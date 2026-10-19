from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from x_claude.core.tools.base import BaseTool, ToolResult

_MAX_OUTPUT_BYTES = 64 * 1024  # 64 KB
_DEFAULT_TIMEOUT = 60


# 分块排空输出，只保留固定字节预算，避免子进程的大量输出堆积在内存中
async def _read_output(proc: asyncio.subprocess.Process) -> tuple[bytes, bool]:
    assert proc.stdout is not None
    output = bytearray()
    truncated = False
    while chunk := await proc.stdout.read(8192):
        remaining = _MAX_OUTPUT_BYTES - len(output)
        output.extend(chunk[:remaining])
        truncated |= len(chunk) > remaining
    await proc.wait()
    return bytes(output), truncated


# 只终止本工具启动的进程树，并限时回收管道；Windows 使用 PID 明确限定 taskkill 目标
async def _stop_process(proc: asyncio.subprocess.Process) -> None:
    try:
        if os.name == "nt":
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/PID", str(proc.pid), "/T", "/F",
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            try:
                await asyncio.wait_for(killer.wait(), timeout=3)
            except TimeoutError:
                killer.kill()
                await killer.wait()
        else:
            getattr(os, "killpg")(proc.pid, getattr(signal, "SIGKILL"))
    except (ProcessLookupError, OSError):
        pass
    if proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(_read_output(proc), timeout=3)
    except TimeoutError:
        pass


# 限时探测 shell 能力，禁用启动脚本且只运行固定命令，不执行或重试用户任务
def _probe_windows_bash(executable: Path) -> bool:
    environment = dict(os.environ)
    environment.pop("BASH_ENV", None)
    environment.pop("ENV", None)
    try:
        result = subprocess.run(
            [str(executable), "--noprofile", "--norc", "-c", "printf xclaude-bash-ready"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=2, check=False, env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return result.returncode == 0 and result.stdout == b"xclaude-bash-ready"
    except (OSError, subprocess.TimeoutExpired):
        return False


# 在 Windows 上优先选择显式配置或 Git 安装内的 Bash，并跳过系统 WSL 启动器
def _find_windows_bash() -> str | None:
    candidates: list[Path] = []
    if configured := os.environ.get("X_BASH_PATH", "").strip():
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            return None
        candidates.append(candidate)
    else:
        if git := shutil.which("git.exe") or shutil.which("git"):
            git_root = Path(git).resolve().parent.parent
            candidates.extend(git_root / p for p in ("bin/bash.exe", "usr/bin/bash.exe"))
        for name in ("bash.exe", "bash"):
            if executable := shutil.which(name):
                candidates.append(Path(executable))
    windows = Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve()
    wsl_directories = {windows / "System32", windows / "SysWOW64", windows / "Sysnative"}
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or candidate.parent in wsl_directories:
            continue
        seen.add(candidate)
        if candidate.is_file() and _probe_windows_bash(candidate):
            return str(candidate)
    return None


class BashParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    command: str
    timeout: int = Field(default=_DEFAULT_TIMEOUT, ge=1, le=120)


class BashTool(BaseTool):
    params_model = BashParams
    name = "bash"
    description = (
        "Execute a POSIX Bash command and return its output (stdout + stderr combined). "
        "On Windows this uses Git Bash, so use POSIX shell syntax. "
        "Non-interactive only — commands requiring user input will hang and time out. "
        "Prefer short, focused commands. Output is truncated at 64 KB."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": f"Maximum seconds to wait (default {_DEFAULT_TIMEOUT}, max 120).",
            },
        },
        "required": ["command"],
    }

    # 在子进程中执行 shell 命令，合并 stdout/stderr，超时或非零退出码时返回错误
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = BashParams.model_validate(params)
        command = p.command
        timeout = p.timeout

        executable: str | None = None
        if os.name == "nt":
            executable = await asyncio.to_thread(_find_windows_bash)
            if executable is None:
                return ToolResult(
                    content=(
                        "No usable Bash found. Install Git for Windows or set X_BASH_PATH "
                        "to its absolute bash.exe path. Check an existing X_BASH_PATH setting; "
                        "the Windows WSL launcher is not supported."
                    ),
                    is_error=True,
                    error_type="configuration_error",
                )

        try:
            if executable is not None:
                proc = await asyncio.create_subprocess_exec(
                    executable,
                    "-c",
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
                    start_new_session=os.name != "nt",
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            try:
                stdout_bytes, truncated = await asyncio.wait_for(
                    _read_output(proc), timeout=timeout
                )
            except TimeoutError:
                await _stop_process(proc)
                return ToolResult(
                    content=f"[timeout after {timeout}s]",
                    is_error=True,
                    error_type="timeout",
                )
            except asyncio.CancelledError:
                cleanup = asyncio.create_task(_stop_process(proc))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                raise
            except Exception:
                await _stop_process(proc)
                raise
        except Exception as exc:
            return ToolResult(content=str(exc), is_error=True, error_type="runtime_error")

        output = stdout_bytes.decode("utf-8", errors="replace")
        if truncated:
            output += "\n[truncated]"

        returncode = proc.returncode or 0
        if returncode != 0:
            return ToolResult(
                content=f"[exit {returncode}]\n{output}",
                is_error=True,
                error_type="runtime_error",
            )
        return ToolResult(content=output or "[no output]")
