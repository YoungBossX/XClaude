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


# 在 Windows 上定位 Git Bash，优先使用 PATH，其次根据 git.exe 的安装目录推导
def _find_windows_bash() -> str | None:
    for name in ("bash.exe", "bash"):
        if executable := shutil.which(name):
            return executable

    if git := shutil.which("git.exe") or shutil.which("git"):
        git_root = Path(git).resolve().parent.parent
        for relative_path in ("bin/bash.exe", "usr/bin/bash.exe"):
            candidate = git_root / relative_path
            if candidate.is_file():
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
            executable = _find_windows_bash()
            if executable is None:
                return ToolResult(
                    content=(
                        "Bash is unavailable. Install Git for Windows, or add its "
                        "bash.exe directory to PATH."
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
