from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import x_claude.core.atomic_file as atomic_module
from x_claude.core.context import ExecutionContext
from x_claude.core.subagent.checkpoint import ContextSnapshot
from x_claude.core.tools.builtin.read_file import ReadFileTool
from x_claude.core.tools.builtin.write_file import WriteFileTool
from x_claude.core.tools.file_access import file_access_scope, observed_version


# 将所有文件副作用限制在测试项目，避免触碰用户工作目录
@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


# 功能：验证写入、短写、同步和替换失败均保留原文件及版本，并清理临时文件
# 设计：故障注入真实临时文件的底层提交阶段，而非模拟工具返回错误
@pytest.mark.parametrize("failure", ["partial", "short", "fsync", "replace"])
async def test_failed_overwrite_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    target = tmp_path / "original.txt"
    target.write_bytes(b"original content")
    versions: dict[str, str | None] = {}
    with file_access_scope(versions):
        await ReadFileTool().invoke({"path": str(target)})
        original_versions = versions.copy()
        if failure in {"partial", "short"}:
            real_fdopen = os.fdopen

            class FaultyStream:
                # 持有真实临时文件流以验证部分字节已落入临时文件
                def __init__(self, stream):
                    self.stream = stream

                # 返回用于注入故障的写入包装器
                def __enter__(self):
                    return self

                # 无论异常与否都关闭底层描述符
                def __exit__(self, *args):
                    self.stream.close()

                # 写入前缀后失败或短写，不允许替换原文件
                def write(self, content):
                    count = self.stream.write(content[:3])
                    self.stream.flush()
                    if failure == "partial":
                        raise OSError("simulated disk full")
                    return count

            # 只包装原子写入产生的描述符，保留真实文件系统行为
            def fdopen(*args, **kwargs):
                return FaultyStream(real_fdopen(*args, **kwargs))

            monkeypatch.setattr(atomic_module.os, "fdopen", fdopen)
        else:
            # 在真实临时文件完整写入后模拟同步或替换失败
            def fail(*args):
                raise OSError(f"simulated {failure} failure")

            monkeypatch.setattr(atomic_module.os, failure, fail)
        with pytest.raises(OSError):
            await WriteFileTool().invoke({"path": str(target), "content": "new content"})
        assert versions == original_versions
    assert target.read_bytes() == b"original content"
    assert list(tmp_path.iterdir()) == [target]


# 功能：验证同任务写入更新版本，后续写入无需重复读取且内容字节准确
# 设计：包含中文和换行，直接比较字节以覆盖 Windows 文本换行与编码差异
async def test_successful_writes_refresh_version(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    versions: dict[str, str | None] = {}
    with file_access_scope(versions):
        first = await WriteFileTool().invoke({"path": str(target), "content": "中文\nfirst"})
        before = versions.copy()
        second = await WriteFileTool().invoke({"path": str(target), "content": "中文\nsecond"})
    assert not first.is_error and not second.is_error
    assert before != versions and target.read_bytes() == "中文\nsecond".encode()
    assert list(tmp_path.iterdir()) == [target]


# 功能：验证未读取已有文件时拒绝覆盖，不允许默认调用绕过冲突检测
# 设计：覆盖既无任务上下文也有空版本表的入口，避免保护只在部分运行路径生效
@pytest.mark.parametrize("versions", [None, {}])
async def test_existing_file_requires_read(tmp_path: Path, versions) -> None:
    target = tmp_path / "file.txt"
    target.write_text("original", encoding="utf-8")
    with file_access_scope(versions):
        result = await WriteFileTool().invoke({"path": str(target), "content": "lost"})
    assert result.is_error and result.error_type == "file_conflict"
    assert target.read_text() == "original"


# 功能：验证两份旧快照不能互相覆盖，重新读取并合并后可提交
# 设计：使用两个独立任务版本表，确保后写方不能借用前写方更新的授权版本
async def test_stale_agent_write_requires_merge(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("base", encoding="utf-8")
    versions_a: dict[str, str | None] = {}
    versions_b: dict[str, str | None] = {}
    for versions in (versions_a, versions_b):
        with file_access_scope(versions):
            await ReadFileTool().invoke({"path": str(target)})
    with file_access_scope(versions_a):
        assert not (await WriteFileTool().invoke(
            {"path": str(target), "content": "base\nA"},
        )).is_error
    with file_access_scope(versions_b):
        rejected = await WriteFileTool().invoke({"path": str(target), "content": "base\nB"})
        assert rejected.error_type == "file_conflict"
        assert target.read_text() == "base\nA"
        read = await ReadFileTool().invoke({"path": str(target)})
        merged = await WriteFileTool().invoke({"path": str(target), "content": read.content + "\nB"})
    assert not merged.is_error and target.read_text() == "base\nA\nB"


# 功能：验证文件删除、外部修改及内容相同的文件替换均使旧版本失效
# 设计：每次先完整读取再改变目标，覆盖哈希相同但文件身份变化的边界
@pytest.mark.parametrize("change", ["delete", "edit", "replace"])
async def test_changed_file_is_rejected(tmp_path: Path, change: str) -> None:
    target = tmp_path / "file.txt"
    target.write_bytes(b"original")
    with file_access_scope({}):
        await ReadFileTool().invoke({"path": str(target)})
        if change == "delete":
            target.unlink()
        elif change == "edit":
            target.write_bytes(b"external edit")
        else:
            replacement = tmp_path / "replacement.txt"
            replacement.write_bytes(b"original")
            os.replace(replacement, target)
        before = target.read_bytes() if target.exists() else None
        result = await WriteFileTool().invoke({"path": str(target), "content": "stale"})
    assert result.error_type == "file_conflict"
    assert (target.read_bytes() if target.exists() else None) == before


# 功能：验证未存在文件被另一个任务抢先创建后不能被覆盖
# 设计：缺失读取仍记录缺失版本，分别测试读过缺失文件和未读的新任务
@pytest.mark.parametrize("read_missing", [True, False])
async def test_new_file_collision_is_rejected(tmp_path: Path, read_missing: bool) -> None:
    target = tmp_path / "new.txt"
    with file_access_scope({}):
        if read_missing:
            with pytest.raises(FileNotFoundError):
                await ReadFileTool().invoke({"path": str(target)})
        target.write_bytes(b"other agent")
        result = await WriteFileTool().invoke({"path": str(target), "content": "overwrite"})
    assert result.error_type == "file_conflict" and target.read_bytes() == b"other agent"


# 功能：验证截断读取不允许用局部内容覆盖整个文件
# 设计：原文件大小位于读取与写入上限之间，排除单纯写入大小限制导致的假通过
async def test_truncated_read_cannot_authorize_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "large.txt"
    original = b"x" * (600 * 1024)
    target.write_bytes(original)
    with file_access_scope({}):
        read = await ReadFileTool().invoke({"path": str(target)})
        assert read.content.endswith("[truncated]")
        result = await WriteFileTool().invoke({"path": str(target), "content": "short"})
    assert result.error_type == "file_conflict" and target.read_bytes() == original


# 功能：验证文件版本可以持久恢复，旧检查点默认无覆盖授权
# 设计：完整 JSON 往返后修改文件，确认恢复不重置版本也不放过外部修改
async def test_versions_survive_checkpoint_restore(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_bytes(b"original")
    context = ExecutionContext("run", "edit file", 5)
    with file_access_scope(context.file_versions):
        await ReadFileTool().invoke({"path": str(target)})
    snapshot = ContextSnapshot.capture(context)
    restored = ContextSnapshot.model_validate_json(snapshot.model_dump_json()).restore("run")
    assert restored.file_versions == context.file_versions
    assert restored.file_versions is not context.file_versions
    target.write_bytes(b"external edit")
    with file_access_scope(restored.file_versions):
        result = await WriteFileTool().invoke({"path": str(target), "content": "stale"})
    assert result.error_type == "file_conflict" and target.read_bytes() == b"external edit"
    old = snapshot.model_dump()
    old.pop("file_versions")
    assert ContextSnapshot.model_validate(old).restore("old").file_versions == {}


# 功能：验证成功替换保留可执行位，拒绝覆盖只读文件
# 设计：真实权限位检查覆盖 POSIX 可执行文件与 Windows 只读属性，清理前恢复写权限
async def test_write_preserves_permissions(tmp_path: Path) -> None:
    target = tmp_path / "script.txt"
    target.write_bytes(b"original")
    mode = 0o755 if os.name != "nt" else stat.S_IREAD | stat.S_IWRITE
    target.chmod(mode)
    mode = stat.S_IMODE(target.stat().st_mode)
    with file_access_scope({}):
        await ReadFileTool().invoke({"path": str(target)})
        assert not (await WriteFileTool().invoke(
            {"path": str(target), "content": "new"},
        )).is_error
    assert stat.S_IMODE(target.stat().st_mode) == mode
    target.chmod(stat.S_IREAD)
    try:
        with file_access_scope({}):
            await ReadFileTool().invoke({"path": str(target)})
            with pytest.raises(PermissionError):
                await WriteFileTool().invoke({"path": str(target), "content": "bad"})
        assert target.read_bytes() == b"new"
    finally:
        target.chmod(stat.S_IREAD | stat.S_IWRITE)


# 功能：验证嵌套上下文退出后恢复父任务版本，独立调用不会继承子任务授权
# 设计：直接检查上下文变量，防止共享字典或错误清理造成跨 Agent 授权泄漏
async def test_file_scope_does_not_leak(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_bytes(b"original")
    with file_access_scope({}):
        await ReadFileTool().invoke({"path": str(target)})
        parent_version = observed_version(target)
        with file_access_scope({}):
            assert observed_version(target) == (False, None)
        assert observed_version(target) == parent_version
    assert observed_version(target) == (False, None)


# 功能：验证不同模块在全新解释器中独立导入，公开导出入口保持兼容
# 设计：子进程不继承 pytest 的模块缓存，避免导入顺序掩盖循环依赖
@pytest.mark.parametrize("statement", [
    "from x_claude.core.tools.builtin.write_file import WriteFileTool",
    "from x_claude.core.tools.invocation import invoke_tool",
    "from x_claude.core.bus.events import ToolCallFailedEvent",
    "from x_claude.core.session.manager import SessionManager",
    "from x_claude.core.subagent.checkpoint import ContextSnapshot",
    "from x_claude.core.loop import AgentLoop",
    "from x_claude.core.llm.base import LLMProvider",
    "from x_claude.core.tools import *; from x_claude.core.session import *; "
    "from x_claude.core.subagent import *",
])
def test_modules_import_independently(statement: str) -> None:
    result = subprocess.run([sys.executable, "-c", statement],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
