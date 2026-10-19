from __future__ import annotations

import asyncio
import io
import json
import logging
import subprocess
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from x_claude.core import atomic_file
from x_claude.core.compact.compactor import Compactor
from x_claude.core.config import XConfig
from x_claude.core.context import ExecutionContext
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.types import LlmResponse
from x_claude.core.logging_setup import setup_logging
from x_claude.core.permissions.storage import load_policy_file, save_policy_file
from x_claude.core.session.store import SessionStore
from x_claude.core.tools.builtin import bash


# 构造独立的 Git 安装与 PATH shell，仅替换外部进程探测，不执行临时假程序
def _shell_install(tmp_path, monkeypatch, *, usable=True):
    git = tmp_path / "Git" / "cmd" / "git.exe"
    preferred = tmp_path / "Git" / "bin" / "bash.exe"
    fallback = tmp_path / "Other" / "bash.exe"
    for path in (git, preferred, fallback):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.delenv("X_BASH_PATH", raising=False)
    monkeypatch.setattr(bash.shutil, "which", lambda name: str(git if name.startswith("git") else fallback))

    # 用真实 CompletedProcess 模拟 shell 能力边界，验证选择器不会执行任务命令
    def probe(args, **kwargs):
        assert "-c" in args and "printf" in args[-1]
        assert kwargs["timeout"] > 0
        ok = usable or str(args[0]) == str(fallback)
        return subprocess.CompletedProcess(args, 0 if ok else 1, b"xclaude-bash-ready" if ok else b"")

    monkeypatch.setattr(bash.subprocess, "run", probe)
    return preferred, fallback


# 功能：验证 PATH 中其它 bash 不抢占已安装的 Git Bash
# 设计：模拟两个不同安装目录，真实文件存在检查与选择逻辑必须选择 Git 安装内的程序
def test_windows_shell_prefers_git(tmp_path, monkeypatch):
    preferred, _ = _shell_install(tmp_path, monkeypatch)
    assert bash._find_windows_bash() == str(preferred)


# 功能：验证显式 shell 配置优先且配置错误时不悄悄换壳
# 设计：合法与不存在路径分别覆盖用户选择和失败关闭，避免同一命令使用意外的解释器
@pytest.mark.parametrize("exists", [True, False])
def test_windows_shell_explicit_path(tmp_path, monkeypatch, exists):
    _shell_install(tmp_path, monkeypatch)
    custom = tmp_path / "Custom Bash" / "bash.exe"
    if exists:
        custom.parent.mkdir()
        custom.touch()
    monkeypatch.setenv("X_BASH_PATH", str(custom))
    assert bash._find_windows_bash() == (str(custom) if exists else None)


# 功能：验证不可用的候选 shell 不被接受，探测失败可继续寻找可用候选
# 设计：Git Bash 探测失败而 PATH 候选成功，验证不是只检查文件是否存在
def test_windows_shell_probes_before_selection(tmp_path, monkeypatch):
    _, fallback = _shell_install(tmp_path, monkeypatch, usable=False)
    assert bash._find_windows_bash() == str(fallback)


# 功能：验证探测超时或无法启动时安全返回不可用
# 设计：给所有真实候选的进程边界注入错误，选择器必须吞掉诊断失败而非误选候选
@pytest.mark.parametrize("error", [OSError("cannot start"), subprocess.TimeoutExpired("bash", 2)])
def test_windows_shell_probe_failure(tmp_path, monkeypatch, error):
    _shell_install(tmp_path, monkeypatch)

    # 在进程创建或等待边界模拟失败
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(bash.subprocess, "run", fail)
    assert bash._find_windows_bash() is None


# 功能：验证系统 WSL 启动器不会被当成原生 Git Bash 使用
# 设计：只提供系统目录候选，并让探测一旦执行就失败，确保在探测前排除不兼容入口
def test_windows_shell_skips_wsl_launcher(tmp_path, monkeypatch):
    system = tmp_path / "Windows"
    launcher = system / "System32" / "bash.exe"
    launcher.parent.mkdir(parents=True)
    launcher.touch()
    monkeypatch.delenv("X_BASH_PATH", raising=False)
    monkeypatch.setenv("SystemRoot", str(system))
    monkeypatch.setattr(bash.shutil, "which", lambda name: None if name.startswith("git") else str(launcher))

    # 被跳过的 WSL 入口不应该触发任何进程探测
    def unexpected(*args, **kwargs):
        raise AssertionError("WSL launcher must not run")

    monkeypatch.setattr(bash.subprocess, "run", unexpected)
    assert bash._find_windows_bash() is None


# 功能：验证 shell 探测不会阻塞 daemon 的异步事件循环
# 设计：探测线程等待事件循环释放屏障，若同步运行则返回特定错误而不会死锁
@pytest.mark.skipif(bash.os.name != "nt", reason="Windows shell selection")
async def test_windows_shell_detection_does_not_block_loop(monkeypatch):
    gate = threading.Event()
    result = []

    # 用线程屏障代替慢 shell 探测，保留真实工具入口和异步调度
    def find():
        result.append(gate.wait(1))
        return None

    monkeypatch.setattr(bash, "_find_windows_bash", find)
    task = asyncio.create_task(bash.BashTool().invoke({"command": "never executed"}))
    try:
        await asyncio.sleep(0)
        gate.set()
        outcome = await task
        assert result == [True]
        assert outcome.error_type == "configuration_error"
    finally:
        gate.set()
        await task


# 功能：验证用户命令失败后不会换壳重跑已发生的副作用
# 设计：真实 Bash 先在临时目录追加标记再失败，退出码和唯一标记共同证明只执行一次
async def test_shell_failure_does_not_repeat_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outcome = await bash.BashTool().invoke({"command": "printf x >> command-count; exit 7"})
    assert outcome.is_error and "[exit 7]" in outcome.content
    assert (tmp_path / "command-count").read_bytes() == b"x"


# 功能：验证自动压缩不能把唯一的长用户请求复制成更长的上下文
# 设计：模型返回短摘要但最新请求原文必须保留，检查内存、磁盘及完成通知均不变
async def test_compaction_rejects_larger_complete_context(tmp_path):
    store = SessionStore(tmp_path)
    context = ExecutionContext("r", "x" * 4000, 5)
    store.append_messages("s", context.messages, "r")
    original = context.messages
    path = store.session_dir("s") / "thread.jsonl"
    original_bytes = path.read_bytes()
    events = []
    bus = EventBus()

    # 收集真实发布的压缩完成事件
    async def collect(event):
        events.append(event)

    bus.subscribe(collect)
    provider = AsyncMock()
    provider.chat.return_value = LlmResponse("end_turn", text="A short summary.")
    result = await Compactor(bus, path.parent, "s", store=store).compact(context, provider)
    assert result is None
    assert context.messages is original
    assert path.read_bytes() == original_bytes
    assert not list(path.parent.glob("summary_*.md"))
    assert not events


# 功能：验证收益比较基于实际截断后的请求，而不是巨大的磁盘工具原文
# 设计：工具原文很长但发送前会截断，新摘要大于实际请求时必须拒绝
async def test_compaction_measures_truncated_request(tmp_path):
    messages = [{"role": "user", "content": "task"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "read_file", "input": {"path": "a"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x" * 20000}]}]
    provider = AsyncMock()
    provider.chat.return_value = LlmResponse("end_turn", text="s" * 2000)
    compactor = Compactor(EventBus(), tmp_path, "s", tool_result_limit=100, tool_result_keep=50)
    assert await compactor.compact_messages(messages, provider) is None


# 功能：验证无收益或模型失败后的自动压缩不会每步重复调用，等待后仍可恢复
# 设计：用可控单调时钟避免真实睡眠，模型调用数用于核对是否额外消耗请求
@pytest.mark.parametrize("model_error", [False, True])
async def test_auto_compaction_failure_backoff(tmp_path, monkeypatch, model_error):
    import time

    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    context = ExecutionContext("r", "x" * 4000, 5)
    provider = AsyncMock()
    if model_error:
        provider.chat.side_effect = RuntimeError("offline")
    else:
        provider.chat.return_value = LlmResponse("end_turn", text="summary")
    compactor = Compactor(EventBus(), tmp_path, "s")
    assert await compactor.compact(context, provider) is None
    assert await compactor.compact(context, provider) is None
    assert provider.chat.await_count == 1
    now[0] = 161.0
    assert time.monotonic() == 161.0
    assert await compactor.compact(context, provider) is None
    assert provider.chat.await_count == 2


# 功能：验证新增大量可压缩历史时不被失败退避阻止，并保留最新请求
# 设计：先触发无收益，再加入大段工具观察；相同实例应立即允许有效压缩
async def test_auto_compaction_growth_can_retry(tmp_path):
    context = ExecutionContext("r", "x" * 4000, 5)
    provider = AsyncMock()
    provider.chat.return_value = LlmResponse("end_turn", text="summary")
    compactor = Compactor(EventBus(), tmp_path, "s")
    assert await compactor.compact(context, provider) is None
    context.add_assistant_message([{"type": "text", "text": "progress " * 2000}])
    assert await compactor.compact(context, provider) is not None
    assert context.messages[-1] == {"role": "user", "content": "x" * 4000}
    assert provider.chat.await_count == 2


# 功能：验证手动压缩也拒绝无收益摘要且不重写会话文件
# 设计：使用真实 SessionManager、存储和错误返回，避免只测自动压缩入口
async def test_manual_compaction_rejects_no_gain(tmp_path):
    from x_claude.core.runner import AgentRunner
    from x_claude.core.session.manager import SessionManager
    from x_claude.core.transport.socket_server import HandlerError

    provider = AsyncMock()
    provider.chat.return_value = LlmResponse("end_turn", text="summary")
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: AgentRunner(XConfig()), EventBus(), provider=provider)
    session = await manager.create("chat")
    store.append_message(session.id, "user", "x" * 4000)
    path = store.session_dir(session.id) / "thread.jsonl"
    original = path.read_bytes()
    with pytest.raises(HandlerError, match="not beneficial"):
        await manager.compact(session.id)
    assert path.read_bytes() == original
    assert not list(path.parent.glob("thread.*.bak*"))


# 功能：验证 JSON 日志中的特殊字符和异常堆栈不会破坏单行结构
# 设计：经真实配置入口和日志 handler 写入内存，再逐行解析并检查异常信息保留
@pytest.mark.parametrize("with_exception", [False, True])
def test_json_logging_is_parseable(tmp_path, monkeypatch, with_exception):
    root = logging.getLogger()
    previous_handlers, previous_level = list(root.handlers), root.level
    output = io.StringIO()
    monkeypatch.setattr("sys.stderr", output)
    config = XConfig()
    config.logging.format = "json"
    config.logging.file = str(tmp_path / "core.log")
    try:
        setup_logging(config)
        if with_exception:
            try:
                raise ValueError('bad "value"\n中文')
            except ValueError:
                logging.getLogger("probe").exception('message "quotes"\nnext', stack_info=True)
        else:
            logging.getLogger("probe").info('message %s', '"quotes"\n中文\\path')
        lines = output.getvalue().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["source"] == "probe"
        assert '\n' in row["msg"] and '"quotes"' in row["msg"]
        if with_exception:
            assert "ValueError" in row["exception"]
            assert "Stack" in row["stack"]
        assert json.loads((tmp_path / "core.log").read_text(encoding="utf-8")) == row
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = previous_handlers
        root.setLevel(previous_level)


# 功能：验证普通文本日志格式不因 JSON 修复改变
# 设计：使用配置入口的真实 handler，断言原有键值格式仍包含格式化后的消息
def test_text_logging_unchanged(monkeypatch):
    root = logging.getLogger()
    previous_handlers, previous_level = list(root.handlers), root.level
    output = io.StringIO()
    monkeypatch.setattr("sys.stderr", output)
    config = XConfig()
    config.logging.file = ""
    try:
        setup_logging(config)
        logging.getLogger("probe").info("count=%d", 3)
        assert output.getvalue().startswith("level=INFO ts=")
        assert 'source=probe msg="count=3"' in output.getvalue()
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = previous_handlers
        root.setLevel(previous_level)


# 功能：验证授权文件同步或替换失败时旧规则完整且无临时文件残留
# 设计：只在真实磁盘提交边界注入错误，不替换保存函数或原子写入函数
@pytest.mark.parametrize("boundary", ["fsync", "replace"])
def test_policy_atomic_failure_preserves_existing(tmp_path, monkeypatch, boundary):
    path = tmp_path / "policy.toml"
    path.write_text('[always]\nbash = "deny"\n', encoding="utf-8")
    original = path.read_bytes()

    # 在已有原子文件设施的提交边界模拟磁盘故障
    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(atomic_file.os, boundary, fail)
    with pytest.raises(OSError, match="disk failure"):
        save_policy_file({"bash": "allow", "write_file": "deny"}, path)
    assert path.read_bytes() == original
    assert load_policy_file(path) == {"bash": "deny"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["policy.toml"]


# 功能：验证授权原子替换只在完整新内容落盘后发生，原有文件格式与加载行为不变
# 设计：在真实 replace 前同时读取旧目标与临时文件，随后执行真实替换并重新加载
def test_policy_atomic_success_reloads(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "policy.toml"
    save_policy_file({"bash": "deny"}, path)
    original = path.read_bytes()
    replaced = []
    real_replace = atomic_file.os.replace

    # 观察提交瞬间，旧目标应仍完整而临时内容已经可读
    def observe(source, target):
        assert Path(target).read_bytes() == original
        assert load_policy_file(Path(source)) == {"bash": "allow", "write_file": "deny"}
        replaced.append(True)
        real_replace(source, target)

    monkeypatch.setattr(atomic_file.os, "replace", observe)
    save_policy_file({"bash": "allow", "write_file": "deny"}, path)
    assert replaced == [True]
    assert load_policy_file(path) == {"bash": "allow", "write_file": "deny"}


# 功能：验证临时文件只写入部分内容后失败也不会截断原授权文件
# 设计：仅包装底层文件写入以模拟磁盘写满，仍执行真实临时文件创建及清理
def test_policy_partial_write_preserves_existing(tmp_path, monkeypatch):
    path = tmp_path / "policy.toml"
    save_policy_file({"bash": "deny"}, path)
    original = path.read_bytes()
    real_fdopen = atomic_file.os.fdopen

    class PartialWriter:
        # 持有原始临时文件，失败后仍由上下文管理器关闭句柄
        def __init__(self, file):
            self.file = file

        # 返回带故障注入的写入接口
        def __enter__(self):
            return self

        # 无论写入是否失败都关闭临时文件
        def __exit__(self, *args):
            self.file.close()

        # 先真实写入部分字节，再模拟磁盘空间耗尽
        def write(self, content):
            self.file.write(content[:8])
            raise OSError("disk full")

    monkeypatch.setattr(atomic_file.os, "fdopen", lambda *args: PartialWriter(real_fdopen(*args)))
    with pytest.raises(OSError, match="disk full"):
        save_policy_file({"bash": "allow"}, path)
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["policy.toml"]
