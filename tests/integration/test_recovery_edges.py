from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from textual.widgets import Static

from tests.integration.test_background_restart import _checkpoint, _manager, _signature
from tests.integration.test_runtime_hardening import _serve
from x_claude.cli.commands import chat
from x_claude.core.app import CoreApp
from x_claude.core.bus.commands import SessionRecoverCommand
from x_claude.core.bus.envelope import HandlerError
from x_claude.core.bus.events import LlmTokenEvent
from x_claude.core.context import ExecutionContext
from x_claude.core.events.bus import EventBus
from x_claude.core.events.writer import EventWriter
from x_claude.core.llm.types import LlmResponse
from x_claude.core.session.store import SessionStore
from x_claude.core.subagent.checkpoint import BackgroundCheckpoint
from x_claude.core.subagent.tool import AgentResultTool
from x_claude.core.tools.builtin.bash import _MAX_OUTPUT_BYTES, BashTool, _read_output
from x_claude.core.transport.ipc_broadcaster import IpcEventBroadcaster
from x_claude.tui.app import ChatTextArea, SlashCompleteWidget, XTuiApp


# 将临时目录作为测试项目，文件工具不能再绕过真实工作目录边界
@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


class Provider:
    # 返回确定性结果并记录输入，验证恢复不会调用原工具或真实模型
    def __init__(self) -> None:
        self.messages: list = []

    # 只完成恢复后的下一步，供测试检查模型收到的工具观察结果
    async def chat(self, messages: list, **kwargs: object) -> LlmResponse:
        self.messages.append(messages)
        return LlmResponse("end_turn", text="resumed")


# 构造包含两个待核对工具的真实检查点，阶段为已推理但工具结果未确认
def _interrupted(store: SessionStore, sid: str) -> BackgroundCheckpoint:
    checkpoint = _checkpoint(store, sid)
    checkpoint.record.tools = ["write_file"]
    checkpoint.record.runtime_signature = _signature(["write_file"])
    context = ExecutionContext("child", "goal", 5, step=1)
    context.add_assistant_message([
        {"type": "tool_use", "id": key, "name": "write_file",
         "input": {"path": str(store._root / key), "content": key}}
        for key in ("a", "b")
    ])
    checkpoint.save(context, "tools")
    return checkpoint


# 功能：验证模型初始化失败或检查点修复后，同一 daemon 的第二次恢复可以成功
# 设计：第一次保留失败缓存，修正故障后不重建 manager，排除依靠 daemon 重启掩盖缓存问题
@pytest.mark.parametrize("failure", ["provider", "checkpoint"])
async def test_transient_recovery_can_retry(tmp_path: Path, failure: str) -> None:
    store = SessionStore(tmp_path)
    manager = _manager(store, None)
    session = await manager.create("chat")
    checkpoint = _checkpoint(store, session.id)
    before = checkpoint.path.read_bytes()
    try:
        if failure == "checkpoint":
            checkpoint.path.write_bytes(b"{broken")
        with patch("x_claude.core.runner.AnthropicProvider", side_effect=RuntimeError("unavailable")):
            await manager.recover_background(session.id)
        if failure == "checkpoint":
            checkpoint.path.write_bytes(before)
        provider = Provider()
        runner = manager._runners[session.id]
        runner._provider = provider
        await asyncio.gather(manager.recover_background(session.id),
                             manager.recover_background(session.id))
        await asyncio.gather(*(task for task, _ in runner._task_registry.all()))
        result = await AgentResultTool(runner._task_registry).invoke({"run_id": "child"})
        assert result.content == "resumed" and not result.is_error
        assert len(provider.messages) == 1
    finally:
        await manager.shutdown()


# 功能：验证人工提供全部结果后续跑且不重放工具，失败观察也保留 is_error
# 设计：用真实检查点与模型输入断言，同时让原写文件工具一旦调用就报错，验证安全边界
async def test_manual_tool_recovery_does_not_replay(tmp_path: Path) -> None:
    provider = Provider()
    store = SessionStore(tmp_path)
    manager = _manager(store, provider)
    session = await manager.create("chat")
    checkpoint = _interrupted(store, session.id)
    try:
        await manager.recover_background(session.id)
        assert not provider.messages
        with patch("x_claude.core.tools.builtin.write_file.WriteFileTool.invoke",
                   side_effect=AssertionError("tool must not replay")):
            await manager.recover(SessionRecoverCommand(
                session_id=session.id, run_id="child", tool_results={
                    "a": "already written and verified",
                    "b": {"content": "not executed; verified no file", "is_error": True},
                },
            ))
            runner = manager._runners[session.id]
            await asyncio.gather(*(task for task, _ in runner._task_registry.all()))
        results = provider.messages[0][-1]["content"]
        assert [r["tool_use_id"] for r in results] == ["a", "b"]
        assert results[1]["is_error"] is True
        assert "User-verified" in results[0]["content"]
        assert BackgroundCheckpoint.load(checkpoint.path, session.id).record.context.status == "success"
        assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    finally:
        await manager.shutdown()


# 功能：验证遗漏、多填或空白的工具结果不能解除中断保护，且磁盘检查点保持不变
# 设计：对三类危险确认进行参数化，逐字节比较文件并确认模型未被启动
@pytest.mark.parametrize("results", [{"a": "done"}, {"a": "done", "b": " "},
                                      {"a": "done", "b": "done", "c": "extra"}])
async def test_manual_recovery_rejects_incomplete_results(tmp_path: Path, results: dict) -> None:
    provider = Provider()
    store = SessionStore(tmp_path)
    manager = _manager(store, provider)
    session = await manager.create("chat")
    checkpoint = _interrupted(store, session.id)
    before = checkpoint.path.read_bytes()
    try:
        with pytest.raises(HandlerError):
            await manager.recover(SessionRecoverCommand(
                session_id=session.id, run_id="child", tool_results=results,
            ))
        assert checkpoint.path.read_bytes() == before
        assert not provider.messages
    finally:
        await manager.shutdown()


# 功能：验证人工结果提交遇到磁盘写入失败时不推进状态，也不启动任何模型或工具
# 设计：模拟原子写入失败并对比检查点字节，确认内存失败缓存仍然保留
async def test_manual_confirmation_write_failure_is_safe(tmp_path: Path) -> None:
    provider = Provider()
    store = SessionStore(tmp_path)
    manager = _manager(store, provider)
    session = await manager.create("chat")
    checkpoint = _interrupted(store, session.id)
    try:
        await manager.recover_background(session.id)
        before = checkpoint.path.read_bytes()
        with patch("x_claude.core.subagent.checkpoint.atomic_write_bytes", side_effect=OSError("disk")):
            with pytest.raises(HandlerError, match="disk"):
                await manager.recover(SessionRecoverCommand(
                    session_id=session.id, run_id="child", tool_results={"a": "done", "b": "done"},
                ))
        assert checkpoint.path.read_bytes() == before
        assert not provider.messages
        entry = manager._runners[session.id]._task_registry.get("child")
        assert entry is not None and "needs_review" in entry[1].reason
    finally:
        await manager.shutdown()


# 功能：验证批次恢复被取消时不会遗留永远等待放行的任务，下一次仍可正常恢复
# 设计：在登记任务后的通知阶段制造取消，检查注册表清理、磁盘状态不变及再次恢复成功
async def test_cancelled_recovery_batch_can_retry(tmp_path: Path) -> None:
    provider = Provider()
    store = SessionStore(tmp_path)
    manager = _manager(store, provider)
    session = await manager.create("chat")
    checkpoint = _checkpoint(store, session.id)
    before = checkpoint.path.read_bytes()
    entered = asyncio.Event()

    # 登记完成但尚未放行时阻塞，使取消真正命中恢复批次的临界边界
    async def block(event):
        if event.type == "subagent.restored":
            entered.set()
            await asyncio.Event().wait()

    manager._bus.subscribe(block)
    task = asyncio.create_task(manager.recover_background(session.id))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        manager._bus.unsubscribe(block)
        runner = manager._runners[session.id]
        assert not runner._task_registry.all()
        assert checkpoint.path.read_bytes() == before and not provider.messages
        await manager.recover_background(session.id)
        await asyncio.gather(*(t for t, _ in runner._task_registry.all()))
        assert len(provider.messages) == 1
    finally:
        manager._bus.unsubscribe(block)
        await manager.shutdown()


# 功能：验证模型、地址、工具契约变化或旧检查点缺少指纹时必须明确确认配置
# 设计：拒绝时不修改检查点；确认后正常完成，并检查保存的检查点不含认证密钥原文
@pytest.mark.parametrize("change", ["model", "endpoint", "schema", "legacy"])
async def test_recovery_requires_config_confirmation(tmp_path: Path, monkeypatch, change: str) -> None:
    provider = Provider()
    store = SessionStore(tmp_path)
    manager = _manager(store, provider)
    session = await manager.create("chat")
    checkpoint = _checkpoint(store, session.id)
    if change == "legacy":
        checkpoint.record.runtime_signature = None
        checkpoint.save(checkpoint.record.context.restore("child"), "planning")
    runner = manager._runner_factory()
    manager._runners[session.id] = runner
    if change == "model":
        runner._config.llm.default_model = "changed-model"
    elif change == "endpoint":
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://changed.example.invalid")
    elif change == "schema":
        monkeypatch.setattr("x_claude.core.tools.builtin.list_dir.ListDirTool.description", "changed")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "SECRET-NOT-FOR-CHECKPOINT")
    before = checkpoint.path.read_bytes()
    try:
        await manager.recover_background(session.id)
        assert not provider.messages
        with pytest.raises(HandlerError, match="accept-config"):
            await manager.recover(SessionRecoverCommand(session_id=session.id, run_id="child"))
        assert checkpoint.path.read_bytes() == before
        await manager.recover(SessionRecoverCommand(
            session_id=session.id, run_id="child", accept_config_change=True,
        ))
        await asyncio.gather(*(task for task, _ in runner._task_registry.all()))
        assert len(provider.messages) == 1
        assert b"SECRET-NOT-FOR-CHECKPOINT" not in checkpoint.path.read_bytes()
    finally:
        await manager.shutdown()


# 功能：验证回放期间新事件不丢失，已落盘但稍后广播的事件不重复
# 设计：真实事件文件配合 drain 屏障制造窗口，断言历史与缓冲事件严格按顺序各出现一次
async def test_replay_and_live_delivery_have_no_gap_or_duplicates(tmp_path: Path) -> None:
    app = CoreApp()
    app._broadcaster = IpcEventBroadcaster()
    bus = EventBus()
    bus.subscribe(app._broadcaster.handle)
    writer = Mock()
    path = tmp_path / "events.jsonl"
    old = LlmTokenEvent(run_id="root", token="old", ts="t")
    new = LlmTokenEvent(run_id="root", token="new", ts="t")
    async with EventWriter(path) as disk:
        disk.subscribe(bus)
        await disk.handle(old)

        # 首次刷新历史时广播已落盘旧事件和全新事件，制造回放与实时重叠
        async def drain() -> None:
            if writer.write.call_count == 1:
                await bus.publish(old)
                await bus.publish(new)

        writer.drain = AsyncMock(side_effect=drain)
        with patch("x_claude.core.app.get_connection_writer", return_value=writer), \
                patch("x_claude.core.app.events_file", return_value=path):
            result = await app._subscribe_handler({
                "topics": ["llm.token"], "scope": "run:root", "replay_from_run": "root",
            })
    events = [json.loads(call.args[0])["event"] for call in writer.write.call_args_list]
    assert result.replayed_count == 1
    assert [event["token"] for event in events] == ["old", "new"]


# 功能：验证回放取消或缓冲溢出后移除订阅，不留下无限缓冲或幽灵连接
# 设计：分别取消真实订阅 handler 和直接触发固定上限，检查后台订阅生命周期
@pytest.mark.parametrize("failure", ["cancel", "overflow"])
async def test_replay_failure_cleans_subscription(failure: str) -> None:
    app = CoreApp()
    app._broadcaster = IpcEventBroadcaster()
    writer = Mock()
    writer.drain = AsyncMock()
    if failure == "overflow":
        app._broadcaster.subscribe(writer, ["llm.token"], replaying=True)
        for i in range(1025):
            await app._broadcaster.handle(LlmTokenEvent(run_id="root", token=str(i), ts="t"))
        assert not app._broadcaster._subscriptions
        writer.close.assert_called_once()
    else:
        with patch("x_claude.core.app.get_connection_writer", return_value=writer), \
                patch.object(app, "_replay_events", side_effect=asyncio.CancelledError()):
            with pytest.raises(asyncio.CancelledError):
                await app._subscribe_handler({"topics": ["*"], "replay_from_run": "root"})
        assert not app._broadcaster._subscriptions


# 功能：验证审批过期、整轮结束和连接断开都能取消未提交的输入，不需再按回车
# 设计：模拟永不返回的 stdin，使用事件屏障确认已经进入审批输入，再触发终止条件
@pytest.mark.parametrize("end", ["timeout", "complete", "disconnect"])
async def test_cli_approval_input_is_cancelled(end: str) -> None:
    printer = chat.ChatPrinter()
    printer._permissions[("run", "approval")] = {"run_id": "run"}
    printer.pending_permission_id = ("run", "approval")
    printer.permission_available.set()
    entered, cancelled, send_done, disconnected = (asyncio.Event() for _ in range(4))

    # 保持运行直到测试触发完成，与审批输入并发等待
    async def send(*args, **kwargs):
        await send_done.wait()

    # 模拟用户一直不回车，取消时记录输入任务确实已经停止
    async def readline(prompt):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client = Mock()
    client.send_command = send
    loop_task = asyncio.create_task(disconnected.wait())
    with patch.object(chat, "_readline", readline):
        task = asyncio.create_task(chat._send_with_approval(client, printer, "sid", "goal", loop_task))
        await asyncio.wait_for(entered.wait(), 1)
        if end == "timeout":
            printer.dismiss_permission(("run", "approval"))
        elif end == "complete":
            send_done.set()
        else:
            disconnected.set()
        await asyncio.wait_for(cancelled.wait(), 1)
        send_done.set()
        if end == "disconnect":
            with pytest.raises(OSError, match="connection closed"):
                await asyncio.wait_for(task, 1)
        else:
            await asyncio.wait_for(task, 1)
    loop_task.cancel()
    await asyncio.gather(loop_task, return_exceptions=True)


# 功能：验证 Windows 控制台实际输入实现可取消，不创建阻塞 input 的线程
# 设计：使用真实 _readline 的轮询分支但模拟空键盘，取消后立即回收任务
@pytest.mark.skipif(os.name != "nt", reason="Windows console implementation")
async def test_windows_console_input_cancellation() -> None:
    with patch.object(sys.stdin, "isatty", return_value=True), \
            patch("msvcrt.kbhit", return_value=False), \
            patch("builtins.input", side_effect=AssertionError("must not use input thread")):
        task = asyncio.create_task(chat._readline("test> "))
        await asyncio.sleep(0.03)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)


# 功能：验证 Windows 控制台保留中文输入及退格操作，按回车提交正确文本
# 设计：模拟 getwch 返回字符而非 mock 整个读取函数，覆盖替换输入实现的基础交互
@pytest.mark.skipif(os.name != "nt", reason="Windows console implementation")
async def test_windows_console_unicode_and_backspace() -> None:
    with patch.object(sys.stdin, "isatty", return_value=True), \
            patch("msvcrt.kbhit", return_value=True), \
            patch("msvcrt.getwch", side_effect=["中", "a", "\b", "文", "\r"]):
        assert await chat._readline("test> ") == "中文"


# 功能：验证输出读取器不会累计超预算内容，并将字节预算应用到多字节字符
# 设计：模拟两千个分块且禁止 communicate，记录所有读取块大小并检查返回缓存固定上限
async def test_shell_output_memory_is_bounded() -> None:
    chunks = iter([b"\xe4\xb8\xad" * 2000] * 2000 + [b""])
    proc = Mock()
    proc.stdout.read = AsyncMock(side_effect=lambda size: next(chunks))
    proc.wait = AsyncMock()
    output, truncated = await _read_output(proc)
    assert len(output) == _MAX_OUTPUT_BYTES and truncated
    assert all(call.args == (8192,) for call in proc.stdout.read.call_args_list)
    proc.communicate.assert_not_called()


# 功能：验证真实 Bash 大量输出仍能正常完成，返回结果只保留固定预算前缀
# 设计：真实子进程输出超过管道容量，排除只截返回值但不排空管道导致死锁的实现
async def test_large_real_shell_output_is_drained() -> None:
    result = await asyncio.wait_for(BashTool().invoke({
        "command": "head -c 2000000 /dev/zero | tr '\\0' x", "timeout": 10,
    }), 15)
    assert not result.is_error
    assert result.content == "x" * _MAX_OUTPUT_BYTES + "\n[truncated]"


# 功能：验证 TUI 的 /recover 经真实 TCP 获取中断工具详情并可提交确认结果
# 设计：使用 Textual 无头运行及真实 Core handler，确保该入口不是仅供 CLI 调试的 API
async def test_tui_manual_recovery_over_tcp(tmp_path: Path) -> None:
    async with _serve(tmp_path, Provider()) as (cfg, manager, _, store):
        app = XTuiApp(cfg.host, cfg.port)
        async with app.run_test() as pilot:
            for _ in range(100):
                if app._session_id:
                    break
                await asyncio.sleep(0.01)
            assert app._session_id
            _interrupted(store, app._session_id)
            await app._do_recover()
            await pilot.pause()
            text = "\n".join(str(w.content) for w in app.query(Static))
            assert "pending_tools" in text and "write_file" in text
            prompt = app.query_one("#prompt", ChatTextArea)
            prompt.text = '/recover child {"a":"checked", "b":"checked"}'
            await app.on_chat_text_area_submitted(ChatTextArea.Submitted(prompt))
            await pilot.pause()
            runner = manager._runners[app._session_id]
            await asyncio.gather(*(task for task, _ in runner._task_registry.all()))
            result = await AgentResultTool(runner._task_registry).invoke({"run_id": "child"})
            assert not result.is_error and result.content == "resumed"
            assert prompt.text == ""
            prompt.text = "/rec"
            app.on_slash_complete_widget_selected(SlashCompleteWidget.Selected("recover"))
            await pilot.pause()
            assert prompt.text == "" and app._session_id is not None
