from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tests.integration.test_runtime_hardening import _serve
from x_claude.core.app import CoreApp
from x_claude.core.bus.events import LlmTokenEvent
from x_claude.core.llm.types import LlmResponse
from x_claude.core.permissions.manager import PermissionManager
from x_claude.core.transport.ipc_broadcaster import IpcEventBroadcaster
from x_claude.core.transport.socket_client import SocketClient
from x_claude.tui.app import ChatTextArea, LLMStreamBlock, PermissionSelect, ToolCallBlock, XTuiApp


# 将实际 TCP 与持久化会话均放入临时项目，避免使用真实模型或用户历史
@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


# 功能：验证同工具 ID 的并发运行分别收到正确成功或失败结果
# 设计：真实 Textual 挂载两个工具块，交换完成顺序覆盖结果匹配对时序的依赖
@pytest.mark.parametrize("order", [("A", "B"), ("B", "A")])
async def test_colliding_tool_ids_have_independent_results(order: tuple[str, str]) -> None:
    app = XTuiApp("127.0.0.1", 0)
    with patch.object(app, "_socket_loop", AsyncMock()):
        async with app.run_test() as pilot:
            for run in ("A", "B"):
                app._handle_event({"type": "tool.call_started", "run_id": run,
                                   "tool_use_id": "same", "tool_name": "read_file",
                                   "params": {"path": run + ".txt"}})
            await pilot.pause()
            for run in order:
                app._handle_event({"type": "tool.call_finished" if run == "A" else "tool.call_failed",
                                   "run_id": run, "tool_use_id": "same", "elapsed_ms": 1,
                                   "output": "A output", "error_message": "B error"})
            await pilot.pause()
            blocks = {block._params["path"]: block for block in app.query(ToolCallBlock)}
            assert blocks["A.txt"]._output == "A output" and not blocks["A.txt"]._is_error
            assert blocks["B.txt"]._output == "B error" and blocks["B.txt"]._is_error
            assert all(block._finished for block in blocks.values())
            assert not app._pending_tool_blocks


# 功能：验证并发流式 token 不拼接，其他运行的步骤事件不截断当前流，重复事件不重复输出
# 设计：真实 UI 中交错两个运行的事件并重复投递，检查最终文本与 Markdown 终结状态
async def test_llm_streams_are_scoped_and_deduplicated() -> None:
    app = XTuiApp("127.0.0.1", 0)
    with patch.object(app, "_socket_loop", AsyncMock()):
        async with app.run_test() as pilot:
            first = {"type": "llm.token", "run_id": "A", "token": "A1", "event_id": "one"}
            for event in [first, {"type": "llm.token", "run_id": "B", "token": "B1"},
                          {"type": "step.started", "run_id": "B", "step": 2},
                          {"type": "llm.token", "run_id": "A", "token": "A2"}, first,
                          {"type": "llm.token", "run_id": "B", "token": "B2"}]:
                app._handle_event(event)
            for run in ("A", "B"):
                app._handle_event({"type": "step.finished", "run_id": run})
            await pilot.pause()
            blocks = list(app.query(LLMStreamBlock))
            assert [block._text for block in blocks] == ["A1A2", "B1", "B2"]
            assert all(block._finalized for block in blocks)
            assert "A" in blocks[0]._label and "B" in blocks[1]._label


# 功能：验证断线期间完成的回复和输入状态在重连时恢复，已收到的前缀不会重复
# 设计：真实 TCP 与 TUI 断线，确认服务端订阅已移除后才完成任务，排除完成事件恰好在线的假通过
@pytest.mark.parametrize("offline_finish", [False, True])
async def test_reconnect_restores_stream_and_actual_busy_state(
    tmp_path: Path, offline_finish: bool,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class Provider:
        # 发出前缀后等待断线控制，再产生尾部；完成的真实文本必须写入磁盘
        async def chat(self, bus, run_id, **kwargs) -> LlmResponse:
            await bus.publish(LlmTokenEvent(run_id=run_id, token="prefix ", ts=datetime.now(UTC).isoformat()))
            entered.set()
            await release.wait()
            await bus.publish(LlmTokenEvent(run_id=run_id, token="suffix", ts=datetime.now(UTC).isoformat()))
            return LlmResponse("end_turn", text="prefix suffix")

    async with _serve(tmp_path, Provider()) as (cfg, manager, bus, store):
        app = XTuiApp(cfg.host, cfg.port)
        async with app.run_test() as pilot:
            async with asyncio.timeout(5):
                while app._session_id is None:
                    await asyncio.sleep(.01)
            sid = app._session_id
            prompt = app.query_one("#prompt", ChatTextArea)
            prompt.text = "finish after disconnect"
            await app.on_chat_text_area_submitted(ChatTextArea.Submitted(prompt))
            await asyncio.wait_for(entered.wait(), 5)
            await pilot.pause()
            assert [block._text for block in app.query(LLMStreamBlock)] == ["prefix "]
            previous = app._client
            assert previous is not None
            await previous.close()
            async with asyncio.timeout(5):
                while app._client is not None or bus._subscribers[0].__self__._subscriptions:
                    await asyncio.sleep(.01)
            if offline_finish:
                release.set()
                async with asyncio.timeout(5):
                    while manager._sessions[sid].status != "waiting_for_input":
                        await asyncio.sleep(.01)
            async with asyncio.timeout(7):
                while app._client is None or app._client is previous or app._session_id != sid:
                    await asyncio.sleep(.01)
            await pilot.pause()
            if not offline_finish:
                assert app._busy and prompt.disabled
                release.set()
            async with asyncio.timeout(5):
                while app._busy:
                    await asyncio.sleep(.01)
            await pilot.pause()
            assert not prompt.disabled and not app.query(PermissionSelect)
            assert [block._text for block in app.query(LLMStreamBlock)] == ["prefix suffix"]
            assert "prefix suffix" in str(store.read_messages(sid, truncate=False))
            # 再次补全不重复已收到的事件，且未产生第二次运行
            await app._subscribe_session()
            await pilot.pause()
            assert [block._text for block in app.query(LLMStreamBlock)] == ["prefix suffix"]
            assert len(manager._sessions[sid].run_ids) == 1


# 功能：验证读取旧会话时补回已完成回复，不恢复历史审批弹窗
# 设计：先完成会话后创建全新 TUI，确保不是旧 App 内存保留了文本
async def test_fresh_resume_replays_completed_conversation(tmp_path: Path) -> None:
    class Provider:
        # 提供可持久化的真实流事件，而非仅返回模型最终文本
        async def chat(self, bus, run_id, **kwargs):
            await bus.publish(LlmTokenEvent(run_id=run_id, token="saved reply", ts=datetime.now(UTC).isoformat()))
            return LlmResponse("end_turn", text="saved reply")

    async with _serve(tmp_path, Provider()) as (cfg, manager, _, store):
        session = await manager.create("chat")
        await manager.send_message(session.id, "original question")
        path = store.runs_dir(session.id) / session.run_ids[-1] / "events.jsonl"
        # 注入一条陈旧审批日志，不能让不存在的 Future 变为可批准请求
        with path.open("a", encoding="utf-8") as stream:
            stream.write('{"type":"permission.requested","run_id":"old","tool_use_id":"old",'
                         '"tool_name":"bash","param_preview":"old","event_id":"old",'
                         '"ts":"2099-01-01T00:00:00Z"}\n')
        app = XTuiApp(cfg.host, cfg.port, resume_session_id=session.id)
        async with app.run_test() as pilot:
            async with asyncio.timeout(5):
                while not app.query(LLMStreamBlock):
                    await asyncio.sleep(.01)
            await pilot.pause()
            assert [block._text for block in app.query(LLMStreamBlock)] == ["saved reply"]
            assert not app.query(PermissionSelect) and not app._busy
            assert not app.query_one("#prompt", ChatTextArea).disabled


# 功能：验证回放与实时事件重叠时不丢失或重复，完成快照与会话状态一致
# 设计：在服务器登记订阅后暂停回放，让真实任务完成并写日志，再放行回放检查两个通道交叠
async def test_live_events_overlapping_replay_are_delivered_once(tmp_path: Path) -> None:
    entered = asyncio.Event()
    finish = asyncio.Event()
    replay_entered = asyncio.Event()
    replay_release = asyncio.Event()
    original = CoreApp._replay_session_events

    # 登记实时订阅之后阻塞回放，以便同一事件同时进入日志与实时暂存区
    async def delayed_replay(app, *args):
        replay_entered.set()
        await replay_release.wait()
        return await original(app, *args)

    class Provider:
        # 两个阶段分别写真实 token 事件，期间由测试协调订阅时序
        async def chat(self, bus, run_id, **kwargs):
            await bus.publish(LlmTokenEvent(run_id=run_id, token="first", ts=datetime.now(UTC).isoformat()))
            entered.set()
            await finish.wait()
            await bus.publish(LlmTokenEvent(run_id=run_id, token="second", ts=datetime.now(UTC).isoformat()))
            return LlmResponse("end_turn", text="firstsecond")

    async with _serve(tmp_path, Provider()) as (cfg, manager, _, _):
        session = await manager.create("chat")
        run = asyncio.create_task(manager.send_message(session.id, "goal"))
        await asyncio.wait_for(entered.wait(), 5)
        client = SocketClient(cfg.host, cfg.port)
        events: list[dict] = []

        # 保存客户端实际收到的所有事件，用于核对去重与最后状态
        async def receive(event):
            events.append(event)

        client.on_event(receive)
        await client.connect()
        loop = asyncio.create_task(client.run_event_loop())
        try:
            with patch.object(CoreApp, "_replay_session_events", delayed_replay):
                subscribe = asyncio.create_task(client.send_command("event.subscribe", {
                    "scope": "session:" + session.id, "replay_session": True,
                    "topics": ["llm.*", "run.*", "session.*"],
                }))
                try:
                    await asyncio.wait_for(replay_entered.wait(), 5)
                    finish.set()
                    await asyncio.wait_for(run, 5)
                finally:
                    replay_release.set()
                await asyncio.wait_for(subscribe, 5)
            assert [event["token"] for event in events if event["type"] == "llm.token"] == ["first", "second"]
            identities = [event["event_id"] for event in events]
            assert len(identities) == len(set(identities))
            states = [event for event in events if event["type"] == "session.synchronized"]
            assert len(states) == 1 and states[0]["busy"] is False
            assert states[0]["status"] == "waiting_for_input"
        finally:
            finish.set()
            await client.close()
            await asyncio.gather(loop, run, return_exceptions=True)


# 功能：验证补发审批等待期间已失效的后续请求不再出现在重连客户端
# 设计：真实 PermissionManager 挂起两个请求，在第一项发送期间解决第二项，检查快照逐项重新核对
async def test_reconnect_does_not_send_expired_pending_snapshot() -> None:
    app = CoreApp()
    manager = PermissionManager(timeout_s=5)
    app._permission_manager = manager
    broadcaster = IpcEventBroadcaster()
    app._broadcaster = broadcaster
    entered = asyncio.Queue()

    # 确认真实请求已登记到审批管理器
    async def emit(event):
        entered.put_nowait(event)

    tasks = [asyncio.create_task(manager.check_and_wait(
        uid, "bash", {"command": "echo ok"}, "sid", emit, run_id="run",
    )) for uid in ("first", "second")]
    sent: list[str] = []

    # 模拟第一个 socket drain 让出执行权期间，第二个审批由其他连接作答
    async def send(writer, event):
        sent.append(event["tool_use_id"])
        assert manager.respond("second", "deny_once", session_id="sid", run_id="run")
        await tasks[1]

    try:
        await asyncio.wait_for(entered.get(), 3)
        await asyncio.wait_for(entered.get(), 3)
        with patch("x_claude.core.app.get_connection_writer", return_value=Mock()), \
                patch.object(broadcaster, "send_pending_permission", side_effect=send):
            await app._subscribe_handler({"scope": "session:sid", "topics": ["permission.*"]})
        assert sent == ["first"]
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
