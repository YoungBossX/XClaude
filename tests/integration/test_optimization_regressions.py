from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Static

from tests.integration.test_background_restart import _checkpoint, _manager, _signature
from tests.integration.test_runtime_hardening import _serve
from tests.unit.test_llm_provider import FakeStream, _make_final
from tests.unit.test_trace_writer import _record
from x_claude.core.bus.events import LlmTokenEvent, RunStartedEvent
from x_claude.core.config import AgentConfig, XConfig, _apply_env, _apply_toml
from x_claude.core.context import ExecutionContext
from x_claude.core.events.bus import EventBus
from x_claude.core.events.writer import EventWriter
from x_claude.core.llm.provider import AnthropicProvider
from x_claude.core.llm.types import LlmResponse, ToolCallBlock, UsageStats
from x_claude.core.loop import AgentLoop
from x_claude.core.permissions.manager import PermissionManager
from x_claude.core.permissions.policy import PermissionDecision, ToolPolicy
from x_claude.core.resources import LimitedProvider, ResourceLimitExceeded, TaskBudget
from x_claude.core.runner import AgentRunner
from x_claude.core.session.manager import SessionManager
from x_claude.core.session.store import SessionStore
from x_claude.core.subagent.checkpoint import BackgroundCheckpoint
from x_claude.core.subagent.tool import AgentResultTool
from x_claude.core.tools.builtin.write_file import WriteFileTool
from x_claude.core.tools.registry import ToolRegistry
from x_claude.core.trace.writer import TraceWriter
from x_claude.core.transport.socket_client import SocketClient
from x_claude.tui.app import LLMStreamBlock, XTuiApp
from x_claude.tui.history import HistoryScreen
from x_claude.tui.replay import ReplayProgress


# 所有文件和会话均放在临时工作目录，不调用真实模型服务
@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


class DroppedStream(FakeStream):
    # 在指定文本后模拟真实流读取异常，覆盖已有前缀与尚未输出两种断流
    @property
    def text_stream(self):
        async def generate():
            for text in self._texts:
                yield text
            raise httpx.ReadError("simulated disconnect")
        return generate()


# 功能：验证模型重试后 TUI 和保存的最终回复一致，不保留旧前缀或重复拼接
# 设计：真实 TCP、真实 TUI 与 AnthropicProvider 共用流替身，覆盖一次及连续两次断流
@pytest.mark.parametrize("failures,prefix", [(1, []), (1, ["partial"]), (2, ["partial"])])
async def test_retry_reconciles_tui_and_persisted_reply(tmp_path, failures, prefix) -> None:
    client = MagicMock()
    client.messages.stream.side_effect = [
        *[DroppedStream(prefix, _make_final()) for _ in range(failures)],
        FakeStream(["complete", " reply"], _make_final()),
    ]
    provider = AnthropicProvider("test", client=client, context_window=200000)
    with patch("x_claude.core.llm.provider._RETRY_BACKOFF_S", (0, 0, 0)):
        async with _serve(tmp_path, provider) as (cfg, manager, _, store):
            app = XTuiApp(cfg.host, cfg.port)
            async with app.run_test() as pilot:
                async with asyncio.timeout(5):
                    while app._session_id is None:
                        await asyncio.sleep(.01)
                await manager.send_message(app._session_id, "reply")
                await pilot.pause()
                assert [b._text for b in app.query(LLMStreamBlock)] == ["complete reply"]
                assert "complete reply" in str(store.read_messages(app._session_id))
                assert client.messages.stream.call_count == failures + 1


# 功能：验证模型失败具有稳定分类和可执行提示，前端事件不泄露原始敏感错误
# 设计：将真实 SDK 异常注入正常 AgentLoop，检查最终状态与发布的结构化事件
@pytest.mark.parametrize("kind,code", [
    (401, "llm_auth"), (429, "llm_rate_limit"), (400, "llm_context_limit"),
    (503, "llm_unavailable"), ("timeout", "llm_timeout"), ("network", "llm_connection"),
])
async def test_llm_failure_classification(kind, code) -> None:
    if isinstance(kind, int):
        response = httpx.Response(kind, request=httpx.Request("POST", "https://example.invalid"))
        error = anthropic.APIStatusError("secret-api-key", response=response,
                                         body={"message": "maximum context length exceeded"})
    else:
        error = httpx.ReadTimeout("secret-api-key") if kind == "timeout" else httpx.ConnectError("secret-api-key")
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=error)
    bus, captured = EventBus(), []

    # 捕获实际循环发布的安全错误描述
    async def record(event):
        captured.append(event.model_dump())

    bus.subscribe(record)
    context = ExecutionContext("r", "goal", 2)
    await AgentLoop(provider, ToolRegistry(), bus).run(context)
    failures = [e for e in captured if e["type"] == "llm.error"]
    assert context.reason == code and len(failures) == 1
    assert failures[0]["hint"] and "secret-api-key" not in json.dumps(failures)


# 功能：验证日志打开、写入或 flush 失败均能结束 stop，且不再接收无限记录
# 设计：直接注入三种文件故障，检查消费者结束与所有队列任务配平
@pytest.mark.parametrize("failure", ["open", "write", "flush"])
async def test_trace_failure_never_strands_shutdown(tmp_path, failure) -> None:
    writer = TraceWriter(tmp_path / "trace.jsonl")
    stream = MagicMock()
    stream.__enter__.return_value = stream
    if failure != "open":
        getattr(stream, failure).side_effect = OSError("simulated disk failure")
    with patch.object(Path, "open", side_effect=OSError("open failed") if failure == "open" else None,
                      return_value=stream):
        await writer.start()
        for _ in range(100):
            writer.emit(_record())
        await asyncio.wait_for(writer.stop(), 1)
    assert writer.failed and writer._task.done()
    assert writer.dropped_records == 100
    writer.emit(_record())
    assert writer._queue.empty()
    await asyncio.wait_for(writer._queue.join(), .1)


# 功能：验证队列上限和从未启动的日志通道不会造成内存无限增长或关闭等待
# 设计：在消费者启动前填满两格队列，超额记录被计数，stop 丢弃剩余数据
async def test_trace_queue_is_bounded_before_start(tmp_path) -> None:
    writer = TraceWriter(tmp_path / "trace.jsonl", queue_size=2)
    for _ in range(10):
        writer.emit(_record())
    assert writer._queue.qsize() == 2 and writer.dropped_records == 8
    await asyncio.wait_for(writer.stop(), .2)
    assert writer.dropped_records == 10


# 功能：验证乱序到达不能越过未收到的事件，有界 ID 缓存淘汰后仍能通过偏移去重
# 设计：先接收后半段，再补前半段，并投递五千个事件验证连续水位与空间上限
def test_replay_cursor_waits_for_gaps_and_bounds_ids() -> None:
    progress = ReplayProgress()
    second = {"event_id": "second", "log_positions": {"r": [10, 20]}}
    progress.accept(second)
    assert progress.offsets["r"] == 0
    progress.accept({"event_id": "first", "log_positions": {"r": [0, 10]}})
    assert progress.offsets["r"] == 20
    for n in range(5000):
        progress.accept({"event_id": str(n), "log_positions": {"r": [20+n, 21+n]}})
    assert len(progress.seen) <= 4096 and progress.duplicate(second)


# 功能：验证真实 IPC 回放只发送断线期间新增日志，保存偏移可跨服务端重建继续使用
# 设计：创建两个真实 EventWriter 日志，第二次服务启动仅回放后追加的中文事件
async def test_replay_offsets_survive_server_restart(tmp_path) -> None:
    progress = ReplayProgress()
    async with _serve(tmp_path, object()) as (cfg, manager, _, store):
        session = await manager.create("chat")
        path = store.runs_dir(session.id) / "run-a" / "events.jsonl"
        async with EventWriter(path) as writer:
            await writer.handle(RunStartedEvent(run_id="run-a", goal="goal", ts="1"))
            for index in range(120):
                await writer.handle(LlmTokenEvent(run_id="run-a", token=f"字{index}", ts="2"))
        client = SocketClient(cfg.host, cfg.port)

        # 按实际到达顺序记录游标，不从测试文件直接构造已消费位置
        async def receive(event):
            progress.accept(event)

        await client.connect()
        client.on_event(receive)
        loop = asyncio.create_task(client.run_event_loop())
        try:
            result = await client.send_command("event.subscribe", {
                "scope": "session:" + session.id, "topics": ["*"], "replay_session": True,
            })
            progress.synchronize(result["replay_offsets"])
            assert result["replayed_count"] == 121
        finally:
            await client.close()
            await asyncio.gather(loop, return_exceptions=True)
    async with EventWriter(path) as writer:
        await writer.handle(LlmTokenEvent(run_id="run-a", token="offline suffix", ts="3"))
    async with _serve(tmp_path, object()) as (cfg, manager, _, _):
        await manager.resume(session.id)
        client = SocketClient(cfg.host, cfg.port)
        captured = []

        # 收集重启后真正收到的后缀事件
        async def receive_suffix(event):
            captured.append(event)

        await client.connect()
        client.on_event(receive_suffix)
        loop = asyncio.create_task(client.run_event_loop())
        try:
            result = await client.send_command("event.subscribe", {
                "scope": "session:" + session.id, "topics": ["*"], "replay_session": True,
                "replay_offsets": progress.offsets,
            })
            assert result["replayed_count"] == 1
            assert [e["token"] for e in captured if e["type"] == "llm.token"] == ["offline suffix"]
        finally:
            await client.close()
            await asyncio.gather(loop, return_exceptions=True)


# 功能：验证历史分页完整覆盖中文和跨块长行，文件更新后拒绝旧游标
# 设计：逆向读取大于 64 KiB 的消息并逐页重组，禁止使用 read_text 全量读取捷径
def test_history_pages_are_complete_and_versioned(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    contents = ["消息" + str(n) for n in range(41)]
    contents[19] = "中文" * 40000
    for content in contents:
        store.append_message("sid", "user", content)
    cursor, rebuilt = None, []
    with patch.object(Path, "read_text", side_effect=AssertionError("must stream history")):
        while True:
            messages, cursor = store.history_page("sid", cursor, 7)
            rebuilt = [m["content"] for m in messages] + rebuilt
            if cursor is None:
                break
    assert rebuilt == contents
    _, cursor = store.history_page("sid", None, 2)
    store.append_message("sid", "assistant", "new")
    with pytest.raises(ValueError, match="对话已更新"):
        store.history_page("sid", cursor, 2)


# 功能：验证历史预览在传输前截断长正文和工具数据，不修改磁盘原文
# 设计：通过真实 TCP 请求两种超长消息，对比有界响应和读取前后的完整历史字节
async def test_history_preview_is_bounded_before_transmission(tmp_path) -> None:
    async with _serve(tmp_path, object()) as (cfg, manager, _, store):
        session = await manager.create("chat")
        store.append_message(session.id, "user", "中文" * 20000)
        store.append_message(session.id, "assistant", [{"type": "text", "text": "内容" * 20000}])
        path = store.session_dir(session.id) / "thread.jsonl"
        original = path.read_bytes()
        client = SocketClient(cfg.host, cfg.port)
        await client.connect()
        loop = asyncio.create_task(client.run_event_loop())
        try:
            result = await client.send_command("session.history_page", {
                "session_id": session.id, "limit": 20,
            })
            assert len(result["messages"]) == 2
            for message in result["messages"]:
                assert len(message["content"]) <= 16000
                assert "显示已截断" in message["content"]
            assert path.read_bytes() == original
        finally:
            await client.close()
            await asyncio.gather(loop, return_exceptions=True)


# 功能：验证模型并发受限，主子调用共用额度，排队取消不泄漏名额
# 设计：真实协程交错二十次模型请求，记录同时执行峰值并重载预算账本
async def test_model_gate_and_budget_are_shared_and_durable(tmp_path) -> None:
    active, peak = 0, 0

    class Provider:
        # 等待一次事件循环以制造真实并发重叠
        async def chat(self, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(.01)
                return LlmResponse("end_turn", text="done", usage=UsageStats(10, 2))
            finally:
                active -= 1

    config = AgentConfig(max_total_tokens=200000)
    budget = TaskBudget("root", config, tmp_path / "budget.json")
    gate = asyncio.Semaphore(2)
    provider = LimitedProvider(Provider(), gate, budget)
    await asyncio.gather(*[provider.chat([], [], EventBus(), f"child-{n}") for n in range(20)])
    assert peak == 2 and active == 0 and budget.used_tokens == 240
    restored = TaskBudget("root", config, tmp_path / "budget.json")
    assert restored.used_tokens == 240 and not restored.reservations
    await gate.acquire()
    await gate.acquire()
    waiting = asyncio.create_task(provider.chat([], [], EventBus(), "cancelled"))
    await asyncio.sleep(0)
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    gate.release()
    gate.release()
    await asyncio.wait_for(provider.chat([], [], EventBus(), "after"), 1)


# 功能：验证预留预算及任务数跨重启仍生效，零预算和时限可显式关闭
# 设计：失败请求不结算预留，重载后不能再次花费同一额度或重新派生超额任务
def test_failed_reservations_and_task_cap_survive_reload(tmp_path) -> None:
    config = AgentConfig(max_total_tokens=10000, max_tasks=2, max_runtime_s=0)
    path = tmp_path / "budget.json"
    budget = TaskBudget("root", config, path)
    budget.register("child")
    budget.reserve(9000)
    restored = TaskBudget("root", config, path)
    with pytest.raises(ResourceLimitExceeded, match="数量"):
        restored.register("other")
    with pytest.raises(ResourceLimitExceeded, match="Token"):
        restored.reserve(2000)
    config.max_total_tokens = 0
    restored.reserve(500000)


# 功能：验证完整 Runner 会限制后台派生数量，限制不会阻止已准入任务完成
# 设计：一次工具响应请求五个后台 Agent，只允许两个，检查真实工具结果及持久预算任务数
async def test_runner_enforces_tree_task_count(tmp_path) -> None:
    class Provider:
        # 根任务派生子任务，子任务立即返回，以便检查前台与后台共用额度
        async def chat(self, messages, step, **kwargs):
            if messages[0]["content"] == "root goal" and step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    str(n), "spawn_agent", {"description": "child", "prompt": "child goal", "run_in_background": True},
                ) for n in range(5)])
            return LlmResponse("end_turn", text="done", usage=UsageStats(10, 2))

    cfg = XConfig()
    cfg.agent.max_tasks = 3
    runner = AgentRunner(cfg, provider=Provider(), runs_dir=tmp_path / "runs")
    try:
        outcome = await runner.run_and_capture("root goal", run_id="root")
        await asyncio.gather(*[task for task, _ in runner._task_registry.all()])
        assert outcome.status == "success" and len(runner._task_registry.all()) == 2
        raw = json.loads((tmp_path / "runs/root/budget.json").read_text())
        assert len(raw["tasks"]) == 3
    finally:
        await runner.shutdown()


# 功能：验证总时限可以中断无限等待的模型调用，并产生可理解的失败状态
# 设计：实际 Runner 在模型等待事件期间达到时限，外层仅设宽松保护以排除测试自行取消
async def test_tree_runtime_deadline_ends_hung_model(tmp_path) -> None:
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=lambda **kwargs: None)

    # 永不返回的模型替身用于验证预算超时，而非 SDK 自身超时
    async def hang(**kwargs):
        await asyncio.Event().wait()

    provider.chat = hang
    cfg = XConfig()
    cfg.agent.max_runtime_s = .05
    runner = AgentRunner(cfg, provider=provider, runs_dir=tmp_path / "runs")
    result = await asyncio.wait_for(runner.run_and_capture("goal"), 1)
    assert result.status == "failed" and result.reason == "runtime_budget"


# 功能：验证配置文件和环境变量资源限制生效，并拒绝 NaN、布尔值及非法零并发
# 设计：覆盖实际配置解析入口，保证运行时限制不会因错误类型被绕过
def test_resource_config_validation(monkeypatch) -> None:
    config = XConfig()
    _apply_toml(config, {"agent": {"max_tasks": 7, "max_total_tokens": 0}})
    monkeypatch.setenv("X_MAX_CONCURRENT_LLM", "2")
    _apply_env(config)
    assert config.agent.max_tasks == 7 and config.agent.max_concurrent_llm == 2
    for key, value in [("max_runtime_s", float("nan")), ("max_tasks", True), ("max_concurrent_llm", 0)]:
        with pytest.raises(SystemExit):
            _apply_toml(config, {"agent": {key: value}})


# 功能：验证大量事件后的 TUI 控件有界且不会移除正在输出的文本块
# 设计：挂载超过上限的真实控件，完成布局与移除后检查实际 DOM 数量和活动引用
async def test_tui_log_prunes_finished_widgets_only() -> None:
    app = XTuiApp("127.0.0.1", 0)
    with patch.object(app, "_socket_loop", AsyncMock()):
        async with app.run_test() as pilot:
            app._handle_event({"type": "llm.token", "run_id": "active", "token": "prefix"})
            for n in range(650):
                app._append(Static(str(n)))
            await pilot.pause()
            await pilot.pause()
            assert len(app.query_one("#log-view", VerticalScroll).children) <= 600
            app._handle_event({"type": "llm.token", "run_id": "active", "token": " suffix"})
            assert app._llm_streams["active"]._text == "prefix suffix"


# 功能：验证真实历史窗口可前后翻页并关闭，浏览期间新事件仍正常进入主界面
# 设计：通过真实 IPC 读取四十五条消息，按钮操作只产生历史请求且不增加模型运行
async def test_history_screen_pages_without_starting_runs(tmp_path) -> None:
    async with _serve(tmp_path, object()) as (cfg, manager, _, store):
        session = await manager.create("chat")
        for n in range(45):
            store.append_message(session.id, "user", f"history-{n:02d}")
        app = XTuiApp(cfg.host, cfg.port, resume_session_id=session.id)
        async with app.run_test(size=(110, 35)) as pilot:
            async with asyncio.timeout(5):
                while app._client is None or app._session_id is None:
                    await asyncio.sleep(.01)
            await pilot.pause()
            app._show_history()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, HistoryScreen)
            assert "history-44" in str(screen.query_one("#history-text", Static).render())
            await pilot.click("#history-older")
            await pilot.pause()
            assert "history-24" in str(screen.query_one("#history-text", Static).render())
            assert not screen.query_one("#history-newer", Button).disabled
            app._handle_event({"type": "llm.token", "run_id": "live", "token": "still arriving"})
            await pilot.click("#history-newer")
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, HistoryScreen)
            assert app._llm_streams["live"]._text == "still arriving"
            assert not session.run_ids


# 功能：验证单模型名额下前台嵌套 Agent 不会因父任务等待子任务而死锁
# 设计：真实 Runner 通过工具派生前台任务，共用一个 semaphore，要求两层循环均成功
async def test_foreground_child_does_not_hold_parent_model_slot(tmp_path) -> None:
    class Provider:
        # 仅根任务第一次请求派生，其余请求正常结束
        async def chat(self, messages, step, **kwargs):
            if messages[0]["content"] == "root goal" and step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "spawn", "spawn_agent", {"description": "child", "prompt": "child goal"},
                )])
            return LlmResponse("end_turn", text="done", usage=UsageStats(1, 1))

    cfg = XConfig()
    cfg.agent.max_concurrent_llm = 1
    runner = AgentRunner(cfg, provider=Provider(), runs_dir=tmp_path / "runs")
    try:
        outcome = await asyncio.wait_for(runner.run_and_capture("root goal"), 2)
        assert outcome.status == "success"
    finally:
        await runner.shutdown()


# 功能：验证恢复的后台任务仍能派生前台子任务，并继承模型与预算
# 设计：检查点仅允许派生工具，经完整 Runner 恢复路径执行两层循环并断言工具结果
async def test_restored_child_can_spawn_with_restored_provider(tmp_path) -> None:
    class Provider:
        # 恢复任务在第二步派生，第三步必须观察到子任务的真实回复
        async def chat(self, messages, step, **kwargs):
            if messages[0]["content"] == "child-goal":
                if step == 2:
                    return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                        "spawn", "spawn_agent", {"description": "nested", "prompt": "nested"},
                    )])
                assert messages[-1]["content"][0]["content"] == "nested result"
                return LlmResponse("end_turn", text="restored result")
            return LlmResponse("end_turn", text="nested result")

    store = SessionStore(tmp_path / "sessions")
    manager = _manager(store, Provider())
    session = await manager.create("chat")
    checkpoint = _checkpoint(store, session.id)
    checkpoint.record.tools = ["spawn_agent"]
    checkpoint.record.runtime_signature = _signature(["spawn_agent"])
    checkpoint.save(checkpoint.record.context.restore("child"), "planning")
    try:
        await manager.recover_background(session.id)
        registry = manager._runners[session.id]._task_registry
        await asyncio.wait_for(asyncio.gather(*(task for task, _ in registry.all())), 2)
        result = await AgentResultTool(registry).invoke({"run_id": "child"})
        assert result.content == "restored result" and not result.is_error
    finally:
        await manager.shutdown()


# 功能：验证预算账本损坏时隔离该任务的恢复错误，不中断整个会话恢复
# 设计：保留有效检查点但破坏关联预算文件，确认返回可查询错误且不调用模型
async def test_corrupt_budget_does_not_abort_session_recovery(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    manager = _manager(store, None)
    session = await manager.create("chat")
    checkpoint = _checkpoint(store, session.id)
    parent = store.runs_dir(session.id) / "parent"
    parent.mkdir()
    (parent / "budget.json").write_text("invalid json", encoding="utf-8")
    before = checkpoint.path.read_bytes()
    try:
        await manager.recover_background(session.id)
        registry = manager._runners[session.id]._task_registry
        result = await AgentResultTool(registry).invoke({"run_id": "child"})
        assert result.is_error and "recovery_error" in result.content
        assert checkpoint.path.read_bytes() == before
    finally:
        await manager.shutdown()


# 功能：验证工具副作用发生后达到总时限时保留未确认检查点，恢复不会自动重放
# 设计：在真实写文件后阻塞工具返回，检查文件、检查点状态和恢复后的模型调用次数
async def test_runtime_limit_preserves_uncertain_tool_checkpoint(tmp_path) -> None:
    target = tmp_path / "changed.txt"
    calls = 0

    class Provider:
        # 返回需审批的写文件请求，统计恢复是否错误地重新请求模型
        async def chat(self, **kwargs):
            nonlocal calls
            calls += 1
            return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                "write", "write_file", {"path": str(target), "content": "once"},
            )])

    # 在工具已产生副作用、但尚未返回结果的窗口内等待总时限
    async def interrupted_write(tool, params):
        target.write_text("once", encoding="utf-8")
        await asyncio.Event().wait()

    cfg = XConfig()
    cfg.agent.max_runtime_s = .15
    bus = EventBus()
    store = SessionStore(tmp_path / "sessions")
    permission = PermissionManager({"write_file": ToolPolicy(PermissionDecision.ALLOW)})
    manager = SessionManager(store, lambda: AgentRunner(
        cfg, bus=bus, provider=Provider(), permission_manager=permission,
    ), bus)
    session = await manager.create("chat")
    try:
        with patch.object(WriteFileTool, "invoke", interrupted_write):
            run_id = await asyncio.wait_for(manager.send_message(session.id, "write"), 2)
        checkpoint = BackgroundCheckpoint.load(store.runs_dir(session.id) / run_id / "root.json", session.id)
        assert target.read_text() == "once"
        assert checkpoint.record.state == "blocked" and checkpoint.pending_tools()[0]["id"] == "write"
        await manager.recover_background(session.id)
        assert calls == 1
    finally:
        await manager.shutdown()
