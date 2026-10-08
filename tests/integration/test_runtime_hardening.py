from __future__ import annotations

import asyncio
import copy
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from x_claude.cli.commands import chat, run
from x_claude.core.app import CoreApp
from x_claude.core.bus.events import LlmTokenEvent, RunStartedEvent, SubagentStartedEvent
from x_claude.core.config import XConfig
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.types import LlmResponse, ToolCallBlock
from x_claude.core.mcp.client import McpClient, McpToolDef
from x_claude.core.mcp.tool import McpTool
from x_claude.core.permissions.manager import PermissionManager
from x_claude.core.runner import AgentRunner
from x_claude.core.session.manager import SessionManager
from x_claude.core.session.store import SessionStore
from x_claude.core.tools.builtin.bash import BashTool
from x_claude.core.tools.invocation import invoke_tool
from x_claude.core.tools.registry import ToolRegistry
from x_claude.core.transport.ipc_broadcaster import IpcEventBroadcaster
from x_claude.core.transport.socket_server import SocketServer
from x_claude.tui.app import XTuiApp


# 将临时目录作为测试项目，文件工具不能再绕过真实工作目录边界
@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


# 在真实 TCP 服务上连接现有 Core handler，模型使用调用者指定的替身
@asynccontextmanager
async def _serve(root: Path, provider: object, port: int = 0, permission_timeout: float = 1):
    bus = EventBus()
    broadcaster = IpcEventBroadcaster()
    bus.subscribe(broadcaster.handle)
    store = SessionStore(root / "sessions")
    cfg = XConfig()
    pm = PermissionManager(timeout_s=permission_timeout)
    manager = SessionManager(store, lambda: AgentRunner(
        cfg, bus=bus, provider=provider, permission_manager=pm,
    ), bus)
    app = CoreApp()
    app._sessions = manager
    app._broadcaster = broadcaster
    app._permission_manager = pm
    server = SocketServer("127.0.0.1", port, broadcaster)
    for method, handler in [
        ("session.create", app._session_create_handler),
        ("session.send_message", app._session_send_handler),
        ("session.close", app._session_close_handler),
        ("session.clear", app._session_clear_handler),
        ("session.recover", app._session_recover_handler),
        ("session.resume", app._session_resume_handler),
        ("session.continue", app._session_continue_handler),
        ("event.subscribe", app._subscribe_handler),
        ("permission.respond", app._permission_respond_handler),
    ]:
        server.register(method, handler)
    await server.start()
    cfg.port = server._server.sockets[0].getsockname()[1]
    try:
        with patch("x_claude.core.runner.load_context_file", return_value=""):
            yield cfg, manager, bus, store
    finally:
        await manager.shutdown()
        await server.stop()


# 功能：验证 CLI 在整轮运行仍未完成时即可批准或拒绝工具调用，审批输入不进入对话
# 设计：使用真实 TCP、现有 Core handler、真实权限管理器和临时文件，模拟 stdin，不调用真实 LLM
@pytest.mark.parametrize("answer", ["y", "n"])
async def test_cli_approval_during_inflight_run(tmp_path: Path, answer: str) -> None:
    goals: list = []
    target = tmp_path / "approved.txt"

    class Provider:
        # 先请求写文件，再完成运行；记录新一轮输入以检测 y/n 被错误当成聊天
        async def chat(self, messages: list, step: int, **kwargs: object) -> LlmResponse:
            if step == 1:
                goals.append(messages[-1]["content"])
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "write", "write_file", {"path": str(target), "content": "approved"},
                )])
            return LlmResponse("end_turn", text="done")

    inputs = iter(["write something", answer])

    # 固定用户输入序列，权限提示出现时才返回审批答案
    async def readline(prompt: str) -> str:
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError() from None

    async with _serve(tmp_path, Provider()) as (cfg, _, bus, _):
        events: list = []

        # 捕获真实运行事件，避免只以客户端退出码判断是否审批成功
        async def record(event: object) -> None:
            events.append(event.model_dump())

        bus.subscribe(record)
        with patch.object(chat, "_readline", side_effect=readline):
            assert await asyncio.wait_for(chat._chat_async(cfg), 4) == 0
    assert goals == ["write something"]
    assert target.exists() == (answer == "y")
    decisions = [e for e in events if e["type"] in ("permission.granted", "permission.denied")]
    assert len(decisions) == 1
    assert decisions[0]["decision"] == ("allow_once" if answer == "y" else "deny_once")


# 功能：验证同时运行的两个 CLI 一次性任务只接收自己的进度，并返回各自成功或失败退出码
# 设计：真实 TCP 上用模型屏障强制重叠，一次成功一次故障，捕获每个真实打印器收到的事件
async def test_parallel_cli_runs_keep_exit_status_isolated(tmp_path: Path) -> None:
    ready = asyncio.Event()
    entered = 0
    seen: dict[object, list[dict]] = {}
    handle = run.StdoutPrinter.handle

    class Provider:
        # 让两个运行同时进入推理，再按输入分别返回成功和异常
        async def chat(self, messages: list, **kwargs: object) -> LlmResponse:
            nonlocal entered
            entered += 1
            if entered == 2:
                ready.set()
            await ready.wait()
            if messages[-1]["content"] == "fail-goal":
                raise RuntimeError("intentional model failure")
            return LlmResponse("end_turn", text="done")

    # 保留打印逻辑，同时按客户端打印器对象区分事件流
    async def record(printer: run.StdoutPrinter, event: dict) -> None:
        seen.setdefault(printer, []).append(event)
        await handle(printer, event)

    async with _serve(tmp_path, Provider()) as (cfg, manager, _, store):
        with patch.object(run.StdoutPrinter, "handle", record):
            codes = await asyncio.wait_for(asyncio.gather(
                run._run_async("success-goal", cfg), run._run_async("fail-goal", cfg),
            ), 4)
        assert codes == [0, 1]
        assert len(seen) == 2
        for events in seen.values():
            assert len({event["session_id"] for event in events}) == 1
            assert len({event["run_id"] for event in events}) == 1
            assert sum(event["type"] == "run.finished" for event in events) == 1
        assert all(session.status == "closed" for session in manager._sessions.values())
        assert all(store.read_meta(sid).status == "closed" for sid in manager._sessions)


# 功能：验证 TUI 只显示自身会话的事件，/clear 后订阅切换到新会话
# 设计：在 Textual 无头运行环境连接真实 TCP，同时执行其他会话，检查实际渲染入口而非只检查 scope 字符串
async def test_tui_isolation_and_clear_resubscribe(tmp_path: Path) -> None:
    class Provider:
        # 模拟模型流式输出，确保走真实 EventBus 和 IPC 推送路径
        async def chat(self, messages: list, bus: EventBus, run_id: str,
                       **kwargs: object) -> LlmResponse:
            await bus.publish(LlmTokenEvent(run_id=run_id, token=str(messages[-1]["content"]), ts="t"))
            return LlmResponse("end_turn", text="done")

    async with _serve(tmp_path, Provider()) as (cfg, manager, _, _):
        app = XTuiApp(cfg.host, cfg.port)
        seen: list[dict] = []
        handle = app._handle_event

        # 记录真正从 SocketClient 收到的事件，再交给原始 TUI 渲染方法
        def record(event: dict) -> None:
            seen.append(event)
            handle(event)

        app._handle_event = record
        async with app.run_test(size=(100, 30)) as pilot:
            async with asyncio.timeout(3):
                while app._session_id is None or app._client is None:
                    await asyncio.sleep(0.01)
                await pilot.pause()
            original = app._session_id
            other = await manager.create("chat")
            await manager.send_message(other.id, "unrelated-stream")
            await pilot.pause()
            assert not any(e.get("token") == "unrelated-stream" for e in seen)
            await manager.send_message(original, "own-stream")
            await pilot.pause()
            assert any(e.get("token") == "own-stream" for e in seen)
            await app._do_clear()
            fresh = app._session_id
            assert fresh != original
            seen.clear()
            await manager.send_message(fresh, "fresh-stream")
            await pilot.pause()
            assert any(e.get("token") == "fresh-stream" for e in seen)
            assert all(e.get("session_id") == fresh for e in seen)


# 功能：验证会话订阅保留子 Agent 事件，同时过滤其他会话及旧会话
# 设计：父子事件按实际发布顺序进入 broadcaster，捕获 JSON 字节并测试同连接重新订阅
async def test_session_scope_includes_children_and_replaces_old_subscription() -> None:
    from unittest.mock import AsyncMock, MagicMock

    writer = MagicMock()
    writer.drain = AsyncMock()
    broadcaster = IpcEventBroadcaster()
    broadcaster.subscribe(writer, ["*"], "session:s1")
    await broadcaster.handle(RunStartedEvent(run_id="r1", session_id="s1", goal="g", ts="t"))
    await broadcaster.handle(RunStartedEvent(run_id="r2", session_id="s2", goal="g", ts="t"))
    await broadcaster.handle(SubagentStartedEvent(
        run_id="child", parent_run_id="r1", description="child", ts="t",
    ))
    await broadcaster.handle(LlmTokenEvent(run_id="child", token="child-result", ts="t"))
    records = [json.loads(call.args[0])["event"] for call in writer.write.call_args_list]
    assert [e["run_id"] for e in records] == ["r1", "child", "child"]
    assert all(e["session_id"] == "s1" for e in records)
    broadcaster.subscribe(writer, ["*"], "session:s2")
    writer.write.reset_mock()
    await broadcaster.handle(LlmTokenEvent(run_id="r1", token="old", ts="t"))
    await broadcaster.handle(LlmTokenEvent(run_id="r2", token="new", ts="t"))
    assert writer.write.call_count == 1


# 功能：验证 daemon 重启后的窄 topic 回放仍包含子 Agent，且不夹带其他运行
# 设计：父子关系只在未订阅的 started 事件中声明，读取真实 UTF-8 JSONL 后仅回放 token
async def test_replay_learns_hierarchy_before_topic_filter(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, MagicMock

    path = tmp_path / "events.jsonl"
    events = [
        {"type": "run.started", "run_id": "root", "session_id": "s1", "goal": "g", "ts": "t"},
        {"type": "subagent.started", "run_id": "child", "parent_run_id": "root", "ts": "t"},
        {"type": "llm.token", "run_id": "child", "token": "子任务结果", "ts": "t"},
        {"type": "run.started", "run_id": "foreign", "session_id": "s2", "goal": "g", "ts": "t"},
        {"type": "llm.token", "run_id": "foreign", "token": "不应回放", "ts": "t"},
    ]
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events), encoding="utf-8")
    app = CoreApp()
    app._broadcaster = IpcEventBroadcaster()
    writer = MagicMock()
    writer.drain = AsyncMock()
    with patch("x_claude.core.app.events_file", return_value=path):
        count = await app._replay_events("root", writer, ["llm.token"], scope="run:root")
    assert count == 1
    record = json.loads(writer.write.call_args.args[0])["event"]
    assert record["token"] == "子任务结果" and record["session_id"] == "s1"


# 功能：验证后台子 Agent 结果可跨轮获取，但不能从另一会话获取
# 设计：使用真实 SessionManager/Runner/子循环，控制后台完成时机并检查模型收到的工具结果
@pytest.mark.parametrize("failed_child", [False, True])
async def test_background_result_survives_next_turn_and_is_session_isolated(
    tmp_path: Path, failed_child: bool,
) -> None:
    gate = asyncio.Event()
    child_id = ""
    results: list = []

    class Provider:
        # 父 Agent 派生后台任务，后续轮次查询；子 Agent 等待屏障后返回独立结果
        async def chat(self, messages: list, step: int, **kwargs: object) -> LlmResponse:
            if messages[0]["content"] == "child-goal":
                await gate.wait()
                if failed_child:
                    raise RuntimeError("child model failed")
                return LlmResponse("end_turn", text="child-result")
            if step == 1:
                if messages[-1]["content"] == "launch":
                    return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                        "spawn", "spawn_agent", {"description": "test", "prompt": "child-goal",
                                                 "run_in_background": True},
                    )])
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "query", "agent_result", {"run_id": child_id},
                )])
            if messages[-1]["content"][0].get("tool_use_id") == "query":
                results.append(copy.deepcopy(messages[-1]["content"][0]))
            return LlmResponse("end_turn", text="parent-done")

    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: AgentRunner(XConfig(), provider=Provider()), EventBus())
    session = await manager.create("chat")
    try:
        await manager.send_message(session.id, "launch")
        registry = manager._runners[session.id]._task_registry
        child_id = next(iter(registry._tasks))
        gate.set()
        await asyncio.gather(*(task for task, _ in registry.all()))
        await manager.send_message(session.id, "collect")
        if failed_child:
            assert results[-1]["is_error"] and "llm_error" in results[-1]["content"]
        else:
            assert results[-1]["content"] == "child-result"
            assert not results[-1].get("is_error")
        other = await manager.create("chat")
        await manager.send_message(other.id, "collect")
        assert results[-1]["is_error"]
        assert "Unknown run_id" in results[-1]["content"]
    finally:
        await manager.shutdown()


# 功能：验证 Skill 展开后的参数真正出现在系统提示中，同时保留原始用户历史
# 设计：使用内建 /review 和真实会话执行链，捕获模型实际入参，排除只测试 loader 的假通过
async def test_skill_arguments_reach_provider(tmp_path: Path) -> None:
    captured: dict = {}

    class Provider:
        # 捕获真实 system 和 messages，不读取目标文件
        async def chat(self, messages: list, system: str, **kwargs: object) -> LlmResponse:
            captured.update(system=system, messages=copy.deepcopy(messages))
            return LlmResponse("end_turn", text="done")

    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: AgentRunner(XConfig(), provider=Provider()), EventBus())
    session = await manager.create("chat")
    await manager.send_message(session.id, "/review audit-target")
    assert "$ARGUMENTS" not in captured["system"]
    assert "audit-target" in captured["system"]
    assert captured["messages"][0]["content"] == "/review audit-target"
    await manager.shutdown()


_MCP_STDIO = '''
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    if "id" not in req: continue
    method = req["method"]
    if method == "initialize": result = {"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"test","version":"1"}}
    elif method == "tools/list": result = {"tools":[{"name":"sample","inputSchema":{"type":"object","required":["path"],"properties":{"path":{"type":"string"}}}}]}
    else: result = {"isError": req["params"]["arguments"]["path"] == "fail","content":[{"type":"text","text":"tool-output"}]}
    print(json.dumps({"jsonrpc":"2.0","id":req["id"],"result":result}), flush=True)
'''


# 功能：验证 MCP stdio 完整握手、工具发现、参数校验和远端工具错误传播
# 设计：启动独立的标准 NDJSON 子进程，不 mock McpClient；统计工具调用以确认非法参数不会发往服务端
async def test_mcp_stdio_schema_and_error_flag() -> None:
    client = McpClient()
    try:
        await client.connect_stdio(sys.executable, ["-u", "-c", _MCP_STDIO])
        definitions = await client.list_tools()
        tool = McpTool(client, "test", definitions[0])
        registry = ToolRegistry()
        registry.register(tool)
        with patch.object(client, "call_tool", wraps=client.call_tool) as calls:
            bad = await invoke_tool(registry, ToolCallBlock("bad", tool.name, {}), EventBus(), "r")
            assert bad.is_error and bad.error_type == "schema_error"
            calls.assert_not_called()
            failed = await invoke_tool(registry, ToolCallBlock(
                "fail", tool.name, {"path": "fail"}), EventBus(), "r")
            assert failed.is_error and "tool-output" in failed.content
            assert calls.call_count == 1
            ok = await invoke_tool(registry, ToolCallBlock(
                "ok", tool.name, {"path": "ok"}), EventBus(), "r")
            assert not ok.is_error and ok.content == "tool-output"
    finally:
        await client.close()


# 功能：验证子 Agent 可调用父级配置的 MCP 工具，并遵守同一工具结果截断配置
# 设计：真实 Runner 派生前台子循环并连接独立 stdio 服务，捕获子模型第二步收到的结果
async def test_child_inherits_mcp_tools_and_budget(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    client = McpClient()
    runner = None
    try:
        await client.connect_stdio(sys.executable, ["-u", "-c", _MCP_STDIO])
        definitions = await client.list_tools()
        tool = McpTool(client, "configured", definitions[0])
        captured: list = []

        class Provider:
            # 父循环派生子任务，子循环调用 MCP，再捕获截断后的远端返回
            async def chat(self, messages: list, step: int,
                           tool_schemas: list, **kwargs: object) -> LlmResponse:
                if messages[0]["content"] == "child-mcp":
                    assert any(schema["name"] == tool.name for schema in tool_schemas)
                    if step == 1:
                        return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                            "mcp", tool.name, {"path": "ok"},
                        )])
                    captured.append(messages[-1]["content"][0]["content"])
                    return LlmResponse("end_turn", text="child-done")
                if step == 1:
                    return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                        "spawn", "spawn_agent", {"description": "mcp", "prompt": "child-mcp"},
                    )])
                assert messages[-1]["content"][0]["content"].startswith("ch\n[...")
                return LlmResponse("end_turn", text="done")

        manager = MagicMock()
        manager.get_tools.return_value = [tool]
        cfg = XConfig()
        cfg.compaction.auto_threshold = 0
        cfg.compaction.tool_result_limit = 4
        cfg.compaction.tool_result_keep = 2
        runner = AgentRunner(cfg, provider=Provider(), mcp_manager=manager, runs_dir=tmp_path)
        result = await runner.run_and_capture("parent-goal", run_id="parent")
        assert result.status == "success"
        assert captured and captured[0].startswith("to\n[...")
    finally:
        if runner is not None:
            await runner.shutdown()
        await client.close()


# 功能：验证 MCP TCP 工具失败标记经 JSON-RPC 返回后仍保留，而不是被当作成功文本
# 设计：真实 asyncio TCP 服务返回标准 isError，调用完整 McpClient/McpTool/统一工具调用链
async def test_mcp_tcp_error_propagation() -> None:
    # 实现最小 initialize 和 tools/call 服务，并正常等待客户端关闭
    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while line := await reader.readline():
                req = json.loads(line)
                if "id" not in req:
                    continue
                result = {} if req["method"] == "initialize" else {
                    "isError": True, "content": [{"type": "text", "text": "remote failure"}],
                }
                writer.write((json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}) + "\n").encode())
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    client = McpClient()
    try:
        await client.connect_tcp("127.0.0.1", server.sockets[0].getsockname()[1])
        tool = McpTool(client, "tcp", McpToolDef("sample", "test", {"type": "object"}))
        registry = ToolRegistry()
        registry.register(tool)
        result = await invoke_tool(registry, ToolCallBlock("t", tool.name, {}), EventBus(), "r")
        assert result.is_error and "remote failure" in result.content
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


# 功能：验证清空会话会取消并回收后台子 Agent，不会把旧任务注册表带到新会话
# 设计：真实后台循环等待屏障，调用 clear_context 后检查 asyncio 任务和子上下文的取消状态
async def test_clear_cancels_background_agent(tmp_path: Path) -> None:
    entered = asyncio.Event()

    class Provider:
        # 父任务返回，子任务保持运行直到 clear 显式取消
        async def chat(self, messages: list, step: int, **kwargs: object) -> LlmResponse:
            if messages[0]["content"] == "wait-child":
                entered.set()
                await asyncio.Event().wait()
            if step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "spawn", "spawn_agent", {"description": "wait", "prompt": "wait-child",
                                             "run_in_background": True},
                )])
            return LlmResponse("end_turn", text="parent-done")

    manager = SessionManager(SessionStore(tmp_path), lambda: AgentRunner(
        XConfig(), provider=Provider()), EventBus())
    session = await manager.create("chat")
    try:
        await manager.send_message(session.id, "launch")
        await asyncio.wait_for(entered.wait(), 2)
        registry = manager._runners[session.id]._task_registry
        task, context = registry.all()[0]
        fresh = await manager.clear_context(session.id)
        assert fresh.id != session.id
        assert task.cancelled()
        assert context.reason == "cancelled"
        assert not registry.all()
        assert session.id not in manager._runners
    finally:
        await manager.shutdown()


# 功能：验证 Bash 取消和超时会清理本工具进程树，子进程不能在稍后继续写文件
# 设计：真实 Shell 先创建 ready 标记再延迟写 escaped，等待超过写入时间后确认取消没有留下后台执行
@pytest.mark.parametrize("cancel", [True, False])
async def test_bash_cleanup_prevents_late_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    command = "printf ready > ready; sleep 2; printf leaked > escaped; sleep 2"
    task = asyncio.create_task(BashTool().invoke({"command": command, "timeout": 1}))
    if cancel:
        async with asyncio.timeout(3):
            while not (tmp_path / "ready").exists():
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        assert result.error_type == "timeout"
    await asyncio.sleep(2.1)
    assert not (tmp_path / "escaped").exists()


# 功能：验证慢订阅客户端不能无限阻塞其他客户端和 Agent 的事件发布
# 设计：用永不完成的 drain 触发限时移除，再确认正常客户端仍收到完整事件
async def test_slow_subscriber_is_disconnected_without_blocking_others() -> None:
    from unittest.mock import AsyncMock, MagicMock

    slow = MagicMock()
    fast = MagicMock()

    # 模拟连接仍存活但客户端停止消费数据的反压状态
    async def blocked() -> None:
        await asyncio.Event().wait()

    slow.drain = AsyncMock(side_effect=blocked)
    fast.drain = AsyncMock()
    broadcaster = IpcEventBroadcaster(write_timeout_s=0.01)
    broadcaster.subscribe(slow, ["*"])
    broadcaster.subscribe(fast, ["*"])
    await asyncio.wait_for(broadcaster.handle(
        RunStartedEvent(run_id="r", goal="g", ts="t")), 1)
    slow.close.assert_called_once()
    fast.write.assert_called_once()
    assert len(broadcaster._subscriptions) == 1
