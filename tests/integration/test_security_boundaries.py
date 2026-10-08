from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

import x_claude.core.app as app_module
from tests.integration.test_background_restart import _manager
from tests.integration.test_root_restart import _wait_root
from tests.integration.test_runtime_hardening import _serve
from x_claude.core.bus.commands import SessionRecoverCommand
from x_claude.core.config import XConfig
from x_claude.core.llm.types import LlmResponse, ToolCallBlock
from x_claude.core.permissions.manager import PermissionManager
from x_claude.core.permissions.policy import PermissionDecision, ToolPolicy
from x_claude.core.session.store import SessionStore
from x_claude.core.subagent.checkpoint import BackgroundCheckpoint
from x_claude.core.tools.builtin.list_dir import ListDirTool
from x_claude.core.tools.builtin.read_file import ReadFileTool
from x_claude.core.tools.builtin.write_file import WriteFileTool
from x_claude.core.tools.file_access import file_access_scope
from x_claude.core.transport.auth import credential_path, read_credential
from x_claude.core.transport.socket_client import SocketClient
from x_claude.core.transport.socket_server import SocketServer
from x_claude.tui.app import ChatTextArea, PermissionSelect, XTuiApp


# 在临时项目内运行，项目外模拟敏感文件和 IPC 凭据保持独立
@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    monkeypatch.chdir(workspace)


# 通过原始 TCP 帧检查服务器认证，不让客户端自动附加凭据掩盖错误
async def _raw(reader, writer, method: str, params: dict, token: str | None = None) -> dict:
    writer.write((json.dumps({"jsonrpc": "2.0", "id": "audit", "method": method,
                             "params": params, "auth_token": token}) + "\n").encode())
    await writer.drain()
    return json.loads(await asyncio.wait_for(reader.readline(), 3))


# 功能：验证三种文件工具拒绝绝对越界和父级遍历，项目内绝对路径仍然可用
# 设计：外部文件是无敏感内容的临时文件，比较写前写后内容以排除拒绝前已执行副作用
@pytest.mark.parametrize("name", ["read", "write", "list"])
async def test_file_tools_reject_external_paths(tmp_path: Path, name: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("dummy-secret", encoding="utf-8")
    tool = {"read": ReadFileTool(), "write": WriteFileTool(), "list": ListDirTool()}[name]
    for value in (str(outside if name == "list" else secret), "../outside/secret.txt"):
        with pytest.raises(PermissionError):
            await tool.invoke({"path": value, "content": "overwritten"})
    assert secret.read_text() == "dummy-secret"
    inside = Path.cwd() / "inside.txt"
    inside.write_text("inside", encoding="utf-8")
    with file_access_scope({}):
        if name == "write":
            await ReadFileTool().invoke({"path": str(inside)})
        result = await tool.invoke({"path": str(Path.cwd() if name == "list" else inside),
                                    "content": "inside"})
    assert not result.is_error


# 功能：验证文件工具和递归目录列表不能通过符号链接或 Windows 联接逃出项目
# 设计：Windows 用无需开发者模式的目录联接，其余平台用目录符号链接，均测试真实文件系统
async def test_link_escape_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("dummy", encoding="utf-8")
    link = Path.cwd() / "link"
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                       check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        link.symlink_to(outside, target_is_directory=True)
    for tool, value in [(ReadFileTool(), link / "secret.txt"),
                        (WriteFileTool(), link / "secret.txt"), (ListDirTool(), link)]:
        with pytest.raises(PermissionError):
            await tool.invoke({"path": str(value), "content": "bad"})
    tree = await ListDirTool().invoke({"path": "."})
    assert "outside workspace; skipped" in tree.content
    assert "secret.txt" not in tree.content and secret.read_text() == "dummy"


# 功能：验证缺失、错误或非 ASCII 凭据都不能调用 IPC，有效凭据允许命令
# 设计：原始 TCP 请求覆盖真实路由入口，并检查生成文件权限和 stop 后凭据清理
async def test_ipc_authentication_and_private_credential() -> None:
    class Provider:
        # 该测试只创建会话，不应请求模型
        async def chat(self, **kwargs):
            raise AssertionError("unexpected model call")

    async with _serve(Path.cwd(), Provider()) as (cfg, _, _, _):
        path = credential_path(cfg.host, cfg.port)
        token = read_credential(cfg.host, cfg.port)
        assert token is not None and len(token) >= 32
        reader, writer = await asyncio.open_connection(cfg.host, cfg.port)
        try:
            for bad in (None, "incorrect", "中文"):
                result = await _raw(reader, writer, "session.create", {}, bad)
                assert result["error"]["code"] == -32001
            accepted = await _raw(reader, writer, "session.create", {}, token)
            assert "result" in accepted
        finally:
            writer.close()
            await writer.wait_closed()
        if os.name != "nt":
            assert path.stat().st_mode & 0o077 == 0
        else:
            acl = subprocess.run(["icacls", str(path)], capture_output=True, check=True)
            assert b"(I)" not in acl.stdout
    assert not path.exists()


# 功能：验证认证令牌只用于验证，不进入 IPC Trace 和错误响应
# 设计：捕获 trace 实际写入参数，覆盖有效命令及非法请求字段类型，避免只检查日志格式
async def test_auth_token_is_not_recorded_in_trace() -> None:
    trace, writer = Mock(), Mock()
    writer.drain = AsyncMock()
    server = SocketServer("127.0.0.1", 0, trace=trace)
    token = "test-secret-token"
    server._auth_token = token
    server.register("core.ping", AsyncMock(return_value={"ok": True}))
    for identifier in ("valid", 1):
        await server._handle_line(json.dumps({"jsonrpc": "2.0", "id": identifier,
                                             "method": "core.ping", "auth_token": token}).encode(),
                                  writer)
    assert trace.emit.called
    assert token not in str(trace.emit.call_args_list)
    assert token not in str(writer.write.call_args_list)


# 功能：验证 daemon 重启更新凭据，客户端重连自动刷新，旧凭据失效
# 设计：同一端口重建真实 SocketServer，复用客户端对象，排除仅在首次构造时读取令牌
async def test_credential_rotates_and_client_refreshes_on_restart() -> None:
    server = SocketServer("127.0.0.1", 0)
    server.register("core.ping", AsyncMock(return_value={"ok": 1}))
    await server.start()
    port = server._server.sockets[0].getsockname()[1]
    old_token = read_credential("127.0.0.1", port)
    client = SocketClient("127.0.0.1", port)
    await client.connect()
    events = asyncio.create_task(client.run_event_loop())
    try:
        assert await client.send_command("core.ping", {}) == {"ok": 1}
    finally:
        await client.close()
        await asyncio.gather(events, return_exceptions=True)
        await server.stop()
    restarted = SocketServer("127.0.0.1", port)
    restarted.register("core.ping", AsyncMock(return_value={"ok": 2}))
    await restarted.start()
    try:
        assert read_credential("127.0.0.1", port) != old_token
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            rejected = await _raw(reader, writer, "core.ping", {}, old_token)
            assert rejected["error"]["code"] == -32001
        finally:
            writer.close()
            await writer.wait_closed()
        await client.connect()
        events = asyncio.create_task(client.run_event_loop())
        assert await asyncio.wait_for(client.send_command("core.ping", {}), 3) == {"ok": 2}
    finally:
        await client.close()
        await asyncio.gather(events, return_exceptions=True)
        await restarted.stop()


# 功能：验证文件截断在载入阶段即受限，不先读取整个大文件
# 设计：捕获真实调用的 read 大小参数，防止仅返回内容截断却仍耗尽内存的假通过
async def test_read_file_limits_bytes_loaded() -> None:
    target = Path.cwd() / "big.txt"
    target.write_bytes(b"x" * (600 * 1024))
    handle = Mock()
    with target.open("rb") as actual:
        handle.fileno.return_value = actual.fileno()
        # 保持句柄打开以验证实际 fstat，不绕过版本检测
        handle.read.return_value = b"x" * (512 * 1024 + 1)
        context = Mock()
        context.__enter__ = Mock(return_value=handle)
        context.__exit__ = Mock(return_value=False)
        with patch.object(Path, "open", return_value=context):
            result = await ReadFileTool().invoke({"path": str(target)})
    handle.read.assert_called_once_with(512 * 1024 + 1)
    assert result.content.endswith("[truncated]")


# 功能：验证多个运行复用同一工具 ID 时审批隔离，错误会话及无运行 ID 不得误放行
# 设计：并发挂起同工具名请求，逐项批准后断言另一 Future 仍未完成
async def test_permission_id_collision_is_scoped_to_run() -> None:
    manager = PermissionManager(timeout_s=2)
    entered = asyncio.Queue()

    # 记录两个真实挂起请求都已登记
    async def emit(event):
        entered.put_nowait(event)

    tasks = [asyncio.create_task(manager.check_and_wait(
        "same", "bash", {"command": "echo ok"}, sid, emit, run_id=run_id,
    )) for sid, run_id in [("s1", "r1"), ("s2", "r2")]]
    try:
        await asyncio.wait_for(entered.get(), 3)
        await asyncio.wait_for(entered.get(), 3)
        assert not manager.respond("same", "allow_once")
        assert not manager.respond("same", "allow_once", session_id="s2", run_id="r1")
        assert manager.respond("same", "allow_once", session_id="s1", run_id="r1")
        assert await tasks[0] == (True, "allow_once") and not tasks[1].done()
        assert manager.respond("same", "deny_once", session_id="s2", run_id="r2")
        assert await tasks[1] == (False, "deny_once")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# 功能：验证 IPC 拒绝监听公网或局域网地址，避免明文令牌被当作远程服务安全保证
# 设计：启动前拒绝无需真正占用端口，对两个非回环地址参数化测试
@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.2"])
async def test_ipc_rejects_non_loopback_host(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        await SocketServer(host, 0).start()


# 功能：验证全局观察者、错误会话、错误 run_id 不能审批，正确客户端可批准且重复批准无效
# 设计：真实 TCP 与真实写文件，第三方客户端持有测试凭据但不拥有正确审批订阅
async def test_approval_is_bound_to_session_run_and_connection() -> None:
    target = Path.cwd() / "approved.txt"
    pending: asyncio.Queue[dict] = asyncio.Queue()

    class Provider:
        # 先请求敏感工具，只有审批通过才写文件
        async def chat(self, step: int, **kwargs):
            if step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "write", "write_file", {"path": str(target), "content": "approved"},
                )])
            return LlmResponse("end_turn", text="done")

    async with _serve(Path.cwd(), Provider()) as (cfg, manager, _, _):
        owner, observer = SocketClient(cfg.host, cfg.port), SocketClient(cfg.host, cfg.port)
        await owner.connect()
        await observer.connect()

        # 事件处理不等待 RPC，避免阻塞客户端响应分发循环
        async def event(data):
            if data["type"] == "permission.requested":
                pending.put_nowait(data)

        owner.on_event(event)
        loops = [asyncio.create_task(c.run_event_loop()) for c in (owner, observer)]
        try:
            sid = (await owner.send_command("session.create", {}))["session_id"]
            await owner.send_command("event.subscribe", {"topics": ["*"], "scope": f"session:{sid}"})
            await observer.send_command("event.subscribe", {"topics": ["*"], "scope": "global"})
            run = asyncio.create_task(owner.send_command(
                "session.send_message", {"session_id": sid, "content": "write"},
            ))
            requested = await asyncio.wait_for(pending.get(), 3)
            params = {"session_id": sid, "run_id": requested["run_id"],
                      "tool_use_id": "write", "decision": "allow_once"}
            for client, updates in [(observer, {}), (owner, {"session_id": "other"}),
                                    (owner, {"run_id": "other"})]:
                result = await client.send_command("permission.respond", {**params, **updates})
                assert result == {"ok": False} and not target.exists()
            assert (await owner.send_command("permission.respond", params))["ok"]
            await asyncio.wait_for(run, 3)
            assert target.read_text() == "approved"
            assert not (await owner.send_command("permission.respond", params))["ok"]
        finally:
            await manager.shutdown()
            for client in (owner, observer):
                await client.close()
            await asyncio.gather(*loops, return_exceptions=True)


# 功能：验证真实 CoreApp 退出先挂起主任务，重建后从原步骤继续且不重放已确认写入
# 设计：不采用测试服务的关闭顺序，运行真正 CoreApp.run 并触发同一 shutdown Event
@pytest.mark.parametrize("phase", ["planning", "tools"])
async def test_real_daemon_shutdown_preserves_root(phase: str) -> None:
    entered = asyncio.Event()
    target = Path.cwd() / "once.txt"
    original = WriteFileTool.invoke

    # 工具阶段执行副作用后阻塞，规划阶段则在已确认结果之后阻塞
    async def interrupted(tool, params):
        result = await original(tool, params)
        if phase == "tools":
            entered.set()
            await asyncio.Event().wait()
        return result

    class First:
        # 第一轮写入，第二轮在推理中等待真实 daemon 退出
        async def chat(self, step: int, **kwargs):
            if step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "write", "write_file", {"path": str(target), "content": "once"},
                )])
            entered.set()
            await asyncio.Event().wait()

    config = XConfig()
    config.port = 0
    config.trace.enabled = False
    config.compaction.auto_threshold = 0
    store = SessionStore(Path.cwd() / "sessions")
    servers, shutdowns = [], []

    # 捕获真实服务器端口，不替换任何关闭行为
    def factory(*args, **kwargs):
        server = SocketServer(*args, **kwargs)
        servers.append(server)
        return server

    permission = PermissionManager({"write_file": ToolPolicy(PermissionDecision.ALLOW)})
    with patch.object(app_module, "get_config", return_value=config), \
            patch.object(app_module, "setup_logging"), \
            patch.object(app_module, "AnthropicProvider", return_value=First()), \
            patch("x_claude.core.runner.AnthropicProvider", return_value=First()), \
            patch.object(app_module, "PermissionManager", return_value=permission), \
            patch.object(app_module, "SessionStore", return_value=store), \
            patch.object(app_module, "SocketServer", side_effect=factory), \
            patch.object(app_module, "_install_shutdown_handlers",
                         side_effect=lambda loop, event: shutdowns.append(event)), \
            patch.object(WriteFileTool, "invoke", interrupted):
        app = app_module.CoreApp()
        daemon = asyncio.create_task(app.run())
        for _ in range(300):
            if shutdowns:
                break
            if daemon.done():
                await daemon
            await asyncio.sleep(.01)
        assert shutdowns
        port = servers[0]._server.sockets[0].getsockname()[1]
        client = SocketClient("127.0.0.1", port)
        await client.connect()
        events = asyncio.create_task(client.run_event_loop())
        try:
            sid = (await client.send_command("session.create", {}))["session_id"]
            request = asyncio.create_task(client.send_command(
                "session.send_message", {"session_id": sid, "content": "write once"},
            ))
            await asyncio.wait_for(entered.wait(), 3)
            path = next(store.runs_dir(sid).glob("*/root.json"))
            before = target.stat().st_mtime_ns
            shutdowns[0].set()
            await asyncio.wait_for(daemon, 5)
            await asyncio.gather(request, return_exceptions=True)
            record = BackgroundCheckpoint.load(path, sid).record
            assert record.state == ("suspended" if phase == "planning" else "blocked")
            assert record.context.status == "running" and record.context.step == 1
        finally:
            if not daemon.done():
                shutdowns[0].set()
                await asyncio.wait_for(daemon, 5)
            await client.close()
            await asyncio.gather(events, return_exceptions=True)

    class Next:
        # 恢复只请求下一步，不能重新写入文件
        async def chat(self, step: int, **kwargs):
            assert step == 2
            return LlmResponse("end_turn", text="resumed")

    manager = _manager(store, Next())
    try:
        await manager.resume(sid)
        if phase == "tools":
            await manager.recover(SessionRecoverCommand(
                session_id=sid, run_id=record.run_id, tool_results={"write": "verified once.txt"},
            ))
        else:
            await manager.recover_background(sid)
        await _wait_root(manager, sid)
        assert BackgroundCheckpoint.load(path, sid).record.thread_committed
        assert target.stat().st_mtime_ns == before
    finally:
        await manager.shutdown()


# 功能：验证 TUI 断线重连后仍能批准原挂起工具，审批归属检查不破坏正常交互
# 设计：真实 TCP、Textual 按键与真实写文件，重连只补发原 Future，不新建任务或延长超时
async def test_tui_approval_survives_connection_restart() -> None:
    target = Path.cwd() / "approved.txt"

    class Provider:
        # 第一轮申请工具，第二轮完成；不能因重连产生新的第一步
        async def chat(self, step: int, **kwargs):
            if step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "write", "write_file", {"path": str(target), "content": "approved"},
                )])
            return LlmResponse("end_turn", text="done")

    async with _serve(Path.cwd(), Provider(), permission_timeout=10) as (cfg, manager, _, _):
        app = XTuiApp(cfg.host, cfg.port)
        async with app.run_test() as pilot:
            for _ in range(150):
                if app._session_id:
                    break
                await asyncio.sleep(.01)
            sid = app._session_id
            assert sid is not None
            prompt = app.query_one("#prompt", ChatTextArea)
            prompt.text = "write"
            await app.on_chat_text_area_submitted(ChatTextArea.Submitted(prompt))
            for _ in range(150):
                if app.query(PermissionSelect):
                    break
                await asyncio.sleep(.01)
            assert len(app.query(PermissionSelect)) == 1
            original_client = app._client
            assert original_client is not None
            await original_client.close()
            for _ in range(600):
                if (app._client is not None and app._client is not original_client
                        and app._session_id == sid and app.query(PermissionSelect)):
                    break
                await asyncio.sleep(.01)
            assert app._client is not original_client and app._session_id == sid
            await pilot.pause()
            assert len(app.query(PermissionSelect)) == 1
            await pilot.press("y")
            for _ in range(150):
                if target.exists() and not app._busy:
                    break
                await asyncio.sleep(.01)
            await pilot.pause()
            assert target.read_text() == "approved"
            assert len(manager._sessions[sid].run_ids) == 1
            assert not app._busy and not app._pending_permission_blocks
            assert not prompt.disabled


# 功能：验证两个子 Agent 使用相同工具 ID 时，TUI 选中一个审批不会批准另一个工具
# 设计：真实并发子循环分别写 a/b，按界面选择允许 a、拒绝 b，核对独立文件副作用
async def test_tui_keeps_colliding_child_approvals_separate() -> None:
    targets = {name: Path.cwd() / f"{name}.txt" for name in ("a", "b")}

    class Provider:
        # 父任务派生两个独立子任务，各自复用同一个工具 ID，检验运行身份不能被省略
        async def chat(self, messages: list, step: int, **kwargs):
            goal = messages[0]["content"]
            if step == 1 and goal in targets:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "same", "write_file", {"path": str(targets[goal]), "content": goal},
                )])
            if step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    name, "spawn_agent", {"description": name, "prompt": name,
                                           "run_in_background": True},
                ) for name in targets])
            return LlmResponse("end_turn", text="done")

    permissions = PermissionManager({
        "spawn_agent": ToolPolicy(PermissionDecision.ALLOW),
        "write_file": ToolPolicy(PermissionDecision.ASK),
    }, timeout_s=10)
    with patch("tests.integration.test_runtime_hardening.PermissionManager",
               return_value=permissions):
        async with _serve(Path.cwd(), Provider()) as (cfg, manager, _, _):
            app = XTuiApp(cfg.host, cfg.port)
            async with app.run_test() as pilot:
                for _ in range(150):
                    if app._session_id:
                        break
                    await asyncio.sleep(.01)
                sid = app._session_id
                assert sid is not None
                prompt = app.query_one("#prompt", ChatTextArea)
                prompt.text = "launch"
                await app.on_chat_text_area_submitted(ChatTextArea.Submitted(prompt))
                for _ in range(150):
                    if len(app.query(PermissionSelect)) == 2:
                        break
                    await asyncio.sleep(.01)
                await pilot.pause()
                assert len(app.query(PermissionSelect)) == 2
                registry = manager._runners[sid]._task_registry
                ids = {context.goal: run_id for run_id, (_, context) in registry._tasks.items()}
                selects = {select._run_id: select for select in app.query(PermissionSelect)}
                selects[ids["a"]].focus()
                await pilot.press("y")
                for _ in range(150):
                    if targets["a"].exists():
                        break
                    await asyncio.sleep(.01)
                assert targets["a"].read_text() == "a" and not targets["b"].exists()
                assert (ids["b"], "same") in app._pending_permission_blocks
                selects[ids["b"]].focus()
                await pilot.press("n")
                await asyncio.wait_for(asyncio.gather(*(task for task, _ in registry.all())), 3)
                await pilot.pause()
                assert not targets["b"].exists()
                assert not app._pending_permission_blocks
