from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Static

from tests.integration.test_runtime_hardening import _serve
from x_claude.core.config import XConfig
from x_claude.core.context import ExecutionContext
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.types import LlmResponse, ToolCallBlock
from x_claude.core.runner import AgentRunner
from x_claude.core.session.manager import SessionManager
from x_claude.core.session.store import SessionStore
from x_claude.core.subagent.checkpoint import (
    BackgroundCheckpoint,
    BackgroundRecord,
    ContextSnapshot,
    runtime_signature,
)
from x_claude.core.subagent.registry import BackgroundTaskRegistry
from x_claude.core.subagent.tool import AgentResultTool, SpawnAgentTool
from x_claude.tui.app import XTuiApp


# 将临时目录作为测试项目，文件工具不能再绕过真实工作目录边界
@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


# 构造与真实后台工具完全一致的运行指纹，避免测试绕过配置变化校验
def _signature(tools: list[str], depth: int = 0) -> str:
    config = XConfig()
    tool = SpawnAgentTool(None, EventBus(), "parent", None, 5, BackgroundTaskRegistry(),
                          Path("."), "sid", depth, config=config)
    registry = tool._build_child_registry(EventBus(), "child", None, allowed_tools=tools)
    return runtime_signature(config, registry.tool_schemas())


# 为指定会话构造真实、可恢复的检查点，不依赖历史日志猜测任务状态
def _checkpoint(store: SessionStore, sid: str, phase: str = "planning") -> BackgroundCheckpoint:
    context = ExecutionContext("child", "child-goal", 5, step=1)
    context.add_assistant_message(
        [{"type": "tool_use", "id": "pending", "name": "list_dir", "input": {"path": "."}}]
        if phase == "tools" else [{"type": "text", "text": "step one done"}]
    )
    record = BackgroundRecord(
        run_id="child", session_id=sid, parent_run_id="parent", description="测试恢复",
        cwd=str(Path.cwd().resolve()), depth=0, tools=["list_dir"],
        runtime_signature=_signature(["list_dir"]), model=XConfig().llm.default_model,
        context=ContextSnapshot.capture(context),
    )
    checkpoint = BackgroundCheckpoint(store.runs_dir(sid) / "child" / "background.json", record)
    checkpoint.save(context, phase)
    return checkpoint


# 构建使用真实存储和后台任务注册表的隔离会话管理器
def _manager(store: SessionStore, provider: object) -> SessionManager:
    cfg = XConfig()
    cfg.compaction.auto_threshold = 0
    bus = EventBus()
    return SessionManager(store, lambda: AgentRunner(cfg, provider=provider, bus=bus), bus)


# 功能：验证正常 daemon 关闭会保存可续跑状态，重建会话后只执行下一步并恢复完成结果
# 设计：真实后台工具写文件后在第二次推理等待，销毁第一管理器，再用全新管理器检查步骤和结果
async def test_graceful_restart_preserves_completed_step_and_result(tmp_path: Path) -> None:
    entered = asyncio.Event()
    target = tmp_path / "once.txt"

    class FirstProvider:
        # 子任务先写入真实文件，在下一步推理中等待 daemon 关闭
        async def chat(self, messages: list, step: int, **kwargs: object) -> LlmResponse:
            if messages[0]["content"] == "child-goal":
                if step == 1:
                    return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                        "write", "write_file", {"path": str(target), "content": "first step"},
                    )])
                entered.set()
                await asyncio.Event().wait()
            if step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "spawn", "spawn_agent", {"description": "persistent", "prompt": "child-goal",
                                             "run_in_background": True},
                )])
            return LlmResponse("end_turn", text="parent-done")

    steps: list[int] = []

    class NextProvider:
        # 拒绝从第一步重新开始，验证历史已有写文件结果后直接完成
        async def chat(self, messages: list, step: int, **kwargs: object) -> LlmResponse:
            steps.append(step)
            assert step == 2
            assert messages[-1]["content"][0]["tool_use_id"] == "write"
            return LlmResponse("end_turn", text="恢复完成")

    store = SessionStore(tmp_path / "sessions")
    first = _manager(store, FirstProvider())
    session = await first.create("chat")
    await first.send_message(session.id, "launch")
    await asyncio.wait_for(entered.wait(), 3)
    child_id = next(iter(first._runners[session.id]._task_registry._tasks))
    path = store.runs_dir(session.id) / child_id / "background.json"
    await first.shutdown()
    checkpoint = BackgroundCheckpoint.load(path, session.id)
    assert checkpoint.record.state == "suspended"
    assert checkpoint.record.phase == "planning"
    assert checkpoint.record.context.step == 1
    before = target.stat().st_mtime_ns

    second = _manager(SessionStore(store._root), NextProvider())
    try:
        await second.resume(session.id)
        await asyncio.gather(second.recover_background(session.id), second.recover_background(session.id))
        registry = second._runners[session.id]._task_registry
        await asyncio.gather(*(task for task, _ in registry.all()))
        result = await AgentResultTool(registry).invoke({"run_id": child_id})
        assert result.content == "恢复完成" and not result.is_error
        assert steps == [2]
        assert target.stat().st_mtime_ns == before
    finally:
        await second.shutdown()

    third = _manager(SessionStore(store._root), None)
    try:
        await third.resume(session.id)
        with patch("x_claude.core.runner.AnthropicProvider", side_effect=AssertionError("no model")):
            await third.recover_background(session.id)
        result = await AgentResultTool(third._runners[session.id]._task_registry).invoke({"run_id": child_id})
        assert result.content == "恢复完成" and not result.is_error
        other = await third.create("chat")
        await third.recover_background(other.id)
        result = await AgentResultTool(third._runners[other.id]._task_registry).invoke({"run_id": child_id})
        assert result.is_error and "Unknown run_id" in result.content
    finally:
        await third.shutdown()


# 功能：验证工具执行阶段、错误工作目录和损坏检查点均不自动重放任务
# 设计：给出三类不安全检查点，并令模型调用立即失败，查询必须返回明确问题而非 Unknown
@pytest.mark.parametrize("problem", ["tools", "workspace", "corrupt"])
async def test_unsafe_checkpoint_requires_review(tmp_path: Path, problem: str) -> None:
    store = SessionStore(tmp_path)
    first = _manager(store, None)
    session = await first.create("chat")
    checkpoint = _checkpoint(store, session.id, "tools" if problem == "tools" else "planning")
    if problem == "workspace":
        checkpoint.record.cwd = str(tmp_path / "different-workspace")
        checkpoint.save(checkpoint.record.context.restore("child"), "planning")
    if problem == "corrupt":
        checkpoint.path.write_bytes(b"{broken")
    second = _manager(store, None)
    try:
        await second.resume(session.id)
        with patch("x_claude.core.runner.AnthropicProvider", side_effect=AssertionError("no replay")):
            await second.recover_background(session.id)
        result = await AgentResultTool(second._runners[session.id]._task_registry).invoke({"run_id": "child"})
        assert result.is_error
        expected = {"tools": "interrupted_tool", "workspace": "workspace_mismatch", "corrupt": "checkpoint_invalid"}
        assert expected[problem] in result.content
    finally:
        await second.shutdown()


# 功能：验证显式 /clear 和关闭会话即使没有加载 Runner，也会永久取消磁盘中的待恢复任务
# 设计：先仅持久化后台任务，再调用用户关闭操作并重新恢复旧会话，模型不得被重新调用
@pytest.mark.parametrize("clear", [True, False])
async def test_close_cancels_saved_unactivated_background(tmp_path: Path, clear: bool) -> None:
    store = SessionStore(tmp_path)
    manager = _manager(store, None)
    session = await manager.create("chat")
    checkpoint = _checkpoint(store, session.id)
    if clear:
        await manager.clear_context(session.id)
    else:
        await manager.close(session.id)
    assert BackgroundCheckpoint.load(checkpoint.path, session.id).record.state == "cancelled"
    try:
        await manager.resume(session.id)
        with patch("x_claude.core.runner.AnthropicProvider", side_effect=AssertionError("cancelled")):
            await manager.recover_background(session.id)
        result = await AgentResultTool(manager._runners[session.id]._task_registry).invoke({"run_id": "child"})
        assert result.is_error and "cancelled" in result.content
    finally:
        await manager.shutdown()


# 功能：验证恢复时保留原角色提示和工具白名单，不因当前默认工具增加而扩大权限
# 设计：仅允许 list_dir 的检查点在默认 Runner 上恢复，检查实际模型入参而非仅检查配置字段
async def test_recovery_preserves_prompt_and_tool_whitelist(tmp_path: Path) -> None:
    class Provider:
        # 检查恢复后的系统提示、可用工具和步数均来自原任务
        async def chat(self, system: str, tool_schemas: list, step: int,
                       **kwargs: object) -> LlmResponse:
            assert system == "original-role"
            assert [schema["name"] for schema in tool_schemas] == ["list_dir"]
            assert step == 2
            return LlmResponse("end_turn", text="restricted done")

    store = SessionStore(tmp_path)
    manager = _manager(store, Provider())
    session = await manager.create("chat")
    checkpoint = _checkpoint(store, session.id)
    context = checkpoint.record.context.restore("child")
    context.system_prompt_override = "original-role"
    checkpoint.save(context, "planning")
    try:
        await manager.recover_background(session.id)
        registry = manager._runners[session.id]._task_registry
        await asyncio.gather(*(task for task, _ in registry.all()))
        result = await AgentResultTool(registry).invoke({"run_id": "child"})
        assert result.content == "restricted done" and not result.is_error
    finally:
        await manager.shutdown()


# 功能：验证嵌套后台任务在任何恢复循环开始前就完成索引登记，不出现虚假的 Unknown run_id
# 设计：让排序在前的父任务立即查询排序在后的子任务，使用零延迟模型最大化暴露加载顺序问题
async def test_recovery_registers_all_children_before_execution(tmp_path: Path) -> None:
    class Provider:
        # 父循环立即查询子任务；子循环立即返回，检查父模型观察到的查询结果
        async def chat(self, messages: list, step: int, **kwargs: object) -> LlmResponse:
            if messages[0]["content"] == "parent-goal":
                if step == 2:
                    return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                        "query", "agent_result", {"run_id": "z-child"},
                    )])
                assert "Unknown run_id" not in messages[-1]["content"][0]["content"]
                return LlmResponse("end_turn", text="parent recovered")
            return LlmResponse("end_turn", text="nested result")

    store = SessionStore(tmp_path)
    manager = _manager(store, Provider())
    session = await manager.create("chat")
    for run_id, goal, parent, tools, depth in [
        ("a-parent", "parent-goal", "root", ["agent_result"], 0),
        ("z-child", "child-goal", "a-parent", [], 1),
    ]:
        context = ExecutionContext(run_id, goal, 5, step=1)
        checkpoint = BackgroundCheckpoint(
            store.runs_dir(session.id) / run_id / "background.json",
            BackgroundRecord(
                run_id=run_id, session_id=session.id, parent_run_id=parent, description=goal,
                cwd=str(Path.cwd().resolve()), depth=depth, tools=tools,
                runtime_signature=_signature(tools, depth), model=XConfig().llm.default_model,
                context=ContextSnapshot.capture(context),
            ),
        )
        checkpoint.save(context, "planning")
    try:
        await manager.recover_background(session.id)
        registry = manager._runners[session.id]._task_registry
        await asyncio.gather(*(task for task, _ in registry.all()))
        result = await AgentResultTool(registry).invoke({"run_id": "a-parent"})
        assert result.content == "parent recovered" and not result.is_error
    finally:
        await manager.shutdown()


# 功能：验证取消记录写入失败不会谎报关闭成功，任务仍可由用户再次取消
# 设计：只在取消持久化时注入磁盘错误，确认 meta 未被标记 closed 且原检查点仍完整
async def test_close_reports_cancel_checkpoint_failure(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = _manager(store, None)
    session = await manager.create("chat")
    checkpoint = _checkpoint(store, session.id)
    before = checkpoint.path.read_bytes()
    with patch.object(BackgroundCheckpoint, "interrupt", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            await manager.close(session.id)
    assert store.read_meta(session.id).status == "active"
    assert checkpoint.path.read_bytes() == before
    await manager.close(session.id)
    assert BackgroundCheckpoint.load(checkpoint.path, session.id).record.state == "cancelled"


# 功能：验证恢复时缺少模型认证配置不会退出 daemon，也不丢失任务检查点
# 设计：模拟 Provider 初始化的 SystemExit，查询应返回具体恢复错误且文件保持可再次恢复
async def test_recovery_reports_unavailable_provider(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = _manager(store, None)
    session = await manager.create("chat")
    checkpoint = _checkpoint(store, session.id)
    before = checkpoint.path.read_bytes()
    try:
        with patch("x_claude.core.runner.AnthropicProvider", side_effect=SystemExit("API_KEY not set")):
            await manager.recover_background(session.id)
        result = await AgentResultTool(manager._runners[session.id]._task_registry).invoke({"run_id": "child"})
        assert result.is_error and "recovery_provider_error" in result.content
        assert checkpoint.path.read_bytes() == before
    finally:
        await manager.shutdown()


# 功能：验证检查点保存失败时不执行后台任务的工具，也不假报任务成功
# 设计：注入工具阶段的原子写入故障，用真实写文件请求确认副作用尚未发生
async def test_checkpoint_failure_blocks_tool_side_effect(tmp_path: Path) -> None:
    target = tmp_path / "must-not-exist.txt"

    class Provider:
        # 子任务要求写文件，父任务只负责派生后台循环
        async def chat(self, messages: list, step: int, **kwargs: object) -> LlmResponse:
            if messages[0]["content"] == "child-goal":
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "write", "write_file", {"path": str(target), "content": "bad"},
                )])
            if step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "spawn", "spawn_agent", {"description": "disk fault", "prompt": "child-goal",
                                             "run_in_background": True},
                )])
            return LlmResponse("end_turn", text="parent done")

    save = BackgroundCheckpoint.save

    # 仅在记录即将执行工具时制造磁盘故障，启动和完成失败记录仍可保存
    def fail_tools(checkpoint: BackgroundCheckpoint, context: ExecutionContext, phase: str) -> None:
        if phase == "tools" and checkpoint.record.kind == "background":
            raise OSError("disk full")
        save(checkpoint, context, phase)

    manager = _manager(SessionStore(tmp_path / "sessions"), Provider())
    session = await manager.create("chat")
    try:
        with patch.object(BackgroundCheckpoint, "save", fail_tools):
            await manager.send_message(session.id, "launch")
            registry = manager._runners[session.id]._task_registry
            await asyncio.gather(*(task for task, _ in registry.all()))
        child_id = next(iter(registry._tasks))
        result = await AgentResultTool(registry).invoke({"run_id": child_id})
        assert result.is_error and not target.exists()
    finally:
        await manager.shutdown()


_CRASH_PROCESS = '''
import asyncio, json, sys
from pathlib import Path
from x_claude.core.config import XConfig
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.types import LlmResponse, ToolCallBlock
from x_claude.core.runner import AgentRunner
from x_claude.core.session.manager import SessionManager
from x_claude.core.session.store import SessionStore
from x_claude.core.tools.builtin.write_file import WriteFileTool
mode = sys.argv[2]
original_write = WriteFileTool.invoke
async def interrupted_write(self, params):
    result = await original_write(self, params)
    return await asyncio.Future()
if mode == "tools": WriteFileTool.invoke = interrupted_write
class Provider:
    async def chat(self, messages, step, **kwargs):
        if messages[0]["content"] == "crash-child":
            if step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock("write", "write_file", {"path":"once.txt","content":"saved-before-crash"})])
            return await asyncio.Future()
        if step == 1:
            return LlmResponse("tool_use", tool_calls=[ToolCallBlock("spawn", "spawn_agent", {"description":"restart proof","prompt":"crash-child","run_in_background":True})])
        return LlmResponse("end_turn", text="parent-done")
async def main():
    cfg = XConfig(); cfg.compaction.auto_threshold = 0
    store = SessionStore(Path(sys.argv[1]) / "sessions"); bus = EventBus()
    manager = SessionManager(store, lambda: AgentRunner(cfg, provider=Provider(), bus=bus), bus)
    session = await manager.create("chat")
    await manager.send_message(session.id, "launch")
    child = next(iter(manager._runners[session.id]._task_registry._tasks))
    path = store.runs_dir(session.id) / child / "background.json"
    while True:
        data = json.loads(path.read_text(encoding="utf-8"))
        if mode == "tools" and data["phase"] == "tools" and Path("once.txt").exists(): break
        if mode == "planning" and data["context"]["step"] == 1: break
        await asyncio.sleep(.01)
    print(json.dumps({"sid":session.id,"child":child}), flush=True)
    await asyncio.Future()
asyncio.run(main())
'''


# 功能：验证原进程被强制终止后，TUI 恢复会话会续跑原 run_id 并展示恢复通知
# 设计：独立 Python 进程真实写入检查点后被 kill；新 TCP Core handler 和 Textual TUI 只从磁盘恢复
@pytest.mark.parametrize("mode", ["planning", "tools"])
async def test_process_crash_then_tui_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c", _CRASH_PROCESS, str(tmp_path), mode, cwd=tmp_path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    try:
        assert proc.stdout is not None
        info = json.loads(await asyncio.wait_for(proc.stdout.readline(), 8))
    finally:
        if proc.returncode is None:
            proc.kill()
        await asyncio.wait_for(proc.wait(), 3)
    before = (tmp_path / "once.txt").stat().st_mtime_ns
    steps: list[int] = []

    class Provider:
        # 新进程只能接着执行第二步，若重放第一步则测试失败
        async def chat(self, messages: list, step: int, **kwargs: object) -> LlmResponse:
            steps.append(step)
            assert step == 2 and messages[0]["content"] == "crash-child"
            assert messages[-1]["content"][0]["tool_use_id"] == "write"
            return LlmResponse("end_turn", text="restarted-result")

    async with _serve(tmp_path, Provider()) as (cfg, manager, _, _):
        app = XTuiApp(cfg.host, cfg.port, resume_session_id=info["sid"])
        async with app.run_test(size=(100, 30)) as pilot:
            async with asyncio.timeout(5):
                while (info["sid"] not in manager._runners
                       or manager._runners[info["sid"]]._task_registry.get(info["child"]) is None):
                    await asyncio.sleep(.01)
                await pilot.pause()
            registry = manager._runners[info["sid"]]._task_registry
            await asyncio.gather(*(task for task, _ in registry.all()))
            await pilot.pause()
            result = await AgentResultTool(registry).invoke({"run_id": info["child"]})
            texts = "\n".join(str(widget.render()) for widget in app.query(Static))
            assert "后台任务恢复" in texts
            if mode == "tools":
                assert result.is_error and "interrupted_tool" in result.content
                assert "工具结果尚未确认" in texts
            else:
                assert result.content == "restarted-result" and not result.is_error
                assert "续跑" in texts
    assert steps == ([] if mode == "tools" else [2])
    assert (tmp_path / "once.txt").stat().st_mtime_ns == before


# 功能：验证普通启动的 TUI 在 daemon 换实例后自动恢复原会话及后台任务，/clear 更新重连目标
# 设计：保持同一 Textual App 打开，在相同 TCP 端口销毁并新建 Core handler，以检查真实重连而非启动参数
async def test_open_tui_reconnects_to_same_session(tmp_path: Path) -> None:
    waiting = asyncio.Event()

    class FirstProvider:
        # 子循环完成一次只读工具调用后等待进程关闭，根任务立即结束
        async def chat(self, messages: list, step: int, **kwargs: object) -> LlmResponse:
            if messages[0]["content"] == "reconnect-child":
                if step == 1:
                    return LlmResponse("tool_use", tool_calls=[ToolCallBlock("list", "list_dir", {})])
                waiting.set()
                await asyncio.Event().wait()
            if step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "spawn", "spawn_agent", {"description": "reconnect", "prompt": "reconnect-child",
                                             "run_in_background": True},
                )])
            return LlmResponse("end_turn", text="parent done")

    steps: list[int] = []

    class NextProvider:
        # 恢复必须接续第二步，不能从第一步重新启动
        async def chat(self, step: int, **kwargs: object) -> LlmResponse:
            steps.append(step)
            assert step == 2
            return LlmResponse("end_turn", text="reconnect done")

    first_ctx = _serve(tmp_path, FirstProvider())
    cfg, first, _, _ = await first_ctx.__aenter__()
    first_open = True
    next_ctx = None
    try:
        app = XTuiApp(cfg.host, cfg.port)
        async with app.run_test(size=(100, 30)) as pilot:
            async with asyncio.timeout(4):
                while app._session_id is None:
                    await asyncio.sleep(.01)
                await pilot.pause()
            original = app._session_id
            send_task = asyncio.create_task(first.send_message(original, "launch"))
            async with asyncio.timeout(3):
                while not any(key[1] == "spawn" for key in app._pending_permission_blocks):
                    await asyncio.sleep(.01)
                await pilot.pause()
                await pilot.press("enter")
            await send_task
            await asyncio.wait_for(waiting.wait(), 4)
            child = next(iter(first._runners[original]._task_registry._tasks))
            await first_ctx.__aexit__(None, None, None)
            first_open = False
            next_ctx = _serve(tmp_path, NextProvider(), port=cfg.port)
            _, second, _, _ = await next_ctx.__aenter__()
            async with asyncio.timeout(7):
                while not steps or app._session_id is None:
                    await asyncio.sleep(.01)
                await pilot.pause()
            assert app._session_id == original
            registry = second._runners[original]._task_registry
            await asyncio.gather(*(task for task, _ in registry.all()))
            assert (await AgentResultTool(registry).invoke({"run_id": child})).content == "reconnect done"
            assert steps == [2]
            await app._do_clear()
            assert app._session_id != original and app._resume_session_id == app._session_id
    finally:
        if next_ctx is not None:
            await next_ctx.__aexit__(None, None, None)
        if first_open:
            await first_ctx.__aexit__(None, None, None)
