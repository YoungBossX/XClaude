from __future__ import annotations

import asyncio
import gc
import time
import weakref
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tests.integration.test_background_restart import _manager
from tests.integration.test_root_restart import _wait_root
from tests.integration.test_runtime_hardening import _serve
from x_claude.core.bus.events import (
    RunStartedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from x_claude.core.config import AgentConfig, XConfig
from x_claude.core.context import ExecutionContext
from x_claude.core.events.bus import EventBus
from x_claude.core.events.writer import EventWriter
from x_claude.core.llm.types import LlmResponse, UsageStats
from x_claude.core.loop import AgentLoop
from x_claude.core.resources import ResourceLimitExceeded, TaskBudget
from x_claude.core.runner import AgentRunner
from x_claude.core.session.store import SessionStore
from x_claude.core.subagent.checkpoint import (
    BackgroundCheckpoint,
    BackgroundRecord,
    ContextSnapshot,
)
from x_claude.core.subagent.registry import BackgroundTaskRegistry
from x_claude.core.subagent.tool import AgentResultTool, SpawnAgentTool
from x_claude.core.tools.registry import ToolRegistry
from x_claude.tui.app import ToolCallBlock, XTuiApp


# 所有测试使用临时项目及模型替身，不触碰用户会话或真实模型 API
@pytest.fixture(autouse=True)
def _workspace(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)


# 功能：验证终态上下文不检查预算、不发布错误事件、不重写原成功或失败状态
# 设计：预算和检查点回调均设置为一旦调用即失败，同时覆盖成功与已有失败原因
@pytest.mark.parametrize("status,reason", [("success", None), ("failed", "llm_auth")])
async def test_terminal_loop_skips_execution_budget(status, reason) -> None:
    context = ExecutionContext("run", "goal", 5, status=status, reason=reason, result="saved")
    budget = Mock(spec=TaskBudget)
    budget.remaining_s.side_effect = AssertionError("terminal context must not check budget")
    checkpoint = Mock(side_effect=AssertionError("terminal context must not overwrite checkpoint"))
    provider = AsyncMock()
    await AgentLoop(provider, ToolRegistry(), EventBus(), budget=budget,
                    checkpoint=checkpoint).run(context)
    assert (context.status, context.reason, context.result) == (status, reason, "saved")
    provider.chat.assert_not_awaited()


# 功能：验证已成功但未提交的主任务在预算过期或账本损坏时仍能幂等补交
# 设计：真实会话写入后中断元数据提交，重建 manager 并禁止恢复时调用模型
@pytest.mark.parametrize("budget_state", ["expired", "corrupt"])
async def test_finished_root_finalizes_without_budget(tmp_path, budget_state) -> None:
    class Provider:
        # 生成确定回复及 usage，使测试覆盖真实预算账本
        async def chat(self, **kwargs):
            return LlmResponse("end_turn", text="saved answer", usage=UsageStats(1, 1))

    store = SessionStore(tmp_path / "sessions")
    first = _manager(store, Provider())
    session = await first.create("chat")
    original = store.write_meta
    fired = False

    # 只中断回答落盘之后的一次元数据提交，保留成功检查点用于恢复
    def fail_once(value):
        nonlocal fired
        if not fired and store.read_messages(value.id)[-1]["role"] == "assistant":
            fired = True
            raise OSError("metadata interrupted")
        return original(value)

    try:
        with patch.object(store, "write_meta", fail_once):
            run_id = await first.send_message(session.id, "answer")
    finally:
        await first.shutdown()
    path = store.runs_dir(session.id) / run_id / "root.json"
    before = BackgroundCheckpoint.load(path, session.id).record
    assert before.context.status == "success" and not before.thread_committed
    history = (store.session_dir(session.id) / "thread.jsonl").read_bytes()
    if budget_state == "corrupt":
        (path.parent / "budget.json").write_text("invalid json", encoding="utf-8")
    second = _manager(store, None)
    try:
        await second.resume(session.id)
        with (
            patch("x_claude.core.resources.time.time", return_value=time.time() + 3600),
            patch("x_claude.core.runner.AnthropicProvider", side_effect=AssertionError("no model")),
        ):
            await second.recover_background(session.id)
            await _wait_root(second, session.id)
        after = BackgroundCheckpoint.load(path, session.id).record
        assert (after.context.status, after.context.reason, after.context.result) == (
            "success", None, "saved answer",
        )
        assert after.thread_committed
        assert (store.session_dir(session.id) / "thread.jsonl").read_bytes() == history
    finally:
        await second.shutdown()


# 功能：验证首次尾部回放丢掉工具开始事件后，成功或失败结果仍完整显示且不会重复
# 设计：真实 TCP 和挂载 TUI，先写超过回放窗口的并行日志，再发布工具结束事件
@pytest.mark.parametrize("failed", [False, True])
async def test_tail_replay_renders_orphan_tool_result(tmp_path, failed) -> None:
    async with _serve(tmp_path, object()) as (cfg, manager, bus, store):
        session = await manager.create("chat")
        path = store.runs_dir(session.id) / "root" / "events.jsonl"
        async with EventWriter(path) as writer:
            writer.subscribe(bus)
            await bus.publish(RunStartedEvent(run_id="root", session_id=session.id,
                                               goal="test", ts="1"))
            await bus.publish(ToolCallStartedEvent(run_id="root", tool_use_id="slow",
                                                    tool_name="bash", params={"command": "long"},
                                                    ts="2"))
            await writer.handle(ToolCallFinishedEvent(
                run_id="parallel-child", tool_use_id="large", tool_name="read_file",
                output="x" * 300000, elapsed_ms=1, ts="3",
            ))
            app = XTuiApp(cfg.host, cfg.port, resume_session_id=session.id)
            async with app.run_test(size=(110, 35)) as pilot:
                async with asyncio.timeout(5):
                    while not app._replay_progress.offsets:
                        await asyncio.sleep(.01)
                await pilot.pause()
                assert not app.query(ToolCallBlock)
                event = (ToolCallFailedEvent(
                    run_id="root", tool_use_id="slow", tool_name="bash", elapsed_ms=20,
                    error_class="runtime_error", error_message="tool error", ts="4",
                ) if failed else ToolCallFinishedEvent(
                    run_id="root", tool_use_id="slow", tool_name="bash", elapsed_ms=20,
                    output="tool answer", ts="4",
                ))
                await bus.publish(event)
                async with asyncio.timeout(3):
                    while event.event_id not in app._seen_event_ids:
                        await asyncio.sleep(.01)
                await pilot.pause()
                block = app.query_one(ToolCallBlock)
                assert block._finished and block._is_error is failed
                assert block._output == ("tool error" if failed else "tool answer")
                assert not block._params_available and "参数未知" in block._params_full
                block.on_click()
                assert "expanded" in block.classes
                app._handle_event(event.model_dump())
                await pilot.pause()
                assert len(app.query(ToolCallBlock)) == 1
                assert not app._pending_tool_blocks


# 创建包含完整历史的终态检查点，缓存测试只消费真实的持久格式
def _saved_child(runs_dir: Path, run_id: str, *, result: str = "answer",
                 status: str = "success", reason: str | None = None) -> ExecutionContext:
    context = ExecutionContext(run_id, "goal", 5, status=status, reason=reason, result=result)
    context.add_assistant_message([{"type": "text", "text": "long history" * 1000}])
    record = BackgroundRecord(
        run_id=run_id, session_id="sid", parent_run_id="root", description="child",
        cwd=str(Path.cwd()), depth=0, context=ContextSnapshot.capture(context),
        tools=[], phase="finished", state="completed",
    )
    BackgroundCheckpoint(runs_dir / run_id / "background.json", record).save(context, "finished")
    return context


# 功能：验证终态结果缓存数量有界，旧结果仍可按需读盘，完整消息不常驻
# 设计：保存并查询超过缓存上限的真实检查点，用弱引用检测旧上下文已释放
async def test_result_cache_evicts_and_reloads_without_history(tmp_path) -> None:
    registry = BackgroundTaskRegistry(max_completed=2)
    registry.bind_storage(tmp_path, "sid")
    refs = []
    for index in range(8):
        run_id = f"child-{index}"
        context = _saved_child(tmp_path, run_id, result=f"answer-{index}")
        refs.append(weakref.ref(context))
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)
        registry.register(run_id, future, context)
        assert len(registry.all()) <= 2
    del context
    gc.collect()
    assert all(ref() is None for ref in refs)
    tool = AgentResultTool(registry)
    for index in range(8):
        result = await tool.invoke({"run_id": f"child-{index}"})
        assert result.content == f"answer-{index}" and not result.is_error
        assert len(registry.all()) <= 2
        assert all(not ctx.messages and not ctx.goal for _, ctx in registry.all())
    assert registry.get("../child-0") is None
    with pytest.raises(ValueError, match="cannot change"):
        registry.bind_storage(tmp_path / "other", "other")


# 功能：验证单个超大结果不突破缓存容量，但查询仍返回磁盘中的完整结果
# 设计：容量设为十字符，查询超过容量的真实结果并确认不驻留缓存
async def test_large_result_is_returned_without_caching(tmp_path) -> None:
    registry = BackgroundTaskRegistry(max_result_chars=10)
    registry.bind_storage(tmp_path, "sid")
    _saved_child(tmp_path, "child", result="内容" * 10000)
    result = await AgentResultTool(registry).invoke({"run_id": "child"})
    assert result.content == "内容" * 10000 and not result.is_error
    assert not registry.all()


# 功能：验证缓存回收不淘汰运行中、未落盘或磁盘读取失败的结果
# 设计：零缓存上限下并列三种状态，真实读盘故障期间仍须保留唯一可查询结果
async def test_cache_retains_active_and_unpersisted_results(tmp_path) -> None:
    registry = BackgroundTaskRegistry(max_completed=0)
    registry.bind_storage(tmp_path, "sid")
    active = asyncio.get_running_loop().create_future()
    context = ExecutionContext("active", "still working", 5)
    registry.register("active", active, context)
    unsaved = ExecutionContext("unsaved", "goal", 5, status="failed", reason="persistence_error")
    done = asyncio.get_running_loop().create_future()
    done.set_result(None)
    registry.register("unsaved", done, unsaved)
    saved = _saved_child(tmp_path, "saved")
    with patch.object(BackgroundCheckpoint, "load", side_effect=OSError("unreadable")):
        registry.register("saved", done, saved)
    assert registry.get("active") == (active, context)
    assert registry.get("unsaved")[1].reason == "persistence_error"
    assert registry.get("unsaved")[1] is unsaved and unsaved.messages
    assert registry.get("saved")[1] is saved and saved.messages
    assert not context.is_done() and context.messages
    await registry.cancel_all()


# 功能：验证终态缓存淘汰后仍区分失败及取消，不把错误结果变为成功
# 设计：零容量强制每次读盘，同时检查 agent_result 的错误标记和取消语义
@pytest.mark.parametrize("reason", ["llm_auth", "cancelled"])
async def test_evicted_failure_and_cancelled_result_keep_semantics(tmp_path, reason) -> None:
    registry = BackgroundTaskRegistry(max_completed=0)
    registry.bind_storage(tmp_path, "sid")
    _saved_child(tmp_path, "child", status="failed", reason=reason, result="")
    result = await AgentResultTool(registry).invoke({"run_id": "child"})
    assert result.is_error
    assert ("cancelled" if reason == "cancelled" else reason) in result.content
    assert not registry.all()


# 功能：验证回收预算对象后从账本重载仍保持已花费额度，活跃引用共享同一对象
# 设计：先消费及持有引用，再释放并用弱引用确认回收，重载后要求拒绝超额请求
async def test_budget_cache_releases_idle_objects_but_not_accounting(tmp_path) -> None:
    cfg = XConfig()
    cfg.agent = AgentConfig(max_total_tokens=100, max_runtime_s=0)
    runner = AgentRunner(cfg, runs_dir=tmp_path)
    budget = runner._budget_for("root", tmp_path)
    identity = budget.reserve(80)
    budget.settle(identity, LlmResponse("end_turn", usage=UsageStats(60, 20)))
    assert runner._budget_for("root", tmp_path) is budget
    reference = weakref.ref(budget)
    del budget
    gc.collect()
    assert reference() is None and not runner._budgets
    reloaded = runner._budget_for("root", tmp_path)
    assert reloaded.used_tokens == 80
    with pytest.raises(ResourceLimitExceeded):
        reloaded.reserve(21)


# 功能：验证真实后台任务完成后释放上下文和预算，淘汰的子任务仍可查询
# 设计：持续运行多个独立任务树并等待实际结束，缓存容量设为二以覆盖跨树回收
async def test_completed_child_trees_release_runtime_memory(tmp_path) -> None:
    class Provider:
        calls = 0

        # 只统计调用次数，不像 Mock 一样保存 bus 参数并意外延长运行时引用
        async def chat(self, **kwargs):
            self.calls += 1
            return LlmResponse("end_turn", text="child result", usage=UsageStats(1, 1))

    cfg = XConfig()
    cfg.compaction.auto_threshold = 0
    provider = Provider()
    runner = AgentRunner(cfg, provider=provider, runs_dir=tmp_path)
    runner._task_registry = BackgroundTaskRegistry(max_completed=2)
    children = []
    try:
        for index in range(8):
            budget = runner._budget_for(f"root-{index}", tmp_path)
            tool = SpawnAgentTool(
                runner._get_provider(budget), EventBus(), f"root-{index}", None, 5,
                runner._task_registry, tmp_path, "sid", config=cfg, budget=budget,
            )
            result = await tool.invoke({"description": "child", "prompt": "child goal",
                                        "run_in_background": True})
            children.append(result.content.split("run_id=")[1].split(".")[0])
            await asyncio.gather(*(task for task, _ in runner._task_registry.all()))
            await asyncio.sleep(0)
            del tool, budget
        gc.collect()
        assert len(runner._task_registry.all()) <= 2
        assert not runner._budgets
        assert all(not context.messages for _, context in runner._task_registry.all())
        result = await AgentResultTool(runner._task_registry).invoke({"run_id": children[0]})
        assert result.content == "child result" and not result.is_error
        assert provider.calls == 8
    finally:
        await runner.shutdown()
