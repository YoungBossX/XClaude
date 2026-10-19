from __future__ import annotations

import asyncio

import pytest

from x_claude.core.config import XConfig
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.types import LlmResponse, ToolCallBlock
from x_claude.core.permissions.manager import PermissionManager
from x_claude.core.permissions.policy import DEFAULT_POLICIES, PermissionDecision, ToolPolicy
from x_claude.core.runner import AgentRunner


class ApprovalScenario:
    # 仅替换外部模型，派生、文件工具、权限、查询、事件及父循环均走真实实现。
    def __init__(self, children=1, root_id="parent"):
        self.children = children
        self.root_id = root_id
        self.child_ids = []
        self.requests = []
        self.parent_steps = []
        self.events = []
        self.ready = asyncio.Event()
        self.queried = asyncio.Event()

    # 观察真实子任务与审批事件，并在首次查询已经完成后提供测试屏障。
    async def collect(self, event):
        data = event.model_dump()
        self.events.append(data)
        if data["type"] == "subagent.started":
            self.child_ids.append(data["run_id"])
        if data["type"] == "permission.requested" and data["tool_name"] == "write_file":
            self.requests.append(data)
            if len(self.requests) == self.children:
                self.ready.set()
        if (data["type"] == "tool.call_finished"
                and data["tool_name"] == "agent_result"):
            self.queried.set()

    # 固定父任务快速查询和子任务待审批写入，排除真实模型随机性及付费请求。
    async def chat(self, messages, run_id, step, **kwargs):
        if self.root_id is None:
            self.root_id = run_id
        if run_id != self.root_id:
            if step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "write", "write_file", {"path": f"{messages[0]['content']}.txt", "content": "approved"},
                )])
            result = messages[-1]["content"][0]
            return LlmResponse("end_turn", text=(
                "CHILD_DENIED: " + result["content"] if result.get("is_error") else "CHILD_WROTE"
            ))
        self.parent_steps.append(step)
        if step == 1:
            return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                f"spawn-{i}", "spawn_agent", {"description": "approval child", "prompt": f"child-{i}",
                                             "run_in_background": True},
            ) for i in range(self.children)])
        await self.ready.wait()
        last = messages[-1]["content"]
        if isinstance(last, list) and len(last) == self.children and all(
            block.get("content", "").startswith("CHILD_") for block in last
        ):
            return LlmResponse("end_turn", text=" | ".join(b["content"] for b in last))
        return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
            f"query-{step}-{i}", "agent_result", {"run_id": child_id, "wait_ms": 0},
        ) for i, child_id in enumerate(self.child_ids)])


# 构造只写临时目录的真实运行器，保留任务预算且禁止读取个人上下文。
def make_runner(tmp_path, monkeypatch, *, children=1, permission_timeout=3, runtime=10,
                scenario=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("x_claude.core.runner.load_context_file", lambda _: "")
    config = XConfig()
    config.agent.max_steps = 8
    config.agent.max_runtime_s = runtime
    config.compaction.auto_threshold = 0
    provider = scenario or ApprovalScenario(children)
    bus = EventBus()
    bus.subscribe(provider.collect)
    permissions = PermissionManager(policies={
        **DEFAULT_POLICIES,
        "spawn_agent": ToolPolicy(PermissionDecision.ALLOW),
        "agent_result": ToolPolicy(PermissionDecision.ALLOW),
    }, timeout_s=permission_timeout)
    runner = AgentRunner(config, provider=provider, bus=bus, permission_manager=permissions,
                         runs_dir=tmp_path / "runs")
    return runner, provider, permissions


# 功能：后台子任务待审批时，父循环不再空查询耗步；批准或拒绝后可恢复并取回真实结果。
# 设计：真实 Runner 使用 wait_ms=0 强制旧问题，审批前检查模型调用次数与文件，审批后检查结果。
@pytest.mark.parametrize("decision", ["allow_once", "deny_once"])
async def test_parent_waits_for_child_approval_without_spending_steps(tmp_path, monkeypatch, decision):
    runner, provider, permissions = make_runner(tmp_path, monkeypatch)
    task = asyncio.create_task(runner.run_and_capture("parent task", run_id="parent"))
    try:
        await asyncio.wait_for(provider.queried.wait(), 3)
        await asyncio.sleep(.05)
        assert provider.parent_steps == [1, 2], "parent polled the model while approval was pending"
        assert not task.done()
        assert not (tmp_path / "child-0.txt").exists()
        request = provider.requests[0]
        assert permissions.respond("write", decision, session_id=request["session_id"],
                                   run_id=request["run_id"])
        outcome = await asyncio.wait_for(task, 3)
        assert outcome.status == "success"
        assert (tmp_path / "child-0.txt").exists() == (decision == "allow_once")
        assert ("CHILD_WROTE" if decision == "allow_once" else "CHILD_DENIED") in outcome.result
        assert not permissions.pending_events()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await runner.shutdown()


# 功能：审批超时后父任务恢复并收到明确超时原因，不因无效查询触发最大步数失败。
# 设计：使用真实短审批时限，验证无文件副作用、timeout 事件和模型最终收到的工具错误。
async def test_approval_timeout_unblocks_parent_with_explicit_reason(tmp_path, monkeypatch):
    runner, provider, permissions = make_runner(tmp_path, monkeypatch, permission_timeout=.15)
    try:
        outcome = await asyncio.wait_for(runner.run_and_capture("parent task", run_id="parent"), 4)
        assert outcome.status == "success", outcome.reason
        assert "timeout" in outcome.result.lower()
        assert not (tmp_path / "child-0.txt").exists()
        assert not permissions.pending_events()
        assert any(e["type"] == "permission.denied" and e["decision"] == "timeout"
                   for e in provider.events)
    finally:
        await runner.shutdown()


# 功能：并行审批只解决一个请求时父任务保持等待，全部结算后分别返回成功和拒绝。
# 设计：三个真实后台写任务采用同名 tool_use_id，按 run_id 审批以验证隔离和通知不会丢失。
async def test_parallel_approvals_do_not_resume_parent_early(tmp_path, monkeypatch):
    runner, provider, permissions = make_runner(tmp_path, monkeypatch, children=3)
    task = asyncio.create_task(runner.run_and_capture("parent task", run_id="parent"))
    try:
        await asyncio.wait_for(provider.queried.wait(), 3)
        await asyncio.sleep(.03)
        for request, decision in zip(provider.requests, ("allow_once", "deny_once", "allow_once"), strict=True):
            assert provider.parent_steps == [1, 2]
            assert permissions.respond("write", decision, session_id=request["session_id"],
                                       run_id=request["run_id"])
            await asyncio.sleep(.03)
        outcome = await asyncio.wait_for(task, 3)
        assert outcome.status == "success", outcome.reason
        assert outcome.result.count("CHILD_WROTE") == 2
        assert outcome.result.count("CHILD_DENIED") == 1
        assert len(list(tmp_path.glob("child-*.txt"))) == 2
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await runner.shutdown()


# 功能：等待审批不延长任务树总运行时限，预算耗尽会终止且不执行未授权工具。
# 设计：审批无独立超时而任务预算很短，验证最终原因是 runtime_budget 而非空轮询耗步。
async def test_wait_preserves_runtime_deadline(tmp_path, monkeypatch):
    runner, provider, permissions = make_runner(tmp_path, monkeypatch, permission_timeout=0, runtime=.2)
    try:
        outcome = await asyncio.wait_for(runner.run_and_capture("parent task", run_id="parent"), 3)
        assert outcome.reason == "runtime_budget"
        assert provider.parent_steps == [1, 2]
        assert not (tmp_path / "child-0.txt").exists()
    finally:
        await runner.shutdown()
    assert not permissions.pending_events()


# 功能：取消等待中的父循环不会误批准或取消共享审批，显式关闭运行器仍清理所有子任务。
# 设计：父取消后检查子任务审批仍有效，再调用生产 shutdown 验证清理与无副作用。
async def test_cancel_waiter_and_shutdown_keep_existing_semantics(tmp_path, monkeypatch):
    runner, provider, permissions = make_runner(tmp_path, monkeypatch, permission_timeout=0)
    task = asyncio.create_task(runner.run_and_capture("parent task", run_id="parent"))
    try:
        await asyncio.wait_for(provider.queried.wait(), 3)
        await asyncio.sleep(.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert permissions.pending_events()
        assert all(not child.done() for child, _ in runner._task_registry.all())
        assert not (tmp_path / "child-0.txt").exists()
    finally:
        await runner.shutdown()
    assert not permissions.pending_events()


# 功能：TUI 在父循环等待子审批时仍可批准、拒绝、清空和退出，不留下未授权写入。
# 设计：挂载真实 Textual UI 并连接 TCP，用键盘结算子审批或调用原有清空退出动作。
@pytest.mark.parametrize("action", ["allow", "deny", "clear", "exit"])
async def test_tui_remains_responsive_during_child_approval(tmp_path, monkeypatch, action):
    from tests.integration.test_runtime_hardening import _serve
    from tests.integration.test_windows_acceptance import wait_until
    from x_claude.tui.app import ChatTextArea, PermissionSelect, XTuiApp

    monkeypatch.chdir(tmp_path)
    for name in ("spawn_agent", "agent_result"):
        monkeypatch.setitem(DEFAULT_POLICIES, name, ToolPolicy(PermissionDecision.ALLOW))
    provider = ApprovalScenario(root_id=None)
    async with _serve(tmp_path, provider, permission_timeout=10) as (cfg, _, bus, store):
        bus.subscribe(provider.collect)
        app = XTuiApp(cfg.host, cfg.port)
        async with app.run_test(size=(110, 40)) as pilot:
            await wait_until(lambda: app._session_id and app._client)
            prompt = app.query_one("#prompt", ChatTextArea)
            await wait_until(lambda: not prompt.disabled and prompt.has_focus)
            prompt.text = "parent task"
            await pilot.press("enter")
            await asyncio.wait_for(provider.queried.wait(), 4)
            await wait_until(lambda: bool(app.query(PermissionSelect)))
            await pilot.pause()
            assert provider.parent_steps == [1, 2]
            assert app._busy
            old_sid = app._session_id
            if action in ("allow", "deny"):
                await pilot.press("y" if action == "allow" else "n")
                await wait_until(lambda: not app._busy and not prompt.disabled)
                assert (tmp_path / "child-0.txt").exists() == (action == "allow")
                assert not app.query(PermissionSelect)
            elif action == "clear":
                await app._do_clear()
                await wait_until(lambda: app._session_id != old_sid and not prompt.disabled)
                assert store.read_messages(app._session_id) == []
                assert not app.query(PermissionSelect)
            else:
                await pilot.press("ctrl+q")
                await wait_until(lambda: not app.is_running)
            if action in ("clear", "exit"):
                assert store.read_meta(old_sid).status == "closed"
                assert not (tmp_path / "child-0.txt").exists()


# 功能：孙任务审批阻塞传播至根任务，批准、拒绝、超时或取消后解除等待且不遗留依赖。
# 设计：组合前台及后台叶子与四种结算，使用真实派生、零等待查询、审批和文件工具。
@pytest.mark.parametrize("background_leaf", [True, False])
@pytest.mark.parametrize("settlement", ["allow_once", "deny_once", "timeout", "cancel"])
async def test_nested_child_permission_wait_propagates_to_root(
    tmp_path, monkeypatch, background_leaf, settlement,
):
    class NestedScenario(ApprovalScenario):
        # 记录父子关联及中间任务首次查询屏障，避免根任务过早访问未派生的叶子。
        def __init__(self):
            super().__init__()
            self.by_parent = {}
            self.middle_queried = asyncio.Event()
            self.middle_steps = []

        # 观察既有派生与查询事件，不伪造权限请求或直接设置上下文字段。
        async def collect(self, event):
            await super().collect(event)
            data = event.model_dump()
            if data["type"] == "subagent.started":
                self.by_parent[data["parent_run_id"]] = data["run_id"]
            if (data["type"] == "tool.call_finished" and data["tool_name"] == "agent_result"
                    and data["run_id"] != "parent"):
                self.middle_queried.set()

        # 根派生中间任务，中间派生叶子；叶子执行真实待审批文件写入。
        async def chat(self, messages, run_id, step, **kwargs):
            if messages[0]["content"] == "leaf":
                return await super().chat(messages, run_id, step, **kwargs)
            (self.parent_steps if run_id == "parent" else self.middle_steps).append(step)
            if step == 1:
                goal = "middle" if run_id == "parent" else "leaf"
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "spawn", "spawn_agent", {"description": goal, "prompt": goal,
                                             "run_in_background": run_id == "parent" or background_leaf},
                )])
            await self.ready.wait()
            if run_id == "parent" and background_leaf:
                await self.middle_queried.wait()
                await asyncio.sleep(0)
            last = messages[-1]["content"][0]["content"]
            if last.startswith(("CHILD_", "Subagent was cancelled.")):
                return LlmResponse("end_turn", text=last)
            return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                f"query-{step}", "agent_result", {"run_id": self.by_parent[run_id], "wait_ms": 0},
            )])

    provider = NestedScenario()
    runner, _, permissions = make_runner(
        tmp_path, monkeypatch, scenario=provider,
        permission_timeout=.5 if settlement == "timeout" else 3,
    )
    task = asyncio.create_task(runner.run_and_capture("parent task", run_id="parent"))
    try:
        await asyncio.wait_for(provider.queried.wait(), 3)
        await asyncio.sleep(.05)
        assert provider.middle_steps == ([1, 2] if background_leaf else [1])
        assert provider.parent_steps == [1, 2]
        request = provider.requests[0]
        middle_id = provider.by_parent["parent"]
        if settlement == "cancel":
            cancelled_id = request["run_id"] if background_leaf else middle_id
            entry = runner._task_registry.get(cancelled_id)
            assert entry is not None
            entry[0].cancel()
        elif settlement != "timeout":
            assert permissions.respond("write", settlement, session_id=request["session_id"],
                                       run_id=request["run_id"])
        outcome = await asyncio.wait_for(task, 3)
        assert outcome.status == "success", outcome.reason
        if settlement == "allow_once":
            assert outcome.result == "CHILD_WROTE"
            assert (tmp_path / "leaf.txt").read_text(encoding="utf-8") == "approved"
        else:
            expected = {"deny_once": "CHILD_DENIED", "timeout": "timeout", "cancel": "cancelled"}
            assert expected[settlement] in outcome.result
            assert not (tmp_path / "leaf.txt").exists()
        assert not permissions.pending_events()
        assert runner._task_registry.permission_dependencies(middle_id) == (middle_id,)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await runner.shutdown()
