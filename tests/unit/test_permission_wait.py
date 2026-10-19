from __future__ import annotations

import asyncio

import pytest

from x_claude.core.context import ExecutionContext
from x_claude.core.permissions.manager import PermissionManager
from x_claude.core.subagent.registry import BackgroundTaskRegistry
from x_claude.core.subagent.tool import AgentResultTool


# 创建真实审批协程并用事件确认其已进入等待，测试不访问管理器的私有字典。
async def request_permission(manager, run_id="child", session_id="s"):
    ready = asyncio.Event()

    # 审批请求已登记后通知调用方，避免测试依赖偶然的调度顺序。
    async def emit(_):
        ready.set()

    task = asyncio.create_task(manager.check_and_wait(
        "write", "write_file", {"path": "target.txt", "content": "value"},
        session_id, emit, run_id,
    ))
    await asyncio.wait_for(ready.wait(), 1)
    return task


# 功能：零等待及普通等待查询均能区分待审批，不等待完整轮询时限、不误返回成功。
# 设计：真实审批 Future 驱动注册子任务，比较批准前查询状态和批准后最终结果。
@pytest.mark.parametrize("wait_ms", [0, 30000])
async def test_result_reports_permission_state_and_eventual_completion(wait_ms):
    manager = PermissionManager(timeout_s=0)
    approval = await request_permission(manager)
    context = ExecutionContext("child", "work", 5)

    # 用户真正批准后才完成子任务；结果查询本身不能改变审批。
    async def child():
        allowed, _ = await approval
        assert allowed
        context.result = "confirmed result"
        context.mark_success()

    task = asyncio.create_task(child())
    registry = BackgroundTaskRegistry()
    registry.register("child", task, context)
    tool = AgentResultTool(registry, manager, "s")
    try:
        result = await asyncio.wait_for(tool.invoke({"run_id": "child", "wait_ms": wait_ms}), .5)
        assert "waiting_for_approval" in result.content
        assert not result.is_error
        assert not task.done()
        assert manager.respond("write", "allow_once", session_id="s", run_id="child")
        await task
        result = await tool.invoke({"run_id": "child", "wait_ms": 0})
        assert result.content == "confirmed result"
        assert not result.permission_wait_runs
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# 功能：等待仅受所选会话和任务约束，其他审批不会阻塞，错误身份不能提前放行。
# 设计：同时登记同会话其他任务与另一会话请求，只解决目标请求即可结束目标等待。
async def test_wait_is_scoped_to_requested_runs_and_session():
    manager = PermissionManager(timeout_s=0)
    tasks = [await request_permission(manager, run, session)
             for run, session in (("target", "s"), ("other", "s"), ("foreign", "s2"))]
    waiter = asyncio.create_task(manager.wait_for_runs("s", ("target", "foreign")))
    try:
        await asyncio.sleep(0)
        assert not waiter.done()
        assert not manager.respond("write", "allow_once", session_id="s2", run_id="target")
        assert manager.respond("write", "deny_once", session_id="s", run_id="target")
        await asyncio.wait_for(waiter, 1)
        assert await tasks[0] == (False, "deny_once")
        assert all(not task.done() for task in tasks[1:])
    finally:
        waiter.cancel()
        for task in tasks:
            task.cancel()
        await asyncio.gather(waiter, *tasks, return_exceptions=True)


# 功能：等待者取消不传播到审批，子任务取消或会话断连则会唤醒审批依赖等待。
# 设计：先取消一个等待者，再分别取消审批协程或断连会话，确认无悬挂 Future。
@pytest.mark.parametrize("settlement", ["cancel_child", "disconnect"])
async def test_wait_cancellation_does_not_cancel_approval(settlement):
    manager = PermissionManager(timeout_s=0)
    approval = await request_permission(manager)
    first = asyncio.create_task(manager.wait_for_runs("s", ("child",)))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not approval.done()
    assert manager.has_pending_requests("s", ("child",))
    second = asyncio.create_task(manager.wait_for_runs("s", ("child",)))
    try:
        if settlement == "cancel_child":
            approval.cancel()
        else:
            manager.cancel_session("s")
        await asyncio.wait_for(second, 1)
        await asyncio.gather(approval, return_exceptions=True)
        assert not manager.pending_events()
    finally:
        approval.cancel()
        second.cancel()
        await asyncio.gather(approval, second, return_exceptions=True)


# 功能：审批通知发布异常时，同样等待该请求的父任务不能永久悬挂。
# 设计：真实审批已登记且等待者已订阅后注入事件投递故障，要求等待结束且无残留请求。
async def test_failed_permission_notification_releases_waiters():
    manager = PermissionManager(timeout_s=0)
    entered, release = asyncio.Event(), asyncio.Event()

    # 模拟外部事件订阅者在异步投递期间抛出故障，不替换审批或等待实现。
    async def emit(_):
        entered.set()
        await release.wait()
        raise OSError("notification unavailable")

    approval = asyncio.create_task(manager.check_and_wait(
        "write", "write_file", {}, "s", emit, "child",
    ))
    await entered.wait()
    waiter = asyncio.create_task(manager.wait_for_runs("s", ("child",)))
    try:
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(OSError, match="notification unavailable"):
            await approval
        await asyncio.wait_for(waiter, .2)
        assert not manager.pending_events()
    finally:
        approval.cancel()
        waiter.cancel()
        await asyncio.gather(approval, waiter, return_exceptions=True)
