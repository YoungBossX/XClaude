from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from x_claude.core.compact.compactor import Compactor
from x_claude.core.context import ExecutionContext
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.types import LlmResponse
from x_claude.core.subagent.registry import BackgroundTaskRegistry
from x_claude.core.subagent.tool import AgentResultTool


# 功能：验证摘要遗漏最新输出要求时，压缩后的真实上下文仍逐字包含用户请求
# 设计：外部模型故意只返回旧目标，验收内存及持久化消息中的原始请求
async def test_compaction_keeps_current_request_verbatim(tmp_path: Path) -> None:
    goal = 'Write answer.json with EXACT keys: project, port; use integer 8310.'
    context = ExecutionContext("r", goal, 5)
    context.messages = [{"role": "user", "content": "Review packaging " * 300},
                        {"role": "assistant", "content": "Older review done."},
                        {"role": "user", "content": goal}]

    class Provider:
        # 返回不含当前任务的短摘要，以隔离摘要模型的遗漏行为
        async def chat(self, **kwargs):
            return LlmResponse("end_turn", text="Packaging constraints: project orbit, port 8310.")

    assert await Compactor(EventBus(), tmp_path, "s").compact(context, Provider())
    assert context.messages[-1] == {"role": "user", "content": goal}


# 功能：验证普通结果查询可等待后台完成，避免模型为了空轮询重复消耗步骤
# 设计：任务在事件循环下一轮完成；旧实现会直接返回still running
async def test_result_waits_for_completion_by_default() -> None:
    context = ExecutionContext("child", "work", 5)

    # 异步完成真实注册任务并提交终态
    async def finish():
        await asyncio.sleep(.01)
        context.result = "finished-value"
        context.mark_success()

    task = asyncio.create_task(finish())
    registry = BackgroundTaskRegistry()
    registry.register("child", task, context)
    try:
        result = await AgentResultTool(registry).invoke({"run_id": "child"})
        assert result.content == "finished-value"
    finally:
        await task


# 功能：验证有界等待超时后子任务仍继续，等待者取消也不能误取消子任务
# 设计：Event控制真实子协程，对同一任务先短超时再取消查询，最后释放并取回结果
async def test_result_timeout_and_cancel_preserve_child() -> None:
    gate = asyncio.Event()
    context = ExecutionContext("child", "work", 5)

    # 等待测试屏障后提交可被后续查询获取的结果
    async def finish():
        await gate.wait()
        context.result = "done"
        context.mark_success()

    task = asyncio.create_task(finish())
    registry = BackgroundTaskRegistry()
    registry.register("child", task, context)
    tool = AgentResultTool(registry)
    try:
        result = await tool.invoke({"run_id": "child", "wait_ms": 1})
        assert result.content == "still running" and not task.cancelled()
        waiter = asyncio.create_task(tool.invoke({"run_id": "child"}))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not task.cancelled()
    finally:
        gate.set()
        await task


# 功能：验证一次性会话压缩后提交结果不会重复追加已持久化的最新请求
# 设计：真实Runner和存储触发自动压缩，外部模型直接结束，检查完整历史的请求次数
async def test_one_shot_compaction_does_not_duplicate_preserved_request(tmp_path, monkeypatch) -> None:
    from x_claude.core.config import XConfig
    from x_claude.core.runner import AgentRunner
    from x_claude.core.session.model import Session
    from x_claude.core.session.store import SessionStore

    goal = "Write the exact pending artifact."
    config = XConfig()
    config.llm.context_window = 1000
    store = SessionStore(tmp_path / "sessions")
    session = Session("s", "one_shot", "active", "test", "", "")
    store.write_meta(session)
    store.append_messages("s", [{"role": "user", "content": "older history " * 500},
                               {"role": "assistant", "content": "done"},
                               {"role": "user", "content": goal}], "history")

    class Provider:
        # 区分摘要和正常推理，只让真实运行时负责持久化消息
        async def chat(self, run_id, **kwargs):
            return LlmResponse("end_turn", text="Prior constraints." if run_id == "compact" else "completed")

    monkeypatch.setattr("x_claude.core.runner.load_context_file", lambda path: "")
    outcome = await AgentRunner(config, provider=Provider()).run_and_capture(
        goal, run_id="root", session=session, store=store)
    assert outcome.status == "success"
    messages = store.read_messages("s", truncate=False)
    assert sum(m.get("content") == goal for m in messages) == 1


# 功能：验证手动compact同样逐字保留最新请求，不能承诺保留却仅提交摘要
# 设计：用真实SessionManager和磁盘存储，模型摘要故意遗漏新的文件名和字段约束
async def test_manual_compaction_preserves_latest_request(tmp_path) -> None:
    from x_claude.core.config import XConfig
    from x_claude.core.runner import AgentRunner
    from x_claude.core.session.manager import SessionManager
    from x_claude.core.session.store import SessionStore

    class Provider:
        # 只摘要旧目标，使原请求是否被确定性保留可直接观察
        async def chat(self, **kwargs):
            return LlmResponse("end_turn", text="Older packaging constraints: port 8310.")

    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: AgentRunner(XConfig()), EventBus(), provider=Provider())
    session = await manager.create("chat")
    goal = "Write pending.json with exact integer port=8310."
    store.append_messages(session.id, [{"role": "user", "content": "older history " * 300},
                                      {"role": "assistant", "content": "done"},
                                      {"role": "user", "content": goal}], "history")
    result = await manager.compact(session.id)
    assert store.read_messages(session.id, truncate=False)[-1] == {"role": "user", "content": goal}
    assert result.saved_tokens < len("older history " * 300) // 4


# 功能：压缩发生在工具观察之后时，保留用户文本块而不把工具结果当成新目标
# 设计：与真实消息角色一致地交错工具调用和结果，直接验证共享提交格式
def test_compacted_messages_skip_tool_only_user_message() -> None:
    from x_claude.core.compact.compactor import build_compacted_messages

    request = {"role": "user", "content": [{"type": "text", "text": "Write exact.json."}]}
    messages = [request, {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t", "name": "read_file", "input": {"path": "a"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "a"}]}]
    assert build_compacted_messages(messages, "summary")[-1] == request
