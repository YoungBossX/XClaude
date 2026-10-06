from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from x_claude.core.compact.compactor import Compactor
from x_claude.core.context import ExecutionContext
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.types import LlmResponse, UsageStats
from x_claude.core.session.store import SessionStore


# 功能：验证摘要生成有超时上限且超时后不改变历史
# 设计：挂起受控 provider 并使用短超时，断言真实超时取消后文件与消息引用均保留
async def test_compaction_timeout_preserves_history(tmp_path: Path) -> None:
    provider = _stub_provider()

    # 挂起摘要调用直到超时取消，不调用外部服务
    async def hang(**kwargs: Any) -> LlmResponse:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    provider.chat = AsyncMock(side_effect=hang)
    context = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    original = context.messages
    compactor = Compactor(EventBus(), tmp_path, "sess-1", timeout_s=0.01)
    assert await compactor.compact(context, provider) is None
    assert context.messages is original
    assert not list(tmp_path.glob("summary_*.md"))


# 功能：验证摘要保存或历史提交失败时内存和磁盘仍使用原历史
# 设计：在两个文件提交边界分别注入错误，覆盖摘要非空却无法安全持久化的情况
@pytest.mark.parametrize("failure_target", ["summary", "thread"])
async def test_compaction_persistence_failure_preserves_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_target: str,
) -> None:
    import os

    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "original")
    path = store.session_dir("sess-1") / "thread.jsonl"
    original_bytes = path.read_bytes()
    context = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    original = context.messages
    compactor = Compactor(EventBus(), path.parent, "sess-1", store=store)
    real_replace = os.replace

    # 在摘要或会话文件的提交边界注入权限异常
    def fail_replace(source: object, target: object) -> None:
        destination = Path(str(target))
        if (failure_target == "summary" and destination.name.startswith("summary_")) or (
            failure_target == "thread" and destination == path
        ):
            raise PermissionError("disk commit failed")
        real_replace(source, target)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", fail_replace)
    assert await compactor.compact(context, _stub_provider()) is None
    assert context.messages is original
    assert path.read_bytes() == original_bytes


# 功能：验证完成通知的订阅者异常不阻止其它通知且内存磁盘状态一致
# 设计：第一个订阅者检查提交顺序后抛异常，第二个仍收到事件，覆盖提交成功后的通知故障
async def test_notification_failure_does_not_undo_committed_compaction(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "original")
    context = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    bus = EventBus()
    received: list[Any] = []
    snapshots: list[list[dict[str, Any]]] = []

    # 确认发布前内存和磁盘已经一致，然后模拟观察者异常
    async def broken_handler(event: Any) -> None:
        snapshots.append(store.read_messages("sess-1"))
        assert store.read_messages("sess-1") == context.messages
        raise RuntimeError("observer unavailable")

    # 收集后续订阅者的完成通知
    async def healthy_handler(event: Any) -> None:
        received.append(event)

    bus.subscribe(broken_handler)
    bus.subscribe(healthy_handler)
    compactor = Compactor(bus, store.session_dir("sess-1"), "sess-1", store=store)
    result = await compactor.compact(context, _stub_provider())
    assert result is not None
    assert len(received) == 1
    assert received[0].type == "context.compacted"
    assert store.read_messages("sess-1") == context.messages
    assert snapshots == [context.messages]


# 功能：验证空白或截断摘要不会替换原历史
# 设计：分别返回仅空白文本和非空 max_tokens 响应，避免把不完整的摘要当成成功结果
@pytest.mark.parametrize(("text", "stop_reason"), [("  \n", "end_turn"), ("partial", "max_tokens")])
async def test_incomplete_summary_preserves_original(
    tmp_path: Path, text: str, stop_reason: str,
) -> None:
    provider = _stub_provider()
    provider.chat = AsyncMock(return_value=LlmResponse(stop_reason=stop_reason, text=text))
    context = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    original = context.messages
    compactor = Compactor(EventBus(), tmp_path, "sess-1")
    assert await compactor.compact(context, provider) is None
    assert context.messages is original
    assert not list(tmp_path.glob("summary_*.md"))


# 功能：验证提交前取消保留旧状态，提交后取消保留一致的新状态并继续传播
# 设计：通过事件屏障在模型生成和完成通知阶段分别取消真实任务，避免依赖时间竞争
@pytest.mark.parametrize("cancel_after_commit", [False, True])
async def test_compaction_cancellation_keeps_consistent_state(
    tmp_path: Path, cancel_after_commit: bool,
) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "original")
    context = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    context.messages = store.read_messages("sess-1")
    original = context.messages
    entered = asyncio.Event()
    bus = EventBus()
    provider = _stub_provider()

    # 在指定取消阶段发出屏障并挂起等待取消
    async def wait_for_cancel(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await asyncio.Event().wait()

    if cancel_after_commit:
        bus.subscribe(wait_for_cancel)
    else:
        provider.chat = AsyncMock(side_effect=wait_for_cancel)
    compactor = Compactor(bus, store.session_dir("sess-1"), "sess-1", store=store)
    task = asyncio.create_task(compactor.compact(context, provider))
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.read_messages("sess-1") == context.messages
    assert (context.messages is not original) == cancel_after_commit


def _stub_provider(summary: str = "## 1. Original Goal\nTest\n## 2. Completed Steps\n- done") -> Any:
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=LlmResponse(
        stop_reason="end_turn",
        text=summary,
        usage=UsageStats(input_tokens=100, output_tokens=30),
    ))
    return provider


def _make_messages(n: int = 5) -> list[dict[str, Any]]:
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": "user message " + "x" * 200})
        msgs.append({"role": "assistant", "content": "assistant reply " + "y" * 200})
    return msgs


# 功能：验证 compact_messages 成功时 provider.chat 被调用一次且不传工具 schema
# 设计：stub provider 返回非空摘要，断言 chat 调用一次，tool_schemas=[]
async def test_compact_messages_calls_provider(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    messages = _make_messages()

    result = await compactor.compact_messages(messages, provider)

    assert result is not None
    provider.chat.assert_called_once()
    call_kwargs = provider.chat.call_args
    assert call_kwargs.kwargs.get("tool_schemas") == [] or call_kwargs.args[1] == []


# 功能：验证 compact_messages 返回的摘要文本来自 provider 响应
# 设计：stub provider 返回固定摘要字符串，断言 result.summary_text 等于该字符串
async def test_compact_messages_returns_summary(tmp_path: Path) -> None:
    expected = "## 1. Original Goal\nDo X\n## 2. Completed\n- step one"
    provider = _stub_provider(summary=expected)
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")

    result = await compactor.compact_messages(_make_messages(), provider)

    assert result is not None
    assert result.summary_text == expected


# 功能：验证 compact() 将 context.messages 替换为两条摘要消息对
# 设计：调用 compact() 后断言 messages 长度为 2，role 分别为 user/assistant
async def test_compact_replaces_context_messages(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    ctx.messages = _make_messages()

    await compactor.compact(ctx, provider)

    assert len(ctx.messages) == 2
    assert ctx.messages[0]["role"] == "user"
    assert ctx.messages[1]["role"] == "assistant"


# 功能：验证 compact() 在 session 目录写入 summary_*.md 文件
# 设计：使用 tmp_path，调用 compact() 后检查目录内是否存在 summary_ 开头的文件
async def test_compact_writes_summary_file(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    ctx.messages = _make_messages()

    await compactor.compact(ctx, provider)

    summary_files = list(tmp_path.glob("summary_*.md"))
    assert len(summary_files) == 1


# 功能：验证 compact() 成功后发布 ContextCompactedEvent 事件
# 设计：订阅 EventBus，收集事件，断言收到类型为 context.compacted 的事件
async def test_compact_publishes_event(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    received: list[Any] = []

    async def handler(event: Any) -> None:
        received.append(event)

    bus.subscribe(handler)
    compactor = Compactor(bus, tmp_path, "sess-1")
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    ctx.messages = _make_messages()

    await compactor.compact(ctx, provider)

    types = [getattr(e, "type", None) for e in received]
    assert "context.compacted" in types


# 功能：验证 provider 抛异常时 context.messages 保持不变
# 设计：stub provider.chat 抛 RuntimeError，断言 compact() 返回 None 且 messages 未被修改
async def test_compact_failure_preserves_context(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=RuntimeError("LLM error"))
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    original_messages = _make_messages()
    ctx.messages = list(original_messages)

    result = await compactor.compact(ctx, provider)

    assert result is None
    assert ctx.messages == original_messages
