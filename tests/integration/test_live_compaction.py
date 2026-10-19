from __future__ import annotations

import json

import pytest

from x_claude.core.compact.budget import estimate_tokens
from x_claude.core.compact.compactor import Compactor
from x_claude.core.context import ExecutionContext
from x_claude.core.events.bus import EventBus
from x_claude.core.runner import AgentRunner
from x_claude.core.session.manager import SessionManager
from x_claude.core.session.store import SessionStore

pytestmark = pytest.mark.integration

_FACTS = (
    "Binding project facts: magic number = 7391; target file = 资料/验收 结果.json; "
    "constraint = do not overwrite original files. These facts remain binding."
)
_REQUEST = (
    "Pending operation code is ZX-2048. Keep that exact code. "
    "Now return ONLY a JSON object with exactly these fields from the agreed facts: "
    "magic (integer), file (string), overwrite_originals (boolean). Do not use tools."
)


class ConversationRunner(AgentRunner):
    # 只限制测试工具范围，模型、循环、压缩、持久化仍使用真实实现。
    async def run_and_capture(self, goal, **kwargs):
        kwargs["tool_whitelist"] = []
        return await super().run_and_capture(goal, **kwargs)


# 功能：真实模型自动/手动压缩后，跨三轮回答保留旧事实、最新要求和精确路径。
# 设计：固定合成历史中加入冗余，最新请求不重述旧事实；同时断言磁盘、实际请求与最终回答。
@pytest.mark.parametrize("mode", ["auto", "manual"])
async def test_live_compaction_preserves_followups(live_api, tmp_path, mode):
    config, provider = live_api
    # 只降低触发阈值控制成本，不伪造模型窗口或 API usage。
    config.compaction.auto_threshold = 0.005 if mode == "auto" else 0
    bus, events = EventBus(), []

    # 收集真实会话事件供压缩、执行结束与错误检查。
    async def collect(event):
        events.append(event.model_dump())

    bus.subscribe(collect)
    store = SessionStore(tmp_path / "sessions")
    manager = SessionManager(store, lambda: ConversationRunner(
        config, bus=bus, provider=provider,
    ), bus, provider=provider, config=config)
    session = await manager.create("chat")
    history = [
        {"role": "user", "content": _FACTS},
        {"role": "assistant", "content": "Acknowledged. " + (
            "Review completed; no files changed, no new facts or decisions. " * 160
        )},
    ]
    store.append_messages(session.id, history, "seed")
    old_size = estimate_tokens(history)
    try:
        if mode == "manual":
            store.append_message(session.id, "user", _REQUEST)
            result = await manager.compact(session.id)
            assert result.saved_tokens > 0
            compacted = store.read_messages(session.id, truncate=False)
            assert compacted[-1] == {"role": "user", "content": _REQUEST}
            await manager.send_message(session.id, "Answer the preceding request now, without tools.")
        else:
            await manager.send_message(session.id, _REQUEST)
            assert any(e["type"] == "context.compacted" for e in events)
        assert provider.calls[0]["run_id"] == "compact"
        main_calls = [c for c in provider.calls if c["run_id"] != "compact"]
        assert any(m.get("content") == _REQUEST for m in main_calls[0]["messages"])
        answer = main_calls[-1]["response"].text
        decoded = json.loads(answer.strip().removeprefix("```json").removesuffix("```").strip())
        assert decoded == {"magic": 7391, "file": "资料/验收 结果.json", "overwrite_originals": False}
        assert type(decoded["magic"]) is int and type(decoded["overwrite_originals"]) is bool
        persisted = store.read_messages(session.id, truncate=False)
        assert estimate_tokens(persisted) < old_size
        assert sum(m.get("content") == _REQUEST for m in persisted) == 1
        # 后续问答只验证压缩产物的可用性，避免同一合成历史重复付费摘要。
        config.compaction.auto_threshold = 0
        for question, expected in [
            ("Return only the exact target file path agreed earlier.", "资料/验收 结果.json"),
            ("Return only the pending operation code agreed earlier.", "ZX-2048"),
        ]:
            await manager.send_message(session.id, question)
            assert expected in provider.calls[-1]["response"].text
        finished = [e for e in events if e["type"] == "run.finished"]
        assert len(finished) == 3
        assert all(e["status"] == "success" for e in finished)
        assert not any(e["type"] in ("llm.error", "tool.call_failed") for e in events)
        backups = list(store.session_dir(session.id).glob("thread_*.jsonl.bak"))
        assert backups and _FACTS in backups[0].read_text(encoding="utf-8")
    finally:
        await manager.shutdown()


# 功能：真实模型生成摘要但完整新上下文更长时拒绝压缩，连续尝试不重复付费。
# 设计：唯一长用户请求必须原文保留，故任何非空摘要都无收益；验证内存、磁盘、通知与调用数。
async def test_live_compaction_no_gain_backoff(live_api, tmp_path):
    _, provider = live_api
    store = SessionStore(tmp_path / "sessions")
    context = ExecutionContext("no-gain", "Keep this request verbatim: " + "abc123 " * 500, 5)
    store.append_messages("s", context.messages, "no-gain")
    path = store.session_dir("s") / "thread.jsonl"
    original, original_bytes = context.messages, path.read_bytes()
    events, bus = [], EventBus()

    # 记录所有完成事件，拒绝压缩不得误发成功通知。
    async def collect(event):
        events.append(event)

    bus.subscribe(collect)
    compactor = Compactor(bus, path.parent, "s", store=store)
    for _ in range(3):
        assert await compactor.compact(context, provider) is None
    assert len(provider.calls) == 1
    assert provider.calls[0]["response"].text.strip()
    assert provider.calls[0]["response"].stop_reason == "end_turn"
    assert context.messages is original
    assert path.read_bytes() == original_bytes
    assert not events
    assert not list(path.parent.glob("summary_*.md"))
