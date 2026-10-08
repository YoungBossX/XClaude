from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from tests.unit.test_llm_provider import _make_provider
from x_claude.core.config import XConfig
from x_claude.core.context import ExecutionContext
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.provider import AnthropicProvider
from x_claude.core.llm.types import LlmResponse, ToolCallBlock, UsageStats
from x_claude.core.loop import AgentLoop
from x_claude.core.runner import AgentRunner
from x_claude.core.session.model import Session
from x_claude.core.session.store import SessionStore
from x_claude.core.tools.base import BaseTool, ToolResult
from x_claude.core.tools.invocation import invoke_tool
from x_claude.core.tools.registry import ToolRegistry


class _CaptureProvider:
    # 捕获独立消息快照，返回确定的最终回复而不访问网络
    async def chat(self, messages: list, **kwargs: object) -> LlmResponse:
        self.messages = copy.deepcopy(messages)
        return LlmResponse("end_turn", text="done")


# 功能：验证并发运行的事件文件相互隔离，重复运行不向 daemon 总线积累订阅
# 设计：两次模型调用使用屏障强制重叠，读取真实 JSONL；同时检查全局订阅数量
async def test_concurrent_runs_isolate_events_and_release_subscriptions(tmp_path: Path) -> None:
    bus = EventBus()
    entered = 0
    ready = asyncio.Event()

    # 等待两个运行都已经打开自己的事件文件，再同时结束
    async def chat(**kwargs: object) -> LlmResponse:
        nonlocal entered
        entered += 1
        if entered == 2:
            ready.set()
        await ready.wait()
        return LlmResponse("end_turn", text="done")

    provider = _CaptureProvider()
    with patch.object(provider, "chat", side_effect=chat):
        runners = [AgentRunner(XConfig(), bus=bus, provider=provider, runs_dir=tmp_path)
                   for _ in range(2)]
        await asyncio.gather(*(r.run_and_capture("goal", run_id=f"r{i}")
                               for i, r in enumerate(runners)))
    for i in range(2):
        events = [json.loads(line) for line in (tmp_path / f"r{i}" / "events.jsonl")
                  .read_text(encoding="utf-8").splitlines()]
        assert {event["run_id"] for event in events} == {f"r{i}"}
    assert not bus._subscribers


# 功能：验证普通工具错误不会重试，即使调用已产生副作用
# 设计：计数型工具模拟先完成副作用再返回错误，断言只执行一次且明确报告失败
async def test_side_effect_tool_is_not_retried() -> None:
    class SideEffectTool(BaseTool):
        name = "side_effect"
        description = "test"
        input_schema: dict[str, object] = {"type": "object"}
        calls = 0

        # 模拟已完成不可重复动作后返回运行错误
        async def invoke(self, params: dict[str, object]) -> ToolResult:
            self.calls += 1
            return ToolResult("failed after side effect", True, "runtime_error")

    tool = SideEffectTool()
    registry = ToolRegistry()
    registry.register(tool)
    result = await invoke_tool(registry, ToolCallBlock("t", tool.name, {}), EventBus(), "r")
    assert tool.calls == 1
    assert result.is_error


# 功能：验证缓存读写均纳入水位，且兼容模型可以覆盖窗口容量
# 设计：注入固定 usage 字段，确认 160000 输入 Token / 200000 窗口为 80%
async def test_context_budget_includes_both_cache_fields() -> None:
    provider, client = _make_provider(input_tokens=3000, cache_read=150000)
    client.messages.stream.return_value._final.usage.cache_creation_input_tokens = 7000
    response = await provider.chat([], [], EventBus(), "r")
    assert response.usage is not None
    assert response.usage.context_pct == 0.8
    assert response.usage.context_window == 200000


# 功能：验证自定义窗口实际参与 Provider 的水位计算，而不是仅能读到配置
# 设计：用相同的缓存 usage 和 64000 容量，检查响应和通知中的水位与容量
async def test_provider_uses_configured_context_window() -> None:
    _, client = _make_provider(input_tokens=1000, cache_read=30000)
    client.messages.stream.return_value._final.usage.cache_creation_input_tokens = 1000
    provider = AnthropicProvider(model="compatible-model", client=client, context_window=64000)
    events: list = []
    bus = EventBus()

    # 捕获 Provider 发布的 usage 通知以验证 TUI 所见值一致
    async def record(event: BaseModel) -> None:
        events.append(event.model_dump())

    bus.subscribe(record)
    response = await provider.chat([], [], bus, "r")
    assert response.usage is not None
    assert response.usage.context_pct == 0.5
    assert response.usage.context_window == 64000
    usage = next(event for event in events if event["type"] == "llm.usage")
    assert usage["context_pct"] == 0.5 and usage["context_window"] == 64000


# 功能：验证本轮超长工具结果按配置截断，但完整原始内容仍保留在历史中
# 设计：同一运行内连续两次模型调用，对比第二次请求与内存原始消息，检查压缩识别未被误触发
async def test_live_tool_result_budget_preserves_raw_history() -> None:
    class LargeTool(BaseTool):
        name = "large"
        description = "test"
        input_schema: dict[str, object] = {"type": "object"}

        # 返回固定的大文本，避免依赖真实文件和外部命令
        async def invoke(self, params: dict[str, object]) -> ToolResult:
            return ToolResult("x" * 20000)

    captured: list = []

    # 第一轮调用工具，第二轮保存实际传给模型的结果
    async def chat(messages: list, step: int, **kwargs: object) -> LlmResponse:
        if step == 1:
            return LlmResponse("tool_use", tool_calls=[ToolCallBlock("t", "large", {})])
        captured.extend(copy.deepcopy(messages))
        return LlmResponse("end_turn", text="done")

    registry = ToolRegistry()
    registry.register(LargeTool())
    provider = _CaptureProvider()
    context = ExecutionContext("r", "goal", 3)
    original = context.messages
    with patch.object(provider, "chat", side_effect=chat):
        await AgentLoop(provider, registry, EventBus(), tool_result_limit=1000,
                        tool_result_keep=100).run(context)
    bounded = captured[-1]["content"][0]["content"]
    assert bounded.startswith("x" * 100 + "\n[...")
    assert len(bounded) < 200
    assert len(context.messages[-2]["content"][0]["content"]) == 20000
    assert context.messages is original


# 功能：验证新增工具结果越过阈值后，在下一次模型调用前触发压缩
# 设计：上次输入只有 70%，新增结果把预算推过 80%；用压缩器替身记录准确调用点
async def test_compaction_accounts_for_new_tool_results() -> None:
    from unittest.mock import AsyncMock

    provider = _CaptureProvider()
    calls = 0

    # 返回低于阈值的输入水位，但同时产生较大的未知工具参数
    async def chat(**kwargs: object) -> LlmResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return LlmResponse("tool_use", tool_calls=[ToolCallBlock("t", "unknown", {})],
                               usage=UsageStats(700, 150, context_pct=0.7, context_window=1000))
        return LlmResponse("end_turn", text="done")

    compactor = AsyncMock()
    with patch.object(provider, "chat", side_effect=chat):
        await AgentLoop(provider, ToolRegistry(), EventBus(), compactor=compactor).run(
            ExecutionContext("r", "goal", 3)
        )
    compactor.compact.assert_awaited_once()


# 功能：验证磁盘写入失败时不会通知运行成功，也不会覆盖原有历史
# 设计：真实历史配合 append_messages 故障注入，同时检查返回状态、终止事件和文件字节
async def test_persistence_failure_is_a_failed_run(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = Session("s", "chat", "active", "", "t", "t")
    store.append_message(session.id, "user", "goal")
    path = store.session_dir(session.id) / "thread.jsonl"
    before = path.read_bytes()
    events: list = []

    # 收集通知，以验证落盘失败时没有先发出 success
    async def record(event: BaseModel) -> None:
        events.append(event.model_dump())

    runner = AgentRunner(XConfig(), provider=_CaptureProvider(), extra_handlers=[record])
    with patch.object(store, "append_messages", side_effect=OSError("disk full")):
        result = await runner.run_and_capture("goal", session=session, store=store)
    assert result.status == "failed"
    assert result.reason == "persistence_error"
    assert path.read_bytes() == before
    finished = [event for event in events if event["type"] == "run.finished"]
    assert len(finished) == 1
    assert finished[0]["status"] == "failed"


# 功能：验证收到成功通知时，下一轮已经可以读取完整的模型回答
# 设计：在 run.finished 回调中立即重读真实 thread，锁定保存与通知的先后顺序
async def test_history_is_durable_before_success_notification(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = Session("s", "chat", "active", "", "t", "t")
    store.append_message(session.id, "user", "goal")

    # 在通知边界验证落盘结果，而不是等整个函数返回后才检查
    async def record(event: BaseModel) -> None:
        if getattr(event, "type", "") == "run.finished":
            assert store.read_messages(session.id)[-1]["content"][0]["text"] == "done"
            assert store.read_meta(session.id).status == "waiting_for_input"

    runner = AgentRunner(XConfig(), provider=_CaptureProvider(), extra_handlers=[record])
    result = await runner.run_and_capture("goal", session=session, store=store)
    assert result.status == "success"
    assert result.session_persisted


# 功能：验证 daemon 的通知观察者失败不会中断执行或混淆持久化状态
# 设计：让全局总线回调每次抛错，检查真实 Runner 仍完成并写入自己的事件文件
async def test_global_observer_failure_does_not_fail_run(tmp_path: Path) -> None:
    bus = EventBus()

    # 模拟一个损坏的外部通知订阅，不能反向改变模型执行结果
    async def broken(event: BaseModel) -> None:
        raise RuntimeError("observer failed")

    bus.subscribe(broken)
    runner = AgentRunner(XConfig(), bus=bus, provider=_CaptureProvider(), runs_dir=tmp_path)
    result = await runner.run_and_capture("goal", run_id="isolated")
    assert result.status == "success"
    events = [json.loads(line) for line in (tmp_path / "isolated" / "events.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "run.finished" and events[-1]["status"] == "success"


# 功能：验证元数据保存失败同样导致运行失败，不会发出绿色成功通知
# 设计：在线程消息成功落盘后注入 meta 写入故障，覆盖多个持久化文件之间的失败边界
async def test_metadata_failure_prevents_success_notification(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = Session("s", "chat", "active", "", "t", "t")
    store.write_meta(session)
    store.append_message(session.id, "user", "goal")
    events: list = []

    # 保存终止事件，并立即确认已写入的消息仍能恢复
    async def record(event: BaseModel) -> None:
        events.append(event.model_dump())

    runner = AgentRunner(XConfig(), provider=_CaptureProvider(), extra_handlers=[record])
    with patch.object(store, "write_meta", side_effect=OSError("metadata write failed")):
        result = await runner.run_and_capture("goal", session=session, store=store)
    assert result.status == "failed" and result.reason == "persistence_error"
    assert store.read_messages(session.id)[-1]["content"][0]["text"] == "done"
    assert store.read_meta(session.id).status == "active"
    assert events[-1]["status"] == "failed"


# 功能：验证兼容模型窗口配置可以来自 TOML，并由环境变量覆盖；非法容量明确报错
# 设计：使用独立配置路径和临时工作目录，不依赖用户真实 .env 和全局配置
@pytest.mark.parametrize("value", ["64000", "0", "-1", "invalid"])
def test_context_window_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    from x_claude.core.config import get_config

    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text("[llm]\ncontext_window=32000\n", encoding="utf-8")
    monkeypatch.setenv("X_CONFIG", str(config))
    monkeypatch.delenv("X_CONTEXT_WINDOW", raising=False)
    assert get_config().llm.context_window == 32000
    monkeypatch.setenv("X_CONTEXT_WINDOW", value)
    if value == "64000":
        assert get_config().llm.context_window == 64000
    else:
        with pytest.raises(SystemExit, match="X_CONTEXT_WINDOW"):
            get_config()


# 功能：验证续接的历史在第一次模型推理之前就可以触发预算保护
# 设计：用明确的小窗口和长历史强制触发预检查，确认先摘要、后正常推理，且原始历史已备份
async def test_resumed_history_budget_is_checked_before_first_inference(tmp_path: Path) -> None:
    order: list[str] = []

    # 为摘要和普通推理分别返回确定文本，以便检查执行顺序和摘要注入
    async def chat(messages: list, run_id: str, **kwargs: object) -> LlmResponse:
        order.append(run_id)
        if run_id == "compact":
            return LlmResponse("end_turn", text="short handoff")
        assert messages[0]["content"] == "short handoff"
        return LlmResponse("end_turn", text="done")

    config = XConfig()
    config.llm.context_window = 10000
    store = SessionStore(tmp_path)
    session = Session("s", "chat", "active", "", "t", "t")
    store.append_message(session.id, "user", "x" * 10000)
    provider = _CaptureProvider()
    with patch.object(provider, "chat", side_effect=chat):
        result = await AgentRunner(config, provider=provider).run_and_capture(
            "goal", run_id="resumed", session=session, store=store,
        )
    assert result.status == "success"
    assert order == ["compact", "resumed"]
    assert store.read_messages(session.id)[0]["content"] == "short handoff"
    assert len(list(store.session_dir(session.id).glob("thread_*.bak"))) == 1
