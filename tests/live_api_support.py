from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from x_claude.core.config import XConfig, get_config
from x_claude.core.llm.provider import AnthropicProvider


# 在切换临时工作目录前读取真实配置，只收紧测试预算，不覆盖用户选择的模型与窗口。
def load_live_config() -> XConfig:
    config = get_config()
    config.agent.max_steps = 5
    config.agent.max_total_tokens = 50000
    config.agent.max_runtime_s = 90.0
    return config


# 同时核对真实文件结果、最终答案、用量与事件归属，拒绝只发起工具就算成功。
def assert_read_file_run(events: list[dict], expected: str) -> None:
    assert events[0]["type"] == "run.started"
    assert events[-1]["type"] == "run.finished"
    assert events[-1]["status"] == "success"
    assert all(e["run_id"] == events[0]["run_id"] for e in events)
    assert any(e["type"] == "step.started" for e in events)
    starts = {e["tool_use_id"] for e in events
              if e["type"] == "tool.call_started" and e["tool_name"] == "read_file"}
    assert any(e["type"] == "tool.call_finished" and e["tool_name"] == "read_file"
               and e["tool_use_id"] in starts and expected in e["output"] for e in events)
    answers = [e["text"] for e in events if e["type"] == "llm.text_completed"]
    assert answers and expected in answers[-1]
    usage = [e for e in events if e["type"] == "llm.usage"]
    assert usage
    for row in usage:
        assert all(type(row.get(key)) is int and row[key] >= 0 for key in (
            "input_tokens", "output_tokens", "cache_read_input_tokens",
        ))


@dataclass
class LiveProvider:
    inner: AnthropicProvider
    calls: list[dict[str, Any]] = field(default_factory=list)
    max_calls: int = 12

    # 真实请求逐次限额和超时，记录合成测试输入与实际响应，绝不替换模型答案。
    async def chat(self, **kwargs: Any):
        if len(self.calls) >= self.max_calls:
            raise RuntimeError("live API test call limit reached")
        row = {"run_id": kwargs["run_id"], "messages": kwargs["messages"]}
        self.calls.append(row)
        response = await asyncio.wait_for(self.inner.chat(**kwargs), timeout=45)
        row["response"] = response
        return response
