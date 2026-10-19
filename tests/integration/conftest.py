from __future__ import annotations

import os
from urllib.parse import urlparse

import anthropic
import pytest

from tests.live_api_support import LiveProvider, load_live_config
from x_claude.core.llm.provider import AnthropicProvider


# 只在已显式开启的真实测试中读配置并创建 SDK，所有上下文与产物留在独立临时目录。
@pytest.fixture
async def live_api(request, tmp_path, monkeypatch, record_property):
    if not request.config.getoption("--run-live-api"):
        pytest.skip("real API disabled")
    config = load_live_config()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.fail("--run-live-api requires ANTHROPIC_API_KEY", pytrace=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("x_claude.core.runner.load_context_file", lambda _: "")
    async with anthropic.AsyncAnthropic(timeout=30, max_retries=0) as client:
        provider = LiveProvider(AnthropicProvider(
            config.llm.default_model, client=client, context_window=config.llm.context_window,
        ))
        record_property("model", config.llm.default_model)
        record_property("api_host", urlparse(str(client.base_url)).hostname)
        try:
            yield config, provider
        finally:
            record_property("model_calls", len(provider.calls))
            record_property("summary_calls", sum(c["run_id"] == "compact" for c in provider.calls))
            usage = [c["response"].usage for c in provider.calls if "response" in c]
            for name in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
                record_property(name, sum(getattr(u, name) for u in usage if u is not None))
