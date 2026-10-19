from __future__ import annotations

import json

import pytest

from tests.live_api_support import assert_read_file_run
from x_claude.core.runner import AgentRunner

pytestmark = pytest.mark.integration


# 功能：通过当前配置的真实模型读取文件，返回正确数字并写出完整事件。
# 设计：仅开放 read_file，在临时项目中读取未出现在目标里的数字，并检查实际结果与最终回答。
async def test_run_e2e_reads_file_and_succeeds(live_api, tmp_path) -> None:
    config, provider = live_api
    (tmp_path / "sample.txt").write_text(
        "# Test Document\n\nThe magic number mentioned in this file is 7391.\n",
        encoding="utf-8",
    )
    runner = AgentRunner(config, provider=provider, runs_dir=tmp_path / "runs")
    try:
        outcome = await runner.run_and_capture(
            "Use read_file to read sample.txt and report the magic number it mentions.",
            tool_whitelist=["read_file"],
        )
        assert outcome.status == "success"
        assert "7391" in outcome.result
        paths = list((tmp_path / "runs").rglob("events.jsonl"))
        assert len(paths) == 1
        events = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]
        assert_read_file_run(events, "7391")
    finally:
        await runner.shutdown()
