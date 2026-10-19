from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.live_api_support import assert_read_file_run, load_live_config


# 功能：真实测试必须沿用用户配置的模型而非固定 Claude 默认值。
# 设计：使用真实配置加载器与独立 TOML、环境变量，不构造 SDK、不读取密钥。
def test_live_config_uses_selected_model(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text("", encoding="utf-8")
    monkeypatch.setenv("X_CONFIG", str(path))
    monkeypatch.setenv("X_LLM_DEFAULT_MODEL", "fixture-compatible-model")
    monkeypatch.setenv("X_CONTEXT_WINDOW", "64000")
    monkeypatch.chdir(tmp_path)
    config = load_live_config()
    assert config.llm.default_model == "fixture-compatible-model"
    assert config.llm.context_window == 64000


# 生成独立手写的成功事件样本，不借用被测验收函数计算期望值。
def _valid_events():
    return [
        {"type": "run.started", "run_id": "r"},
        {"type": "step.started", "run_id": "r"},
        {"type": "tool.call_started", "run_id": "r", "tool_use_id": "t", "tool_name": "read_file"},
        {"type": "tool.call_finished", "run_id": "r", "tool_use_id": "t", "tool_name": "read_file", "output": "The number is 7391."},
        {"type": "llm.text_completed", "run_id": "r", "text": "The answer is 7391."},
        {"type": "llm.usage", "run_id": "r", "input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 0},
        {"type": "run.finished", "run_id": "r", "status": "success"},
    ]


# 功能：错误答案、未完成工具、错配工具 ID 或混入其他 run 不得被判为真实成功。
# 设计：从手写有效事件中各破坏一个验收条件，验证断言能拒绝假阳性。
@pytest.mark.parametrize("defect", ["answer", "tool", "tool_id", "run", "usage"])
def test_live_assertions_reject_false_success(defect):
    events = copy.deepcopy(_valid_events())
    if defect == "answer":
        events[4]["text"] = "The answer is 123."
    elif defect == "tool":
        events.pop(3)
    elif defect == "tool_id":
        events[3]["tool_use_id"] = "different"
    elif defect == "run":
        events[4]["run_id"] = "other"
    else:
        events[5].pop("input_tokens")
    with pytest.raises(AssertionError):
        assert_read_file_run(events, "7391")


# 功能：完整事件和正确答案应通过验收，避免负向检查一律拒绝。
# 设计：使用相同独立手写事件作为正向对照。
def test_live_assertions_accept_valid_run():
    assert_read_file_run(_valid_events(), "7391")


# 功能：未显式启用时真实 API 标记用例绝不执行，启用后不会被错误跳过。
# 设计：在新 pytest 进程使用会立即失败的本地哨兵用例，不连接网络、不读取真实密钥。
@pytest.mark.parametrize("enabled", [False, True])
def test_live_api_requires_opt_in(tmp_path, enabled):
    sample = tmp_path / "test_sentinel.py"
    sample.write_text(
        "import pytest\npytestmark = pytest.mark.integration\n"
        "def test_sentinel():\n    raise AssertionError('LIVE_SENTINEL_EXECUTED')\n",
        encoding="utf-8",
    )
    command = [sys.executable, "-m", "pytest", "-p", "tests.conftest", str(sample), "-q"]
    if enabled:
        command.append("--run-live-api")
    env = {key: value for key, value in os.environ.items()
           if key not in {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"}}
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], env=env,
                            capture_output=True, timeout=20)
    output = result.stdout.decode("utf-8", errors="replace")
    assert result.returncode == (1 if enabled else 0), result.stderr
    assert ("LIVE_SENTINEL_EXECUTED" in output) == enabled
    if not enabled:
        assert "1 skipped" in output
