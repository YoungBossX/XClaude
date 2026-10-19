from __future__ import annotations

import pytest

from x_claude.core.config import XConfig
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.types import LlmResponse, ToolCallBlock
from x_claude.core.permissions.manager import PermissionManager
from x_claude.core.permissions.policy import PermissionDecision, ToolPolicy
from x_claude.core.runner import AgentRunner


# 功能：固定流程中原子任务纠正错误命令且第四次派生被拒；默认模式仍能动态派生四个子任务。
# 设计：真实 Runner、预算、审批、Shell 和子循环，仅替换模型以稳定复现同一子任务内的修正。
@pytest.mark.parametrize("fixed", [True, False])
async def test_fixed_recovery_and_default_dynamic_spawning(tmp_path, monkeypatch, fixed):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("x_claude.core.runner.load_context_file", lambda _: "")
    cfg = XConfig()
    cfg.agent.max_steps = 6
    cfg.compaction.auto_threshold = 0
    if fixed:
        cfg.agent.max_tasks = 4
        cfg.agent.allowed_shell_commands = ["printf recovered"]
    events, children, child_steps = [], [], {}
    bus = EventBus()

    # 记录真实派生身份，保证恢复发生于原有子任务而不是替代子任务。
    async def collect(event):
        data = event.model_dump()
        events.append(data)
        if data["type"] == "subagent.started":
            children.append(data["run_id"])

    bus.subscribe(collect)

    class Provider:
        # 父任务尝试四次派生；固定模式子任务先偏离命令协议，再在同一 run 内纠正一次。
        async def chat(self, messages, run_id, step, **kwargs):
            if run_id == "root":
                if step == 1:
                    return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                        f"spawn-{i}", "spawn_agent",
                        {"description": f"child-{i}", "prompt": f"child-{i}", "run_in_background": True},
                    ) for i in range(4)])
                if step == 2:
                    return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                        f"query-{i}", "agent_result", {"run_id": child, "wait_ms": 30000},
                    ) for i, child in enumerate(children)])
                return LlmResponse("end_turn", text=" | ".join(b["content"] for b in messages[-1]["content"]))
            child_steps.setdefault(run_id, []).append(step)
            if step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "first", "bash", {"command": "cd . && printf recovered" if fixed else "printf dynamic"},
                )])
            last = messages[-1]["content"][0]
            if fixed and step == 2:
                assert last["is_error"] and "command_contract" in last["content"]
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "corrected", "bash", {"command": "printf recovered"},
                )])
            assert not last.get("is_error")
            return LlmResponse("end_turn", text=last["content"])

    pm = PermissionManager(policies={name: ToolPolicy(PermissionDecision.ALLOW)
                                    for name in ("bash", "spawn_agent", "agent_result")}, timeout_s=.02)
    runner = AgentRunner(cfg, provider=Provider(), bus=bus, permission_manager=pm,
                         runs_dir=tmp_path / "runs")
    try:
        outcome = await runner.run_and_capture("run children", run_id="root")
        assert outcome.status == "success", outcome.reason
        assert len(children) == (3 if fixed else 4)
        assert outcome.result == " | ".join(["recovered" if fixed else "dynamic"] * len(children))
        assert all(steps == ([1, 2, 3] if fixed else [1, 2]) for steps in child_steps.values())
        assert not any(e["type"] == "permission.requested" for e in events)
        spawn_errors = [e for e in events if e["type"] == "tool.call_failed" and e["tool_name"] == "spawn_agent"]
        assert len(spawn_errors) == (1 if fixed else 0)
        if fixed:
            assert "task_budget" in spawn_errors[0]["error_message"]
    finally:
        await runner.shutdown()
