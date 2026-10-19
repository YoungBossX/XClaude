from __future__ import annotations

import json

import pytest


# 构造完整事件轨迹，各断言期望来自独立的字面量，不调用验收器生成预期。
def workflow_fixture(extra=False):
    expected = {"part-0": {"count": 2, "total": 3, "maximum": 2},
                "part-1": {"count": 2, "total": 7, "maximum": 4},
                "part-2": {"count": 2, "total": 11, "maximum": 6}}
    events = []
    states = []
    for index in range(4 if extra else 3):
        child, part = f"c{index}", f"part-{index % 3}"
        events.extend([
            {"type": "tool.call_started", "run_id": "root", "tool_use_id": f"s{index}",
             "tool_name": "spawn_agent", "params": {"run_in_background": True}, "ts": "01"},
            {"type": "subagent.started", "run_id": child, "parent_run_id": "root", "ts": "02"},
            {"type": "tool.call_started", "run_id": child, "tool_use_id": "calc", "tool_name": "bash",
             "params": {"command": f"python calculate.py {part}.json"}, "ts": "03"},
            {"type": "tool.call_finished", "run_id": child, "tool_use_id": "calc", "tool_name": "bash",
             "output": json.dumps(expected[part]), "ts": "04"},
            {"type": "subagent.finished", "run_id": child, "status": "success", "ts": "05"},
            {"type": "tool.call_started", "run_id": "root", "tool_use_id": f"q{index}",
             "tool_name": "agent_result", "params": {"run_id": child}, "ts": "06"},
            {"type": "tool.call_finished", "run_id": "root", "tool_use_id": f"q{index}",
             "tool_name": "agent_result", "output": json.dumps(expected[part]), "ts": "07"},
        ])
        states.append({"run_id": child, "done": True, "cancelled": False, "status": "success"})
    events.extend([
        {"type": "tool.call_started", "run_id": "root", "tool_name": "write_file", "tool_use_id": "answer",
         "params": {"path": "answer.json"}, "ts": "08"},
        {"type": "tool.call_finished", "run_id": "root", "tool_name": "write_file", "tool_use_id": "answer", "ts": "09"},
        {"type": "run.finished", "run_id": "root", "status": "success", "ts": "10"},
    ])
    return events, expected, states


# 功能：额外派生只报真实的数量缺陷，不谎报子任务未结束、非后台派生或提前写入。
# 设计：四个子任务均真实完成并按序回收，正确产物不能掩盖主协议不合规。
def test_extra_child_does_not_generate_false_timing_or_state_errors():
    from tests.fixed_workflow_support import assess_workflow

    events, expected, states = workflow_fixture(extra=True)
    result = assess_workflow(events, expected, expected, states)
    assert result["artifact_valid"]
    assert not result["passed"]
    assert set(result["errors"]) == {"child_count", "spawn_attempt_count"}


# 功能：分别检出数量、真实计算、产物类型、待审批通知、提前回收和提前写入问题。
# 设计：每次只破坏一个证据边界，以独立错误键验证，不根据错误数量推断任务数量。
@pytest.mark.parametrize("defect", [None, "artifact", "approval_notice", "early_query", "early_write", "not_done", "no_calculation"])
def test_strict_workflow_checks_independent_boundaries(defect):
    from tests.fixed_workflow_support import assess_workflow

    events, expected, states = workflow_fixture()
    actual = json.loads(json.dumps(expected))
    if defect == "artifact":
        actual["part-0"]["count"] = 2.0
    if defect == "not_done":
        states[0]["done"] = False
    for event in events:
        if event.get("tool_use_id") == "q0" and event["type"] == "tool.call_finished":
            if defect == "approval_notice":
                event["output"] = "waiting_for_approval: wait"
            if defect == "early_query":
                event["ts"] = "04"
        if defect == "early_write" and event.get("tool_use_id") == "answer":
            event["ts"] = "02"
        if defect == "no_calculation" and event.get("run_id") == "c0" and event.get("tool_name") == "bash":
            event["tool_name"] = "read_file"
    result = assess_workflow(events, expected, actual, states)
    assert result["passed"] is (defect is None)
    if defect is not None:
        error = {"artifact": "artifact", "approval_notice": "final_results", "early_query": "final_results",
                 "early_write": "write_order", "not_done": "children_terminal", "no_calculation": "calculations"}[defect]
        assert error in result["errors"]


# 功能：只有回收到对应子任务真实算出的统计，才算拿到最终结果。
# 设计：空结果、占位符和其他分片结果均拒绝；允许正常回复附带文字或代码块。
@pytest.mark.parametrize("reply,valid", [
    ("", False), ("Subagent completed with no text result.", False),
    ('{"count": 2, "total": 7, "maximum": 4}', False),
    ('{"count": 2.0, "total": 3, "maximum": 2}', False),
    ('Result:\n```json\n{"count": 2, "total": 3, "maximum": 2}\n```', True),
])
def test_final_result_content_matches_child_calculation(reply, valid):
    from tests.fixed_workflow_support import assess_workflow

    events, expected, states = workflow_fixture()
    for event in events:
        if event["type"] == "tool.call_finished" and event.get("tool_use_id") == "q0":
            event["output"] = reply
    result = assess_workflow(events, expected, expected, states)
    assert result["checks"]["final_results"] is valid
    assert result["passed"] is valid
