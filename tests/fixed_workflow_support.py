from __future__ import annotations

import json


# 严格比较 JSON 类型和值，避免布尔或浮点数冒充整数统计结果。
def strict_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(strict_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(strict_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


# 从最终回复的正文或代码块提取对象，并与该子任务的真实计算结果严格比对。
def contains_result(text, expected):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if strict_equal(value, expected):
            return True
    return False


# 分开判断产物、派生数量、终态、计算和回收时序，不把数量违规伪装成多项状态故障。
def assess_workflow(events, expected, actual, states, *, root_id="root"):
    roots = [e for e in events if e.get("run_id") == root_id]
    children = {e["run_id"] for e in events if e["type"] == "subagent.started" and e.get("parent_run_id") == root_id}
    spawns = [e for e in roots if e["type"] == "tool.call_started" and e.get("tool_name") == "spawn_agent"]
    terminal = {e["run_id"]: e["ts"] for e in events if e["type"] == "subagent.finished" and e.get("status") == "success"}
    finished = {(e["run_id"], e.get("tool_use_id")): e for e in events if e["type"] == "tool.call_finished"}
    calculated = {}
    child_values = {}
    for event in events:
        if event["type"] != "tool.call_started" or event.get("tool_name") != "bash" or event["run_id"] not in children:
            continue
        result = finished.get((event["run_id"], event["tool_use_id"]))
        if result is None:
            continue
        for part, wanted in expected.items():
            if event["params"].get("command") == f"python calculate.py {part}.json":
                try:
                    if strict_equal(json.loads(result.get("output", "")), wanted):
                        calculated[part] = event["run_id"]
                        child_values.setdefault(event["run_id"], []).append(wanted)
                except ValueError:
                    pass
    retrieved = {}
    for event in roots:
        if event["type"] != "tool.call_started" or event.get("tool_name") != "agent_result":
            continue
        child = event["params"].get("run_id")
        result = finished.get((root_id, event["tool_use_id"]))
        if (child in children and child in terminal and result is not None
                and result["ts"] >= terminal[child]
                and not result.get("output", "").startswith(("waiting_for_approval:", "still running"))
                and any(contains_result(result.get("output", ""), value) for value in child_values.get(child, []))):
            retrieved.setdefault(child, result["ts"])
    writes = [e for e in roots if e["type"] == "tool.call_started" and e.get("tool_name") == "write_file"
              and e["params"].get("path") == "answer.json"]
    checks = {
        "artifact": strict_equal(actual, expected),
        "root_completion": [e.get("status") for e in roots if e["type"] == "run.finished"] == ["success"],
        "child_count": len(children) == 3,
        "spawn_attempt_count": len(spawns) == 3,
        "background_spawning": all(e["params"].get("run_in_background") is True for e in spawns),
        "children_success": bool(children) and children.issubset(terminal),
        "children_terminal": ({s["run_id"] for s in states} == children
                              and all(s["done"] and not s["cancelled"] and s["status"] == "success" for s in states)),
        "child_tool_contract": not any(e["type"] == "tool.call_started" and e["run_id"] in children
                                       and e.get("tool_name") not in {"read_file", "list_dir", "bash"} for e in events),
        "calculations": set(calculated) == set(expected) and len(set(calculated.values())) == 3,
        "final_results": bool(children) and set(retrieved) == children,
        "artifact_write": bool(writes) and all((root_id, e["tool_use_id"]) in finished for e in writes),
        # 缺少最终结果时不能断言先后顺序；缺失事实由 final_results 单独报告。
        "write_order": (all(e["ts"] >= max(retrieved.values()) for e in writes)
                        if children and set(retrieved) == children else None),
    }
    errors = [name for name, valid in checks.items() if valid is False]
    return {"passed": not errors, "artifact_valid": checks["artifact"],
            "protocol_valid": not any(name != "artifact" for name in errors),
            "child_count": len(children), "checks": checks, "errors": errors}
