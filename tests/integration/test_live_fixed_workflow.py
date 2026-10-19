from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path

import pytest

from tests.fixed_workflow_support import assess_workflow, strict_equal
from x_claude.core.context import ExecutionContext
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.types import ToolCallBlock
from x_claude.core.loop import AgentLoop
from x_claude.core.permissions.manager import PermissionManager
from x_claude.core.permissions.policy import PermissionDecision, ToolPolicy
from x_claude.core.runner import AgentRunner
from x_claude.core.subagent.tool import SpawnAgentTool
from x_claude.core.tools.builtin.bash import BashTool
from x_claude.core.tools.invocation import invoke_tool
from x_claude.core.tools.registry import ToolRegistry

pytestmark = pytest.mark.integration
COMMANDS = [f"python calculate.py part-{i}.json" for i in range(3)]


# 每个付费试验使用独立目录，沿用原六组数据公式但不导入或修改原始试验驱动。
def prepare_case(path, case_number):
    expected, inputs = {}, {}
    for part in range(3):
        values = [(i * (part + 2) + case_number) % 97 for i in range(120)]
        inputs[f"part-{part}.json"] = json.dumps(values).encode()
        expected[f"part-{part}"] = {"count": 120, "total": sum(values), "maximum": max(values)}
    inputs["calculate.py"] = (
        b"import json, sys\nfrom pathlib import Path\n"
        b"name = sys.argv[1]\nassert name in {'part-0.json', 'part-1.json', 'part-2.json'}\n"
        b"xs = json.loads(Path(name).read_text(encoding='utf-8'))\n"
        b"print(json.dumps({'count': len(xs), 'total': sum(xs), 'maximum': max(xs)}))\n"
    )
    for name, content in inputs.items():
        (path / name).write_bytes(content)
    return expected, inputs


# 保存合成输入对应的真实模型轮次及事件，不输出或保存 SDK 密钥、请求头等凭据。
def save_evidence(path, provider, events, result):
    calls = [{"run_id": call["run_id"], "messages": call["messages"],
              "response": asdict(call["response"]) if "response" in call else None}
             for call in provider.calls]
    usage = [e for e in events if e["type"] == "llm.usage"]
    result = {**result, "model_turns": len(calls), "usage_records": len(usage),
              "usage_complete": len(usage) == len(calls),
              "input_tokens": sum(e["input_tokens"] for e in usage),
              "output_tokens": sum(e["output_tokens"] for e in usage),
              "cache_read_input_tokens": sum(e["cache_read_input_tokens"] for e in usage),
              "cache_creation_input_tokens": sum(e["cache_creation_input_tokens"] for e in usage),
              "stream_retries": sum(e["type"] == "llm.retrying" for e in events)}
    for name, value in (("result.json", result), ("api_turns.json", calls), ("observed_events.json", events)):
        (path / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


# 真实进程使用当前 Python；不改变本机配置，只约束本次固定任务的资源和命令集合。
def configure_fixed(config, provider, monkeypatch):
    config.agent.max_steps = 14
    config.agent.max_tasks = 4
    config.agent.max_runtime_s = 145
    config.agent.max_total_tokens = 120000
    config.agent.allowed_shell_commands = list(COMMANDS)
    config.compaction.auto_threshold = 0
    provider.max_calls = 32
    import sys
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    original = SpawnAgentTool._build_child_registry

    # 与原评测保持相同子工具范围，生产派生、预算、命令校验及权限链均使用真实实现。
    def child_registry(tool, *args, **kwargs):
        source, selected = original(tool, *args, **kwargs), ToolRegistry()
        for name in ("read_file", "list_dir", "bash"):
            if actual := source.get(name):
                selected.register(actual)
        return selected

    monkeypatch.setattr(SpawnAgentTool, "_build_child_registry", child_registry)


# 功能：真实模型在固定三分片协议下创建恰好三个子任务，取得真实计算结果后输出正确产物。
# 设计：六组既定数据各三次独立运行，严格分开产物与过程验收，失败尝试原样留档。
@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.parametrize("case_number", range(6))
async def test_live_fixed_three_children(live_api, tmp_path, monkeypatch, case_number, repeat):
    config, provider = live_api
    configure_fixed(config, provider, monkeypatch)
    expected, inputs = prepare_case(tmp_path, case_number)
    events, bus = [], EventBus()

    # 记录真实事件流，所有主验收均读取实际工具参数、输出和时序。
    async def collect(event):
        events.append(event.model_dump())

    bus.subscribe(collect)
    pm = PermissionManager(policies={
        **{name: ToolPolicy(PermissionDecision.ALLOW) for name in
           ("read_file", "list_dir", "write_file", "spawn_agent", "agent_result")},
        "bash": ToolPolicy(PermissionDecision.DENY, allow_patterns=[r"^python calculate\.py part-[012]\.json$"]),
    })
    runner = AgentRunner(config, provider=provider, bus=bus, permission_manager=pm, runs_dir=tmp_path / "runs")
    result = {"passed": False, "case_number": case_number, "repeat": repeat,
              "protocol": "fixed-three-v2", "fault_injection": False, "effective_agent": asdict(config.agent)}
    try:
        outcome = await runner.run_and_capture(
            "Compute count, total and maximum for part-0.json, part-1.json and part-2.json. "
            "FIXED PROTOCOL: make EXACTLY THREE spawn_agent calls total, one background child for each part. "
            "Never create additional or replacement children, including after errors. "
            "Only children calculate. Each child must use its exact command: " + "; ".join(COMMANDS) + ". "
            "Pass these rules explicitly to each child: the working directory is already correct; "
            "do NOT add cd, wrappers, prefixes or suffixes. Only read files or run the exact calculator command. "
            "Return the calculator's JSON output unchanged in your final reply. "
            "Do not write files or spawn agents. On command_contract schema_error the command was NOT executed; "
            "correct it once within the SAME child. Never bypass permission denial or retry uncertain side effects. "
            "If calculation still fails, report failure, do not invent results. "
            "Parent must retrieve each child's final successful calculation with agent_result before writing answer.json. "
            "JSON keys: part-0, part-1, part-2; each has exactly count, total, maximum as integers. "
            "Do not modify inputs or calculate.py. Complete the file before replying briefly.",
            run_id="root", tool_whitelist=["spawn_agent", "agent_result", "write_file", "read_file", "list_dir"],
        )
        states = [{"run_id": ctx.run_id, "done": task.done(), "cancelled": task.cancelled(), "status": ctx.status}
                  for task, ctx in runner._task_registry.all()]
        try:
            actual = json.loads((tmp_path / "answer.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            actual = None
        result.update(assess_workflow(events, expected, actual, states), outcome=asdict(outcome), states=states)
        result["inputs_unchanged"] = all((tmp_path / name).read_bytes() == raw for name, raw in inputs.items())
        result["passed"] = result["passed"] and result["inputs_unchanged"]
        assert result["passed"], json.dumps(result, ensure_ascii=False)
    finally:
        try:
            save_evidence(tmp_path, provider, events, result)
        finally:
            await runner.shutdown()


# 功能：真实模型看到已拒绝且未执行的错误命令后，在同一个子任务内修正，不新增任务。
# 设计：仅初始错误工具调用为故障注入，其校验、后续模型推理、Shell 执行均真实；单列诊断不混入18次自然试验。
async def test_live_same_child_recovers_contract_error(live_api, tmp_path, monkeypatch):
    config, provider = live_api
    configure_fixed(config, provider, monkeypatch)
    expected, _ = prepare_case(tmp_path, 4)
    registry, bus, events = ToolRegistry(), EventBus(), []
    registry.register(BashTool(allowed_commands=[COMMANDS[1]]))

    # 保存故障注入与后续真实调用的完整事件，检查没有审批绕过和替代派生。
    async def collect(event):
        events.append(event.model_dump())

    bus.subscribe(collect)
    pm = PermissionManager(policies={"bash": ToolPolicy(PermissionDecision.DENY,
                           allow_patterns=[r"^python calculate\.py part-1\.json$"])})
    context = ExecutionContext("original-child", "Run exactly python calculate.py part-1.json and report its JSON output. "
                               "The current directory is already correct. Correct a rejected command within this task; "
                               "do not bypass permission denial or invent values.", 5)
    result = {"passed": False, "fault_injection": "first real tool call input only", "protocol": "same-child-recovery"}
    try:
        first = await provider.chat(messages=list(context.messages), tool_schemas=registry.tool_schemas(),
                                    bus=bus, run_id=context.run_id, step=0)
        assert first.stop_reason == "tool_use" and len(first.tool_calls) == 1
        original = first.tool_calls[0]
        assert original.name == "bash"
        call = ToolCallBlock(original.id, "bash", {"command": "cd . && python calculate.py part-1.json"})
        blocks = list(first.thinking_blocks)
        if first.text:
            blocks.append({"type": "text", "text": first.text})
        blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.input})
        context.add_assistant_message(blocks)
        rejected = await invoke_tool(registry, call, bus, context.run_id, permission_manager=pm)
        assert rejected.error_type == "schema_error" and "command_contract" in rejected.content
        context.add_tool_result(call.id, rejected.content, is_error=rejected.is_error)
        await AgentLoop(provider, registry, bus, permission_manager=pm).run(context)
        executed = [e for e in events if e["type"] == "tool.call_finished" and e["tool_name"] == "bash"]
        assert context.status == "success"
        assert len(executed) == 1 and executed[0]["run_id"] == "original-child"
        assert strict_equal(json.loads(executed[0]["output"]), expected["part-1"])
        assert not any(e["type"] in ("subagent.started", "permission.requested") for e in events)
        result["passed"] = True
    finally:
        save_evidence(tmp_path, provider, events, result)


# 功能：未启用固定协议的真实多 Agent 仍可派生第四个子任务并回收其结果。
# 设计：普通预算与默认命令配置，四个独立文件交给四个真实子任务读取并写出汇总。
async def test_live_default_dynamic_four_children(live_api, tmp_path):
    config, provider = live_api
    config.agent.max_steps, config.agent.max_tasks, provider.max_calls = 14, 32, 32
    config.agent.max_total_tokens, config.agent.max_runtime_s = 120000, 145
    config.agent.allowed_shell_commands = None
    config.compaction.auto_threshold = 0
    expected = {f"item-{i}": 7310 + i * 17 for i in range(4)}
    for name, value in expected.items():
        (tmp_path / f"{name}.txt").write_text(str(value), encoding="utf-8")
    events, bus = [], EventBus()

    # 保留普通模式的真实派生与文件读取事件，证明没有全局三子任务限制。
    async def collect(event):
        events.append(event.model_dump())

    bus.subscribe(collect)
    pm = PermissionManager(policies={name: ToolPolicy(PermissionDecision.ALLOW) for name in
                                     ("spawn_agent", "agent_result", "read_file", "write_file")})
    runner = AgentRunner(config, provider=provider, bus=bus, permission_manager=pm, runs_dir=tmp_path / "runs")
    result = {"passed": False, "protocol": "default-dynamic-control", "fault_injection": False,
              "effective_agent": asdict(config.agent)}
    try:
        outcome = await runner.run_and_capture(
            "Spawn exactly FOUR background child agents, one for each item-0.txt through item-3.txt. "
            "Each child must use read_file on its assigned file and return the integer without writing or spawning. "
            "Retrieve all four final results with agent_result, then write answer.json mapping item-0 through item-3 "
            "to the integers. Do not read the files yourself, invent values, or end before the artifact is written.",
            run_id="root", tool_whitelist=["spawn_agent", "agent_result", "write_file"],
        )
        children = {e["run_id"] for e in events if e["type"] == "subagent.started" and e.get("parent_run_id") == "root"}
        readers = {e["run_id"] for e in events if e["type"] == "tool.call_finished" and e.get("tool_name") == "read_file"}
        assert outcome.status == "success"
        assert len(children) == 4 and readers == children
        terminal = {e["run_id"]: e["ts"] for e in events
                    if e["type"] == "subagent.finished" and e.get("status") == "success"}
        finished = {(e["run_id"], e.get("tool_use_id")): e for e in events if e["type"] == "tool.call_finished"}
        ownership = {}
        for event in events:
            if event["type"] != "tool.call_started" or event.get("tool_name") != "read_file":
                continue
            name = Path(event["params"]["path"]).stem
            read = finished.get((event["run_id"], event["tool_use_id"]))
            if name in expected and read and re.search(rf"\b{expected[name]}\b", read.get("output", "")):
                ownership[event["run_id"]] = name
        assert set(ownership) == children and set(ownership.values()) == set(expected)
        retrieved = {}
        for event in events:
            if (event["run_id"] != "root" or event["type"] != "tool.call_started"
                    or event.get("tool_name") != "agent_result"):
                continue
            child = event["params"].get("run_id")
            reply = finished.get(("root", event["tool_use_id"]))
            if (child in ownership and child in terminal and reply and reply["ts"] >= terminal[child]
                    and re.search(rf"\b{expected[ownership[child]]}\b", reply.get("output", ""))):
                retrieved.setdefault(child, reply["ts"])
        assert set(retrieved) == children
        writes = [e for e in events if e["type"] == "tool.call_started" and e["run_id"] == "root"
                  and e.get("tool_name") == "write_file" and e["params"].get("path") == "answer.json"]
        assert writes and all(e["ts"] >= max(retrieved.values()) for e in writes)
        assert strict_equal(json.loads((tmp_path / "answer.json").read_text(encoding="utf-8")), expected)
        assert all(task.done() and not task.cancelled() and ctx.status == "success" for task, ctx in runner._task_registry.all())
        result.update(passed=True, child_count=len(children))
    finally:
        try:
            save_evidence(tmp_path, provider, events, result)
        finally:
            await runner.shutdown()
