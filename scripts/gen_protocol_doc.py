#!/usr/bin/env python3
"""Generate WIRE_PROTOCOL.md from pydantic models in x_claude.core.bus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel

import x_claude
from x_claude.core.bus import commands, events
from x_claude.core.bus.commands import (
    AgentRunCommand,
    AgentRunResult,
    EventSubscribeCommand,
    EventSubscribeResult,
    PermissionRespondCommand,
    PermissionRespondResult,
    PingCommand,
    PongResult,
    SessionCloseCommand,
    SessionCloseResult,
    SessionCreateCommand,
    SessionCreateResult,
    SessionGetHistoryCommand,
    SessionGetHistoryResult,
    SessionRecoverCommand,
    SessionRecoverResult,
    SessionSendMessageCommand,
    SessionSendMessageResult,
)
from x_claude.core.bus.envelope import EventPushEnvelope, JsonRpcRequest
from x_claude.core.bus.events import (
    CoreStartedEvent,
    LlmModelSelectedEvent,
    LlmTokenEvent,
    LlmUsageEvent,
    LogLineEvent,
    RunFinishedEvent,
    RunRestoredEvent,
    RunStartedEvent,
    SessionClosedEvent,
    SessionCreatedEvent,
    SessionMessageReceivedEvent,
    SessionResumedEvent,
    SessionSynchronizedEvent,
    SessionWaitingForInputEvent,
    StepFinishedEvent,
    StepStartedEvent,
    SubagentFinishedEvent,
    SubagentRestoredEvent,
    SubagentStartedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)

_OUTPUT_PATH = Path(__file__).parent.parent / "WIRE_PROTOCOL.md"


# 从 pydantic 模型生成一个带字段表、JSON Schema 和可选示例的 Markdown 小节
def _model_section(name: str, model: type, example: dict | None = None) -> str:  # type: ignore[type-arg]
    schema = model.model_json_schema()  # type: ignore[attr-defined]
    props = schema.get("properties", {})
    required: set[str] = set(schema.get("required", []))

    table = ""
    if props:
        table = "\n| Field | Type | Required |\n|---|---|---|\n"
        for field_name, field_info in props.items():
            ftype = field_info.get("type", "object")
            if "anyOf" in field_info:
                ftype = " | ".join(t.get("type", "?") for t in field_info["anyOf"])
            req = "yes" if field_name in required else "no"
            table += f"| `{field_name}` | `{ftype}` | {req} |\n"

    schema_block = f"\n```json\n{json.dumps(schema, indent=2)}\n```\n"

    example_block = ""
    if example:
        example_block = f"\n**Example:**\n\n```json\n{json.dumps(example, indent=2)}\n```\n"

    return f"### {name}\n{table}{schema_block}{example_block}"


# 生成完整的 WIRE_PROTOCOL.md 文档字符串
def generate() -> str:
    run_id = "20260516-100000-abc123"
    ts = "2026-05-16T10:00:00.001Z"

    ping_req_example = {
        "jsonrpc": "2.0",
        "id": "u-1",
        "method": "core.ping",
        "params": {"client": "cli/0.0.1"},
    }
    pong_resp_example = {
        "jsonrpc": "2.0",
        "id": "u-1",
        "result": {
            "server_version": x_claude.__version__,
            "uptime_ms": 12,
            "received_at": ts,
        },
    }
    agent_run_req_example = {
        "jsonrpc": "2.0",
        "id": "u-2",
        "method": "agent.run",
        "params": {"goal": "总结 README.md 的主要章节"},
    }
    agent_run_resp_example = {
        "jsonrpc": "2.0",
        "id": "u-2",
        "result": {"run_id": run_id},
    }
    subscribe_req_example = {
        "jsonrpc": "2.0",
        "id": "u-3",
        "method": "event.subscribe",
        "params": {
            "topics": ["run.*", "step.*", "tool.*", "llm.token"],
            "scope": "global",
            "replay_from_run": None,
        },
    }
    subscribe_resp_example = {
        "jsonrpc": "2.0",
        "id": "u-3",
        "result": {"subscription_id": "sub-abc123", "replayed_count": 0},
    }
    session_id = "sess-abc123def456"
    session_create_req_example = {
        "jsonrpc": "2.0",
        "id": "u-4",
        "method": "session.create",
        "params": {"mode": "chat", "title": ""},
    }
    session_create_resp_example = {
        "jsonrpc": "2.0",
        "id": "u-4",
        "result": {"session_id": session_id, "status": "active"},
    }
    session_send_req_example = {
        "jsonrpc": "2.0",
        "id": "u-5",
        "method": "session.send_message",
        "params": {"session_id": session_id, "content": "总结 README.md"},
    }
    session_send_resp_example = {
        "jsonrpc": "2.0",
        "id": "u-5",
        "result": {"run_id": run_id},
    }
    event_push_example = {
        "kind": "event",
        "event": {
            "type": "step.started",
            "run_id": run_id,
            "step": 1,
            "ts": ts,
        },
    }
    for request in (ping_req_example, agent_run_req_example, subscribe_req_example,
                    session_create_req_example, session_send_req_example):
        request["auth_token"] = "<private local IPC credential>"

    sections = [
        "# Wire Protocol\n\n",
        "> Generated by `scripts/gen_protocol_doc.py`. **Do not edit manually.**\n\n",
        "## Transport\n\n",
        "- TCP loopback `127.0.0.1:7437` (override via `X_HOST` / `X_PORT`)\n",
        "- Each message is one `\\n`-terminated JSON line (NDJSON)\n",
        "- Commands use JSON-RPC 2.0 (client → server); Events use `kind=event` envelope (server → client)\n\n",
        "- Non-loopback bind addresses are rejected. Every command requires `auth_token`.\n",
        "- Daemon generates a private per-endpoint token in `~/.x/ipc/`; local clients read it automatically.\n",
        "- Token files are owner-only (Windows protected DACL / POSIX 0600). Tokens are not traced.\n",
        "- Approval replies must match session, run and tool ID and the connection's live subscription.\n\n",
        _model_section("JsonRpcRequest", JsonRpcRequest),
        "## Commands\n\n",
        "All commands are sent as authenticated JSON-RPC 2.0 requests. `method` selects the handler.\n\n",
        _model_section("PingCommand", PingCommand, ping_req_example),
        "\n",
        _model_section("PongResult", PongResult, pong_resp_example),
        "\n",
        _model_section("AgentRunCommand", AgentRunCommand, agent_run_req_example),
        "\n",
        _model_section("AgentRunResult", AgentRunResult, agent_run_resp_example),
        "\n",
        _model_section("EventSubscribeCommand", EventSubscribeCommand, subscribe_req_example),
        "\n",
        _model_section("EventSubscribeResult", EventSubscribeResult, subscribe_resp_example),
        "\n",
        _model_section("SessionCreateCommand", SessionCreateCommand, session_create_req_example),
        "\n",
        _model_section("SessionCreateResult", SessionCreateResult, session_create_resp_example),
        "\n",
        _model_section("SessionSendMessageCommand", SessionSendMessageCommand, session_send_req_example),
        "\n",
        _model_section("SessionSendMessageResult", SessionSendMessageResult, session_send_resp_example),
        "\n",
        _model_section("SessionGetHistoryCommand", SessionGetHistoryCommand),
        "\n",
        _model_section("SessionGetHistoryResult", SessionGetHistoryResult),
        "\n",
        _model_section("SessionCloseCommand", SessionCloseCommand),
        "\n",
        _model_section("SessionCloseResult", SessionCloseResult),
        _model_section("SessionRecoverCommand", SessionRecoverCommand),
        _model_section("SessionRecoverResult", SessionRecoverResult),
        _model_section("PermissionRespondCommand", PermissionRespondCommand),
        _model_section("PermissionRespondResult", PermissionRespondResult),
        "\n## Server Push\n\n",
        "Events pushed from daemon to subscribed clients over the same TCP connection.\n\n",
        _model_section("EventPushEnvelope", EventPushEnvelope, event_push_example),
        "\n## IPC Events\n\n",
        "Events sent over the IPC socket (daemon → client).\n\n",
        _model_section("CoreStartedEvent", CoreStartedEvent),
        "\n## Run Events\n\n",
        "Events are written to `~/.x/sessions/<session_id>/runs/<run_id>/events.jsonl` "
        "and forwarded over IPC. Direct standalone runners may use `runs/<run_id>/events.jsonl`.\n\n",
        "`log_positions` carries per-log `[start, end]` byte offsets on IPC; it is reconstructed "
        "during replay and is not stored recursively in JSONL. A client must acknowledge only "
        "contiguous received ranges, never jump over an unseen event.\n\n",
        "Session replay accepts `replay_offsets`; the response confirms delivered snapshot offsets. "
        "Initial TUI history is bounded using `replay_tail_runs` / `replay_tail_bytes`; "
        "subsequent reconnects request the complete missing suffix. Historical approvals are "
        "not reactivated. `session.history_page` reads stored messages without executing tasks. "
        "Message previews exceeding 16000 characters are truncated before transmission, "
        "with an explicit notice; persisted history and model context are unchanged.\n\n",
        "`llm.retrying` resets the incomplete text for that run. `llm.text_completed` replaces "
        "the attempt's displayed text with the complete response. `llm.error` carries a stable "
        "code and a safe, actionable hint.\n\n",
        _model_section("RunStartedEvent", RunStartedEvent,
            {"type": "run.started", "run_id": run_id, "goal": "总结 README.md", "ts": ts}),
        "\n",
        _model_section("RunRestoredEvent", RunRestoredEvent),
        "\n",
        _model_section("RunFinishedEvent", RunFinishedEvent, {
            "type": "run.finished", "run_id": run_id,
            "status": "success", "reason": None, "steps": 2, "ts": ts}),
        "\n",
        _model_section("StepStartedEvent", StepStartedEvent,
            {"type": "step.started", "run_id": run_id, "step": 1, "ts": ts}),
        "\n",
        _model_section("StepFinishedEvent", StepFinishedEvent,
            {"type": "step.finished", "run_id": run_id, "step": 1, "ts": ts}),
        "\n",
        _model_section("ToolCallStartedEvent", ToolCallStartedEvent,
            {"type": "tool.call_started", "run_id": run_id, "tool_use_id": "toolu_01",
             "tool_name": "read_file", "params": {"path": "README.md"}, "ts": ts}),
        "\n",
        _model_section("ToolCallFinishedEvent", ToolCallFinishedEvent,
            {"type": "tool.call_finished", "run_id": run_id, "tool_use_id": "toolu_01",
             "tool_name": "read_file", "elapsed_ms": 3, "ts": ts}),
        "\n",
        _model_section("ToolCallFailedEvent", ToolCallFailedEvent,
            {"type": "tool.call_failed", "run_id": run_id, "tool_use_id": "toolu_02",
             "tool_name": "read_file", "error_class": "runtime_error",
             "error_message": "file not found", "elapsed_ms": 1, "attempt": 1, "ts": ts}),
        "\n",
        _model_section("LlmModelSelectedEvent", LlmModelSelectedEvent,
            {"type": "llm.model_selected", "run_id": run_id,
             "model": "claude-sonnet-4-6", "strategy": "static", "ts": ts}),
        "\n",
        _model_section("LlmTokenEvent", LlmTokenEvent,
            {"type": "llm.token", "run_id": run_id, "token": "The ", "ts": ts}),
        "\n",
        _model_section("LlmUsageEvent", LlmUsageEvent,
            {"type": "llm.usage", "run_id": run_id, "input_tokens": 512, "output_tokens": 48,
             "cache_read_input_tokens": 490, "cache_creation_input_tokens": 0, "ts": ts}),
        "\n",
        _model_section("LogLineEvent", LogLineEvent,
            {"type": "log.line", "run_id": run_id, "level": "INFO",
             "source": "x_claude.core.loop", "message": "step 1 started", "ts": ts}),
        "\n## Session Events\n\n",
        _model_section("SessionCreatedEvent", SessionCreatedEvent,
            {"type": "session.created", "session_id": session_id, "mode": "chat", "ts": ts}),
        "\n",
        _model_section("SessionMessageReceivedEvent", SessionMessageReceivedEvent,
            {"type": "session.message_received", "session_id": session_id,
             "content": "总结 README.md", "ts": ts}),
        "\n",
        _model_section("SessionWaitingForInputEvent", SessionWaitingForInputEvent,
            {"type": "session.waiting_for_input", "session_id": session_id,
             "last_run_id": run_id, "ts": ts}),
        "\n",
        _model_section("SessionResumedEvent", SessionResumedEvent,
            {"type": "session.resumed", "session_id": session_id, "ts": ts}),
        "\n",
        _model_section("SessionClosedEvent", SessionClosedEvent,
            {"type": "session.closed", "session_id": session_id, "ts": ts}),
        "\n",
        _model_section("SessionSynchronizedEvent", SessionSynchronizedEvent),
        "\n## Subagent Events\n\n",
        _model_section("SubagentStartedEvent", SubagentStartedEvent),
        "\n",
        _model_section("SubagentRestoredEvent", SubagentRestoredEvent),
        "\n",
        _model_section("SubagentFinishedEvent", SubagentFinishedEvent),
        "\n## Error Codes\n\n",
        "| Code | Name | Meaning |\n",
        "|------|------|---------|\n",
        "| -32700 | Parse Error | Invalid JSON received |\n",
        "| -32600 | Invalid Request | Missing required JSON-RPC fields |\n",
        "| -32601 | Method Not Found | Unknown method |\n",
        "| -32602 | Invalid Params | Parameter validation failed |\n",
        "| -32603 | Internal Error | Handler raised an unhandled exception |\n",
        "| -32000 | Application Error | Application-specific failure |\n",
        "| -32001 | Unauthorized | Missing or invalid local IPC credential |\n",
        "| -32010 | Session Not Found | Unknown session or no prior chat |\n",
        "| -32011 | Session Closed | Session no longer accepts messages |\n",
        "| -32012 | Session Busy | Active or unfinished task |\n",
        "| -32030 | Recovery Review Required | Invalid checkpoint or unconfirmed recovery |\n",
    ]
    document = "".join(sections)
    document = document.replace("\n## Server Push", _remaining_models(commands, document)
                                + "\n## Server Push")
    return document.replace("\n## Error Codes", _remaining_models(events, document)
                            + "\n## Error Codes")


# 自动补齐新增契约模型，避免生成器只覆盖早期 S0 模型或遗漏后续命令与事件
def _remaining_models(module: ModuleType, document: str) -> str:
    return "".join(
        "\n" + _model_section(name, model)
        for name, model in sorted(vars(module).items())
        if isinstance(model, type) and issubclass(model, BaseModel)
        and model.__module__ == module.__name__ and f"### {name}\n" not in document
    )


# 解析命令行参数，写出或校验 WIRE_PROTOCOL.md
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WIRE_PROTOCOL.md")
    parser.add_argument("--check", action="store_true", help="Verify file matches generated output")
    parser.add_argument("--output", default=str(_OUTPUT_PATH))
    args = parser.parse_args()

    content = generate()

    if args.check:
        output_path = Path(args.output)
        if not output_path.exists():
            print(f"ERROR: {output_path} not found — run: make docs", file=sys.stderr)
            sys.exit(1)
        if output_path.read_text(encoding="utf-8") != content:
            print(f"ERROR: {output_path} out of sync with code — run: make docs", file=sys.stderr)
            sys.exit(1)
        print(f"OK: {output_path} is up to date.")
    else:
        output_path = Path(args.output)
        output_path.write_text(content, encoding="utf-8")
        print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
