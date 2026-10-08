from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Discriminator, Field


class RuntimeEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex)


class SessionSynchronizedEvent(RuntimeEvent):
    type: Literal["session.synchronized"] = "session.synchronized"
    session_id: str
    status: Literal["active", "waiting_for_input", "closed"]
    busy: bool
    ts: str


class CoreStartedEvent(RuntimeEvent):
    type: Literal["core.started"] = "core.started"
    listen_addr: str  # e.g. "127.0.0.1:7437"
    version: str


class RunStartedEvent(RuntimeEvent):
    type: Literal["run.started"] = "run.started"
    run_id: str
    session_id: str = ""
    resumed: bool = False
    goal: str
    ts: str  # ISO 8601


class RunRestoredEvent(RuntimeEvent):
    type: Literal["run.restored"] = "run.restored"
    run_id: str
    session_id: str
    state: str
    step: int
    message: str
    ts: str


class RunFinishedEvent(RuntimeEvent):
    type: Literal["run.finished"] = "run.finished"
    run_id: str
    status: str  # "success" | "failed"
    reason: str | None = None  # "exceeded_max_steps" | "cancelled" | "llm_error" | ...
    steps: int
    ts: str


class StepStartedEvent(RuntimeEvent):
    type: Literal["step.started"] = "step.started"
    run_id: str
    step: int
    ts: str


class StepFinishedEvent(RuntimeEvent):
    type: Literal["step.finished"] = "step.finished"
    run_id: str
    step: int
    ts: str


class ToolCallStartedEvent(RuntimeEvent):
    type: Literal["tool.call_started"] = "tool.call_started"
    run_id: str
    tool_use_id: str
    tool_name: str
    params: dict[str, Any]
    ts: str


class ToolCallFinishedEvent(RuntimeEvent):
    type: Literal["tool.call_finished"] = "tool.call_finished"
    run_id: str
    tool_use_id: str
    tool_name: str
    elapsed_ms: int
    output: str = ""  # tool result content, for TUI display
    ts: str


class ToolCallFailedEvent(RuntimeEvent):
    type: Literal["tool.call_failed"] = "tool.call_failed"
    run_id: str
    tool_use_id: str
    tool_name: str
    # "runtime_error" | "timeout" | "schema_error" | "permission_denied" | "rate_limited"
    error_class: str
    error_message: str
    elapsed_ms: int
    attempt: int = 1  # 1=first attempt, 2=first retry, 3=second retry
    ts: str


class LlmTokenEvent(RuntimeEvent):
    type: Literal["llm.token"] = "llm.token"
    run_id: str
    token: str
    ts: str


class LlmUsageEvent(RuntimeEvent):
    type: Literal["llm.usage"] = "llm.usage"
    run_id: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    context_pct: float = 0.0
    context_window: int = 0
    ts: str


class LlmModelSelectedEvent(RuntimeEvent):
    type: Literal["llm.model_selected"] = "llm.model_selected"
    run_id: str
    model: str
    strategy: str  # "static" | "rule_based" | "cost_budget"
    ts: str


class LogLineEvent(RuntimeEvent):
    type: Literal["log.line"] = "log.line"
    run_id: str
    level: str  # "DEBUG" | "INFO" | "WARNING" | "ERROR"
    source: str
    message: str
    ts: str


class SessionCreatedEvent(RuntimeEvent):
    type: Literal["session.created"] = "session.created"
    session_id: str
    mode: str
    ts: str


class SessionMessageReceivedEvent(RuntimeEvent):
    type: Literal["session.message_received"] = "session.message_received"
    session_id: str
    content: str
    ts: str


class SessionWaitingForInputEvent(RuntimeEvent):
    type: Literal["session.waiting_for_input"] = "session.waiting_for_input"
    session_id: str
    last_run_id: str
    ts: str


class SessionResumedEvent(RuntimeEvent):
    type: Literal["session.resumed"] = "session.resumed"
    session_id: str
    ts: str


class SessionClosedEvent(RuntimeEvent):
    type: Literal["session.closed"] = "session.closed"
    session_id: str
    ts: str


class ContextCompactedEvent(RuntimeEvent):
    type: Literal["context.compacted"] = "context.compacted"
    session_id: str
    run_id: str
    original_tokens: int
    summary_tokens: int
    ts: str


class PermissionRequestedEvent(RuntimeEvent):
    type: Literal["permission.requested"] = "permission.requested"
    run_id: str
    tool_use_id: str
    tool_name: str
    params: dict[str, Any]
    param_preview: str
    session_id: str
    ts: str


class PermissionGrantedEvent(RuntimeEvent):
    type: Literal["permission.granted"] = "permission.granted"
    run_id: str
    tool_use_id: str
    # "allow_once" | "always_allow" | "auto_allow"
    decision: str
    ts: str


class PermissionDeniedEvent(RuntimeEvent):
    type: Literal["permission.denied"] = "permission.denied"
    run_id: str
    tool_use_id: str
    # "deny_once" | "always_deny" | "auto_deny"
    decision: str
    ts: str


class SubagentStartedEvent(RuntimeEvent):
    type: Literal["subagent.started"] = "subagent.started"
    run_id: str          # 子 agent run_id
    parent_run_id: str
    description: str
    ts: str
    session_id: str = ""
    resumed: bool = False


class SubagentRestoredEvent(RuntimeEvent):
    type: Literal["subagent.restored"] = "subagent.restored"
    run_id: str
    session_id: str
    parent_run_id: str
    description: str
    state: str
    step: int
    message: str
    ts: str


class SubagentFinishedEvent(RuntimeEvent):
    type: Literal["subagent.finished"] = "subagent.finished"
    run_id: str
    parent_run_id: str
    status: str          # "success" | "failed"
    ts: str
    session_id: str = ""


class SkillInvokedEvent(RuntimeEvent):
    type: Literal["skill.invoked"] = "skill.invoked"
    skill_name: str
    arguments: str
    run_id: str
    session_id: str = ""
    ts: str


# 根据 type 字段决定事件类型的判别联合
Event = Annotated[
    CoreStartedEvent
    | RunStartedEvent
    | RunRestoredEvent
    | RunFinishedEvent
    | StepStartedEvent
    | StepFinishedEvent
    | ToolCallStartedEvent
    | ToolCallFinishedEvent
    | ToolCallFailedEvent
    | LlmTokenEvent
    | LlmUsageEvent
    | LlmModelSelectedEvent
    | LogLineEvent
    | SessionCreatedEvent
    | SessionMessageReceivedEvent
    | SessionWaitingForInputEvent
    | SessionResumedEvent
    | SessionClosedEvent
    | SessionSynchronizedEvent
    | ContextCompactedEvent
    | PermissionRequestedEvent
    | PermissionGrantedEvent
    | PermissionDeniedEvent
    | SubagentStartedEvent
    | SubagentRestoredEvent
    | SubagentFinishedEvent
    | SkillInvokedEvent,
    Discriminator("type"),
]
