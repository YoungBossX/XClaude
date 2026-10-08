from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field

from x_claude.core.session.model import SessionMode, SessionStatus


class PingCommand(BaseModel):
    type: Literal["core.ping"] = "core.ping"
    client: str


class PongResult(BaseModel):
    server_version: str
    uptime_ms: int
    received_at: str  # ISO 8601


class AgentRunCommand(BaseModel):
    type: Literal["agent.run"] = "agent.run"
    goal: str


class AgentRunResult(BaseModel):
    run_id: str


class EventSubscribeCommand(BaseModel):
    type: Literal["event.subscribe"] = "event.subscribe"
    topics: list[str]          # fnmatch 模式，如 ["step.*", "tool.*"]
    scope: str = "global"      # "global" | "run:<run_id>" | "session:<session_id>"
    replay_from_run: str | None = None  # 设置则先从 events.jsonl 回放历史再接实时流


class EventSubscribeResult(BaseModel):
    subscription_id: str
    replayed_count: int = 0


class SessionCreateCommand(BaseModel):
    type: Literal["session.create"] = "session.create"
    mode: SessionMode = "chat"
    title: str = ""


class SessionCreateResult(BaseModel):
    session_id: str
    status: SessionStatus


class SessionContinueCommand(BaseModel):
    type: Literal["session.continue"] = "session.continue"


class SessionContinueResult(BaseModel):
    session_id: str
    status: SessionStatus


class SessionResumeCommand(BaseModel):
    type: Literal["session.resume"] = "session.resume"
    session_id: str


class SessionResumeResult(BaseModel):
    session_id: str
    status: SessionStatus


class SessionSendMessageCommand(BaseModel):
    type: Literal["session.send_message"] = "session.send_message"
    session_id: str
    content: str


class SessionSendMessageResult(BaseModel):
    run_id: str


class SessionGetHistoryCommand(BaseModel):
    type: Literal["session.get_history"] = "session.get_history"
    session_id: str


class SessionGetHistoryResult(BaseModel):
    messages: list[dict[str, Any]]


class SessionCloseCommand(BaseModel):
    type: Literal["session.close"] = "session.close"
    session_id: str


class SessionCloseResult(BaseModel):
    status: SessionStatus


class SessionClearCommand(BaseModel):
    type: Literal["session.clear"] = "session.clear"
    session_id: str


class SessionClearResult(BaseModel):
    session_id: str
    status: SessionStatus


class PermissionRespondCommand(BaseModel):
    type: Literal["permission.respond"] = "permission.respond"
    tool_use_id: str
    session_id: str
    run_id: str
    # "allow_once" | "always_allow" | "deny_once" | "always_deny"
    decision: Literal["allow_once", "always_allow", "deny_once", "always_deny"]


class PermissionRespondResult(BaseModel):
    ok: bool = True


class SessionCompactCommand(BaseModel):
    type: Literal["session.compact"] = "session.compact"
    session_id: str
    focus: str = ""


class SessionCompactResult(BaseModel):
    summary_tokens: int
    saved_tokens: int


class RecoveredToolResult(BaseModel):
    content: str
    is_error: bool = False


class SessionRecoverCommand(BaseModel):
    type: Literal["session.recover"] = "session.recover"
    session_id: str
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$")
    tool_results: dict[str, str | RecoveredToolResult] | None = None
    accept_config_change: bool = False


class BackgroundTaskInfo(BaseModel):
    run_id: str
    kind: Literal["background", "root"] = "background"
    state: str
    phase: str
    step: int
    message: str = ""
    pending_tools: list[dict[str, Any]] = Field(default_factory=list)


class SessionRecoverResult(BaseModel):
    tasks: list[BackgroundTaskInfo]


# 根据 type 字段决定命令类型的判别联合
Command = Annotated[
    PingCommand
    | AgentRunCommand
    | EventSubscribeCommand
    | SessionCreateCommand
    | SessionContinueCommand
    | SessionResumeCommand
    | SessionSendMessageCommand
    | SessionGetHistoryCommand
    | SessionCloseCommand
    | SessionClearCommand
    | PermissionRespondCommand
    | SessionCompactCommand
    | SessionRecoverCommand,
    Discriminator("type"),
]
