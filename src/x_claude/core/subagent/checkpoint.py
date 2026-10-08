from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from x_claude.core.atomic_file import atomic_write_bytes
from x_claude.core.config import XConfig
from x_claude.core.context import ExecutionContext

if TYPE_CHECKING:
    from x_claude.core.bus.commands import RecoveredToolResult

log = logging.getLogger(__name__)
Phase = Literal["ready", "planning", "tools", "finished"]
State = Literal["running", "completed", "suspended", "cancelled", "blocked"]


# 仅保存配置及工具契约指纹，不把 API key 或 MCP 环境变量原文写入检查点
def runtime_signature(config: XConfig, schemas: list[dict[str, Any]]) -> str:
    payload = {
        "llm": asdict(config.llm),
        "base_url": os.environ.get("ANTHROPIC_BASE_URL", ""),
        "mcp": asdict(config.mcp),
        "tools": sorted(schemas, key=lambda item: str(item["name"])),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ContextSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    goal: str
    max_steps: int = Field(ge=1)
    messages: list[dict[str, Any]]
    step: int = Field(ge=0)
    status: Literal["running", "success", "failed"]
    reason: str | None
    result: str
    system_prompt_override: str | None
    session_notes: str = ""
    global_context: str = ""
    project_context: str = ""
    file_versions: dict[str, str | None] = Field(default_factory=dict)

    # 只记录恢复子循环必需的字段，不保存客户端、协程或认证配置
    @classmethod
    def capture(cls, context: ExecutionContext) -> ContextSnapshot:
        return cls(**{name: getattr(context, name) for name in cls.model_fields})

    # 使用已完成步骤和完整消息重建上下文，不从任务开头重新执行
    def restore(self, run_id: str) -> ExecutionContext:
        if self.step > self.max_steps:
            raise ValueError("checkpoint step exceeds max_steps")
        return ExecutionContext(run_id=run_id, **self.model_dump())


class BackgroundRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1] = 1
    kind: Literal["background", "root"] = "background"
    thread_committed: bool = False
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    session_id: str
    parent_run_id: str
    description: str
    cwd: str
    depth: int = Field(ge=0, le=1)
    tools: list[str]
    runtime_signature: str | None = None
    model: str = ""
    phase: Phase = "ready"
    state: State = "running"
    context: ContextSnapshot
    updated_at: str = ""


class BackgroundCheckpoint:
    # 绑定单个后台任务的原子检查点文件及经过校验的数据
    def __init__(self, path: Path, record: BackgroundRecord) -> None:
        self.path = path
        self.record = record

    # 提取最后一条工具调用消息，供用户逐项核对中断工具而不是重新执行
    def pending_tools(self) -> list[dict[str, Any]]:
        if self.record.phase != "tools":
            return []
        if not self.record.context.messages:
            raise ValueError("missing pending tool message")
        messages = self.record.context.messages
        index = next((i for i in range(len(messages) - 1, -1, -1)
                      if messages[i].get("role") == "assistant"), -1)
        if index < 0:
            raise ValueError("missing pending assistant message")
        message = messages[index]
        content = message.get("content")
        if message.get("role") != "assistant" or not isinstance(content, list):
            raise ValueError("invalid pending tool message")
        tools = [block for block in content
                 if isinstance(block, dict) and block.get("type") == "tool_use"]
        ids = [block.get("id") for block in tools]
        if not ids or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("invalid pending tool ids")
        confirmed: set[str] = set()
        for result_message in messages[index + 1:]:
            blocks = result_message.get("content")
            if result_message.get("role") != "user" or not isinstance(blocks, list):
                raise ValueError("invalid tool result message")
            for block in blocks:
                if (not isinstance(block, dict) or block.get("type") != "tool_result"
                        or block.get("tool_use_id") not in ids
                        or block["tool_use_id"] in confirmed):
                    raise ValueError("invalid or duplicate tool result")
                confirmed.add(block["tool_use_id"])
        return [tool for tool in tools if tool["id"] not in confirmed]

    # 人工核对所有工具结果后原子提交安全边界；不执行或猜测任何中断工具
    def confirm_tools(
        self, results: dict[str, str | RecoveredToolResult], *, signature: str | None = None,
        model: str = "",
    ) -> None:
        from x_claude.core.bus.commands import RecoveredToolResult

        tools = self.pending_tools()
        if (not tools or self.record.context.status != "running"
                or set(results) != {t["id"] for t in tools}):
            raise ValueError("provide exactly one result for every pending tool_use id")
        parsed = {key: RecoveredToolResult(content=value) if isinstance(value, str) else value
                  for key, value in results.items()}
        if any(not value.content.strip() for value in parsed.values()):
            raise ValueError("tool results must not be empty")
        context = self.record.context.restore(self.record.run_id)
        for tool in tools:
            context.add_tool_result(
                tool["id"], "[User-verified recovery result; tool was not replayed]\n"
                + parsed[tool["id"]].content, is_error=parsed[tool["id"]].is_error,
            )
        update: dict[str, Any] = {
            "context": ContextSnapshot.capture(context), "phase": "ready", "state": "running",
        }
        if signature is not None:
            update.update(runtime_signature=signature, model=model)
        self._commit(self.record.model_copy(update=update))

    # 从磁盘恢复检查点，并核对目录身份，禁止跨会话或任意路径恢复
    @classmethod
    def load(cls, path: Path, session_id: str) -> BackgroundCheckpoint:
        record = BackgroundRecord.model_validate_json(path.read_bytes())
        if record.session_id != session_id or record.run_id != path.parent.name:
            raise ValueError("checkpoint identity mismatch")
        if ((path.name == "root.json") != (record.kind == "root")):
            raise ValueError("checkpoint kind mismatch")
        record.context.restore(record.run_id)
        if (record.phase == "finished" or record.state in ("completed", "cancelled")) and (
            record.context.status == "running"
        ):
            raise ValueError("unfinished context in terminal checkpoint")
        return cls(path, record)

    # 完整保存消息及执行阶段；提交失败向上传播，工具执行前不得绕过检查点
    def save(self, context: ExecutionContext, phase: Phase) -> None:
        state: State = "completed" if context.is_done() else "running"
        self._commit(self.record.model_copy(update={
            "context": ContextSnapshot.capture(context), "phase": phase, "state": state,
        }))

    # daemon 退出保留最后已确认状态，显式关闭或清空会话则永久取消未完成任务
    def interrupt(self, *, suspend: bool) -> None:
        if self.record.state in ("completed", "cancelled"):
            return
        if suspend:
            state: State = "blocked" if self.record.phase == "tools" else "suspended"
            self._commit(self.record.model_copy(update={"state": state}))
        else:
            context = self.record.context.restore(self.record.run_id)
            context.mark_failed("cancelled")
            self._commit(self.record.model_copy(update={
                "context": ContextSnapshot.capture(context), "phase": "finished",
                "state": "cancelled",
            }))

    # 在原目录原子写入 UTF-8 JSON，成功后才更新内存记录
    def _commit(self, record: BackgroundRecord) -> None:
        record = record.model_copy(update={"updated_at": datetime.now(UTC).isoformat()})
        atomic_write_bytes(self.path, (record.model_dump_json(indent=2) + "\n").encode("utf-8"))
        self.record = record

    # 即使会话尚未激活 Runner，关闭操作也必须取消其持久化的待恢复任务
    @classmethod
    def cancel_saved(cls, runs_dir: Path, session_id: str) -> None:
        paths = list(runs_dir.glob("*/background.json")) + list(runs_dir.glob("*/root.json"))
        for path in paths:
            try:
                checkpoint = cls.load(path, session_id)
            except ValueError:
                log.exception("cannot cancel invalid background checkpoint path=%s", path)
                continue
            checkpoint.interrupt(suspend=False)
