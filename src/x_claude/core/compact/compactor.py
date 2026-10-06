from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from x_claude.core.atomic_file import atomic_write_bytes
from x_claude.core.bus.events import ContextCompactedEvent
from x_claude.core.events.bus import EventBus

if TYPE_CHECKING:
    from x_claude.core.context import ExecutionContext
    from x_claude.core.llm.base import LLMProvider
    from x_claude.core.session.store import SessionStore

logger = logging.getLogger(__name__)

_COMPACT_PROMPT = """\
You are compressing an agent conversation into a handoff summary.
Another LLM instance will continue this task from your summary alone — make it complete.

Structure your response with exactly these six sections:

## 1. Original Goal
One sentence describing what the user asked the agent to accomplish.

## 2. Completed Steps
Bullet list of what has been done. Be specific (file paths, commands run, decisions made).

## 3. Key Constraints & Discoveries
Facts learned during the run that affect future decisions \
(e.g., API limitations, file formats, user preferences stated mid-conversation).

## 4. Current File State
For each file that was created or modified: path, a one-line description of its current state.

## 5. Remaining TODOs
Ordered list of what still needs to be done to complete the original goal.

## 6. Critical Data
Any values the next LLM needs verbatim: IDs, tokens, exact error messages, config values \
discovered during the run.

Be concise. Omit reasoning steps and intermediate attempts. Keep conclusions.\
"""


# 返回当前 UTC 时间的简短时间戳字符串（用于文件名）
def _ts_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class CompactionResult:
    summary_text: str
    original_token_estimate: int
    summary_tokens: int


class Compactor:
    # 初始化压缩器，绑定事件总线、session 目录和 session ID
    def __init__(
        self, bus: EventBus, session_dir: Path, session_id: str,
        *, store: SessionStore | None = None, timeout_s: float = 60.0,
    ) -> None:
        self._bus = bus
        self._session_dir = session_dir
        self._session_id = session_id
        self._store = store
        self._timeout_s = timeout_s

    # 先保存摘要和可选会话历史，再无等待地切换内存上下文，最后发送可隔离异常的完成通知
    async def compact(
        self,
        context: ExecutionContext,
        provider: LLMProvider,
        focus: str = "",
    ) -> CompactionResult | None:
        result = await self.compact_messages(context.messages, provider, focus=focus)
        if result is None:
            return None

        await asyncio.sleep(0)  # 在同步提交前交付待处理的取消，提交区间内不再让出执行权
        new_messages: list[dict[str, Any]] = [
            {"role": "user", "content": result.summary_text},
            {"role": "assistant", "content": "Understood, I'll continue from this summary."},
        ]
        try:
            self._write_summary(result.summary_text)
            if self._store is not None:
                self._store.write_compacted(self._session_id, new_messages)
        except Exception:
            logger.exception("compactor: persistence failed, retaining original history")
            return None
        context.messages = new_messages
        await self._bus.publish(
            ContextCompactedEvent(
                session_id=self._session_id,
                run_id=context.run_id,
                original_tokens=result.original_token_estimate,
                summary_tokens=result.summary_tokens,
                ts=_now(),
            ),
            isolate_errors=True,
        )
        logger.info(
            "context compacted session=%s run=%s original≈%d summary=%d tokens",
            self._session_id, context.run_id,
            result.original_token_estimate, result.summary_tokens,
        )
        return result

    # 纯函数式压缩：接收消息列表，返回 CompactionResult；失败时返回 None
    async def compact_messages(
        self,
        messages: list[dict[str, Any]],
        provider: LLMProvider,
        focus: str = "",
    ) -> CompactionResult | None:
        from x_claude.core.events.bus import EventBus as _Bus

        original_estimate = sum(
            len(str(m.get("content", ""))) for m in messages
        ) // 4  # 粗略 token 估算（字符数 / 4）

        history_text = _messages_to_text(messages)
        prompt = _COMPACT_PROMPT
        if focus.strip():
            prompt += f"\n\nIMPORTANT: Pay special attention to: {focus.strip()}"

        compress_request: list[dict[str, object]] = [
            {"role": "user", "content": f"{prompt}\n\n---\n\n{history_text}"}
        ]

        try:
            silent_bus = _Bus()
            async with asyncio.timeout(self._timeout_s):
                response = await provider.chat(
                    messages=compress_request,
                    tool_schemas=[],
                    bus=silent_bus,
                    run_id="compact",
                    step=0,
                    system="You are a helpful assistant that summarizes conversations.",
                )
        except Exception:
            logger.exception("compactor: LLM call failed, skipping compaction")
            return None

        summary_text = response.text.strip()
        if response.stop_reason != "end_turn" or not summary_text:
            logger.warning("compactor: summary empty or incomplete, skipping compaction")
            return None

        summary_tokens = response.usage.output_tokens if response.usage else len(summary_text) // 4

        return CompactionResult(
            summary_text=summary_text,
            original_token_estimate=original_estimate,
            summary_tokens=summary_tokens,
        )

    # 将摘要完整写入 session 目录，原子提交失败向调用者传播以保留旧上下文
    def _write_summary(self, text: str) -> None:
        path = self._session_dir / f"summary_{_ts_compact()}.md"
        atomic_write_bytes(path, text.encode("utf-8"))


# 将消息列表序列化为可供 LLM 阅读的纯文本
def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role}]\n{content}")
        elif isinstance(content, list):
            blocks: list[str] = []
            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    blocks.append(block.get("text", ""))
                elif btype == "tool_use":
                    blocks.append(
                        f"<tool_call name={block.get('name')} id={block.get('id')}>\n"
                        f"{block.get('input', {})}\n</tool_call>"
                    )
                elif btype == "tool_result":
                    blocks.append(
                        f"<tool_result id={block.get('tool_use_id')}>\n"
                        f"{block.get('content', '')}\n</tool_result>"
                    )
            parts.append(f"[{role}]\n" + "\n".join(blocks))
    return "\n\n".join(parts)
