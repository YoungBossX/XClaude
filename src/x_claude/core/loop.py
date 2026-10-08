from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from x_claude.core.bus.events import StepFinishedEvent, StepStartedEvent
from x_claude.core.compact.budget import estimate_tokens, truncate_tool_results
from x_claude.core.context import ExecutionContext
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.base import LLMProvider
from x_claude.core.tools.invocation import invoke_tool
from x_claude.core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from x_claude.core.compact.compactor import Compactor
    from x_claude.core.permissions.manager import PermissionManager
    from x_claude.core.subagent.checkpoint import Phase


log = logging.getLogger(__name__)

def _now() -> str:
    return datetime.now(UTC).isoformat()


class AgentLoop:
    # 初始化循环依赖及可选的权限管理器、压缩器和 session ID
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        bus: EventBus,
        *,
        permission_manager: PermissionManager | None = None,
        compactor: Compactor | None = None,
        compact_threshold: float = 0.80,
        session_id: str = "",
        tool_result_limit: int = 8_000,
        tool_result_keep: int = 4_000,
        context_window: int = 0,
        checkpoint: Callable[[ExecutionContext, Phase], None] | None = None,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._bus = bus
        self._permission_manager = permission_manager
        self._compactor = compactor
        self._compact_threshold = compact_threshold
        self._session_id = session_id
        self._tool_result_limit = tool_result_limit
        self._tool_result_keep = tool_result_keep
        self._context_window = context_window
        self._checkpoint = checkpoint

    # 持久任务在推理前、工具执行前、每项观察后及完整步骤后保存检查点
    def _save_checkpoint(self, context: ExecutionContext, phase: Phase) -> None:
        if self._checkpoint is not None:
            self._checkpoint(context, phase)

    # 驱动 plan→act→observe 循环直到上下文终止；CancelledError 向上传播
    async def run(self, context: ExecutionContext) -> None:
        while not context.is_done():
            if context.step >= context.max_steps:
                context.mark_failed("exceeded_max_steps")
                self._save_checkpoint(context, "finished")
                break
            system = context.system_prompt(
                "You are a helpful AI assistant. "
                "Use the available tools to complete the user's goal. "
                "When the goal is fully achieved, respond with a final answer "
                "and do not call any more tools."
            )
            schemas = self._registry.tool_schemas()
            request_messages = truncate_tool_results(
                context.messages, self._tool_result_limit, self._tool_result_keep,
            )
            # 冷启动和跨轮续接先检查保守预算，避免等模型拒绝请求后才尝试压缩
            if (self._context_window > 0 and self._compact_threshold > 0
                    and self._compactor is not None
                    and estimate_tokens([request_messages, system, schemas])
                    >= self._context_window * self._compact_threshold):
                await self._compactor.compact(context, self._provider)
                request_messages = truncate_tool_results(
                    context.messages, self._tool_result_limit, self._tool_result_keep,
                )
            prefill_len = len(context.messages)
            self._save_checkpoint(context, "planning")
            context.step += 1
            await self._bus.publish(
                StepStartedEvent(run_id=context.run_id, step=context.step, ts=_now())
            )

            # [plan] call LLM — API errors terminate the run
            try:
                response = await self._provider.chat(
                    messages=request_messages,
                    tool_schemas=schemas,
                    bus=self._bus,
                    run_id=context.run_id,
                    step=context.step,
                    system=system,
                )
            except asyncio.CancelledError:
                context.mark_failed("cancelled")
                raise
            except Exception:
                logging.getLogger(__name__).exception(
                    "LLM call failed run_id=%s step=%d", context.run_id, context.step
                )
                context.mark_failed("llm_error")
                self._save_checkpoint(context, "finished")
                break

            # [observe] append assistant content blocks to context
            # thinking blocks must come first and be preserved verbatim for extended thinking mode
            blocks: list[dict[str, object]] = list(response.thinking_blocks)
            if response.text:
                blocks.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                )
            context.add_assistant_message(blocks)

            # [act] execute each requested tool; errors become tool results so loop continues
            if response.stop_reason == "tool_use":
                # 中断于此阶段时工具效果可能不确定，恢复不得自动重放该批调用
                self._save_checkpoint(context, "tools")
                for tc in response.tool_calls:
                    result = await invoke_tool(
                        self._registry, tc, self._bus, context.run_id,
                        permission_manager=self._permission_manager,
                        session_id=self._session_id,
                        file_versions=context.file_versions,
                    )
                    context.add_tool_result(tc.id, result.content, is_error=result.is_error)
                    # 逐项确认已返回的结果；若提交失败，禁止继续执行后面的工具
                    self._save_checkpoint(context, "tools")
            elif response.stop_reason == "max_tokens" and response.tool_calls:
                # Output token limit hit mid-tool-call; input is incomplete.
                # Add synthetic error results so the conversation stays balanced.
                for tc in response.tool_calls:
                    context.add_tool_result(
                        tc.id,
                        "Error: output token limit reached before this tool call "
                        "could be completed. "
                        "Please break the task into smaller steps and try again.",
                        is_error=True,
                    )

            # Termination check — end_turn wins over max_steps if both hit on same step
            if response.stop_reason == "end_turn":
                context.result = response.text or ""
                context.mark_success()
            elif context.step >= context.max_steps:
                context.mark_failed("exceeded_max_steps")

            # 工具结果追加完毕（messages 末尾为 user）后检查压缩，仅在 run 继续时触发
            # 此时压缩结果 [user_summary, assistant_ack] 对下一次 LLM 调用是合法输入
            next_context_pct = response.usage.context_pct if response.usage else 0.0
            if response.usage and response.usage.context_window:
                # usage 表示上次请求输入；把本步输出和截断后的新工具结果计入下一步预算
                delta = truncate_tool_results(
                    context.messages[prefill_len + 1:],
                    self._tool_result_limit, self._tool_result_keep,
                )
                next_context_pct += (
                    response.usage.output_tokens + estimate_tokens(delta)
                ) / response.usage.context_window
            if (
                not context.is_done()
                and response.stop_reason == "tool_use"
                and self._compactor is not None
                and self._compact_threshold > 0
                and response.usage is not None
                and next_context_pct >= self._compact_threshold
            ):
                await self._compactor.compact(context, self._provider)

            self._save_checkpoint(context, "finished" if context.is_done() else "ready")
            await self._bus.publish(
                StepFinishedEvent(run_id=context.run_id, step=context.step, ts=_now())
            )
