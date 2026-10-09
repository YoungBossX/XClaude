from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from x_claude.core.agents.loader import AgentProfile, AgentProfileLoader
from x_claude.core.bus.events import (
    SubagentFinishedEvent,
    SubagentRestoredEvent,
    SubagentStartedEvent,
)
from x_claude.core.compact.compactor import Compactor
from x_claude.core.config import XConfig
from x_claude.core.context import ExecutionContext
from x_claude.core.events.bus import EventBus
from x_claude.core.events.writer import EventWriter
from x_claude.core.loop import AgentLoop
from x_claude.core.resources import ResourceLimitExceeded, TaskBudget
from x_claude.core.runs import new_run_id
from x_claude.core.subagent.checkpoint import (
    BackgroundCheckpoint,
    BackgroundRecord,
    ContextSnapshot,
    runtime_signature,
)
from x_claude.core.subagent.registry import BackgroundTaskRegistry
from x_claude.core.tools.base import BaseTool, ToolResult
from x_claude.core.tools.builtin.bash import BashTool
from x_claude.core.tools.builtin.list_dir import ListDirTool
from x_claude.core.tools.builtin.read_file import ReadFileTool
from x_claude.core.tools.builtin.task_create import TaskCreateTool
from x_claude.core.tools.builtin.task_get import TaskGetTool
from x_claude.core.tools.builtin.task_list import TaskListTool
from x_claude.core.tools.builtin.task_update import TaskUpdateTool
from x_claude.core.tools.builtin.write_file import WriteFileTool
from x_claude.core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from x_claude.core.llm.base import LLMProvider
    from x_claude.core.permissions.manager import PermissionManager

_profile_loader = AgentProfileLoader()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SpawnAgentParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    description: str
    prompt: str
    run_in_background: bool = False
    subagent_type: str = ""


# 在隔离的冷启动上下文中派生子 agent，支持前台阻塞和后台并行两种模式
class SpawnAgentTool(BaseTool):
    name = "spawn_agent"
    description = (
        "Spawn an isolated sub-agent to handle a self-contained sub-task. "
        "The sub-agent starts with a clean context containing only the provided prompt — "
        "it does not inherit the current conversation history. "
        "Use run_in_background=true to run in parallel; retrieve result later with agent_result."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "3-5 word task description shown in progress display",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Complete task description including all context the sub-agent needs. "
                    "The sub-agent cannot see the parent conversation, so be explicit."
                ),
            },
            "run_in_background": {
                "type": "boolean",
                "description": "When true, returns immediately with a run_id; use agent_result to poll.",  # noqa: E501
            },
            "subagent_type": {
                "type": "string",
                "description": "Agent role profile (planner/executor/reviewer). Leave empty for default.",  # noqa: E501
            },
        },
        "required": ["description", "prompt"],
    }
    params_model = SpawnAgentParams

    # 构造 SpawnAgentTool；depth=0 表示根 agent，最大允许嵌套深度为 2
    def __init__(
        self,
        provider: LLMProvider | None,
        parent_bus: EventBus,
        parent_run_id: str,
        permission_manager: PermissionManager | None,
        max_steps: int,
        task_registry: BackgroundTaskRegistry,
        runs_dir: Path,
        session_id: str,
        depth: int = 0,
        config: XConfig | None = None,
        mcp_tools: Sequence[BaseTool] | None = None,
        budget: TaskBudget | None = None,
    ) -> None:
        self._provider = provider
        self._parent_bus = parent_bus
        self._parent_run_id = parent_run_id
        self._permission_manager = permission_manager
        self._max_steps = max_steps
        self._task_registry = task_registry
        self._task_registry.bind_storage(runs_dir, session_id)
        self._runs_dir = runs_dir
        self._session_id = session_id
        self._depth = depth
        self._config = config or XConfig()
        self._mcp_tools = list(mcp_tools or [])
        self._budget = budget

    # 派生子 agent，前台时阻塞直到完成并返回结果，后台时立即返回 run_id
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = SpawnAgentParams.model_validate(params)

        if self._depth >= 2:
            return ToolResult(
                content="Subagent nesting limit (2) reached; cannot spawn further subagents.",
                is_error=True,
                error_type="runtime_error",
            )

        profile: AgentProfile | None = None
        if p.subagent_type:
            profile = _profile_loader.load(p.subagent_type)

        child_run_id = new_run_id()
        if self._budget is not None:
            try:
                self._budget.register(child_run_id)
            except ResourceLimitExceeded as exc:
                return ToolResult(content=f"{exc.code}: {exc}", is_error=True,
                                  error_type="runtime_error")
        child_context = ExecutionContext(
            run_id=child_run_id,
            goal=p.prompt,
            max_steps=self._max_steps,
            system_prompt_override=profile.system_prompt if profile else None,
        )

        child_bus = EventBus()

        # 将子 bus 所有事件桥接到父 bus，TUI 据此渲染嵌套进度
        async def _bridge(event: BaseModel) -> None:
            await self._parent_bus.publish(event)

        child_bus.subscribe(_bridge)

        child_registry = self._build_child_registry(child_bus, child_run_id, profile)
        child_run_path = self._runs_dir / child_run_id
        child_run_path.mkdir(parents=True, exist_ok=True)
        checkpoint = None
        if p.run_in_background:
            checkpoint = BackgroundCheckpoint(child_run_path / "background.json", BackgroundRecord(
                run_id=child_run_id, parent_run_id=self._parent_run_id,
                session_id=self._session_id, description=p.description,
                cwd=str(Path.cwd().resolve()), depth=self._depth,
                tools=[str(schema["name"]) for schema in child_registry.tool_schemas()],
                runtime_signature=runtime_signature(self._config, child_registry.tool_schemas()),
                model=self._config.llm.default_model,
                budget_root_id=self._budget.root_id if self._budget is not None else "",
                context=ContextSnapshot.capture(child_context),
            ))
            checkpoint.save(child_context, "ready")
        assert self._provider is not None
        child_loop = AgentLoop(
            self._provider,
            child_registry,
            child_bus,
            permission_manager=self._permission_manager,
            session_id=self._session_id,
            compactor=Compactor(
                child_bus, child_run_path, self._session_id,
                tool_result_limit=self._config.compaction.tool_result_limit,
                tool_result_keep=self._config.compaction.tool_result_keep,
            ),
            compact_threshold=self._config.compaction.auto_threshold,
            tool_result_limit=self._config.compaction.tool_result_limit,
            tool_result_keep=self._config.compaction.tool_result_keep,
            context_window=self._config.llm.context_window or 200_000,
            checkpoint=checkpoint.save if checkpoint is not None else None,
            budget=self._budget,
        )

        await self._parent_bus.publish(
            SubagentStartedEvent(
                run_id=child_run_id,
                parent_run_id=self._parent_run_id,
                description=p.description,
                session_id=self._session_id,
                ts=_now(),
            )
        )

        if p.run_in_background:
            task: asyncio.Task[None] = asyncio.create_task(
                self._run_background(
                    child_loop, child_context, child_bus, child_run_path, child_run_id,
                    checkpoint=checkpoint,
                )
            )
            self._task_registry.register(child_run_id, task, child_context)
            return ToolResult(
                content=(
                    f"Subagent started in background. run_id={child_run_id}. "
                    f"Use agent_result(run_id='{child_run_id}') to retrieve result."
                )
            )

        await self._run_background(
            child_loop, child_context, child_bus, child_run_path, child_run_id,
        )

        if child_context.status == "success":
            return ToolResult(
                content=child_context.result or "Subagent completed with no text output."
            )
        return ToolResult(
            content=(
                child_context.result
                or f"Subagent failed (status={child_context.status}, reason={child_context.reason})"
            ),
            is_error=True,
            error_type="runtime_error",
        )

    # 后台任务协程：写事件文件，运行 loop，发布完成事件
    async def _run_background(
        self,
        loop: AgentLoop,
        context: ExecutionContext,
        bus: EventBus,
        run_path: Path,
        run_id: str,
        *, checkpoint: BackgroundCheckpoint | None = None,
        start_gate: asyncio.Event | None = None,
    ) -> None:
        if start_gate is not None:
            await start_gate.wait()
        async with EventWriter(run_path / "events.jsonl") as writer:
            writer.subscribe(bus)
            try:
                await loop.run(context)
                if checkpoint is not None:
                    if checkpoint.record.phase == "tools" and checkpoint.pending_tools():
                        checkpoint.interrupt(suspend=True)
                    else:
                        checkpoint.save(context, "finished")
            except asyncio.CancelledError:
                suspend = self._task_registry.suspending
                context.mark_failed("suspended" if suspend else "cancelled")
                if checkpoint is not None:
                    checkpoint.interrupt(suspend=suspend)
                raise
            except Exception:
                context.mark_failed("subagent_error")
                logging.getLogger(__name__).exception("subagent failed run_id=%s", run_id)
                if checkpoint is not None:
                    try:
                        checkpoint.save(context, "finished")
                    except Exception:
                        logging.getLogger(__name__).exception("checkpoint failed run_id=%s", run_id)
            finally:
                await bus.publish(SubagentFinishedEvent(
                    run_id=run_id, parent_run_id=self._parent_run_id,
                    status="suspended" if context.reason == "suspended" else context.status,
                    session_id=self._session_id, ts=_now(),
                ), isolate_errors=True)

    # 从安全检查点续跑；工具阶段不自动重放，已完成或取消的结果只重建查询索引
    async def restore(
        self, checkpoint: BackgroundCheckpoint,
        *, provider_factory: Callable[[], LLMProvider],
        start_gate: asyncio.Event | None = None,
    ) -> None:
        record = checkpoint.record
        context = record.context.restore(record.run_id)
        message = "已恢复保存的任务结果"
        terminal = record.state in ("completed", "cancelled") or context.is_done()
        blocked = ""
        child_bus = EventBus()
        registry = self._build_child_registry(
            child_bus, record.run_id, None, allowed_tools=record.tools,
        )
        signature = runtime_signature(self._config, registry.tool_schemas())
        if not terminal and record.phase == "tools" and checkpoint.pending_tools():
            blocked = (
                "interrupted_tool: 工具结果尚未确认，请核对文件、Shell 或 MCP 操作；不会自动重放"
            )
        elif not terminal and Path(record.cwd).resolve() != Path.cwd().resolve():
            blocked = f"workspace_mismatch: 请在原工作目录恢复会话：{record.cwd}"
        elif not terminal and record.runtime_signature != signature:
            blocked = (
                "config_changed: 模型、窗口、服务地址或工具契约变化（旧模型："
                + (record.model or "未记录") + "），请用 /recover 核对并明确接受当前配置"
            )
        if blocked:
            context.mark_failed("needs_review: " + blocked)
            message = blocked
        if terminal or blocked:
            future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            future.set_result(None)
            self._task_registry.register(record.run_id, future, context)
            state = "blocked" if blocked else record.state
        else:
            if record.phase == "tools":
                checkpoint.save(context, "ready")
            try:
                if self._provider is None:
                    self._provider = provider_factory()
            except (Exception, SystemExit) as exc:
                context.mark_failed(f"recovery_provider_error: {exc}")
                future = asyncio.get_running_loop().create_future()
                future.set_result(None)
                self._task_registry.register(record.run_id, future, context)
                await self._parent_bus.publish(SubagentRestoredEvent(
                    run_id=record.run_id, session_id=self._session_id,
                    parent_run_id=record.parent_run_id, description=record.description,
                    state="blocked", step=context.step, message=context.reason or "", ts=_now(),
                ), isolate_errors=True)
                return
            # 验证通过且模型就绪后重建工具，嵌套派生必须继承恢复后的模型和预算
            registry = self._build_child_registry(
                child_bus, record.run_id, None, allowed_tools=record.tools,
            )
            context.status = "running"
            context.reason = None

            # 续跑事件只桥接到该会话的 daemon 总线，显式携带会话身份
            async def bridge(event: BaseModel) -> None:
                await self._parent_bus.publish(event, isolate_errors=True)

            child_bus.subscribe(bridge)
            loop = AgentLoop(
                self._provider, registry, child_bus,
                permission_manager=self._permission_manager, session_id=self._session_id,
                compactor=Compactor(
                    child_bus, checkpoint.path.parent, self._session_id,
                    tool_result_limit=self._config.compaction.tool_result_limit,
                    tool_result_keep=self._config.compaction.tool_result_keep,
                ),
                compact_threshold=self._config.compaction.auto_threshold,
                tool_result_limit=self._config.compaction.tool_result_limit,
                tool_result_keep=self._config.compaction.tool_result_keep,
                context_window=self._config.llm.context_window or 200_000,
                checkpoint=checkpoint.save,
                budget=self._budget,
            )
            await self._parent_bus.publish(SubagentStartedEvent(
                run_id=record.run_id, parent_run_id=record.parent_run_id,
                description=record.description, session_id=self._session_id,
                resumed=True, ts=_now(),
            ), isolate_errors=True)
            task = asyncio.create_task(self._run_background(
                loop, context, child_bus, checkpoint.path.parent, record.run_id,
                checkpoint=checkpoint, start_gate=start_gate,
            ))
            self._task_registry.register(record.run_id, task, context)
            state, message = "running", "从最后已确认步骤续跑，不重放已完成工具"
        await self._parent_bus.publish(SubagentRestoredEvent(
            run_id=record.run_id, session_id=self._session_id,
            parent_run_id=record.parent_run_id, description=record.description,
            state=state, step=context.step, message=message, ts=_now(),
        ), isolate_errors=True)

    # 构造子 registry；基于角色配置过滤工具，深度允许时注册嵌套 SpawnAgentTool
    def _build_child_registry(
        self,
        child_bus: EventBus,
        child_run_id: str,
        profile: AgentProfile | None,
        *, allowed_tools: list[str] | None = None,
    ) -> ToolRegistry:
        from x_claude.core.task.manager import TaskManager

        allowed: set[str] | None = set(allowed_tools) if allowed_tools is not None else (
            set(profile.allowed_tools) if profile and profile.allowed_tools else None
        )

        def _allowed(name: str) -> bool:
            return allowed is None or name in allowed

        registry = ToolRegistry()
        _all_tools = [
            ReadFileTool(),
            BashTool(),
            WriteFileTool(),
            ListDirTool(),
        ]
        for t in _all_tools:
            if _allowed(t.name):
                registry.register(t)
        for tool in self._mcp_tools:
            if _allowed(tool.name):
                registry.register(tool)

        child_task_manager = TaskManager(self._runs_dir / child_run_id / ".tasks")
        for t in [
            TaskCreateTool(child_task_manager),
            TaskUpdateTool(child_task_manager),
            TaskListTool(child_task_manager),
            TaskGetTool(child_task_manager),
        ]:
            if _allowed(t.name):
                registry.register(t)

        if self._depth < 1:
            nested = SpawnAgentTool(
                provider=self._provider,
                parent_bus=child_bus,
                parent_run_id=child_run_id,
                permission_manager=self._permission_manager,
                max_steps=self._max_steps,
                task_registry=self._task_registry,
                runs_dir=self._runs_dir,
                session_id=self._session_id,
                depth=self._depth + 1,
                config=self._config,
                mcp_tools=self._mcp_tools,
                budget=self._budget,
            )
            if _allowed("spawn_agent"):
                registry.register(nested)
            if _allowed("agent_result"):
                registry.register(AgentResultTool(self._task_registry))

        return registry


class AgentResultParams(BaseModel):
    run_id: str


# 查询后台 subagent 的执行状态和最终结果
class AgentResultTool(BaseTool):
    name = "agent_result"
    description = (
        "Retrieve the result of a background sub-agent previously started with spawn_agent. "
        "Returns 'still running' if the sub-agent has not yet completed."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run_id returned by spawn_agent(run_in_background=true)",
            },
        },
        "required": ["run_id"],
    }
    params_model = AgentResultParams

    # 初始化，持有共享的后台任务注册表
    def __init__(self, task_registry: BackgroundTaskRegistry) -> None:
        self._task_registry = task_registry

    # 查询指定 run_id 的后台任务状态，返回结果或错误
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = AgentResultParams.model_validate(params)
        entry = self._task_registry.get(p.run_id)
        if entry is None:
            return ToolResult(
                content=f"Unknown run_id: {p.run_id}. Only background subagents can be queried.",
                is_error=True,
                error_type="runtime_error",
            )
        task, context = entry
        if not task.done():
            return ToolResult(content="still running")
        if task.cancelled():
            return ToolResult(
                content="Subagent was cancelled.", is_error=True, error_type="runtime_error"
            )
        exc = task.exception()
        if exc is not None:
            return ToolResult(
                content=f"Subagent raised an exception: {exc}",
                is_error=True,
                error_type="runtime_error",
            )
        if context.status != "success":
            return ToolResult(
                content=f"Subagent failed: {context.reason or context.status}",
                is_error=True, error_type="runtime_error",
            )
        return ToolResult(content=context.result or "Subagent completed with no text result.")
