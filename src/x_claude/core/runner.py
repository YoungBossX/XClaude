from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from weakref import WeakValueDictionary

from pydantic import BaseModel

from x_claude.core.bus.commands import BackgroundTaskInfo, RecoveredToolResult
from x_claude.core.bus.events import (
    RunFinishedEvent,
    RunRestoredEvent,
    RunStartedEvent,
    SubagentRestoredEvent,
)
from x_claude.core.compact.compactor import Compactor
from x_claude.core.config import XConfig
from x_claude.core.context import ExecutionContext
from x_claude.core.events.bus import EventBus, EventHandler
from x_claude.core.events.writer import EventWriter
from x_claude.core.llm.base import LLMProvider
from x_claude.core.llm.provider import AnthropicProvider, _context_window
from x_claude.core.loop import AgentLoop
from x_claude.core.mcp.server import McpServerManager
from x_claude.core.memory.loader import load_context_file
from x_claude.core.permissions.manager import PermissionManager
from x_claude.core.resources import LimitedProvider, TaskBudget, budget_root
from x_claude.core.runs import RUNS_DIR, new_run_id
from x_claude.core.session.model import Session
from x_claude.core.session.store import SessionStore
from x_claude.core.subagent.checkpoint import (
    BackgroundCheckpoint,
    BackgroundRecord,
    ContextSnapshot,
    runtime_signature,
)
from x_claude.core.subagent.registry import BackgroundTaskRegistry
from x_claude.core.subagent.tool import AgentResultTool, SpawnAgentTool
from x_claude.core.task.manager import TaskManager
from x_claude.core.tools.builtin import (
    BashTool,
    ListDirTool,
    NoteSaveTool,
    ReadFileTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
    WriteFileTool,
)
from x_claude.core.tools.registry import ToolRegistry
from x_claude.core.trace.provider import TracingProvider
from x_claude.core.trace.writer import TraceWriter


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class RunOutcome:
    status: str
    result: str
    reason: str | None
    session_persisted: bool = False


class AgentRunner:
    # 组装所有运行时依赖，准备执行一次完整的 agent run
    def __init__(
        self,
        config: XConfig,
        *,
        bus: EventBus | None = None,
        provider: LLMProvider | None = None,
        extra_handlers: list[EventHandler] | None = None,
        runs_dir: Path | None = None,
        trace: TraceWriter | None = None,
        permission_manager: PermissionManager | None = None,
        mcp_manager: McpServerManager | None = None,
        model_gate: asyncio.Semaphore | None = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._provider = provider
        self._extra_handlers: list[EventHandler] = extra_handlers or []
        self._runs_dir = runs_dir or RUNS_DIR
        self._trace = trace
        self._permission_manager = permission_manager
        self._mcp_manager = mcp_manager
        self._model_gate = model_gate or asyncio.Semaphore(config.agent.max_concurrent_llm)
        self._budgets: WeakValueDictionary[str, TaskBudget] = WeakValueDictionary()
        # 跨 run 共享的后台 subagent 任务注册表
        self._task_registry = BackgroundTaskRegistry()
        self._recovery_lock = asyncio.Lock()
        self._suspend_roots = False
        self._root_errors: dict[str, str] = {}
        self._root_checkpoint: BackgroundCheckpoint | None = None

    # 在取消主循环之前明确区分 daemon 挂起与用户永久取消
    def prepare_shutdown(self, *, suspend: bool) -> None:
        self._suspend_roots = suspend

    # 查找唯一未提交的主任务，完成但尚未写回历史的任务也需要恢复提交
    def pending_root(self, session: Session, store: SessionStore) -> BackgroundCheckpoint | None:
        pending = []
        for path in store.runs_dir(session.id).glob("*/root.json"):
            checkpoint = BackgroundCheckpoint.load(path, session.id)
            record = checkpoint.record
            if not record.thread_committed and record.state != "cancelled":
                pending.append(checkpoint)
        if len(pending) > 1:
            raise ValueError("multiple unfinished root checkpoints; review required")
        self._root_checkpoint = pending[0] if pending else None
        return self._root_checkpoint

    # 重建主任务原工具白名单的契约，不为检查配置而初始化真实模型客户端
    def root_schemas(
        self, session: Session, store: SessionStore, record: BackgroundRecord,
    ) -> list[dict[str, Any]]:
        registry = self._build_registry(
            TaskManager(store.runs_dir(session.id) / record.run_id / ".tasks"),
            session=session, store=store, run_id=record.run_id,
            provider=cast(LLMProvider, object()), bus=EventBus(),
            child_runs_dir=store.runs_dir(session.id), session_id=session.id,
            tool_whitelist=record.tools,
        )
        return registry.tool_schemas()

    # 校验主任务安全边界并通知前端，未确认工具、目录或配置变化不会自动执行
    async def root_can_resume(
        self, checkpoint: BackgroundCheckpoint, session: Session, store: SessionStore,
    ) -> bool:
        record = checkpoint.record
        problem = ""
        try:
            if record.context.status == "running":
                if record.phase == "tools" and checkpoint.pending_tools():
                    problem = "interrupted_tool: 请用 /recover 核对未确认的工具结果"
                elif Path(record.cwd).resolve() != Path.cwd().resolve():
                    problem = f"workspace_mismatch: 请在原目录恢复：{record.cwd}"
                elif record.runtime_signature != runtime_signature(
                    self._config, self.root_schemas(session, store, record),
                ):
                    problem = "config_changed: 请核对配置并通过 /recover --accept-config 确认"
                else:
                    self._get_provider()
                    if record.phase == "tools":
                        checkpoint.save(record.context.restore(record.run_id), "ready")
        except (Exception, SystemExit) as exc:
            problem = f"recovery_error: {exc}"
        if problem:
            self._root_errors[record.run_id] = problem
        else:
            self._root_errors.pop(record.run_id, None)
        if self._bus is not None:
            await self._bus.publish(RunRestoredEvent(
                run_id=record.run_id, session_id=session.id,
                state="blocked" if problem else "running",
                step=record.context.step, message=problem or "从主任务最后已确认状态续跑",
                ts=_now(),
            ), isolate_errors=True)
        return not problem

    # 会话结束或 daemon 退出时回收后台子 Agent
    async def shutdown(self, *, suspend: bool = False) -> None:
        async with self._recovery_lock:
            await self._task_registry.cancel_all(suspend=suspend)
            if self._root_checkpoint is not None:
                checkpoint = BackgroundCheckpoint.load(
                    self._root_checkpoint.path, self._root_checkpoint.record.session_id,
                )
                checkpoint.interrupt(suspend=suspend)
                self._root_checkpoint = checkpoint

    # 惰性创建或装饰模型客户端，恢复已完成结果时不要求创建模型客户端
    def _get_provider(self, budget: TaskBudget | None = None) -> LLMProvider:
        provider: LLMProvider = self._provider or AnthropicProvider(
            self._config.llm.default_model, context_window=self._config.llm.context_window,
        )
        if self._trace is not None:
            provider = TracingProvider(
                provider, self._trace, include_payload=self._config.trace.include_llm_payload,
            )
        return (LimitedProvider(provider, self._model_gate, budget)
                if budget is not None else provider)

    # 同一根任务的前台和后台子任务共用一份预算对象，重启时从原子账本恢复
    def _budget_for(self, run_id: str, runs_dir: Path, parent_id: str = "") -> TaskBudget:
        root_id = budget_root(runs_dir, run_id, parent_id)
        budget = self._budgets.get(root_id)
        if budget is None:
            budget = TaskBudget(
                root_id, self._config.agent, runs_dir / root_id / "budget.json",
            )
            self._budgets[root_id] = budget
        return budget

    # 恢复本会话保存的后台任务，串行登记避免重连与新消息同时触发重复续跑
    async def restore_background(self, session: Session, store: SessionStore) -> None:
        async with self._recovery_lock:
            gate = asyncio.Event()
            original = {key: value[0] for key, value in self._task_registry._tasks.items()}
            try:
                await self._restore_background_batch(session, store, gate)
            except BaseException:
                new = {key: value[0] for key, value in self._task_registry._tasks.items()
                       if original.get(key) is not value[0]}
                for task in new.values():
                    task.cancel()
                await asyncio.gather(*new.values(), return_exceptions=True)
                for key in new:
                    self._task_registry.forget_stopped(key)
                raise
            gate.set()

    # 在放行屏障之前登记整个恢复批次，取消时由外层撤销尚未启动的任务
    async def _restore_background_batch(
        self, session: Session, store: SessionStore, start_gate: asyncio.Event,
    ) -> None:
        runs_dir = store.runs_dir(session.id)
        self._task_registry.bind_storage(runs_dir, session.id)
        bus = self._bus or EventBus()
        for path in sorted(runs_dir.glob("*/background.json")):
            entry = self._task_registry.get(path.parent.name)
            if entry is not None:
                retryable = (entry[1].reason or "").startswith((
                    "checkpoint_invalid:", "recovery_provider_error:", "recovery_error:",
                    "needs_review: workspace_mismatch:",
                ))
                if not entry[0].done() or not retryable:
                    continue
                self._task_registry.forget_stopped(path.parent.name)
            try:
                checkpoint = BackgroundCheckpoint.load(path, session.id)
            except (OSError, ValueError):
                logging.getLogger(__name__).exception("invalid checkpoint path=%s", path)
                context = ExecutionContext(path.parent.name, "", 1)
                context.mark_failed("checkpoint_invalid: 无法读取任务检查点，不会自动重放")
                future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                future.set_result(None)
                self._task_registry.register(context.run_id, future, context)
                await bus.publish(SubagentRestoredEvent(
                    run_id=context.run_id, session_id=session.id, parent_run_id="",
                    description="检查点损坏", state="blocked", step=0,
                    message=context.reason or "checkpoint_invalid", ts=_now(),
                ), isolate_errors=True)
                continue
            record = checkpoint.record
            try:
                budget = (self._budget_for(record.run_id, runs_dir, record.parent_run_id)
                          if record.context.status == "running" else None)
                tool = SpawnAgentTool(
                    provider=None, parent_bus=bus, parent_run_id=record.parent_run_id,
                    permission_manager=self._permission_manager,
                    max_steps=record.context.max_steps, task_registry=self._task_registry,
                    runs_dir=runs_dir, session_id=session.id, depth=record.depth,
                    config=self._config,
                    mcp_tools=self._mcp_manager.get_tools() if self._mcp_manager else [],
                    budget=budget,
                )
                await tool.restore(
                    checkpoint, provider_factory=lambda: self._get_provider(budget),
                    start_gate=start_gate,
                )
            except (Exception, SystemExit) as exc:
                logging.getLogger(__name__).exception(
                    "cannot restore task run_id=%s", record.run_id,
                )
                context = record.context.restore(record.run_id)
                context.mark_failed(f"recovery_error: {exc}")
                future = asyncio.get_running_loop().create_future()
                future.set_result(None)
                self._task_registry.register(record.run_id, future, context)
                await bus.publish(SubagentRestoredEvent(
                    run_id=record.run_id, session_id=session.id,
                    parent_run_id=record.parent_run_id, description=record.description,
                    state="blocked", step=context.step, message=context.reason or "", ts=_now(),
                ), isolate_errors=True)

    # 人工确认任务的配置及工具观察结果，原子提交后再走正常恢复路径
    async def recover_task(
        self, session: Session, store: SessionStore, run_id: str,
        results: dict[str, str | RecoveredToolResult] | None, accept_config: bool,
    ) -> None:
        async with self._recovery_lock:
            entry = self._task_registry.get(run_id)
            if entry is not None and not entry[0].done():
                raise ValueError("background task is still running")
            path = store.runs_dir(session.id) / run_id
            checkpoint = BackgroundCheckpoint.load(
                path / ("root.json" if (path / "root.json").exists() else "background.json"),
                session.id,
            )
            record = checkpoint.record
            if record.context.status != "running":
                raise ValueError("completed or cancelled task cannot be resumed")
            if Path(record.cwd).resolve() != Path.cwd().resolve():
                raise ValueError(f"resume in the original workspace: {record.cwd}")
            tool = SpawnAgentTool(
                self._provider, self._bus or EventBus(), record.parent_run_id,
                self._permission_manager, record.context.max_steps, self._task_registry,
                store.runs_dir(session.id), session.id, record.depth, config=self._config,
                mcp_tools=self._mcp_manager.get_tools() if self._mcp_manager else [],
            )
            schemas = (self.root_schemas(session, store, record) if record.kind == "root" else
                       tool._build_child_registry(
                           EventBus(), run_id, None, allowed_tools=record.tools,
                       ).tool_schemas())
            if {str(s["name"]) for s in schemas} != set(record.tools):
                raise ValueError("restore missing tools before accepting configuration changes")
            signature = runtime_signature(self._config, schemas)
            if record.runtime_signature != signature and not accept_config:
                raise ValueError(
                    "configuration changed; explicitly use --accept-config after review"
                )
            if record.phase == "tools" and checkpoint.pending_tools():
                if results is None:
                    raise ValueError(
                        "verify every interrupted tool and provide its result; "
                        "tools will not replay"
                    )
                checkpoint.confirm_tools(
                    results, signature=signature, model=self._config.llm.default_model,
                )
            elif results is not None:
                raise ValueError("this checkpoint has no pending tools")
            elif accept_config:
                checkpoint._commit(record.model_copy(update={
                    "runtime_signature": signature, "model": self._config.llm.default_model,
                }))
            self._task_registry.forget_stopped(run_id)
        await self.restore_background(session, store)

    # 返回本会话的任务检查点及等待核对信息，不跨会话搜索任务
    def background_status(self, session: Session, store: SessionStore) -> list[BackgroundTaskInfo]:
        tasks: list[BackgroundTaskInfo] = []
        runs_dir = store.runs_dir(session.id)
        paths = list(runs_dir.glob("*/background.json")) + list(runs_dir.glob("*/root.json"))
        for path in sorted(paths):
            try:
                checkpoint = BackgroundCheckpoint.load(path, session.id)
                record = checkpoint.record
                entry = self._task_registry.get(record.run_id)
                reason = (self._root_errors.get(record.run_id) if record.kind == "root" else
                          entry[1].reason if entry is not None else None)
                state = "blocked" if reason and record.context.status == "running" else record.state
                tasks.append(BackgroundTaskInfo(
                    run_id=record.run_id, kind=record.kind, state=state,
                    phase=record.phase, step=record.context.step, message=reason or "",
                    pending_tools=checkpoint.pending_tools(),
                ))
            except (ValueError, OSError) as exc:
                tasks.append(BackgroundTaskInfo(
                    run_id=path.parent.name,
                    kind="root" if path.name == "root.json" else "background",
                    state="blocked", phase="unknown", step=0,
                    message=f"invalid checkpoint: {exc}",
                ))
        return tasks

    # 构建工具注册表，注入 TaskManager（任务工具共享同一实例）；可选注入 SpawnAgentTool
    def _build_registry(
        self,
        task_manager: TaskManager,
        *,
        session: Session | None = None,
        store: SessionStore | None = None,
        run_id: str | None = None,
        provider: LLMProvider | None = None,
        bus: EventBus | None = None,
        child_runs_dir: Path | None = None,
        session_id: str = "",
        tool_whitelist: list[str] | None = None,
        budget: TaskBudget | None = None,
    ) -> ToolRegistry:
        allowed: set[str] | None = set(tool_whitelist) if tool_whitelist is not None else None

        def _ok(name: str) -> bool:
            return allowed is None or name in allowed

        registry = ToolRegistry()
        for t in [ReadFileTool(), BashTool(), WriteFileTool(), ListDirTool()]:
            if _ok(t.name):
                registry.register(t)
        for t in [
            TaskCreateTool(task_manager),
            TaskUpdateTool(task_manager),
            TaskListTool(task_manager),
            TaskGetTool(task_manager),
        ]:
            if _ok(t.name):
                registry.register(t)
        if session is not None and store is not None and run_id is not None:
            note_tool = NoteSaveTool(store, session.id, run_id)
            if _ok(note_tool.name):
                registry.register(note_tool)
        if provider is not None and bus is not None and run_id is not None:
            runs_dir = child_runs_dir or self._runs_dir
            if _ok("spawn_agent"):
                registry.register(
                    SpawnAgentTool(
                        provider=provider,
                        parent_bus=bus,
                        parent_run_id=run_id,
                        permission_manager=self._permission_manager,
                        max_steps=self._config.agent.max_steps,
                        budget=budget,
                        task_registry=self._task_registry,
                        runs_dir=runs_dir,
                        session_id=session_id,
                        depth=0,
                        config=self._config,
                        mcp_tools=self._mcp_manager.get_tools() if self._mcp_manager else [],
                    )
                )
            if _ok("agent_result"):
                registry.register(AgentResultTool(self._task_registry))
        if self._mcp_manager is not None:
            for mcp_tool in self._mcp_manager.get_tools():
                if _ok(mcp_tool.name):
                    registry.register(mcp_tool)
        return registry

    # 执行一次完整的 agent run（委托给 run_and_capture，忽略返回值）
    async def run(self, goal: str, *, run_id: str | None = None) -> None:
        await self.run_and_capture(goal, run_id=run_id)

    # 执行 agent run 并返回 RunOutcome（含最终文字结果）
    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str | None = None,
        session: Session | None = None,
        store: SessionStore | None = None,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
        checkpoint: BackgroundCheckpoint | None = None,
    ) -> RunOutcome:
        resuming = checkpoint is not None
        run_id = run_id or new_run_id()
        if session is not None and store is not None:
            run_path = store.runs_dir(session.id) / run_id
            history = store.read_messages(session.id, truncate=False)
            notes = store.read_notes(session.id)
        else:
            run_path = self._runs_dir / run_id
            history = [{"role": "user", "content": goal}]
            notes = ""
        run_path.mkdir(parents=True, exist_ok=True)
        budget = (self._budget_for(run_id, run_path.parent)
                  if checkpoint is None or checkpoint.record.context.status == "running" else None)

        global_ctx = load_context_file(Path("~/.x/context.md").expanduser())
        project_ctx = load_context_file(Path(".x/context.md"))

        task_manager = TaskManager(run_path / ".tasks")

        # 每次运行拥有独立事件流，仅单向桥接到 daemon 总线，不接收其他运行事件
        bus = EventBus()
        if self._bus is not None:
            parent_bus = self._bus

            # daemon 的观察者异常只影响通知，不能反向中断当前运行
            async def forward(event: BaseModel) -> None:
                await parent_bus.publish(event, isolate_errors=True)

            bus.subscribe(forward)
        for h in self._extra_handlers:
            bus.subscribe(h)

        context = (checkpoint.record.context.restore(run_id) if checkpoint is not None
                   else ExecutionContext(
            run_id=run_id,
            goal=goal,
            max_steps=self._config.agent.max_steps,
            prefill_messages=history,
            session_notes=notes,
            global_context=global_ctx,
            project_context=project_ctx,
            system_prompt_override=system_prompt_override,
        ))
        prefill_len = len(history)
        initial_messages = context.messages

        if (checkpoint is None and session is not None and store is not None
                and session.mode == "chat"):
            preview = self._build_registry(
                task_manager, session=session, store=store, run_id=run_id,
                provider=cast(LLMProvider, object()), bus=bus,
                child_runs_dir=store.runs_dir(session.id), session_id=session.id,
                tool_whitelist=tool_whitelist,
            )
            checkpoint = BackgroundCheckpoint(run_path / "root.json", BackgroundRecord(
                kind="root", run_id=run_id, session_id=session.id, parent_run_id="",
                budget_root_id=budget.root_id if budget is not None else "",
                description=goal[:80], cwd=str(Path.cwd().resolve()), depth=0,
                tools=[str(schema["name"]) for schema in preview.tool_schemas()],
                model=self._config.llm.default_model,
                runtime_signature=runtime_signature(self._config, preview.tool_schemas()),
                context=ContextSnapshot.capture(context),
            ))
            checkpoint.save(context, "ready")
        self._root_checkpoint = checkpoint

        async with EventWriter(run_path / "events.jsonl") as writer:
            writer.subscribe(bus)
            await bus.publish(RunStartedEvent(
                run_id=run_id, goal=goal, ts=_now(),
                session_id=session.id if session is not None else "",
                resumed=resuming,
            ))

            cancelled = False
            session_persisted = False
            try:
                provider = (self._get_provider(budget) if not context.is_done()
                            else cast(LLMProvider, object()))
                session_id_str = session.id if session is not None else ""
                child_runs_dir = (
                    store.runs_dir(session.id)
                    if session is not None and store is not None
                    else self._runs_dir
                )
                registry = self._build_registry(
                    task_manager,
                    session=session,
                    store=store,
                    run_id=run_id,
                    provider=provider,
                    bus=bus,
                    child_runs_dir=child_runs_dir,
                    session_id=session_id_str,
                    tool_whitelist=tool_whitelist,
                    budget=budget,
                )
                session_dir = (
                    store.session_dir(session.id)
                    if session is not None and store is not None
                    else run_path
                )
                compactor = Compactor(
                    bus, session_dir, session_id_str, store=store,
                    tool_result_limit=self._config.compaction.tool_result_limit,
                    tool_result_keep=self._config.compaction.tool_result_keep,
                )
                loop = AgentLoop(
                    provider, registry, bus,
                    permission_manager=self._permission_manager,
                    compactor=compactor,
                    compact_threshold=self._config.compaction.auto_threshold,
                    session_id=session_id_str,
                    tool_result_limit=self._config.compaction.tool_result_limit,
                    tool_result_keep=self._config.compaction.tool_result_keep,
                    context_window=(self._config.llm.context_window
                                    or _context_window(self._config.llm.default_model)),
                    checkpoint=checkpoint.save if checkpoint is not None else None,
                    budget=budget,
                )
                await loop.run(context)
                if (context.reason == "runtime_budget" and checkpoint is not None
                        and checkpoint.record.phase == "tools" and checkpoint.pending_tools()):
                    checkpoint.interrupt(suspend=True)
            except asyncio.CancelledError:
                cancelled = True
                if not context.is_done():
                    context.mark_failed("cancelled")
                if checkpoint is not None:
                    checkpoint.interrupt(suspend=self._suspend_roots)
            except (Exception, SystemExit):
                logging.getLogger(__name__).exception(
                    "agent run failed run_id=%s step=%d", run_id, context.step
                )
                if not context.is_done():
                    context.mark_failed("llm_error")

            if (session is not None and store is not None and not cancelled
                    and (checkpoint is None or checkpoint.record.context.status != "running")):
                try:
                    # 摘要已在压缩提交时保存，仅追加本次运行尚未持久化的尾部
                    tail = context.messages[2:] if context.messages is not initial_messages else (
                        context.messages[prefill_len:]
                    )
                    store.append_messages(
                        session.id, context.messages if checkpoint is not None else tail,
                        run_id=run_id, replace_history=checkpoint is not None,
                    )
                    session.updated_at = _now()
                    session.status = "closed" if session.mode == "one_shot" else "waiting_for_input"
                    store.write_meta(session)
                    if checkpoint is not None:
                        checkpoint._commit(checkpoint.record.model_copy(update={
                            "thread_committed": True,
                        }))
                    session_persisted = True
                except Exception:
                    logging.getLogger(__name__).exception(
                        "session persistence failed run_id=%s", run_id
                    )
                    context.mark_failed("persistence_error")

            await bus.publish(
                RunFinishedEvent(
                    run_id=run_id,
                    status="suspended" if cancelled and self._suspend_roots else context.status,
                    reason=context.reason,
                    steps=context.step,
                    ts=_now(),
                )
            )

        if cancelled:
            raise asyncio.CancelledError()

        return RunOutcome(
            status=context.status,
            result=context.result,
            reason=context.reason,
            session_persisted=session_persisted,
        )
