from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from x_claude.core.bus.envelope import HandlerError
from x_claude.core.bus.events import (
    RunRestoredEvent,
    SessionClosedEvent,
    SessionCreatedEvent,
    SessionMessageReceivedEvent,
    SessionResumedEvent,
    SessionWaitingForInputEvent,
    SkillInvokedEvent,
)
from x_claude.core.config import XConfig
from x_claude.core.events.bus import EventBus
from x_claude.core.runs import new_run_id
from x_claude.core.session.model import Session, SessionMode
from x_claude.core.session.store import SessionStore
from x_claude.core.skills.loader import SkillLoader
from x_claude.core.subagent.checkpoint import BackgroundCheckpoint

if TYPE_CHECKING:
    from x_claude.core.bus.commands import SessionRecoverCommand, SessionRecoverResult
    from x_claude.core.llm.base import LLMProvider
    from x_claude.core.runner import AgentRunner

SESSION_NOT_FOUND = -32010
SESSION_CLOSED = -32011
SESSION_BUSY = -32012


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionManager:
    # 初始化会话管理器，接入文件存储、runner 工厂、事件总线和可选的 LLM provider（用于手动压缩）
    def __init__(
        self,
        store: SessionStore,
        runner_factory: Callable[[], AgentRunner],
        bus: EventBus,
        provider: LLMProvider | None = None,
        config: XConfig | None = None,
    ) -> None:
        self._store = store
        self._runner_factory = runner_factory
        self._bus = bus
        self._provider = provider
        self._config = config or XConfig()
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._skill_loader = SkillLoader()
        self._runners: dict[str, AgentRunner] = {}
        self._active_runs: set[asyncio.Task[Any]] = set()
        self._active_by_session: dict[str, asyncio.Task[Any]] = {}
        self._root_tasks: dict[str, asyncio.Task[None]] = {}
        self._suspending = False
        self._closing_sessions: set[str] = set()

    # 创建新 session 并写入 meta.json
    async def create(self, mode: SessionMode, title: str = "") -> Session:
        sid = f"sess-{uuid.uuid4().hex[:12]}"
        ts = _now()
        session = Session(
            id=sid,
            mode=mode,
            status="active",
            title=title,
            created_at=ts,
            updated_at=ts,
            run_ids=[],
        )
        self._sessions[sid] = session
        self._locks[sid] = asyncio.Lock()
        self._store.write_meta(session)
        await self._bus.publish(SessionCreatedEvent(session_id=sid, mode=mode, ts=ts))
        return session

    # 恢复磁盘中最近更新的 chat session，使其重新接受用户消息
    async def continue_latest(self) -> Session:
        session = self._store.latest_chat_session()
        if session is None:
            raise HandlerError(SESSION_NOT_FOUND, "no prior chat session found")
        return self._restore_session(session)

    # 恢复指定 ID 的持久化 chat session，使其重新接受用户消息
    async def resume(self, sid: str) -> Session:
        try:
            session = self._store.read_meta(sid)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            raise HandlerError(SESSION_NOT_FOUND, "session not found") from None
        if session.mode != "chat":
            raise HandlerError(SESSION_NOT_FOUND, "session is not a chat session")
        return self._restore_session(session)

    # 将持久化 session 重新注册到当前 manager 并切换为可输入状态
    def _restore_session(self, session: Session) -> Session:
        if self._locks.get(session.id, asyncio.Lock()).locked():
            existing = self._sessions.get(session.id)
            if existing is not None:
                return existing
            raise HandlerError(SESSION_BUSY, "session busy")
        session.status = "waiting_for_input"
        self._closing_sessions.discard(session.id)
        session.updated_at = _now()
        self._sessions[session.id] = session
        self._locks.setdefault(session.id, asyncio.Lock())
        self._store.write_meta(session)
        return session

    # 客户端订阅建立或发送新消息后才激活恢复，避免无客户端时启动新的工具审批
    async def recover_background(self, sid: str) -> None:
        session = self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked():
            if sid in self._active_by_session:
                await self._bus.publish(RunRestoredEvent(
                    run_id=session.run_ids[-1] if session.run_ids else "unknown",
                    session_id=sid, state="running", step=0,
                    message="已重新连接仍在运行的主任务", ts=_now(),
                ), isolate_errors=True)
            return
        async with lock:
            await self._recover_background(session)

    # 在持有会话锁时加载 Runner，确保恢复不能越过同时发生的关闭或清空操作
    async def _recover_background(self, session: Session) -> None:
        if session.status == "closed" or self._suspending:
            return
        sid = session.id
        runner = self._runners.get(sid)
        if runner is None:
            runner = self._runner_factory()
            self._runners[sid] = runner
        restore = getattr(runner, "restore_background", None)
        if restore is not None:
            await restore(session, self._store)
        await self._activate_root(session, runner)

    # 订阅成功后才启动唯一未完成主任务；不等待整轮 RPC，以便权限响应仍能并发处理
    async def _activate_root(self, session: Session, runner: AgentRunner) -> None:
        task = self._root_tasks.get(session.id)
        if task is not None and not task.done():
            return
        pending = getattr(runner, "pending_root", None)
        if pending is None or session.status == "closed" or self._suspending:
            return
        try:
            checkpoint = pending(session, self._store)
        except (ValueError, OSError) as exc:
            await self._bus.publish(RunRestoredEvent(
                run_id=session.run_ids[-1] if session.run_ids else "unknown",
                session_id=session.id, state="blocked", step=0,
                message=f"checkpoint_invalid: {exc}", ts=_now(),
            ), isolate_errors=True)
            return
        if checkpoint is not None and await runner.root_can_resume(
            checkpoint, session, self._store,
        ):
            self._root_tasks[session.id] = asyncio.create_task(
                self._resume_root(session, runner, checkpoint),
            )

    # 恢复循环持有会话锁并沿用原 run_id，完成后通知输入状态，不追加新的用户目标
    async def _resume_root(
        self, session: Session, runner: AgentRunner, checkpoint: BackgroundCheckpoint,
    ) -> None:
        sid = session.id
        task = asyncio.current_task()
        try:
            async with self._locks[sid]:
                if session.status == "closed" or sid in self._closing_sessions:
                    return
                if task is not None:
                    self._active_runs.add(task)
                    self._active_by_session[sid] = task
                session.status = "active"
                try:
                    await runner.run_and_capture(
                        checkpoint.record.context.goal, run_id=checkpoint.record.run_id,
                        session=session, store=self._store, checkpoint=checkpoint,
                        tool_whitelist=checkpoint.record.tools,
                    )
                finally:
                    if not self._suspending and sid not in self._closing_sessions:
                        session.status = "waiting_for_input"
                        session.updated_at = _now()
                        self._store.write_meta(session)
                        await self._bus.publish(SessionWaitingForInputEvent(
                            session_id=sid, last_run_id=checkpoint.record.run_id, ts=_now(),
                        ), isolate_errors=True)
        except Exception:
            logging.getLogger(__name__).exception("root recovery failed session=%s", sid)
        finally:
            if task is not None:
                self._active_runs.discard(task)
                if self._active_by_session.get(sid) is task:
                    self._active_by_session.pop(sid, None)
                if self._root_tasks.get(sid) is task:
                    self._root_tasks.pop(sid, None)

    # 在会话锁内处理用户主动恢复，关闭会话及运行中的任务不能被人工确认覆盖
    async def recover(self, command: SessionRecoverCommand) -> SessionRecoverResult:
        from x_claude.core.bus.commands import SessionRecoverResult

        session = self._get_session(command.session_id)
        lock = self._locks[session.id]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        async with lock:
            if session.status == "closed":
                raise HandlerError(SESSION_CLOSED, "session already closed")
            runner = self._runners.get(session.id)
            if runner is None:
                runner = self._runner_factory()
                self._runners[session.id] = runner
            try:
                if command.run_id is not None:
                    await runner.recover_task(
                        session, self._store, command.run_id,
                        command.tool_results, command.accept_config_change,
                    )
                    await self._activate_root(session, runner)
                elif command.tool_results is not None or command.accept_config_change:
                    raise ValueError("run_id is required for recovery confirmation")
                else:
                    await self._recover_background(session)
            except (ValueError, OSError) as exc:
                raise HandlerError(-32030, str(exc)) from None
            return SessionRecoverResult(tasks=runner.background_status(session, self._store))

    # 处理用户消息，追加 thread 并启动一次 agent run
    async def send_message(self, sid: str, content: str, *, run_id: str | None = None) -> str:
        session = self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")

        async with lock:
            task = asyncio.current_task()
            if task is not None:
                self._active_runs.add(task)
                self._active_by_session[sid] = task
            try:
                return await self._send_message_locked(session, content, run_id)
            finally:
                if task is not None:
                    self._active_runs.discard(task)
                    if self._active_by_session.get(sid) is task:
                        self._active_by_session.pop(sid, None)

    # 在会话锁内准备目标、执行循环并保存状态，取消覆盖准备和通知阶段
    async def _send_message_locked(self, session: Session, content: str, run_id: str | None) -> str:
        sid = session.id
        if self._suspending:
            raise HandlerError(SESSION_BUSY, "daemon is shutting down")
        if session.status == "closed":
            raise HandlerError(SESSION_CLOSED, "session already closed")

        await self._recover_background(session)
        runner = self._runners[sid]
        pending = getattr(runner, "pending_root", None)
        if pending is not None:
            try:
                if pending(session, self._store) is not None:
                    raise HandlerError(SESSION_BUSY, "unfinished task: use /recover or /clear")
            except (ValueError, OSError) as exc:
                raise HandlerError(-32030, f"checkpoint requires review: {exc}") from None

        if session.status == "waiting_for_input":
            await self._bus.publish(SessionResumedEvent(session_id=sid, ts=_now()))

        self._store.append_message(sid, "user", content)
        await self._bus.publish(
            SessionMessageReceivedEvent(session_id=sid, content=content, ts=_now())
        )

        if not session.title:
            session.title = content[:40]

        run_id = run_id or new_run_id()
        session.run_ids.append(run_id)
        session.updated_at = _now()
        self._store.write_meta(session)

        # Skill 解析：检测 "/" 前缀，展开为系统提示覆盖和工具白名单
        goal = content
        system_prompt_override: str | None = None
        tool_whitelist: list[str] | None = None
        if content.startswith("/") and content[1:].strip():
            parts = content[1:].split(None, 1)
            skill_name = parts[0]
            arguments = parts[1] if len(parts) > 1 else ""
            skill = self._skill_loader.resolve(skill_name)
            if skill is not None:
                goal = self._skill_loader.render_prompt(skill, arguments)
                system_prompt_override = goal
                tool_whitelist = skill.allowed_tools or None
                await self._bus.publish(
                    SkillInvokedEvent(
                        skill_name=skill_name,
                        arguments=arguments,
                        run_id=run_id,
                        session_id=sid,
                        ts=_now(),
                    )
                )

        # Runner 按会话复用，后台任务可跨轮查询，但不能被其他会话访问
        current_runner = self._runners.get(sid)
        if current_runner is None:
            current_runner = self._runner_factory()
            self._runners[sid] = current_runner
        runner = current_runner
        outcome = None
        try:
            outcome = await runner.run_and_capture(
                goal, run_id=run_id, session=session, store=self._store,
                system_prompt_override=system_prompt_override, tool_whitelist=tool_whitelist,
            )
        finally:
            if outcome is None or not outcome.session_persisted:
                session.updated_at = _now()
                session.status = "closed" if session.mode == "one_shot" else "waiting_for_input"
                self._store.write_meta(session)

        if session.mode == "one_shot":
            session.status = "closed"
            await self._stop_runner(sid)
            await self._bus.publish(SessionClosedEvent(session_id=sid, ts=session.updated_at))
        else:
            session.status = "waiting_for_input"
            await self._bus.publish(
                SessionWaitingForInputEvent(
                    session_id=sid,
                    last_run_id=run_id,
                    ts=session.updated_at,
                )
            )
        return run_id

    # 关闭指定 session 并更新 meta.json
    async def close(self, sid: str) -> None:
        session = self._get_session(sid)
        await self._cancel_root(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        async with lock:
            await self._stop_runner(sid)
            session.status = "closed"
            session.updated_at = _now()
            self._store.write_meta(session)
            await self._bus.publish(SessionClosedEvent(session_id=sid, ts=session.updated_at))

    # 结束当前 session 并创建空白 chat session，保留旧 session 供按 ID 恢复
    async def clear_context(self, sid: str) -> Session:
        session = self._get_session(sid)
        await self._cancel_root(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        async with lock:
            await self._stop_runner(sid)
            session.status = "closed"
            session.updated_at = _now()
            self._store.write_meta(session)
            return await self.create("chat")

    # 用户关闭或清空时取消主任务，已确认结果保留，但原任务不会在下次恢复时复活
    async def _cancel_root(self, sid: str) -> None:
        self._closing_sessions.add(sid)
        runner = self._runners.get(sid)
        prepare = getattr(runner, "prepare_shutdown", None)
        if prepare is not None:
            prepare(suspend=False)
        task = self._active_by_session.get(sid) or self._root_tasks.get(sid)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    # 结束会话时取消尚未完成的后台任务，避免清空后继续修改旧任务文件
    async def _stop_runner(self, sid: str, *, suspend: bool = False) -> None:
        runner = self._runners.pop(sid, None)
        shutdown = getattr(runner, "shutdown", None)
        if shutdown is not None:
            if suspend:
                await shutdown(suspend=True)
            else:
                await shutdown()
        if not suspend:
            BackgroundCheckpoint.cancel_saved(self._store.runs_dir(sid), sid)

    # daemon 退出时清理所有会话的后台子 Agent
    async def shutdown(self) -> None:
        self._suspending = True
        for runner in self._runners.values():
            prepare = getattr(runner, "prepare_shutdown", None)
            if prepare is not None:
                prepare(suspend=True)
        tasks = [task for task in set(self._active_runs) | set(self._root_tasks.values())
                 if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for sid in list(self._runners):
            await self._stop_runner(sid, suspend=True)

    # 手动压缩指定 session 的 thread，将摘要持久化写入 thread.jsonl
    async def compact(self, sid: str, focus: str = "") -> Any:
        session = self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        if self._provider is None:
            raise HandlerError(-32020, "provider not available for compaction")
        async with lock:
            runner = self._runners.get(sid)
            pending = getattr(runner, "pending_root", None)
            if pending is not None and pending(session, self._store) is not None:
                raise HandlerError(
                    SESSION_BUSY, "unfinished task: recover or clear before compaction"
                )
            from x_claude.core.bus.commands import SessionCompactResult
            from x_claude.core.compact.compactor import Compactor
            messages = self._store.read_messages(sid, truncate=False)
            session_dir = self._store.session_dir(sid)
            compactor = Compactor(
                self._bus, session_dir, sid,
                tool_result_limit=self._config.compaction.tool_result_limit,
                tool_result_keep=self._config.compaction.tool_result_keep,
            )
            result = await compactor.compact_messages(messages, self._provider, focus=focus)
            if result is None:
                raise HandlerError(-32021, "compaction failed or not beneficial")
            self._store.write_compacted(sid, [
                {"role": "user", "content": result.summary_text},
                {"role": "assistant", "content": "Understood, I'll continue from this summary."},
            ])
            return SessionCompactResult(
                summary_tokens=result.summary_tokens,
                saved_tokens=max(0, result.original_token_estimate - result.summary_tokens),
            )

    # 读取指定 session 的完整 thread 历史
    async def get_history(self, sid: str) -> list[dict[str, Any]]:
        self._get_session(sid)
        return self._store.read_messages(sid)

    # 从内存索引取 session，不存在时抛 JSON-RPC 结构化错误
    def _get_session(self, sid: str) -> Session:
        session = self._sessions.get(sid)
        if session is None:
            raise HandlerError(SESSION_NOT_FOUND, "session not found")
        return session
