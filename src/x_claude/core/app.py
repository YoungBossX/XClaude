from __future__ import annotations

import asyncio
import datetime
import fnmatch
import heapq
import json
import logging
import signal
import time
from collections.abc import Generator
from contextlib import AsyncExitStack
from datetime import UTC
from pathlib import Path
from typing import Any

from pydantic import BaseModel

import x_claude
from x_claude.core.bus.commands import (
    AgentRunCommand,
    AgentRunResult,
    EventSubscribeCommand,
    EventSubscribeResult,
    PermissionRespondCommand,
    PermissionRespondResult,
    PongResult,
    SessionClearCommand,
    SessionClearResult,
    SessionCloseCommand,
    SessionCloseResult,
    SessionCompactCommand,
    SessionCompactResult,
    SessionContinueCommand,
    SessionContinueResult,
    SessionCreateCommand,
    SessionCreateResult,
    SessionGetHistoryCommand,
    SessionGetHistoryResult,
    SessionRecoverCommand,
    SessionRecoverResult,
    SessionResumeCommand,
    SessionResumeResult,
    SessionSendMessageCommand,
    SessionSendMessageResult,
)
from x_claude.core.bus.envelope import INVALID_PARAMS, EventPushEnvelope, HandlerError
from x_claude.core.config import XConfig, get_config
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.provider import AnthropicProvider
from x_claude.core.logging_setup import setup_logging
from x_claude.core.mcp.server import McpServerManager
from x_claude.core.permissions.manager import PermissionManager
from x_claude.core.permissions.storage import load_policy_file
from x_claude.core.runner import AgentRunner
from x_claude.core.runs import events_file, new_run_id
from x_claude.core.session import SessionManager, SessionStore
from x_claude.core.session.lock import SessionStoreLock, StoreInUseError
from x_claude.core.trace.record import TraceRecord
from x_claude.core.trace.writer import TraceWriter
from x_claude.core.transport.ipc_broadcaster import IpcEventBroadcaster
from x_claude.core.transport.socket_server import SocketServer, get_connection_writer

logger = logging.getLogger(__name__)


# 注册 Unix 事件循环信号处理器，并在 Windows 回退到标准 signal 回调
def _install_shutdown_handlers(
    loop: asyncio.AbstractEventLoop,
    shutdown: asyncio.Event,
) -> None:
    def request_shutdown(*_args: object) -> None:
        loop.call_soon_threadsafe(shutdown.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except NotImplementedError:
            signal.signal(sig, request_shutdown)


def _now() -> str:
    return datetime.datetime.now(UTC).isoformat()


class CoreApp:
    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self._bus = EventBus()
        self._broadcaster: IpcEventBroadcaster | None = None
        self._trace: TraceWriter | None = None
        self._config: XConfig | None = None
        self._running_runs: set[asyncio.Task[Any]] = set()
        self._sessions: SessionManager | None = None
        self._permission_manager: PermissionManager | None = None
        self._mcp_manager: McpServerManager | None = None

    # 处理 core.ping 请求，返回服务版本、运行时长和接收时间
    async def _ping_handler(self, params: dict[str, Any]) -> PongResult:
        client = params.get("client", "unknown")
        logger.debug("ping from %s", client)
        return PongResult(
            server_version=x_claude.__version__,
            uptime_ms=int((time.monotonic() - self._start_time) * 1000),
            received_at=datetime.datetime.now(datetime.UTC).isoformat(),
        )

    # 将 EventBus 事件写入 trace（作为 EventBus 订阅者）
    async def _trace_event_handler(self, event: BaseModel) -> None:
        assert self._trace is not None
        event_dict = event.model_dump()
        self._trace.emit(
            TraceRecord(
                ts=_now(),
                direction="CORE",
                layer="event",
                kind="event",
                run_id=event_dict.get("run_id"),
                data=event_dict,
            )
        )

    # 启动一次 agent run：异步创建 AgentRunner 并立即返回 run_id
    async def _agent_run_handler(self, params: dict[str, Any]) -> AgentRunResult:
        assert self._sessions is not None
        cmd = AgentRunCommand.model_validate(params)
        session = await self._sessions.create(mode="one_shot", title=cmd.goal[:40])
        run_id = new_run_id()
        run_task = asyncio.create_task(
            self._sessions.send_message(session.id, cmd.goal, run_id=run_id)
        )
        self._running_runs.add(run_task)
        run_task.add_done_callback(self._running_runs.discard)
        return AgentRunResult(run_id=run_id)

    # 创建 chat 或 one_shot session，并返回 session_id
    async def _session_create_handler(self, params: dict[str, Any]) -> SessionCreateResult:
        assert self._sessions is not None
        cmd = SessionCreateCommand.model_validate(params)
        session = await self._sessions.create(mode=cmd.mode, title=cmd.title)
        return SessionCreateResult(session_id=session.id, status=session.status)

    # 恢复最近一次 chat session，并返回其 ID 与可输入状态
    async def _session_continue_handler(self, params: dict[str, Any]) -> SessionContinueResult:
        assert self._sessions is not None
        SessionContinueCommand.model_validate(params)
        session = await self._sessions.continue_latest()
        return SessionContinueResult(session_id=session.id, status=session.status)

    # 恢复指定 ID 的 chat session，并返回其可输入状态
    async def _session_resume_handler(self, params: dict[str, Any]) -> SessionResumeResult:
        assert self._sessions is not None
        cmd = SessionResumeCommand.model_validate(params)
        session = await self._sessions.resume(cmd.session_id)
        return SessionResumeResult(session_id=session.id, status=session.status)

    # 向 session 发送一条用户消息并同步等待对应 run 完成
    async def _session_send_handler(self, params: dict[str, Any]) -> SessionSendMessageResult:
        assert self._sessions is not None
        cmd = SessionSendMessageCommand.model_validate(params)
        run_id = await self._sessions.send_message(cmd.session_id, cmd.content)
        return SessionSendMessageResult(run_id=run_id)

    # 返回 session 的完整 Anthropic messages 历史
    async def _session_history_handler(self, params: dict[str, Any]) -> SessionGetHistoryResult:
        assert self._sessions is not None
        cmd = SessionGetHistoryCommand.model_validate(params)
        messages = await self._sessions.get_history(cmd.session_id)
        return SessionGetHistoryResult(messages=messages)

    # 接收客户端权限审批响应，resolve 对应挂起的 Future
    async def _permission_respond_handler(self, params: dict[str, Any]) -> PermissionRespondResult:
        cmd = PermissionRespondCommand.model_validate(params)
        logger.info(
            "permission.respond received tool_use_id=%s decision=%s",
            cmd.tool_use_id, cmd.decision,
        )
        if self._permission_manager is None:
            logger.error("permission.respond: PermissionManager not initialized")
            return PermissionRespondResult(ok=False)
        if self._broadcaster is None or not self._broadcaster.can_approve(
            get_connection_writer(), cmd.session_id, cmd.run_id, cmd.tool_use_id,
        ):
            return PermissionRespondResult(ok=False)
        accepted = self._permission_manager.respond(
            cmd.tool_use_id, cmd.decision, session_id=cmd.session_id, run_id=cmd.run_id,
        )
        return PermissionRespondResult(ok=accepted)

    # 手动压缩 session thread，将摘要持久化写入 thread.jsonl
    async def _session_compact_handler(self, params: dict[str, Any]) -> SessionCompactResult:
        assert self._sessions is not None
        cmd = SessionCompactCommand.model_validate(params)
        result = await self._sessions.compact(cmd.session_id, cmd.focus)
        return result  # type: ignore[no-any-return]

    # 关闭 session 并返回 closed 状态
    async def _session_close_handler(self, params: dict[str, Any]) -> SessionCloseResult:
        assert self._sessions is not None
        cmd = SessionCloseCommand.model_validate(params)
        await self._sessions.close(cmd.session_id)
        return SessionCloseResult(status="closed")

    # 清空当前 session 的对话上下文并保留 session 可继续对话
    async def _session_clear_handler(self, params: dict[str, Any]) -> SessionClearResult:
        assert self._sessions is not None
        cmd = SessionClearCommand.model_validate(params)
        session = await self._sessions.clear_context(cmd.session_id)
        return SessionClearResult(session_id=session.id, status=session.status)

    # 显示后台任务恢复状态，或接收用户核对的工具结果及配置变更确认
    async def _session_recover_handler(self, params: dict[str, Any]) -> SessionRecoverResult:
        assert self._sessions is not None
        return await self._sessions.recover(SessionRecoverCommand.model_validate(params))

    # 注册客户端事件订阅，可选先回放 events.jsonl 历史再接收实时流
    async def _subscribe_handler(self, params: dict[str, Any]) -> EventSubscribeResult:
        cmd = EventSubscribeCommand.model_validate(params)
        writer = get_connection_writer()

        if cmd.replay_session and (not cmd.scope.startswith("session:")
                                   or cmd.replay_from_run is not None):
            raise HandlerError(INVALID_PARAMS, "replay_session requires session scope only")

        assert self._broadcaster is not None
        replayed_count = 0
        sub_id = self._broadcaster.subscribe(
            writer, cmd.topics, cmd.scope,
            replaying=cmd.replay_from_run is not None or cmd.replay_session,
        )
        try:
            if cmd.replay_session:
                replayed_ids: set[str] = set()
                replayed_count = await self._replay_session_events(
                    cmd.scope[8:], writer, cmd.topics, replayed_ids,
                )
                await self._broadcaster.finish_replay(writer, replayed_ids)
            elif cmd.replay_from_run is not None:
                replayed_ids = set()
                replayed_count = await self._replay_events(
                    cmd.replay_from_run, writer, cmd.topics, scope=cmd.scope,
                    replayed_ids=replayed_ids,
                )
                await self._broadcaster.finish_replay(writer, replayed_ids)
            if self._permission_manager is not None:
                for event in self._permission_manager.pending_events():
                    # 上一个补发的 drain 期间审批可能已超时或被其他连接处理，不能发送失效快照
                    if event not in self._permission_manager.pending_events():
                        continue
                    await self._broadcaster.send_pending_permission(writer, event)
            if cmd.scope.startswith("session:") and self._sessions is not None:
                sid = cmd.scope[8:]
                await self._sessions.recover_background(sid)
                if any(fnmatch.fnmatch("session.synchronized", p) for p in cmd.topics):
                    state = self._sessions.synchronization_event(sid)
                    writer.write(EventPushEnvelope(event=state.model_dump()).model_dump_json()
                                 .encode() + b"\n")
                    await asyncio.wait_for(writer.drain(), timeout=2.0)
        except BaseException:
            self._broadcaster.unsubscribe(writer)
            raise
        return EventSubscribeResult(subscription_id=sub_id, replayed_count=replayed_count)

    # 按时间合并会话日志，去重父子日志的镜像事件，实时事件在回放期间暂存
    async def _replay_session_events(
        self, sid: str, writer: asyncio.StreamWriter, topics: list[str], seen: set[str],
    ) -> int:
        assert self._sessions is not None and self._broadcaster is not None

        # 只读取订阅开始时已落盘的完整行，不把后续写入混入回放快照
        def read_events(path: Path, size: int) -> Generator[dict[str, Any], None, None]:
            with path.open("rb") as stream:
                while stream.tell() < size:
                    line = stream.readline(size - stream.tell())
                    if not line.endswith(b"\n"):
                        break
                    try:
                        item = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if isinstance(item, dict):
                        yield {**item, "session_id": sid}

        paths = self._sessions.event_paths(sid)
        streams = [read_events(path, path.stat().st_size) for path in paths]
        count = 0
        try:
            for raw in heapq.merge(*streams, key=lambda item: str(item.get("ts", ""))):
                event = self._broadcaster.associate(raw)
                identity = str(event.get("event_id", ""))
                if identity and identity in seen:
                    continue
                if identity:
                    seen.add(identity)
                # 历史审批只作为日志，不能重新挂载可操作弹窗；真实待审批稍后单独补发
                if event.get("type") == "permission.requested":
                    continue
                if not any(fnmatch.fnmatch(event.get("type", ""), p) for p in topics):
                    continue
                writer.write(EventPushEnvelope(event=event).model_dump_json().encode() + b"\n")
                count += 1
                if count % 100 == 0:
                    await asyncio.wait_for(writer.drain(), timeout=2.0)
        finally:
            for stream in streams:
                stream.close()
        if count:
            await asyncio.wait_for(writer.drain(), timeout=2.0)
        return count

    # 从 events.jsonl 向 writer 回放匹配 topic 的历史事件，返回已回放条数
    async def _replay_events(
        self,
        run_id: str,
        writer: asyncio.StreamWriter,
        topics: list[str],
        *, scope: str = "global", replayed_ids: set[str] | None = None,
    ) -> int:
        path = events_file(run_id)
        if not path.exists():
            candidate = (self._sessions.run_event_path(run_id)
                         if self._sessions is not None else None)
            if candidate is not None:
                path = candidate
        if not path.exists():
            return 0

        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if scope.startswith("session:") and path.parent.parent.name == "runs":
                event["session_id"] = event.get("session_id") or path.parent.parent.parent.name
            if self._broadcaster is not None:
                event = self._broadcaster.associate(event)
                if not self._broadcaster.matches_scope(event, scope):
                    continue
            # 即使未订阅 started 事件，也必须先建立父子关系以过滤后续 token
            event_type: str = event.get("type", "")
            if not any(fnmatch.fnmatch(event_type, p) for p in topics):
                continue
            envelope = EventPushEnvelope(event=event)
            writer.write(envelope.model_dump_json().encode() + b"\n")
            if replayed_ids is not None and event.get("event_id"):
                replayed_ids.add(event["event_id"])
            count += 1

        if count:
            await asyncio.wait_for(writer.drain(), timeout=2.0)
        return count

    # 启动守护进程：加载配置、初始化日志、启动 trace、启动 TCP 服务器，并等待退出信号
    async def run(self) -> None:
        self._start_time = time.monotonic()
        self._config = get_config()
        store = SessionStore(Path("~/.x/sessions").expanduser())
        try:
            with SessionStoreLock(store.root):
                await self._run_locked(store)
        except StoreInUseError as exc:
            raise SystemExit(str(exc)) from None

    # 持有存储锁期间启动资源，退出或启动失败时按依赖顺序完整关闭后才释放锁
    async def _run_locked(self, store: SessionStore) -> None:
        assert self._config is not None
        setup_logging(self._config)
        async with AsyncExitStack() as cleanup:
            if self._config.trace.enabled:
                trace_path = Path(self._config.trace.file).expanduser()
                self._trace = TraceWriter(trace_path)
                cleanup.push_async_callback(self._trace.stop)
                await self._trace.start()
                self._bus.subscribe(self._trace_event_handler)

            policy_file = Path("~/.x/policy.toml").expanduser()
            self._permission_manager = PermissionManager(
                policy_file=policy_file, timeout_s=self._config.permission.timeout_s,
            )
            logger.info(
                "permission manager: timeout_s=%.1f  persistent=%d entries",
                self._config.permission.timeout_s, len(load_policy_file(policy_file)),
            )
            self._broadcaster = IpcEventBroadcaster(trace=self._trace)
            self._bus.subscribe(self._broadcaster.handle)
            compact_provider = AnthropicProvider(
                self._config.llm.default_model, context_window=self._config.llm.context_window,
            )
            self._mcp_manager = McpServerManager()
            cleanup.push_async_callback(self._mcp_manager.stop_all)
            if self._config.mcp.servers:
                await self._mcp_manager.start_all(self._config.mcp.servers)
            self._sessions = SessionManager(
                store,
                runner_factory=lambda: AgentRunner(
                    self._config,  # type: ignore[arg-type]
                    bus=self._bus, trace=self._trace,
                    permission_manager=self._permission_manager, mcp_manager=self._mcp_manager,
                ),
                bus=self._bus, provider=compact_provider, config=self._config,
            )
            server = SocketServer(
                self._config.host, self._config.port, self._broadcaster, trace=self._trace,
            )
            cleanup.push_async_callback(server.stop)
            cleanup.push_async_callback(self._shutdown_sessions)
            cleanup.callback(server.stop_accepting)
            for method, handler in [
                ("core.ping", self._ping_handler),
                ("agent.run", self._agent_run_handler),
                ("event.subscribe", self._subscribe_handler),
                ("session.create", self._session_create_handler),
                ("session.continue", self._session_continue_handler),
                ("session.resume", self._session_resume_handler),
                ("session.send_message", self._session_send_handler),
                ("session.get_history", self._session_history_handler),
                ("session.close", self._session_close_handler),
                ("session.clear", self._session_clear_handler),
                ("session.recover", self._session_recover_handler),
                ("permission.respond", self._permission_respond_handler),
                ("session.compact", self._session_compact_handler),
            ]:
                server.register(method, handler)
            addr = await server.start()
            logger.info("x-core %s listening addr=%s", x_claude.__version__, addr)
            logger.info("config: %s", self._config)
            shutdown = asyncio.Event()
            _install_shutdown_handlers(asyncio.get_running_loop(), shutdown)
            await shutdown.wait()
            logger.info("shutting down")

    # 先标记并挂起会话，再取消一次性请求，即使前者失败也不遗留请求协程
    async def _shutdown_sessions(self) -> None:
        try:
            if self._sessions is not None:
                await self._sessions.shutdown()
        finally:
            for run_task in list(self._running_runs):
                run_task.cancel()
            if self._running_runs:
                await asyncio.gather(*self._running_runs, return_exceptions=True)


# 同步入口：启动 CoreApp 事件循环
def run() -> None:
    asyncio.run(CoreApp().run())
