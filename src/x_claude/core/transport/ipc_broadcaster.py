from __future__ import annotations

import asyncio
import fnmatch
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from x_claude.core.bus.envelope import EventPushEnvelope
from x_claude.core.trace.record import TraceRecord
from x_claude.core.trace.writer import TraceWriter

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class _Subscription:
    sub_id: str
    writer: asyncio.StreamWriter
    topics: list[str]
    scope: str
    pending: list[dict[str, Any]] | None = None
    replayed_ids: set[str] = field(default_factory=set)
    approvals: set[tuple[str, str, str]] = field(default_factory=set)


class IpcEventBroadcaster:
    def __init__(
        self, trace: TraceWriter | None = None, *, write_timeout_s: float = 2.0,
    ) -> None:
        self._subscriptions: list[_Subscription] = []
        self._trace = trace
        self._run_sessions: dict[str, str] = {}
        self._run_parents: dict[str, str] = {}
        self._write_timeout_s = write_timeout_s

    # 注册一个客户端订阅，返回 subscription_id
    def subscribe(
        self,
        writer: asyncio.StreamWriter,
        topics: list[str],
        scope: str = "global",
        *, replaying: bool = False,
    ) -> str:
        # 一个连接只保留当前订阅，切换 /clear 后不能继续接收旧会话
        self.unsubscribe(writer)
        sub_id = f"sub-{uuid.uuid4().hex[:8]}"
        sub = _Subscription(
            sub_id=sub_id, writer=writer, topics=topics, scope=scope,
            pending=[] if replaying else None,
        )
        self._subscriptions.append(sub)
        return sub_id

    # 完成回放后按顺序发送暂存事件，使用持久事件 ID 消除回放与实时流重叠
    async def finish_replay(self, writer: asyncio.StreamWriter, replayed_ids: set[str]) -> None:
        sub = next((s for s in self._subscriptions if s.writer is writer), None)
        if sub is None:
            raise ConnectionError("replay subscription disconnected")
        sub.replayed_ids = replayed_ids
        for event in sub.pending or []:
            if event.get("event_id") in replayed_ids:
                continue
            writer.write(EventPushEnvelope(event=event).model_dump_json().encode() + b"\n")
        sub.pending = None
        await asyncio.wait_for(writer.drain(), timeout=self._write_timeout_s)

    # 移除指定 writer 的所有订阅
    def unsubscribe(self, writer: asyncio.StreamWriter) -> None:
        self._subscriptions = [s for s in self._subscriptions if s.writer is not writer]

    # 只允许收到实时审批且订阅对应会话或运行的连接作答，全局观察连接不能审批
    def can_approve(
        self, writer: asyncio.StreamWriter, session_id: str, run_id: str, tool_use_id: str,
    ) -> bool:
        key = (session_id, run_id, tool_use_id)
        return any(s.writer is writer and s.scope != "global" and key in s.approvals
                   for s in self._subscriptions)

    # 向重连的对应订阅补发真实挂起审批，旧日志回放不能获得审批权
    async def send_pending_permission(
        self, writer: asyncio.StreamWriter, event: dict[str, Any],
    ) -> None:
        sub = next((s for s in self._subscriptions if s.writer is writer), None)
        event = self.associate(event)
        if (sub is None or sub.scope == "global" or not self.matches_scope(event, sub.scope)
                or not self._matches_topic("permission.requested", sub.topics)):
            return
        sub.approvals.add((str(event["session_id"]), str(event["run_id"]),
                           str(event["tool_use_id"])))
        writer.write(EventPushEnvelope(event=event).model_dump_json().encode() + b"\n")
        await asyncio.wait_for(writer.drain(), timeout=self._write_timeout_s)

    # 将事件推送到所有匹配的订阅客户端，写入失败时延迟清理死连接
    async def handle(self, event: BaseModel) -> None:
        event_dict = self.associate(event.model_dump())
        event_type: str = event_dict.get("type", "")
        run_id: str | None = event_dict.get("run_id")

        dead: list[asyncio.StreamWriter] = []

        for sub in list(self._subscriptions):
            if not self._matches_topic(event_type, sub.topics):
                continue
            if not self.matches_scope(event_dict, sub.scope):
                continue
            if event_type == "permission.requested" and sub.scope != "global":
                sub.approvals.add((str(event_dict.get("session_id", "")),
                                   str(run_id or ""), str(event_dict.get("tool_use_id", ""))))
            elif event_type in ("permission.granted", "permission.denied"):
                sub.approvals = {key for key in sub.approvals
                                 if key[1:] != (str(run_id), event_dict.get("tool_use_id"))}
            if event_dict.get("event_id") in sub.replayed_ids:
                continue
            if sub.pending is not None:
                if len(sub.pending) >= 1024:
                    dead.append(sub.writer)
                    sub.writer.close()
                else:
                    sub.pending.append(event_dict)
                continue
            try:
                envelope = EventPushEnvelope(event=event_dict)
                sub.writer.write(envelope.model_dump_json().encode() + b"\n")
                await asyncio.wait_for(sub.writer.drain(), timeout=self._write_timeout_s)
                if self._trace is not None:
                    client_id = str(sub.writer.get_extra_info("peername", "<unknown>"))
                    self._trace.emit(
                        TraceRecord(
                            ts=_now(),
                            direction="CORE→CLIENT",
                            layer="ipc",
                            kind="push",
                            run_id=run_id,
                            client_id=client_id,
                            data={"sub_id": sub.sub_id, "event_type": event_type},
                        )
                    )
            except (ConnectionResetError, BrokenPipeError, OSError, RuntimeError, ValueError):
                logger.debug("dead connection for sub %s, scheduling cleanup", sub.sub_id)
                dead.append(sub.writer)
                sub.writer.close()

        for writer in dead:
            self.unsubscribe(writer)

    # 关联运行与会话，实时推送和回放共用相同的父子关系规则
    def associate(self, event: dict[str, Any]) -> dict[str, Any]:
        event = dict(event)
        run_id = event.get("run_id")
        session_id = event.get("session_id", "")
        parent_id = event.get("parent_run_id")
        if run_id and parent_id:
            self._run_parents[run_id] = parent_id
            session_id = session_id or self._run_sessions.get(parent_id, "")
        if run_id and session_id:
            self._run_sessions[run_id] = session_id
        if run_id:
            session_id = session_id or self._run_sessions.get(run_id, "")
        if session_id:
            event["session_id"] = session_id
        return event

    # 检查事件类型是否匹配订阅的 topic 列表（支持 fnmatch glob 模式）
    @staticmethod
    def _matches_topic(event_type: str, topics: list[str]) -> bool:
        return any(fnmatch.fnmatch(event_type, pattern) for pattern in topics)

    # 检查事件 run_id 是否匹配订阅的 scope（global 全通，run:<id> 精确匹配）
    @staticmethod
    def _matches_scope(run_id: str | None, scope: str) -> bool:
        if scope == "global":
            return True
        if scope.startswith("run:"):
            return run_id == scope[4:]
        return False

    # 根据会话和父子运行关系过滤事件，保留同一运行树的子 Agent 进度
    def matches_scope(self, event: dict[str, object], scope: str) -> bool:
        if scope.startswith("session:"):
            sid = event.get("session_id") or self._run_sessions.get(str(event.get("run_id")), "")
            return sid == scope[8:]
        run_id = str(event.get("run_id", ""))
        seen: set[str] = set()
        while run_id and run_id not in seen:
            if self._matches_scope(run_id, scope):
                return True
            seen.add(run_id)
            run_id = self._run_parents.get(run_id, "")
        return scope == "global"
