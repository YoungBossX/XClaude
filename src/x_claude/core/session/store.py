from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from x_claude.core.atomic_file import atomic_write_bytes
from x_claude.core.session.model import Session

logger = logging.getLogger(__name__)

MessageContent = str | list[dict[str, Any]]


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionStore:
    # 初始化 session 文件存储根目录
    def __init__(self, root: Path) -> None:
        self._root = root.expanduser()
        self._root.mkdir(parents=True, exist_ok=True)

    # 返回会话存储的实际目录，供 daemon 取得跨进程写入互斥锁
    @property
    def root(self) -> Path:
        return self._root.resolve()

    # 返回指定 session 的目录路径
    def session_dir(self, sid: str) -> Path:
        return self._root / sid

    # 返回指定 session 下的 runs 目录路径
    def runs_dir(self, sid: str) -> Path:
        return self.session_dir(sid) / "runs"

    # 分块逆向读取完整 JSONL 行，历史分页不需要把整个会话文件加载到内存
    @staticmethod
    def _previous_lines(stream: BinaryIO, end: int) -> Iterator[tuple[int, bytes]]:
        position, buffer = end, b""
        while position:
            start = max(0, position - 65536)
            stream.seek(start)
            buffer = stream.read(position - start) + buffer
            position = start
            boundary = len(buffer)
            while True:
                split = buffer.rfind(b"\n", 0, boundary - 1)
                if split < 0:
                    break
                yield position + split + 1, buffer[split + 1:boundary]
                boundary = split + 1
            buffer = buffer[:boundary]
        if buffer.strip():
            yield 0, buffer

    # 使用文件版本与字节偏移翻页，原子替换或压缩后拒绝旧游标，避免跳页或错读
    def history_page(
        self, sid: str, cursor: str | None, limit: int,
        *, max_chars: int | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        path = self.session_dir(sid) / "thread.jsonl"
        if not path.exists():
            return [], None
        messages: list[dict[str, Any]] = []
        with path.open("rb") as stream:
            import os

            stat = os.fstat(stream.fileno())
            revision = f"{stat.st_mtime_ns}-{stat.st_size}"
            end = stat.st_size
            if cursor is not None:
                version, raw_offset = cursor.rsplit(":", 1)
                if version != revision:
                    raise ValueError("对话已更新或压缩，请重新打开 /history 查看最新历史。")
                end = int(raw_offset)
                if not 0 <= end <= stat.st_size:
                    raise ValueError("invalid history cursor")
            offset = end
            for start, line in self._previous_lines(stream, end):
                offset = start
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if isinstance(row, dict) and row.get("role") in ("user", "assistant"):
                    content = row.get("content", "")
                    if max_chars is not None:
                        text = (content if isinstance(content, str)
                                else json.dumps(content, ensure_ascii=False))
                        if len(text) > max_chars:
                            suffix = "\n[本条过长，显示已截断；完整记录保留在本机会话文件]"
                            content = text[:max(0, max_chars - len(suffix))] + suffix
                    messages.append({"role": row["role"], "content": content})
                if len(messages) >= limit:
                    break
        messages.reverse()
        return messages, f"{revision}:{offset}" if offset else None

    # 将 session meta 写入 meta.json
    def write_meta(self, session: Session) -> None:
        path = self.session_dir(session.id)
        path.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path / "meta.json", (
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8"))

    # 从 meta.json 读取 session meta
    def read_meta(self, sid: str) -> Session:
        data = json.loads((self.session_dir(sid) / "meta.json").read_text(encoding="utf-8"))
        return Session.from_dict(data)

    # 返回最近更新的 chat session；无可用会话或损坏元数据时返回 None
    def latest_chat_session(self) -> Session | None:
        sessions: list[Session] = []
        for path in self._root.iterdir():
            if not path.is_dir() or not (path / "meta.json").exists():
                continue
            try:
                session = self.read_meta(path.name)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("skip invalid session metadata path=%s error=%s", path, exc)
                continue
            if session.mode == "chat":
                sessions.append(session)
        return max(sessions, key=lambda session: session.updated_at, default=None)

    # 追加一条 Anthropic API 消息到 thread.jsonl
    def append_message(
        self,
        sid: str,
        role: str,
        content: MessageContent,
        run_id: str | None = None,
    ) -> None:
        row: dict[str, Any] = {"ts": _now(), "role": role, "content": content}
        if run_id is not None:
            row["run_id"] = run_id
        path = self.session_dir(sid)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "thread.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 普通批次原子追加，主任务按提交标识幂等写回完整上下文，避免恢复后重复追加
    def append_messages(
        self,
        sid: str,
        messages: list[dict[str, Any]],
        run_id: str,
        *, replace_history: bool = False,
    ) -> None:
        if not messages:
            return
        path = self.session_dir(sid) / "thread.jsonl"
        original = path.read_bytes() if path.exists() else b""
        existing = ([json.loads(line) for line in original.decode("utf-8").splitlines() if line]
                    if replace_history else [])
        if replace_history and any(row.get("root_commit") == run_id for row in existing):
            return
        rows = [
            json.dumps(
                {"ts": _now(), "role": str(msg["role"]),
                 "content": msg["content"], "run_id": run_id},
                ensure_ascii=False,
            ) + "\n"
            for msg in messages
        ]
        if replace_history:
            replacement: list[dict[str, Any]] = []
            for index, message in enumerate(messages):
                old = existing[index] if index < len(existing) else {}
                if old.get("role") == message["role"] and old.get("content") == message["content"]:
                    replacement.append(dict(old))
                else:
                    replacement.append({"ts": _now(), **message, "run_id": run_id})
            replacement[-1]["root_commit"] = run_id
            payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in replacement)
            atomic_write_bytes(path, payload.encode("utf-8"))
        else:
            atomic_write_bytes(path, original + "".join(rows).encode("utf-8"))

    # 读取完整 thread 并返回可直接传给 Anthropic 的 messages
    def read_messages(self, sid: str, *, truncate: bool = True) -> list[dict[str, Any]]:
        path = self.session_dir(sid) / "thread.jsonl"
        if not path.exists():
            return []

        messages: list[dict[str, Any]] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skip broken thread row sid=%s line=%s", sid, line_no)
                continue
            role = row.get("role")
            if role not in ("user", "assistant"):
                logger.warning(
                    "skip unknown thread role sid=%s line=%s role=%s",
                    sid,
                    line_no,
                    role,
                )
                continue
            messages.append({"role": role, "content": row.get("content", "")})

        messages = self._trim_orphan_tool_use(messages)
        if not truncate:
            return messages
        from x_claude.core.compact.budget import truncate_tool_results
        return truncate_tool_results(messages)

    # 裁掉尾部未配对 tool_use 以及其后的消息，避免 Anthropic messages.invalid
    def _trim_orphan_tool_use(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pending: set[str] = set()
        last_balanced = 0
        for idx, msg in enumerate(messages, start=1):
            content = msg.get("content")
            if isinstance(content, list):
                if msg.get("role") == "assistant":
                    for block in content:
                        if block.get("type") == "tool_use":
                            pending.add(str(block.get("id", "")))
                elif msg.get("role") == "user":
                    for block in content:
                        if block.get("type") == "tool_result":
                            pending.discard(str(block.get("tool_use_id", "")))
            if not pending:
                last_balanced = idx
        if pending:
            logger.warning("trim orphan tool_use blocks from thread")
            return messages[:last_balanced]
        return messages

    # 完整序列化新历史，保存独立备份，再原子替换 thread.jsonl，任一提交前失败都保留原历史
    def write_compacted(self, sid: str, messages: list[dict[str, Any]]) -> None:
        path = self.session_dir(sid) / "thread.jsonl"
        rows = [
            json.dumps(
                {"ts": _now(), "role": msg["role"], "content": msg["content"]},
                ensure_ascii=False,
            ) + "\n"
            for msg in messages
        ]
        payload = "".join(rows).encode("utf-8")
        ts_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        bak = self.session_dir(sid) / f"thread_{ts_str}_{uuid4().hex}.jsonl.bak"
        if path.exists():
            atomic_write_bytes(bak, path.read_bytes())
        atomic_write_bytes(path, payload)

    # 读取 notes.md 全文，文件不存在时返回空字符串
    def read_notes(self, sid: str) -> str:
        path = self.session_dir(sid) / "notes.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    # 将一条主动笔记追加到 notes.md
    def append_note(self, sid: str, content: str, run_id: str) -> None:
        path = self.session_dir(sid)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "notes.md").open("a", encoding="utf-8") as f:
            f.write(f"## Note ({_now()}, {run_id})\n{content}\n\n")
