from __future__ import annotations

from collections import deque
from typing import Any


class ReplayProgress:
    # 记录连续已收日志偏移与有界乱序区间，不能用最大偏移跳过尚未到达的事件
    def __init__(self) -> None:
        self.offsets: dict[str, int] = {}
        self._pending: dict[str, dict[int, int]] = {}
        self.seen: set[str] = set()
        self._recent: deque[str] = deque()

    # 通过已确认日志区间或近期事件 ID 判断重复，旧去重项可安全淘汰
    def duplicate(self, event: dict[str, Any]) -> bool:
        if event.get("event_id") in self.seen:
            return True
        return any(end <= self.offsets.get(run_id, 0)
                   for run_id, (_, end) in event.get("log_positions", {}).items())

    # 渲染成功后登记区间，遇到间隙时保留后续区间并等待缺失部分
    def accept(self, event: dict[str, Any]) -> None:
        identity = str(event.get("event_id", ""))
        if identity and identity not in self.seen:
            self.seen.add(identity)
            self._recent.append(identity)
            if len(self._recent) > 4096:
                self.seen.discard(self._recent.popleft())
        for run_id, (start, end) in event.get("log_positions", {}).items():
            pending = self._pending.setdefault(run_id, {})
            pending[start] = end
            self._advance(run_id)
            if len(pending) > 1024:
                pending.clear()  # 保守停留在连续位置，重连时重新读取间隙后的历史

    # 回放响应确认服务器已发送的完整区间，再衔接回放过程中收到的实时后缀
    def synchronize(self, offsets: dict[str, int]) -> None:
        for run_id, offset in offsets.items():
            self.offsets[run_id] = max(offset, self.offsets.get(run_id, 0))
            self._advance(run_id)

    # 连续推进单个日志文件，丢弃已经被较大确认区间覆盖的重复项
    def _advance(self, run_id: str) -> None:
        pending = self._pending.setdefault(run_id, {})
        offset = self.offsets.get(run_id, 0)
        while offset in pending:
            offset = pending.pop(offset)
        self.offsets[run_id] = offset
        for start in list(pending):
            if pending[start] <= offset:
                del pending[start]
