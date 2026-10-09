from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from x_claude.core.trace.record import TraceRecord

log = logging.getLogger(__name__)


class TraceWriter:
    # 限制观测队列大小，日志故障不得无限积压内存或阻塞 daemon 关闭
    def __init__(self, path: Path, *, queue_size: int = 4096, stop_timeout_s: float = 2.0) -> None:
        if queue_size < 1 or stop_timeout_s <= 0:
            raise ValueError("trace queue size and stop timeout must be positive")
        self._path = path
        self._queue: asyncio.Queue[TraceRecord] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task[None] | None = None
        self._stop_timeout_s = stop_timeout_s
        self._accepting = True
        self.failed = False
        self.dropped_records = 0

    # 启动写入任务，重复调用不创建第二个消费者
    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.failed = False
        self._accepting = True
        self._task = asyncio.create_task(self._drain())

    # 有限等待队列落盘，消费者失败、未启动或超时均可结束关闭
    async def stop(self) -> None:
        self._accepting = False
        try:
            if self._task is not None and not self._task.done():
                try:
                    await asyncio.wait_for(self._queue.join(), timeout=self._stop_timeout_s)
                except TimeoutError:
                    log.warning("trace shutdown timed out; dropping pending diagnostic records")
        finally:
            if self._task is not None:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
            self._discard_pending()

    # 非阻塞提交诊断记录，队列满时丢弃新记录并仅在首次发生时告警
    def emit(self, record: TraceRecord) -> None:
        if not self._accepting or self.failed:
            self.dropped_records += 1
            return
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            if not self.dropped_records:
                log.warning("trace queue full; dropping diagnostic records")
            self.dropped_records += 1

    # 清理不能再写出的记录，并配平 join 的未完成计数
    def _discard_pending(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
            self.dropped_records += 1

    # UTF-8 批量写入诊断记录，写入失败后关闭观测通道并释放全部等待者
    async def _drain(self) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as stream:
                while True:
                    records = [await self._queue.get()]
                    while len(records) < 64 and not self._queue.empty():
                        records.append(self._queue.get_nowait())
                    try:
                        stream.write("".join(record.model_dump_json() + "\n" for record in records))
                        stream.flush()
                    except Exception:
                        self.dropped_records += len(records)
                        raise
                    finally:
                        for _ in records:
                            self._queue.task_done()
                    await asyncio.sleep(0)
        except Exception:
            self.failed = True
            self._accepting = False
            log.exception("trace writer disabled after write failure")
        finally:
            self._discard_pending()
