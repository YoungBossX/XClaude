from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

type EventHandler = Callable[[BaseModel], Awaitable[None]]
logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[EventHandler] = []

    # 注册一个事件处理函数
    def subscribe(self, handler: EventHandler, *, first: bool = False) -> None:
        if first:
            self._subscribers.insert(0, handler)
        else:
            self._subscribers.append(handler)

    # 移除事件订阅，避免已结束运行的回调长期驻留
    def unsubscribe(self, handler: EventHandler) -> None:
        self._subscribers = [h for h in self._subscribers if h != handler]

    # 按注册顺序调用订阅者，可选隔离普通异常以继续完成通知，取消始终向上传播
    async def publish(self, event: BaseModel, *, isolate_errors: bool = False) -> None:
        for handler in list(self._subscribers):
            try:
                await handler(event)
            except Exception:
                if not isolate_errors:
                    raise
                logger.exception("event subscriber failed event=%s", type(event).__name__)
