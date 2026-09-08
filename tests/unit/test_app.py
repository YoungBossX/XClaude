from __future__ import annotations

import asyncio
import signal
from unittest.mock import MagicMock

import pytest

from x_claude.core.app import _install_shutdown_handlers


# 功能：验证事件循环不支持 add_signal_handler 时回退到 Windows 信号处理器
# 设计：让 mock loop 稳定抛出 NotImplementedError，再调用回退回调并断言线程安全地设置 shutdown，避免依赖宿主平台信号实现
def test_install_shutdown_handlers_falls_back_when_event_loop_lacks_signal_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.add_signal_handler.side_effect = NotImplementedError
    shutdown = asyncio.Event()
    handlers: dict[signal.Signals, object] = {}

    def register_handler(sig: signal.Signals, handler: object) -> object:
        handlers[sig] = handler
        return handler

    monkeypatch.setattr(signal, "signal", register_handler)
    _install_shutdown_handlers(loop, shutdown)

    assert set(handlers) == {signal.SIGINT, signal.SIGTERM}
    handler = handlers[signal.SIGINT]
    assert callable(handler)
    handler(signal.SIGINT, None)  # type: ignore[operator]
    loop.call_soon_threadsafe.assert_called_once_with(shutdown.set)
