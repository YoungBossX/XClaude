from typing import TYPE_CHECKING, Any

from x_claude.core.session.model import Session, SessionMode, SessionStatus
from x_claude.core.session.store import MessageContent, SessionStore

if TYPE_CHECKING:
    from x_claude.core.session.manager import SessionManager as SessionManager


# 包导入只加载数据类型，按需导出管理器以免协议层反向加载完整运行时
def __getattr__(name: str) -> Any:
    if name == "SessionManager":
        from x_claude.core.session.manager import SessionManager

        return SessionManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MessageContent",
    "Session",
    "SessionManager",
    "SessionMode",
    "SessionStatus",
    "SessionStore",
]
