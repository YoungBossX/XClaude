from typing import TYPE_CHECKING, Any

from x_claude.core.tools.base import BaseTool, ToolResult
from x_claude.core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from x_claude.core.tools.invocation import invoke_tool as invoke_tool


# 基础工具可独立导入，按需加载含事件依赖的调用器且保留原公开入口
def __getattr__(name: str) -> Any:
    if name == "invoke_tool":
        from x_claude.core.tools.invocation import invoke_tool

        return invoke_tool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
__all__ = ["BaseTool", "ToolResult", "ToolRegistry", "invoke_tool"]
