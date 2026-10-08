from typing import TYPE_CHECKING, Any

from x_claude.core.subagent.registry import BackgroundTaskRegistry

if TYPE_CHECKING:
    from x_claude.core.subagent.tool import AgentResultTool as AgentResultTool
    from x_claude.core.subagent.tool import SpawnAgentTool as SpawnAgentTool


# 检查点和注册表可独立导入，派生工具只在调用方请求时加载
def __getattr__(name: str) -> Any:
    if name in {"SpawnAgentTool", "AgentResultTool"}:
        from x_claude.core.subagent import tool

        return getattr(tool, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["BackgroundTaskRegistry", "SpawnAgentTool", "AgentResultTool"]
