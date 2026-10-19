from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from x_claude.core.config import XConfig, _apply_toml
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.types import ToolCallBlock
from x_claude.core.permissions.manager import PermissionManager
from x_claude.core.permissions.policy import PermissionDecision, ToolPolicy
from x_claude.core.subagent.checkpoint import runtime_signature
from x_claude.core.tools.builtin.bash import BashTool
from x_claude.core.tools.invocation import invoke_tool
from x_claude.core.tools.registry import ToolRegistry


# 功能：固定命令协议在审批和执行前拒绝改写，返回错误后允许同一调用方提交原命令。
# 设计：真实 Bash、审批管理器与工具事件链，检查非法命令没有副作用或审批请求。
async def test_contract_rejects_before_approval_and_allows_explicit_correction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tool = BashTool(allowed_commands=["printf recovered"])
    registry, bus, events = ToolRegistry(), EventBus(), []
    registry.register(tool)

    # 记录实际工具事件，检查被拒命令是否错误进入审批或执行成功分支。
    async def collect(event):
        events.append(event.model_dump())

    bus.subscribe(collect)
    manager = PermissionManager(policies={"bash": ToolPolicy(PermissionDecision.ALLOW)}, timeout_s=.01)
    approval = AsyncMock(wraps=manager.check_and_wait)
    monkeypatch.setattr(manager, "check_and_wait", approval)
    bad = await invoke_tool(registry, ToolCallBlock(
        "bad", "bash", {"command": "cd . && printf leaked > leaked.txt"},
    ), bus, "child", permission_manager=manager)
    assert bad.is_error and bad.error_type == "schema_error"
    assert "not executed" in bad.content
    assert not (tmp_path / "leaked.txt").exists()
    assert not any(e["type"] == "permission.requested" for e in events)
    approval.assert_not_awaited()
    good = await invoke_tool(registry, ToolCallBlock(
        "good", "bash", {"command": "printf recovered"},
    ), bus, "child", permission_manager=manager)
    assert not good.is_error and good.content == "recovered"
    approval.assert_awaited_once()


# 功能：命令符合固定协议仍必须获得原有权限，协议不能自动批准或重试被拒操作。
# 设计：真实审批事件后显式拒绝，验证没有 Shell 成功事件或文件写入。
async def test_contract_does_not_override_permission_denial(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    command = "printf denied > denied.txt"
    registry, bus, events = ToolRegistry(), EventBus(), []
    registry.register(BashTool(allowed_commands=[command]))
    manager = PermissionManager(timeout_s=.1)

    # 在真实审批请求上提交用户拒绝，绝不改动工具或 Shell 的返回。
    async def deny(event):
        events.append(event.model_dump())
        if event.type == "permission.requested":
            assert manager.respond(event.tool_use_id, "deny_once", session_id="s", run_id="child")

    bus.subscribe(deny)
    result = await invoke_tool(registry, ToolCallBlock("write", "bash", {"command": command}),
                               bus, "child", permission_manager=manager, session_id="s")
    assert result.error_type == "permission_denied"
    assert not (tmp_path / "denied.txt").exists()
    assert not any(e["type"] == "tool.call_finished" for e in events)


# 功能：固定协议只影响显式配置的工具实例，不污染默认 Shell 工具或配置对象。
# 设计：先创建受限实例，再用默认实例执行非枚举命令，并核对修改源列表不扩大授权。
async def test_default_shell_and_contract_instances_are_isolated():
    commands = ["printf fixed"]
    restricted = BashTool(allowed_commands=commands)
    commands.append("printf dynamic")
    ordinary = BashTool()
    assert "enum" not in ordinary.input_schema["properties"]["command"]
    assert restricted.input_schema["properties"]["command"]["enum"] == ["printf fixed"]
    result = await ordinary.invoke({"command": "printf dynamic"})
    assert not result.is_error and result.content == "dynamic"


# 功能：配置可显式启用命令协议，且改变协议后恢复签名变化，防止重启绕过约束。
# 设计：从真实 TOML 字典入口加载并比较没有 Bash 的父任务签名，同时检查默认配置未被改变。
def test_config_contract_is_opt_in_and_changes_recovery_signature():
    config = XConfig()
    before = runtime_signature(config, [])
    _apply_toml(config, {"agent": {"allowed_shell_commands": ["printf fixed"], "max_tasks": 4}})
    assert config.agent.allowed_shell_commands == ["printf fixed"]
    assert runtime_signature(config, []) != before
    assert XConfig().agent.allowed_shell_commands is None
    assert XConfig().agent.max_tasks == 32


# 功能：拒绝含糊或空的命令协议配置，不把错误配置降级成无限制执行。
# 设计：真实配置入口覆盖字符串、空列表、空命令及非字符串元素。
@pytest.mark.parametrize("value", ["printf x", [], [""], [False]])
def test_invalid_command_contract_is_rejected(value):
    with pytest.raises(SystemExit):
        _apply_toml(XConfig(), {"agent": {"allowed_shell_commands": value}})
