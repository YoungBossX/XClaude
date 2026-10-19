from __future__ import annotations

import asyncio
import os
import re

import pytest
from textual.widgets import Static

from tests.helpers.windows_scenario import WindowsScenarioProvider
from tests.integration.test_runtime_hardening import _serve
from x_claude.tui.app import ChatTextArea, PermissionSelect, SlashCompleteWidget, XTuiApp

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows acceptance")


# 等待可观察状态，不假设一帧 UI 或一次 RPC 已完成。
async def wait_until(predicate, timeout=10):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.02)


# 功能：Windows TUI 通过 TCP 审批真实 Bash，正确处理中文空格路径并拒绝未授权写入。
# 设计：固定 Provider 避免模型随机性，真实输入键、审批控件、文件结果与事件共同验收。
async def test_windows_tui_shell_clear_and_exit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    async with _serve(tmp_path, WindowsScenarioProvider(), permission_timeout=10) as (cfg, _, bus, store):
        events = []

        # 收集真实总线事件以核对工具与运行完成状态。
        async def collect(event):
            events.append(event.model_dump())

        bus.subscribe(collect)
        app = XTuiApp(cfg.host, cfg.port)
        async with app.run_test(size=(110, 40)) as pilot:
            await wait_until(lambda: app._session_id and app._client)
            prompt = app.query_one("#prompt", ChatTextArea)
            for goal, key in (("unicode", "y"), ("denied", "n")):
                await wait_until(lambda: not prompt.disabled)
                prompt.text = goal
                await pilot.press("enter")
                await wait_until(lambda: bool(app.query(PermissionSelect)))
                await pilot.press(key)
                await wait_until(lambda: not app._busy and not prompt.disabled)
            assert (tmp_path / "中文 空格" / "结果.txt").read_text(encoding="utf-8") == "验收通过"
            assert not (tmp_path / "denied.txt").exists()
            assert any(e["type"] == "tool.call_finished" and "验收通过" in e["output"] for e in events)
            assert any(e["type"] == "tool.call_failed" and e["error_class"] == "permission_denied" for e in events)
            old_sid = app._session_id
            await pilot.press("/", "c")
            await wait_until(lambda: bool(app.query(SlashCompleteWidget)))
            await pilot.press("enter")
            await wait_until(lambda: app._session_id != old_sid and not prompt.disabled)
            assert prompt.text == ""
            assert not app.query(SlashCompleteWidget)
            assert store.read_messages(app._session_id) == []
            assert store.read_meta(old_sid).status == "closed"
            prompt.text = "hello"
            await pilot.press("enter")
            await wait_until(lambda: not app._busy and not prompt.disabled)
            assert len(store.read_messages(app._session_id)) == 2
            sid = app._session_id
            await pilot.press("/", "e")
            await wait_until(lambda: bool(app.query(SlashCompleteWidget)))
            await pilot.press("enter")
            await wait_until(lambda: store.read_meta(sid).status == "closed")
            await wait_until(lambda: not app.is_running)
        assert not app.is_running


# 功能：Windows TUI 授权后的超时或退出取消会终止真实 Shell 子进程，不留下延迟写入。
# 设计：通过输入和审批控件执行带 ready 标记的延迟命令，等过副作用时间后核对文件不存在。
@pytest.mark.parametrize("mode", ["timeout", "cancel"])
async def test_windows_tui_shell_interruption(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    async with _serve(tmp_path, WindowsScenarioProvider(), permission_timeout=10) as (cfg, _, bus, store):
        events = []

        # 保存真实失败分类，不能只检查 Shell 返回了某个输出。
        async def collect(event):
            events.append(event.model_dump())

        bus.subscribe(collect)
        app = XTuiApp(cfg.host, cfg.port)
        async with app.run_test(size=(110, 40)) as pilot:
            await wait_until(lambda: app._session_id and app._client)
            sid = app._session_id
            prompt = app.query_one("#prompt", ChatTextArea)
            advertised_exit = re.search(r"Ctrl\+([A-Z]) 退出", str(app.query_one("#banner", Static).render()))
            assert advertised_exit is not None
            prompt.text = mode
            await pilot.press("enter")
            await wait_until(lambda: bool(app.query(PermissionSelect)))
            await pilot.press("y")
            await wait_until(lambda: (tmp_path / "ready").exists())
            if mode == "cancel":
                await pilot.press("ctrl+" + advertised_exit.group(1).lower())
                await wait_until(lambda: store.read_meta(sid).status == "closed")
                await wait_until(lambda: not app.is_running)
            else:
                await wait_until(lambda: not app._busy and not prompt.disabled)
                assert any(e["type"] == "tool.call_failed" and e["error_class"] == "timeout" for e in events)
        await asyncio.sleep(3.1)
        assert not (tmp_path / "escaped.txt").exists()
