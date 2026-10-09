from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.integration.test_background_restart import _manager
from tests.integration.test_root_restart import _wait_root
from tests.integration.test_runtime_hardening import _serve
from x_claude.core.bus.events import ToolCallFinishedEvent
from x_claude.core.llm.types import LlmResponse, ToolCallBlock
from x_claude.core.permissions.manager import PermissionManager
from x_claude.core.permissions.policy import PermissionDecision, ToolPolicy
from x_claude.core.session.store import SessionStore
from x_claude.core.subagent.checkpoint import BackgroundCheckpoint
from x_claude.tui.app import ChatTextArea, XTuiApp
from x_claude.tui.app import ToolCallBlock as ToolWidget


# 将所有 Agent 文件操作与持久检查点限制在临时项目中
@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


# 功能：验证并发子 Agent 的旧版本写入被拒绝，TUI 展示冲突，重新读取后保留双方修改
# 设计：真实 TCP、主任务派生后台子任务与 Textual 渲染，模型替身用屏障固定冲突顺序
async def test_background_agents_merge_after_visible_conflict(tmp_path: Path) -> None:
    target = tmp_path / "shared.txt"
    target.write_bytes(b"base")
    both_read = asyncio.Event()
    a_written = asyncio.Event()
    b_merged = asyncio.Event()
    readers: set[str] = set()
    failures: list = []
    finished_versions: dict[str, dict[str, str | None]] = {}

    class Provider:
        # 两个子任务读相同原文，B 在 A 提交后尝试旧版本，再读新内容合并
        async def chat(self, messages: list, step: int, **kwargs) -> LlmResponse:
            goal = messages[0]["content"]
            if goal not in {"A", "B"}:
                if step == 1:
                    return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                        f"spawn-{name}", "spawn_agent",
                        {"description": name, "prompt": name, "run_in_background": True},
                    ) for name in ("A", "B")])
                return LlmResponse("end_turn", text="children launched")
            if step == 1 or (goal == "B" and step == 3):
                if step == 3:
                    assert messages[-1]["content"][0]["is_error"] is True
                    assert "file_conflict" in messages[-1]["content"][0]["content"]
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    f"{goal}-read-{step}", "read_file", {"path": str(target)},
                )])
            if step == 2:
                assert messages[-1]["content"][0]["content"] == "base"
                readers.add(goal)
                if len(readers) == 2:
                    both_read.set()
                await asyncio.wait_for(both_read.wait(), 5)
                if goal == "B":
                    await asyncio.wait_for(a_written.wait(), 5)
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    f"{goal}-write", "write_file", {"path": str(target), "content": "base\n" + goal},
                )])
            if goal == "B" and step == 4:
                content = messages[-1]["content"][0]["content"]
                assert content == "base\nA"
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "B-merge", "write_file", {"path": str(target), "content": content + "\nB"},
                )])
            return LlmResponse("end_turn", text="merged")

    permissions = PermissionManager({name: ToolPolicy(PermissionDecision.ALLOW)
                                     for name in ("spawn_agent", "write_file", "read_file")})
    with patch("tests.integration.test_runtime_hardening.PermissionManager", return_value=permissions):
        async with _serve(tmp_path, Provider()) as (cfg, manager, bus, store):
            # 观察真实完成事件作为屏障，不用定时等待猜测文件是否已提交
            async def observe(event):
                if isinstance(event, ToolCallFinishedEvent):
                    if event.tool_use_id == "A-write":
                        a_written.set()
                    elif event.tool_use_id == "B-merge":
                        b_merged.set()
                if getattr(event, "type", "") == "tool.call_failed":
                    failures.append(event)
                if getattr(event, "type", "") == "subagent.finished":
                    entry = manager._runners[event.session_id]._task_registry.get(event.run_id)
                    assert entry is not None
                    finished_versions[event.run_id] = dict(entry[1].file_versions)

            bus.subscribe(observe)
            app = XTuiApp(cfg.host, cfg.port)
            async with app.run_test() as pilot:
                async with asyncio.timeout(5):
                    while app._session_id is None:
                        await asyncio.sleep(.01)
                sid = app._session_id
                prompt = app.query_one("#prompt", ChatTextArea)
                prompt.text = "launch two editors"
                await app.on_chat_text_area_submitted(ChatTextArea.Submitted(prompt))
                await asyncio.wait_for(b_merged.wait(), 8)
                tasks = manager._runners[sid]._task_registry.all()
                await asyncio.wait_for(asyncio.gather(*(task for task, _ in tasks)), 5)
                await pilot.pause()
                assert len(tasks) == 2 and all(ctx.status == "success" for _, ctx in tasks)
                assert target.read_bytes() == b"base\nA\nB"
                assert len(failures) == 1 and failures[0].error_class == "file_conflict"
                conflicts = [widget for widget in app.query(ToolWidget)
                             if "file_conflict" in widget._output]
                assert len(conflicts) == 1 and conflicts[0]._is_error
                conflicts[0].on_click()
                assert "expanded" in conflicts[0].classes
                for _, ctx in tasks:
                    path = store.runs_dir(sid) / ctx.run_id / "background.json"
                    saved = BackgroundCheckpoint.load(path, sid).record.context
                    assert saved.file_versions == finished_versions[ctx.run_id] and saved.file_versions
                cached = manager._runners[sid]._task_registry.all()
                assert all(not ctx.file_versions and not ctx.messages for _, ctx in cached)


# 功能：验证主任务停机续跑保留读取版本，文件未变可写，停机期间被修改则拒绝旧写入
# 设计：销毁并重建真实 SessionManager，只由磁盘检查点恢复执行而不复用内存版本表
@pytest.mark.parametrize("external_edit", [False, True])
async def test_root_restart_preserves_file_version(tmp_path: Path, external_edit: bool) -> None:
    target = tmp_path / "shared.txt"
    target.write_bytes(b"original")
    entered = asyncio.Event()

    class First:
        # 完整读取后在推理阶段等待停机，确保检查点中已提交读取版本
        async def chat(self, step: int, **kwargs) -> LlmResponse:
            if step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "read", "read_file", {"path": str(target)},
                )])
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    class Next:
        # 从第二步尝试原计划修改，并明确检查模型确实收到冲突或成功结果
        async def chat(self, messages: list, step: int, **kwargs) -> LlmResponse:
            if step == 2:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "write", "write_file", {"path": str(target), "content": "planned edit"},
                )])
            assert step == 3
            result = messages[-1]["content"][0]
            assert result.get("is_error", False) is external_edit
            if external_edit:
                assert "file_conflict" in result["content"]
            return LlmResponse("end_turn", text="checked")

    store = SessionStore(tmp_path / "sessions")
    first = _manager(store, First())
    session = await first.create("chat")
    run = asyncio.create_task(first.send_message(session.id, "edit"))
    try:
        await asyncio.wait_for(entered.wait(), 5)
    finally:
        await first.shutdown()
        await asyncio.gather(run, return_exceptions=True)
    run_id = session.run_ids[-1]
    path = store.runs_dir(session.id) / run_id / "root.json"
    before = BackgroundCheckpoint.load(path, session.id)
    assert before.record.context.file_versions and before.record.state == "suspended"
    if external_edit:
        target.write_bytes(b"external edit")
    second = _manager(SessionStore(store._root), Next())
    try:
        await second.resume(session.id)
        await second.recover_background(session.id)
        await _wait_root(second, session.id)
        after = BackgroundCheckpoint.load(path, session.id)
        assert after.record.context.status == "success" and after.record.thread_committed
        assert target.read_bytes() == (b"external edit" if external_edit else b"planned edit")
        assert store.read_messages(session.id, truncate=False) == after.record.context.messages
    finally:
        await second.shutdown()
