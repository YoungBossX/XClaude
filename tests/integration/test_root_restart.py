from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Static

from tests.integration.test_background_restart import _manager
from tests.integration.test_runtime_hardening import _serve
from x_claude.core.bus.commands import SessionRecoverCommand
from x_claude.core.bus.envelope import HandlerError
from x_claude.core.llm.types import LlmResponse, ToolCallBlock
from x_claude.core.session.store import SessionStore
from x_claude.core.subagent.checkpoint import BackgroundCheckpoint
from x_claude.core.tools.builtin.write_file import WriteFileTool
from x_claude.tui.app import ChatTextArea, SlashCompleteWidget, XTuiApp


# 将临时目录作为测试项目，文件工具不能再绕过真实工作目录边界
@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


# 等待主任务恢复协程完成，保留任务引用以防完成后索引被清理
async def _wait_root(manager, sid: str) -> None:
    task = manager._root_tasks.get(sid)
    if task is not None:
        await asyncio.wait_for(task, 3)


# 功能：验证主任务正常关闭后续跑原 run_id、步数、工具结果与记忆，不重复已完成工具
# 设计：真实写文件后阻塞下一次模型推理，销毁 manager，再使用全新实例只从磁盘恢复
async def test_root_graceful_restart_preserves_execution(tmp_path: Path) -> None:
    entered = asyncio.Event()
    target = tmp_path / "once.txt"

    class First:
        # 先写文件，再等待 daemon 退出；第二步不能被误认为已经完成
        async def chat(self, step: int, **kwargs) -> LlmResponse:
            if step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "write", "write_file", {"path": str(target), "content": "once"},
                )])
            entered.set()
            await asyncio.Event().wait()

    steps: list[int] = []

    class Next:
        # 原记忆必须保留，恢复后只能执行第二步推理而非再次调用写文件工具
        async def chat(self, messages: list, step: int, system: str, **kwargs) -> LlmResponse:
            steps.append(step)
            assert step == 2
            assert messages[-1]["content"][0]["tool_use_id"] == "write"
            assert "original-context" in system and "original-note" in system
            return LlmResponse("end_turn", text="root resumed")

    store = SessionStore(tmp_path / "sessions")
    first = _manager(store, First())
    session = await first.create("chat")
    store.append_note(session.id, "original-note", "memory")
    with patch("x_claude.core.runner.load_context_file", return_value="original-context"):
        run = asyncio.create_task(first.send_message(session.id, "write once"))
        await asyncio.wait_for(entered.wait(), 3)
        run_id = session.run_ids[-1]
        path = store.runs_dir(session.id) / run_id / "root.json"
        before = target.stat().st_mtime_ns
        await first.shutdown()
        assert run.cancelled()
    checkpoint = BackgroundCheckpoint.load(path, session.id)
    assert checkpoint.record.state == "suspended" and checkpoint.record.context.step == 1
    second = _manager(SessionStore(store._root), Next())
    try:
        await second.resume(session.id)
        with patch("x_claude.core.runner.load_context_file", return_value="changed-context"):
            await second.recover_background(session.id)
            await _wait_root(second, session.id)
        checkpoint = BackgroundCheckpoint.load(path, session.id)
        assert checkpoint.record.thread_committed and checkpoint.record.context.status == "success"
        assert steps == [2] and target.stat().st_mtime_ns == before
        assert store.read_meta(session.id).run_ids == [run_id]
        assert store.read_messages(session.id, truncate=False) == checkpoint.record.context.messages
        await second.recover_background(session.id)
        assert steps == [2]
    finally:
        await second.shutdown()


# 功能：验证工具批次中断只要求核对未确认的调用，已有结果不被遗漏或重放
# 设计：第一项工具完整返回，第二项执行副作用后阻塞，检查磁盘仅把第二项标为待核对
async def test_root_partial_tool_batch_requires_only_unknown_result(tmp_path: Path) -> None:
    entered = asyncio.Event()
    paths = {key: tmp_path / key for key in ("a", "b")}
    original = WriteFileTool.invoke

    # 第二项真实写入后不返回，模拟外部操作成功但结果未确认的危险窗口
    async def interrupt(tool, params):
        result = await original(tool, params)
        if params["path"] == str(paths["b"]):
            entered.set()
            await asyncio.Event().wait()
        return result

    class First:
        # 在一个步骤内请求两个工具，测试逐项持久化而不是仅保存整批结果
        async def chat(self, **kwargs) -> LlmResponse:
            return LlmResponse("tool_use", tool_calls=[
                ToolCallBlock(key, "write_file", {"path": str(path), "content": key})
                for key, path in paths.items()
            ])

    called: list[int] = []

    class Next:
        # 自动结果和用户核对结果必须按同一批次顺序交给下一步模型
        async def chat(self, messages: list, step: int, **kwargs) -> LlmResponse:
            called.append(step)
            assert [r["tool_use_id"] for r in messages[-1]["content"]] == ["a", "b"]
            assert "User-verified" not in messages[-1]["content"][0]["content"]
            assert "User-verified" in messages[-1]["content"][1]["content"]
            return LlmResponse("end_turn", text="checked and resumed")

    store = SessionStore(tmp_path / "sessions")
    first = _manager(store, First())
    session = await first.create("chat")
    with patch.object(WriteFileTool, "invoke", interrupt):
        run = asyncio.create_task(first.send_message(session.id, "two writes"))
        await asyncio.wait_for(entered.wait(), 3)
        run_id = session.run_ids[-1]
        await first.shutdown()
        assert run.cancelled()
    path = store.runs_dir(session.id) / run_id / "root.json"
    checkpoint = BackgroundCheckpoint.load(path, session.id)
    assert [t["id"] for t in checkpoint.pending_tools()] == ["b"]
    before = {key: p.stat().st_mtime_ns for key, p in paths.items()}
    second = _manager(store, Next())
    try:
        await second.resume(session.id)
        await second.recover_background(session.id)
        assert not called
        history = (store.session_dir(session.id) / "thread.jsonl").read_bytes()
        with pytest.raises(HandlerError, match="unfinished task"):
            await second.send_message(session.id, "do something else")
        assert (store.session_dir(session.id) / "thread.jsonl").read_bytes() == history
        await second.recover(SessionRecoverCommand(
            session_id=session.id, run_id=run_id, tool_results={"b": "verified file content is b"},
        ))
        await _wait_root(second, session.id)
        assert called == [2]
        assert before == {key: p.stat().st_mtime_ns for key, p in paths.items()}
    finally:
        await second.shutdown()


# 功能：验证历史写入后、元数据提交前中断可再次提交，且不调用模型或重复追加回答
# 设计：给最终元数据提交注入一次磁盘故障，然后使用没有模型客户端的全新 manager 恢复
async def test_root_finalization_retry_is_idempotent(tmp_path: Path) -> None:
    class Provider:
        # 直接完成目标，以定位最终历史与元数据两个文件之间的提交窗口
        async def chat(self, **kwargs) -> LlmResponse:
            return LlmResponse("end_turn", text="final answer")

    store = SessionStore(tmp_path / "sessions")
    first = _manager(store, Provider())
    session = await first.create("chat")
    original = store.write_meta
    fired = False

    # 仅在回答已经写入后阻止第一次元数据提交，后续清理仍能完成
    def fail_once(value):
        nonlocal fired
        if not fired and store.read_messages(value.id)[-1]["role"] == "assistant":
            fired = True
            raise OSError("metadata interrupted")
        return original(value)

    with patch.object(store, "write_meta", fail_once):
        run_id = await first.send_message(session.id, "answer")
    path = store.runs_dir(session.id) / run_id / "root.json"
    checkpoint = BackgroundCheckpoint.load(path, session.id)
    assert fired and not checkpoint.record.thread_committed
    before = (store.session_dir(session.id) / "thread.jsonl").read_bytes()
    await first.shutdown()
    second = _manager(store, None)
    try:
        await second.resume(session.id)
        with patch("x_claude.core.runner.AnthropicProvider", side_effect=AssertionError("no LLM")):
            await second.recover_background(session.id)
            await _wait_root(second, session.id)
        assert (store.session_dir(session.id) / "thread.jsonl").read_bytes() == before
        assert BackgroundCheckpoint.load(path, session.id).record.thread_committed
        assert len(store.read_messages(session.id)) == 2
    finally:
        await second.shutdown()


# 功能：验证 /clear 和 /exit 对运行中主任务执行永久取消，而非保留为自动续跑
# 设计：阻塞第一次推理，直接调用真实会话清空或关闭 API，再恢复旧会话检查取消记录
@pytest.mark.parametrize("operation", ["clear", "close"])
async def test_explicit_stop_cancels_root_permanently(tmp_path: Path, operation: str) -> None:
    entered = asyncio.Event()

    class Provider:
        # 保持推理进行中，要求关闭命令能主动取消持有会话锁的主任务
        async def chat(self, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    store = SessionStore(tmp_path / "sessions")
    manager = _manager(store, Provider())
    session = await manager.create("chat")
    task = asyncio.create_task(manager.send_message(session.id, "long task"))
    await asyncio.wait_for(entered.wait(), 3)
    run_id = session.run_ids[-1]
    if operation == "clear":
        new = await manager.clear_context(session.id)
        assert new.id != session.id and not store.read_messages(new.id)
    else:
        await manager.close(session.id)
    assert task.cancelled()
    checkpoint = BackgroundCheckpoint.load(store.runs_dir(session.id) / run_id / "root.json", session.id)
    assert checkpoint.record.state == "cancelled"
    await manager.resume(session.id)
    await manager.recover_background(session.id)
    assert session.id not in manager._root_tasks
    await manager.shutdown()


# 功能：验证所有工具结果都已确认、但步骤边界通知前中断时可自动续跑而不再人工核对
# 设计：在 ready 检查点前注入取消，磁盘仍处于 tools 阶段但 pending_tools 为空
async def test_all_tool_results_confirmed_can_resume_automatically(tmp_path: Path) -> None:
    target = tmp_path / "once.txt"

    class First:
        # 返回单个写工具，准备在结果逐项落盘后触发中断
        async def chat(self, **kwargs):
            return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                "write", "write_file", {"path": str(target), "content": "once"},
            )])

    save = BackgroundCheckpoint.save

    # 只在工具结果已写入后、整步 ready 提交前取消，不影响逐项工具结果检查点
    def interrupt(checkpoint, context, phase):
        if phase == "ready" and context.step == 1:
            raise asyncio.CancelledError()
        save(checkpoint, context, phase)

    store = SessionStore(tmp_path / "sessions")
    first = _manager(store, First())
    session = await first.create("chat")
    first._runners[session.id] = first._runner_factory()
    first._runners[session.id].prepare_shutdown(suspend=True)
    with patch.object(BackgroundCheckpoint, "save", interrupt):
        with pytest.raises(asyncio.CancelledError):
            await first.send_message(session.id, "write once")
    run_id = session.run_ids[-1]
    path = store.runs_dir(session.id) / run_id / "root.json"
    checkpoint = BackgroundCheckpoint.load(path, session.id)
    assert checkpoint.record.phase == "tools" and not checkpoint.pending_tools()
    before = target.stat().st_mtime_ns
    await first.shutdown()

    class Next:
        # 已确认的结果直接交给下一步，不再次发起工具调用
        async def chat(self, step: int, **kwargs):
            assert step == 2
            return LlmResponse("end_turn", text="done")

    second = _manager(store, Next())
    try:
        await second.resume(session.id)
        await second.recover_background(session.id)
        await _wait_root(second, session.id)
        assert target.stat().st_mtime_ns == before
        assert BackgroundCheckpoint.load(path, session.id).record.thread_committed
    finally:
        await second.shutdown()


# 功能：验证第一次推理前的通知阶段被取消时，主任务也已有检查点并能跨重启恢复
# 设计：阻塞 run.started 通知，确保模型尚未调用，然后正常关闭并用全新 manager 续跑
async def test_root_checkpoint_exists_before_first_notification(tmp_path: Path) -> None:
    entered = asyncio.Event()

    class Provider:
        # 第一次通知被阻塞时不能提前调用模型，恢复后才完成原目标
        async def chat(self, **kwargs):
            return LlmResponse("end_turn", text="done")

    first = _manager(SessionStore(tmp_path / "sessions"), Provider())
    session = await first.create("chat")

    # 在模型调用之前拦截通知，以覆盖循环外准备阶段的恢复边界
    async def block(event):
        if event.type == "run.started":
            entered.set()
            await asyncio.Event().wait()

    first._bus.subscribe(block)
    task = asyncio.create_task(first.send_message(session.id, "original goal"))
    await asyncio.wait_for(entered.wait(), 2)
    run_id = session.run_ids[-1]
    path = first._store.runs_dir(session.id) / run_id / "root.json"
    assert path.exists()
    await first.shutdown()
    assert task.cancelled()
    assert BackgroundCheckpoint.load(path, session.id).record.state == "suspended"
    second = _manager(first._store, Provider())
    try:
        await second.resume(session.id)
        await second.recover_background(session.id)
        await _wait_root(second, session.id)
        assert BackgroundCheckpoint.load(path, session.id).record.thread_committed
    finally:
        await second.shutdown()


# 功能：验证实际工具成功但结果检查点提交失败时，不自动重放而要求用户核对
# 设计：工具确实写入文件，仅拒绝观察结果的落盘，检查未知调用仍保留且没有新工具执行
async def test_root_tool_result_commit_failure_stays_unknown(tmp_path: Path) -> None:
    target = tmp_path / "once.txt"

    class Provider:
        # 先写文件，再在下一步完成；恢复只能在人工确认后进入第二步
        async def chat(self, step: int, **kwargs):
            if step == 1:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "write", "write_file", {"path": str(target), "content": "once"},
                )])
            return LlmResponse("end_turn", text="done")

    manager = _manager(SessionStore(tmp_path / "sessions"), Provider())
    session = await manager.create("chat")
    save = BackgroundCheckpoint.save

    # 工具执行前的检查点成功，返回结果后的提交失败，模拟不可消除的副作用确认窗口
    def fail_result(checkpoint, context, phase):
        if phase == "tools" and context.messages[-1]["role"] == "user":
            raise OSError("result journal unavailable")
        save(checkpoint, context, phase)

    try:
        with patch.object(BackgroundCheckpoint, "save", fail_result):
            run_id = await manager.send_message(session.id, "write once")
        path = manager._store.runs_dir(session.id) / run_id / "root.json"
        checkpoint = BackgroundCheckpoint.load(path, session.id)
        assert [t["id"] for t in checkpoint.pending_tools()] == ["write"]
        assert target.read_text() == "once"
        before = target.stat().st_mtime_ns
        await manager.recover_background(session.id)
        assert not manager._root_tasks
        await manager.recover(SessionRecoverCommand(
            session_id=session.id, run_id=run_id, tool_results={"write": "verified once.txt contains once"},
        ))
        await _wait_root(manager, session.id)
        assert target.stat().st_mtime_ns == before
        assert BackgroundCheckpoint.load(path, session.id).record.thread_committed
    finally:
        await manager.shutdown()


# 功能：验证真实 TUI 选择 /exit 会先取消运行中的主任务，再退出而非先断开控制通道
# 设计：通过 TCP 启动一个阻塞目标，触发斜杠菜单退出，检查服务端的取消记录及会话关闭状态
async def test_tui_exit_cancels_running_root(tmp_path: Path) -> None:
    entered = asyncio.Event()

    class Provider:
        # 保持主任务推理中，直到退出流程主动取消
        async def chat(self, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    async with _serve(tmp_path, Provider()) as (cfg, manager, _, store):
        app = XTuiApp(cfg.host, cfg.port)
        async with app.run_test() as pilot:
            for _ in range(100):
                if app._session_id:
                    break
                await asyncio.sleep(0.01)
            sid = app._session_id
            assert sid is not None
            prompt = app.query_one("#prompt", ChatTextArea)
            prompt.text = "long task"
            await app.on_chat_text_area_submitted(ChatTextArea.Submitted(prompt))
            await asyncio.wait_for(entered.wait(), 2)
            run_id = manager._sessions[sid].run_ids[-1]
            app.on_slash_complete_widget_selected(SlashCompleteWidget.Selected("exit"))
            await pilot.pause()
            # 退出 worker 跨多个 RPC/文件写入，等待关闭提交而不是假设一帧即可完成
            for _ in range(300):
                if store.read_meta(sid).status == "closed":
                    break
                await asyncio.sleep(.01)
            assert store.read_meta(sid).status == "closed"
            checkpoint = BackgroundCheckpoint.load(store.runs_dir(sid) / run_id / "root.json", sid)
            assert checkpoint.record.state == "cancelled"


_CRASH_ROOT = '''
import asyncio, json, sys
from pathlib import Path
from x_claude.core.config import XConfig
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.types import LlmResponse, ToolCallBlock
from x_claude.core.runner import AgentRunner
from x_claude.core.session.manager import SessionManager
from x_claude.core.session.store import SessionStore
from x_claude.core.tools.builtin.write_file import WriteFileTool
mode = sys.argv[2]
entered = asyncio.Event()
original = WriteFileTool.invoke
async def interrupted(tool, params):
    result = await original(tool, params)
    if mode == "tools" and params["path"] == "b.txt":
        entered.set()
        await asyncio.Future()
    return result
WriteFileTool.invoke = interrupted
class Provider:
    async def chat(self, step, **kwargs):
        if step == 1:
            return LlmResponse("tool_use", tool_calls=[ToolCallBlock(key, "write_file", {"path":key+".txt","content":key}) for key in ("a", "b")])
        entered.set()
        await asyncio.Future()
async def main():
    cfg = XConfig(); cfg.compaction.auto_threshold = 0
    store = SessionStore(Path(sys.argv[1]) / "sessions"); bus = EventBus()
    manager = SessionManager(store, lambda: AgentRunner(cfg, provider=Provider(), bus=bus), bus)
    session = await manager.create("chat")
    task = asyncio.create_task(manager.send_message(session.id, "root crash"))
    await entered.wait()
    print(json.dumps({"sid":session.id, "run":session.run_ids[-1]}), flush=True)
    await asyncio.Future()
asyncio.run(main())
'''


# 功能：验证主任务进程被强制终止后，TUI 能续跑安全阶段或核对唯一未知工具后续跑
# 设计：独立进程真实执行两个写入并 kill，新 TCP Core 与 Textual 只依赖检查点恢复
@pytest.mark.parametrize("mode", ["planning", "tools"])
async def test_root_process_crash_then_tui_resume(tmp_path: Path, monkeypatch, mode: str) -> None:
    monkeypatch.chdir(tmp_path)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c", _CRASH_ROOT, str(tmp_path), mode, cwd=tmp_path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    try:
        assert proc.stdout is not None
        info = json.loads(await asyncio.wait_for(proc.stdout.readline(), 8))
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
    times = [(tmp_path / name).stat().st_mtime_ns for name in ("a.txt", "b.txt")]
    calls: list[int] = []

    class Provider:
        # 拒绝从头执行，验证真实进程崩溃后仍保留第二步所需结果
        async def chat(self, messages: list, step: int, **kwargs) -> LlmResponse:
            assert step == 2
            calls.append(step)
            assert [b["tool_use_id"] for b in messages[-1]["content"]] == ["a", "b"]
            return LlmResponse("end_turn", text="crash recovered")

    async with _serve(tmp_path, Provider()) as (cfg, manager, _, store):
        app = XTuiApp(cfg.host, cfg.port, resume_session_id=info["sid"])
        async with app.run_test() as pilot:
            for _ in range(150):
                if app._session_id:
                    break
                await asyncio.sleep(0.01)
            assert app._session_id == info["sid"]
            await pilot.pause()
            if mode == "tools":
                assert not calls
                await app._do_recover(info["run"] + ' {"b":"verified b.txt contains b"}')
            await _wait_root(manager, info["sid"])
            await pilot.pause()
            text = "\n".join(str(w.content) for w in app.query(Static))
            assert "主任务" in text and info["run"] in text
            assert not app._busy and not app.query_one("#prompt", ChatTextArea).disabled
            checkpoint = BackgroundCheckpoint.load(
                store.runs_dir(info["sid"]) / info["run"] / "root.json", info["sid"],
            )
            assert checkpoint.record.thread_committed
            assert calls == [2]
            assert times == [(tmp_path / name).stat().st_mtime_ns for name in ("a.txt", "b.txt")]
