from __future__ import annotations

from pathlib import Path

import pytest

from x_claude.core.bus.envelope import HandlerError
from x_claude.core.events.bus import EventBus
from x_claude.core.runner import RunOutcome
from x_claude.core.session.manager import SESSION_CLOSED, SESSION_NOT_FOUND, SessionManager
from x_claude.core.session.model import Session
from x_claude.core.session.store import SessionStore


class _Runner:
    # 模拟 AgentRunner，将 run 新消息写入 thread 后返回成功
    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str | None = None,
        session: Session | None = None,
        store: SessionStore | None = None,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
    ) -> RunOutcome:
        assert run_id is not None
        assert session is not None
        assert store is not None
        store.append_messages(
            session.id,
            [{"role": "assistant", "content": [{"type": "text", "text": f"done {goal}"}]}],
            run_id,
        )
        return RunOutcome(status="success", result="done", reason=None)


# 功能：验证 create 会创建 active session、写入 meta 并发布 session.created 事件
# 设计：用真实 SessionStore + EventBus 收集事件，覆盖 manager 与 store/bus 的协作边界
async def test_create_session_writes_meta_and_event(tmp_path: Path) -> None:
    events: list[object] = []
    bus = EventBus()

    async def collect(event: object) -> None:
        events.append(event)

    bus.subscribe(collect)
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: _Runner(), bus)  # type: ignore[arg-type]

    session = await manager.create("chat", "title")

    assert session.status == "active"
    assert store.read_meta(session.id).title == "title"
    assert [e.type for e in events] == ["session.created"]  # type: ignore[attr-defined]


# 功能：验证 chat session 处理一条消息后进入 waiting_for_input，并保留 user/assistant thread
# 设计：mock runner 主动追加 assistant 消息，确认 send_message 负责 user 消息、状态流转和 run_id 记录
async def test_send_message_chat_enters_waiting_and_writes_thread(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    session = await manager.create("chat")

    run_id = await manager.send_message(session.id, "hello")

    loaded = store.read_meta(session.id)
    assert loaded.status == "waiting_for_input"
    assert loaded.run_ids == [run_id]
    messages = store.read_messages(session.id)
    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[1]["role"] == "assistant"


# 功能：验证 one_shot session 在单次消息完成后自动 closed
# 设计：复用 mock runner 的成功路径，聚焦 mode 对最终状态的影响，保证 x run 的统一路径正确
async def test_one_shot_auto_closes(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    session = await manager.create("one_shot")

    await manager.send_message(session.id, "hello")

    assert store.read_meta(session.id).status == "closed"


# 功能：验证不存在的 session_id 返回 session_not_found 错误码
# 设计：直接调用 get_history 的查找路径，断言 HandlerError code，覆盖 IPC handler 可结构化返回错误
async def test_missing_session_raises_handler_error(tmp_path: Path) -> None:
    manager = SessionManager(SessionStore(tmp_path), lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    with pytest.raises(HandlerError) as exc:
        await manager.get_history("missing")
    assert exc.value.code == SESSION_NOT_FOUND


# 功能：验证 closed session 不能继续 send_message
# 设计：先显式 close，再发送消息，断言 session_closed 错误码，覆盖状态机拒绝路径
async def test_closed_session_rejects_message(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    session = await manager.create("chat")
    await manager.close(session.id)

    with pytest.raises(HandlerError) as exc:
        await manager.send_message(session.id, "again")
    assert exc.value.code == SESSION_CLOSED


# 功能：验证 clear_context 创建新 session 并完整保留旧 session 的消息和主动 notes
# 设计：直接预置真实持久化文件后调用 manager，断言新旧 session 隔离且旧记录可恢复
async def test_clear_context_starts_new_session_and_preserves_previous_context(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    session = await manager.create("chat")
    store.append_message(session.id, "user", "remember this")
    store.append_note(session.id, "durable fact", "run-1")

    fresh = await manager.clear_context(session.id)

    assert fresh.id != session.id
    assert store.read_messages(fresh.id) == []
    assert store.read_notes(fresh.id) == ""
    assert store.read_messages(session.id) == [{"role": "user", "content": "remember this"}]
    assert "durable fact" in store.read_notes(session.id)
    assert store.read_meta(session.id).id == session.id
    assert store.read_meta(session.id).status == "closed"


# 功能：验证指定 session ID 可在清理后恢复其原始对话上下文
# 设计：先 clear 创建新会话，再由全新 manager 按旧 ID 恢复，覆盖进程重启后的精确续接路径
async def test_resume_restores_cleared_previous_session_by_id(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    original = SessionManager(store, lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    session = await original.create("chat")
    await original.send_message(session.id, "keep this conversation")
    await original.clear_context(session.id)

    restored = SessionManager(store, lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    resumed = await restored.resume(session.id)

    assert resumed.id == session.id
    assert resumed.status == "waiting_for_input"
    assert store.read_messages(session.id)[0] == {"role": "user", "content": "keep this conversation"}


# 功能：验证新建 manager 后能续接磁盘中最近关闭的 chat session
# 设计：先持久化一条 chat 历史并关闭会话，再用全新的 manager 恢复，覆盖进程重启后的发现、装载与状态转换
async def test_continue_latest_reopens_most_recent_persisted_chat_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    original = SessionManager(store, lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    older = await original.create("chat")
    await original.send_message(older.id, "old conversation")
    await original.close(older.id)
    session = await original.create("chat")
    await original.send_message(session.id, "remember this")
    await original.close(session.id)

    restored = SessionManager(store, lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    continued = await restored.continue_latest()

    assert continued.id == session.id
    assert continued.status == "waiting_for_input"
    assert store.read_meta(session.id).status == "waiting_for_input"
    assert store.read_messages(session.id)[0] == {"role": "user", "content": "remember this"}
