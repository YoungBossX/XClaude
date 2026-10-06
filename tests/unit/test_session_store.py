from __future__ import annotations

from pathlib import Path

import pytest

from x_claude.core.session.model import Session
from x_claude.core.session.store import SessionStore


# 功能：验证运行结果批量追加失败时不会留下半批消息或改变原历史
# 设计：第二条消息序列化失败，覆盖压缩后保存多条回复时原来的逐条追加风险
def test_batch_append_serialization_failure_preserves_original(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "summary")
    path = store.session_dir("sess-1") / "thread.jsonl"
    original = path.read_bytes()
    with pytest.raises(TypeError):
        store.append_messages("sess-1", [
            {"role": "assistant", "content": "new reply"},
            {"role": "user", "content": object()},
        ], run_id="run-1")
    assert path.read_bytes() == original


# 功能：验证压缩历史序列化失败时原文件逐字节保留
# 设计：第二条消息不可序列化，覆盖已经准备部分新历史但尚未提交的失败路径
def test_compaction_serialization_failure_preserves_original(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "原始历史")
    path = store.session_dir("sess-1") / "thread.jsonl"
    original = path.read_bytes()
    with pytest.raises(TypeError):
        store.write_compacted("sess-1", [
            {"role": "user", "content": "summary"},
            {"role": "assistant", "content": object()},
        ])
    assert path.read_bytes() == original


# 功能：验证最终原子替换失败时原历史可读且没有临时文件残留
# 设计：只阻断 thread.jsonl 的 os.replace，让备份正常完成，再检查原文件和备份都完整
@pytest.mark.parametrize("operation", ["compact", "append"])
def test_compaction_replace_failure_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    import os

    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "original")
    path = store.session_dir("sess-1") / "thread.jsonl"
    original = path.read_bytes()
    real_replace = os.replace

    # 在新历史提交点模拟 Windows 文件锁造成的替换失败
    def fail_thread_replace(source: object, target: object) -> None:
        if Path(str(target)) == path:
            raise PermissionError("thread is locked")
        real_replace(source, target)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", fail_thread_replace)
    with pytest.raises(PermissionError):
        if operation == "compact":
            store.write_compacted("sess-1", [{"role": "user", "content": "summary"}])
        else:
            store.append_messages(
                "sess-1", [{"role": "assistant", "content": "reply"}], run_id="run-1",
            )
    assert path.read_bytes() == original
    if operation == "compact":
        assert next(path.parent.glob("thread_*.jsonl.bak")).read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))


# 功能：验证磁盘写入同步失败时原历史保持完整且临时文件被清理
# 设计：在临时文件已写入后注入 fsync 异常，覆盖磁盘满等提交前 I/O 故障
def test_compaction_sync_failure_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "original")
    path = store.session_dir("sess-1") / "thread.jsonl"
    original = path.read_bytes()

    # 模拟磁盘无法同步完整临时文件
    def fail_sync(descriptor: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError):
        store.write_compacted("sess-1", [{"role": "user", "content": "summary"}])
    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))


# 功能：验证连续压缩能成功保存且每次备份拥有独立路径
# 设计：同一会话立即压缩两次，防止秒级时间戳重复造成 Windows rename 失败或覆盖已有备份
def test_repeated_compaction_has_unique_backups(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "original")
    first = [{"role": "user", "content": "first summary"}]
    second = [{"role": "user", "content": "second summary"}]
    store.write_compacted("sess-1", first)
    store.write_compacted("sess-1", second)
    assert store.read_messages("sess-1") == second
    assert len(list(store.session_dir("sess-1").glob("thread_*.jsonl.bak"))) == 2


# 功能：验证 SessionStore 初始化时自动创建 sessions 根目录
# 设计：传入 tmp_path 下不存在的目录，断言目录被创建，覆盖首次启动 daemon 的冷路径
def test_store_creates_root(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    SessionStore(root)
    assert root.exists()


# 功能：验证 session meta 写入后能完整读回
# 设计：构造含 run_ids 的 Session，经过 JSON 文件往返后断言字段保持，覆盖 meta.json 的持久化契约
def test_meta_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = Session(
        id="sess-1",
        mode="chat",
        status="waiting_for_input",
        title="hello",
        created_at="t1",
        updated_at="t2",
        run_ids=["run-1"],
    )
    store.write_meta(session)
    loaded = store.read_meta("sess-1")
    assert loaded == session


# 功能：验证含 tool_use/tool_result block 的 thread 消息能按 Anthropic 格式读回
# 设计：追加 assistant tool_use 和 user tool_result，读取时应剥离 ts/run_id，只保留 API messages 所需字段
def test_thread_message_roundtrip_with_tool_blocks(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "read file")
    store.append_message(
        "sess-1",
        "assistant",
        [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "x"}}],
        run_id="run-1",
    )
    store.append_message(
        "sess-1",
        "user",
        [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        run_id="run-1",
    )

    messages = store.read_messages("sess-1")
    assert messages == [
        {"role": "user", "content": "read file"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "x"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        },
    ]


# 功能：验证 thread 尾部孤儿 tool_use 会被裁掉
# 设计：构造一条未配对 tool_result 的 assistant tool_use，读取时只返回最后一次配平之前的消息，避免 API 报 messages.invalid
def test_read_messages_trims_orphan_tool_use_tail(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "hello")
    store.append_message(
        "sess-1",
        "assistant",
        [{"type": "tool_use", "id": "orphan", "name": "read_file", "input": {}}],
        run_id="run-1",
    )
    assert store.read_messages("sess-1") == [{"role": "user", "content": "hello"}]


# 功能：验证 notes.md 不存在时读为空，追加笔记后能读到内容和 run_id
# 设计：先读空状态再追加，覆盖 chat 第一轮前和 note_save 调用后的两个关键状态
def test_notes_read_and_append(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    assert store.read_notes("sess-1") == ""
    store.append_note("sess-1", "Python 3.12", "run-1")
    notes = store.read_notes("sess-1")
    assert "Python 3.12" in notes
    assert "run-1" in notes
