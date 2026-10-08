from __future__ import annotations

import asyncio

from rich.markdown import Markdown
from textual.widget import Widget

from x_claude.tui.app import (
    LLMStreamBlock,
    SlashCompleteWidget,
    ToolCallBlock,
    XTuiApp,
    _param_summary,
    _preview,
)


# 功能：验证 TUI 欢迎横幅展示 XCLAUDE 而非旧项目名称
# 设计：断言块状字形的首行，直接覆盖用户可见的品牌区域，同时避免绑定底部操作提示的非品牌文案
def test_banner_uses_xclaude_branding() -> None:
    assert "██   ██   █████  ██       █████  ██   ██ ██████  ███████" in XTuiApp._BANNER


# 功能：验证 TUI 接收 --continue 时保留最近会话续接标记
# 设计：仅构造应用并断言内部标记，隔离 CLI 参数解析和网络连接，确保 socket 建立时可选择续接路径
def test_tui_stores_continue_session_flag() -> None:
    app = XTuiApp("127.0.0.1", 9999, continue_session=True)
    assert app._continue_session  # type: ignore[attr-defined]


# 功能：验证 TUI 接收 --resume 时保留指定 session ID
# 设计：仅构造应用并断言内部参数，隔离 CLI 解析与网络连接，确保 socket 建立时优先精确续接
def test_tui_stores_resume_session_id() -> None:
    app = XTuiApp("127.0.0.1", 9999, resume_session_id="sess-previous")
    assert app._resume_session_id == "sess-previous"  # type: ignore[attr-defined]


# 功能：验证斜杠补全列表展示内置 /exit 退出命令
# 设计：直接检查候选项，避免依赖 Textual 渲染，同时防止内置命令遗漏导致用户无法发现
def test_slash_completion_lists_exit_command() -> None:
    app = XTuiApp("127.0.0.1", 9999)
    assert ("exit", "exit TUI") in app._build_slash_items()


# 功能：验证在补全框选中 /exit 后立即启动安全退出
# 设计：直接发送 Selected 事件，断言仅调度退出协程，避免依赖终端按键和实际 socket
def test_slash_completion_selecting_exit_starts_safe_quit() -> None:
    class _FakePrompt:
        def __init__(self) -> None:
            self.text = "/e"

    class _FakePopup:
        def remove(self) -> None:
            pass

    app = XTuiApp("127.0.0.1", 9999)
    workers: list[tuple[object, dict[str, object]]] = []
    prompt = _FakePrompt()
    popup = _FakePopup()
    app.run_worker = lambda coroutine, **kwargs: workers.append((coroutine, kwargs))  # type: ignore[method-assign]
    app.query_one = lambda selector, *args: prompt if selector == "#prompt" else popup  # type: ignore[method-assign]

    try:
        app.on_slash_complete_widget_selected(SlashCompleteWidget.Selected("exit"))
        assert len(workers) == 1
        assert workers[0][1] == {"name": "quit", "group": "shutdown", "exclusive": True}
        assert prompt.text == ""
    finally:
        for coroutine, _ in workers:
            coroutine.close()  # type: ignore[union-attr]


# 功能：验证在补全框选中 /clear 后立即启动上下文清理
# 设计：直接发送 Selected 事件，断言路由到 clear worker，避免再次要求用户确认输入框内容
def test_slash_completion_selecting_clear_starts_context_clear() -> None:
    class _FakePrompt:
        def __init__(self) -> None:
            self.text = "/c"

    class _FakePopup:
        def remove(self) -> None:
            pass

    app = XTuiApp("127.0.0.1", 9999)
    workers: list[tuple[object, dict[str, object]]] = []
    prompt = _FakePrompt()
    popup = _FakePopup()
    app.run_worker = lambda coroutine, **kwargs: workers.append((coroutine, kwargs))  # type: ignore[method-assign]
    app.query_one = lambda selector, *args: prompt if selector == "#prompt" else popup  # type: ignore[method-assign]

    try:
        app.on_slash_complete_widget_selected(SlashCompleteWidget.Selected("clear"))
        assert len(workers) == 1
        assert workers[0][1] == {"name": "clear", "exclusive": False}
        assert prompt.text == ""
    finally:
        for coroutine, _ in workers:
            coroutine.close()  # type: ignore[union-attr]


# 功能：验证选中 /compact 后立即压缩上下文并清空输入草稿
# 设计：模拟补全选择与输入框，断言无需二次回车，且残留的 /c 不会留在下一条消息中
def test_slash_completion_selecting_compact_starts_context_compaction() -> None:
    class _FakePrompt:
        def __init__(self) -> None:
            self.text = "/c"

    class _FakePopup:
        def remove(self) -> None:
            pass

    app = XTuiApp("127.0.0.1", 9999)
    workers: list[tuple[object, dict[str, object]]] = []
    prompt = _FakePrompt()
    popup = _FakePopup()
    app.run_worker = lambda coroutine, **kwargs: workers.append((coroutine, kwargs))  # type: ignore[method-assign]
    app.query_one = lambda selector, *args: prompt if selector == "#prompt" else popup  # type: ignore[method-assign]

    try:
        app.on_slash_complete_widget_selected(SlashCompleteWidget.Selected("compact"))
        assert len(workers) == 1
        assert workers[0][1] == {"name": "compact", "exclusive": False}
        assert prompt.text == ""
    finally:
        for coroutine, _ in workers:
            coroutine.close()  # type: ignore[union-attr]


# 功能：验证 _preview 超出长度时截断并追加省略号
# 设计：不依赖任何 TUI 组件，纯函数测试
def test_preview_truncates() -> None:
    assert _preview("abcde", 3) == "abc…"
    assert _preview("ab", 5) == "ab"


# 功能：验证工具参数摘要优先展示工具最关键字段
# 设计：覆盖 read_file/bash/note_save 三类常见工具，避免工具块摘要退化成整段 JSON
def test_param_summary_prefers_key_fields() -> None:
    assert _param_summary("read_file", {"path": "README.md"}) == "path='README.md'"
    assert _param_summary("bash", {"command": "echo hi", "timeout": 1}) == "command='echo hi'"
    assert _param_summary("note_save", {"content": "Python 3.12"}) == "content='Python 3.12'"


# 功能：验证 llm.token 事件累积到 LLMStreamBlock，不连续 token 各自新开一块
# 设计：monkey-patch _append 收集追加的 widgets，断言 token 追加到同一块；
#       发送非 token 事件后新 block 被重置，下一个 token 开启新块
def test_llm_tokens_accumulate_in_block() -> None:
    app = XTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({"type": "llm.token", "token": "Hello", "run_id": "r", "ts": "t"})
    app._handle_event({"type": "llm.token", "token": " world", "run_id": "r", "ts": "t"})

    assert len(appended) == 1  # same block reused
    assert isinstance(appended[0], LLMStreamBlock)
    assert appended[0]._text == "Hello world"  # type: ignore[attr-defined]


# 功能：验证 LLMStreamBlock 结束时会把累积文本渲染为 Rich Markdown
# 设计：直接调用 finalize_markdown，断言 renderable 类型，覆盖 Markdown polish 的核心行为
def test_llm_block_finalize_renders_markdown() -> None:
    block = LLMStreamBlock()
    block.append_token("## Title\n\n- one\n\n```python\nprint('hi')\n```")
    block.finalize_markdown()
    assert isinstance(block.content, Markdown)


# 功能：验证非 token 事件后 _current_llm 被重置，下一个 token 开启新块
# 设计：插入 step.started 中断流，验证之前的 block 被 finalize，之后的 llm.token 创建新 LLMStreamBlock
def test_llm_block_resets_after_non_token_event() -> None:
    app = XTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({"type": "llm.token", "token": "A", "run_id": "r", "ts": "t"})
    app._handle_event({"type": "step.started", "run_id": "r", "step": 2, "ts": "t"})
    app._handle_event({"type": "llm.token", "token": "B", "run_id": "r", "ts": "t"})

    llm_blocks = [w for w in appended if isinstance(w, LLMStreamBlock)]
    assert len(llm_blocks) == 2
    assert llm_blocks[0]._finalized  # type: ignore[attr-defined]


# 功能：验证 run.started 事件追加 Static widget 且包含 run_id 和 goal
# 设计：monkey-patch _append，断言追加的 widget 的 renderable 包含关键字段
def test_run_started_appends_widget_with_content() -> None:
    app = XTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({
        "type": "run.started", "run_id": "run-abc", "goal": "do the thing", "ts": "t"
    })

    assert len(appended) == 1
    rendered = appended[0].content
    assert "run-abc" in rendered
    assert "do the thing" in rendered


# 功能：验证 run.finished success 追加包含 "completed" 的 widget
# 设计：monkey-patch _append，检查 rendered 内容包含 completed 和 green
def test_run_finished_success_shows_completed() -> None:
    app = XTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({
        "type": "run.finished", "run_id": "r", "status": "success", "steps": 3, "ts": "t"
    })

    rendered = appended[0].content
    assert "completed" in rendered
    assert "green" in rendered


# 功能：验证 run.finished failed 追加包含 "failed" 和 red 的 widget
# 设计：与 success 对称，检查颜色标记差异
def test_run_finished_failed_shows_red() -> None:
    app = XTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({
        "type": "run.finished", "run_id": "r", "status": "failed",
        "steps": 1, "reason": "llm_error", "ts": "t"
    })

    rendered = appended[0].content
    assert "failed" in rendered
    assert "red" in rendered


# 功能：验证 tool.call_started 追加 ToolCallBlock，call_finished 更新其结果
# 设计：直接调用 _handle_event 两次，通过 _pending_tool_blocks 验证状态流转
def test_tool_call_started_and_finished() -> None:
    app = XTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({
        "type": "tool.call_started",
        "tool_use_id": "uid-1",
        "tool_name": "bash",
        "params": {"command": "echo hi"},
        "run_id": "r", "ts": "t",
    })
    assert "uid-1" in app._pending_tool_blocks  # type: ignore[attr-defined]

    app._handle_event({
        "type": "tool.call_finished",
        "tool_use_id": "uid-1",
        "tool_name": "bash",
        "elapsed_ms": 42,
        "output": "hi",
        "run_id": "r", "ts": "t",
    })
    assert "uid-1" not in app._pending_tool_blocks  # type: ignore[attr-defined]
    block = appended[0]
    assert isinstance(block, ToolCallBlock)
    assert block._finished  # type: ignore[attr-defined]
    assert block._output == "hi"  # type: ignore[attr-defined]


# 功能：验证 note_save 成功完成时工具块摘要显示 remembered
# 设计：直接操作 ToolCallBlock，覆盖 note_save 的特殊低噪声展示策略
def test_note_save_tool_block_shows_remembered() -> None:
    block = ToolCallBlock("note_save", {"content": "Python 3.12"})
    block.set_result("saved", 3)
    assert "remembered" in block._summary()  # type: ignore[attr-defined]


# 功能：验证输入 /exit 时直接启动安全退出，而不发送给 agent
# 设计：捕获 run_worker 的协程与名称，断言只调度 action_quit 且不添加用户消息，避免依赖真实 TUI 生命周期
async def test_exit_command_starts_safe_quit_without_sending_message() -> None:
    class _FakeArea:
        def __init__(self) -> None:
            self.text = "/exit"

    class _FakeEvent:
        def __init__(self, area: _FakeArea) -> None:
            self.value = area.text
            self.text_area = area

    app = XTuiApp("127.0.0.1", 9999)
    workers: list[tuple[object, dict[str, object]]] = []
    appended: list[Widget] = []
    app.run_worker = lambda coroutine, **kwargs: workers.append((coroutine, kwargs))  # type: ignore[method-assign]
    app._append = lambda widget: appended.append(widget)  # type: ignore[method-assign]

    try:
        await app.on_chat_text_area_submitted(_FakeEvent(_FakeArea()))  # type: ignore[arg-type]
        assert len(workers) == 1
        assert workers[0][1] == {"name": "quit", "group": "shutdown", "exclusive": True}
        assert appended == []
    finally:
        for coroutine, _ in workers:
            coroutine.close()  # type: ignore[union-attr]


# 功能：验证输入 /clear 时调度会话上下文清理，而不发送给 Agent
# 设计：复用简化输入事件并捕获 worker，锁定内置命令路由，避免依赖 socket 或 Textual 生命周期
async def test_clear_command_starts_context_clear_without_sending_message() -> None:
    class _FakeArea:
        def __init__(self) -> None:
            self.text = "/clear"
            self.disabled = False
            self.border_title = ""

    class _FakeEvent:
        def __init__(self, area: _FakeArea) -> None:
            self.value = area.text
            self.text_area = area

    class _FakeClient:
        async def send_command(self, method: str, params: dict) -> dict:
            return {"run_id": "run-1"}

    app = XTuiApp("127.0.0.1", 9999)
    workers: list[tuple[object, dict[str, object]]] = []
    app._client = _FakeClient()  # type: ignore[assignment]
    app._session_id = "sess-1"
    app._append = lambda widget: None  # type: ignore[method-assign]
    app._update_header = lambda state: None  # type: ignore[method-assign]
    app.run_worker = lambda coroutine, **kwargs: workers.append((coroutine, kwargs))  # type: ignore[method-assign]

    try:
        await app.on_chat_text_area_submitted(_FakeEvent(_FakeArea()))  # type: ignore[arg-type]
        assert len(workers) == 1
        assert workers[0][1] == {"name": "clear", "exclusive": False}
    finally:
        for coroutine, _ in workers:
            coroutine.close()  # type: ignore[union-attr]


# 功能：验证关闭请求因断线被取消时，/exit 仍会退出 TUI
# 设计：模拟 SocketClient 在 Windows 断连后取消 pending Future，断言 action_quit 不传播取消且调用 exit
async def test_action_quit_exits_when_session_close_is_cancelled() -> None:
    class _DisconnectedClient:
        async def send_command(self, method: str, params: dict) -> dict:
            raise asyncio.CancelledError

    app = XTuiApp("127.0.0.1", 9999)
    exit_calls: list[None] = []
    app._client = _DisconnectedClient()  # type: ignore[assignment]
    app._session_id = "sess-1"
    app._append = lambda widget: None  # type: ignore[method-assign]
    app.exit = lambda: exit_calls.append(None)  # type: ignore[method-assign]

    await app.action_quit()

    assert exit_calls == [None]


# 功能：验证提交用户输入时会追加 user turn，并进入 busy 状态
# 设计：用 fake client 替代 SocketClient，直接调用 on_chat_text_area_submitted，
#       覆盖 TextArea 清空内容 + 设置 busy 占位符的核心状态迁移
async def test_input_submit_appends_user_turn_and_disables_prompt() -> None:
    class _FakeArea:
        def __init__(self) -> None:
            self.disabled = False
            self.border_title = ""
            self.text = "hello"

    class _FakeEvent:
        def __init__(self, area: _FakeArea) -> None:
            self.value = area.text
            self.text_area = area

    class _FakeClient:
        async def send_command(self, method: str, params: dict) -> dict:
            return {"run_id": "run-1"}

    app = XTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]
    app._update_header = lambda state: None  # type: ignore[method-assign]
    app._client = _FakeClient()  # type: ignore[assignment]
    app._session_id = "sess-1"

    area = _FakeArea()
    event = _FakeEvent(area)
    await app.on_chat_text_area_submitted(event)  # type: ignore[arg-type]

    assert app._busy  # type: ignore[attr-defined]
    assert area.disabled
    assert area.text == ""
    assert "agent is working" in area.border_title.lower()
    assert appended[0].content == "[bold]>[/bold] hello"


# 功能：验证未知事件类型不抛异常也不追加任何 widget
# 设计：发送 type 为 unknown 的事件，断言 appended 为空
def test_unknown_event_silently_ignored() -> None:
    app = XTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({"type": "some.unknown.type", "run_id": "r", "ts": "t"})
    assert appended == []
