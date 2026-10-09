from __future__ import annotations

import json
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from x_claude.core.transport.socket_client import SocketClient


# 展示消息正文，工具数据以纯文本显示并限制单页控件负载
def message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        content = "\n".join(str(block.get("text", block.get("content", "")))
                            if block.get("type") in ("text", "tool_result")
                            else json.dumps(block, ensure_ascii=False)
                            for block in content if isinstance(block, dict))
    text = str(content)
    suffix = "\n[本条过长，显示已截断；完整记录保留在本机会话文件]" if len(text) > 16000 else ""
    return text[:16000] + suffix


class HistoryScreen(ModalScreen[None]):
    BINDINGS = [("escape", "dismiss", "关闭")]
    DEFAULT_CSS = """
    HistoryScreen { align: center middle; }
    #history-panel { width: 90%; height: 90%; border: round cyan; background: $surface; }
    #history-content { height: 1fr; padding: 1 2; }
    #history-buttons { height: 3; }
    #history-title { height: 2; padding: 0 2; }
    """

    # 初始化只读历史浏览器，单页保留二十条消息
    def __init__(self, client: SocketClient, session_id: str) -> None:
        super().__init__()
        self._client = client
        self._session_id = session_id
        self._cursors: list[str | None] = [None]
        self._page = 0
        self._next: str | None = None

    # 构建独立历史窗口，不向当前任务日志追加旧消息
    def compose(self) -> ComposeResult:
        with Vertical(id="history-panel"):
            yield Static("对话历史（只读）", id="history-title")
            with VerticalScroll(id="history-content"):
                yield Static("加载中…", id="history-text", markup=False)
            with Horizontal(id="history-buttons"):
                yield Button("更早", id="history-older")
                yield Button("更新", id="history-newer")
                yield Button("关闭", id="history-close")

    # 挂载后异步读取最新页，窗口关闭时 Textual 自动取消所属 worker
    def on_mount(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    # 切换游标并读取对应页，翻页只影响历史窗口
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "history-close":
            self.dismiss()
            return
        if event.button.id == "history-older" and self._next:
            self._cursors = self._cursors[:self._page + 1] + [self._next]
            self._page += 1
        elif event.button.id == "history-newer" and self._page:
            self._page -= 1
        self.run_worker(self._load(), exclusive=True)

    # 读取一页并渲染纯文本，明确提示历史版本失效或连接错误
    async def _load(self) -> None:
        older = self.query_one("#history-older", Button)
        newer = self.query_one("#history-newer", Button)
        older.disabled = newer.disabled = True
        try:
            result = await self._client.send_command("session.history_page", {
                "session_id": self._session_id, "cursor": self._cursors[self._page], "limit": 20,
            })
            self._next = result.get("next_cursor")
            messages = result.get("messages", [])
            text = "\n\n".join(f"{m['role']}\n{message_text(m)}" for m in messages)
            self.query_one("#history-text", Static).update(text or "暂无保存的对话。")
            self.query_one("#history-title", Static).update(
                f"对话历史（只读）· 倒序第 {self._page + 1} 页",
            )
            self.query_one("#history-content", VerticalScroll).scroll_home(animate=False)
            older.disabled = self._next is None
            newer.disabled = self._page == 0
        except Exception as exc:
            self.query_one("#history-text", Static).update(f"无法读取历史：{exc}")
