from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from x_claude.core.config import XConfig
from x_claude.core.transport.socket_client import IpcError, SocketClient

_DECISION_MAP: dict[str, str] = {
    "y": "allow_once",
    "a": "always_allow",
    "n": "deny_once",
    "d": "always_deny",
}


class ChatPrinter:
    # 初始化 chat 模式的流式输出状态和待审批权限请求
    def __init__(self) -> None:
        self._inline = False
        self.pending_permission_id: tuple[str, str] | None = None
        self._permissions: dict[tuple[str, str], dict[str, Any]] = {}
        self.permission_available = asyncio.Event()
        self.permission_changed = asyncio.Event()

    # 移除已处理的审批并唤醒下一个请求，后台子 Agent 可同时申请多个权限
    def dismiss_permission(self, tool_use_id: tuple[str, str]) -> None:
        self._permissions.pop(tool_use_id, None)
        self.permission_changed.set()
        self.pending_permission_id = next(iter(self._permissions), None)
        if self.pending_permission_id is None:
            self.permission_available.clear()
        else:
            self.permission_available.set()

    # 若当前 LLM token 尚未换行，则补一个换行
    def _ensure_newline(self) -> None:
        if self._inline:
            print()
            self._inline = False

    # 按事件类型打印 chat 输出、等待提示和权限审批请求
    async def handle(self, event: dict[str, Any]) -> None:
        t = event.get("type", "")
        if t == "llm.token":
            print(event.get("token", ""), end="", flush=True)
            self._inline = True
        elif t == "tool.call_started":
            self._ensure_newline()
            print(f"[tool] {event.get('tool_name', '')}")
        elif t == "permission.requested":
            self._ensure_newline()
            tool_name = str(event.get("tool_name", ""))
            param_preview = str(event.get("param_preview", ""))
            tool_use_id = str(event.get("tool_use_id", ""))
            print(f"[permission] {tool_name}  {param_preview}")
            print("  y=allow once  a=always allow  n=deny once  d=always deny")
            self._permissions[(str(event.get("run_id", "")), tool_use_id)] = event
            self.pending_permission_id = next(iter(self._permissions), None)
            self.permission_available.set()
        elif t in ("permission.granted", "permission.denied"):
            self.dismiss_permission((str(event.get("run_id", "")),
                                     str(event.get("tool_use_id", ""))))
        elif t == "session.waiting_for_input":
            self._ensure_newline()
            print("[waiting for input]")
        elif t == "session.closed":
            self._ensure_newline()
            self._permissions.clear()
            self.pending_permission_id = None
            self.permission_available.clear()
            print("session closed.")


# 使用可取消的终端输入，避免审批过期后线程池 input 仍阻止进程退出
async def _readline(prompt: str) -> str:
    print(prompt, end="", flush=True)
    if os.name == "nt":
        if not sys.stdin.isatty():
            raise OSError("interactive chat requires a console; use x run for redirected stdin")
        import msvcrt

        chars: list[str] = []
        try:
            while True:
                if not msvcrt.kbhit():
                    await asyncio.sleep(0.02)
                    continue
                char = msvcrt.getwch()
                if char in ("\x00", "\xe0"):
                    msvcrt.getwch()
                elif char in ("\r", "\n"):
                    return "".join(chars)
                elif char == "\x03":
                    raise EOFError()
                elif char in ("\x04", "\x1a"):
                    raise EOFError()
                elif char == "\b":
                    if chars:
                        chars.pop()
                        print("\b \b", end="", flush=True)
                else:
                    chars.append(char)
                    print(char, end="", flush=True)
        finally:
            print()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()
    fd = sys.stdin.fileno()
    buffer = bytearray()

    # 终端或管道可读时取一行；取消后移除监听，不遗留后台读线程
    def ready() -> None:
        if future.done():
            return
        try:
            chunk = os.read(fd, 1)
            if chunk == b"\n" or (not chunk and buffer):
                future.set_result(buffer.decode(sys.stdin.encoding or "utf-8", errors="replace"))
            elif not chunk:
                future.set_exception(EOFError())
            else:
                buffer.extend(chunk)
        except Exception as exc:
            future.set_exception(exc)
        finally:
            if future.done():
                loop.remove_reader(fd)

    loop.add_reader(fd, ready)
    try:
        return await future
    finally:
        loop.remove_reader(fd)


# 普通输入也监听连接关闭，断连后取消输入并返回，而不是要求再按回车
async def _read_connected(prompt: str, loop_task: asyncio.Task[None]) -> str:
    task = asyncio.create_task(_readline(prompt))
    try:
        done, _ = await asyncio.wait({task, loop_task}, return_when=asyncio.FIRST_COMPLETED)
        if loop_task in done:
            raise OSError("core connection closed while awaiting input")
        return task.result()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# 等待单轮运行期间继续处理审批，不让同步 RPC 等待阻塞终端输入
async def _send_with_approval(
    client: SocketClient, printer: ChatPrinter, session_id: str, content: str,
    loop_task: asyncio.Task[None],
) -> None:
    send_task = asyncio.create_task(client.send_command(
        "session.send_message", {"session_id": session_id, "content": content},
    ))
    permission_wait: asyncio.Task[bool] | None = None
    try:
        while not send_task.done():
            permission_wait = asyncio.create_task(printer.permission_available.wait())
            done, _ = await asyncio.wait(
                {send_task, permission_wait, loop_task}, return_when=asyncio.FIRST_COMPLETED,
            )
            if loop_task in done:
                raise OSError("core connection closed while running")
            if send_task in done:
                break
            tool_use_id = printer.pending_permission_id
            if tool_use_id is not None:
                printer.permission_changed.clear()
                input_task = asyncio.create_task(_readline("permission> "))
                changed = asyncio.create_task(printer.permission_changed.wait())
                try:
                    done, _ = await asyncio.wait(
                        {input_task, changed, send_task, loop_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if loop_task in done:
                        raise OSError("core connection closed while awaiting approval")
                    if send_task in done:
                        break
                    if tool_use_id not in printer._permissions:
                        continue
                    if input_task not in done:
                        continue
                    answer = input_task.result().strip().lower()
                finally:
                    input_task.cancel()
                    changed.cancel()
                    await asyncio.gather(input_task, changed, return_exceptions=True)
                decision = _DECISION_MAP.get(answer)
                if decision is None:
                    print("  enter y (allow once), a (always allow), "
                          "n (deny once), d (always deny)")
                    continue
                if tool_use_id in printer._permissions:
                    await client.send_command("permission.respond", {
                        "tool_use_id": tool_use_id[1], "decision": decision,
                        "session_id": session_id,
                        "run_id": printer._permissions[tool_use_id].get("run_id", ""),
                    })
                printer.dismiss_permission(tool_use_id)
            permission_wait.cancel()
            await asyncio.gather(permission_wait, return_exceptions=True)
        await send_task
    finally:
        if permission_wait is not None:
            permission_wait.cancel()
            await asyncio.gather(permission_wait, return_exceptions=True)
        if not send_task.done():
            send_task.cancel()
        await asyncio.gather(send_task, return_exceptions=True)


# 异步核心：创建 chat session，循环读取用户输入并发送到 daemon；权限请求时优先处理审批
async def _chat_async(config: XConfig) -> int:
    client = SocketClient(config.host, config.port)
    try:
        await client.connect()
    except (ConnectionRefusedError, OSError):
        print(f"error: core not running ({config.host}:{config.port})", file=sys.stderr)
        return 1

    printer = ChatPrinter()
    client.on_event(printer.handle)
    loop_task = asyncio.create_task(client.run_event_loop())

    try:
        created = await client.send_command("session.create", {"mode": "chat"})
        session_id = str(created["session_id"])
        await client.send_command(
            "event.subscribe",
            {
                "topics": ["session.*", "run.*", "tool.*", "llm.token", "permission.*"],
                "scope": f"session:{session_id}",
            },
        )
        print(f"[session: {session_id}]")

        while True:
            try:
                line = await _read_connected("> ", loop_task)
            except (EOFError, KeyboardInterrupt):
                break
            content = line.strip()
            if not content:
                continue

            # 有待审批的权限请求时，将用户输入解释为决策而非聊天消息
            if printer.pending_permission_id:
                decision = _DECISION_MAP.get(content.lower())
                if decision is None:
                    print("  enter y (allow once), a (always allow), "
                          "n (deny once), d (always deny)")
                    continue
                tool_use_id = printer.pending_permission_id
                await client.send_command(
                    "permission.respond",
                    {"tool_use_id": tool_use_id[1], "decision": decision,
                     "session_id": session_id,
                     "run_id": printer._permissions[tool_use_id].get("run_id", "")},
                )
                printer.dismiss_permission(tool_use_id)
                continue

            try:
                await _send_with_approval(client, printer, session_id, content, loop_task)
            except EOFError:
                break

        await client.send_command("session.close", {"session_id": session_id})
    except (IpcError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass
        await client.close()
    return 0


# 执行 x chat 命令
def cmd_chat(config: XConfig) -> None:
    try:
        exit_code = asyncio.run(_chat_async(config))
    except KeyboardInterrupt:
        sys.exit(130)
    sys.exit(exit_code)
