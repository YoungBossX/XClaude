from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from x_claude.core.bus.envelope import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    UNAUTHORIZED,
    HandlerError,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcSuccess,
    make_error,
)
from x_claude.core.trace.record import TraceRecord
from x_claude.core.trace.writer import TraceWriter
from x_claude.core.transport.auth import local_host, publish_credential
from x_claude.core.transport.ipc_broadcaster import IpcEventBroadcaster

logger = logging.getLogger(__name__)

type CommandHandler = Callable[[dict[str, Any]], Awaitable[Any]]

# 每个连接处理协程中，当前正在处理的 writer（供 handler 读取连接上下文）
_writer_var: ContextVar[asyncio.StreamWriter] = ContextVar("_writer_var")


def _now() -> str:
    return datetime.now(UTC).isoformat()


# 返回当前 handler 调用所属连接的 StreamWriter
def get_connection_writer() -> asyncio.StreamWriter:
    return _writer_var.get()

_MAX_LINE_BYTES = 64 * 1024 * 1024  # 64 MB per frame，兼容 MCP 大文件工具结果


class SocketServer:
    def __init__(
        self,
        host: str,
        port: int,
        broadcaster: IpcEventBroadcaster | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._handlers: dict[str, CommandHandler] = {}
        self._server: asyncio.AbstractServer | None = None
        self._broadcaster = broadcaster
        self._trace = trace
        self._active_writers: set[asyncio.StreamWriter] = set()
        self._requests: set[asyncio.Task[None]] = set()
        self._auth_token: str | None = None
        self._credential_path: Path | None = None
        self._stopping = False

    # 注册一个方法名对应的命令处理函数
    def register(self, method: str, handler: CommandHandler) -> None:
        self._handlers[method] = handler

    # 启动 TCP 服务器；若端口已被占用则退出进程
    async def start(self) -> str:
        self._host = local_host(self._host)
        try:
            _r, w = await asyncio.open_connection(self._host, self._port)
            w.close()
            await w.wait_closed()
            raise SystemExit(f"core already running at {self._host}:{self._port}")
        except (ConnectionRefusedError, OSError):
            pass

        self._server = await asyncio.start_server(
            self._handle_connection,
            host=self._host,
            port=self._port,
            limit=_MAX_LINE_BYTES,
        )
        port = self._server.sockets[0].getsockname()[1]
        self._auth_token = secrets.token_urlsafe(32)
        try:
            self._credential_path = publish_credential(self._host, port, self._auth_token)
        except BaseException:
            self._server.close()
            await self._server.wait_closed()
            raise
        return f"{self._host}:{self._port}"

    # 先停止接收新连接与新命令，不取消现有主任务，让 manager 先设置挂起状态
    def stop_accepting(self) -> None:
        self._stopping = True
        if self._server is not None:
            self._server.close()

    # 关闭服务器：先断开所有活跃连接，再等待服务器完全关闭（最多 2 秒）
    async def stop(self) -> None:
        if self._server is None:
            return
        self.stop_accepting()
        for writer in list(self._active_writers):
            try:
                writer.close()
            except Exception:
                pass
        self._server.close()
        requests = [task for task in self._requests if task is not asyncio.current_task()]
        for task in requests:
            task.cancel()
        if requests:
            await asyncio.gather(*requests, return_exceptions=True)
        try:
            await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            pass
        if self._credential_path is not None:
            try:
                if self._credential_path.read_text(encoding="ascii") == self._auth_token:
                    self._credential_path.unlink()
            except FileNotFoundError:
                pass

    # 处理单个客户端连接，完成后关闭写流
    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername", "<unknown>")
        logger.debug("client connected: %s", peer)
        self._active_writers.add(writer)
        try:
            await self._read_loop(reader, writer)
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("client connection reset: %s", peer)
        finally:
            self._active_writers.discard(writer)
            if self._broadcaster is not None:
                self._broadcaster.unsubscribe(writer)
            try:
                writer.close()
            except Exception:
                pass
            logger.debug("client disconnected: %s", peer)

    # 持续读取换行分隔的 JSON 行并逐行分发处理
    async def _read_loop(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        while True:
            try:
                line = await reader.readline()
            except (asyncio.LimitOverrunError, ValueError):
                await self._send(writer, make_error(None, INVALID_REQUEST, "Request too large"))
                return

            if not line:
                return

            # 每条命令独立作为 task 执行，避免长时间运行的 handler（如 session.send_message）
            # 阻塞读循环，使 permission.respond 等并发命令能被及时处理
            task = asyncio.create_task(self._handle_line(line, writer))
            self._requests.add(task)
            task.add_done_callback(self._requests.discard)

    # 解析单行 JSON-RPC 请求并调用对应 handler，将结果或错误写回客户端
    async def _handle_line(self, line: bytes, writer: asyncio.StreamWriter) -> None:
        try:
            raw: Any = json.loads(line)
        except json.JSONDecodeError as e:
            await self._send(writer, make_error(None, PARSE_ERROR, f"Parse error: {e}"))
            return

        try:
            req = JsonRpcRequest.model_validate(raw)
        except ValidationError:
            await self._send(writer, make_error(None, INVALID_REQUEST, "Invalid Request"))
            return

        if (self._auth_token is None or req.auth_token is None
                or not secrets.compare_digest(req.auth_token.encode(), self._auth_token.encode())):
            await self._send(
                writer, make_error(req.id, UNAUTHORIZED, "IPC authentication required"),
            )
            return
        if self._stopping:
            await self._send(writer, make_error(req.id, INTERNAL_ERROR, "daemon is shutting down"))
            return

        if self._trace is not None:
            client_id = str(writer.get_extra_info("peername", "<unknown>"))
            self._trace.emit(
                TraceRecord(
                    ts=_now(),
                    direction="CLIENT→CORE",
                    layer="ipc",
                    kind="command",
                    client_id=client_id,
                    data={"method": req.method, "id": req.id, "params": req.params},
                )
            )

        handler = self._handlers.get(req.method)
        if handler is None:
            await self._send(
                writer,
                make_error(req.id, METHOD_NOT_FOUND, f"Method not found: {req.method}"),
            )
            return

        _writer_var.set(writer)
        try:
            result = await handler(req.params)
        except HandlerError as e:
            await self._send(writer, make_error(req.id, e.code, str(e), e.data))
            return
        except ValidationError as e:
            await self._send(
                writer,
                make_error(req.id, INVALID_REQUEST, "Invalid params", str(e)),
            )
            return
        except Exception as e:
            logger.exception("handler %s raised: %s", req.method, e)
            await self._send(writer, make_error(req.id, INTERNAL_ERROR, "Internal error"))
            return

        result_data: Any = result.model_dump() if isinstance(result, BaseModel) else result
        try:
            await self._send(writer, JsonRpcSuccess(id=req.id, result=result_data))
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("client disconnected before response for %s", req.method)

    # 将 pydantic 消息序列化为 JSON 行并写入流，随后刷新缓冲区
    async def _send(self, writer: asyncio.StreamWriter, msg: BaseModel) -> None:
        writer.write(msg.model_dump_json().encode() + b"\n")
        await writer.drain()
        if self._trace is not None:
            kind = "error" if isinstance(msg, JsonRpcError) else "response"
            client_id = str(writer.get_extra_info("peername", "<unknown>"))
            self._trace.emit(
                TraceRecord(
                    ts=_now(),
                    direction="CORE→CLIENT",
                    layer="ipc",
                    kind=kind,
                    client_id=client_id,
                    data=msg.model_dump(),
                )
            )
