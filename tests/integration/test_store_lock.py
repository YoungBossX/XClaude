from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import x_claude.core.app as app_module
from x_claude.core.config import XConfig
from x_claude.core.session.lock import SessionStoreLock, StoreInUseError
from x_claude.core.session.store import SessionStore
from x_claude.core.transport.socket_server import SocketServer


# 功能：验证真正的跨进程互斥，同存储目录拒绝第二进程而其他目录可用
# 设计：子进程独立打开内核锁，不能用 Python 进程内变量伪造互斥通过
def test_store_lock_excludes_another_process(tmp_path: Path) -> None:
    code = """
import sys
from pathlib import Path
from x_claude.core.session.lock import SessionStoreLock, StoreInUseError
try:
    with SessionStoreLock(Path(sys.argv[1])):
        print('acquired')
except StoreInUseError:
    print('blocked')
"""
    root = tmp_path / "sessions"
    with SessionStoreLock(root):
        for path, expected in [(root, "blocked"), (tmp_path / "other", "acquired")]:
            result = subprocess.run([sys.executable, "-c", code, str(path)],
                                    capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == expected
    with SessionStoreLock(root):
        assert (root / ".daemon.lock").exists()


# 功能：验证崩溃或正常关闭均释放锁，不因遗留锁文件拒绝启动
# 设计：独立进程取得锁后用 os._exit 模拟不执行清理的崩溃，父进程再次取得同一锁
@pytest.mark.parametrize("crash", [False, True])
def test_process_exit_releases_store_lock(tmp_path: Path, crash: bool) -> None:
    code = """
import os, sys
from pathlib import Path
from x_claude.core.session.lock import SessionStoreLock
with SessionStoreLock(Path(sys.argv[1])):
    if sys.argv[2] == 'crash':
        os._exit(7)
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path),
                             "crash" if crash else "normal"], timeout=10)
    assert result.returncode == (7 if crash else 0)
    assert (tmp_path / ".daemon.lock").exists()
    with SessionStoreLock(tmp_path):
        assert (tmp_path / ".daemon.lock").stat().st_size >= (1 if sys.platform == "win32" else 0)


# 功能：验证不同端口的真实 CoreApp 也不能启动共享存储的第二个 daemon
# 设计：第一实例实际监听 TCP，第二实例直接调用同一生产启动入口，失败后第一实例仍然运行
async def test_core_app_rejects_shared_store_before_resources(tmp_path: Path) -> None:
    config = XConfig()
    config.port = 0
    config.trace.enabled = False
    store = SessionStore(tmp_path / "sessions")
    ready = asyncio.Event()
    shutdown = asyncio.Event()
    servers: list[SocketServer] = []

    # 保留服务器全部行为，仅记录实例以核对初始化次数
    def server_factory(*args, **kwargs):
        server = SocketServer(*args, **kwargs)
        servers.append(server)
        return server

    # 监听成功后用事件控制正常停机，避免实际注册全局信号
    def install(loop, event):
        nonlocal shutdown
        shutdown = event
        ready.set()

    with patch.object(app_module, "get_config", return_value=config), \
            patch.object(app_module, "SessionStore", return_value=store), \
            patch.object(app_module, "setup_logging") as logging_setup, \
            patch.object(app_module, "AnthropicProvider", return_value=object()) as provider, \
            patch.object(app_module, "SocketServer", side_effect=server_factory), \
            patch.object(app_module, "_install_shutdown_handlers", side_effect=install):
        first = asyncio.create_task(app_module.CoreApp().run())
        try:
            await asyncio.wait_for(ready.wait(), 5)
            first_port = servers[0]._server.sockets[0].getsockname()[1]
            assert first_port != 0
            # port=0 会选择另一个空闲端口，不能把拒绝启动误认为端口占用
            with pytest.raises(SystemExit, match="session storage already in use"):
                await app_module.CoreApp().run()
            assert len(servers) == 1 and provider.call_count == logging_setup.call_count == 1
            assert not first.done() and servers[0]._server.is_serving()
        finally:
            shutdown.set()
            await asyncio.wait_for(first, 5)
    with SessionStoreLock(store.root):
        pass


# 功能：验证监听失败与任务取消均完整清理资源并释放存储锁
# 设计：覆盖初始化失败和正常运行中取消，不仅测试独立锁对象的退出
@pytest.mark.parametrize("failure", ["start", "cancel"])
async def test_core_failure_or_cancel_releases_store_lock(tmp_path: Path, failure: str) -> None:
    config = XConfig()
    config.trace.enabled = False
    config.port = 0
    store = SessionStore(tmp_path / "sessions")
    server = SocketServer(config.host, 0)
    ready = asyncio.Event()
    mcp = AsyncMock()
    with patch.object(app_module, "get_config", return_value=config), \
            patch.object(app_module, "SessionStore", return_value=store), \
            patch.object(app_module, "setup_logging"), \
            patch.object(app_module, "AnthropicProvider", return_value=object()), \
            patch.object(app_module, "McpServerManager", return_value=mcp), \
            patch.object(app_module, "SocketServer", return_value=server), \
            patch.object(app_module, "_install_shutdown_handlers", side_effect=lambda *args: ready.set()):
        if failure == "start":
            with patch.object(server, "start", AsyncMock(side_effect=OSError("failed bind"))):
                with pytest.raises(OSError, match="failed bind"):
                    await app_module.CoreApp().run()
        else:
            task = asyncio.create_task(app_module.CoreApp().run())
            try:
                await asyncio.wait_for(ready.wait(), 5)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            assert task.cancelled() and not server._server.is_serving()
            assert not server._server.sockets
        mcp.stop_all.assert_awaited_once()
    with SessionStoreLock(store.root):
        pass


# 功能：验证同一目录的规范化路径不能绕过锁
# 设计：用包含父级折返的别名获取第二锁，验证不是简单字符串路径判重
def test_normalized_store_path_is_same_lock(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "child").mkdir()
    with SessionStoreLock(root):
        with pytest.raises(StoreInUseError):
            with SessionStoreLock(root / "child" / ".."):
                pytest.fail("lock bypassed")
