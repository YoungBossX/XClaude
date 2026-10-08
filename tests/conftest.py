from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from x_claude.core.transport.auth import read_credential


@pytest.fixture
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return port  # socket released; daemon can bind to this port


@pytest.fixture
async def running_daemon(
    free_port: int, tmp_path: Path,
) -> AsyncGenerator[subprocess.Popen[bytes], None]:
    env = os.environ.copy()
    env["X_PORT"] = str(free_port)
    env["X_LOG_FILE"] = ""
    env["X_LOG_LEVEL"] = "WARNING"

    # 测试守护进程使用独立存储，不与用户正在运行的 daemon 争用存储锁
    bootstrap = (
        "import sys; from pathlib import Path; from unittest.mock import patch; "
        "import x_claude.core.app as app; "
        "from x_claude.core.session.store import SessionStore; "
        "factory = lambda _: SessionStore(Path(sys.argv[1])); "
        "scope = patch.object(app, 'SessionStore', side_effect=factory); "
        "scope.start(); app.run()"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", bootstrap, str(tmp_path / "sessions")], env=env,
    )

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"Daemon exited during startup with code {proc.returncode}")
        await asyncio.sleep(0.05)
        try:
            _reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
            writer.close()
            await writer.wait_closed()
            if read_credential("127.0.0.1", free_port) is not None:
                break
        except (ConnectionRefusedError, OSError):
            pass
    else:
        proc.terminate()
        proc.wait()
        pytest.fail("Daemon did not start within 10 seconds")

    try:
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
