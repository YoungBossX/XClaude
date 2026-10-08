from __future__ import annotations

import sys
from pathlib import Path
from typing import BinaryIO


class StoreInUseError(RuntimeError):
    pass


class SessionStoreLock:
    # 规范化实际存储目录，锁文件长期保留以避免删除后产生两个锁对象
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._stream: BinaryIO | None = None

    # 非阻塞取得内核文件锁，Windows 锁首字节，POSIX 使用 flock
    def __enter__(self) -> SessionStoreLock:
        self.root.mkdir(parents=True, exist_ok=True)
        stream = (self.root / ".daemon.lock").open("a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                if stream.tell() == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise StoreInUseError(
                f"session storage already in use or cannot be locked: {self.root}; "
                "stop the other x-core before starting another daemon"
            ) from exc
        self._stream = stream
        return self

    # 关闭句柄释放内核锁，异常退出和进程终止也由操作系统释放
    def __exit__(self, *args: object) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
