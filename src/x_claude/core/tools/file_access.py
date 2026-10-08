from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import RLock

# 文件操作临界区内不等待协程；同时保护同进程线程，版本记录按执行上下文隔离
FILE_LOCK = RLock()
_versions: ContextVar[dict[str, str | None] | None] = ContextVar("file_versions", default=None)


# 为单次工具调用绑定所属任务的文件版本，退出或取消后还原父任务上下文
@contextmanager
def file_access_scope(versions: dict[str, str | None] | None) -> Iterator[None]:
    token = _versions.set(versions)
    try:
        yield
    finally:
        _versions.reset(token)


# 使用文件身份、修改时间和原始字节构造版本，检测替换以及内容修改
def file_version(raw: bytes, metadata: os.stat_result) -> str:
    # Windows 文件替换后路径与句柄的创建时间可能不同，不用它判定版本
    changed_at = metadata.st_ctime_ns if os.name != "nt" else 0
    identity = (metadata.st_dev, metadata.st_ino, metadata.st_size,
                metadata.st_mtime_ns, changed_at, metadata.st_mode)
    return hashlib.sha256(repr(identity).encode() + b"\0" + raw).hexdigest()


# 保存实际读到或成功写入的完整文件版本；截断内容不得授权覆盖
def remember_version(path: Path, version: str | None) -> None:
    versions = _versions.get()
    if versions is not None:
        versions[os.path.normcase(str(path))] = version


# 获取当前任务之前观察到的版本，未读取文件返回独立的存在标志
def observed_version(path: Path) -> tuple[bool, str | None]:
    versions = _versions.get()
    key = os.path.normcase(str(path))
    return (versions is not None and key in versions,
            versions.get(key) if versions is not None else None)
