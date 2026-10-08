from __future__ import annotations

import os
import tempfile
from pathlib import Path


# 在目标目录完整写入并同步临时文件，然后原子替换目标，失败时保留原文件并清理临时文件
def atomic_write_bytes(path: Path, content: bytes, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if stream.write(content) != len(content):
                raise OSError("incomplete temporary file write")
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
