from __future__ import annotations

from pathlib import Path


# 解析实际路径并限制在工作目录内，拒绝父级遍历以及链接、联接指向的外部目录
def workspace_path(value: str) -> Path:
    requested = Path(value)
    if ".." in requested.parts:
        raise PermissionError(f"path traversal not allowed: {value}")
    root = Path.cwd().resolve()
    target = requested.resolve()
    if not target.is_relative_to(root):
        raise PermissionError(f"path outside workspace not allowed: {value}")
    return target
