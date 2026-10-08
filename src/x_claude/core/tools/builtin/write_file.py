from __future__ import annotations

import os
import stat

from pydantic import BaseModel, ConfigDict

from x_claude.core.atomic_file import atomic_write_bytes
from x_claude.core.tools.base import BaseTool, ToolResult
from x_claude.core.tools.file_access import (
    FILE_LOCK,
    file_version,
    observed_version,
    remember_version,
)
from x_claude.core.tools.workspace import workspace_path

_MAX_BYTES = 1 * 1024 * 1024  # 1 MB


class WriteFileParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str
    content: str


class WriteFileTool(BaseTool):
    params_model = WriteFileParams
    name = "write_file"
    description = (
        "Write text content to a file, creating it (and any parent directories) if it "
        "does not exist, or overwriting it if it does. "
        "Path must stay inside the current working directory, including resolved links. "
        "Content size is limited to 1 MB."
        " Before overwriting an existing file, read_file its complete content in this run. "
        "If file_conflict is returned, read_file again and merge your changes; do not "
        "blindly retry stale content. New files need no prior read."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file (relative to current working directory).",
            },
            "content": {
                "type": "string",
                "description": "Text content to write.",
            },
        },
        "required": ["path", "content"],
    }

    # 仅写入项目内实际路径，拒绝超 1MB 内容并自动创建父目录
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = WriteFileParams.model_validate(params)
        path_str = p.path
        content = p.content

        path = workspace_path(path_str)

        encoded = content.encode("utf-8")
        if len(encoded) > _MAX_BYTES:
            return ToolResult(
                content=f"content too large: {len(encoded)} bytes (limit 1 MB)",
                is_error=True,
                error_type="runtime_error",
            )

        with FILE_LOCK:
            observed, expected = observed_version(path)
            try:
                with path.open("rb") as stream:
                    metadata = os.fstat(stream.fileno())
                    raw = stream.read(_MAX_BYTES + 1)
                    after = os.fstat(stream.fileno())
                current = file_version(raw, metadata)
                if (current != file_version(raw, after)
                        or current != file_version(raw, path.stat())):
                    return ToolResult(
                        content="file_conflict: file changed during validation; read_file again.",
                        is_error=True, error_type="file_conflict",
                    )
                mode = stat.S_IMODE(metadata.st_mode)
            except FileNotFoundError:
                current, mode = None, None
            if ((current is not None and not observed)
                    or (observed and current != expected)):
                return ToolResult(
                    content="file_conflict: file was not fully read in this run or changed "
                    "since your last read/write. Read the current file with read_file, "
                    "merge your changes, then write_file again. No content was changed.",
                    is_error=True, error_type="file_conflict",
                )
            # 临界区内检查与替换不让出执行权，写入失败不提前更新任务版本
            if mode is not None and not mode & stat.S_IWUSR:
                raise PermissionError(f"file is read-only: {path}")
            atomic_write_bytes(path, encoded, mode=mode)
            remember_version(path, file_version(encoded, path.stat()))

        return ToolResult(content=f"wrote {len(encoded)} bytes to {path_str}")
