from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict

from x_claude.core.tools.base import BaseTool, ToolResult
from x_claude.core.tools.file_access import FILE_LOCK, file_version, remember_version
from x_claude.core.tools.workspace import workspace_path

_MAX_BYTES = 512 * 1024  # 512 KB


class ReadFileParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str


class ReadFileTool(BaseTool):
    params_model = ReadFileParams
    name = "read_file"
    description = (
        "Read the text content of a file. "
        "Path must stay inside the current working directory, including resolved links. "
        "Files larger than 512 KB are truncated."
        " Read an existing file in this run before write_file; truncated reads do not "
        "authorize overwriting the full file."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file (relative to current working directory).",
            }
        },
        "required": ["path"],
    }

    # 仅读取项目内文件，限量载入并截断；实际链接目标也必须在项目内
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        path_str = ReadFileParams.model_validate(params).path

        path = workspace_path(path_str)
        with FILE_LOCK:
            try:
                with path.open("rb") as stream:
                    before = os.fstat(stream.fileno())
                    raw = stream.read(_MAX_BYTES + 1)
                    after = os.fstat(stream.fileno())
            except FileNotFoundError:
                remember_version(path, None)
                raise
            if file_version(raw, before) != file_version(raw, after):
                remember_version(path, "unreadable")
                return ToolResult(
                    content="file_conflict: file changed while reading; read_file again.",
                    is_error=True, error_type="file_conflict",
                )
            remember_version(path, file_version(raw, after)
                             if len(raw) <= _MAX_BYTES else "truncated")
        truncated = len(raw) > _MAX_BYTES
        text = raw[:_MAX_BYTES].decode("utf-8", errors="replace")
        if truncated:
            text += "\n[truncated]"

        return ToolResult(content=text)
