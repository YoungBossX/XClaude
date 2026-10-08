from __future__ import annotations

import json
from typing import Any

TOOL_RESULT_LIMIT = 8_000
TOOL_RESULT_KEEP = 4_000


# 对消息列表中超长的 tool_result 内容做内存截断，返回处理后的新列表
def truncate_tool_results(
    messages: list[dict[str, Any]],
    limit: int = TOOL_RESULT_LIMIT,
    keep: int = TOOL_RESULT_KEEP,
) -> list[dict[str, Any]]:
    result = []
    for msg in messages:
        if msg.get("role") != "user":
            result.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue
        new_blocks = []
        for block in content:
            if block.get("type") == "tool_result" and isinstance(block.get("content"), str):
                text = block["content"]
                if len(text) > limit:
                    prefix = min(keep, limit)
                    omitted = len(text) - prefix
                    block = dict(block)
                    block["content"] = (
                        text[:prefix]
                        + f"\n[... {omitted} chars omitted. Full output in run events.]"
                    )
            new_blocks.append(block)
        result.append({**msg, "content": new_blocks})
    return result


# 用序列化后的 UTF-8 字节数保守估算新增内容，避免中文和工具参数被字符除四低估
def estimate_tokens(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
