from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from jsonschema import validate
from pydantic import BaseModel


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    # "runtime_error" | "timeout" | "schema_error" | "permission_denied" | "file_conflict"
    error_type: str | None = None


class BaseTool(ABC):
    name: str
    description: str
    input_schema: dict[str, object]
    params_model: ClassVar[type[BaseModel] | None] = None
    # 只有显式声明幂等的工具才允许自动重试；Shell、写文件和派生 Agent 默认禁止
    retry_safe: ClassVar[bool] = False

    # 内置工具使用 Pydantic，动态工具使用声明的 JSON Schema；调用前统一校验
    def validate_params(self, params: dict[str, object]) -> None:
        if self.params_model is not None:
            self.params_model.model_validate(params)
        else:
            validate(instance=params, schema=self.input_schema)

    # 执行工具调用，返回结果或错误
    @abstractmethod
    async def invoke(self, params: dict[str, object]) -> ToolResult: ...
