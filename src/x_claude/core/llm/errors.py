from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic
import httpx

from x_claude.core.resources import ResourceLimitExceeded


@dataclass(frozen=True)
class LlmFailure:
    code: str
    message: str
    hint: str


# 将模型异常转换为固定的用户提示，不向事件、TUI 或历史泄露服务端原始错误和密钥
def describe_llm_error(exc: Exception) -> LlmFailure:
    if isinstance(exc, ResourceLimitExceeded):
        return LlmFailure(exc.code, str(exc), "缩小任务范围，或调整 [agent] 的资源预算后再运行。")
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return LlmFailure("llm_auth", "模型服务认证或权限校验失败",
                          "检查 ANTHROPIC_API_KEY 与 ANTHROPIC_BASE_URL 是否属于同一服务；"
                          "修改后重启 x-core。")
    if status == 429:
        return LlmFailure("llm_rate_limit", "模型服务限流或额度不足",
                          "稍后重试，并检查服务额度；频繁出现时降低并发请求数。")
    if isinstance(exc, (TimeoutError, httpx.TimeoutException, anthropic.APITimeoutError)):
        return LlmFailure("llm_timeout", "等待模型响应超时",
                          "检查网络与代理连接，稍后重试或缩小任务范围。")
    if isinstance(exc, (httpx.TransportError, anthropic.APIConnectionError)):
        return LlmFailure("llm_connection", "无法连接模型服务或连接中断",
                          "检查服务地址、网络和代理配置后重试。")
    if status in (400, 413, 422):
        body = json.dumps(getattr(exc, "body", ""), default=str).lower()
        if status == 413 or any(term in body for term in (
            "context_length", "context window", "context length", "too many tokens",
            "prompt is too long", "maximum context", "max context",
        )):
            return LlmFailure("llm_context_limit", "请求超出模型上下文容量",
                              "先执行 /compact 或 /clear；"
                              "核对 X_CONTEXT_WINDOW 是否符合模型实际窗口。")
        return LlmFailure("llm_invalid_request", "模型服务拒绝了请求参数",
                          "检查模型名称及服务的 Anthropic 协议兼容性。")
    if isinstance(status, int) and status >= 500:
        return LlmFailure("llm_unavailable", "模型服务暂时不可用", "稍后重试。")
    return LlmFailure("llm_error", "模型调用失败", "查看本机 daemon 日志定位原因。")
