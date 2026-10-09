from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from x_claude.core.atomic_file import atomic_write_bytes
from x_claude.core.compact.budget import estimate_tokens
from x_claude.core.config import AgentConfig
from x_claude.core.events.bus import EventBus
from x_claude.core.llm.base import LLMProvider
from x_claude.core.llm.types import LlmResponse


class ResourceLimitExceeded(RuntimeError):
    # 携带稳定错误码供运行事件与前端提示消费
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class TaskBudget:
    # 整棵任务树共享并持久化额度，失败请求保留预留消费，daemon 重启不能重置预算
    def __init__(self, root_id: str, config: AgentConfig, path: Path | None = None) -> None:
        self.root_id = root_id
        self.config = config
        self.path = path
        self.started_at = time.time()
        self.used_tokens = 0
        self.tasks = {root_id}
        self.reservations: dict[str, int] = {}
        if path is not None and path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("root_id") != root_id:
                raise ValueError("resource budget identity mismatch")
            self.started_at = float(raw["started_at"])
            self.used_tokens = int(raw["used_tokens"])
            self.tasks = set(raw["tasks"])
            self.reservations = {str(k): int(v) for k, v in raw["reservations"].items()}
            if self.used_tokens < 0 or any(v < 0 for v in self.reservations.values()):
                raise ValueError("invalid resource budget")

    # 提交额度前完成序列化和原子替换，写入失败不能继续调用外部模型
    def _save(self) -> None:
        if self.path is not None:
            payload = {"root_id": self.root_id, "started_at": self.started_at,
                       "used_tokens": self.used_tokens, "tasks": sorted(self.tasks),
                       "reservations": self.reservations}
            atomic_write_bytes(self.path, json.dumps(payload).encode("utf-8"))

    # 返回整棵树剩余墙钟时间，包含排队和 daemon 离线时间；零配置表示禁用
    def remaining_s(self) -> float | None:
        limit = self.config.max_runtime_s
        if not limit:
            return None
        remaining = limit - (time.time() - self.started_at)
        if remaining <= 0:
            raise ResourceLimitExceeded("runtime_budget", "任务树已达到总运行时限")
        return remaining

    # 限制任务树总派生数，已登记任务恢复时不重复占用名额
    def register(self, run_id: str) -> None:
        self.remaining_s()
        if run_id in self.tasks:
            return
        if len(self.tasks) >= self.config.max_tasks:
            raise ResourceLimitExceeded("task_budget", "任务树已达到任务数量上限")
        self.tasks.add(run_id)
        self._save()

    # 请求前预留输入估算与最大输出，防止多个并发请求同时花费同一剩余额度
    def reserve(self, tokens: int) -> str:
        self.remaining_s()
        projected = self.used_tokens + sum(self.reservations.values()) + tokens
        if self.config.max_total_tokens and projected > self.config.max_total_tokens:
            raise ResourceLimitExceeded("token_budget", "任务树剩余 Token 预算不足以发起下一次请求")
        identity = uuid4().hex
        self.reservations[identity] = tokens
        self._save()
        return identity

    # 模型返回 usage 后校正消费；失败和取消请求保留估算预留，避免恢复后免费重复调用
    def settle(self, identity: str, response: LlmResponse) -> None:
        estimate = self.reservations.pop(identity)
        usage = response.usage
        self.used_tokens += (usage.input_tokens + usage.output_tokens
                             + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
                             if usage is not None else estimate)
        self._save()


class LimitedProvider:
    # 模型调用共用 daemon 信号量，仅在实际请求期间占槽，前台父任务等待子任务不会占槽死锁
    def __init__(
        self, inner: LLMProvider, gate: asyncio.Semaphore, budget: TaskBudget | None = None,
    ) -> None:
        self._inner = inner
        self._gate = gate
        self.budget = budget

    # 排队后再预留预算，自动压缩请求也经由同一预算与并发限制
    async def chat(
        self, messages: list[dict[str, object]], tool_schemas: list[dict[str, object]],
        bus: EventBus, run_id: str, *, step: int = 0, system: str | None = None,
    ) -> LlmResponse:
        if self.budget is None:
            async with self._gate:
                return await self._inner.chat(
                    messages=messages, tool_schemas=tool_schemas, bus=bus, run_id=run_id,
                    step=step, system=system,
                )
        async with asyncio.timeout(self.budget.remaining_s()):
            async with self._gate:
                reservation = self.budget.reserve(estimate_tokens([messages, tool_schemas, system])
                                                  + 8192)
                response = await self._inner.chat(
                    messages=messages, tool_schemas=tool_schemas, bus=bus, run_id=run_id,
                    step=step, system=system,
                )
                self.budget.settle(reservation, response)
                return response


# 恢复旧检查点时沿父链找到树根，新检查点直接使用持久化根 ID
def budget_root(runs_dir: Path, run_id: str, parent_id: str = "") -> str:
    current = run_id
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        folder = runs_dir / current
        path = folder / ("root.json" if (folder / "root.json").exists() else "background.json")
        if not path.exists():
            return parent_id or current
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        root = raw.get("budget_root_id")
        if root:
            return str(root)
        parent = str(raw.get("parent_run_id", ""))
        if not parent:
            return current
        if not parent.replace("-", "").replace("_", "").isalnum():
            raise ValueError("invalid parent run identity")
        current, parent_id = parent, ""
    raise ValueError("cyclic task parent identities")
