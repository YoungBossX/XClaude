from __future__ import annotations

import asyncio

from x_claude.core.context import ExecutionContext


# 管理后台 subagent 任务的生命周期：注册、查询、批量取消
class BackgroundTaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, tuple[asyncio.Future[None], ExecutionContext]] = {}
        self.suspending = False

    # 注册一个后台任务及其执行上下文
    def register(
        self,
        run_id: str,
        task: asyncio.Future[None],
        context: ExecutionContext,
    ) -> None:
        self._tasks[run_id] = (task, context)

    # 查询后台任务及其上下文；不存在时返回 None
    def get(self, run_id: str) -> tuple[asyncio.Future[None], ExecutionContext] | None:
        return self._tasks.get(run_id)

    # 返回所有已注册的 (task, context) 对，用于 daemon 退出时批量清理
    def all(self) -> list[tuple[asyncio.Future[None], ExecutionContext]]:
        return list(self._tasks.values())

    # 只释放已停止任务的缓存，运行中的任务不能被恢复请求覆盖或重复启动
    def forget_stopped(self, run_id: str) -> None:
        entry = self._tasks.get(run_id)
        if entry is not None and not entry[0].done():
            raise ValueError("background task is still running")
        self._tasks.pop(run_id, None)

    # 取消并等待后台任务结束，回收异常，最后清空会话内任务注册表
    async def cancel_all(self, *, suspend: bool = False) -> None:
        self.suspending = suspend
        tasks = [task for task, _ in self._tasks.values()]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
