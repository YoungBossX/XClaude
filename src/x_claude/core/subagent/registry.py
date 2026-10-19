from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from x_claude.core.context import ExecutionContext
from x_claude.core.subagent.checkpoint import BackgroundCheckpoint


# 管理后台 subagent 任务的生命周期：注册、查询、批量取消
class BackgroundTaskRegistry:
    # 限制已完成结果缓存，运行中与尚未安全落盘的状态不参与淘汰
    def __init__(self, *, max_completed: int = 64, max_result_chars: int = 1_000_000) -> None:
        if max_completed < 0 or max_result_chars < 0:
            raise ValueError("cache limits must be nonnegative")
        self._tasks: OrderedDict[str, tuple[asyncio.Future[None], ExecutionContext]] = OrderedDict()
        self._max_completed = max_completed
        self._max_result_chars = max_result_chars
        self._runs_dir: Path | None = None
        self._session_id = ""
        self.suspending = False
        self._foreground_children: dict[str, dict[str, ExecutionContext]] = {}

    # 仅在同步派生执行期间记录依赖；结束、异常或取消时立即移除，不持久化通知状态。
    @contextmanager
    def foreground_child(self, parent_run_id: str, context: ExecutionContext) -> Iterator[None]:
        children = self._foreground_children.setdefault(parent_run_id, {})
        children[context.run_id] = context
        try:
            yield
        finally:
            children.pop(context.run_id, None)
            if not children:
                self._foreground_children.pop(parent_run_id, None)

    # 遍历当前查询任务的前台子调用及后台等待依赖，不等待同会话内无关任务的审批。
    def permission_dependencies(self, run_id: str) -> tuple[str, ...]:
        visited: set[str] = set()
        pending: list[tuple[str, ExecutionContext | None]] = [(run_id, None)]
        while pending:
            current, context = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            entry = self._tasks.get(current)
            if context is None and entry is not None:
                context = entry[1]
            if context is not None:
                pending.extend((child, None) for child in context.permission_wait_runs)
            pending.extend(self._foreground_children.get(current, {}).items())
        return tuple(sorted(visited))

    # 查询磁盘结果只允许绑定的会话目录，不跨会话共享索引
    def bind_storage(self, runs_dir: Path, session_id: str) -> None:
        resolved = runs_dir.resolve()
        if self._runs_dir is not None and (resolved, session_id) != (
            self._runs_dir, self._session_id,
        ):
            raise ValueError("background registry cannot change session storage")
        self._runs_dir, self._session_id = resolved, session_id

    # 注册一个后台任务及其执行上下文
    def register(
        self,
        run_id: str,
        task: asyncio.Future[None],
        context: ExecutionContext,
    ) -> None:
        self._tasks[run_id] = (task, context)
        if task.done():
            self._completed(run_id, task)
        else:
            task.add_done_callback(partial(self._completed, run_id))

    # 查询缓存或已完成检查点；按需读盘不启动模型、恢复执行或重新调用工具
    def get(self, run_id: str) -> tuple[asyncio.Future[None], ExecutionContext] | None:
        entry = self._tasks.get(run_id)
        if entry is not None:
            self._tasks.move_to_end(run_id)
            return entry
        context = self._saved_result(run_id)
        if context is None:
            return None
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        if context.reason == "cancelled":
            future.cancel()
        else:
            future.set_result(None)
        self.register(run_id, future, context)
        return future, context

    # 校验目录及会话身份，仅把终态检查点作为可重新读取的结果
    def _saved_result(self, run_id: str) -> ExecutionContext | None:
        if (self._runs_dir is None or not run_id
                or not run_id.isascii() or not run_id.replace("-", "").replace("_", "").isalnum()):
            return None
        path = self._runs_dir / run_id / "background.json"
        try:
            checkpoint = BackgroundCheckpoint.load(path, self._session_id)
        except (OSError, ValueError):
            return None
        if checkpoint.record.context.status == "running":
            return None
        saved = checkpoint.record.context
        result = ExecutionContext(run_id, "", saved.max_steps, step=saved.step,
                                  status=saved.status, reason=saved.reason, result=saved.result)
        result.messages.clear()
        return result

    # 完成回调不能覆盖后来重新登记的同名任务，也不能修改执行上下文本身
    def _completed(self, run_id: str, task: asyncio.Future[None]) -> None:
        entry = self._tasks.get(run_id)
        if entry is None or entry[0] is not task:
            return
        context = entry[1]
        saved = self._saved_result(run_id)
        if saved is not None and (saved.status, saved.reason, saved.result) == (
            context.status, context.reason, context.result,
        ):
            self._tasks[run_id] = task, saved
        self._tasks.move_to_end(run_id)
        self._prune()

    # 仅淘汰已经确认落盘且可重读的终态；磁盘失败时保留最后的内存结果
    def _prune(self) -> None:
        completed = [(key, task, context) for key, (task, context) in self._tasks.items()
                     if task.done()]
        count = len(completed)
        chars = sum(len(ctx.result) + len(ctx.reason or "") for _, _, ctx in completed)
        for key, task, context in completed:
            if count <= self._max_completed and chars <= self._max_result_chars:
                break
            if task.cancelled() and context.reason != "cancelled":
                continue
            if not task.cancelled() and task.exception() is not None:
                continue
            saved = self._saved_result(key)
            if saved is None or (saved.status, saved.reason, saved.result) != (
                context.status, context.reason, context.result,
            ):
                continue
            self._tasks.pop(key)
            count -= 1
            chars -= len(context.result) + len(context.reason or "")

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
