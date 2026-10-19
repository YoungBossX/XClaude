from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from x_claude.core import atomic_file
from x_claude.core.permissions.manager import PermissionManager
from x_claude.core.permissions.storage import load_policy_file, save_policy_file


# 在指定真实文件提交窗口异常退出，不运行 finally 清理或正常关闭路径。
def crash_write(path: Path, boundary: str) -> None:
    real_fdopen, real_fsync, real_replace = os.fdopen, os.fsync, os.replace

    class PartialWriter:
        # 持有真实临时文件句柄，仅将退出点插入一次写入内部。
        def __init__(self, stream):
            self.stream = stream

        # 返回可观察实际写入的句柄包装。
        def __enter__(self):
            return self

        # 正常路径关闭句柄，异常退出不经过此方法。
        def __exit__(self, *args):
            self.stream.close()

        # 写入前半部并同步，再立即退出以留下真实未完成的临时文件。
        def write(self, content):
            self.stream.write(content[:len(content) // 2])
            self.stream.flush()
            real_fsync(self.stream.fileno())
            os._exit(73)

    # 完成真实 fsync 后退出，目标文件还未替换。
    def synced(fd):
        real_fsync(fd)
        os._exit(73)

    # 在真实 replace 前或后退出，测试两个可观察提交状态。
    def replaced(source, target):
        if boundary == "before_replace":
            os._exit(73)
        real_replace(source, target)
        os._exit(73)

    if boundary == "partial":
        atomic_file.os.fdopen = lambda *args: PartialWriter(real_fdopen(*args))
    elif boundary == "synced":
        atomic_file.os.fsync = synced
    else:
        atomic_file.os.replace = replaced
    save_policy_file({"bash": "allow", "write_file": "deny"}, path)
    raise AssertionError("crash boundary was not reached")


# 新进程重新加载权限文件并走真实审批逻辑，默认询问由测试显式拒绝。
async def inspect_policy(path: Path) -> dict:
    manager = PermissionManager(policy_file=path, timeout_s=1)
    requested = []

    # 对没有持久授权的操作模拟拒绝，不将其误算为自动拒绝。
    async def deny(event):
        requested.append(event["tool_name"])
        assert manager.respond(event["tool_use_id"], "deny_once", session_id="s", run_id="r")

    decisions = {}
    for tool in ("bash", "write_file", "unregistered"):
        decisions[tool] = await manager.check_and_wait(
            tool, tool, {"command": "printf safe"}, "s", deny, run_id="r",
        )
    return {"policy": load_policy_file(path), "decisions": decisions, "requested": requested}


# 只接受测试父进程提供的绝对路径，写入或读取同一个临时策略文件。
def main() -> None:
    mode, name = sys.argv[1:3]
    path = Path(name)
    assert path.is_absolute() and path.name == "policy.toml"
    if mode == "inspect":
        print(json.dumps(asyncio.run(inspect_policy(path))))
    else:
        crash_write(path, mode)


if __name__ == "__main__":
    main()
