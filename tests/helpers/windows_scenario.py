from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from tests.integration.test_runtime_hardening import _serve
from x_claude.core.llm.types import LlmResponse, ToolCallBlock

COMMANDS = {
    "unicode": "mkdir -p '中文 空格'; printf '验收通过' > '中文 空格/结果.txt'; cat '中文 空格/结果.txt'",
    "denied": "printf forbidden > denied.txt",
    "timeout": "printf ready > ready; sleep 3; printf escaped > escaped.txt",
    "cancel": "printf ready > ready; sleep 3; printf escaped > escaped.txt",
}


class WindowsScenarioProvider:
    # 固定测试动作，真实工具、审批、循环和事件仍走生产代码，不调用网络模型。
    async def chat(self, messages, step, **kwargs):
        if step == 1:
            goal = messages[-1]["content"]
            if goal in COMMANDS:
                return LlmResponse("tool_use", tool_calls=[ToolCallBlock(
                    "shell", "bash", {"command": COMMANDS[goal], "timeout": 1 if goal == "timeout" else 10},
                )])
        return LlmResponse("end_turn", text="WINDOWS_SCENARIO_DONE")


# 独立终端验收服务仅运行固定场景，写出事件后由外层测试观察并结束。
async def serve(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    async with _serve(root, WindowsScenarioProvider(), permission_timeout=45) as (cfg, _, bus, _):
        # 记录真实服务端事件，供外层终端验收核对授权与会话生命周期。
        async def collect(event):
            with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(event.model_dump_json() + "\n")

        bus.subscribe(collect)
        (root / "ready.json").write_text(json.dumps({"port": cfg.port}), encoding="utf-8")
        print(json.dumps({"port": cfg.port, "root": str(root)}), flush=True)
        async with asyncio.timeout(300):
            while not (root / "stop").exists():
                await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(serve(Path(sys.argv[1]).resolve()))
