# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install / sync dependencies
uv sync

# Lint
uv run ruff check src tests scripts
uv run mypy src

# Tests
uv run pytest tests/unit -v           # unit only (fast, no daemon)
uv run pytest tests/integration -v -m "not integration" # local TCP/TUI/process tests
uv run pytest tests/ -v -m "not integration"            # excludes the real API test

# Single test
uv run pytest tests/unit/test_envelope.py::test_request_roundtrip -v

# Regenerate WIRE_PROTOCOL.md after changing bus models
uv run python scripts/gen_protocol_doc.py

# Verify generated protocol documentation locally
uv run python scripts/gen_protocol_doc.py --check

# Run daemon manually
uv run x-core                        # foreground; Ctrl+C to stop
# PowerShell: $env:X_PORT = "8000"; uv run x-core
# POSIX shell: X_PORT=8000 uv run x-core

# Send a ping
uv run x ping
uv run x --version
```

## Architecture

This is a **dual-process** local AI agent system. `x-core` is a persistent daemon; `x` and `x-tui` connect through authenticated loopback TCP using JSON-RPC 2.0 and NDJSON.

```
x-core (daemon)
  └─ listens on 127.0.0.1:7437 (TCP)
       ↑ JSON-RPC 2.0 NDJSON
x (CLI)   x-tui (TUI)
```

**`x-tui` is the primary frontend.** All user-facing work on task management, observability, and interaction should be designed for and validated in the TUI first. The `x` CLI exists only for quick scripted testing and debugging — it is not a product surface. When implementing features that touch the user interface, invest in the TUI layout, event rendering, and keyboard interactions. Do not shortcut TUI work by pointing to the CLI as an alternative.

### Protocol layer (`src/x_claude/core/bus/`)

All IPC messages are typed pydantic v2 models with a **discriminated union on the `type` field**. This is the contract boundary — adding a new command or event means adding a new model class to `commands.py` or `events.py` and extending the `Command`/`Event` union.

- `envelope.py` — `JsonRpcRequest`, `JsonRpcSuccess`, `JsonRpcError`, error code constants, `make_error()`
- `commands.py` — ping, one-shot runs, session create/continue/resume/send/history/close/clear/compact/recover, event subscriptions and permission responses, with typed results.
- `events.py` — run/step/tool/LLM/session/subagent/permission/compaction/skill events. Runtime events carry durable IDs; IPC also carries per-log byte ranges for incremental replay.

`WIRE_PROTOCOL.md` is **generated** from these models by `scripts/gen_protocol_doc.py`. Always regenerate and commit it after changing bus models.

### Transport layer (`src/x_claude/core/transport/`)

- `socket_server.py` validates local IPC credentials, dispatches concurrent requests, and isolates disconnected clients. Register handlers via `server.register("method.name", handler_fn)`.
- `ipc_broadcaster.py` filters session/run scopes, queues live events during replay, and binds permission responses to the authorized connection.
- `event.subscribe` supports incremental session replay using acknowledged log offsets. Initial TUI replay is bounded; `/history` opens read-only message pages.
- One OS lock per session-store directory prevents concurrent daemon writers even when ports differ. Keep the reusable `.daemon.lock` file.

### Config (`src/x_claude/core/config.py`)

Priority: **built-in defaults → `~/.x/config.toml` → project `.x/config.toml` → environment overrides**. `.env` fills missing environment variables; existing OS variables win. Setting `X_CONFIG` selects one TOML file instead of the global/project pair.

TOML sections: `[core]`, `[logging]`, `[agent]`, `[llm]`, `[trace]`, `[permission]`, `[compaction]` and `[mcp]`. Unknown keys fail validation. See `RUNBOOK.md` and `.env.example` for supported settings.

Agent limits include steps, concurrent model requests, total tasks, task-tree Token budget and runtime deadline. Auto compaction defaults to 80%; compatibility models should set their actual `X_CONTEXT_WINDOW`.

### Daemon entry (`src/x_claude/core/app.py`)

`CoreApp.run()` loads config, acquires the store lock, starts trace/permissions/MCP/session management and TCP, then waits for shutdown. Cleanup stops new requests, suspends recoverable tasks, closes clients and resources, and finally releases the lock. Windows uses a standard signal callback fallback.

Task checkpoints and budget ledgers are persisted atomically. Unconfirmed interrupted tools require `/recover` review and are never automatically replayed. `/clear` and `/exit` retain explicit cancellation semantics.

### Testing

Integration tests use temporary projects/stores, real TCP and subprocesses, and Textual's mounted test UI. The `running_daemon` fixture starts a real daemon on a random port with isolated session storage. The pytest marker named `integration` specifically selects real model API tests; it is not synonymous with the tests/integration directory.

### Code style

All functions must have a **single-line Chinese comment** immediately above the `def` line explaining what the function does. Example:

```python
# 发送 JSON-RPC 响应并刷新写缓冲区
async def _send(self, writer: asyncio.StreamWriter, msg: BaseModel) -> None:
    ...
```

Do not write multi-line docstrings; one concise Chinese line is enough.

**Test functions** require **two Chinese comment lines** immediately above the `def` line:

```python
# 功能：验证 publish 后订阅者能收到事件对象
# 设计：用内联 handler 收集事件引用，断言 is 而非 ==，排除序列化中间步骤的干扰
async def test_publish_reaches_subscriber() -> None:
    ...
```

- `# 功能：` — 该测试验证的具体行为或不变式，一句话说清楚"测什么"
- `# 设计：` — 为什么选择这种测试方式：覆盖了什么边界条件、为什么用这个 stub/fixture、这种断言方式相比其他方式的优势

两行注释缺一不可。功能行让读者 5 秒内判断测试意图；设计行让读者理解测试背后的决策，而非只看到操作步骤。

### Maintained references

- `README.md` — user-facing features and launch commands.
- `RUNBOOK.md` — configuration, recovery and operational boundaries.
- `WIRE_PROTOCOL.md` — generated current command/event contracts.
