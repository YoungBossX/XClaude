# 运维手册（RUNBOOK）

## 日常操作

### 启动守护进程

```bash
uv run x-core
```

默认监听 `127.0.0.1:7437`，按 `Ctrl+C` 优雅退出。

### 验证连通

```bash
uv run x ping
# → pong server=0.0.1 uptime=12ms latency=2ms
```

### 停止守护进程

```bash
kill $(pgrep -f x-core)
```

---

## 配置

优先级（低 → 高）：**内建默认值 → `~/.x/config.toml` → `.env` → 系统环境变量**。

### `~/.x/config.toml`

```toml
[core]
host = "127.0.0.1"
port = 7437

[logging]
level  = "INFO"
file   = "~/.x/logs/core.log"
format = "text"    # "text" | "json"
```

### `.env`

从 `.env.example` 复制后修改，存放本机配置与密钥（不提交 git）：

```bash
cp .env.example .env
```

### 系统环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `X_CONFIG` | `~/.x/config.toml` | 覆盖配置文件路径 |
| `X_HOST` | `127.0.0.1` | TCP 监听地址 |
| `X_PORT` | `7437` | TCP 监听端口 |
| `X_LOG_LEVEL` | `INFO` | 日志级别（DEBUG / INFO / WARNING / ERROR） |
| `X_LOG_FILE` | `~/.x/logs/core.log` | 日志文件路径（留空则仅输出 stderr） |
| `X_LOG_FORMAT` | `text` | 日志格式（`text` 或 `json`） |
| `X_COMPACT_THRESHOLD` | `0.8` | 自动摘要压缩阈值，范围 0–1；设为 0 禁用 |

自动压缩默认开启：在工具调用步骤结束、任务仍需继续且 `context_pct >= 0.8` 时触发。
压缩成功后以摘要和助手确认消息替换历史，保存摘要文件，并备份原会话历史；后续对话从压缩后的上下文续接。
压缩失败或摘要为空时保留原上下文，继续执行。`/compact` 手动压缩仍然可用。

`context_pct` 当前使用最近一次模型响应的 `input_tokens / context_window`，尚未计入缓存 token
和本步新追加的工具结果；压缩不会额外保留最新 1–2 轮原始消息。
修改配置后需重启 `x-core` 生效。

---

## 开发

```bash
uv run ruff check src tests scripts   # lint
uv run mypy src                       # 类型检查
uv run pytest tests/ -v               # 全量测试
uv run pytest tests/unit/ -v         # 仅单元测试（无需启动 daemon）

make docs                             # 重新生成 WIRE_PROTOCOL.md
make verify-s0                        # 完整验证（lint + 类型 + 测试 + 协议同源检查）
```

---

## 日志

```bash
tail -f ~/.x/logs/core.log
```

---

## 常见错误

| 报错 | 原因 | 处理 |
|------|------|------|
| `core already running at 127.0.0.1:7437` | 已有守护进程在运行 | `kill $(pgrep -f x-core)` |
| `core not running` | 未启动守护进程 | `uv run x-core` |
| `Address already in use` | 端口被其他进程占用 | `X_PORT=8000 uv run x-core` |
| `Config error: X_PORT must be an integer` | `.env` 或环境变量中端口值非整数 | 检查 `X_PORT` 的值 |
