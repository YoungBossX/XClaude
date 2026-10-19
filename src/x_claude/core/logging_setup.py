from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path

from x_claude.core.config import XConfig

_TEXT_FMT = 'level=%(levelname)s ts=%(asctime)s source=%(name)s msg="%(message)s"'


class _JsonFormatter(logging.Formatter):
    # 将日志正文和可选堆栈完整转义为一行 JSON，保留原有字段名称
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "level": record.levelname,
            "ts": self.formatTime(record, self.datefmt),
            "source": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            data["exception"] = record.exc_text
        if record.stack_info:
            data["stack"] = self.formatStack(record.stack_info)
        return json.dumps(data, ensure_ascii=False)


# 根据配置初始化 root logger：设置级别、格式，并挂载 stderr 和可选的滚动文件 handler
def setup_logging(config: XConfig) -> None:
    level = getattr(logging, config.logging.level.upper(), logging.INFO)
    formatter = (
        _JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        if config.logging.format == "json"
        else logging.Formatter(_TEXT_FMT, datefmt="%Y-%m-%dT%H:%M:%S")
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    if config.logging.file:
        log_path = Path(config.logging.file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
