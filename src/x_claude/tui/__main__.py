from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
from pathlib import Path

from x_claude.core.config import get_config
from x_claude.tui.app import XTuiApp

_DEFAULT_TUI_LOG = "~/.x/logs/tui.log"


# TUI 文件日志初始化：不写 stderr（避免干扰 Textual 渲染），只写滚动文件
def _setup_logging(level: str) -> None:
    log_path = Path(os.environ.get("X_TUI_LOG_FILE", _DEFAULT_TUI_LOG)).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter(
            'level=%(levelname)s ts=%(asctime)s source=%(name)s msg="%(message)s"',
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    root.handlers.clear()
    root.addHandler(handler)


# x-tui 入口：解析回放与会话续接参数后启动 TUI 应用
def main() -> None:
    parser = argparse.ArgumentParser(prog="x-tui", description="XClaude TUI")
    parser.add_argument(
        "--replay",
        metavar="RUN_ID",
        help="Replay events from a past run on connect",
    )
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "-c",
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Continue the most recent chat session",
    )
    session_group.add_argument(
        "-r",
        "--resume",
        metavar="SESSION_ID",
        dest="resume_session_id",
        help="Resume a chat session by ID",
    )
    args = parser.parse_args()

    config = get_config()
    _setup_logging(config.logging.level)
    app = XTuiApp(
        config.host,
        config.port,
        replay_run_id=args.replay,
        continue_session=args.continue_session,
        resume_session_id=args.resume_session_id,
    )
    app.run()


if __name__ == "__main__":
    main()
