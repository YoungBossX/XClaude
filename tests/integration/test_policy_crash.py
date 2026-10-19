from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from x_claude.core.permissions.storage import save_policy_file


# 功能：真实异常退出后授权只能是完整旧版或新版，首次保存中断不得意外放行。
# 设计：子进程在四个原子写窗口 os._exit，新进程从磁盘加载并执行权限决策，残留临时文件不参与加载。
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("boundary", ["partial", "synced", "before_replace", "after_replace"])
def test_policy_crash_reload(tmp_path, boundary, existing):
    path = tmp_path / "policy.toml"
    old = {"bash": "deny", "write_file": "allow"}
    if existing:
        save_policy_file(old, path)
    original = path.read_bytes() if existing else None
    env = {key: value for key, value in os.environ.items()
           if key not in {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"}}
    command = [sys.executable, "-m", "tests.helpers.policy_crash_worker"]
    kwargs = dict(cwd=Path(__file__).resolve().parents[2], env=env,
                  capture_output=True, timeout=15)
    crashed = subprocess.run([*command, boundary, str(path)], **kwargs)
    assert crashed.returncode == 73, crashed.stderr.decode("utf-8", errors="replace")
    after = boundary == "after_replace"
    if not after:
        assert (path.read_bytes() if path.exists() else None) == original
    expected = {"bash": "allow", "write_file": "deny"} if after else (old if existing else {})
    if path.exists():
        assert tomllib.loads(path.read_text(encoding="utf-8"))["always"] == expected
    inspected = subprocess.run([*command, "inspect", str(path)], **kwargs)
    assert inspected.returncode == 0, inspected.stderr.decode("utf-8", errors="replace")
    result = json.loads(inspected.stdout)
    assert result["policy"] == expected
    if after:
        assert result["decisions"]["bash"] == [True, "auto_allow"]
        assert result["decisions"]["write_file"] == [False, "auto_deny"]
    elif existing:
        assert result["decisions"]["bash"] == [False, "auto_deny"]
        assert result["decisions"]["write_file"] == [True, "auto_allow"]
    else:
        assert result["decisions"]["bash"] == [False, "deny_once"]
        assert result["decisions"]["write_file"] == [False, "deny_once"]
    assert result["decisions"]["unregistered"] == [False, "deny_once"]
    assert result["requested"] == (["unregistered"] if expected else ["bash", "write_file", "unregistered"])
    assert bool(list(tmp_path.glob(".policy.toml.*.tmp"))) == (not after)
