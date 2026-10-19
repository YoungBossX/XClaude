from __future__ import annotations

from pathlib import Path

from x_claude.core.atomic_file import atomic_write_bytes

_DEFAULT_POLICY_PATH = Path("~/.x/policy.toml")


# 加载 policy.toml 中 [always] 节，返回 {tool_name: "allow"/"deny"}；文件不存在时返回空字典
def load_policy_file(path: Path | None = None) -> dict[str, str]:
    p = (path or _DEFAULT_POLICY_PATH).expanduser()
    if not p.exists():
        return {}
    result: dict[str, str] = {}
    in_always = False
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[always]":
            in_always = True
            continue
        if stripped.startswith("["):
            in_always = False
            continue
        if in_always and "=" in stripped and not stripped.startswith("#"):
            k, _, v = stripped.partition("=")
            k = k.strip()
            v = v.strip().strip('"')
            if v in ("allow", "deny"):
                result[k] = v
    return result


# 将授权规则完整同步后原子替换 policy.toml，提交失败保留原文件
def save_policy_file(always: dict[str, str], path: Path | None = None) -> None:
    p = (path or _DEFAULT_POLICY_PATH).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ~/.x/policy.toml",
        "# 由 x-core 自动管理，手动编辑生效但格式须正确",
        "",
        "[always]",
    ]
    for tool, decision in sorted(always.items()):
        lines.append(f'{tool} = "{decision}"')
    atomic_write_bytes(p, ("\n".join(lines) + "\n").encode("utf-8"))
