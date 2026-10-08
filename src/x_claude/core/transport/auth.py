from __future__ import annotations

import csv
import ctypes
import hashlib
import ipaddress
import os
import subprocess
from pathlib import Path

from x_claude.core.atomic_file import atomic_write_bytes


# 仅允许回环地址，认证令牌不能把明文 TCP 变成可安全暴露的远程服务
def local_host(host: str) -> str:
    if host.lower() == "localhost":
        return "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError("IPC host must be a loopback address") from None
    if not address.is_loopback:
        raise ValueError("IPC host must be a loopback address")
    return str(address)


# 按本机监听地址和端口定位凭据，不把令牌放进项目配置或版本库
def credential_path(host: str, port: int) -> Path:
    endpoint = f"{local_host(host)}:{port}".encode()
    name = hashlib.sha256(endpoint).hexdigest()[:24]
    return Path("~/.x/ipc").expanduser() / f"{name}.token"


# 设置仅当前用户可访问的 ACL；Windows 使用受保护 DACL，失败时拒绝发布凭据
def restrict_access(path: Path) -> None:
    if os.name != "nt":
        path.chmod(0o700 if path.is_dir() else 0o600)
        return
    result = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True,
        text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    sid = next(csv.reader(result.stdout.splitlines()))[1]
    if not sid.startswith("S-1-") or any(c not in "S-0123456789" for c in sid):
        raise OSError("cannot determine Windows account SID")
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong,
                        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    convert.restype = ctypes.c_int
    apply = advapi.SetFileSecurityW
    apply.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_void_p]
    apply.restype = ctypes.c_int
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    descriptor = ctypes.c_void_p()
    flags = "OICI" if path.is_dir() else ""
    if not convert(f"D:P(A;{flags};FA;;;{sid})", 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not apply(str(path), 0x80000004, descriptor):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.LocalFree(descriptor)


# 在私有目录原子发布随机凭据，新文件继承私有目录 ACL
def publish_credential(host: str, port: int, token: str) -> Path:
    path = credential_path(host, port)
    path.parent.mkdir(parents=True, exist_ok=True)
    if (path.parent.is_symlink() or path.is_symlink()
            or path.parent.is_junction() or path.is_junction()):
        raise PermissionError("IPC credential path must not be a link")
    restrict_access(path.parent)
    atomic_write_bytes(path, token.encode("ascii"))
    restrict_access(path)
    return path


# 客户端每次重连重新读取本机凭据；未发现凭据时由服务器明确拒绝认证
def read_credential(host: str, port: int) -> str | None:
    try:
        return credential_path(host, port).read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
