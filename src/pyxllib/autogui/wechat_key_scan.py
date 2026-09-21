"""只读提取指定微信进程的 WCDB Config.Cipher，并逐库 HMAC 验证。

4.1.13 协议布局参考：
https://github.com/fanyuantaier/wechatauto-replica/blob/main/wechatauto/db.py
密钥在配置 blob 中经过 XOR 编码，扫描明文 key 或旧 cfg 主密钥不能替代此契约。
"""
import ctypes
from ctypes import wintypes
import hashlib
import hmac
from pathlib import Path
import re
import struct

CONFIG_NAME = b"com.Tencent.WCDB.Config.Cipher"
CONFIG_MASK = bytes.fromhex("d2c7442458020000004889442450488b450048844c2448488944254048584c24")


def verify_database_key(key: bytes, page: bytes) -> bool:
    """候选必须通过目标库第 1 页认证，不能因进程/目录名称匹配就认为可用。"""
    if len(key) != 32 or len(page) != 4096:
        return False
    mac_key = hashlib.pbkdf2_hmac("sha512", key, bytes(x ^ 0x3a for x in page[:16]), 2, 32)
    digest = hmac.new(mac_key, page[16:4032] + struct.pack("<I", 1), hashlib.sha512).digest()
    return hmac.compare_digest(digest, page[4032:])


def scan_account_keys(pid: int, db_root: Path) -> dict:
    """仅访问指定 PID，返回目标库认证通过的 key；不输出密钥，不更换账号。"""
    class Region(ctypes.Structure):
        _fields_ = [("base", ctypes.c_void_p), ("allocation_base", ctypes.c_void_p),
                    ("allocation_protect", wintypes.DWORD), ("size", ctypes.c_size_t),
                    ("state", wintypes.DWORD), ("protect", wintypes.DWORD), ("type", wintypes.DWORD)]

    win = ctypes.WinDLL("kernel32", use_last_error=True)
    win.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    win.OpenProcess.restype = wintypes.HANDLE
    win.VirtualQueryEx.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.POINTER(Region), ctypes.c_size_t]
    win.VirtualQueryEx.restype = ctypes.c_size_t
    win.ReadProcessMemory.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    win.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = win.OpenProcess(0x410, False, pid)
    if not handle:
        raise OSError("无法读取指定微信进程")

    def read(address, length):
        if not 0x10000 <= address < 0x800000000000:
            return b""
        buffer = ctypes.create_string_buffer(length)
        count = ctypes.c_size_t()
        win.ReadProcessMemory(handle, address, buffer, length, ctypes.byref(count))
        return buffer.raw[:count.value]

    def find(needles):
        hits = set()
        address = 0
        region = Region()
        while win.VirtualQueryEx(handle, address, ctypes.byref(region), ctypes.sizeof(region)):
            base, size = int(region.base or 0), int(region.size)
            if region.state == 0x1000 and not region.protect & 0x101:
                for offset in range(0, size, 4 * 1024 * 1024):
                    data = read(base + offset, min(size - offset, 4 * 1024 * 1024 + 64))
                    for needle in needles:
                        position = data.find(needle)
                        while position >= 0:
                            hits.add(base + offset + position)
                            position = data.find(needle, position + 1)
            if base + size <= address:
                break
            address = base + size
        return hits

    candidates = set()
    try:
        names = find([CONFIG_NAME])
        pairs = [struct.pack("<QQ", address, len(CONFIG_NAME)) for address in names]
        for address in find(pairs) if pairs else []:
            node = read(address - 16, 64)
            if len(node) != 64 or struct.unpack_from("<Q", node, 16)[0] not in names:
                continue
            config = struct.unpack_from("<Q", node, 40)[0]
            descriptor = read(config + 0x88, 24)
            if len(descriptor) != 24:
                continue
            pointer, length = struct.unpack_from("<QQ", descriptor, 8)
            if not 0 < length <= 1024:
                continue
            blob = read(pointer, length)
            if len(blob) != length:
                continue
            decoded = bytes(value ^ CONFIG_MASK[index % len(CONFIG_MASK)] for index, value in enumerate(blob))
            for match in re.finditer(rb"[xX]'([0-9a-fA-F]{64,192})'", decoded):
                run = match.group(1).decode()
                for offset in range(0, len(run) - 63, 32):
                    candidates.add(bytes.fromhex(run[offset:offset + 64]))
    finally:
        win.CloseHandle(handle)
    matches = {}
    for db in Path(db_root).rglob("*.db"):
        with db.open("rb") as stream:
            page = stream.read(4096)
        for key in candidates:
            if verify_database_key(key, page):
                matches[db.relative_to(db_root).as_posix()] = {"key_hex": key.hex(), "mode": "raw-derived-key"}
                break
    return matches
