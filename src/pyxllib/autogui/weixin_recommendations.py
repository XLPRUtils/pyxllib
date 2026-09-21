"""微信推荐页的账号隔离诊断入口；读取现场，不发送消息、不驱动 GUI。

推荐流与聊天库是独立数据源。调查时通过账号所属进程中的已知卡片标题定位
加载过的响应片段；这些片段只是协议取证，不可当作完整原图采集清单。
"""
import ctypes
from ctypes import wintypes
import psutil

from pyxllib.autogui.weixin4_instrumentation import resolve_sender


def inspect_loaded_feed(account_id: str, titles: list[str], *, max_hits: int = 4) -> dict:
    """只读定位指定账号已加载的标题片段；每进程/标题/编码最多 max_hits 处。

    允许检索协议字段以调查推荐响应，返回内容可能包含凭证，不应写入公开日志。
    同一内存块也可出现多条卡片，不能仅提取首次匹配，否则会遗漏后续推荐。
    """
    if not titles or any(len(title) < 5 for title in titles):
        raise ValueError("必须提供当前推荐卡片中至少五字的标题片段")
    sender = resolve_sender(account_id)
    parent = psutil.Process(sender["pid"])
    processes = [parent] + [p for p in parent.children(recursive=True)
                            if p.name().lower() in {"weixin.exe", "wechatappex.exe"}]

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
    patterns = [(title, encoding, title.encode(encoding)) for title in titles for encoding in ("utf-8", "utf-16-le")]
    fragments = []
    for process in processes:
        handle = win.OpenProcess(0x410, False, process.pid)
        if not handle:
            continue
        counts = {}
        try:
            region = Region()
            address = 0
            while win.VirtualQueryEx(handle, address, ctypes.byref(region), ctypes.sizeof(region)):
                base, size = int(region.base or 0), int(region.size)
                if region.state == 0x1000 and region.type == 0x20000 and not region.protect & 0x101:
                    for offset in range(0, size, 4 * 1024 * 1024):
                        length = min(size - offset, 4 * 1024 * 1024 + 65536)
                        buffer = ctypes.create_string_buffer(length)
                        count = ctypes.c_size_t()
                        win.ReadProcessMemory(handle, base + offset, buffer, length, ctypes.byref(count))
                        data = buffer.raw[:count.value]
                        for title, encoding, pattern in patterns:
                            position = data.find(pattern)
                            while position >= 0 and counts.get((title, encoding), 0) < max_hits:
                                start = max(0, position - 16384)
                                end = min(len(data), position + 32768)
                                fragments.append({"pid": process.pid, "title": title, "encoding": encoding,
                                                  "text": data[start:end].decode(encoding, errors="replace")})
                                counts[title, encoding] = counts.get((title, encoding), 0) + 1
                                # 相邻卡片已包含在窗口内；推进到窗口末端，避免重复大段现场。
                                position = data.find(pattern, end)
                if base + size <= address:
                    break
                address = base + size
        finally:
            win.CloseHandle(handle)
    if resolve_sender(account_id) != sender:
        raise RuntimeError("取证期间账号进程变化，丢弃结果")
    return {"account_id": account_id, "process_ids": [p.pid for p in processes], "fragments": fragments}
