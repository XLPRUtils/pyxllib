"""Weixin.dll 发送链布局解析（版本快路径 + 签名/拓扑回退）。

分层契约（对齐凡修接口层 Runtime 目标分层解析与恢复）：

- L0 版本快路径：已知 ``sha256`` 直接命中不可变 offsets/struct 表。
- L1 单点强锚：``getService`` 的 46 字节签名在整个 DLL 内唯一。
- L2 有界拓扑回退：沿 ``sendEntry -> getCoro -> getService -> getContext -> doSend``
  的紧邻调用关系消歧短签名；再从 message 构造器的字符串初始化模式推导正文偏移。
- L4 失败关闭：任何一层歧义或缺失都抛 :class:`RebindError` 并附带证据。

解析结果按 ``sha256`` 缓存；DLL 哈希变化即视为新代际，重新解析。
本模块只读取 DLL 文件字节，不附加、不注入、不调用微信进程。
"""

from __future__ import annotations

import bisect
import struct
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

# --- 稳定签名（跨版本未见变化的前导字节） ---

SIG_GET_CORO = bytes.fromhex("564883ec404889ce")
SIG_GET_SERVICE = bytes.fromhex(
    "4889d0488b51304885d27414f0ff4208488b5130488b4928488908"
    "48895008c331d2488b492848890848895008c3"
)
SIG_GET_CONTEXT = bytes.fromhex("5556574881ec90000000488d")
SIG_DO_SEND = bytes.fromhex("554157415641554154565753")
SIG_SEND_ENTRY = bytes.fromhex("5556574881ecd0000000488d")
SIG_MESSAGE_CTOR = bytes.fromhex("5556574883ec50488d6c2450")

# 消息结构常量；除 content 会随版本漂移外，其余在 4.x 发送链上保持稳定。
DEFAULT_STRUCT: dict[str, int] = {
    "object_size": 0x1000,
    "content": 0x758,
    "wxid": 0xB0,
    "length": 0x1C8,
    "flag": 0x118,
    "kind": 0x9C,
    "subtype": 0x180,
    "flag_value": 1,
    "kind_value": 1,
    "subtype_value": 0x77,
}

# L0：sha256 -> 已验证布局。新增版本可加一行，或交给 L2 自动 rebind。
KNOWN_LAYOUTS: dict[str, dict[str, Any]] = {
    "7AD9753D11C2BAF5C900AAC50DDF56A8170AA85C46129D661325FF88505BEFB1": {
        "version": "4.1.12.55",
        "offsets": {
            "get_coro": 0x42010,
            "get_service": 0x339480,
            "get_context": 0x6CEA20,
            "message_ctor": 0x738E90,
            "do_send": 0x1734330,
            "send_entry": 0x197D500,
        },
        "struct": dict(DEFAULT_STRUCT),
    },
    "8F7406A8A465E851EE10EAECCCB572D0E5B4E00D480C770938C714C396097B74": {
        "version": "4.1.13.65",
        "offsets": {
            "get_coro": 0x43AB0,
            "get_service": 0x3426C0,
            "get_context": 0x6EC950,
            "message_ctor": 0x7574F0,
            "do_send": 0x17A3620,
            "send_entry": 0x19D14C0,
        },
        "struct": dict(DEFAULT_STRUCT),
    },
}


class RebindError(RuntimeError):
    """发送链无法安全定位；调用方必须失败关闭。"""


@dataclass(frozen=True)
class WeixinLayout:
    sha256: str
    version: str | None
    offsets: dict[str, int]
    struct: dict[str, int]
    source: str
    evidence: dict[str, Any] = field(default_factory=dict)


class _PeImage:
    """极简 PE 解析：只取节表和 .pdata 函数边界，用于 RVA/文件偏移互转与拓扑定位。"""

    def __init__(self, data: bytes):
        self.data = data
        if len(data) < 0x40:
            raise RebindError("文件过小，不是有效的 PE 文件")
        e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
        if e_lfanew + 24 > len(data) or data[e_lfanew:e_lfanew + 4] != b"PE\0\0":
            raise RebindError("不是有效的 PE 文件")
        coff = e_lfanew + 4
        try:
            num_sections = struct.unpack_from("<H", data, coff + 2)[0]
            size_optional = struct.unpack_from("<H", data, coff + 16)[0]
        except struct.error as exc:
            raise RebindError("PE 头解析失败") from exc
        base = coff + 20 + size_optional
        self.sections: dict[str, tuple[int, int, int, int]] = {}
        try:
            for index in range(num_sections):
                off = base + index * 40
                name = data[off:off + 8].rstrip(b"\0").decode("ascii", "replace")
                vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, off + 8)
                self.sections[name] = (vaddr, vsize, rawptr, rawsize)
        except struct.error as exc:
            raise RebindError("PE 节表解析失败") from exc
        try:
            self._fn_bounds = self._parse_pdata()
            self._fn_starts = [begin for begin, _end in self._fn_bounds]
        except struct.error as exc:
            raise RebindError("pdata 解析失败") from exc

    def _parse_pdata(self) -> list[tuple[int, int]]:
        section = self.sections.get(".pdata")
        if section is None:
            return []
        _vaddr, _vsize, rawptr, rawsize = section
        bounds = []
        for index in range(rawsize // 12):
            begin, end, _unwind = struct.unpack_from("<III", self.data, rawptr + index * 12)
            bounds.append((begin, end))
        bounds.sort()
        return bounds

    def section_bytes(self, name: str) -> tuple[int, bytes]:
        vaddr, _vsize, rawptr, rawsize = self.sections[name]
        return vaddr, self.data[rawptr:rawptr + rawsize]

    def offset(self, rva: int) -> int:
        for vaddr, vsize, rawptr, rawsize in self.sections.values():
            if vaddr <= rva < vaddr + max(vsize, rawsize):
                return rawptr + (rva - vaddr)
        raise RebindError(f"RVA 0x{rva:x} 不在任何节内")

    def rva(self, file_offset: int) -> int:
        for vaddr, _vsize, rawptr, rawsize in self.sections.values():
            if rawptr <= file_offset < rawptr + rawsize:
                return vaddr + (file_offset - rawptr)
        raise RebindError(f"文件偏移 0x{file_offset:x} 不在任何节内")

    def bytes_at(self, rva: int, size: int) -> bytes:
        start = self.offset(rva)
        return self.data[start:start + size]

    def containing(self, rva: int) -> tuple[int, int] | None:
        index = bisect.bisect_right(self._fn_starts, rva) - 1
        if index < 0:
            return None
        begin, end = self._fn_bounds[index]
        return (begin, end) if begin <= rva < end else None


def _iter_call_targets(text: bytes, text_vaddr: int, lo: int, hi: int):
    """产出 ``(文件内偏移, 目标 RVA)``，仅扫描范围内 ``E8 rel32`` 直接调用。"""
    pos = lo
    hi = min(hi, len(text))
    while True:
        index = text.find(b"\xe8", pos, hi)
        if index < 0:
            return
        pos = index + 1
        if index + 5 > hi:
            return
        rel = struct.unpack_from("<i", text, index + 1)[0]
        yield index, (text_vaddr + index) + 5 + rel


def _sig_at(pe: _PeImage, rva: int, signature: bytes) -> bool:
    try:
        return pe.bytes_at(rva, len(signature)) == signature
    except RebindError:
        return False


def _locate_service(text: bytes, text_vaddr: int) -> int:
    hits = []
    pos = text.find(SIG_GET_SERVICE)
    while pos >= 0:
        hits.append(pos)
        pos = text.find(SIG_GET_SERVICE, pos + 1)
    if len(hits) != 1:
        raise RebindError(f"getService 唯一锚点命中 {len(hits)} 处，无法安全定位")
    return text_vaddr + hits[0]


def _closest_call(text: bytes, text_vaddr: int, lo: int, hi: int, predicate, *, reverse: bool):
    found = None
    for index, target in _iter_call_targets(text, text_vaddr, lo, hi):
        if predicate(target):
            found = (index, target)
            if not reverse:
                break
    return found


def _resolve_send_chain(pe: _PeImage):
    text_vaddr, text = pe.section_bytes(".text")
    text_len = len(text)
    service_rva = _locate_service(text, text_vaddr)

    service_calls = [
        (index, target)
        for index, target in _iter_call_targets(text, text_vaddr, 0, text_len)
        if target == service_rva
    ]
    if not service_calls:
        raise RebindError("没有任何函数调用 getService，拓扑回退失败")

    # sendEntry 指纹：prologue + `mov r9b,1` 旗标 + `lea r8,[rsi+8]` 消息向量。
    # 三个条件缺一不可，用于把文本发送链从同样调用 getCoro/getService 的
    # 兄弟发送链（图片/文件等）中区分出来。
    vector_lea = b"\x4c\x8d\x46\x08"
    flag_set = b"\x41\xb1\x01"
    tuples: dict[tuple[int, int, int, int], list[int]] = {}
    for index, _target in service_calls:
        call_rva = text_vaddr + index
        fn = pe.containing(call_rva)
        if fn is None:
            continue
        fn_start, fn_end = fn
        if pe.bytes_at(fn_start, len(SIG_SEND_ENTRY)) != SIG_SEND_ENTRY:
            continue
        # _iter_call_targets 的坐标是 .text 切片内位置，需要把 RVA 边界换成切片位置。
        fn_lo = fn_start - text_vaddr
        fn_hi = fn_end - text_vaddr

        coro = _closest_call(
            text, text_vaddr, fn_lo, index,
            lambda rva: _sig_at(pe, rva, SIG_GET_CORO), reverse=True,
        )
        if coro is None:
            continue
        context = _closest_call(
            text, text_vaddr, index + 5, min(fn_hi, index + 5 + 0x30),
            lambda rva: _sig_at(pe, rva, SIG_GET_CONTEXT), reverse=False,
        )
        if context is None:
            continue
        do_send = _closest_call(
            text, text_vaddr, context[0] + 5, min(fn_hi, context[0] + 5 + 0x30),
            lambda rva: _sig_at(pe, rva, SIG_DO_SEND), reverse=False,
        )
        if do_send is None:
            continue
        if text[do_send[0] - 3:do_send[0]] != flag_set:
            continue
        if vector_lea not in text[max(0, do_send[0] - 0x18):do_send[0]]:
            continue
        key = (coro[1], context[1], do_send[1], fn_start)
        tuples.setdefault(key, []).append(call_rva)

    if len(tuples) != 1:
        raise RebindError(f"发送链拓扑候选不唯一：{len(tuples)} 组，无法确认")
    (coro_rva, context_rva, do_send_rva, send_entry_rva), _sites = next(iter(tuples.items()))
    return {
        "get_coro": coro_rva,
        "get_service": service_rva,
        "get_context": context_rva,
        "do_send": do_send_rva,
        "send_entry": send_entry_rva,
    }


def _ctor_string_pair_at(text: bytes, index: int) -> int | None:
    """识别 ``msgCtor`` 的双 std::string 初始化，返回第一个字符串字段偏移。

    组合指纹：``lea rax,[rip+disp]; mov [rsi],rax; xorps xmm0,xmm0;`` 后再接两组
    ``movups [rsi+d],xmm0; mov qword [rsi+d+0x10],0; mov qword [rsi+d+0x18],0xf``。
    """
    if index < 13 or index + 69 > len(text) or text[index - 3:index] != b"\x0f\x57\xc0":
        return None
    if text[index - 6:index - 3] != b"\x48\x89\x06":
        return None
    if text[index - 13:index - 10] != b"\x48\x8d\x05":
        return None
    if text[index:index + 2] != b"\x0f\x11":
        return None
    modrm = text[index + 2]
    if modrm != 0x86:  # 与上方 vtable 写入相同的 rsi 对象 / xmm0
        return None
    rm = modrm & 7
    disp = struct.unpack_from("<i", text, index + 3)[0]
    cursor = index + 7
    for expected_delta, expected_value in ((0x10, 0), (0x18, 0xF)):
        if text[cursor:cursor + 2] != b"\x48\xc7" or text[cursor + 2] != (0x80 | rm):
            return None
        if struct.unpack_from("<i", text, cursor + 3)[0] != disp + expected_delta:
            return None
        if struct.unpack_from("<I", text, cursor + 7)[0] != expected_value:
            return None
        cursor += 11
    if text[cursor:cursor + 2] != b"\x0f\x11" or text[cursor + 2] != (0x80 | rm):
        return None
    if struct.unpack_from("<i", text, cursor + 3)[0] != disp + 0x20:
        return None
    cursor += 7
    for expected_delta, expected_value in ((0x30, 0), (0x38, 0xF)):
        if text[cursor:cursor + 3] != b"\x48\xc7\x86":
            return None
        if struct.unpack_from("<i", text, cursor + 3)[0] != disp + expected_delta:
            return None
        if struct.unpack_from("<I", text, cursor + 7)[0] != expected_value:
            return None
        cursor += 11
    # 文本消息恰好初始化两个字段后返回。4.1.15.50 的另一派生消息有相同
    # 前缀，但继续初始化额外字段；只匹配前缀会误将它当作文本构造器。
    if text[cursor:cursor + 11] != bytes.fromhex("4889f04883c4505f5e5dc3"):
        return None
    return disp


def _locate_message_ctor(pe: _PeImage) -> tuple[int, int]:
    text_vaddr, text = pe.section_bytes(".text")
    candidates: dict[int, int] = {}
    pos = 0
    while True:
        pos = text.find(b"\x0f\x11", pos)
        if pos < 0:
            break
        offset = _ctor_string_pair_at(text, pos)
        pos += 1
        if offset is None:
            continue
        pattern_rva = text_vaddr + pos - 1
        fn = pe.containing(pattern_rva)
        if fn is None or not _sig_at(pe, fn[0], SIG_MESSAGE_CTOR):
            continue
        candidates[fn[0]] = offset
    if len(candidates) != 1:
        listed = ", ".join(f"0x{rva:x}" for rva in sorted(candidates))
        raise RebindError(
            f"message 构造器模式命中 {len(candidates)} 个同类函数，无法自动确认：{listed}。"
            "需人工在 KNOWN_LAYOUTS 增补该版本 pins，L2 不做猜测。"
        )
    fn_start, content_offset = next(iter(candidates.items()))
    return fn_start, content_offset


def rebind(image_bytes: bytes, sha: str | None = None) -> WeixinLayout:
    """L1/L2：不依赖历史 DLL，直接从当前镜像重建布局。"""
    digest = (sha or sha256(image_bytes).hexdigest()).upper()
    pe = _PeImage(image_bytes)
    offsets = _resolve_send_chain(pe)
    message_ctor, content_offset = _locate_message_ctor(pe)
    offsets["message_ctor"] = message_ctor

    if not 0 < content_offset < 0x1000 or content_offset % 8 != 0:
        raise RebindError(f"推导出的正文字段偏移异常：0x{content_offset:x}")
    struct_layout = dict(DEFAULT_STRUCT)
    struct_layout["content"] = content_offset

    _validate(pe, offsets)
    return WeixinLayout(
        sha256=digest,
        version=None,
        offsets=offsets,
        struct=struct_layout,
        source="rebound",
        evidence={"content_offset": hex(content_offset)},
    )


def _validate(pe: _PeImage, offsets: dict[str, int]) -> None:
    expected = {
        "get_coro": SIG_GET_CORO,
        "get_service": SIG_GET_SERVICE,
        "get_context": SIG_GET_CONTEXT,
        "message_ctor": SIG_MESSAGE_CTOR,
        "do_send": SIG_DO_SEND,
        "send_entry": SIG_SEND_ENTRY,
    }
    for name, signature in expected.items():
        if not _sig_at(pe, offsets[name], signature):
            raise RebindError(f"{name} 前导签名校验失败：0x{offsets[name]:x}")


def resolve(image_bytes: bytes, sha: str | None = None) -> WeixinLayout:
    """L0 命中不可变版本表，否则进入 L1/L2 有界回退。"""
    digest = (sha or sha256(image_bytes).hexdigest()).upper()
    known = KNOWN_LAYOUTS.get(digest)
    if known is not None:
        return WeixinLayout(
            sha256=digest,
            version=known["version"],
            offsets=dict(known["offsets"]),
            struct=dict(known["struct"]),
            source="pinned",
        )
    return rebind(image_bytes, digest)


def resolve_file(path: str | Path, sha: str | None = None) -> WeixinLayout:
    return resolve(Path(path).read_bytes(), sha)
