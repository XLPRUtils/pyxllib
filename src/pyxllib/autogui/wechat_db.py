#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Read decrypted WeChat 4.x ``db_storage`` databases.

This module is intentionally about local database structure, not GUI capture.
It expects databases that can already be opened by SQLite.  Decryption helpers
can prepare such a directory, but ordinary browsing should use read-only
connections to a decrypted snapshot.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import html
import json
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import hmac
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PAGE_SIZE = 80
MAX_PAGE_SIZE = 500
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
SQLITE_HEADER = b"SQLite format 3\0"
WX_DB_PAGE_SIZE = 4096
WX_DB_SALT_SIZE = 16
WX_DB_IV_SIZE = 16
WX_DB_HMAC_SIZE = 64
WX_DB_KEY_SIZE = 32
WX_DB_AES_BLOCK_SIZE = 16
WX_DB_ROUND_COUNT = 256000
WX_IMAGE_V4_AES_KEYS = {
    b"\x07\x08V1\x08\x07": b"cfcd208495d565ef",
    b"\x07\x08V2\x08\x07": None,  # V2 is account-specific, never a universal fixed key.
}
_WECHAT_IMAGE_AES_KEY_CACHE: dict[str, bytes | None] = {}
_WECHAT_IMAGE_KEY_RETRY_AT: dict[str, float] = {}
_WECHAT_EXPORTED_RESOURCE_CACHE: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}
_WECHAT_EXPORTED_RESOURCE_CACHE_TTL = 300.0


class WeChatDbError(RuntimeError):
    """Raised when a WeChat database snapshot cannot be read."""


def _connect_readonly(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<blob:{len(value)}>"
    return value


def _jsonable_row(row: sqlite3.Row) -> dict[str, Any]:
    return {key: _jsonable_value(row[key]) for key in row.keys()}


def _decode_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, bytes):
        return str(value)
    data = value
    if data.startswith(ZSTD_MAGIC):
        try:
            import zstandard as zstd

            data = zstd.ZstdDecompressor().decompress(data, max_output_size=8 * 1024 * 1024)
        except Exception:
            return ""
    for encoding in ("utf-8", "gb18030"):
        try:
            text = data.decode(encoding).strip("\x00\r\n\t ")
        except UnicodeDecodeError:
            continue
        if text:
            return text
    return ""


def _image_type_from_header(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff") and len(data) >= 4 and data[3] in (*range(0xE0, 0xF0), 0xDB, 0xC0, 0xC2, 0xC4, 0xFE):
        return "jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if (len(data) >= 14 and data.startswith(b"BM") and data[6:10] == b"\0" * 4
            and 26 <= int.from_bytes(data[10:14], "little") <= int.from_bytes(data[2:6], "little")):
        return "bmp"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return ""


def _wechat_v4_image_aes_key(header: bytes) -> bytes | None:
    return WX_IMAGE_V4_AES_KEYS.get(header[:6])


def _detect_image_format(data: bytes) -> str:
    return _image_type_from_header(data) or ("wxgf" if data.startswith(b"wxgf") else "bin")


def _readable_image(path: Path) -> bool:
    """A signature alone can match a wrong AES key; validate the entire image."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def _convert_wxgf(data: bytes) -> bytes | None:
    """Extract length-prefixed Annex B HEVC and render a still as PNG.

    WXGF's header length is byte 4; each stream begins after a big-endian
    32-bit length. Prefer the largest stream (the colour image). Keep unknown
    containers unreadable instead of passing encrypted bytes to a viewer.
    Format reference: sjzar/chatlog a16b689/pkg/util/dat2img/wxgf.go.
    """
    if len(data) < 15 or data[:4] != b"wxgf" or not 5 <= data[4] < len(data):
        return None
    streams = []
    for marker in (b"\0\0\0\1", b"\0\0\1"):
        offset = data[4]
        while offset < len(data):
            start = data.find(marker, offset)
            if start < 0:
                break
            size = int.from_bytes(data[start - 4:start], "big") if start >= 4 else 0
            if size > 0 and start + size <= len(data):
                streams.append(data[start:start + size])
                offset = start + size
            else:
                offset = start + 1
        if streams:
            break
    if not streams:
        return None
    executable = os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg")
    if not executable:
        return None
    result = subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-xerror",
                             "-f", "hevc", "-i", "pipe:0", "-frames:v", "1",
                             "-f", "image2pipe", "-c:v", "png", "pipe:1"],
                            input=max(streams, key=len), capture_output=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return result.stdout if result.returncode == 0 and result.stdout.startswith(b"\x89PNG") else None


def _verify_wechat_v4_image_aes_key(aes_key: bytes, templates: list[bytes]) -> bool:
    if len(aes_key) != 16 or not templates:
        return False
    try:
        from Crypto.Cipher import AES

        cipher = AES.new(aes_key, AES.MODE_ECB)
        return all(_detect_image_format(cipher.decrypt(template)) != "bin" for template in templates)
    except Exception:
        return False


def _find_wechat_v4_image_templates(attach_dir: Path, max_templates: int = 3, max_files: int = 64) -> list[bytes]:
    if not attach_dir.exists():
        return []
    templates: list[bytes] = []
    seen: set[bytes] = set()
    examined = 0
    for suffix in ("*_t.dat", "*.dat"):
        for source in sorted(attach_dir.rglob(suffix)):
            examined += 1
            if examined > max_files and templates:
                return templates
            try:
                with source.open("rb") as stream:
                    data = stream.read(0x1F)
            except OSError:
                continue
            if len(data) >= 0x1F and data.startswith(b"\x07\x08V2\x08\x07"):
                template = data[0x0F:0x1F]
                if template not in seen:
                    templates.append(template)
                    seen.add(template)
                    if len(templates) >= max_templates:
                        return templates
        if templates:
            return templates
    return templates


def _scan_windows_weixin_image_aes_key(templates: list[bytes], *, pid: int | None = None) -> bytes | None:
    if os.name != "nt" or not templates:
        return None
    cache_key = str(pid) + "|" + "|".join(template.hex() for template in templates)
    if _WECHAT_IMAGE_AES_KEY_CACHE.get(cache_key):
        return _WECHAT_IMAGE_AES_KEY_CACHE[cache_key]
    env_key = (os.environ.get("CODEYUN_WECHAT_IMAGE_AES_KEY") or "").strip()
    if env_key:
        candidates = [env_key.encode("ascii", errors="ignore")]
        try:
            candidates.append(bytes.fromhex(env_key))
        except ValueError:
            pass
        for candidate in candidates:
            if _verify_wechat_v4_image_aes_key(candidate[:16], templates):
                _WECHAT_IMAGE_AES_KEY_CACHE[cache_key] = candidate[:16]
                return candidate[:16]

    if pid is not None:
        pids = [pid]
    else:
        try:
            output = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='Weixin.exe'\" | Select-Object -ExpandProperty ProcessId"],
                text=True, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except Exception:
            return None
        pids = [int(part) for part in output.split() if part.isdigit()]
    if not pids:
        return None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process_query_information = 0x0400
    process_vm_read = 0x0010
    mem_commit = 0x1000
    page_noaccess = 0x01
    page_guard = 0x100
    readable_pages = {0x04, 0x08, 0x40, 0x80}
    max_region_size = 50 * 1024 * 1024
    chunk_size = 2 * 1024 * 1024

    class MemoryBasicInformation64(ctypes.Structure):
        _fields_ = [
            ("BaseAddress", ctypes.c_ulonglong),
            ("AllocationBase", ctypes.c_ulonglong),
            ("AllocationProtect", ctypes.c_ulong),
            ("__alignment1", ctypes.c_ulong),
            ("RegionSize", ctypes.c_ulonglong),
            ("State", ctypes.c_ulong),
            ("Protect", ctypes.c_ulong),
            ("Type", ctypes.c_ulong),
            ("__alignment2", ctypes.c_ulong),
        ]

    open_process = kernel32.OpenProcess
    open_process.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    open_process.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    virtual_query_ex = kernel32.VirtualQueryEx
    virtual_query_ex.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(MemoryBasicInformation64),
        ctypes.c_size_t,
    ]
    virtual_query_ex.restype = ctypes.c_size_t
    read_process_memory = kernel32.ReadProcessMemory
    read_process_memory.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    read_process_memory.restype = ctypes.c_int

    def is_readable_page(protect: int) -> bool:
        if protect == page_noaccess or (protect & page_guard):
            return False
        base = protect & ~(page_guard | 0x200 | 0x400)
        return base in readable_pages

    patterns = [re.compile(rb"(?<![A-Za-z0-9])([A-Za-z0-9]{32})(?![A-Za-z0-9])"), re.compile(rb"(?<![A-Za-z0-9])([A-Za-z0-9]{16})(?![A-Za-z0-9])")]
    for pid in pids:
        handle = open_process(process_query_information | process_vm_read, 0, pid)
        if not handle:
            continue
        seen: set[bytes] = set()
        address = 0
        try:
            while address < 0x7FFFFFFFFFFF:
                mbi = MemoryBasicInformation64()
                if not virtual_query_ex(handle, ctypes.c_void_p(address), ctypes.byref(mbi), ctypes.sizeof(mbi)):
                    break
                base = int(mbi.BaseAddress)
                size = int(mbi.RegionSize)
                if int(mbi.State) == mem_commit and is_readable_page(int(mbi.Protect)) and size <= max_region_size:
                    offset = 0
                    while offset < size:
                        n = min(chunk_size, size - offset)
                        buffer = (ctypes.c_ubyte * n)()
                        bytes_read = ctypes.c_size_t(0)
                        ok = read_process_memory(
                            handle,
                            ctypes.c_void_p(base + offset),
                            buffer,
                            n,
                            ctypes.byref(bytes_read),
                        )
                        if ok and bytes_read.value:
                            chunk = bytes(buffer[: bytes_read.value])
                            for pattern in patterns:
                                for match in pattern.finditer(chunk):
                                    candidate = match.group(1)[:16]
                                    if candidate in seen:
                                        continue
                                    seen.add(candidate)
                                    if _verify_wechat_v4_image_aes_key(candidate, templates):
                                        _WECHAT_IMAGE_AES_KEY_CACHE[cache_key] = candidate
                                        return candidate
                        offset += n - 31 if n > 31 else n
                next_address = base + size
                if next_address <= address:
                    break
                address = next_address
        finally:
            close_handle(handle)
    _WECHAT_IMAGE_AES_KEY_CACHE[cache_key] = None
    return None


def _decode_wechat_v4_image_dat(source: Path, target_dir: Path, stem: str, xor_key: int, aes_key: bytes | None = None) -> Path | None:
    try:
        with source.open("rb") as f:
            header = f.read(0xF)
            if header[:6] not in WX_IMAGE_V4_AES_KEYS:
                data = header + f.read()
                candidates = [data]
                # Legacy .dat uses one XOR byte for the entire file. Infer it
                # from a recognized signature, then validate the whole image.
                for key in range(256):
                    if _image_type_from_header(bytes(value ^ key for value in data[:16])):
                        candidates.append(bytes(value ^ key for value in data))
                from io import BytesIO
                from PIL import Image
                for plain in candidates:
                    kind = _image_type_from_header(plain[:16])
                    if not kind:
                        continue
                    try:
                        with Image.open(BytesIO(plain)) as image:
                            image.verify()
                        with Image.open(BytesIO(plain)) as image:
                            image.load()
                    except (OSError, ValueError, SyntaxError):
                        continue
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target = target_dir / f"{stem}.{kind}"
                    temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
                    temporary.write_bytes(plain)
                    temporary.replace(target)
                    return target
                return None
            aes_key = aes_key or _wechat_v4_image_aes_key(header)
            if not aes_key:
                return None
            encrypt_length, xor_length = struct.unpack_from("<II", header, 6)
            encrypt_length0 = encrypt_length // 16 * 16 + 16
            encrypted_data = f.read(encrypt_length0)
            rest_data = f.read()
        if not encrypted_data:
            return None
        if len(encrypted_data) != encrypt_length0 or xor_length > len(rest_data):
            return None
        from Crypto.Cipher import AES

        decrypted_data = AES.new(aes_key, AES.MODE_ECB).decrypt(encrypted_data)
        image_type = _detect_image_format(decrypted_data[:16])
        if image_type == "bin":
            return None
        pad_length = decrypted_data[-1]
        if not (1 <= pad_length <= 16 and decrypted_data[-pad_length:] == bytes([pad_length]) * pad_length):
            return None
        decrypted_data = decrypted_data[:-pad_length]
        plain_data = decrypted_data + (rest_data[:-xor_length] + bytes(byte ^ xor_key for byte in rest_data[-xor_length:]) if xor_length else rest_data)
        if _detect_image_format(plain_data[:16]) != image_type:
            return None
        if image_type == "wxgf":
            converted = _convert_wxgf(plain_data)
            if converted:
                plain_data, image_type = converted, "png"
        if image_type != "wxgf":
            from PIL import Image
            from io import BytesIO
            with Image.open(BytesIO(plain_data)) as image:
                image.verify()
            with Image.open(BytesIO(plain_data)) as image:
                image.load()
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{stem}.{image_type}"
        temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(plain_data)
        temporary.replace(target)
        return target
    except Exception:
        return None


def _extract_xml_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}\b(?![^>]*\/>)[^>]*>([\s\S]*?)</{tag}>", text, re.IGNORECASE)
    if not match:
        return ""
    value = re.sub(r"<!\[CDATA\[|\]\]>", "", match.group(1))
    return html.unescape(value.strip())


def _extract_xml_block(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}\b(?![^>]*\/>)[^>]*>([\s\S]*?)</{tag}>", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _strip_xml_sender_prefix(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("<?xml") or stripped.startswith("<msg"):
        return stripped
    match = re.search(r"<(?:\?xml|msg)\b", stripped, re.IGNORECASE)
    if match:
        return stripped[match.start() :]
    return stripped


def _parse_appmsg(text: str) -> dict[str, Any] | None:
    xml_text = _strip_xml_sender_prefix(text)
    if "<appmsg" not in xml_text.lower():
        return None
    refer_block = _extract_xml_block(xml_text, "refermsg")
    appmsg_text = re.sub(r"<refermsg\b(?![^>]*\/>)[^>]*>[\s\S]*?</refermsg>", "", xml_text, flags=re.IGNORECASE)
    title = _extract_xml_tag(appmsg_text, "title")
    description = _extract_xml_tag(appmsg_text, "des")
    url = _extract_xml_tag(appmsg_text, "url")
    app_type = _extract_xml_tag(appmsg_text, "type")
    total_size = _extract_xml_tag(appmsg_text, "totallen")
    file_ext = _extract_xml_tag(appmsg_text, "fileext")
    md5 = _extract_xml_tag(appmsg_text, "md5")
    thumb_url = _extract_xml_tag(appmsg_text, "thumburl") or _extract_xml_tag(appmsg_text, "cdnthumburl")
    forwarded_items = _parse_forwarded_items(appmsg_text)
    refer = None
    if refer_block:
        refer = {
            "content": _extract_xml_tag(refer_block, "content"),
            "display_name": _extract_xml_tag(refer_block, "displayname"),
            "from_user": _extract_xml_tag(refer_block, "fromusr"),
            "chat_user": _extract_xml_tag(refer_block, "chatusr"),
            "type": int(_extract_xml_tag(refer_block, "type") or 0) or None,
            "create_time": int(_extract_xml_tag(refer_block, "createtime") or 0) or None,
        }
        refer = {key: value for key, value in refer.items() if value not in ("", None)}
    item: dict[str, Any] = {
        "title": title,
        "description": description,
        "url": url,
        "app_type": int(app_type) if app_type.isdigit() else None,
        "file_ext": file_ext,
        "total_size": int(total_size) if total_size.isdigit() else None,
        "md5": md5,
        "thumb_url": thumb_url,
        "refer": refer,
        "forwarded_items": forwarded_items,
    }
    return {key: value for key, value in item.items() if value not in ("", None)}


def _parse_forwarded_items(text: str) -> list[dict[str, Any]]:
    record_xml = html.unescape(_extract_xml_tag(text, "recorditem")).strip()
    if not record_xml:
        return []
    try:
        root = ET.fromstring(record_xml)
    except ET.ParseError:
        return []

    items: list[dict[str, Any]] = []
    for data_index, node in enumerate(root.findall(".//dataitem")):
        datatype_text = str(node.get("datatype") or "").strip()
        datatype = int(datatype_text) if datatype_text.isdigit() else None

        def node_text(tag: str) -> str:
            return str(node.findtext(tag) or "").strip()

        def node_int(tag: str) -> int | None:
            value = node_text(tag)
            return int(value) if value.isdigit() and int(value) > 0 else None

        item = {
            "data_index": data_index,
            "data_id": str(node.get("dataid") or "").strip(),
            "datatype": datatype,
            "speaker": node_text("sourcename"),
            "source_time": node_text("sourcetime"),
            "text": node_text("datadesc"),
            "data_format": node_text("datafmt"),
            "data_size": node_int("datasize"),
            "full_md5": node_text("fullmd5"),
            "thumb_size": node_int("thumbsize"),
            "thumb_md5": node_text("thumbfullmd5"),
            "cdn_data_url": node_text("cdndataurl"),
            "cdn_data_key": node_text("cdndatakey"),
            "cdn_thumb_url": node_text("cdnthumburl"),
            "cdn_thumb_key": node_text("cdnthumbkey"),
        }
        items.append({key: value for key, value in item.items() if value not in ("", None)})
    return items


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def message_table_name(username: str) -> str:
    return "Msg_" + hashlib.md5(username.encode("utf-8")).hexdigest()


def _safe_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def normalize_message_type(value: int | None) -> int:
    raw = int(value or 0)
    if raw > 0xFFFFFFFF and (raw & 0xFFFFFFFF) < 100000:
        return raw & 0xFFFFFFFF
    if raw > 0xFFFF and (raw & 0xFFFF) in {1, 3, 34, 37, 42, 43, 47, 48, 49, 50, 51, 10000, 10002}:
        return raw & 0xFFFF
    return raw


def _wx_db_reserve_size() -> int:
    reserve = WX_DB_IV_SIZE + WX_DB_HMAC_SIZE
    return ((reserve + WX_DB_AES_BLOCK_SIZE - 1) // WX_DB_AES_BLOCK_SIZE) * WX_DB_AES_BLOCK_SIZE


def _derive_wx_db_key(key_hex: str, mode: str, salt: bytes) -> bytes:
    from Crypto.Hash import SHA512
    from Crypto.Protocol.KDF import PBKDF2

    key = bytes.fromhex(key_hex)
    if mode == "raw-derived-key":
        return key
    if mode == "passphrase-pbkdf2":
        return PBKDF2(key, salt, dkLen=WX_DB_KEY_SIZE, count=WX_DB_ROUND_COUNT, hmac_hash_module=SHA512)
    raise WeChatDbError(f"不支持的数据库 key 模式：{mode}")


def decrypt_wechat_v4_db(in_path: Path, out_path: Path, key_hex: str, mode: str) -> bool:
    from Crypto.Cipher import AES
    from Crypto.Hash import SHA512
    from Crypto.Protocol.KDF import PBKDF2

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page_count = 0
    reserve = _wx_db_reserve_size()
    with in_path.open("rb") as f_in, tmp_path.open("wb") as f_out:
        salt = f_in.read(WX_DB_SALT_SIZE)
        if len(salt) != WX_DB_SALT_SIZE:
            return False
        key = _derive_wx_db_key(key_hex, mode, salt)
        mac_salt = bytes(x ^ 0x3A for x in salt)
        mac_key = PBKDF2(key, mac_salt, dkLen=WX_DB_KEY_SIZE, count=2, hmac_hash_module=SHA512)
        f_out.write(SQLITE_HEADER)
        while True:
            if page_count == 0:
                rest = f_in.read(WX_DB_PAGE_SIZE - WX_DB_SALT_SIZE)
                if not rest:
                    break
                page = salt + rest
                offset = WX_DB_SALT_SIZE
            else:
                page = f_in.read(WX_DB_PAGE_SIZE)
                if not page:
                    break
                offset = 0
            if len(page) != WX_DB_PAGE_SIZE:
                return False
            mac = hmac.new(mac_key, page[offset : WX_DB_PAGE_SIZE - reserve + WX_DB_IV_SIZE], SHA512)
            mac.update(struct.pack("<I", page_count + 1))
            expected = page[
                WX_DB_PAGE_SIZE - reserve + WX_DB_IV_SIZE : WX_DB_PAGE_SIZE - reserve + WX_DB_IV_SIZE + WX_DB_HMAC_SIZE
            ]
            if not hmac.compare_digest(mac.digest(), expected):
                return False
            iv = page[WX_DB_PAGE_SIZE - reserve : WX_DB_PAGE_SIZE - reserve + WX_DB_IV_SIZE]
            plain = AES.new(key, AES.MODE_CBC, iv).decrypt(page[offset : WX_DB_PAGE_SIZE - reserve])
            f_out.write(plain)
            f_out.write(page[WX_DB_PAGE_SIZE - reserve :])
            page_count += 1
    if page_count:
        # A live database can keep its newest committed pages entirely in WAL.
        # Publish one atomic snapshot containing only authenticated, committed frames.
        from pyxllib.autogui.wechat_updates import apply_committed_wal
        apply_committed_wal(in_path, tmp_path, key_hex, mode)
        tmp_path.replace(out_path)
        return True
    tmp_path.unlink(missing_ok=True)
    return False


def _display_name(username: str, contacts: dict[str, dict[str, Any]]) -> str:
    contact = contacts.get(username) or {}
    return contact.get("remark") or contact.get("nick_name") or contact.get("alias") or username


@dataclass(frozen=True)
class WeChatDbPaths:
    root: Path
    contact: Path
    session: Path
    message: Path
    biz_message: Path
    media: Path
    resource: Path
    hardlink: Path
    head_image: Path

    @classmethod
    def from_root(cls, root: os.PathLike[str] | str) -> "WeChatDbPaths":
        root_path = Path(root)
        return cls(
            root=root_path,
            contact=root_path / "contact" / "contact.db",
            session=root_path / "session" / "session.db",
            message=root_path / "message" / "message_0.db",
            biz_message=root_path / "message" / "biz_message_0.db",
            media=root_path / "message" / "media_0.db",
            resource=root_path / "message" / "message_resource.db",
            hardlink=root_path / "hardlink" / "hardlink.db",
            head_image=root_path / "head_image" / "head_image.db",
        )


class WeChatDbStorage:
    """Query a decrypted WeChat 4.x ``db_storage`` directory."""

    def __init__(self, root: os.PathLike[str] | str):
        self.paths = WeChatDbPaths.from_root(root)
        self._image_xor_key_cache: int | None = None
        self._image_aes_key_cache: bytes | None = None
        self._exported_resource_files_cache: dict[str, dict[str, Any]] | None = None

    @property
    def root(self) -> Path:
        return self.paths.root

    def _sync_state_path(self) -> Path:
        return self.root.parent / "sync_state.json"

    def _load_sync_state(self) -> dict[str, Any]:
        path = self._sync_state_path()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _save_sync_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = int(time.time())
        path = self._sync_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _update_sync_state(self, **updates: Any) -> None:
        state = self._load_sync_state()
        state.update(updates)
        self._save_sync_state(state)

    def _file_fingerprint(self, path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _same_fingerprint(self, left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
        if not left or not right:
            return False
        return left.get("size") == right.get("size") and left.get("mtime_ns") == right.get("mtime_ns")

    def _valid_wechat_account_root(self, path: Path | None) -> Path | None:
        if not path:
            return None
        try:
            candidate = path.expanduser()
        except RuntimeError:
            return None
        if (candidate / "msg").exists() and (candidate / "db_storage").exists():
            return candidate
        return None

    def status(self) -> dict[str, Any]:
        dbs = {
            "contact": self.paths.contact,
            "session": self.paths.session,
            "message": self.paths.message,
            "biz_message": self.paths.biz_message,
            "media": self.paths.media,
            "resource": self.paths.resource,
            "hardlink": self.paths.hardlink,
            "head_image": self.paths.head_image,
        }
        exists = {name: path.exists() for name, path in dbs.items()}
        return {
            "db_storage_path": os.fspath(self.root),
            "live_account_root": self._load_sync_state().get("live_account_root"),
            "exists": self.root.exists(),
            "databases": exists,
            "ready": exists["session"] and exists["message"],
        }

    def _contact_map(self, *, include_avatar: bool = False) -> dict[str, dict[str, Any]]:
        if not self.paths.contact.exists():
            return {}
        conn = _connect_readonly(self.paths.contact)
        try:
            if not _table_exists(conn, "contact"):
                return {}
            rows = conn.execute(
                """
                SELECT username, remark, nick_name, alias, local_type, flag, chat_room_type
                FROM contact
                """
            ).fetchall()
            contacts = {row["username"]: dict(row) for row in rows if row["username"]}
            if include_avatar:
                avatar_map = self._avatar_data_urls(set(contacts))
                for username, contact in contacts.items():
                    contact["avatar_data_url"] = avatar_map.get(username)
            return contacts
        finally:
            conn.close()

    def _avatar_data_urls(self, usernames: set[str] | None = None) -> dict[str, str]:
        if not self.paths.head_image.exists():
            return {}
        conn = _connect_readonly(self.paths.head_image)
        try:
            if not _table_exists(conn, "head_image"):
                return {}
            params: list[Any] = []
            where_sql = ""
            if usernames:
                placeholders = ",".join("?" for _ in usernames)
                where_sql = f" WHERE username IN ({placeholders})"
                params = sorted(usernames)
            rows = conn.execute(f"SELECT username, image_buffer FROM head_image{where_sql}", params).fetchall()
            avatars: dict[str, str] = {}
            for row in rows:
                data = row["image_buffer"]
                if isinstance(data, bytes) and data:
                    avatars[row["username"]] = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
            return avatars
        finally:
            conn.close()

    def _message_conn(self, source: str = "message") -> sqlite3.Connection:
        if source == "biz":
            return _connect_readonly(self.paths.biz_message)
        return _connect_readonly(self.paths.message)

    def _database_path(self, database: str) -> Path:
        mapping = {
            "contact": self.paths.contact,
            "session": self.paths.session,
            "message": self.paths.message,
            "biz_message": self.paths.biz_message,
            "media": self.paths.media,
            "resource": self.paths.resource,
            "hardlink": self.paths.hardlink,
            "head_image": self.paths.head_image,
        }
        try:
            return mapping[database]
        except KeyError as exc:
            raise WeChatDbError(f"未知数据库：{database}") from exc

    def _session_map(self) -> dict[str, dict[str, Any]]:
        if not self.paths.session.exists():
            return {}
        conn = _connect_readonly(self.paths.session)
        try:
            if not _table_exists(conn, "SessionTable"):
                return {}
            return {
                row["username"]: dict(row)
                for row in conn.execute(
                    """
                    SELECT username, type, unread_count, summary, last_timestamp, sort_timestamp,
                           last_msg_locald_id, last_msg_type, last_msg_sender, last_sender_display_name
                    FROM SessionTable
                    """
                )
                if row["username"]
            }
        finally:
            conn.close()

    def _wechat_account_root(self) -> Path | None:
        candidates: list[Path] = []
        state = self._load_sync_state()
        cached_account_root = self._valid_wechat_account_root(Path(str(state.get("live_account_root")))) if state.get("live_account_root") else None
        if cached_account_root:
            return cached_account_root
        if state.get("live_account_root"):
            raise WeChatDbError("已绑定的微信账号目录不可用，拒绝自动切换账号")
        if self.root.name == "db_storage":
            candidates.append(self.root.parent)
        env_path = (os.environ.get("CODEYUN_WECHAT_ACCOUNT_ROOT") or "").strip()
        if env_path:
            candidates.append(Path(env_path).expanduser())
        secret_path = self.root.parent.parent / "secrets" / "wechat_v4_key.json"
        if secret_path.exists():
            try:
                payload = json.loads(secret_path.read_text(encoding="utf-8"))
                for item in payload.get("candidates") or []:
                    wx_dir = str(item.get("wx_dir") or "").strip()
                    if wx_dir:
                        candidates.append(Path(wx_dir))
            except Exception:
                pass
        if self.paths.hardlink.exists():
            conn = _connect_readonly(self.paths.hardlink)
            try:
                if _table_exists(conn, "db_info"):
                    row = conn.execute("SELECT ValueStdStr FROM db_info WHERE Key='uuid'").fetchone()
                    text = str(row["ValueStdStr"] or "") if row else ""
                    if "xwechat_files" in text:
                        match = re.search(r"[A-Za-z]:\\.*?xwechat_files", text)
                        if match:
                            candidates.extend(Path(match.group(0)).glob("wxid_*"))
            finally:
                conn.close()
        valid_candidates = {}
        for candidate in candidates:
            valid = self._valid_wechat_account_root(candidate)
            if valid:
                valid_candidates[os.path.normcase(os.fspath(valid.resolve()))] = valid
        if len(valid_candidates) > 1:
            raise WeChatDbError("发现多个微信账号目录，请显式绑定独立的账号存储")
        if valid_candidates:
            valid = next(iter(valid_candidates.values()))
            self._update_sync_state(live_account_root=os.fspath(valid))
            return valid
        return None

    def _raw_snapshot_db_storage(self, live_account_root: Path) -> Path:
        data_root = self.root.parent.parent
        return data_root / "raw_snapshot" / "xwechat_files" / live_account_root.name / "db_storage"

    def _db_key_matches(self) -> dict[str, dict[str, Any]]:
        secret = self.root.parent.parent / "secrets" / "wechat_v4_db_keys.json"
        if not secret.exists():
            return {}
        payload = json.loads(secret.read_text(encoding="utf-8"))
        return payload.get("matches") or {}

    def _copy_live_db_storage(self, live_account_root: Path) -> dict[str, Any]:
        source_root = live_account_root / "db_storage"
        if not source_root.exists():
            raise WeChatDbError(f"本机微信 db_storage 不存在：{source_root}")
        target_root = self._raw_snapshot_db_storage(live_account_root)
        state = self._load_sync_state()
        live_db_files = dict(state.get("live_db_files") or {})
        copied = 0
        unchanged = 0
        removed = 0
        errors: list[str] = []
        # Treat DB and its WAL as one source generation. A checkpoint during
        # copying must not publish a mixture of two generations.
        before = {p.relative_to(source_root).as_posix(): self._file_fingerprint(p)
                  for p in source_root.rglob("*") if p.is_file()}
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            rel = source.relative_to(source_root)
            rel_key = rel.as_posix()
            target = target_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                source_fingerprint = self._file_fingerprint(source)
                target_fingerprint = self._file_fingerprint(target) if target.exists() else None
                cached = live_db_files.get(rel_key)
                if self._same_fingerprint(source_fingerprint, target_fingerprint):
                    if not self._same_fingerprint(source_fingerprint, cached):
                        live_db_files[rel_key] = {
                            **source_fingerprint,
                            "source": os.fspath(source),
                            "target": os.fspath(target),
                        }
                    unchanged += 1
                    continue
                tmp = target.with_suffix(target.suffix + ".copying")
                tmp.unlink(missing_ok=True)
                shutil.copy2(source, tmp)
                if not self._same_fingerprint(source_fingerprint, self._file_fingerprint(source)):
                    tmp.unlink(missing_ok=True)
                    raise WeChatDbError(f"同步期间文件变化，下一轮重试：{rel}")
                tmp.replace(target)
                live_db_files[rel_key] = {
                    **source_fingerprint,
                    "source": os.fspath(source),
                    "target": os.fspath(target),
                }
                copied += 1
            except Exception as exc:
                errors.append(f"{rel}: {type(exc).__name__}: {exc}")
        after = {p.relative_to(source_root).as_posix(): self._file_fingerprint(p)
                 for p in source_root.rglob("*") if p.is_file()}
        if before != after:
            errors.append("Source generation changed while copying; retry before publishing")
        if target_root.exists():
            source_rels = {path.relative_to(source_root).as_posix() for path in source_root.rglob("*") if path.is_file()}
            for rel_key in list(live_db_files):
                if rel_key not in source_rels:
                    live_db_files.pop(rel_key, None)
                    removed += 1
        state["live_account_root"] = os.fspath(live_account_root)
        state["live_db_storage"] = os.fspath(source_root)
        state["raw_snapshot_db_storage"] = os.fspath(target_root)
        state["live_db_files"] = live_db_files
        self._save_sync_state(state)
        return {
            "source": os.fspath(source_root),
            "target": os.fspath(target_root),
            "copied": copied,
            "unchanged": unchanged,
            "removed": removed,
            "errors": errors[:20],
            "error_count": len(errors),
        }

    def _decrypt_snapshot_dbs(self, live_account_root: Path) -> dict[str, Any]:
        source_root = self._raw_snapshot_db_storage(live_account_root)
        matches = self._db_key_matches()
        state = self._load_sync_state()
        decrypted_dbs = dict(state.get("decrypted_dbs") or {})
        decrypted = 0
        wal_only_updates = []
        unchanged = 0
        skipped = 0
        failed: list[str] = []
        for source in sorted(source_root.rglob("*.db")):
            rel = source.relative_to(source_root)
            rel_key = rel.as_posix()
            key_info = matches.get(str(rel)) or matches.get(rel_key) or matches.get(rel_key.replace("/", "\\"))
            if not key_info:
                skipped += 1
                continue
            target = self.root / rel
            try:
                source_fingerprint = self._file_fingerprint(source)
                wal_path = Path(str(source) + "-wal")
                wal_fingerprint = self._file_fingerprint(wal_path) if wal_path.exists() else None
                mode = key_info.get("mode") or "raw-derived-key"
                previous = decrypted_dbs.get(rel_key)
                if (
                    target.exists()
                    and previous
                    and self._same_fingerprint(source_fingerprint, previous.get("source_fingerprint"))
                    and previous.get("key_hex") == key_info["key_hex"]
                    and previous.get("mode") == mode
                    and previous.get("wal_fingerprint") == wal_fingerprint
                ):
                    unchanged += 1
                    continue
                ok = decrypt_wechat_v4_db(source, target, key_info["key_hex"], mode)
                if ok:
                    if previous and self._same_fingerprint(source_fingerprint, previous.get("source_fingerprint")) and previous.get("wal_fingerprint") != wal_fingerprint:
                        wal_only_updates.append(rel_key)
                    decrypted_dbs[rel_key] = {
                        "source": os.fspath(source),
                        "target": os.fspath(target),
                        "source_fingerprint": source_fingerprint,
                        "wal_fingerprint": wal_fingerprint,
                        "target_fingerprint": self._file_fingerprint(target),
                        "key_hex": key_info["key_hex"],
                        "mode": mode,
                        "decrypted_at": int(time.time()),
                    }
                    decrypted += 1
                else:
                    failed.append(f"{rel}: decrypt-failed")
            except Exception as exc:
                failed.append(f"{rel}: {type(exc).__name__}: {exc}")
        state["decrypted_dbs"] = decrypted_dbs
        self._save_sync_state(state)
        return {
            "source": os.fspath(source_root),
            "target": os.fspath(self.root),
            "decrypted": decrypted,
            "wal_only_updates": wal_only_updates,
            "unchanged": unchanged,
            "skipped": skipped,
            "failed": failed[:20],
            "failed_count": len(failed),
        }

    def export_all_resources(self) -> dict[str, Any]:
        export_root = self._resource_export_root()
        before = len([path for path in export_root.rglob("*") if path.is_file()]) if export_root.exists() else 0
        try:
            exported = self._export_resource_files(decode_missing=True)
            errors: list[str] = []
        except Exception as exc:
            exported = {}
            errors = [f"{type(exc).__name__}: {exc}"]
        after = len([path for path in export_root.rglob("*") if path.is_file()]) if export_root.exists() else 0
        unique_downloads = {
            str(item.get("download_name") or item.get("file_name") or "")
            for item in exported.values()
            if item.get("download_name") or item.get("file_name")
        }
        return {
            "scanned_chats": 0,
            "resource_items": len(unique_downloads),
            "exported_files": after,
            "new_files": max(0, after - before),
            "errors": errors[:20],
            "error_count": len(errors),
        }

    def initialize_from_live(self, sender: dict, *, export_media: bool = False) -> dict[str, Any]:
        """将空归档绑定到指定在线账号，校验密钥后同步；已有归档不可换账号。"""
        from pyxllib.autogui.weixin4_instrumentation import resolve_sender
        from pyxllib.autogui.wechat_key_scan import scan_account_keys

        if resolve_sender(sender["account_id"]) != sender:
            raise WeChatDbError("账号进程已变化")
        live_root = Path(sender["account_root"])
        state = self._load_sync_state()
        if state.get("live_account_root") and Path(state["live_account_root"]).resolve() != live_root.resolve():
            raise WeChatDbError("归档已属于另一账号，拒绝覆盖")
        matches = scan_account_keys(sender["pid"], live_root / "db_storage")
        required = {"contact/contact.db", "session/session.db", "message/message_0.db"}
        if not required.issubset(matches):
            raise WeChatDbError(f"账号核心数据库密钥不完整：{sorted(required - matches.keys())}")
        if resolve_sender(sender["account_id"]) != sender:
            raise WeChatDbError("密钥校验期间账号进程变化")
        secret = self.root.parent.parent / "secrets" / "wechat_v4_db_keys.json"
        secret.parent.mkdir(parents=True, exist_ok=True)
        secret.write_text(json.dumps({"matches": matches}), encoding="utf-8")
        self._update_sync_state(live_account_root=str(live_root))
        return self.sync_from_live(export_media=export_media)

    def sync_from_live(self, *, export_media: bool = True) -> dict[str, Any]:
        """Serialize account snapshot publication; media export is opt-in for monitors."""
        from filelock import FileLock
        self.root.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.root.parent / "live-sync.lock"), timeout=120):
            return self._sync_from_live(export_media=export_media)

    def poll_updates(self, cursor: dict | None = None, *, limit: int = 1000) -> dict:
        """Refresh all account chats and return new messages plus an opaque cursor.

        None establishes a baseline without replaying history. Persist returned
        cursor atomically with accepted events; retrying the old cursor is safe.
        Resources are fetched separately through list_messages when required.
        """
        from pyxllib.autogui.wechat_updates import poll_storage_updates
        return poll_storage_updates(self, cursor, limit=limit)

    def message_resources(self, chat_id: str, local_id: int) -> dict:
        """Export readable assets on demand for one message; no GUI fallback.

        Returned items contain export.stored_path when decoding succeeds.
        An empty result does not prove the original message had no attachment.
        """
        from filelock import FileLock
        # Sync and media export both update account state. Share ownership so
        # key recovery cannot overwrite a concurrently published DB generation.
        with FileLock(str(self.root.parent / "live-sync.lock"), timeout=120):
            return self._message_resources(chat_id, local_id)

    def _message_resources(self, chat_id: str, local_id: int) -> dict:
        username = self._resolve_chat_username(chat_id)
        result = self._resource_summary(username, export=True, decode_missing=True, local_id=int(local_id)).get(int(local_id), {})
        for item in result.get("items", []):
            exported = item.get("export")
            if not exported:
                continue
            path = Path(exported["stored_path"])
            readable = path.is_file()
            if readable and exported.get("kind") == "image":
                with path.open("rb") as stream:
                    readable = bool(_image_type_from_header(stream.read(64))) and _readable_image(path)
            exported["readable"] = readable
            if readable and exported.get("kind") == "image":
                from PIL import Image
                with Image.open(path) as image:
                    exported["width"], exported["height"] = image.size
                original = str(exported.get("original_file_name") or "")
                exported["variant"] = "high" if original.endswith("_h.dat") else "thumbnail" if original.endswith("_t.dat") else "standard"
                exported.pop("read_error", None)
            if not readable:
                exported["read_error"] = "资源尚未成功解码；不能把原始加密文件当作可读图片"
        return result

    def _sync_from_live(self, *, export_media: bool = True) -> dict[str, Any]:
        started_at = time.time()
        live_account_root = self._wechat_account_root()
        if not live_account_root:
            raise WeChatDbError("未找到本机微信账号目录")
        copy_result = self._copy_live_db_storage(live_account_root)
        if copy_result.get("error_count"):
            raise WeChatDbError("微信源快照不完整，尚未发布：" + "; ".join(copy_result["errors"][:3]))
        decrypt_result = self._decrypt_snapshot_dbs(live_account_root)
        self._exported_resource_files_cache = None
        for decode in (0, 1):
            _WECHAT_EXPORTED_RESOURCE_CACHE.pop(f"{self.root.resolve()}|decode={decode}", None)
        media_result = self.export_all_resources() if export_media else None
        return {
            "live_account_root": os.fspath(live_account_root),
            "elapsed_seconds": round(time.time() - started_at, 3),
            "copy": copy_result,
            "decrypt": decrypt_result,
            "media": media_result,
        }

    def _hardlink_dirs(self) -> dict[int, str]:
        if not self.paths.hardlink.exists():
            return {}
        conn = _connect_readonly(self.paths.hardlink)
        try:
            if not _table_exists(conn, "dir2id"):
                return {}
            return {int(row["rowid"]): row["username"] for row in conn.execute("SELECT rowid, username FROM dir2id")}
        finally:
            conn.close()

    def _hardlink_rows(self, table: str) -> list[dict[str, Any]]:
        if not self.paths.hardlink.exists():
            return []
        conn = _connect_readonly(self.paths.hardlink)
        try:
            if not _table_exists(conn, table):
                return []
            return [dict(row) for row in conn.execute(f'SELECT * FROM "{table}"')]
        finally:
            conn.close()

    def _resource_export_root(self) -> Path:
        return self.root.parent / "exported_media"

    def _resource_manifest_path(self) -> Path:
        return self._resource_export_root() / "manifest.json"

    def _load_exported_resource_manifest(self) -> dict[str, dict[str, Any]]:
        manifest = self._resource_manifest_path()
        if not manifest.exists():
            return {}
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            return {}
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return {}
        exported: dict[str, dict[str, Any]] = {}
        for raw in items:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            stored_path = item.get("stored_path")
            if stored_path and not Path(str(stored_path)).exists():
                continue
            for key in [
                item.get("file_name"),
                item.get("original_file_name"),
                item.get("md5"),
                f"size:{item.get('size')}" if item.get("size") is not None else "",
                f"size:{int(item.get('size')) + 31}" if item.get("size") is not None else "",
            ]:
                if key:
                    exported[str(key)] = item
        return exported

    def _write_exported_resource_manifest(self, exported: dict[str, dict[str, Any]]) -> None:
        manifest = self._resource_manifest_path()
        unique: dict[str, dict[str, Any]] = {}
        for item in exported.values():
            download_name = item.get("download_name")
            if download_name:
                unique[str(download_name)] = item
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "generated_at": int(time.time()),
                    "items": sorted(unique.values(), key=lambda value: str(value.get("download_name") or "")),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(manifest)

    def _wechat_v4_image_xor_key(self, account_root: Path) -> int:
        """Infer the account XOR byte by thumbnail JPEG-end votes.

        One coincidental tail must not poison all images. A failed discovery
        is retried on the next request; only positive evidence is cached.
        """
        if self._image_xor_key_cache is not None:
            return self._image_xor_key_cache
        from collections import Counter
        votes = Counter()
        for dirname in ("msg", "cache", "temp"):
            base = account_root / dirname
            if not base.exists():
                continue
            for index, source in enumerate(base.rglob("*_t.dat")):
                if index >= 2000:
                    break
                try:
                    with source.open("rb") as stream:
                        if stream.read(6) not in WX_IMAGE_V4_AES_KEYS or source.stat().st_size < 31:
                            continue
                        stream.seek(-2, os.SEEK_END)
                        tail = stream.read(2)
                    key = tail[0] ^ 0xFF
                    if key == tail[1] ^ 0xD9:
                        votes[key] += 1
                except OSError:
                    continue
        if not votes:
            return 0
        key, count = votes.most_common(1)[0]
        if len(votes) > 1 and count <= votes.most_common(2)[1][1]:
            return 0
        self._image_xor_key_cache = key
        state = self._load_sync_state()
        state["image_decode"] = {**state.get("image_decode", {}), "xor_key": key}
        self._save_sync_state(state)
        return key

    def _wechat_v4_image_dynamic_aes_key(self, account_root: Path, source: Path | None = None) -> bytes | None:
        attach_dir = account_root / "msg" / "attach"
        if source is not None:
            with source.open("rb") as stream:
                stream.seek(15)
                templates = [stream.read(16)]
        else:
            templates = _find_wechat_v4_image_templates(attach_dir)
        if self._image_aes_key_cache and _verify_wechat_v4_image_aes_key(self._image_aes_key_cache, templates):
            return self._image_aes_key_cache
        retry_key = str(account_root.resolve()) + "|" + "|".join(block.hex() for block in templates)
        state = self._load_sync_state()
        image_state = dict(state.get("image_decode") or {})
        cached_hex = str(image_state.get("aes_key_hex") or "")
        if cached_hex:
            try:
                cached_key = bytes.fromhex(cached_hex)
                if len(cached_key) == 16 and _verify_wechat_v4_image_aes_key(cached_key, templates):
                    self._image_aes_key_cache = cached_key
                    return cached_key
            except ValueError:
                pass
        if _WECHAT_IMAGE_KEY_RETRY_AT.get(retry_key, 0) > time.monotonic():
            return None
        from pyxllib.autogui.weixin4_instrumentation import account_id_from_root, resolve_sender
        # Account UIN is also present in some Windows login-config versions.
        # Never trust a guessed field offset: prove every derived key against
        # this message's actual ciphertext before persisting it.
        account_id = account_id_from_root(account_root)
        for name in ("login_config", "login_configv2"):
            config_path = account_root / "config" / name
            if not config_path.is_file():
                continue
            data = config_path.read_bytes()[:65536]
            candidates = {str(int.from_bytes(data[i:i + 4], order)) for order in ("little", "big")
                          for i in range(max(0, len(data) - 3))}
            candidates.update(part.decode() for part in re.findall(rb"\d{5,10}", data))
            for uin in candidates:
                candidate = hashlib.md5((uin + account_id).encode()).hexdigest()[:16].encode()
                if _verify_wechat_v4_image_aes_key(candidate, templates):
                    self._image_aes_key_cache = candidate
                    image_state["aes_key_hex"] = candidate.hex()
                    image_state["aes_key_verified_at"] = int(time.time())
                    state["image_decode"] = image_state
                    self._save_sync_state(state)
                    return candidate
        suffix = account_root.name.rsplit("_", 1)[-1].lower()
        if re.fullmatch(r"[0-9a-f]{4}", suffix):
            # WeChat's account suffix and image XOR byte constrain the 32-bit
            # UIN to 2^24 candidates. This recovers keys even when image AES
            # material has not yet been loaded into the running process.
            xor = self._wechat_v4_image_xor_key(account_root)
            for uin in range(xor, 1 << 32, 256):
                number = str(uin)
                if hashlib.md5(number.encode()).hexdigest()[:4] != suffix:
                    continue
                for identity in (account_id, account_root.name):
                    candidate = hashlib.md5((number + identity).encode()).hexdigest()[:16].encode()
                    if _verify_wechat_v4_image_aes_key(candidate, templates):
                        self._image_aes_key_cache = candidate
                        image_state["aes_key_hex"] = candidate.hex()
                        image_state["aes_key_verified_at"] = int(time.time())
                        state["image_decode"] = image_state
                        self._save_sync_state(state)
                        return candidate
        sender = resolve_sender(account_id_from_root(account_root))
        self._image_aes_key_cache = _scan_windows_weixin_image_aes_key(templates, pid=sender["pid"])
        if self._image_aes_key_cache:
            image_state["aes_key_hex"] = self._image_aes_key_cache.hex()
            image_state["aes_key_verified_at"] = int(time.time())
            state["image_decode"] = image_state
            self._save_sync_state(state)
        else:
            _WECHAT_IMAGE_KEY_RETRY_AT[retry_key] = time.monotonic() + 30
        return self._image_aes_key_cache

    def _relative_media_path(self, row: dict[str, Any], dirs: dict[int, str], media_kind: str) -> Path | None:
        dir1 = dirs.get(int(row.get("dir1") or 0))
        dir2 = dirs.get(int(row.get("dir2") or 0))
        file_name = row.get("file_name")
        if not dir1 or not file_name:
            return None
        if media_kind == "image" and dir2:
            return Path("msg") / "attach" / dir1 / dir2 / "Img" / str(file_name)
        if media_kind == "video":
            return Path("msg") / "video" / dir1 / str(file_name)
        if media_kind == "file":
            return Path("msg") / "file" / dir1 / str(file_name)
        return None

    def _existing_exported_image(self, export_dir: Path, prefix: str, stem: str) -> Path | None:
        for ext in ("jpg", "png", "gif", "webp", "bmp"):
            candidate = export_dir / f"{prefix}{stem}.{ext}"
            if candidate.exists() and _readable_image(candidate):
                return candidate
        return None

    def _export_resource_files(self, *, decode_missing: bool = True, resource_hints: list[dict] | None = None,
                               chat_username: str | None = None, image_md5: str = "") -> dict[str, dict[str, Any]]:
        if resource_hints is None and decode_missing and self._exported_resource_files_cache is not None:
            return self._exported_resource_files_cache
        cache_key = f"{self.root.resolve()}|decode={int(decode_missing)}"
        cached = _WECHAT_EXPORTED_RESOURCE_CACHE.get(cache_key)
        if resource_hints is None and cached and time.time() - cached[0] < _WECHAT_EXPORTED_RESOURCE_CACHE_TTL:
            if decode_missing:
                self._exported_resource_files_cache = cached[1]
            return cached[1]
        if not decode_missing:
            exported = self._load_exported_resource_manifest()
            _WECHAT_EXPORTED_RESOURCE_CACHE[cache_key] = (time.time(), exported)
            return exported
        manifest_items = self._load_exported_resource_manifest()
        exported = manifest_items if resource_hints is None else {}
        account_root = self._wechat_account_root()
        if not account_root:
            return exported
        dirs = self._hardlink_dirs()
        export_root = self._resource_export_root()
        image_xor_key: int | None = None
        image_aes_key: bytes | None = None
        for table, media_kind in [
            ("image_hardlink_info_v4", "image"),
            ("video_hardlink_info_v4", "video"),
            ("file_hardlink_info_v4", "file"),
        ]:
            rows = self._hardlink_rows(table)
            if media_kind == "image" and resource_hints is not None:
                chat_hash = hashlib.md5((chat_username or "").encode()).hexdigest()
                rows = [row for row in rows if dirs.get(int(row.get("dir1") or 0)) == chat_hash]
                if image_md5:
                    families = {re.sub(r"_[ht]$", "", Path(str(row.get("file_name") or "")).stem)
                                for row in rows if str(row.get("md5") or "") == image_md5}
                    if families:
                        rows = [row for row in rows if re.sub(r"_[ht]$", "", Path(str(row.get("file_name") or "")).stem) in families]
                        # Thumbnail files can be present without a hardlink row.
                        for row in list(rows):
                            if str(row.get("file_name") or "").endswith("_h.dat"):
                                thumb = {**row, "file_name": str(row["file_name"]).replace("_h.dat", "_t.dat"), "md5": ""}
                                path = account_root / self._relative_media_path(thumb, dirs, "image")
                                if path.is_file() and not any(x["file_name"] == thumb["file_name"] for x in rows):
                                    thumb["file_size"] = path.stat().st_size - 31
                                    rows.append(thumb)
                    else:
                        # A different image with the same byte length is not
                        # evidence that this message's missing asset is local.
                        rows = []
            for row in rows:
                if resource_hints is not None:
                    tokens = [str(row.get("md5") or ""), str(row.get("file_name") or "")]
                    size = int(row.get("file_size") or 0)
                    if not any(any(token and token in hint["packed_text"] for token in tokens)
                               or (size and int(hint["size"] or 0) in (size, size + 31))
                               for hint in resource_hints):
                        continue
                relative_path = self._relative_media_path(row, dirs, media_kind)
                if not relative_path:
                    continue
                source = account_root / relative_path
                if not source.exists():
                    continue
                md5_text = str(row.get("md5") or "")
                prefix = f"{md5_text[:8]}_" if md5_text else ""
                original_file_name = str(row.get("file_name") or relative_path.name)
                target_name = f"{prefix}{relative_path.name}"
                cached_item = manifest_items.get(md5_text) or manifest_items.get(original_file_name) or manifest_items.get(target_name)
                cached_stored_path = str(cached_item.get("stored_path") or "") if cached_item else ""
                if (cached_item and cached_stored_path and Path(cached_stored_path).exists()
                        and (media_kind != "image" or _readable_image(Path(cached_stored_path)))):
                    exported[str(cached_item["file_name"])] = cached_item
                    exported[original_file_name] = cached_item
                    if md5_text:
                        exported[md5_text] = cached_item
                    size = int(cached_item.get("size") or row.get("file_size") or 0)
                    if size:
                        exported[f"size:{size}"] = cached_item
                        exported[f"size:{size + 31}"] = cached_item
                    continue
                target = export_root / media_kind / target_name
                decoded_from_dat = False
                if media_kind == "image" and relative_path.suffix.lower() == ".dat":
                    decoded = self._existing_exported_image(export_root / media_kind, prefix, relative_path.stem)
                    if not decoded and decode_missing:
                        if image_xor_key is None:
                            image_xor_key = self._wechat_v4_image_xor_key(account_root)
                        with source.open("rb") as stream:
                            signature = stream.read(6)
                        if signature == b"\x07\x08V2\x08\x07":
                            image_aes_key = self._wechat_v4_image_dynamic_aes_key(account_root, source)
                        else:
                            image_aes_key = None
                        decoded = _decode_wechat_v4_image_dat(
                            source,
                            export_root / media_kind,
                            f"{prefix}{relative_path.stem}",
                            image_xor_key or 0,
                            image_aes_key,
                        )
                    if decoded:
                        target = decoded
                        target_name = target.name
                        decoded_from_dat = True
                if not decoded_from_dat:
                    if not decode_missing and not target.exists():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if decode_missing and (not target.exists() or target.stat().st_size != source.stat().st_size):
                        shutil.copy2(source, target)
                item = {
                    "kind": media_kind,
                    "file_name": target.name,
                    "original_file_name": original_file_name,
                    "size": int(row.get("file_size") or source.stat().st_size),
                    "source_path": os.fspath(source),
                    "stored_path": os.fspath(target),
                    "download_name": f"{media_kind}/{target_name}",
                    "md5": md5_text,
                    "decoded_from_dat": decoded_from_dat,
                }
                exported[item["file_name"]] = item
                exported[original_file_name] = item
                if item["md5"]:
                    exported[item["md5"]] = item
                exported[f"size:{item['size']}"] = item
                exported[f"size:{item['size'] + 31}"] = item

        # Merged-forward records are stored outside hardlink.db under
        # msg/attach/<chat>/<month>/Rec/<record>/Img/<data_index>.  The
        # message_resource rows point at them by data_index and encrypted
        # size, so export these files as first-class image resources too.
        attach_root = account_root / "msg" / "attach"
        if resource_hints is not None:
            attach_root = attach_root / hashlib.md5((chat_username or "").encode()).hexdigest()
        if attach_root.exists():
            forward_pattern = "*/*/Rec/*/Img/*" if resource_hints is None else "*/Rec/*/Img/*"
            for source in attach_root.glob(forward_pattern):
                if not source.is_file():
                    continue
                try:
                    source_size = source.stat().st_size
                    relative = source.relative_to(account_root).as_posix()
                except OSError:
                    continue
                if resource_hints is not None and not any(source_size == int(hint["size"] or 0)
                        or source.name in hint["packed_text"] for hint in resource_hints):
                    continue
                digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
                stem = f"forward_{digest}_{source.name}"
                target = self._existing_exported_image(export_root / "image", "", stem)
                if target is None and decode_missing:
                    if image_xor_key is None:
                        image_xor_key = self._wechat_v4_image_xor_key(account_root)
                    if image_aes_key is None:
                        image_aes_key = self._wechat_v4_image_dynamic_aes_key(account_root, source)
                    target = _decode_wechat_v4_image_dat(
                        source,
                        export_root / "image",
                        stem,
                        image_xor_key or 0,
                        image_aes_key,
                    )
                if target is None:
                    continue
                item = {
                    "kind": "image",
                    "file_name": target.name,
                    "original_file_name": relative,
                    "size": source_size,
                    "source_path": os.fspath(source),
                    "stored_path": os.fspath(target),
                    "download_name": f"image/{target.name}",
                    "md5": hashlib.md5(target.read_bytes()).hexdigest(),
                    "decoded_from_dat": True,
                    "forwarded_record": True,
                }
                exported[item["file_name"]] = item
                exported[relative] = item
                exported[f"size:{source_size}"] = item
        if decode_missing:
            if resource_hints is None:
                self._exported_resource_files_cache = exported
            self._write_exported_resource_manifest({**manifest_items, **exported})
        if resource_hints is None:
            _WECHAT_EXPORTED_RESOURCE_CACHE[cache_key] = (time.time(), exported)
        return exported

    def list_chats(
        self,
        limit: int = 500,
        q: str | None = None,
        offset: int = 0,
        folded: bool | None = None,
        include_folded_entry: bool = False,
    ) -> list[dict[str, Any]]:
        contacts = self._contact_map(include_avatar=True)
        sessions = self._session_map()
        needle = q.strip().lower() if q else ""
        conn = self._message_conn("message")
        try:
            chats: list[dict[str, Any]] = []
            rows = conn.execute(
                "SELECT rowid, user_name, is_session FROM Name2Id WHERE is_session=1 ORDER BY rowid"
            ).fetchall()
            for row in rows:
                username = row["user_name"] or ""
                table = message_table_name(username)
                if not _table_exists(conn, table):
                    continue
                stats = conn.execute(
                    f"""
                    SELECT COUNT(*) AS n, MIN(create_time) AS first_time, MAX(create_time) AS last_time
                    FROM "{table}"
                    """
                ).fetchone()
                message_count = int(stats["n"] or 0)
                session = sessions.get(username) or {}
                display_name = _display_name(username, contacts)
                if display_name == username:
                    display_name = session.get("last_sender_display_name") or username
                searchable = " ".join(
                    str(part or "") for part in [username, display_name, session.get("summary")]
                ).lower()
                if needle and needle not in searchable:
                    continue
                last_type = session.get("last_msg_type")
                last_sender = session.get("last_msg_sender")
                contact = contacts.get(username) or {}
                contact_flag = int(contact.get("flag") or 0)
                chats.append(
                    {
                        "username": username,
                        "name": display_name,
                        "table_name": table,
                        "chat_type": "chatroom" if username.endswith("@chatroom") else "contact",
                        "is_folded": bool(contact_flag & 0x10000000),
                        "message_count": message_count,
                        "first_time": stats["first_time"],
                        "last_time": stats["last_time"],
                        "summary": session.get("summary"),
                        "unread_count": session.get("unread_count"),
                        "last_msg_type": last_type,
                        "last_msg_type_normalized": normalize_message_type(last_type),
                        "last_msg_sender": last_sender,
                        "last_msg_sender_name": _display_name(last_sender, contacts) if last_sender else None,
                        "avatar_data_url": contact.get("avatar_data_url"),
                    }
                )
            chats.sort(key=lambda item: (item["last_time"] or 0, item["message_count"]), reverse=True)
            if include_folded_entry:
                folded_chats = [item for item in chats if item.get("is_folded")]
                normal_chats = [item for item in chats if not item.get("is_folded")]
                if folded_chats:
                    first = folded_chats[0]
                    folded_entry = {
                        **first,
                        "username": "@placeholder_foldgroup",
                        "name": "折叠的聊天",
                        "table_name": "",
                        "chat_type": "folded",
                        "is_folded": False,
                        "is_folded_entry": True,
                        "message_count": len(folded_chats),
                        "summary": f"{first.get('name')}: {first.get('summary') or ''}",
                        "unread_count": sum(int(item.get("unread_count") or 0) for item in folded_chats),
                        "avatar_data_url": None,
                    }
                    chats = normal_chats + [folded_entry]
                    chats.sort(key=lambda item: (item["last_time"] or 0, item["message_count"]), reverse=True)
                else:
                    chats = normal_chats
            elif folded is not None:
                chats = [item for item in chats if bool(item.get("is_folded")) == folded]
            return chats[offset : offset + limit]
        finally:
            conn.close()

    def count_chats(self, q: str | None = None, folded: bool | None = None, include_folded_entry: bool = False) -> int:
        return len(
            self.list_chats(
                limit=100000,
                q=q,
                offset=0,
                folded=folded,
                include_folded_entry=include_folded_entry,
            )
        )

    def _resolve_chat_username(self, chat: str) -> str:
        if chat.startswith("Msg_"):
            conn = self._message_conn("message")
            try:
                if not _table_exists(conn, chat):
                    raise WeChatDbError(f"消息表不存在：{chat}")
            finally:
                conn.close()
            return chat[4:]
        return chat

    def list_messages(
        self,
        chat_username: str,
        q: str | None = None,
        message_type: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
        order: str = "desc",
        include_resources: bool = True,
    ) -> dict[str, Any]:
        limit = min(max(1, int(limit)), MAX_PAGE_SIZE)
        offset = max(0, int(offset))
        order_sql = "ASC" if order == "asc" else "DESC"
        username = self._resolve_chat_username(chat_username)
        table = username if username.startswith("Msg_") else message_table_name(username)
        contacts = self._contact_map(include_avatar=True)
        resource_by_message = (
            self._resource_summary(chat_username, export=True, decode_missing=False) if include_resources else {}
        )
        conn = self._message_conn("message")
        try:
            if not _table_exists(conn, table):
                return {"total": 0, "items": [], "table_name": table}
            clauses = []
            params: list[Any] = []
            if q:
                clauses.append("(msg.message_content LIKE ? OR sender.user_name LIKE ? OR msg.source LIKE ?)")
                needle = f"%{_safe_like(q.strip())}%"
                params.extend([needle, needle, needle])
            if message_type:
                normalized_type = int(message_type)
                clauses.append("(msg.local_type = ? OR (msg.local_type > 65535 AND (msg.local_type & 65535) = ?))")
                params.extend([normalized_type, normalized_type])
            where_sql = " WHERE " + " AND ".join(clauses) if clauses else ""
            total = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM "{table}" msg
                LEFT JOIN Name2Id sender ON sender.rowid = msg.real_sender_id
                {where_sql}
                """,
                params,
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT
                    msg.local_id,
                    msg.server_id,
                    msg.local_type,
                    msg.sort_seq,
                    sender.user_name AS sender_username,
                    msg.create_time,
                    datetime(msg.create_time, 'unixepoch', 'localtime') AS create_time_text,
                    msg.status,
                    msg.upload_status,
                    msg.download_status,
                    msg.server_seq,
                    msg.origin_source,
                    msg.source,
                    msg.message_content,
                    msg.compress_content,
                    length(msg.packed_info_data) AS packed_info_size
                FROM "{table}" msg
                LEFT JOIN Name2Id sender ON sender.rowid = msg.real_sender_id
                {where_sql}
                ORDER BY msg.sort_seq {order_sql}, msg.create_time {order_sql}, msg.local_id {order_sql}
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
            items = []
            for row in rows:
                item = _jsonable_row(row)
                sender_username = item.get("sender_username")
                local_type = item.get("local_type")
                local_id = item.get("local_id")
                item["sender_name"] = _display_name(str(sender_username or ""), contacts) if sender_username else None
                item["sender_avatar_data_url"] = (
                    (contacts.get(str(sender_username or "")) or {}).get("avatar_data_url") if sender_username else None
                )
                item["local_type_normalized"] = normalize_message_type(local_type)
                item["resource"] = resource_by_message.get(int(local_id or 0))
                message_text = _decode_text_value(row["message_content"]) or _decode_text_value(row["compress_content"])
                source_text = _decode_text_value(row["source"])
                item["message_text"] = message_text
                item["source_text"] = source_text
                item["appmsg"] = _parse_appmsg(message_text) or _parse_appmsg(source_text)
                if message_text:
                    item["message_content"] = message_text
                if source_text:
                    item["source"] = source_text
                items.append(item)
            return {
                "total": total,
                "items": items,
                "table_name": table,
            }
        finally:
            conn.close()

    def count_messages(
        self,
        chat_username: str,
        q: str | None = None,
        message_type: str | None = None,
    ) -> dict[str, Any]:
        username = self._resolve_chat_username(chat_username)
        table = username if username.startswith("Msg_") else message_table_name(username)
        conn = self._message_conn("message")
        try:
            if not _table_exists(conn, table):
                return {"total": 0, "table_name": table}
            clauses = []
            params: list[Any] = []
            if q:
                clauses.append("(msg.message_content LIKE ? OR sender.user_name LIKE ? OR msg.source LIKE ?)")
                needle = f"%{_safe_like(q.strip())}%"
                params.extend([needle, needle, needle])
            if message_type:
                normalized_type = int(message_type)
                clauses.append("(msg.local_type = ? OR (msg.local_type > 65535 AND (msg.local_type & 65535) = ?))")
                params.extend([normalized_type, normalized_type])
            where_sql = " WHERE " + " AND ".join(clauses) if clauses else ""
            total = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM "{table}" msg
                LEFT JOIN Name2Id sender ON sender.rowid = msg.real_sender_id
                {where_sql}
                """,
                params,
            ).fetchone()[0]
            return {"total": total, "table_name": table}
        finally:
            conn.close()

    def _resource_summary(
        self,
        chat_username: str,
        *,
        export: bool = False,
        decode_missing: bool = True,
        local_id: int | None = None,
    ) -> dict[int, dict[str, Any]]:
        if not self.paths.resource.exists() or chat_username.startswith("Msg_"):
            return {}
        conn = _connect_readonly(self.paths.resource)
        try:
            if not (_table_exists(conn, "ChatName2Id") and _table_exists(conn, "MessageResourceInfo")):
                return {}
            chat_row = conn.execute("SELECT rowid FROM ChatName2Id WHERE user_name=?", (chat_username,)).fetchone()
            if not chat_row:
                return {}
            rows = conn.execute(
                """
                SELECT
                    info.message_local_id,
                    detail.resource_id,
                    detail.type,
                    detail.size,
                    detail.data_index,
                    detail.packed_info
                FROM MessageResourceInfo info
                LEFT JOIN MessageResourceDetail detail ON detail.message_id = info.message_id
                WHERE info.chat_id = ? AND (? IS NULL OR info.message_local_id = ?)
                """,
                (chat_row["rowid"], local_id, local_id),
            ).fetchall()
            hints = [{"packed_text": _decode_text_value(row["packed_info"]), "size": row["size"]} for row in rows]
            image_md5 = ""
            if local_id is not None:
                message_conn = self._message_conn("message")
                try:
                    table = message_table_name(chat_username)
                    if _table_exists(message_conn, table):
                        message = message_conn.execute(f'SELECT message_content FROM "{table}" WHERE local_id=?', (local_id,)).fetchone()
                        if message:
                            match = re.search(r'<img\b[^>]*\bmd5="([a-fA-F0-9]{32})"', _decode_text_value(message[0]))
                            image_md5 = match.group(1).lower() if match else ""
                finally:
                    message_conn.close()
            exported_files = (self._export_resource_files(decode_missing=decode_missing,
                                resource_hints=hints if local_id is not None else None,
                                chat_username=chat_username, image_md5=image_md5) if export else {})
            grouped: dict[int, dict[str, Any]] = {}
            for row in rows:
                local_id = int(row["message_local_id"] or 0)
                item = grouped.setdefault(
                    local_id,
                    {
                        "resource_count": 0,
                        "total_size": 0,
                        "resource_types": set(),
                        "data_indexes": set(),
                        "items": [],
                    },
                )
                item["resource_count"] += 1
                item["total_size"] += int(row["size"] or 0)
                if row["type"] is not None:
                    item["resource_types"].add(str(row["type"]))
                if row["data_index"] is not None:
                    item["data_indexes"].add(str(row["data_index"]))
                packed_text = _decode_text_value(row["packed_info"])
                exported = None
                for key, value in exported_files.items():
                    if key and key in packed_text:
                        exported = value
                        break
                if not exported:
                    exported = exported_files.get(f"size:{int(row['size'] or 0)}")
                resource_item = {
                    "resource_id": row["resource_id"],
                    "type": row["type"],
                    "size": int(row["size"] or 0),
                    "data_index": row["data_index"],
                    "packed_text": packed_text,
                }
                if exported:
                    resource_item["export"] = exported
                item["items"].append(resource_item)
            return {
                local_id: {
                    **item,
                    "resource_types": ",".join(sorted(item["resource_types"])),
                    "data_indexes": ",".join(sorted(item["data_indexes"])),
                }
                for local_id, item in grouped.items()
            }
        finally:
            conn.close()

    def message_types(self, chat_username: str | None = None) -> list[dict[str, Any]]:
        conn = self._message_conn("message")
        try:
            tables: list[str]
            if chat_username:
                table = message_table_name(chat_username)
                tables = [table] if _table_exists(conn, table) else []
            else:
                tables = [
                    row["name"]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'Msg_%'"
                    ).fetchall()
                    if re.fullmatch(r"Msg_[0-9a-f]{32}", row["name"])
                ]
            counts: dict[int, int] = {}
            raw_counts: dict[int, int] = {}
            for table in tables[:300]:
                for row in conn.execute(f'SELECT local_type, COUNT(*) AS n FROM "{table}" GROUP BY local_type'):
                    raw_key = int(row["local_type"] or 0)
                    key = normalize_message_type(raw_key)
                    raw_counts[raw_key] = raw_counts.get(raw_key, 0) + int(row["n"] or 0)
                    counts[key] = counts.get(key, 0) + int(row["n"] or 0)
            return [
                {"local_type": key, "count": value}
                for key, value in sorted(counts.items(), key=lambda item: item[1], reverse=True)
            ]
        finally:
            conn.close()

    def list_tables(self, database: str) -> list[dict[str, Any]]:
        path = self._database_path(database)
        conn = _connect_readonly(path)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            items = []
            for row in rows:
                table = row["name"]
                count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                columns = [col["name"] for col in conn.execute(f'PRAGMA table_info("{table}")')]
                items.append({"name": table, "count": int(count), "columns": columns})
            return items
        finally:
            conn.close()

    def browse_table(
        self,
        database: str,
        table: str,
        q: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_]+", table):
            raise WeChatDbError(f"非法表名：{table}")
        limit = min(max(1, int(limit)), MAX_PAGE_SIZE)
        offset = max(0, int(offset))
        path = self._database_path(database)
        conn = _connect_readonly(path)
        try:
            if not _table_exists(conn, table):
                raise WeChatDbError(f"数据表不存在：{database}.{table}")
            columns = [col["name"] for col in conn.execute(f'PRAGMA table_info("{table}")')]
            clauses = []
            params: list[Any] = []
            if q:
                text_cols = [
                    col["name"]
                    for col in conn.execute(f'PRAGMA table_info("{table}")')
                    if "TEXT" in (col["type"] or "").upper()
                ]
                if text_cols:
                    clauses.append("(" + " OR ".join(f'"{col}" LIKE ?' for col in text_cols) + ")")
                    params.extend([f"%{_safe_like(q.strip())}%"] * len(text_cols))
            where_sql = " WHERE " + " AND ".join(clauses) if clauses else ""
            total = conn.execute(f'SELECT COUNT(*) FROM "{table}" {where_sql}', params).fetchone()[0]
            rows = conn.execute(
                f'SELECT * FROM "{table}" {where_sql} LIMIT ? OFFSET ?',
                [*params, limit, offset],
            ).fetchall()
            return {
                "database": database,
                "table": table,
                "columns": columns,
                "total": int(total),
                "items": [_jsonable_row(row) for row in rows],
            }
        finally:
            conn.close()

    def schema_overview(self) -> list[dict[str, Any]]:
        items = []
        for name, path in {
            "contact": self.paths.contact,
            "session": self.paths.session,
            "message": self.paths.message,
            "biz_message": self.paths.biz_message,
            "media": self.paths.media,
            "resource": self.paths.resource,
        }.items():
            if not path.exists():
                items.append({"name": name, "path": os.fspath(path), "exists": False, "objects": 0, "tables": []})
                continue
            conn = _connect_readonly(path)
            try:
                rows = conn.execute(
                    "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                ).fetchall()
                tables = [row["name"] for row in rows if row["type"] == "table"]
                items.append(
                    {
                        "name": name,
                        "path": os.fspath(path),
                        "exists": True,
                        "objects": len(rows),
                        "tables": tables[:20],
                    }
                )
            finally:
                conn.close()
        return items
