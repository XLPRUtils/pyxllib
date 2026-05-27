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
    b"\x07\x08V2\x08\x07": b"43e7d25eb1b9bb64",
}
WX_IMAGE_V4_XOR_TAIL_SIZE = 0x100000
_WECHAT_IMAGE_AES_KEY_CACHE: dict[str, bytes | None] = {}
_WECHAT_IMAGE_XOR_KEY_CACHE: dict[str, int] = {}
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
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if data.startswith(b"BM"):
        return "bmp"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return ""


def _wechat_v4_image_aes_key(header: bytes) -> bytes | None:
    return WX_IMAGE_V4_AES_KEYS.get(header[:6])


def _detect_image_format(data: bytes) -> str:
    return _image_type_from_header(data) or "bin"


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
                data = source.read_bytes()[:0x1F]
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


def _scan_windows_weixin_image_aes_key(templates: list[bytes]) -> bytes | None:
    if os.name != "nt" or not templates:
        return None
    cache_key = "|".join(template.hex() for template in templates)
    if cache_key in _WECHAT_IMAGE_AES_KEY_CACHE:
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

    try:
        output = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='Weixin.exe'\" | Select-Object -ExpandProperty ProcessId",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
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
            aes_key = aes_key or _wechat_v4_image_aes_key(header)
            if not aes_key:
                return None
            encrypt_length = struct.unpack_from("<H", header, 6)[0]
            encrypt_length0 = encrypt_length // 16 * 16 + 16
            encrypted_data = f.read(encrypt_length0)
            rest_data = f.read()
        if not encrypted_data:
            return None
        if len(encrypted_data) % 16:
            encrypted_data += b"\x00" * (16 - len(encrypted_data) % 16)
        from Crypto.Cipher import AES

        decrypted_data = AES.new(aes_key, AES.MODE_ECB).decrypt(encrypted_data)
        image_type = _image_type_from_header(decrypted_data[:12])
        if not image_type:
            return None
        pad_length = decrypted_data[-1]
        if 1 <= pad_length <= 16:
            decrypted_data = decrypted_data[:-pad_length]
        plain_tail = bytes(byte ^ xor_key for byte in rest_data[-WX_IMAGE_V4_XOR_TAIL_SIZE:])
        plain_data = decrypted_data + rest_data[:-WX_IMAGE_V4_XOR_TAIL_SIZE] + plain_tail
        if _image_type_from_header(plain_data[:12]) != image_type:
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{stem}.{image_type}"
        if not target.exists() or target.stat().st_size != len(plain_data):
            target.write_bytes(plain_data)
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
    }
    return {key: value for key, value in item.items() if value not in ("", None)}


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
        self._image_aes_key_scanned = False
        self._exported_resource_files_cache: dict[str, dict[str, Any]] | None = None

    @property
    def root(self) -> Path:
        return self.paths.root

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
        for candidate in candidates:
            if (candidate / "msg").exists():
                return candidate
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
        copied = 0
        unchanged = 0
        errors: list[str] = []
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            rel = source.relative_to(source_root)
            target = target_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                source_stat = source.stat()
                target_stat = target.stat() if target.exists() else None
                if (
                    target_stat
                    and target_stat.st_size == source_stat.st_size
                    and int(target_stat.st_mtime) == int(source_stat.st_mtime)
                ):
                    unchanged += 1
                    continue
                shutil.copy2(source, target)
                copied += 1
            except Exception as exc:
                errors.append(f"{rel}: {type(exc).__name__}: {exc}")
        return {
            "source": os.fspath(source_root),
            "target": os.fspath(target_root),
            "copied": copied,
            "unchanged": unchanged,
            "errors": errors[:20],
            "error_count": len(errors),
        }

    def _decrypt_snapshot_dbs(self, live_account_root: Path) -> dict[str, Any]:
        source_root = self._raw_snapshot_db_storage(live_account_root)
        matches = self._db_key_matches()
        decrypted = 0
        skipped = 0
        failed: list[str] = []
        for source in sorted(source_root.rglob("*.db")):
            rel = source.relative_to(source_root)
            rel_key = str(rel)
            key_info = matches.get(rel_key) or matches.get(rel_key.replace("/", "\\"))
            if not key_info:
                skipped += 1
                continue
            target = self.root / rel
            try:
                ok = decrypt_wechat_v4_db(source, target, key_info["key_hex"], key_info.get("mode") or "raw-derived-key")
                if ok:
                    decrypted += 1
                else:
                    failed.append(f"{rel}: decrypt-failed")
            except Exception as exc:
                failed.append(f"{rel}: {type(exc).__name__}: {exc}")
        return {
            "source": os.fspath(source_root),
            "target": os.fspath(self.root),
            "decrypted": decrypted,
            "skipped": skipped,
            "failed": failed[:20],
            "failed_count": len(failed),
        }

    def export_all_resources(self) -> dict[str, Any]:
        export_root = self._resource_export_root()
        before = len([path for path in export_root.rglob("*") if path.is_file()]) if export_root.exists() else 0
        chats = self.list_chats(limit=1000)
        scanned = 0
        errors: list[str] = []
        for chat in chats:
            username = chat.get("username")
            if not username:
                continue
            try:
                self._resource_summary(str(username), export=True)
                scanned += 1
            except Exception as exc:
                errors.append(f"{chat.get('name') or username}: {type(exc).__name__}: {exc}")
        after = len([path for path in export_root.rglob("*") if path.is_file()]) if export_root.exists() else 0
        return {
            "scanned_chats": scanned,
            "exported_files": after,
            "new_files": max(0, after - before),
            "errors": errors[:20],
            "error_count": len(errors),
        }

    def sync_from_live(self, *, export_media: bool = True) -> dict[str, Any]:
        started_at = time.time()
        live_account_root = self._wechat_account_root()
        if not live_account_root:
            raise WeChatDbError("未找到本机微信账号目录")
        copy_result = self._copy_live_db_storage(live_account_root)
        decrypt_result = self._decrypt_snapshot_dbs(live_account_root)
        self._exported_resource_files_cache = None
        _WECHAT_EXPORTED_RESOURCE_CACHE.pop(os.fspath(self.root.resolve()), None)
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
        manifest.write_text(
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

    def _wechat_v4_image_xor_key(self, account_root: Path) -> int:
        if self._image_xor_key_cache is not None:
            return self._image_xor_key_cache
        cache_key = os.fspath(account_root.resolve())
        if cache_key in _WECHAT_IMAGE_XOR_KEY_CACHE:
            self._image_xor_key_cache = _WECHAT_IMAGE_XOR_KEY_CACHE[cache_key]
            return self._image_xor_key_cache
        for dirname in ("cache", "temp", "msg"):
            base = account_root / dirname
            if not base.exists():
                continue
            checked = 0
            for source in base.rglob("*_t.dat"):
                checked += 1
                if checked > 2000:
                    break
                try:
                    with source.open("rb") as f:
                        header = f.read(6)
                        if _wechat_v4_image_aes_key(header) is None:
                            continue
                        if source.stat().st_size < 2:
                            continue
                        f.seek(-2, os.SEEK_END)
                        tail = f.read(2)
                    key1 = tail[0] ^ 0xFF
                    key2 = tail[1] ^ 0xD9
                    if key1 == key2:
                        self._image_xor_key_cache = key1
                        _WECHAT_IMAGE_XOR_KEY_CACHE[cache_key] = key1
                        return key1
                except OSError:
                    continue
        self._image_xor_key_cache = 0
        _WECHAT_IMAGE_XOR_KEY_CACHE[cache_key] = 0
        return 0

    def _wechat_v4_image_dynamic_aes_key(self, account_root: Path) -> bytes | None:
        if self._image_aes_key_scanned:
            return self._image_aes_key_cache
        attach_dir = account_root / "msg" / "attach"
        templates = _find_wechat_v4_image_templates(attach_dir)
        self._image_aes_key_cache = _scan_windows_weixin_image_aes_key(templates)
        self._image_aes_key_scanned = True
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
            if candidate.exists():
                return candidate
        return None

    def _export_resource_files(self, *, decode_missing: bool = True) -> dict[str, dict[str, Any]]:
        if decode_missing and self._exported_resource_files_cache is not None:
            return self._exported_resource_files_cache
        cache_key = f"{self.root.resolve()}|decode={int(decode_missing)}"
        cached = _WECHAT_EXPORTED_RESOURCE_CACHE.get(cache_key)
        if cached and time.time() - cached[0] < _WECHAT_EXPORTED_RESOURCE_CACHE_TTL:
            if decode_missing:
                self._exported_resource_files_cache = cached[1]
            return cached[1]
        if not decode_missing:
            exported = self._load_exported_resource_manifest()
            _WECHAT_EXPORTED_RESOURCE_CACHE[cache_key] = (time.time(), exported)
            return exported
        account_root = self._wechat_account_root()
        if not account_root:
            return {}
        dirs = self._hardlink_dirs()
        export_root = self._resource_export_root()
        exported: dict[str, dict[str, Any]] = {}
        image_xor_key = self._wechat_v4_image_xor_key(account_root) if decode_missing else 0
        image_aes_key = self._wechat_v4_image_dynamic_aes_key(account_root) if decode_missing else None
        for table, media_kind in [
            ("image_hardlink_info_v4", "image"),
            ("video_hardlink_info_v4", "video"),
            ("file_hardlink_info_v4", "file"),
        ]:
            for row in self._hardlink_rows(table):
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
                target = export_root / media_kind / target_name
                decoded_from_dat = False
                if media_kind == "image" and relative_path.suffix.lower() == ".dat":
                    decoded = self._existing_exported_image(export_root / media_kind, prefix, relative_path.stem)
                    if not decoded and decode_missing:
                        decoded = _decode_wechat_v4_image_dat(
                            source,
                            export_root / media_kind,
                            f"{prefix}{relative_path.stem}",
                            image_xor_key,
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
        if decode_missing:
            self._exported_resource_files_cache = exported
            self._write_exported_resource_manifest(exported)
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
    ) -> dict[int, dict[str, Any]]:
        if not self.paths.resource.exists() or chat_username.startswith("Msg_"):
            return {}
        exported_files = self._export_resource_files(decode_missing=decode_missing) if export else {}
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
                WHERE info.chat_id = ?
                """,
                (chat_row["rowid"],),
            ).fetchall()
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
