"""Account-wide committed message updates. Storage, WAL and cursors stay here.

Call WeChatDbStorage.poll_updates; consumers never open WeChat databases.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import struct
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def _checksum(data: bytes, sums: tuple[int, int], endian: str) -> tuple[int, int]:
    words = struct.unpack(endian + "I" * (len(data) // 4), data)
    a, b = sums
    for i in range(0, len(words), 2):
        a = (a + words[i] + b) & 0xffffffff
        b = (b + words[i + 1] + a) & 0xffffffff
    return a, b


def committed_wal_frames(wal: bytes, page_size: int = 4096) -> list[tuple[int, bytes, int]]:
    """Validate rolling checksums/salts and exclude the uncommitted/torn tail."""
    if len(wal) < 32:
        return []
    magic, version, size = struct.unpack(">III", wal[:12])
    if magic not in (0x377f0682, 0x377f0683) or version != 3007000 or size != page_size:
        raise ValueError("Unsupported WAL header")
    endian = "<" if magic == 0x377f0682 else ">"
    sums = _checksum(wal[:24], (0, 0), endian)
    if sums != struct.unpack(">II", wal[24:32]):
        raise ValueError("Invalid WAL header checksum")
    frames, committed = [], 0
    for offset in range(32, len(wal) - 23 - page_size, page_size + 24):
        header = wal[offset:offset + 24]
        page = wal[offset + 24:offset + 24 + page_size]
        if header[8:16] != wal[16:24]:
            break
        sums = _checksum(header[:8] + page, sums, endian)
        if sums != struct.unpack(">II", header[16:24]):
            break
        pgno, db_size = struct.unpack(">II", header[:8])
        if not pgno:
            break
        frames.append((pgno, page, db_size))
        if db_size:
            committed = len(frames)
    return frames[:committed]


def apply_committed_wal(source: Path, snapshot: Path, key_hex: str, mode: str) -> None:
    from Crypto.Cipher import AES
    from Crypto.Hash import SHA512
    from Crypto.Protocol.KDF import PBKDF2
    from pyxllib.autogui.wechat_db import _derive_wx_db_key, _wx_db_reserve_size, SQLITE_HEADER

    wal_path = Path(str(source) + "-wal")
    if not wal_path.exists():
        return
    frames = committed_wal_frames(wal_path.read_bytes())
    if not frames:
        return
    with source.open("rb") as stream:
        salt = stream.read(16)
    key = _derive_wx_db_key(key_hex, mode, salt)
    mac_key = PBKDF2(key, bytes(x ^ 0x3a for x in salt), dkLen=32, count=2, hmac_hash_module=SHA512)
    reserve = _wx_db_reserve_size()
    with snapshot.open("r+b") as stream:
        for pgno, page, db_size in frames:
            offset = 16 if pgno == 1 else 0
            expected = page[4096 - reserve + 16:4096 - reserve + 80]
            mac = hmac.new(mac_key, page[offset:4096 - reserve + 16], hashlib.sha512)
            mac.update(struct.pack("<I", pgno))
            if not hmac.compare_digest(mac.digest(), expected):
                raise ValueError(f"WAL page authentication failed: {pgno}")
            iv = page[4096 - reserve:4096 - reserve + 16]
            plain = AES.new(key, AES.MODE_CBC, iv).decrypt(page[offset:4096 - reserve])
            stream.seek((pgno - 1) * 4096)
            if pgno == 1:
                stream.write(SQLITE_HEADER)
            stream.write(plain + page[4096 - reserve:])
        stream.truncate(frames[-1][2] * 4096)


def mention_ids(source: str) -> list[str]:
    """Only structured @ metadata is authoritative; plain @ text is not an @."""
    if not source:
        return []
    try:
        root = ET.fromstring(source)
        text = root.findtext(".//atuserlist") or ""
    except ET.ParseError:
        return []
    return [value.strip() for value in text.split(",") if value.strip()]


def poll_storage_updates(storage, cursor: dict | None, *, limit: int = 1000) -> dict:
    from filelock import FileLock
    from pyxllib.autogui.wechat_db import _connect_readonly, _decode_text_value, _table_exists, message_table_name, normalize_message_type
    import re

    if not 1 <= limit <= 10000:
        raise ValueError("limit must be 1..10000")
    with FileLock(str(storage.root.parent / "updates.lock"), timeout=120):
        # This is a live API: an archived account must not masquerade as an
        # online listener, and a restarted process cannot borrow another root.
        from pyxllib.autogui.weixin4_instrumentation import account_id_from_root, resolve_sender
        import psutil
        owner = storage.status().get("live_account_root")
        if not owner:
            raise RuntimeError("Live account archive is not initialized")
        # Enumerating every open Windows file handle dominates an idle poll.
        # Cache ownership briefly, validate PID + creation time on every check,
        # and rediscover immediately when the process changes. Sends retain the
        # instrumentation provider's fresh account validation.
        checked_at, sender = getattr(storage, "_updates_live_sender", (0, None))
        valid = False
        if sender:
            try:
                process = psutil.Process(sender["pid"])
                valid = (process.name().lower() == "weixin.exe"
                         and process.create_time() == sender["create_time"])
            except psutil.Error:
                pass
        if not valid or time.monotonic() - checked_at >= 30:
            sender = resolve_sender(account_id_from_root(owner))
            storage._updates_live_sender = (time.monotonic(), sender)
        if Path(sender["account_root"]).resolve() != Path(owner).resolve():
            raise RuntimeError("Live account root changed; prepare the account through its public API")
        sync = storage.sync_from_live(export_media=False)
        if sync["copy"].get("error_count") or sync["decrypt"].get("failed_count"):
            raise RuntimeError("Account snapshot incomplete; cursor was not advanced")
        if cursor is not None and cursor.get("root") != str(storage.root.resolve()):
            raise ValueError("Cursor belongs to another account snapshot")
        if cursor is not None and cursor.get("version", 1) not in (1, 2):
            raise ValueError("Unsupported message cursor version")
        positions = dict((cursor or {}).get("positions") or {})
        started_at = (cursor or {}).get("started_at", int(time.time()))
        contacts = None
        events, has_more = [], False
        connections = []
        try:
            names = []
            # Normal chat history is sharded; local IDs are scoped to a shard.
            # Do not silently limit an account listener to message_0.db.
            paths = sorted(p for p in storage.paths.message.parent.glob("*.db")
                           if re.fullmatch(r"(?:biz_)?message_\d+\.db", p.name))
            for path in paths:
                conn = _connect_readonly(path)
                connections.append(conn)
                if _table_exists(conn, "Name2Id"):
                    names.extend((conn, row, path.name) for row in conn.execute("SELECT user_name FROM Name2Id"))
            for conn, entry, shard in names:
                chat = entry["user_name"]
                if not chat:
                    continue
                table = message_table_name(chat)
                if not _table_exists(conn, table):
                    continue
                position_key = f"{shard}/{chat}"
                if cursor and cursor.get("version", 1) == 1 and shard == storage.paths.message.name and chat in positions:
                    positions[position_key] = positions.pop(chat)
                if cursor is None:
                    positions[position_key] = conn.execute(f'SELECT COALESCE(MAX(local_id),0) FROM "{table}"').fetchone()[0]
                    continue
                remaining = limit - len(events)
                if remaining <= 0:
                    has_more = True
                    break
                rows = conn.execute(f'SELECT msg.*, sender.user_name AS sender_username FROM "{table}" msg '
                                    'LEFT JOIN Name2Id sender ON sender.rowid=msg.real_sender_id '
                                    'WHERE msg.local_id>? ORDER BY msg.local_id LIMIT ?',
                                    (positions.get(position_key, 0), remaining + 1)).fetchall()
                has_more = has_more or len(rows) > remaining
                newly_discovered = position_key not in positions
                for row in rows[:remaining]:
                    positions[position_key] = int(row["local_id"])
                    # A newly discovered table may contain imported history.
                    if newly_discovered and int(row["create_time"] or 0) < started_at:
                        continue
                    if contacts is None:
                        contacts = storage._contact_map(include_avatar=False)
                    sender = row["sender_username"] or ""
                    contact = contacts.get(sender) or {}
                    source = _decode_text_value(row["source"])
                    text = _decode_text_value(row["message_content"]) or _decode_text_value(row["compress_content"])
                    identity = f"{chat}:{row['local_id']}" if shard == storage.paths.message.name else f"{shard}/{chat}:{row['local_id']}"
                    events.append(dict(message_id=identity, chat_id=chat, source_db=shard,
                                       local_id=int(row["local_id"]), server_id=str(row["server_id"] or ""),
                                       sender_id=sender, sender_name=contact.get("remark") or contact.get("nick_name") or sender,
                                       timestamp=int(row["create_time"] or 0), text=text, source=source,
                                       mentions=mention_ids(source), message_type=normalize_message_type(row["local_type"])))
            return dict(events=events, cursor=dict(version=2, root=str(storage.root.resolve()),
                                                  started_at=started_at, positions=positions),
                        has_more=has_more, sync=sync)
        finally:
            for conn in connections:
                conn.close()
