#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""不操作微信界面，直接从本机微信数据库读取『懒人信息转发服务』转发的手机短信。

每次读取只解密 ``message/biz_message_0.db`` 并合并已提交的 ``-wal`` 帧（最新消息只在 WAL），
解密结果缓存在系统临时目录，主库/WAL 未变化时不重复解密。

用法::

    from pyxllib.autogui.wechat_sms import wait_sms_code
    sms = wait_sms_code(after=datetime.now(), timeout=120, contains='微信支付')
    sms['code']

命令行::

    python -m pyxllib.autogui.wechat_sms list
    python -m pyxllib.autogui.wechat_sms wait --after now --timeout 120 --contains 微信支付
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from pyxllib.autogui.wechat_db import (
    _decode_text_value,
    _extract_xml_tag,
    _table_exists,
    decrypt_wechat_v4_db,
    message_table_name,
)

SMS_FORWARD_USERNAME = "gh_09ba3c651273"  # 懒人信息转发服务
SMS_TABLE = message_table_name(SMS_FORWARD_USERNAME)
BIZ_DB_REL = "message/biz_message_0.db"
CODE_PATTERNS = (
    re.compile(r"【(\d{4,8})】"),
    re.compile(r"(?:验证码|校验码|动态码|确认码|code)[^\d]{0,8}(\d{4,8})", re.IGNORECASE),
    re.compile(r"(\d{4,8})[^\d]{0,6}(?:是|为)?(?:您的)?(?:验证码|校验码|动态码)"),
)
_SMS_TIME_RE = re.compile(r"来电时间[：:]\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_SENDER_RE = re.compile(r"来电号码[：:]\s*([^\r\n]*)")
AFTER_TOLERANCE = timedelta(seconds=2)


class WeChatSmsUnavailable(RuntimeError):
    """没有任何账号的转发短信库可读（无 key、无库、解密失败）。"""


def extract_sms_code(text: str) -> str | None:
    for pattern in CODE_PATTERNS:
        if match := pattern.search(text or ""):
            return match.group(1)
    return None


def parse_forwarded_sms(content: Any) -> dict[str, Any] | None:
    """解析一条转发消息；非短信推送（无来电时间、无来电号码）返回 None。"""
    text = _decode_text_value(content)
    des = _extract_xml_tag(text, "des") if "<des" in text else ""
    if not des:
        return None
    sms_time = _SMS_TIME_RE.search(des)
    sender = _SENDER_RE.search(des)
    if not sms_time and not sender:
        return None
    return {
        "sms_time": datetime.strptime(sms_time.group(1), "%Y-%m-%d %H:%M:%S") if sms_time else None,
        "sender": sender.group(1).strip() if sender else "",
        "text": des,
        "code": extract_sms_code(des),
    }


def read_sms_table(db_path: Path, wxid: str = "", since: datetime | None = None) -> list[dict[str, Any]]:
    """从已解密的 biz_message 库读取转发短信，表不存在当作空。"""
    conn = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)
    try:
        if not _table_exists(conn, SMS_TABLE):
            return []
        sql = f"SELECT create_time, message_content FROM {SMS_TABLE}"
        params: tuple = ()
        if since is not None:
            sql += " WHERE create_time >= ?"
            params = (int(since.timestamp()),)
        rows = conn.execute(sql + " ORDER BY create_time DESC, local_id DESC", params).fetchall()
    finally:
        conn.close()
    items = []
    for create_time, content in rows:
        sms = parse_forwarded_sms(content)
        if sms:
            items.append({"received_at": datetime.fromtimestamp(int(create_time or 0)), **sms, "wxid": wxid})
    return items


# ---------------------------------------------------------------- 账号与解密


def _account_sources() -> list[dict[str, Any]]:
    """已初始化归档的账号：wxid、实时库路径、key。key 与账号目录由归档提供方维护。"""
    from pyxllib.autogui.wechat_accounts import list_account_archive_roots
    from pyxllib.autogui.weixin4_instrumentation import DEFAULT_CONTACT_DB, account_id_from_root

    roots = [DEFAULT_CONTACT_DB.parent.parent]
    roots += [path / "wechat_db" / "decrypted" / "db_storage" for path in list_account_archive_roots()]
    sources = []
    for root in roots:
        try:
            state = json.loads((root.parent / "sync_state.json").read_text(encoding="utf-8"))
            live_root = Path(state["live_account_root"])
            matches = json.loads((root.parent.parent / "secrets" / "wechat_v4_db_keys.json")
                                 .read_text(encoding="utf-8")).get("matches") or {}
        except Exception:
            continue
        key_info = matches.get(BIZ_DB_REL) or matches.get(BIZ_DB_REL.replace("/", "\\"))
        sources.append({
            "wxid": account_id_from_root(live_root),
            "db": live_root / "db_storage" / BIZ_DB_REL,
            "key_hex": (key_info or {}).get("key_hex"),
            "mode": (key_info or {}).get("mode") or "raw-derived-key",
        })
    return sources


def _fingerprint(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return st.st_size, st.st_mtime_ns


def _cache_dir(wxid: str) -> Path:
    path = Path(tempfile.gettempdir()) / "pyxllib" / "wechat_sms" / wxid
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mark_rollback_journal(path: Path) -> None:
    """解密库头第 18、19 字节为 2（WAL）时改成 1，否则无 -wal/-shm 只读打开会失败。"""
    with path.open("r+b") as f:
        f.seek(18)
        if f.read(2) == b"":
            f.seek(18)
            f.write(b"")


def decrypt_biz_db(source: dict[str, Any]) -> Path:
    """解密主库并合并已提交 WAL；主库与 WAL 都未变化时直接复用上次结果（跨进程共享缓存）。"""
    from filelock import FileLock

    db: Path = source["db"]
    if not source.get("key_hex"):
        raise WeChatSmsUnavailable(f"{source['wxid']} 缺少 biz_message_0.db 的 key")
    if not db.exists():
        raise WeChatSmsUnavailable(f"{source['wxid']} 的 biz_message_0.db 不存在")
    cache = _cache_dir(source["wxid"])
    base, merged, state_path = cache / "biz_base.db", cache / "biz_message_0.db", cache / "state.json"
    wal = Path(str(db) + "-wal")
    with FileLock(str(cache / "decrypt.lock"), timeout=60):
        db_fp, wal_fp = _fingerprint(db), _fingerprint(wal)
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
        fresh = {"db": list(db_fp or ()), "wal": list(wal_fp or ()), "key": source["key_hex"][:8]}
        if state == fresh and merged.exists():
            return merged
        state_path.unlink(missing_ok=True)
        if state.get("db") != fresh["db"] or state.get("key") != fresh["key"] or not base.exists():
            # decrypt_wechat_v4_db 已合并当时的 WAL；WAL 在检查点前只追加，重放全部已提交帧是幂等的。
            if not decrypt_wechat_v4_db(db, base, source["key_hex"], source["mode"]):
                raise WeChatSmsUnavailable(f"{source['wxid']} 的 biz_message_0.db 解密失败（key 不匹配或读到半写页）")
        tmp = merged.with_suffix(".tmp")
        shutil.copyfile(base, tmp)
        from pyxllib.autogui.wechat_updates import apply_committed_wal
        apply_committed_wal(db, tmp, source["key_hex"], source["mode"])
        _mark_rollback_journal(tmp)
        os.replace(tmp, merged)
        state_path.write_text(json.dumps(fresh), encoding="utf-8")
    return merged


def _select_sources(accounts: Iterable[str] | None) -> list[dict[str, Any]]:
    sources = _account_sources()
    if accounts is not None:
        wanted = set(accounts)
        sources = [s for s in sources if s["wxid"] in wanted]
    return sources


def list_forwarded_sms(since: datetime | None = None, accounts: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """读取各账号收到的转发短信，按 received_at 倒序（最新在前）。

    单个账号读不到时跳过；所有账号都读不到时抛 WeChatSmsUnavailable。
    """
    items, errors, readable = [], [], 0
    for source in _select_sources(accounts):
        for attempt in range(2):  # 微信同时在写/检查点时偶尔读到半写页，重试一次
            try:
                items += read_sms_table(decrypt_biz_db(source), source["wxid"], since)
                readable += 1
                break
            except Exception as exc:
                if attempt:
                    errors.append(f"{source['wxid']}: {type(exc).__name__}: {exc}")
    if not readable:
        raise WeChatSmsUnavailable("没有可读的转发短信库：" + ("; ".join(errors) or "未发现已初始化的微信账号归档"))
    items.sort(key=lambda x: x["received_at"], reverse=True)
    return items


def wait_sms_code(after: datetime, timeout: float | None = 120, interval: float = 1, contains: str | None = None,
                  accounts: Iterable[str] | None = None) -> dict[str, Any]:
    """等待 received_at >= after-2秒、带验证码、正文含 contains 的短信，返回最近收到的那条。

    返回字典含 received_at、sms_time、sender、text、code、wxid；超时抛 TimeoutError。
    """
    since = after - AFTER_TOLERANCE
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        for sms in list_forwarded_sms(since=since, accounts=accounts):
            if sms["code"] and sms["received_at"] >= since and (not contains or contains in sms["text"]):
                return sms
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            raise TimeoutError(f"等待转发短信验证码超时：after={after:%Y-%m-%d %H:%M:%S} timeout={timeout}s contains={contains!r}")
        time.sleep(interval if deadline is None else max(0, min(interval, deadline - now)))


# ---------------------------------------------------------------- CLI


def _parse_after(value: str) -> datetime:
    if value == "now":
        return datetime.now()
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        clock = datetime.strptime(value, "%H:%M:%S").time()
        return datetime.combine(datetime.now().date(), clock)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="读取懒人信息转发服务转发到微信的手机短信（不操作微信界面）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list", help="列出最近的转发短信")
    p_list.add_argument("--hours", type=float, default=24)
    p_list.add_argument("--account", action="append", help="限定 wxid，可重复")
    p_wait = sub.add_parser("wait", help="等待验证码，只打印验证码")
    p_wait.add_argument("--after", default="now", help="now、HH:MM:SS 或 'YYYY-MM-DD HH:MM:SS'")
    p_wait.add_argument("--timeout", type=float, default=120)
    p_wait.add_argument("--interval", type=float, default=1)
    p_wait.add_argument("--contains")
    p_wait.add_argument("--account", action="append")
    args = parser.parse_args(argv)
    if args.cmd == "list":
        since = datetime.now() - timedelta(hours=args.hours)
        for sms in list_forwarded_sms(since=since, accounts=args.account):
            print(json.dumps({**sms, "received_at": str(sms["received_at"]), "sms_time": str(sms["sms_time"] or "")},
                             ensure_ascii=False))
        return 0
    sms = wait_sms_code(_parse_after(args.after), timeout=args.timeout, interval=args.interval,
                        contains=args.contains, accounts=args.account)
    print(sms["code"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
