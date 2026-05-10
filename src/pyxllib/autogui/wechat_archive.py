#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Archive WeChat GUI messages into SQLite.

This module intentionally reads messages through the logged-in WeChat GUI.  It
does not decrypt or parse WeChat's private database files.
"""

import argparse
import hashlib
import json
import os
import re
import socket
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta


class WeChatArchiveError(RuntimeError):
    """Base error for WeChat archive operations."""


class WeChatChatNotFoundError(WeChatArchiveError):
    """Raised when wxautox cannot enter the requested chat."""


def _now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _json_dumps(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha1_text(data):
    return hashlib.sha1(data.encode("utf-8", errors="replace")).hexdigest()


def _safe_getattr(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_datetime_text(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value.replace(microsecond=0)
    text = str(value).strip()
    if not text:
        return None
    for parser in (
            lambda x: datetime.fromisoformat(x.replace(" ", "T")),
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"),
            lambda x: datetime.strptime(x, "%Y-%m-%d"),
    ):
        try:
            return parser(text).replace(microsecond=0)
        except ValueError:
            pass
    return None


def _hours_since(now, value):
    parsed = _parse_datetime_text(value)
    if parsed is None:
        return None
    return max(0.0, (now - parsed).total_seconds() / 3600)


def parse_wechat_time(value, now=None):
    """Parse common WeChat time labels into ``YYYY-mm-dd HH:MM:SS`` text."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")

    text = str(value).strip()
    if not text:
        return None

    now = now or datetime.now()
    text = text.replace("星期天", "星期日")

    patterns = [
        (r"^(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?$", True),
        (r"^(\d{4})/(\d{1,2})/(\d{1,2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?$", True),
        (r"^(\d{4})年(\d{1,2})月(\d{1,2})日 ?(\d{1,2}):(\d{2})(?::(\d{2}))?$", True),
    ]
    for pattern, _ in patterns:
        match = re.match(pattern, text)
        if match:
            year, month, day, hour, minute, second = match.groups()
            return datetime(
                int(year), int(month), int(day), int(hour), int(minute), int(second or 0)
            ).strftime("%Y-%m-%d %H:%M:%S")

    match = re.match(r"^(\d{1,2})-(\d{1,2}) (\d{1,2}):(\d{2})(?::(\d{2}))?$", text)
    if match:
        month, day, hour, minute, second = match.groups()
        return datetime(now.year, int(month), int(day), int(hour), int(minute), int(second or 0)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    match = re.match(r"^(\d{1,2}):(\d{2})$", text)
    if match:
        hour, minute = match.groups()
        return datetime(now.year, now.month, now.day, int(hour), int(minute)).strftime("%Y-%m-%d %H:%M:%S")

    match = re.match(r"^昨天 ?(\d{1,2}):(\d{2})$", text)
    if match:
        hour, minute = match.groups()
        day = now - timedelta(days=1)
        return datetime(day.year, day.month, day.day, int(hour), int(minute)).strftime("%Y-%m-%d %H:%M:%S")

    match = re.match(r"^星期([一二三四五六日]) ?(\d{1,2}):(\d{2})$", text)
    if match:
        weekday, hour, minute = match.groups()
        weekday_num = ["一", "二", "三", "四", "五", "六", "日"].index(weekday)
        delta_days = (now.weekday() - weekday_num) % 7
        day = now - timedelta(days=delta_days)
        return datetime(day.year, day.month, day.day, int(hour), int(minute)).strftime("%Y-%m-%d %H:%M:%S")

    match = re.match(r"^(\d{4})年(\d{1,2})月(\d{1,2})日 (上午|下午) ?(\d{1,2}):(\d{2})$", text)
    if match:
        year, month, day, period, hour, minute = match.groups()
        hour = _parse_period_hour(period, hour)
        return datetime(int(year), int(month), int(day), hour, int(minute)).strftime("%Y-%m-%d %H:%M:%S")

    match = re.match(r"^(\d{1,2})-(\d{1,2}) (上午|下午) ?(\d{1,2}):(\d{2})$", text)
    if match:
        month, day, period, hour, minute = match.groups()
        hour = _parse_period_hour(period, hour)
        return datetime(now.year, int(month), int(day), hour, int(minute)).strftime("%Y-%m-%d %H:%M:%S")

    return text


def _parse_period_hour(period, hour):
    hour = int(hour)
    if period == "下午" and hour != 12:
        hour += 12
    elif period == "上午" and hour == 12:
        hour = 0
    return hour


def classify_message_type(msg_type, mtype, content):
    if msg_type in ("time", "sys", "recall"):
        return msg_type
    if mtype:
        return mtype

    text = content or ""
    placeholder_map = [
        ("[图片]", "image"),
        ("[视频]", "video"),
        ("[文件]", "file"),
        ("[语音]", "voice"),
        ("[链接]", "link"),
        ("[音乐]", "music"),
        ("[位置]", "location"),
    ]
    for prefix, kind in placeholder_map:
        if text.startswith(prefix):
            return kind
    return "text"


def normalize_wx_messages(messages, chat_name, collected_at=None, now=None):
    """Normalize wxautox message objects into plain dictionaries."""
    collected_at = collected_at or _now_text()
    current_time = None
    current_time_label = None
    rows = []

    for index, msg in enumerate(messages):
        msg_type = _safe_getattr(msg, "type", "message") or "message"
        mtype = _safe_getattr(msg, "mtype", "") or ""
        content = _safe_getattr(msg, "content", "")
        raw_id = _safe_getattr(msg, "id", None)
        sender = _safe_getattr(msg, "sender", None)
        sender_remark = _safe_getattr(msg, "sender_remark", None)

        if msg_type == "time":
            current_time_label = content
            current_time = parse_wechat_time(_safe_getattr(msg, "time", content), now=now)

        if msg_type == "self":
            direction = "out"
            sender = sender or "self"
        elif msg_type == "friend":
            direction = "in"
        else:
            direction = "system"

        normalized_time = current_time
        raw_time_label = current_time_label
        if msg_type == "time":
            normalized_time = current_time
            raw_time_label = content

        message_type = classify_message_type(msg_type, mtype, content)
        media_path = None
        if message_type in ("image", "video", "file", "voice") and content and not content.startswith("["):
            media_path = content

        raw = {
            "class": msg.__class__.__name__,
            "index": index,
            "id": raw_id,
            "type": msg_type,
            "mtype": mtype,
            "sender": sender,
            "sender_remark": sender_remark,
            "content": content,
            "time": _safe_getattr(msg, "time", None),
            "info": _safe_info(msg),
        }
        fingerprint_data = {
            "chat": chat_name,
            "direction": direction,
            "sender": sender,
            "sender_remark": sender_remark,
            "type": message_type,
            "content": content,
            "time": normalized_time,
            "raw_time_label": raw_time_label,
        }
        if not normalized_time and raw_id:
            fingerprint_data["raw_id"] = raw_id

        rows.append(
            {
                "chat_name": chat_name,
                "direction": direction,
                "sender": sender,
                "sender_remark": sender_remark,
                "message_type": message_type,
                "content": content,
                "media_path": media_path,
                "normalized_time": normalized_time,
                "raw_time_label": raw_time_label,
                "raw_id": raw_id,
                "raw_json": _json_dumps(raw),
                "fingerprint": _sha1_text(_json_dumps(fingerprint_data)),
                "collected_at": collected_at,
            }
        )

    return rows


def _safe_info(msg):
    info = _safe_getattr(msg, "info", None)
    if info is None:
        return None
    try:
        json.dumps(info, ensure_ascii=False)
        return info
    except TypeError:
        return repr(info)


class WeChatArchive:
    """High-level API for archiving a WeChat conversation through GUI/UIA."""

    def __init__(self, db_path, *, wx=None, lock_timeout=-1, chat_timeout=5):
        self.db_path = os.path.abspath(os.fspath(db_path))
        self.wx = wx
        self.lock_timeout = lock_timeout
        self.chat_timeout = chat_timeout
        self.ensure_schema()

    def full_chat(self, chat_name, **kwargs):
        kwargs.setdefault("until", "top")
        kwargs.setdefault("max_scrolls", None)
        return self.pull_chat(chat_name, **kwargs)

    def pull_chat(
            self,
            chat_name,
            *,
            until="loaded",
            max_scrolls=0,
            exact=True,
            save_media=False,
            savepic=None,
            savevideo=None,
            savefile=None,
            savevoice=None,
            parseurl=False,
            scroll_interval=0.3,
            settle_timeout=1.5,
            max_no_progress=2,
            max_runtime=None,
            lock=True,
            enable_sync=True):
        """Archive one chat.

        Args:
            until: ``"loaded"`` reads only loaded messages, ``"top"`` scrolls
                upward until WeChat reports no more history.
            max_scrolls: safety limit.  Use ``None`` together with
                ``until="top"`` for full history.
        """
        if until not in ("loaded", "top"):
            raise ValueError("until must be 'loaded' or 'top'")
        if max_scrolls is None and until != "top":
            until = "top"

        start_time = time.monotonic()
        total_seen = 0
        total_inserted = 0
        scroll_count = 0
        no_progress = 0
        reached_top = False
        last_error = None
        previous_oldest = None
        previous_screen = None
        chat_id = None

        with self._connect() as conn:
            with self._wechat_session(lock=lock) as wx:
                matched_name = self._enter_chat(wx, chat_name, exact=exact)
                self._wait_messages_stable(wx, timeout=settle_timeout)
                chat_info = self._get_chat_info(wx, matched_name or chat_name)
                account_id = self._upsert_account(conn, wx)
                chat_id = self._upsert_chat(conn, account_id, chat_info, enable_sync=enable_sync)

                while True:
                    if max_runtime is not None and time.monotonic() - start_time >= max_runtime:
                        last_error = "max_runtime reached"
                        break

                    wx_messages = wx.GetAllMessage(
                        savepic=save_media if savepic is None else savepic,
                        savevideo=save_media if savevideo is None else savevideo,
                        savefile=save_media if savefile is None else savefile,
                        savevoice=save_media if savevoice is None else savevoice,
                        parseurl=parseurl,
                    )
                    records = normalize_wx_messages(wx_messages, chat_info["chat_name"])
                    inserted = self._insert_messages(conn, account_id, chat_id, records)
                    conn.commit()

                    total_seen += len(records)
                    total_inserted += inserted

                    fingerprints = [row["fingerprint"] for row in records]
                    oldest = fingerprints[0] if fingerprints else None
                    newest = fingerprints[-1] if fingerprints else None
                    screen = _sha1_text(_json_dumps(fingerprints))
                    if oldest and (oldest == previous_oldest or screen == previous_screen):
                        no_progress += 1
                    else:
                        no_progress = 0
                    previous_oldest = oldest
                    previous_screen = screen

                    self._update_sync_state(
                        conn,
                        chat_id,
                        oldest_fingerprint=oldest,
                        newest_fingerprint=newest,
                        loaded_count=total_seen,
                        scroll_count=scroll_count,
                        reached_top=reached_top,
                        last_error=last_error,
                    )
                    conn.commit()

                    if until == "loaded":
                        break
                    if max_scrolls is not None and scroll_count >= max_scrolls:
                        break
                    if max_no_progress is not None and no_progress >= max_no_progress:
                        last_error = "no progress while loading history"
                        break

                    loaded = wx.LoadMoreMessage(interval=scroll_interval)
                    scroll_count += 1
                    if not loaded:
                        reached_top = True
                        last_error = None
                        break

                self._update_sync_state(
                    conn,
                    chat_id,
                    loaded_count=total_seen,
                    scroll_count=scroll_count,
                    reached_top=reached_top,
                    last_error=last_error,
                    sync_latest=until == "loaded",
                    sync_history=until == "top",
                    sync_success=True,
                )
                conn.commit()

        return {
            "chat_name": chat_name,
            "matched_name": matched_name,
            "chat_id": chat_id,
            "db_path": self.db_path,
            "seen": total_seen,
            "inserted": total_inserted,
            "scroll_count": scroll_count,
            "reached_top": reached_top,
            "last_error": last_error,
        }

    def sync_latest(self, chat_names, **kwargs):
        results = []
        for chat_name in chat_names:
            options = dict(kwargs)
            options.setdefault("until", "loaded")
            options.setdefault("max_scrolls", 0)
            results.append(self.pull_chat(chat_name, **options))
        return results

    def sync_chat_latest(self, chat_name, **kwargs):
        """Read only the currently loaded tail of one chat."""
        kwargs.setdefault("until", "loaded")
        kwargs.setdefault("max_scrolls", 0)
        return self.pull_chat(chat_name, **kwargs)

    def backfill_chat_history(self, chat_name, *, max_scrolls=1, **kwargs):
        """Load a small slice of older history for one chat."""
        kwargs.setdefault("until", "top")
        kwargs.setdefault("max_scrolls", max_scrolls)
        return self.pull_chat(chat_name, **kwargs)

    def sync_incremental(
            self,
            *,
            chat_names=None,
            max_runtime=90,
            max_chats=6,
            max_scrolls_total=8,
            max_scrolls_per_chat=1,
            sync_latest=True,
            backfill_history=True,
            exact=True,
            save_media=False,
            use_wechat_hints=True):
        """Synchronize recent tails first, then backfill a bounded amount of history."""
        if self.wx is None:
            with self._wechat_session(lock=True) as wx:
                original_wx = self.wx
                self.wx = wx
                try:
                    return self.sync_incremental(
                        chat_names=chat_names,
                        max_runtime=max_runtime,
                        max_chats=max_chats,
                        max_scrolls_total=max_scrolls_total,
                        max_scrolls_per_chat=max_scrolls_per_chat,
                        sync_latest=sync_latest,
                        backfill_history=backfill_history,
                        exact=exact,
                        save_media=save_media,
                        use_wechat_hints=use_wechat_hints,
                    )
                finally:
                    self.wx = original_wx

        started_at = _now_text()
        start_time = time.monotonic()
        recent_chat_names = []
        unread_chat_names = {}
        if use_wechat_hints:
            try:
                hints = self._read_wechat_session_hints()
                recent_chat_names = hints.get("recent_chat_names") or []
                unread_chat_names = hints.get("unread_chat_names") or {}
            except Exception:
                recent_chat_names = []
                unread_chat_names = {}

        plan = self.plan_sync_chats(
            manual_chat_names=chat_names,
            recent_chat_names=recent_chat_names,
            unread_chat_names=unread_chat_names,
            limit=max_chats,
        )
        remaining_scrolls = max_scrolls_total
        results = []
        processed_chats = 0
        total_seen = 0
        total_inserted = 0
        total_scroll_count = 0
        stopped_reason = None

        for item in plan:
            if max_chats is not None and processed_chats >= max_chats:
                stopped_reason = "max_chats reached"
                break
            if max_runtime is not None and time.monotonic() - start_time >= max_runtime:
                stopped_reason = "max_runtime reached"
                break

            chat_name = item["name"]
            chat_id = item.get("chat_id")
            processed_chats += 1

            if sync_latest and item.get("sync_latest", True):
                result = self._run_sync_phase(
                    chat_name,
                    "latest",
                    chat_id=chat_id,
                    exact=exact,
                    save_media=save_media,
                    max_runtime=self._remaining_runtime(start_time, max_runtime),
                )
                results.append(result)
                total_seen += result.get("seen", 0) or 0
                total_inserted += result.get("inserted", 0) or 0
                if result.get("error"):
                    continue

            can_backfill = (
                backfill_history
                and item.get("backfill_history", True)
                and not item.get("reached_top")
                and (remaining_scrolls is None or remaining_scrolls > 0)
            )
            if can_backfill:
                phase_scrolls = max_scrolls_per_chat
                if remaining_scrolls is not None:
                    phase_scrolls = min(phase_scrolls, remaining_scrolls)
                if phase_scrolls > 0:
                    result = self._run_sync_phase(
                        chat_name,
                        "history",
                        chat_id=chat_id,
                        exact=exact,
                        save_media=save_media,
                        max_scrolls=phase_scrolls,
                        max_runtime=self._remaining_runtime(start_time, max_runtime),
                    )
                    results.append(result)
                    scroll_count = result.get("scroll_count", 0) or 0
                    total_seen += result.get("seen", 0) or 0
                    total_inserted += result.get("inserted", 0) or 0
                    total_scroll_count += scroll_count
                    if remaining_scrolls is not None:
                        remaining_scrolls = max(0, remaining_scrolls - scroll_count)

        return {
            "db_path": self.db_path,
            "started_at": started_at,
            "finished_at": _now_text(),
            "max_runtime": max_runtime,
            "max_chats": max_chats,
            "max_scrolls_total": max_scrolls_total,
            "max_scrolls_per_chat": max_scrolls_per_chat,
            "planned": plan,
            "processed_chats": processed_chats,
            "results": results,
            "seen": total_seen,
            "inserted": total_inserted,
            "scroll_count": total_scroll_count,
            "remaining_scrolls": remaining_scrolls,
            "stopped_reason": stopped_reason,
        }

    def plan_sync_chats(
            self,
            *,
            manual_chat_names=None,
            recent_chat_names=None,
            unread_chat_names=None,
            limit=None,
            include_disabled=False,
            now=None):
        """Build a priority-ordered chat sync plan from archive state and optional GUI hints."""
        now = now or datetime.now()
        manual_names = _as_list(manual_chat_names)
        manual_set = set(manual_names)
        recent_names = _as_list(recent_chat_names)
        recent_rank = {name: index for index, name in enumerate(recent_names)}
        unread_map = unread_chat_names or {}
        if isinstance(unread_map, (list, tuple, set)):
            unread_map = {name: 1 for name in unread_map}

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.id AS chat_id,
                    c.name,
                    c.chat_type,
                    c.remark,
                    cfg.enabled,
                    cfg.priority,
                    cfg.sync_latest,
                    cfg.backfill_history,
                    ss.reached_top,
                    ss.last_error,
                    ss.last_incremental_at,
                    ss.last_history_at,
                    ss.last_success_at,
                    ss.consecutive_failures,
                    ss.next_due_at,
                    COUNT(m.id) AS message_count,
                    MAX(m.collected_at) AS latest_collected_at,
                    MAX(m.normalized_time) AS last_message_time
                FROM chats c
                LEFT JOIN chat_sync_config cfg ON cfg.chat_id = c.id
                LEFT JOIN sync_state ss ON ss.chat_id = c.id
                LEFT JOIN messages m ON m.chat_id = c.id
                GROUP BY c.id
                """
            ).fetchall()

        items = []
        known_names = set()
        for row in rows:
            item = dict(row)
            known_names.add(item["name"])
            enabled = bool(item.get("enabled", 1))
            if not enabled and item["name"] not in manual_set and not include_disabled:
                continue
            scored = self._score_sync_plan_item(item, now, manual_set, recent_rank, unread_map)
            if scored is not None:
                items.append(scored)

        for name in manual_names:
            if name in known_names:
                continue
            items.append(
                {
                    "chat_id": None,
                    "name": name,
                    "enabled": True,
                    "priority": 0,
                    "sync_latest": True,
                    "backfill_history": False,
                    "reached_top": False,
                    "message_count": 0,
                    "latest_collected_at": None,
                    "last_message_time": None,
                    "last_incremental_at": None,
                    "last_history_at": None,
                    "last_success_at": None,
                    "consecutive_failures": 0,
                    "next_due_at": None,
                    "score": 10000,
                    "reasons": ["manual"],
                    "due": True,
                }
            )

        items.sort(key=lambda item: (-item["score"], item["name"]))
        if limit is not None:
            items = items[:limit]
        return items

    def _remaining_runtime(self, start_time, max_runtime):
        if max_runtime is None:
            return None
        return max(1.0, max_runtime - (time.monotonic() - start_time))

    def _run_sync_phase(self, chat_name, phase, *, chat_id=None, exact=True, save_media=False, max_scrolls=0, max_runtime=None):
        try:
            if phase == "history":
                result = self.backfill_chat_history(
                    chat_name,
                    max_scrolls=max_scrolls,
                    exact=exact,
                    save_media=save_media,
                    max_runtime=max_runtime,
                    enable_sync=False,
                )
            else:
                result = self.sync_chat_latest(
                    chat_name,
                    exact=exact,
                    save_media=save_media,
                    max_runtime=max_runtime,
                    enable_sync=False,
                )
            result["phase"] = phase
            return result
        except Exception as exc:
            if chat_id:
                with self._connect() as conn:
                    self._record_sync_failure(conn, chat_id, str(exc))
                    conn.commit()
            return {
                "chat_name": chat_name,
                "chat_id": chat_id,
                "phase": phase,
                "seen": 0,
                "inserted": 0,
                "scroll_count": 0,
                "reached_top": False,
                "last_error": str(exc),
                "error": str(exc),
            }

    def _score_sync_plan_item(self, item, now, manual_set, recent_rank, unread_map):
        name = item["name"]
        next_due_at = _parse_datetime_text(item.get("next_due_at"))
        manual = name in manual_set
        if next_due_at and next_due_at > now and not manual:
            return None

        reasons = []
        score = float(_as_int(item.get("priority"), 0))
        if manual:
            score += 10000
            reasons.append("manual")

        unread_count = _as_int(unread_map.get(name), 0)
        if unread_count:
            score += 800 + min(unread_count, 99) * 4
            reasons.append("unread:{}".format(unread_count))

        if name in recent_rank:
            rank = recent_rank[name]
            score += max(0, 400 - rank * 8)
            reasons.append("recent:{}".format(rank + 1))

        latest_age = _hours_since(now, item.get("latest_collected_at"))
        if latest_age is None:
            score += 120
            reasons.append("never_collected")
        else:
            score += max(0, 160 - latest_age)
            if latest_age >= 12:
                score += min(160, latest_age / 2)
                reasons.append("stale:{:.1f}h".format(latest_age))

        success_age = _hours_since(now, item.get("last_success_at"))
        if success_age is not None and success_age >= 24:
            score += min(120, success_age / 6)
            reasons.append("success_stale:{:.1f}h".format(success_age))

        if item.get("reached_top") in (0, None, False) and item.get("backfill_history", 1):
            score += 20
            reasons.append("history_pending")

        failures = _as_int(item.get("consecutive_failures"), 0)
        if failures:
            score -= min(1000, 240 * failures)
            reasons.append("failures:{}".format(failures))

        return {
            "chat_id": item.get("chat_id"),
            "name": name,
            "chat_type": item.get("chat_type"),
            "remark": item.get("remark"),
            "enabled": bool(item.get("enabled", 1)),
            "priority": _as_int(item.get("priority"), 0),
            "sync_latest": bool(item.get("sync_latest", 1)),
            "backfill_history": bool(item.get("backfill_history", 1)),
            "reached_top": bool(item.get("reached_top", 0)),
            "message_count": _as_int(item.get("message_count"), 0),
            "latest_collected_at": item.get("latest_collected_at"),
            "last_message_time": item.get("last_message_time"),
            "last_incremental_at": item.get("last_incremental_at"),
            "last_history_at": item.get("last_history_at"),
            "last_success_at": item.get("last_success_at"),
            "consecutive_failures": failures,
            "next_due_at": item.get("next_due_at"),
            "score": round(score, 2),
            "reasons": reasons,
            "due": True,
        }

    def _read_wechat_session_hints(self):
        recent_chat_names = []
        unread_chat_names = {}
        with self._wechat_session(lock=True) as wx:
            if hasattr(wx, "GetSession"):
                sessions = wx.GetSession()
                for session in sessions:
                    name = _safe_getattr(session, "name", None)
                    if not name:
                        continue
                    recent_chat_names.append(name)
                    if _safe_getattr(session, "isnew", False):
                        unread_chat_names[name] = _as_int(_safe_getattr(session, "new_count", 1), 1)
            elif hasattr(wx, "GetSessionList"):
                sessions = wx.GetSessionList(reset=True)
                if isinstance(sessions, dict):
                    recent_chat_names = list(sessions)
                    unread_chat_names = {name: _as_int(count, 0) for name, count in sessions.items() if _as_int(count, 0)}
        return {
            "recent_chat_names": recent_chat_names,
            "unread_chat_names": unread_chat_names,
        }

    def ensure_schema(self):
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_key TEXT NOT NULL UNIQUE,
                    nickname TEXT,
                    computer_name TEXT,
                    wx_version TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    chat_type TEXT,
                    remark TEXT,
                    group_member_count INTEGER,
                    status TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(account_id, name),
                    FOREIGN KEY(account_id) REFERENCES accounts(id)
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    direction TEXT,
                    sender TEXT,
                    sender_remark TEXT,
                    message_type TEXT,
                    content TEXT,
                    media_path TEXT,
                    normalized_time TEXT,
                    raw_time_label TEXT,
                    raw_id TEXT,
                    raw_json TEXT,
                    fingerprint TEXT NOT NULL UNIQUE,
                    collected_at TEXT NOT NULL,
                    FOREIGN KEY(account_id) REFERENCES accounts(id),
                    FOREIGN KEY(chat_id) REFERENCES chats(id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_chat_time ON messages(chat_id, normalized_time, id);
                CREATE INDEX IF NOT EXISTS idx_messages_chat_type ON messages(chat_id, message_type);

                CREATE TABLE IF NOT EXISTS sync_state (
                    chat_id INTEGER PRIMARY KEY,
                    oldest_fingerprint TEXT,
                    newest_fingerprint TEXT,
                    loaded_count INTEGER NOT NULL DEFAULT 0,
                    scroll_count INTEGER NOT NULL DEFAULT 0,
                    reached_top INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    last_incremental_at TEXT,
                    last_history_at TEXT,
                    last_success_at TEXT,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    next_due_at TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(id)
                );

                CREATE TABLE IF NOT EXISTS chat_sync_config (
                    chat_id INTEGER PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    priority INTEGER NOT NULL DEFAULT 0,
                    sync_latest INTEGER NOT NULL DEFAULT 1,
                    backfill_history INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(id)
                );

                CREATE INDEX IF NOT EXISTS idx_chat_sync_config_enabled ON chat_sync_config(enabled, priority);
                """
            )
            self._ensure_table_columns(
                conn,
                "sync_state",
                {
                    "last_incremental_at": "TEXT",
                    "last_history_at": "TEXT",
                    "last_success_at": "TEXT",
                    "consecutive_failures": "INTEGER NOT NULL DEFAULT 0",
                    "next_due_at": "TEXT",
                },
            )
            now = _now_text()
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_sync_config(
                    chat_id, enabled, priority, sync_latest, backfill_history, created_at, updated_at
                )
                SELECT id, 1, 0, 1, 1, ?, ? FROM chats
                """,
                (now, now),
            )
            conn.commit()

    def _ensure_table_columns(self, conn, table_name, columns):
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info({})".format(table_name)).fetchall()
        }
        for name, definition in columns.items():
            if name not in existing:
                conn.execute("ALTER TABLE {} ADD COLUMN {} {}".format(table_name, name, definition))

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _wechat_session(self, lock=True):
        if self.wx is not None:
            return _NullWechatContext(self.wx)
        if not lock:
            from pyxllib.autogui.wxautolib import WeChat

            return _NullWechatContext(WeChat())
        from pyxllib.autogui.wxautolib import WeChatSingletonLock

        return WeChatSingletonLock(self.lock_timeout)

    def _enter_chat(self, wx, chat_name, exact=True):
        if not hasattr(wx, "ChatWith"):
            return chat_name
        matched_name = wx.ChatWith(chat_name, timeout=self.chat_timeout, exact=exact)
        if exact:
            if not matched_name:
                raise WeChatChatNotFoundError("Cannot find WeChat chat: {}".format(chat_name))
            if matched_name != chat_name:
                raise WeChatChatNotFoundError("Expected chat {!r}, got {!r}".format(chat_name, matched_name))
        elif not matched_name:
            raise WeChatChatNotFoundError("Cannot find WeChat chat: {}".format(chat_name))
        return matched_name

    def _get_chat_info(self, wx, fallback_name):
        info = {}
        try:
            if hasattr(wx, "CurrentChat"):
                info = wx.CurrentChat(details=True) or {}
        except Exception:
            info = {}
        name = info.get("chat_name") or fallback_name
        return {
            "chat_name": name,
            "chat_type": info.get("chat_type"),
            "chat_remark": info.get("chat_remark"),
            "group_member_count": info.get("group_member_count"),
        }

    def _wait_messages_stable(self, wx, timeout=1.5, poll_interval=0.2):
        if not timeout or timeout <= 0:
            return
        msg_list = _safe_getattr(wx, "C_MsgList", None)
        if msg_list is None or not hasattr(msg_list, "GetChildren"):
            time.sleep(min(timeout, poll_interval))
            return

        deadline = time.monotonic() + timeout
        last_count = None
        stable_count = 0
        while time.monotonic() < deadline:
            try:
                count = len(msg_list.GetChildren())
            except Exception:
                time.sleep(poll_interval)
                continue
            if count and count == last_count:
                stable_count += 1
                if stable_count >= 2:
                    return
            else:
                stable_count = 0
                last_count = count
            time.sleep(poll_interval)

    def _upsert_account(self, conn, wx):
        nickname = _safe_getattr(wx, "nickname", None)
        computer_name = socket.gethostname()
        account_key = nickname or os.environ.get("WECHAT_ACCOUNT") or "{}:{}".format(
            computer_name, os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"
        )
        wx_version = _safe_getattr(wx, "VERSION", None)
        now = _now_text()
        conn.execute(
            """
            INSERT INTO accounts(account_key, nickname, computer_name, wx_version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_key) DO UPDATE SET
                nickname=excluded.nickname,
                computer_name=excluded.computer_name,
                wx_version=excluded.wx_version,
                updated_at=excluded.updated_at
            """,
            (account_key, nickname, computer_name, wx_version, now, now),
        )
        row = conn.execute("SELECT id FROM accounts WHERE account_key=?", (account_key,)).fetchone()
        return int(row["id"])

    def _upsert_chat(self, conn, account_id, chat_info, *, enable_sync=True):
        now = _now_text()
        conn.execute(
            """
            INSERT INTO chats(account_id, name, chat_type, remark, group_member_count, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_id, name) DO UPDATE SET
                chat_type=excluded.chat_type,
                remark=excluded.remark,
                group_member_count=excluded.group_member_count,
                status=excluded.status,
                updated_at=excluded.updated_at
            """,
            (
                account_id,
                chat_info["chat_name"],
                chat_info.get("chat_type"),
                chat_info.get("chat_remark"),
                chat_info.get("group_member_count"),
                "active",
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT id FROM chats WHERE account_id=? AND name=?", (account_id, chat_info["chat_name"])
        ).fetchone()
        chat_id = int(row["id"])
        self._ensure_chat_sync_config(conn, chat_id, enable=enable_sync)
        return chat_id

    def _ensure_chat_sync_config(self, conn, chat_id, *, enable=True):
        now = _now_text()
        conn.execute(
            """
            INSERT OR IGNORE INTO chat_sync_config(
                chat_id, enabled, priority, sync_latest, backfill_history, created_at, updated_at
            )
            VALUES (?, ?, 0, 1, 1, ?, ?)
            """,
            (chat_id, int(bool(enable)), now, now),
        )
        if enable:
            conn.execute(
                """
                UPDATE chat_sync_config
                SET enabled=1, sync_latest=1, updated_at=?
                WHERE chat_id=?
                """,
                (now, chat_id),
            )

    def _insert_messages(self, conn, account_id, chat_id, records):
        inserted = 0
        for row in records:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO messages(
                    account_id, chat_id, direction, sender, sender_remark, message_type,
                    content, media_path, normalized_time, raw_time_label, raw_id,
                    raw_json, fingerprint, collected_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    account_id,
                    chat_id,
                    row["direction"],
                    row["sender"],
                    row["sender_remark"],
                    row["message_type"],
                    row["content"],
                    row["media_path"],
                    row["normalized_time"],
                    row["raw_time_label"],
                    row["raw_id"],
                    row["raw_json"],
                    row["fingerprint"],
                    row["collected_at"],
                ),
            )
            inserted += cursor.rowcount
        return inserted

    def _get_sync_state(self, conn, chat_id):
        row = conn.execute("SELECT * FROM sync_state WHERE chat_id=?", (chat_id,)).fetchone()
        return dict(row) if row else None

    def _update_sync_state(
            self,
            conn,
            chat_id,
            *,
            oldest_fingerprint=None,
            newest_fingerprint=None,
            loaded_count=None,
            scroll_count=None,
            reached_top=None,
            last_error=None,
            sync_latest=False,
            sync_history=False,
            sync_success=False):
        previous = self._get_sync_state(conn, chat_id) or {}
        now = _now_text()
        values = {
            "oldest_fingerprint": oldest_fingerprint or previous.get("oldest_fingerprint"),
            "newest_fingerprint": newest_fingerprint or previous.get("newest_fingerprint"),
            "loaded_count": loaded_count if loaded_count is not None else previous.get("loaded_count", 0),
            "scroll_count": scroll_count if scroll_count is not None else previous.get("scroll_count", 0),
            "reached_top": int(reached_top if reached_top is not None else previous.get("reached_top", 0)),
            "last_error": last_error,
            "last_incremental_at": now if sync_latest else previous.get("last_incremental_at"),
            "last_history_at": now if sync_history else previous.get("last_history_at"),
            "last_success_at": now if sync_success else previous.get("last_success_at"),
            "consecutive_failures": 0 if sync_success else previous.get("consecutive_failures", 0),
            "next_due_at": None if sync_success else previous.get("next_due_at"),
            "updated_at": now,
        }
        conn.execute(
            """
            INSERT INTO sync_state(
                chat_id, oldest_fingerprint, newest_fingerprint, loaded_count,
                scroll_count, reached_top, last_error, last_incremental_at,
                last_history_at, last_success_at, consecutive_failures, next_due_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                oldest_fingerprint=excluded.oldest_fingerprint,
                newest_fingerprint=excluded.newest_fingerprint,
                loaded_count=excluded.loaded_count,
                scroll_count=excluded.scroll_count,
                reached_top=excluded.reached_top,
                last_error=excluded.last_error,
                last_incremental_at=excluded.last_incremental_at,
                last_history_at=excluded.last_history_at,
                last_success_at=excluded.last_success_at,
                consecutive_failures=excluded.consecutive_failures,
                next_due_at=excluded.next_due_at,
                updated_at=excluded.updated_at
            """,
            (
                chat_id,
                values["oldest_fingerprint"],
                values["newest_fingerprint"],
                values["loaded_count"],
                values["scroll_count"],
                values["reached_top"],
                values["last_error"],
                values["last_incremental_at"],
                values["last_history_at"],
                values["last_success_at"],
                values["consecutive_failures"],
                values["next_due_at"],
                values["updated_at"],
            ),
        )

    def _record_sync_failure(self, conn, chat_id, error):
        previous = self._get_sync_state(conn, chat_id) or {}
        failures = _as_int(previous.get("consecutive_failures"), 0) + 1
        delay_minutes = min(24 * 60, 5 * (2 ** min(failures - 1, 6)))
        next_due_at = (datetime.now() + timedelta(minutes=delay_minutes)).strftime("%Y-%m-%d %H:%M:%S")
        self._update_sync_state(
            conn,
            chat_id,
            last_error=error,
            sync_success=False,
        )
        conn.execute(
            """
            UPDATE sync_state
            SET consecutive_failures=?, next_due_at=?, last_error=?, updated_at=?
            WHERE chat_id=?
            """,
            (failures, next_due_at, error, _now_text(), chat_id),
        )


class _NullWechatContext:
    def __init__(self, wx):
        self.wx = wx

    def __enter__(self):
        return self.wx

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Archive one WeChat chat through the logged-in GUI.")
    parser.add_argument("db_path")
    parser.add_argument("chat_name")
    parser.add_argument("--full", action="store_true", help="scroll to the top of the chat history")
    parser.add_argument("--max-scrolls", type=int, default=0)
    parser.add_argument("--not-exact", action="store_true", help="allow fuzzy chat match")
    parser.add_argument("--save-media", action="store_true")
    args = parser.parse_args(argv)

    archive = WeChatArchive(args.db_path)
    if args.full:
        result = archive.full_chat(args.chat_name, exact=not args.not_exact, save_media=args.save_media)
    else:
        result = archive.pull_chat(
            args.chat_name,
            max_scrolls=args.max_scrolls,
            until="top" if args.max_scrolls else "loaded",
            exact=not args.not_exact,
            save_media=args.save_media,
        )
    print(_json_dumps(result))


if __name__ == "__main__":
    _main()
