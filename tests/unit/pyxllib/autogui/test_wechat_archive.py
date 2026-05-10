#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
from datetime import datetime, timedelta

from pyxllib.autogui.wechat_archive import WeChatArchive, normalize_wx_messages, parse_wechat_time


class FakeMsg:
    def __init__(self, msg_type, content, *, sender=None, sender_remark=None, msg_id=None, mtype="", time=None):
        self.type = msg_type
        self.content = content
        self.sender = sender
        self.sender_remark = sender_remark
        self.id = msg_id or "{}:{}".format(msg_type, content)
        self.mtype = mtype
        self.time = time
        self.info = [msg_type, content, self.id]


class FakeWx:
    VERSION = "test"
    nickname = "tester"

    def __init__(self, pages):
        self.pages = pages
        self.page_index = 0
        self.chat_with_calls = []

    def ChatWith(self, who, timeout=3, exact=False):
        self.chat_with_calls.append((who, timeout, exact))
        return who

    def CurrentChat(self, details=False):
        if details:
            return {
                "chat_name": "文件传输助手",
                "chat_type": "friend",
                "chat_remark": None,
                "group_member_count": None,
            }
        return "文件传输助手"

    def GetAllMessage(self, **kwargs):
        return self.pages[self.page_index]

    def LoadMoreMessage(self, interval=0.3):
        if self.page_index + 1 >= len(self.pages):
            return False
        self.page_index += 1
        return True


def test_parse_wechat_time_common_labels():
    now = datetime(2026, 5, 10, 16, 30, 0)

    assert parse_wechat_time("15:01", now=now) == "2026-05-10 15:01:00"
    assert parse_wechat_time("昨天 09:02", now=now) == "2026-05-09 09:02:00"
    assert parse_wechat_time("星期六 08:00", now=now) == "2026-05-09 08:00:00"
    assert parse_wechat_time("2025年12月3日 下午 3:04", now=now) == "2025-12-03 15:04:00"


def test_normalize_messages_propagates_time_and_types():
    now = datetime(2026, 5, 10, 16, 30, 0)
    rows = normalize_wx_messages(
        [
            FakeMsg("time", "15:01", time="15:01"),
            FakeMsg("self", "发给自己", sender="Self", msg_id="1"),
            FakeMsg("friend", "群消息", sender="张三", sender_remark="张三备注", msg_id="2"),
            FakeMsg("friend", "[图片]", sender="李四", msg_id="3"),
            FakeMsg("friend", "[文件] report.xlsx", sender="李四", msg_id="4"),
            FakeMsg("friend", "[链接]", sender="李四", msg_id="5"),
            FakeMsg("recall", "你撤回了一条消息", msg_id="6"),
        ],
        "文件传输助手",
        collected_at="2026-05-10 16:30:00",
        now=now,
    )

    assert rows[0]["message_type"] == "time"
    assert rows[1]["direction"] == "out"
    assert rows[1]["normalized_time"] == "2026-05-10 15:01:00"
    assert rows[2]["direction"] == "in"
    assert rows[2]["sender_remark"] == "张三备注"
    assert rows[3]["message_type"] == "image"
    assert rows[4]["message_type"] == "file"
    assert rows[5]["message_type"] == "link"
    assert rows[6]["message_type"] == "recall"


def test_sqlite_write_is_idempotent(tmp_path):
    db_path = tmp_path / "wechat.sqlite"
    messages = [
        FakeMsg("time", "15:01", time="15:01"),
        FakeMsg("self", "hello", sender="Self", msg_id="1"),
        FakeMsg("friend", "world", sender="文件传输助手", msg_id="2"),
    ]
    wx = FakeWx([messages])
    archive = WeChatArchive(db_path, wx=wx)

    first = archive.pull_chat("文件传输助手")
    second = archive.pull_chat("文件传输助手")

    assert first["inserted"] == 3
    assert second["inserted"] == 0
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0] == 1
    finally:
        conn.close()


def test_full_chat_loads_until_top(tmp_path):
    db_path = tmp_path / "wechat.sqlite"
    wx = FakeWx(
        [
            [
                FakeMsg("time", "15:02", time="15:02"),
                FakeMsg("self", "new", sender="Self", msg_id="new"),
            ],
            [
                FakeMsg("time", "14:00", time="14:00"),
                FakeMsg("friend", "old", sender="文件传输助手", msg_id="old"),
            ],
        ]
    )
    archive = WeChatArchive(db_path, wx=wx)

    result = archive.full_chat("文件传输助手")

    assert result["reached_top"] is True
    assert result["scroll_count"] == 2
    assert result["inserted"] == 4
    conn = sqlite3.connect(db_path)
    try:
        contents = [row[0] for row in conn.execute("SELECT content FROM messages ORDER BY id")]
    finally:
        conn.close()
    assert contents == ["15:02", "new", "14:00", "old"]


def test_sync_incremental_reads_tail_without_duplicates(tmp_path):
    db_path = tmp_path / "wechat.sqlite"
    base_messages = [
        FakeMsg("time", "15:01", time="15:01"),
        FakeMsg("self", "hello", sender="Self", msg_id="1"),
    ]
    wx = FakeWx([base_messages])
    archive = WeChatArchive(db_path, wx=wx)

    assert archive.pull_chat("文件传输助手")["inserted"] == 2
    wx.pages[0] = [
        *base_messages,
        FakeMsg("friend", "new tail", sender="文件传输助手", msg_id="2"),
    ]

    result = archive.sync_incremental(
        chat_names=["文件传输助手"],
        max_chats=1,
        max_scrolls_total=0,
        use_wechat_hints=False,
    )

    assert result["inserted"] == 1
    assert result["results"][0]["phase"] == "latest"
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
    finally:
        conn.close()


def test_sync_incremental_backfill_respects_scroll_budget(tmp_path):
    db_path = tmp_path / "wechat.sqlite"
    wx = FakeWx(
        [
            [
                FakeMsg("time", "15:02", time="15:02"),
                FakeMsg("self", "new", sender="Self", msg_id="new"),
            ],
            [
                FakeMsg("time", "14:00", time="14:00"),
                FakeMsg("friend", "old", sender="文件传输助手", msg_id="old"),
            ],
            [
                FakeMsg("time", "13:00", time="13:00"),
                FakeMsg("friend", "older", sender="文件传输助手", msg_id="older"),
            ],
        ]
    )
    archive = WeChatArchive(db_path, wx=wx)
    archive.pull_chat("文件传输助手")

    result = archive.sync_incremental(
        chat_names=["文件传输助手"],
        max_chats=1,
        max_scrolls_total=1,
        max_scrolls_per_chat=1,
        sync_latest=False,
        use_wechat_hints=False,
    )

    assert result["scroll_count"] == 1
    assert result["inserted"] == 2
    conn = sqlite3.connect(db_path)
    try:
        contents = [row[0] for row in conn.execute("SELECT content FROM messages ORDER BY id")]
    finally:
        conn.close()
    assert contents == ["15:02", "new", "14:00", "old"]


def test_plan_sync_chats_prioritizes_unread_and_skips_backoff(tmp_path):
    db_path = tmp_path / "wechat.sqlite"
    archive = WeChatArchive(db_path, wx=FakeWx([[]]))
    base = datetime(2026, 5, 10, 12, 0, 0)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO accounts(account_key, nickname, computer_name, wx_version, created_at, updated_at)
            VALUES ('a', 'a', 'pc', 'test', '2026-05-10 00:00:00', '2026-05-10 00:00:00')
            """
        )
        account_id = conn.execute("SELECT id FROM accounts WHERE account_key='a'").fetchone()[0]
        for name in ["A", "B", "C"]:
            conn.execute(
                """
                INSERT INTO chats(account_id, name, status, created_at, updated_at)
                VALUES (?, ?, 'active', '2026-05-10 00:00:00', '2026-05-10 00:00:00')
                """,
                (account_id, name),
            )
        rows = conn.execute("SELECT id, name FROM chats").fetchall()
        for chat_id, name in rows:
            conn.execute(
                """
                INSERT INTO chat_sync_config(chat_id, enabled, priority, sync_latest, backfill_history, created_at, updated_at)
                VALUES (?, 1, 0, 1, 1, '2026-05-10 00:00:00', '2026-05-10 00:00:00')
                """,
                (chat_id,),
            )
            conn.execute(
                """
                INSERT INTO sync_state(chat_id, reached_top, last_success_at, consecutive_failures, next_due_at, updated_at)
                VALUES (?, 0, ?, ?, ?, '2026-05-10 00:00:00')
                """,
                (
                    chat_id,
                    "2026-05-09 00:00:00",
                    2 if name == "C" else 0,
                    (base + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S") if name == "C" else None,
                ),
            )

    plan = archive.plan_sync_chats(
        recent_chat_names=["A", "B"],
        unread_chat_names={"B": 3},
        now=base,
    )

    assert [item["name"] for item in plan] == ["B", "A"]
    assert "unread:3" in plan[0]["reasons"]


def test_schema_migration_adds_incremental_tables(tmp_path):
    db_path = tmp_path / "old.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_key TEXT NOT NULL UNIQUE,
                nickname TEXT,
                computer_name TEXT,
                wx_version TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE chats (
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
                UNIQUE(account_id, name)
            );
            CREATE TABLE messages (
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
                collected_at TEXT NOT NULL
            );
            CREATE TABLE sync_state (
                chat_id INTEGER PRIMARY KEY,
                oldest_fingerprint TEXT,
                newest_fingerprint TEXT,
                loaded_count INTEGER NOT NULL DEFAULT 0,
                scroll_count INTEGER NOT NULL DEFAULT 0,
                reached_top INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at TEXT NOT NULL
            );
            INSERT INTO accounts(account_key, created_at, updated_at) VALUES ('a', 'x', 'x');
            INSERT INTO chats(account_id, name, created_at, updated_at) VALUES (1, '文件传输助手', 'x', 'x');
            INSERT INTO sync_state(chat_id, updated_at) VALUES (1, 'x');
            """
        )

    WeChatArchive(db_path, wx=FakeWx([[]]))

    conn = sqlite3.connect(db_path)
    try:
        sync_columns = {row[1] for row in conn.execute("PRAGMA table_info(sync_state)")}
        assert {"last_incremental_at", "last_history_at", "last_success_at", "consecutive_failures", "next_due_at"} <= sync_columns
        assert conn.execute("SELECT COUNT(*) FROM chat_sync_config").fetchone()[0] == 1
    finally:
        conn.close()
