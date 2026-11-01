import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyxllib.autogui import weixin4_instrumentation
from pyxllib.autogui.wechat_db import message_table_name
from pyxllib.autogui.wechat_updates import poll_storage_updates

CHAT = "44577659873@chatroom"
REVOKE = ('<?xml version="1.0"?><sysmsg type="revokemsg"><revokemsg><content>你撤回了一条消息</content>'
          '<revoketime>1793375951</revoketime></revokemsg></sysmsg>')


class Storage:
    def __init__(self, root: Path):
        self.root = root / "db_storage"
        (self.root / "message").mkdir(parents=True)
        self.paths = SimpleNamespace(message=self.root / "message" / "message_0.db")
        self.table = message_table_name(CHAT)
        with sqlite3.connect(self.paths.message) as conn:
            conn.execute("CREATE TABLE Name2Id(user_name TEXT)")
            conn.executemany("INSERT INTO Name2Id VALUES(?)", [(CHAT,), ("wxid_me",)])
            conn.execute(f'CREATE TABLE "{self.table}"(local_id INTEGER PRIMARY KEY, server_id INTEGER, local_type INTEGER, '
                         'real_sender_id INTEGER, create_time INTEGER, source TEXT, message_content TEXT, compress_content BLOB)')

    def write(self, sql, *args):
        with sqlite3.connect(self.paths.message) as conn:
            conn.execute(sql.format(T=f'"{self.table}"'), args)
        # Mimic a new WAL generation landing in the snapshot.
        stat = self.paths.message.stat()
        os.utime(self.paths.message, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    def add(self, local_id, text, local_type=1):
        self.write("INSERT INTO {T} VALUES(?,?,?,?,?,?,?,?)", local_id, 900 + local_id, local_type, 2,
                   2_000_000_000, "", text, None)

    def recall(self, local_id):
        self.write("UPDATE {T} SET local_type=10000, message_content=? WHERE local_id=?", REVOKE, local_id)

    def status(self):
        return {"live_account_root": str(self.root)}

    def sync_from_live(self, export_media=False):
        return {"copy": {}, "decrypt": {}}

    def _contact_map(self, include_avatar=False):
        return {}


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(weixin4_instrumentation, "account_id_from_root", lambda root: "wxid_me")
    monkeypatch.setattr(weixin4_instrumentation, "resolve_sender",
                        lambda account: {"pid": os.getpid(), "create_time": 0, "account_root": str(tmp_path / "db_storage")})
    return Storage(tmp_path)


def poll(storage, cursor):
    return poll_storage_updates(storage, cursor)


def test_in_place_recall_is_reported_once_as_revision(storage):
    cursor = poll(storage, None)["cursor"]
    storage.add(54, "唯一码")
    storage.add(55, "撤回测试")
    batch = poll(storage, cursor)
    assert [(e["local_id"], e.get("revision")) for e in batch["events"]] == [(54, None), (55, None)]
    storage.recall(55)
    again = poll(storage, batch["cursor"])
    assert [(e["message_id"], e["message_type"], e.get("revision")) for e in again["events"]] == [
        (f"{CHAT}:55", 10000, "recalled")]
    assert "revokemsg" in again["events"][0]["text"]
    assert poll(storage, again["cursor"])["events"] == []  # 不重复上报
    assert poll(storage, batch["cursor"])["events"][0]["local_id"] == 55  # 旧游标重试仍可重放


def test_unchanged_shard_and_idle_chat_are_not_rescanned(storage):
    cursor = poll(storage, None)["cursor"]
    storage.add(1, "hi")
    cursor = poll(storage, cursor)["cursor"]
    cursor["recall"]["active"] = {}  # 超出撤回窗口的会话
    storage.recall(1)
    assert poll(storage, cursor)["events"] == []
    cursor = poll(storage, None)["cursor"]
    storage.add(2, "x")
    cursor = poll(storage, cursor)["cursor"]
    storage.recall(2)
    cursor["recall"]["shards"] = {"message_0.db": [storage.paths.message.stat().st_mtime_ns,
                                                   storage.paths.message.stat().st_size]}  # 签名未变的分片
    assert poll(storage, cursor)["events"] == []

def test_legacy_cursor_bootstraps_existing_recalls(storage):
    storage.add(55, REVOKE, local_type=10000)
    storage.add(56, "表情")
    cursor = poll(storage, None)["cursor"]
    cursor.pop("recall")
    events = poll(storage, cursor)["events"]
    assert [(e["local_id"], e["revision"]) for e in events] == [(55, "recalled")]


def test_recall_seen_as_new_row_is_not_repeated(storage):
    cursor = poll(storage, None)["cursor"]
    storage.add(3, REVOKE, local_type=10000)
    batch = poll(storage, cursor)
    assert [e.get("revision") for e in batch["events"]] == [None]
    storage.add(4, "next")
    assert [e["local_id"] for e in poll(storage, batch["cursor"])["events"]] == [4]
