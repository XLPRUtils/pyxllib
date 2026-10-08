import sqlite3
from types import SimpleNamespace

import pytest

from pyxllib.autogui import wechat_group_profile as profile
from pyxllib.autogui.wechat_db import WeChatDbStorage, WeChatDbError

ACCOUNT = "wxid_bot123"
ROOM = "12345@chatroom"


def field(number, value):
    raw = value.encode() if isinstance(value, str) else value
    return bytes([number * 8 + 2, len(raw)]) + raw


def storage(tmp_path, raw):
    directory = tmp_path / "contact"
    directory.mkdir()
    conn = sqlite3.connect(directory / "contact.db")
    conn.execute("CREATE TABLE chat_room(username TEXT,ext_buffer BLOB)")
    conn.execute("INSERT INTO chat_room VALUES (?,?)", (ROOM, raw))
    conn.commit()
    conn.close()
    return WeChatDbStorage(tmp_path)


def test_group_members_decode_and_exact_target(tmp_path):
    raw = field(1, field(1, ACCOUNT) + field(2, "管家") + b"\x18\x01")
    raw += field(1, field(1, "wxid_other")) + b"\x20\x01"
    source = storage(tmp_path, raw)
    assert source.group_members(ROOM) == [
        {"username": ACCOUNT, "display_name": "管家"},
        {"username": "wxid_other", "display_name": ""},
    ]
    with pytest.raises(WeChatDbError):
        source.group_members("1234@chatroom")


@pytest.mark.parametrize("raw", [b"\x0a\x20\x01", b"\x80", b"\0"])
def test_corrupt_member_data_fails_visibly(tmp_path, raw):
    with pytest.raises(WeChatDbError):
        storage(tmp_path, raw).group_members(ROOM)


@pytest.mark.parametrize("account,room,name", [
    ("other", ROOM, "管家"), (ACCOUNT, "未来社", "管家"),
    (ACCOUNT, ROOM, ""), (ACCOUNT, ROOM, "a\0b"),
])
def test_invalid_targets_never_access_account(monkeypatch, account, room, name):
    monkeypatch.setattr(profile, "get_account_storage", lambda *a: pytest.fail("invalid input reached account"))
    with pytest.raises(ValueError):
        profile.set_group_nickname(account, room, name)


def test_already_equal_does_not_attach_or_mutate(monkeypatch, tmp_path):
    monkeypatch.setattr(profile, "get_account_storage", lambda *a: SimpleNamespace(root=tmp_path / "db_storage"))
    monkeypatch.setattr(profile, "resolve_sender", lambda *a: {"pid": 1})
    monkeypatch.setattr(profile, "get_group_nickname", lambda *a: {"nickname": "管家"})
    monkeypatch.setattr(profile, "running_weixin_dll", lambda *a: pytest.fail("idempotent call loaded DLL"))
    assert profile.set_group_nickname(ACCOUNT, ROOM, "管家")["changed"] is False


def test_unknown_dll_fails_before_attach(monkeypatch, tmp_path):
    dll = tmp_path / "Weixin.dll"
    dll.write_bytes(b"unknown")
    monkeypatch.setattr(profile, "get_account_storage", lambda *a: SimpleNamespace(root=tmp_path / "db_storage"))
    monkeypatch.setattr(profile, "resolve_sender", lambda *a: {"pid": 1})
    monkeypatch.setattr(profile, "get_group_nickname", lambda *a: {"nickname": ""})
    monkeypatch.setattr(profile, "running_weixin_dll", lambda *a: dll)
    with pytest.raises(profile.WeixinInstrumentationUnavailable, match="未执行修改"):
        profile.set_group_nickname(ACCOUNT, ROOM, "管家")


def test_missing_self_is_not_an_empty_nickname(monkeypatch):
    monkeypatch.setattr(profile, "get_account_storage", lambda *a: SimpleNamespace(
        group_members=lambda *a: [{"username": "wxid_other", "display_name": "管家"}]))
    with pytest.raises(profile.WeixinInstrumentationUnavailable):
        profile.get_group_nickname(ACCOUNT, ROOM, refresh=False)
