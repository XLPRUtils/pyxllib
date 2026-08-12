import sqlite3

import pytest

from pyxllib.autogui.weixin4_instrumentation import WeixinInstrumentationError, resolve_recipient_id
from pyxllib.autogui.wxautolib import wechat_lock_send


def make_db(tmp_path, rows):
    path = tmp_path / "contact.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE contact(username TEXT, alias TEXT, remark TEXT, nick_name TEXT, delete_flag INTEGER)")
    conn.executemany("INSERT INTO contact VALUES (?, ?, ?, ?, 0)", rows)
    conn.commit()
    conn.close()
    return path


def test_resolve_business_groups_exactly(tmp_path):
    path = make_db(
        tmp_path,
        [
            ("52281119334@chatroom", "", "", "考勤中台"),
            ("51653518650@chatroom", "", "", "考勤后台"),
        ],
    )
    assert resolve_recipient_id("考勤中台", path) == "52281119334@chatroom"
    assert resolve_recipient_id("考勤后台", path) == "51653518650@chatroom"
    assert resolve_recipient_id("文件传输助手", path) == "filehelper"


def test_resolve_repairs_legacy_gbk_text(tmp_path):
    mojibake = "考勤后台".encode("gbk").decode("latin1")
    path = make_db(tmp_path, [("room@chatroom", "", "", mojibake)])
    assert resolve_recipient_id("考勤后台", path) == "room@chatroom"


def test_resolve_rejects_ambiguous_recipient(tmp_path):
    path = make_db(tmp_path, [("a", "", "", "考勤中台"), ("b", "", "考勤中台", "")])
    with pytest.raises(WeixinInstrumentationError, match="必须唯一匹配"):
        resolve_recipient_id("考勤中台", path)


def test_wechat_lock_send_prefers_instrumentation_for_plain_text(monkeypatch):
    sent = []
    monkeypatch.setattr(
        "pyxllib.autogui.weixin4_instrumentation.send_text",
        lambda recipient, text: sent.append((recipient, text)),
    )
    monkeypatch.setattr(
        "pyxllib.autogui.wxautolib.WeChatSingletonLock",
        lambda *args, **kwargs: pytest.fail("纯文本发送不应启动 GUI"),
    )

    wechat_lock_send("考勤中台", "原日报文案")

    assert sent == [("考勤中台", "原日报文案")]
