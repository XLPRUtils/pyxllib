import sqlite3
from datetime import datetime, timedelta

import pytest
import zstandard

from pyxllib.autogui import wechat_sms


def _content(des: str) -> bytes:
    xml = f"<msg><appmsg><title>新短信提醒</title><des><![CDATA[{des}]]></des><type>5</type></appmsg></msg>"
    return zstandard.ZstdCompressor().compress(xml.encode("utf-8"))


def _sms(code_text: str, at: datetime) -> bytes:
    return _content(f"来电号码：{code_text}106900\n来电时间：{at:%Y-%m-%d %H:%M:%S}")


def _make_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE {wechat_sms.SMS_TABLE}(local_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "create_time INTEGER, message_content TEXT)")
    conn.executemany(f"INSERT INTO {wechat_sms.SMS_TABLE}(create_time, message_content) VALUES (?, ?)",
                     [(int(t.timestamp()), c) for t, c in rows])
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def now():
    return datetime.now().replace(microsecond=0)


@pytest.fixture
def fake_accounts(monkeypatch, tmp_path):
    dbs = {}
    monkeypatch.setattr(wechat_sms, "_account_sources",
                        lambda: [{"wxid": wxid, "db": path, "key_hex": "00", "mode": "raw"} for wxid, path in dbs.items()])
    monkeypatch.setattr(wechat_sms, "decrypt_biz_db", lambda source: source["db"])
    return dbs, tmp_path


@pytest.mark.parametrize("text, code", [
    ("验证码【123456】微信支付", "123456"),
    ("您的验证码是 8842，5分钟内有效", "8842"),
    ("Your CODE: 99887766", "99887766"),
    ("778899是您的动态码", "778899"),
    ("没有数字码", None),
])
def test_extract_sms_code(text, code):
    assert wechat_sms.extract_sms_code(text) == code


def test_parse_skips_non_sms_push():
    assert wechat_sms.parse_forwarded_sms(_content("欢迎关注，观看使用视频介绍")) is None
    sms = wechat_sms.parse_forwarded_sms(_content("来电号码：验证码【654321】10685\n来电时间：2026-10-21 20:50:08"))
    assert sms["code"] == "654321"
    assert sms["sms_time"] == datetime(2026, 10, 21, 20, 50, 8)
    assert sms["sender"] == "验证码【654321】10685"


def test_missing_table_is_empty(tmp_path):
    path = tmp_path / "empty.db"
    sqlite3.connect(path).close()
    assert wechat_sms.read_sms_table(path) == []


def test_list_merges_accounts_newest_first(fake_accounts, now):
    dbs, tmp = fake_accounts
    dbs["wxid_a"] = _make_db(tmp / "a.db", [(now - timedelta(minutes=5), _sms("验证码【111111】", now)),
                                            (now - timedelta(minutes=1), _content("视频介绍"))])
    dbs["wxid_b"] = _make_db(tmp / "b.db", [(now - timedelta(minutes=2), _sms("验证码【222222】", now))])
    items = wechat_sms.list_forwarded_sms()
    assert [(x["wxid"], x["code"]) for x in items] == [("wxid_b", "222222"), ("wxid_a", "111111")]
    assert [x["wxid"] for x in wechat_sms.list_forwarded_sms(accounts=["wxid_a"])] == ["wxid_a"]


def test_list_raises_when_no_account_readable(monkeypatch):
    monkeypatch.setattr(wechat_sms, "_account_sources", lambda: [{"wxid": "wxid_x", "db": None, "key_hex": None}])
    with pytest.raises(wechat_sms.WeChatSmsUnavailable):
        wechat_sms.list_forwarded_sms()


def test_wait_filters_after_and_contains(fake_accounts, now):
    dbs, tmp = fake_accounts
    dbs["wxid_a"] = _make_db(tmp / "a.db", [
        (now - timedelta(minutes=3), _sms("微信支付验证码【000001】", now)),  # 早于 after
        (now - timedelta(seconds=1), _sms("淘宝验证码【000002】", now)),  # 在 2 秒容差内，但不含 contains
        (now, _sms("微信支付验证码【000003】", now)),
        (now + timedelta(seconds=1), _content("来电号码：无码通知\n来电时间：2026-10-21 20:00:00")),
    ])
    assert wechat_sms.wait_sms_code(now, timeout=0, contains="微信支付")["code"] == "000003"
    assert wechat_sms.wait_sms_code(now, timeout=0)["code"] == "000003"
    assert wechat_sms.wait_sms_code(now - timedelta(seconds=1), timeout=0, contains="淘宝")["code"] == "000002"


def test_wait_times_out(fake_accounts, now):
    dbs, tmp = fake_accounts
    dbs["wxid_a"] = _make_db(tmp / "a.db", [(now - timedelta(minutes=3), _sms("验证码【000001】", now))])
    with pytest.raises(TimeoutError):
        wechat_sms.wait_sms_code(now, timeout=0.2, interval=0.05)


def test_wait_picks_up_new_message(fake_accounts, now, monkeypatch):
    dbs, tmp = fake_accounts
    path = dbs["wxid_a"] = _make_db(tmp / "a.db", [])
    real_sleep = wechat_sms.time.sleep

    def sleep_then_arrive(seconds):
        conn = sqlite3.connect(path)
        conn.execute(f"INSERT INTO {wechat_sms.SMS_TABLE}(create_time, message_content) VALUES (?, ?)",
                     (int(now.timestamp()), _sms("验证码【135790】", now)))
        conn.commit()
        conn.close()
        real_sleep(0)

    monkeypatch.setattr(wechat_sms.time, "sleep", sleep_then_arrive)
    assert wechat_sms.wait_sms_code(now, timeout=5, interval=0.01)["code"] == "135790"


def test_mark_rollback_journal(tmp_path):
    path = tmp_path / "wal.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t(x)")
    conn.close()
    assert path.read_bytes()[18:20] == b"\x02\x02"
    wechat_sms._mark_rollback_journal(path)
    assert path.read_bytes()[18:20] == b"\x01\x01"
    sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True).execute("SELECT * FROM t").fetchall()
