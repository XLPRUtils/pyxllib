import sqlite3

import pytest

from pyxllib.autogui import weixin4_instrumentation
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


def test_wechat_lock_send_uses_api_for_plain_text(monkeypatch):
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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"files": ["report.xlsx"]},
        {"url": "https://example.com"},
        {"text": "日报", "at": "所有人"},
        {"text": None},
    ],
)
def test_wechat_lock_send_rejects_unsupported_types_without_gui(monkeypatch, kwargs):
    monkeypatch.setattr(
        "pyxllib.autogui.wxautolib.WeChatSingletonLock",
        lambda *args, **kwargs: pytest.fail("不支持的消息类型也不允许启动 GUI"),
    )

    with pytest.raises(NotImplementedError, match="禁止降级 GUI"):
        wechat_lock_send("考勤中台", **kwargs)


def test_wechat_lock_send_propagates_api_failure_without_gui(monkeypatch):
    monkeypatch.setattr(
        "pyxllib.autogui.weixin4_instrumentation.send_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("api unavailable")),
    )
    monkeypatch.setattr(
        "pyxllib.autogui.wxautolib.WeChatSingletonLock",
        lambda *args, **kwargs: pytest.fail("API 失败时不允许启动 GUI"),
    )

    with pytest.raises(RuntimeError, match="api unavailable"):
        wechat_lock_send("考勤中台", "日报")


def test_native_script_loads_version_pinned_adapter(tmp_path):
    adapter = tmp_path / "native adapter.dll"

    source = weixin4_instrumentation._script_source(adapter)

    assert f'Module.load("{str(adapter).replace(chr(92), chr(92) * 2)}")' in source
    assert "api-only-native-coroutine" in source
    assert "native adapter failed" in source


def test_native_adapter_reuses_current_build(monkeypatch, tmp_path):
    source = tmp_path / "adapter.c"
    adapter = tmp_path / "adapter.dll"
    source.write_text("source", encoding="utf-8")
    adapter.write_bytes(b"dll")
    monkeypatch.setattr(weixin4_instrumentation, "NATIVE_SOURCE", source)
    monkeypatch.setattr(weixin4_instrumentation, "NATIVE_ADAPTER", adapter)

    assert weixin4_instrumentation._ensure_native_adapter() == adapter
