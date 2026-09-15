import sqlite3
from pathlib import Path

import pytest

from pyxllib.autogui import weixin4_instrumentation, weixin4_offsets
from pyxllib.autogui.weixin4_instrumentation import WeixinInstrumentationError, resolve_recipient_id
from pyxllib.autogui.wxautolib import wechat_lock_send

WEIXIN_DLL = Path(r"C:\Program Files\Tencent\Weixin\4.1.13.65\Weixin.dll")
WEIXIN_BYTES = WEIXIN_DLL.read_bytes() if WEIXIN_DLL.exists() else None
requires_weixin = pytest.mark.skipif(WEIXIN_BYTES is None, reason="缺少 4.1.13.65 Weixin.dll")


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
    layout = weixin4_offsets.WeixinLayout(
        sha256="test",
        version="9.9.9",
        offsets={
            "get_coro": 0x10,
            "get_service": 0x20,
            "get_context": 0x30,
            "message_ctor": 0x40,
            "do_send": 0x50,
            "send_entry": 0x60,
        },
        struct={"content": 0x758, "object_size": 0x1000},
        source="pinned",
    )

    source = weixin4_instrumentation._script_source(adapter, layout)

    assert f'Module.load("{str(adapter).replace(chr(92), chr(92) * 2)}")' in source
    assert "api-only-native-coroutine" in source
    assert "native adapter failed" in source
    assert '"content": 1880' in source
    assert f'"get_coro": {0x10}' in source


@requires_weixin
def test_preflight_reports_pinned_layout():
    result = weixin4_instrumentation.preflight()

    assert result["ok"] is True
    assert result["source"] == "pinned"
    assert result["offsets"]["get_service"] == hex(0x3426C0)


@requires_weixin
def test_resolve_matches_pinned_offsets():
    layout = weixin4_offsets.resolve(WEIXIN_BYTES)

    assert layout.source == "pinned"
    assert layout.version == "4.1.13.65"
    assert layout.offsets["get_service"] == 0x3426C0
    assert layout.struct["content"] == 0x758


@requires_weixin
def test_rebind_recovers_send_chain_from_live_image():
    pe = weixin4_offsets._PeImage(WEIXIN_BYTES)

    chain = weixin4_offsets._resolve_send_chain(pe)

    assert chain == {
        "get_coro": 0x43AB0,
        "get_service": 0x3426C0,
        "get_context": 0x6EC950,
        "do_send": 0x17A3620,
        "send_entry": 0x19D14C0,
    }


def test_resolve_fails_closed_on_unknown_image():
    with pytest.raises(weixin4_offsets.RebindError):
        weixin4_offsets.resolve(b"not a portable executable")


def test_native_adapter_reuses_current_build(monkeypatch, tmp_path):
    source = tmp_path / "adapter.c"
    adapter = tmp_path / "adapter.dll"
    source.write_text("source", encoding="utf-8")
    adapter.write_bytes(b"dll")
    monkeypatch.setattr(weixin4_instrumentation, "NATIVE_SOURCE", source)
    monkeypatch.setattr(weixin4_instrumentation, "NATIVE_ADAPTER", adapter)

    assert weixin4_instrumentation._ensure_native_adapter() == adapter
