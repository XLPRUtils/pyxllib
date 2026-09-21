from types import SimpleNamespace
import pytest
from kq5034 import wechat
from pyxllib.autogui import wxautolib


def test_attendance_sender_is_fixed(monkeypatch):
    calls = []
    monkeypatch.setattr("pyxllib.autogui.weixin4_instrumentation.send_text", lambda *args, **kwargs: calls.append(kwargs))
    wechat.wechat_lock_send("考勤中台", "mock only")
    assert calls == [{"sender_account_id": wechat.ATTENDANCE_WECHAT_ACCOUNT_ID}]
    with pytest.raises(ValueError):
        wechat.wechat_lock_send("考勤中台", "mock only", sender_account_id="main")


def test_attendance_logger_keeps_account_and_general_logger_keeps_default(monkeypatch):
    calls = []
    monkeypatch.setattr(wxautolib, "wechat_lock_send", lambda *args, **kwargs: calls.append(kwargs))
    wxautolib.wechat_handler(SimpleNamespace(record={"extra": {"wechat_user": "filehelper", "wechat_sender_account_id": wechat.ATTENDANCE_WECHAT_ACCOUNT_ID}}))
    wxautolib.wechat_handler(SimpleNamespace(record={"extra": {"wechat_user": "filehelper"}}))
    assert calls == [{"sender_account_id": wechat.ATTENDANCE_WECHAT_ACCOUNT_ID}, {}]
