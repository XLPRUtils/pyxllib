from types import SimpleNamespace

import pytest
from PIL import Image

from pyxllib.autogui.weixin4 import Weixin4Error, Weixin4TextClient


def bare_client():
    client = object.__new__(Weixin4TextClient)
    client.hwnd = 1
    return client


def test_search_result_clicks_exact_conversation(monkeypatch):
    client = bare_client()
    image = Image.new("RGB", (400, 500), "white")
    clicks = []
    monkeypatch.setattr(client, "_find_search_popup", lambda: 2)
    monkeypatch.setattr(client, "_capture", lambda *a, **k: image)
    monkeypatch.setattr(
        client,
        "_ocr_payload",
        lambda _: {
            "rec_texts": ["考勤后台管理系统", "考勒后台"],
            "rec_boxes": [[20, 40, 180, 70], [30, 210, 150, 250]],
        },
    )
    monkeypatch.setattr(
        client,
        "_click_relative",
        lambda x, y, **kwargs: clicks.append((x, y, kwargs)),
    )

    client._select_search_result("考勤后台")

    assert clicks == [(pytest.approx(0.225), pytest.approx(0.46), {"hwnd": 2})]


def test_chatwith_uses_alias_and_never_presses_enter(monkeypatch):
    client = bare_client()
    chats = iter(["文件传输助手", "考勤后台(3)"])
    selected = []
    hotkeys = []
    fake_gui = SimpleNamespace(
        hotkey=lambda *keys: hotkeys.append(keys),
        press=lambda key: pytest.fail(f"ChatWith must not press {key!r}"),
    )
    clipboard = {"value": "old"}
    fake_clipboard = SimpleNamespace(
        paste=lambda: clipboard["value"],
        copy=lambda value: clipboard.__setitem__("value", value),
    )
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", fake_gui)
    monkeypatch.setitem(__import__("sys").modules, "pyperclip", fake_clipboard)
    monkeypatch.setattr("pyxllib.autogui.weixin4.time.sleep", lambda _: None)
    monkeypatch.setattr(client, "_show", lambda: None)
    monkeypatch.setattr(client, "_current_chat", lambda: next(chats))
    monkeypatch.setattr(client, "_click_relative", lambda *a, **k: None)
    monkeypatch.setattr(client, "_select_search_result", selected.append)

    assert client.ChatWith("考勤中台") == "考勤中台"
    assert selected == ["考勤后台"]
    assert hotkeys == [("ctrl", "a"), ("ctrl", "v")]
    assert clipboard["value"] == "old"


def test_send_failure_removes_inserted_draft(monkeypatch):
    client = bare_client()
    chats = iter(["文件传输助手", "别的群"])
    pressed = []
    fake_gui = SimpleNamespace(
        hotkey=lambda *keys: pressed.append(keys),
        press=lambda key: pressed.append((key,)),
    )
    clipboard = {"value": "old"}
    fake_clipboard = SimpleNamespace(
        paste=lambda: clipboard["value"],
        copy=lambda value: clipboard.__setitem__("value", value),
    )
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", fake_gui)
    monkeypatch.setitem(__import__("sys").modules, "pyperclip", fake_clipboard)
    monkeypatch.setattr("pyxllib.autogui.weixin4.time.sleep", lambda _: None)
    monkeypatch.setattr(client, "_show", lambda: None)
    monkeypatch.setattr(client, "_current_chat", lambda: next(chats))
    monkeypatch.setattr(client, "_click_relative", lambda *a, **k: None)

    with pytest.raises(Weixin4Error, match="落键前目标校验失败"):
        client.SendMsg("真正正文", user="文件传输助手")

    assert pressed.count(("backspace",)) >= 2
    assert clipboard["value"] == "old"
