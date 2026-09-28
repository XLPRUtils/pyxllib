from types import SimpleNamespace
import sys

import pytest

from pyxllib.autogui import weixin4_instrumentation as api
from pyxllib.autogui.wechat_db import WeChatDbStorage, WeChatDbError


MAIN = {"account_id": api.DEFAULT_SENDER_ACCOUNT_ID, "pid": 10, "create_time": 1, "account_root": "main"}
SECOND = {"account_id": "wxid_second", "pid": 20, "create_time": 2, "account_root": "second"}


def test_select_main_with_two_accounts(monkeypatch):
    monkeypatch.setattr(api, "list_live_accounts", lambda: [SECOND, MAIN])
    assert api.resolve_sender(api.DEFAULT_SENDER_ACCOUNT_ID) == MAIN
    assert api.resolve_sender("wxid_second") == SECOND


@pytest.mark.parametrize("accounts", [[SECOND], [MAIN, MAIN], [MAIN, dict(SECOND, pid=10)]])
def test_refuse_offline_duplicate_or_ambiguous_main(monkeypatch, accounts):
    monkeypatch.setattr(api, "list_live_accounts", lambda: accounts)
    with pytest.raises(api.WeixinInstrumentationUnavailable):
        api.resolve_sender(api.DEFAULT_SENDER_ACCOUNT_ID)


def test_reject_contact_snapshot_from_other_account(monkeypatch):
    monkeypatch.setattr(api, "list_live_accounts", lambda: [MAIN, SECOND])
    monkeypatch.setattr(api, "contact_account_id", lambda _: MAIN["account_id"])
    with pytest.raises(api.WeixinInstrumentationUnavailable, match="联系人快照"):
        api.send_text("someone", "never sent", sender_account_id=SECOND["account_id"], contact_db="main.db")


@pytest.mark.parametrize("change_account", [False, True])
def test_send_attaches_only_selected_account_and_rechecks(monkeypatch, change_account):
    calls = []
    snapshots = iter([[MAIN, SECOND], [SECOND] if change_account else [MAIN, SECOND]])
    monkeypatch.setattr(api, "list_live_accounts", lambda: next(snapshots))
    monkeypatch.setattr(api, "load_layout", lambda: None)
    monkeypatch.setattr(api, "_ensure_native_adapter", lambda: None)
    monkeypatch.setattr(api, "_script_source", lambda *args: "")
    exports = SimpleNamespace(probe=lambda: {"pid": 10}, sendtext=lambda *args: calls.append("send") or {"result": 1})
    script = SimpleNamespace(on=lambda *args: None, load=lambda: None, exports_sync=exports)
    session = SimpleNamespace(create_script=lambda _: script, detach=lambda: calls.append("detach"))
    processes = [SimpleNamespace(pid=20, name="Weixin.exe"), SimpleNamespace(pid=10, name="Weixin.exe")]
    fake = SimpleNamespace(get_local_device=lambda: SimpleNamespace(enumerate_processes=lambda: processes),
                           attach=lambda pid: calls.append(pid) or session)
    monkeypatch.setitem(sys.modules, "frida", fake)
    if change_account:
        with pytest.raises(api.WeixinInstrumentationError):
            api.send_text("filehelper", "never sent")
        assert calls == [10, "detach"]
    else:
        assert api.send_text("filehelper", "mock only")["sender"] == MAIN
        assert calls == [10, "send", "detach"]


def test_missing_bound_archive_never_rebinds(monkeypatch, tmp_path):
    storage = WeChatDbStorage(tmp_path / "decrypted" / "db_storage")
    monkeypatch.setattr(storage, "_load_sync_state", lambda: {"live_account_root": str(tmp_path / "missing")})
    with pytest.raises(WeChatDbError, match="拒绝自动切换"):
        storage.sync_from_live(export_media=False)


def test_enumerate_accounts_by_open_paths(monkeypatch, tmp_path):
    root = tmp_path / "xwechat_files" / "wxid_main_ab12"
    process = SimpleNamespace(pid=10, name=lambda: "Weixin.exe", create_time=lambda: 1,
                              open_files=lambda: [SimpleNamespace(path=str(root / "db_storage" / "contact" / "contact.db")),
                                                  SimpleNamespace(path=str(root / "db_storage" / "session" / "session.db"))])
    monkeypatch.setattr(api.psutil, "process_iter", lambda: [process])
    assert api.list_live_accounts() == [{"account_id": "wxid_main", "account_root": str(root), "pid": 10, "create_time": 1}]


def test_enumerate_accounts_when_deleted_handle_stat_is_denied(monkeypatch, tmp_path):
    root = tmp_path / "xwechat_files" / "wxid_second_ab12"

    def denied():
        raise api.psutil.AccessDenied(pid=20)

    process = SimpleNamespace(
        pid=20, name=lambda: "Weixin.exe", create_time=lambda: 2,
        open_files=denied,
        memory_maps=lambda: [SimpleNamespace(path=str(root / "db_storage" / "contact" / "contact.db-shm"))],
    )
    monkeypatch.setattr(api.psutil, "process_iter", lambda: [process])
    assert api.list_live_accounts() == [
        {"account_id": "wxid_second", "account_root": str(root), "pid": 20, "create_time": 2}
    ]
    process.memory_maps = denied
    assert api.list_live_accounts() == []
