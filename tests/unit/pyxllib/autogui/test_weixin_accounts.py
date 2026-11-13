from types import SimpleNamespace
import sys

import pytest

from pyxllib.autogui import weixin4_instrumentation as api
from pyxllib.autogui import wechat_accounts as accounts
from pyxllib.autogui.wechat_db import WeChatDbStorage, WeChatDbError


MAIN = {"account_id": api.DEFAULT_SENDER_ACCOUNT_ID, "pid": 10, "create_time": 1, "account_root": "main"}
SECOND = {"account_id": "wxid_second", "pid": 20, "create_time": 2, "account_root": "second"}


def test_check_account_preflights_the_selected_account(monkeypatch, tmp_path):
    archive = SimpleNamespace(root=tmp_path, status=lambda: {"ready": True})
    dll = tmp_path / "Weixin.dll"
    calls = []
    monkeypatch.setattr(accounts, "resolve_sender", lambda account_id: SECOND)
    monkeypatch.setattr(accounts, "get_account_storage", lambda account_id: archive)
    monkeypatch.setattr(accounts, "running_weixin_dll", lambda account_id: calls.append(("dll", account_id)) or dll)
    monkeypatch.setitem(sys.modules, "pyxllib.autogui.weixin4_instrumentation",
                        SimpleNamespace(resolve_recipient_id=lambda name, path: name, preflight=lambda path: {
                            "ok": path == dll, "dll": str(path)}))
    result = accounts.check_account(SECOND["account_id"], ["group"])
    assert result["layout"] == {"ok": True, "dll": str(dll)}
    assert calls == [("dll", SECOND["account_id"])]


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
def test_send_attaches_only_selected_account_and_rechecks(monkeypatch, change_account, tmp_path):
    calls = []
    # Before/after the process lock, before attach, immediately before send.
    snapshots = iter([[MAIN, SECOND]] * 3 + [[SECOND] if change_account else [MAIN, SECOND]])
    monkeypatch.setattr(api, "list_live_accounts", lambda: next(snapshots))
    # DLL identity is now validated before and after attaching. Keep those
    # probes local so the account snapshots test only the intended race.
    dll = tmp_path / 'Weixin.dll'
    monkeypatch.setattr(api, "running_weixin_dll", lambda account: dll)
    monkeypatch.setattr(api, "load_layout", lambda path: None)
    monkeypatch.setattr(api, "_ensure_native_adapter", lambda: None)
    monkeypatch.setattr(api, "_script_source", lambda *args: "")
    exports = SimpleNamespace(probe=lambda: {"pid": 10, "path": str(dll)},
                              sendtext=lambda *args: calls.append("send") or {"result": 1})
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


def test_account_changed_while_waiting_for_lock_never_attaches(monkeypatch):
    snapshots = iter([[MAIN, SECOND], [SECOND]])
    monkeypatch.setattr(api, 'list_live_accounts', lambda: next(snapshots))
    monkeypatch.setattr(api, 'running_weixin_dll', lambda *a: pytest.fail('must stop before native inspection'))
    monkeypatch.setitem(sys.modules, 'frida', SimpleNamespace(
        attach=lambda *a: pytest.fail('must not attach after identity changes')))
    with pytest.raises(api.WeixinInstrumentationUnavailable):
        api.send_text('filehelper', 'mock only')


def test_enumerate_accounts_by_open_paths(monkeypatch, tmp_path):
    root = tmp_path / "xwechat_files" / "wxid_main_ab12"
    process = SimpleNamespace(pid=10, name=lambda: "Weixin.exe", create_time=lambda: 1,
                              memory_maps=lambda: [SimpleNamespace(path=r"C:\Windows\System32\ntdll.dll")],
                              open_files=lambda: [SimpleNamespace(path=str(root / "db_storage" / "contact" / "contact.db")),
                                                  SimpleNamespace(path=str(root / "db_storage" / "session" / "session.db"))])
    monkeypatch.setattr(api.psutil, "process_iter", lambda: [process])
    assert api.list_live_accounts() == [{"account_id": "wxid_main", "account_root": str(root), "pid": 10, "create_time": 1}]


def test_mapped_db_paths_avoid_system_handle_enumeration(monkeypatch, tmp_path):
    root = tmp_path / "xwechat_files" / "wxid_main_ab12"

    def must_not_enumerate_handles():
        raise AssertionError("open_files must not run when mapped db paths already prove the account")

    process = SimpleNamespace(pid=10, name=lambda: "Weixin.exe", create_time=lambda: 1,
                              memory_maps=lambda: [SimpleNamespace(path=str(root / "db_storage" / "message" / "message_0.db"))],
                              open_files=must_not_enumerate_handles)
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
