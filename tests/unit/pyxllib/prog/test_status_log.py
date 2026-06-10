from __future__ import annotations

from pyxllib.prog.status_log import (
    append_status_log,
    append_status_log_once,
    filter_status_logs,
    normalize_guard_items,
    status_live_empty,
    status_logs,
    trim_status_logs,
)


def test_append_status_log_adds_structured_item_and_updates_timestamp(monkeypatch):
    monkeypatch.setattr("pyxllib.prog.status_log.time.time", lambda: 123.0)
    status = {}

    item = append_status_log(
        status,
        "info",
        "queued",
        scope="manual_job",
        item_id="job",
        time_text="01:02:03",
    )

    assert item == {
        "time": "01:02:03",
        "kind": "info",
        "scope": "manual_job",
        "item_id": "job",
        "message": "queued",
    }
    assert status["updated_at"] == 123.0


def test_append_status_log_trims_and_filters_malformed_logs():
    status = {"logs": ["bad", {"kind": "old", "message": "1"}, {"kind": "old", "message": "2"}]}

    append_status_log(status, "info", "3", time_text="00", update_timestamp=False, limit=2)

    assert [item["message"] for item in status_logs(status)] == ["2", "3"]


def test_append_status_log_once_deduplicates_by_kind_and_message():
    status = {}

    assert append_status_log_once(status, "stop", "done", time_text="00") is True
    assert append_status_log_once(status, "stop", "done", time_text="01") is False
    assert append_status_log_once(status, "stop", "other", time_text="02") is True

    assert [item["message"] for item in status["logs"]] == ["done", "other"]
    assert status["logs"][0]["time"] == "00"


def test_trim_status_logs_keeps_recent_dict_items():
    status = {"logs": [{"id": 1}, "bad", {"id": 2}, {"id": 3}]}

    assert trim_status_logs(status, limit=2) == [{"id": 2}, {"id": 3}]


def test_filter_status_logs_by_scope_item_and_limit():
    status = {
        "logs": [
            {"message": "bad"},
            {"scope": "manual", "item_id": "1", "message": "one"},
            {"scope": "manual", "item_id": "2", "message": "two"},
            {"scope": "guard", "item_id": "1", "message": "three"},
        ]
    }

    assert [item["message"] for item in filter_status_logs(status, scope="manual")] == ["one", "two"]
    assert [item["message"] for item in filter_status_logs(status, item_id="1")] == ["one", "three"]
    assert [item["message"] for item in filter_status_logs(status, scope="manual", limit=1)] == ["two"]


def test_status_live_empty_checks_running_task_text_logs_and_started_flag():
    assert status_live_empty({"running": False, "status": "idle"}) is True
    assert status_live_empty({"running": True, "status": "idle"}) is False
    assert status_live_empty({"running": False, "status": "running"}) is False
    assert status_live_empty({"running": False, "status": "idle", "task_type": "go_scene"}) is False
    assert status_live_empty({"running": False, "status": "idle", "current_task": "到场景"}) is False
    assert status_live_empty({"running": False, "status": "idle", "logs": [{"message": "x"}]}) is False
    assert status_live_empty({"running": False, "status": "idle", "started_at": 123}) is False


def test_normalize_guard_items_merges_definitions_raw_state_and_overrides():
    definitions = {
        "close_popups": {"label": "关闭弹窗", "message": "默认"},
        "invite": {"label": "邀请", "message": "等待"},
    }
    raw_items = {
        "close_popups": {"enabled": False, "running": False, "entry_id": "old", "updated_at": 2, "message": "旧消息"},
        "invite": {"enabled": True, "running": False, "entry_id": "entry", "updated_at": 3},
    }

    normalized = normalize_guard_items(
        definitions,
        raw_items,
        overrides={
            "close_popups": {"enabled": True, "running": True, "entry_id": "new", "message": "弹窗标题"},
        },
    )

    assert normalized["close_popups"] == {
        "label": "关闭弹窗",
        "message": "弹窗标题",
        "enabled": True,
        "running": True,
        "entry_id": "new",
        "updated_at": 2.0,
    }
    assert normalized["invite"] == {
        "label": "邀请",
        "message": "等待",
        "enabled": True,
        "running": False,
        "entry_id": "entry",
        "updated_at": 3.0,
    }
