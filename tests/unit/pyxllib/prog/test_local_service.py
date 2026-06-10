from __future__ import annotations

import json
from datetime import datetime

from pyxllib.prog.local_service import (
    acquire_json_lease,
    clear_stale_json_lease,
    is_json_lease_active,
    json_lease,
    local_service_enabled,
    owner_active_for_other_process,
    parse_datetime_text,
    read_json_lease,
    read_json_object_status,
    release_json_lease,
    seconds_since,
    should_enqueue_local_run,
    tail_text,
    write_json_command,
)


def test_read_json_object_status_reports_missing_active_and_stale(tmp_path):
    path = tmp_path / "owner.json"

    assert read_json_object_status(path)["active"] is False

    path.write_text(json.dumps({"pid": 1, "updated_at": 95.0}), encoding="utf-8")

    active = read_json_object_status(path, stale_after_seconds=30.0, now=100.0)
    stale = read_json_object_status(path, stale_after_seconds=3.0, now=100.0)

    assert active["active"] is True
    assert active["age_seconds"] == 5.0
    assert stale["active"] is False
    assert stale["stale"] is True


def test_write_json_command_persists_command_file(tmp_path):
    path = tmp_path / "control.json"

    request = write_json_command(
        path,
        command="stop_current_task",
        request_id="request-id",
        created_at=123.0,
        entry_id="entry",
        reason="test",
    )

    assert request["path"] == str(path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "id": "request-id",
        "command": "stop_current_task",
        "entry_id": "entry",
        "reason": "test",
        "created_at": 123.0,
    }


def test_json_lease_lifecycle_and_token_ownership(tmp_path):
    path = tmp_path / "lease.json"

    token = acquire_json_lease(path, reason="test", ttl_seconds=5.0, token="token-1", now=100.0)

    status = read_json_lease(path, now=101.0)
    assert token == "token-1"
    assert status["active"] is True
    assert status["reason"] == "test"
    assert status["token"] == "token-1"

    release_json_lease(path, "other-token")
    assert path.exists()

    release_json_lease(path, "token-1")
    assert not path.exists()


def test_json_lease_active_check_removes_stale_file(tmp_path, monkeypatch):
    path = tmp_path / "lease.json"

    acquire_json_lease(path, reason="test", ttl_seconds=5.0, token="token-1", now=100.0)
    monkeypatch.setattr("pyxllib.prog.local_service.time.time", lambda: 106.0)

    assert is_json_lease_active(path) is False
    assert not path.exists()


def test_json_lease_context_and_clear_stale(tmp_path, monkeypatch):
    path = tmp_path / "lease.json"

    with json_lease(path, reason="script") as token:
        assert token
        assert read_json_lease(path)["reason"] == "script"

    assert not path.exists()

    acquire_json_lease(path, reason="test", ttl_seconds=5.0, token="token-1", now=100.0)
    monkeypatch.setattr("pyxllib.prog.local_service.time.time", lambda: 106.0)

    result = clear_stale_json_lease(path)

    assert result["cleared"] is True
    assert not path.exists()


def test_owner_active_for_other_process_handles_owner_diagnostics():
    assert owner_active_for_other_process({"active": False, "pid": 2}, current_pid=1) is False
    assert owner_active_for_other_process({"active": True, "pid": 1}, current_pid=1) is False
    assert owner_active_for_other_process({"active": True, "pid": 2}, current_pid=1) is True
    assert owner_active_for_other_process({"active": True, "pid": "bad"}, current_pid=1) is True


def test_should_enqueue_local_run_resolves_modes():
    assert should_enqueue_local_run("enqueue", owner_active_elsewhere=False) is True
    assert should_enqueue_local_run("direct", owner_active_elsewhere=True) is False
    assert should_enqueue_local_run("auto", owner_active_elsewhere=True) is True
    assert should_enqueue_local_run("auto", owner_active_elsewhere=False) is False

    try:
        should_enqueue_local_run("bad", owner_active_elsewhere=False)
    except ValueError as exc:
        assert "auto/direct/enqueue" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_local_service_enabled_resolves_runtime_services_first():
    names = {"fanxiu-behavior-tree", "fanxiu", "all"}

    assert local_service_enabled(service_names=names, runtime_services_text="other,fanxiu") is True
    assert local_service_enabled(
        service_names=names,
        runtime_services_text="",
        enabled_text="1",
        default_hosts={"current"},
        current_hostname="current",
    ) is False


def test_local_service_enabled_resolves_explicit_enabled_before_hosts():
    names = {"fanxiu"}

    assert local_service_enabled(service_names=names, enabled_text="1") is True
    assert local_service_enabled(
        service_names=names,
        enabled_text="disabled",
        default_hosts={"current"},
        current_hostname="current",
    ) is False


def test_local_service_enabled_falls_back_to_hosts():
    names = {"fanxiu"}

    assert local_service_enabled(service_names=names, hosts_text="Mi15, current", current_hostname="CURRENT") is True
    assert local_service_enabled(service_names=names, default_hosts={"mi15"}, current_hostname="other") is False


def test_parse_datetime_text_and_seconds_since():
    assert parse_datetime_text("2026-06-09 10:11:12") == datetime(2026, 6, 9, 10, 11, 12)
    assert parse_datetime_text("2026-06-09T10:11:12.456") == datetime(2026, 6, 9, 10, 11, 12)
    assert parse_datetime_text("bad") is None

    assert seconds_since("2026-06-09 10:11:12", now=datetime(2026, 6, 9, 10, 11, 15)) == 3
    assert seconds_since("2026-06-09 10:11:12", now=datetime(2026, 6, 9, 10, 11, 10)) == 0
    assert seconds_since("bad", now=datetime(2026, 6, 9, 10, 11, 10)) is None


def test_tail_text_reads_recent_non_empty_lines(tmp_path):
    path = tmp_path / "service.log"
    path.write_text("a\n\nb\nc\n", encoding="utf-8")

    assert tail_text(path, lines=2) == ["b", "c"]
    assert tail_text(path, lines=0) == ["a", "b", "c"]

    missing = tmp_path / "missing.log"
    assert tail_text(missing) == [f"{missing} 不存在"]
