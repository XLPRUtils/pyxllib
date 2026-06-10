from __future__ import annotations

import time
from datetime import datetime
from typing import Any


__all__ = [
    "append_status_log",
    "append_status_log_once",
    "filter_status_logs",
    "normalize_guard_items",
    "status_live_empty",
    "status_logs",
    "trim_status_logs",
]


def status_logs(status: dict[str, Any]) -> list[dict[str, Any]]:
    logs = status.get("logs")
    if not isinstance(logs, list):
        return []
    return [item for item in logs if isinstance(item, dict)]


def trim_status_logs(status: dict[str, Any], *, limit: int = 500) -> list[dict[str, Any]]:
    logs = status_logs(status)[-max(1, int(limit)) :]
    status["logs"] = logs
    return logs


def filter_status_logs(
    status: dict[str, Any],
    *,
    limit: int = 500,
    scope: str = "",
    item_id: str = "",
) -> list[dict[str, Any]]:
    logs = status_logs(status)
    resolved_scope = str(scope or "").strip()
    resolved_item_id = str(item_id or "").strip()
    if resolved_scope:
        logs = [item for item in logs if str(item.get("scope") or "") == resolved_scope]
    if resolved_item_id:
        logs = [item for item in logs if str(item.get("item_id") or "") == resolved_item_id]
    return logs[-max(1, int(limit)) :]


def status_live_empty(
    status: dict[str, Any],
    *,
    running_key: str = "running",
    status_key: str = "status",
    idle_status: str = "idle",
    text_keys: tuple[str, ...] = ("task_type", "current_task"),
    logs_key: str = "logs",
    started_key: str = "started_at",
) -> bool:
    """Check whether a runtime status still has no live task data."""

    return (
        not bool(status.get(running_key))
        and str(status.get(status_key) or idle_status) == idle_status
        and all(not str(status.get(key) or "") for key in text_keys)
        and not status.get(logs_key)
        and not status.get(started_key)
    )


def normalize_guard_items(
    guard_definitions: dict[str, dict[str, Any]],
    raw_items: Any,
    *,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Merge guard definitions, persisted item state, and runtime overrides."""

    source = raw_items if isinstance(raw_items, dict) else {}
    resolved_overrides = overrides if isinstance(overrides, dict) else {}
    normalized: dict[str, dict[str, Any]] = {}
    for guard_id, definition in guard_definitions.items():
        raw_item = source.get(guard_id)
        if not isinstance(raw_item, dict):
            raw_item = {}
        override = resolved_overrides.get(guard_id)
        if not isinstance(override, dict):
            override = {}
        enabled = bool(override.get("enabled", raw_item.get("enabled")))
        running = bool(override.get("running", raw_item.get("running")))
        entry_id = str(override.get("entry_id", raw_item.get("entry_id") or ""))
        message = str(override.get("message", raw_item.get("message") or definition.get("message") or ""))
        normalized[guard_id] = {
            **definition,
            "enabled": enabled,
            "running": running,
            "entry_id": entry_id,
            "updated_at": float(override.get("updated_at", raw_item.get("updated_at") or 0)),
            "message": message,
        }
    return normalized


def append_status_log(
    status: dict[str, Any],
    kind: str,
    message: str,
    *,
    scope: str = "",
    item_id: str = "",
    time_text: str | None = None,
    updated_at: float | None = None,
    update_timestamp: bool = True,
    limit: int = 500,
) -> dict[str, Any]:
    logs = status_logs(status)
    logs.append({
        "time": time_text or datetime.now().strftime("%H:%M:%S"),
        "kind": kind,
        "scope": scope,
        "item_id": item_id,
        "message": message,
    })
    status["logs"] = logs[-max(1, int(limit)) :]
    if update_timestamp:
        status["updated_at"] = time.time() if updated_at is None else updated_at
    return status["logs"][-1]


def append_status_log_once(
    status: dict[str, Any],
    kind: str,
    message: str,
    *,
    time_text: str | None = None,
    limit: int = 500,
) -> bool:
    logs = status_logs(status)
    if any(item.get("kind") == kind and item.get("message") == message for item in logs):
        status["logs"] = logs[-max(1, int(limit)) :]
        return False
    logs.append({
        "time": time_text or datetime.now().strftime("%H:%M:%S"),
        "kind": kind,
        "message": message,
    })
    status["logs"] = logs[-max(1, int(limit)) :]
    return True
