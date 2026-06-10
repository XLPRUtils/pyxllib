from __future__ import annotations

import time
from typing import Any, MutableMapping


__all__ = [
    "append_fact_event",
    "ensure_mapping_bucket",
    "fact_key",
    "trim_fact_events",
]


def fact_key(prefix: str, *parts: Any) -> str:
    """Build a stable colon-separated fact key from non-empty parts."""

    text = ":".join(str(part or "").strip() for part in parts if str(part or "").strip())
    return f"{prefix}:{text}" if text else str(prefix)


def ensure_mapping_bucket(root: MutableMapping[str, Any], *keys: str) -> dict[str, Any]:
    """Return a nested dict bucket, replacing malformed values on the path."""

    current: MutableMapping[str, Any] = root
    for key in keys:
        value = current.get(key)
        if not isinstance(value, dict):
            value = {}
            current[key] = value
        current = value
    return dict(current) if not isinstance(current, dict) else current


def trim_fact_events(
    facts: MutableMapping[str, Any],
    *,
    key: str = "events",
    limit: int = 200,
) -> None:
    """Keep only dict events and cap the event list length in-place."""

    events = facts.get(key)
    if isinstance(events, list):
        facts[key] = [item for item in events if isinstance(item, dict)][-max(0, int(limit)) :]


def append_fact_event(
    facts: MutableMapping[str, Any],
    kind: str,
    payload: dict[str, Any],
    *,
    key: str = "events",
    time_key: str = "time",
    now: float | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Append a timestamped event to a JSON facts object."""

    event = {**payload, time_key: time.time() if now is None else now, "kind": str(kind)}
    events = facts.setdefault(key, [])
    if not isinstance(events, list):
        events = []
        facts[key] = events
    events.append(event)
    if limit is not None:
        trim_fact_events(facts, key=key, limit=limit)
    return event
