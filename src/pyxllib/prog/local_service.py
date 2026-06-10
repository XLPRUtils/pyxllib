from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


__all__ = [
    "acquire_json_lease",
    "clear_stale_json_lease",
    "is_json_lease_active",
    "json_lease",
    "local_service_enabled",
    "owner_active_for_other_process",
    "parse_datetime_text",
    "read_json_lease",
    "read_json_object_status",
    "release_json_lease",
    "seconds_since",
    "should_enqueue_local_run",
    "tail_text",
    "write_json_command",
]


def read_json_object_status(
    path: Path,
    *,
    stale_after_seconds: float | None = None,
    updated_at_key: str = "updated_at",
    invalid_message: str = "文件不是 JSON object",
    now: float | None = None,
) -> dict[str, Any]:
    """Read a JSON object and add exists/active/stale diagnostics."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"exists": False, "active": False, "stale": False, "path": str(path)}
    except Exception as exc:
        return {"exists": True, "active": False, "stale": True, "path": str(path), "error": str(exc)}
    if not isinstance(payload, dict):
        return {"exists": True, "active": False, "stale": True, "path": str(path), "error": invalid_message}

    if stale_after_seconds is None:
        return {**payload, "exists": True, "active": True, "stale": False, "path": str(path)}

    updated_at = float(payload.get(updated_at_key) or 0.0)
    age_seconds = (time.time() if now is None else now) - updated_at if updated_at > 0 else None
    stale = updated_at <= 0 or (
        age_seconds is not None and age_seconds > max(1.0, float(stale_after_seconds or 0.0))
    )
    return {
        **payload,
        "exists": True,
        "active": not stale,
        "stale": stale,
        "age_seconds": age_seconds,
        "path": str(path),
    }


def write_json_command(
    path: Path,
    *,
    command: str,
    request_id: str | None = None,
    created_at: float | None = None,
    **payload: Any,
) -> dict[str, Any]:
    """Write a small JSON command file and return the written command payload."""

    request = {
        "id": request_id or uuid.uuid4().hex,
        "command": str(command or ""),
        **payload,
        "created_at": time.time() if created_at is None else created_at,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**request, "path": str(path)}


def owner_active_for_other_process(owner: dict[str, Any], *, current_pid: int | None = None) -> bool:
    """Return whether a live owner record belongs to another process."""

    if not bool(owner.get("active")):
        return False
    try:
        owner_pid = int(owner.get("pid") or 0)
    except (TypeError, ValueError):
        return True
    return owner_pid != (os.getpid() if current_pid is None else int(current_pid))


def should_enqueue_local_run(run_mode: str, *, owner_active_elsewhere: bool) -> bool:
    """Resolve auto/direct/enqueue for local task submission."""

    mode = str(run_mode or "auto").strip().lower()
    if mode == "enqueue":
        return True
    if mode == "direct":
        return False
    if mode != "auto":
        raise ValueError("run_mode only supports auto/direct/enqueue")
    return bool(owner_active_elsewhere)


def local_service_enabled(
    *,
    service_names: set[str] | list[str] | tuple[str, ...],
    runtime_services_text: str | None = None,
    enabled_text: str | None = None,
    hosts_text: str | None = None,
    default_hosts: set[str] | list[str] | tuple[str, ...] = (),
    current_hostname: str = "",
) -> bool:
    """Resolve whether a local service should be enabled from common env-style controls."""

    normalized_names = {str(item or "").strip().lower() for item in service_names if str(item or "").strip()}
    if runtime_services_text is not None:
        services = {item.strip().lower() for item in str(runtime_services_text).split(",") if item.strip()}
        return bool(services & normalized_names)

    if enabled_text is not None:
        return str(enabled_text).strip().lower() not in {"0", "false", "no", "off", "disabled"}

    hosts = (
        {item.strip().lower() for item in str(hosts_text).split(",") if item.strip()}
        if hosts_text
        else {str(item or "").strip().lower() for item in default_hosts if str(item or "").strip()}
    )
    return str(current_hostname or "").strip().lower() in hosts


def parse_datetime_text(value: Any, *, formats: tuple[str, ...] | None = None) -> datetime | None:
    """Parse common service timestamp text into a naive ``datetime``."""

    text = str(value or "").strip()
    if not text:
        return None
    for fmt in formats or ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            pass
    return None


def seconds_since(value: Any, *, now: datetime | None = None, formats: tuple[str, ...] | None = None) -> int | None:
    """Return elapsed seconds since a timestamp text, clamped at zero."""

    parsed = parse_datetime_text(value, formats=formats)
    if not parsed:
        return None
    current_time = now or datetime.now()
    return max(0, int((current_time - parsed).total_seconds()))


def tail_text(path: Path, *, lines: int = 80, max_bytes: int = 512 * 1024, encoding: str = "utf-8") -> list[str]:
    """Read the non-empty tail lines of a text file without loading large files fully."""

    if not path.exists():
        return [f"{path} 不存在"]
    try:
        size = path.stat().st_size
        with path.open("rb") as file:
            file.seek(max(0, size - max_bytes))
            data = file.read()
    except OSError as exc:
        return [f"读取 {path} 失败：{exc}"]
    text = data.decode(encoding, errors="replace")
    rows = [row.rstrip() for row in text.splitlines() if row.strip()]
    return rows[-lines:] if lines > 0 else rows


def read_json_lease(
    path: Path,
    *,
    ttl_seconds: float | None = None,
    ttl_key: str = "ttl_seconds",
    min_ttl_seconds: float = 5.0,
    now: float | None = None,
    invalid_message: str = "lease 文件不是 JSON object",
) -> dict[str, Any]:
    """Read a JSON lease file whose freshness is determined by updated_at + ttl."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"exists": False, "active": False, "stale": False, "path": str(path)}
    except Exception as exc:
        return {"exists": True, "active": False, "stale": True, "path": str(path), "error": str(exc)}
    if not isinstance(payload, dict):
        return {"exists": True, "active": False, "stale": True, "path": str(path), "error": invalid_message}

    updated_at = float(payload.get("updated_at") or 0.0)
    resolved_ttl = ttl_seconds if ttl_seconds is not None else payload.get(ttl_key)
    resolved_ttl = max(float(min_ttl_seconds), float(resolved_ttl or min_ttl_seconds))
    current_time = time.time() if now is None else now
    age_seconds = current_time - updated_at if updated_at > 0 else None
    stale = updated_at <= 0 or (age_seconds is not None and age_seconds > resolved_ttl)
    return {
        **payload,
        "exists": True,
        "active": not stale,
        "stale": stale,
        "age_seconds": age_seconds,
        "path": str(path),
    }


def acquire_json_lease(
    path: Path,
    *,
    reason: str,
    ttl_seconds: float = 300.0,
    token: str | None = None,
    now: float | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Create or replace a JSON lease file and return its token."""

    resolved_token = token or uuid.uuid4().hex
    payload = {
        "pid": os.getpid(),
        "token": resolved_token,
        "reason": str(reason or ""),
        "ttl_seconds": max(5.0, float(ttl_seconds or 300.0)),
        "updated_at": time.time() if now is None else now,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved_token


def release_json_lease(path: Path, token: str) -> None:
    """Remove a JSON lease only when it is unowned or still owned by token."""

    resolved_token = str(token or "")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except Exception:
        payload = {}
    if not isinstance(payload, dict) or str(payload.get("token") or "") == resolved_token:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass


def is_json_lease_active(path: Path) -> bool:
    status = read_json_lease(path)
    if not bool(status.get("active")):
        if bool(status.get("stale")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
        return False
    return True


def clear_stale_json_lease(path: Path) -> dict[str, Any]:
    status = read_json_lease(path)
    if not bool(status.get("stale")):
        return {"cleared": False, **status}
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except Exception as exc:
        return {"cleared": False, **status, "error": str(exc)}
    return {"cleared": True, **status}


@contextmanager
def json_lease(
    path: Path,
    *,
    reason: str,
    ttl_seconds: float = 300.0,
) -> Iterator[str]:
    token = acquire_json_lease(path, reason=reason, ttl_seconds=ttl_seconds)
    try:
        yield token
    finally:
        release_json_lease(path, token)
