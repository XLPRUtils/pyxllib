from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


__all__ = [
    "read_json_state",
    "read_json_state_dict",
    "write_json_state",
]


def write_json_state(
    path: Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int | None = 2,
    encoding: str = "utf-8",
    permission_retries: int = 8,
) -> None:
    """Atomically write a small JSON state file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding=encoding)
    try:
        for attempt in range(max(1, int(permission_retries))):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt >= max(1, int(permission_retries)) - 1:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def read_json_state(path: Path, default: Any, *, encoding: str = "utf-8") -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding=encoding))
    except (OSError, json.JSONDecodeError):
        return default


def read_json_state_dict(path: Path, *, default: dict[str, Any] | None = None, encoding: str = "utf-8") -> dict[str, Any]:
    payload = read_json_state(path, default or {}, encoding=encoding)
    return payload if isinstance(payload, dict) else dict(default or {})
