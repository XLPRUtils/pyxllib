"""Bounded per-process stage diagnostics for the WeChat storage provider.

Only fixed stage names, call counts, elapsed seconds, work units and the
process private-memory change across each stage are recorded.  No paths,
keys, chat identities or message content are stored.  Other threads may
allocate during a stage, so a single delta is noisy; consistent positive
totals over many calls are the signal.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading
import time

SYNC_STAGES = frozenset({
    "sync.copy", "sync.decrypt", "decrypt.db", "decrypt.wal_apply",
    "updates.resolve_sender", "updates.query", "updates.contacts",
})
_LOCK = threading.Lock()
_STATS: dict[str, dict[str, float]] = {}


def _private_mb() -> float | None:
    try:
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, "private", info.rss) / 2 ** 20
    except Exception:
        return None


@contextmanager
def sync_stage(name: str):
    """Measure one provider stage; unknown names are rejected to stay bounded."""
    if name not in SYNC_STAGES:
        raise ValueError(f"unknown provider stage {name!r}")
    started = time.perf_counter()
    before = _private_mb()
    units = {"value": 0}
    try:
        yield units
    finally:
        after = _private_mb()
        delta = after - before if after is not None and before is not None else None
        with _LOCK:
            stat = _STATS.setdefault(name, {"count": 0, "seconds": 0.0, "units": 0,
                                            "private_delta_mb": 0.0, "positive_delta_mb": 0.0,
                                            "max_delta_mb": 0.0})
            stat["count"] += 1
            stat["seconds"] += time.perf_counter() - started
            stat["units"] += int(units["value"] or 0)
            if delta is not None:
                stat["private_delta_mb"] += delta
                stat["positive_delta_mb"] += max(0.0, delta)
                stat["max_delta_mb"] = max(stat["max_delta_mb"], delta)


def sync_stage_diagnostics() -> dict[str, dict[str, float]]:
    """Return a rounded copy of the per-stage totals since process start."""
    with _LOCK:
        return {name: {key: round(value, 3) if isinstance(value, float) else value
                       for key, value in stat.items()}
                for name, stat in sorted(_STATS.items())}
