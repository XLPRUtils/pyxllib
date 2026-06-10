from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Iterable


__all__ = [
    "clear_job_queue",
    "create_job_record",
    "job_payload_values",
    "job_queue_state",
    "normalize_job_record",
    "pop_next_job",
    "read_job_queue",
    "remove_job_by_id",
    "requeue_running_jobs",
]


JobNormalizer = Callable[[Any], dict[str, Any] | None]
RunningJobKeepPredicate = Callable[[dict[str, Any]], bool]
JobKeepPredicate = Callable[[dict[str, Any]], bool]


def normalize_job_record(
    item: Any,
    *,
    task_type_key: str = "task_type",
    default_group: str = "manual_job",
    default_status: str = "pending",
    now: float | None = None,
) -> dict[str, Any] | None:
    """Normalize a loose dict into the minimal JSON job-queue shape."""

    if not isinstance(item, dict):
        return None
    task_type = str(item.get(task_type_key) or "").strip()
    if not task_type:
        return None
    current_time = time.time() if now is None else now
    created_at = float(item.get("created_at") or current_time)
    job_id = str(item.get("id") or f"manual-{uuid.uuid4().hex}")
    return {
        "id": job_id,
        task_type_key: task_type,
        "label": str(item.get("label") or task_type),
        "group": str(item.get("group") or default_group),
        "status": str(item.get("status") or default_status),
        "interruptible": bool(item.get("interruptible", True)),
        "payload": item.get("payload") if isinstance(item.get("payload"), dict) else {},
        "created_at": created_at,
        "updated_at": float(item.get("updated_at") or created_at),
    }


def create_job_record(
    task_type: str,
    payload: dict[str, Any] | None = None,
    *,
    label: str = "",
    group: str = "manual_job",
    status: str = "pending",
    interruptible: bool = True,
    id_prefix: str = "manual",
    now: float | None = None,
) -> dict[str, Any]:
    current_time = time.time() if now is None else now
    resolved_task_type = str(task_type or "").strip()
    if not resolved_task_type:
        raise ValueError("job task_type is required")
    return {
        "id": f"{id_prefix}-{int(current_time * 1000)}-{uuid.uuid4().hex[:8]}",
        "task_type": resolved_task_type,
        "label": str(label or resolved_task_type),
        "group": str(group or "manual_job"),
        "status": str(status or "pending"),
        "interruptible": bool(interruptible),
        "payload": payload if isinstance(payload, dict) else {},
        "created_at": current_time,
        "updated_at": current_time,
    }


def read_job_queue(
    raw: Any,
    *,
    normalizer: JobNormalizer = normalize_job_record,
    allowed_statuses: Iterable[str] = ("pending", "running", "queued"),
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Read a JSON-list job queue, filtering malformed and unsupported records."""

    source = raw if isinstance(raw, list) else []
    status_set = {str(status) for status in allowed_statuses}
    jobs = [
        job
        for item in source
        if (job := normalizer(item))
        and str(job.get("status") or "") in status_set
    ]
    return jobs[-max(1, int(limit)) :]


def job_queue_state(jobs: list[dict[str, Any]], *, limit: int = 100) -> list[dict[str, Any]]:
    return jobs[-max(1, int(limit)) :]


def job_payload_values(
    jobs: Iterable[dict[str, Any]],
    key: str,
    *,
    payload_key: str = "payload",
) -> set[str]:
    """Collect non-empty string values from job payloads."""

    values: set[str] = set()
    for job in jobs:
        payload = job.get(payload_key) if isinstance(job.get(payload_key), dict) else {}
        value = str(payload.get(key) or "").strip()
        if value:
            values.add(value)
    return values


def requeue_running_jobs(
    jobs: list[dict[str, Any]],
    *,
    keep_running_job: RunningJobKeepPredicate | None = None,
    now: float | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Turn running jobs back to queued when keep_running_job allows replay."""

    current_time = time.time() if now is None else now
    updated: list[dict[str, Any]] = []
    changed_count = 0
    for job in jobs:
        if str(job.get("status") or "") != "running":
            updated.append(job)
            continue
        changed_count += 1
        if keep_running_job is None or keep_running_job(job):
            updated.append({
                **job,
                "status": "queued",
                "updated_at": current_time,
                "last_requeue_reason": "backend_reload",
            })
    return updated, changed_count


def pop_next_job(
    jobs: list[dict[str, Any]],
    *,
    runnable_statuses: Iterable[str] = ("pending", "queued"),
    group_order: dict[str, int] | None = None,
    default_group: str = "manual_job",
    now: float | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Claim the next runnable job and mark it running."""

    runnable_status_set = {str(status) for status in runnable_statuses}
    order = group_order or {}
    runnable = [job for job in jobs if str(job.get("status") or "") in runnable_status_set]
    if not runnable:
        return None, jobs
    runnable.sort(
        key=lambda item: (
            order.get(str(item.get("group") or default_group), 1000),
            float(item.get("created_at") or 0),
        )
    )
    selected_id = str(runnable[0].get("id") or "")
    current_time = time.time() if now is None else now
    updated: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for job in jobs:
        if str(job.get("id") or "") == selected_id:
            job = {**job, "status": "running", "updated_at": current_time}
            selected = job
        updated.append(job)
    return selected, updated


def remove_job_by_id(jobs: list[dict[str, Any]], job_id: str) -> list[dict[str, Any]]:
    resolved_job_id = str(job_id or "")
    if not resolved_job_id:
        return jobs
    return [job for job in jobs if str(job.get("id") or "") != resolved_job_id]


def clear_job_queue(
    jobs: list[dict[str, Any]],
    *,
    keep_job: JobKeepPredicate | None = None,
) -> tuple[list[dict[str, Any]], int]:
    kept = [job for job in jobs if keep_job is not None and keep_job(job)]
    return kept, len(jobs) - len(kept)
