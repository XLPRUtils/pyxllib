from __future__ import annotations

from pyxllib.prog.job_queue import (
    clear_job_queue,
    create_job_record,
    job_payload_values,
    job_queue_state,
    normalize_job_record,
    pop_next_job,
    read_job_queue,
    remove_job_by_id,
    requeue_running_jobs,
)


def test_normalize_job_record_keeps_minimal_queue_shape():
    job = normalize_job_record(
        {"task_type": "go_scene", "payload": {"target": "#121"}, "created_at": 10},
        now=20,
    )

    assert job is not None
    assert job["task_type"] == "go_scene"
    assert job["label"] == "go_scene"
    assert job["group"] == "manual_job"
    assert job["status"] == "pending"
    assert job["payload"] == {"target": "#121"}
    assert job["created_at"] == 10.0
    assert job["updated_at"] == 10.0


def test_create_job_record_builds_pending_json_queue_record():
    job = create_job_record(
        "go_scene",
        {"target": "#121"},
        label="到场景",
        group="manual_job",
        interruptible=False,
        id_prefix="manual",
        now=10.5,
    )

    assert job["id"].startswith("manual-10500-")
    assert job["task_type"] == "go_scene"
    assert job["label"] == "到场景"
    assert job["group"] == "manual_job"
    assert job["status"] == "pending"
    assert job["interruptible"] is False
    assert job["payload"] == {"target": "#121"}
    assert job["created_at"] == 10.5
    assert job["updated_at"] == 10.5


def test_read_job_queue_filters_bad_records_status_and_limits():
    raw = [
        {},
        {"task_type": "a", "status": "done"},
        {"task_type": "b", "status": "pending", "created_at": 1},
        {"task_type": "c", "status": "queued", "created_at": 2},
    ]

    jobs = read_job_queue(raw, limit=1)

    assert [job["task_type"] for job in jobs] == ["c"]


def test_requeue_running_jobs_can_drop_non_replayable_jobs():
    jobs = [
        {"id": "safe", "task_type": "detect_scene", "status": "running"},
        {"id": "unsafe", "task_type": "daily_signup", "status": "running"},
        {"id": "pending", "task_type": "go_scene", "status": "pending"},
    ]

    updated, changed_count = requeue_running_jobs(
        jobs,
        keep_running_job=lambda job: job["task_type"] == "detect_scene",
        now=100,
    )

    assert changed_count == 2
    assert [job["id"] for job in updated] == ["safe", "pending"]
    assert updated[0]["status"] == "queued"
    assert updated[0]["last_requeue_reason"] == "backend_reload"
    assert updated[0]["updated_at"] == 100


def test_pop_next_job_respects_group_order_then_created_time():
    jobs = [
        {"id": "job", "group": "job", "status": "pending", "created_at": 1},
        {"id": "guard", "group": "guard", "status": "pending", "created_at": 3},
        {"id": "manual", "group": "manual_job", "status": "queued", "created_at": 2},
    ]

    selected, updated = pop_next_job(
        jobs,
        group_order={"guard": 10, "manual_job": 50, "job": 100},
        now=200,
    )

    assert selected is not None
    assert selected["id"] == "guard"
    assert selected["status"] == "running"
    assert selected["updated_at"] == 200
    assert next(job for job in updated if job["id"] == "guard")["status"] == "running"


def test_job_queue_state_and_remove_job_by_id():
    jobs = [{"id": str(index)} for index in range(3)]

    assert job_queue_state(jobs, limit=2) == [{"id": "1"}, {"id": "2"}]
    assert remove_job_by_id(jobs, "1") == [{"id": "0"}, {"id": "2"}]


def test_job_payload_values_collects_non_empty_payload_strings():
    jobs = [
        {"payload": {"__scheduler_task_id": "daily"}},
        {"payload": {"__scheduler_task_id": " daily "}},
        {"payload": {"__scheduler_task_id": ""}},
        {"payload": "bad"},
        {},
    ]

    assert job_payload_values(jobs, "__scheduler_task_id") == {"daily"}


def test_clear_job_queue_can_keep_running_jobs():
    jobs = [
        {"id": "running", "status": "running"},
        {"id": "pending", "status": "pending"},
    ]

    kept, removed = clear_job_queue(
        jobs,
        keep_job=lambda job: str(job.get("status") or "") == "running",
    )

    assert kept == [{"id": "running", "status": "running"}]
    assert removed == 1
