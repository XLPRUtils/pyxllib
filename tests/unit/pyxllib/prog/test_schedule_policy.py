import datetime as dt

from pyxllib.prog.schedule_policy import (
    RESULT_FAILURE,
    RESULT_SUCCESS,
    apply_schedule_result,
    build_scheduled_task_plan,
    compute_next_by_trigger,
    first_valid_schedule_time_text,
    initialize_schedule_state,
    merge_scheduled_task_updates,
    next_daily_time,
    normalize_scheduled_task_record,
    parse_daily_clock,
    parse_schedule_time,
    repair_orphaned_scheduled_runs,
    schedule_policy_label,
    schedule_task_due,
    schedule_task_due_timestamp,
    schedule_task_order_key,
    scheduled_task_plan_reason,
    scheduled_task_payload_with_meta,
    scheduled_task_run_copy,
    scheduled_task_state,
    select_due_scheduled_tasks,
    sync_scheduled_tasks_from_facts,
)


def test_monthly_trigger_skips_missing_day():
    next_time = compute_next_by_trigger(
        {"type": "monthly", "day": 31, "time": "00:00"},
        base_time=dt.datetime(2026, 2, 1, 0, 0),
    )

    assert next_time == dt.datetime(2026, 3, 31, 0, 0)


def test_daily_trigger_supports_multiple_times():
    next_time = compute_next_by_trigger(
        {"type": "daily", "times": ["06:50", "21:00"]},
        base_time=dt.datetime(2026, 5, 19, 7, 0),
    )

    assert next_time == dt.datetime(2026, 5, 19, 21, 0)


def test_interval_trigger_combines_duration_units():
    next_time = compute_next_by_trigger(
        {"type": "interval", "minutes": 1, "seconds": 30},
        base_time=dt.datetime(2026, 5, 19, 7, 0),
    )

    assert next_time == dt.datetime(2026, 5, 19, 7, 1, 30)


def test_failure_retry_takes_priority_before_normal_next():
    policy = {
        "enabled": True,
        "trigger": {"type": "monthly", "day": 27, "time": "00:00"},
        "outcome": {"on_failure": {"type": "retry_after", "minutes": 10}},
    }

    state = initialize_schedule_state(policy, base_time=dt.datetime(2026, 5, 19, 12, 0))
    next_state = apply_schedule_result(
        policy,
        state,
        result=RESULT_FAILURE,
        base_time=dt.datetime(2026, 5, 27, 0, 0),
    )

    assert next_state["next_trigger_at"] == "2026-05-27T00:10:00"


def test_result_normalization_resets_failure_count_on_success():
    policy = {
        "enabled": True,
        "trigger": {"type": "daily", "time": "00:00"},
    }

    next_state = apply_schedule_result(
        policy,
        {"failure_count": 2},
        result=RESULT_SUCCESS.upper(),
        base_time=dt.datetime(2026, 5, 19, 7, 0),
    )

    assert next_state["last_result"] == RESULT_SUCCESS
    assert next_state["failure_count"] == 0


def test_zero_interval_policy_has_no_label():
    assert schedule_policy_label({"enabled": True, "trigger": {"type": "interval"}}) == ""


def test_success_returns_to_normal_trigger_rule():
    policy = {
        "enabled": True,
        "trigger": {"type": "monthly", "day": 27, "time": "00:00"},
        "outcome": {"on_failure": {"type": "retry_after", "minutes": 10}},
    }

    next_state = apply_schedule_result(
        policy,
        {},
        result=RESULT_SUCCESS,
        base_time=dt.datetime(2026, 5, 27, 0, 0),
    )

    assert next_state["next_trigger_at"] == "2026-06-27T00:00:00"
    assert schedule_policy_label(policy) == "每月 27 日 00:00"


def test_public_schedule_helpers_match_json_task_needs():
    assert parse_schedule_time("2026-06-02 05:00:00") == dt.datetime(2026, 6, 2, 5, 0).timestamp()
    assert parse_schedule_time("2026-06-02T05:00:00") == dt.datetime(2026, 6, 2, 5, 0).timestamp()
    assert parse_schedule_time("123.5") == 123.5
    assert parse_schedule_time("bad") is None
    assert parse_daily_clock("05:00").hour == 5
    assert parse_daily_clock("05:00:01").second == 1
    assert parse_daily_clock("25:00") is None


def test_first_valid_schedule_time_text_skips_empty_and_invalid_values():
    source = {
        "bad": "not-time",
        "empty": "",
        "good": "2026-06-02 05:00:00",
        "later": "2026-06-03 05:00:00",
    }

    assert first_valid_schedule_time_text(source, "missing", "empty", "bad", "good", "later") == "2026-06-02 05:00:00"
    assert first_valid_schedule_time_text(source, "missing", "bad") is None


def test_next_daily_time_selects_next_future_clock():
    assert next_daily_time(
        ["05:00", "23:00"],
        base_time=dt.datetime(2026, 6, 2, 6, 0),
    ) == "2026-06-02 23:00:00"
    assert next_daily_time(
        ["05:00"],
        base_time=dt.datetime(2026, 6, 2, 6, 0),
    ) == "2026-06-03 05:00:00"
    assert next_daily_time(["bad"], base_time=dt.datetime(2026, 6, 2, 6, 0)) is None


def test_schedule_task_due_uses_retry_before_next_time():
    task = {
        "enabled": True,
        "next_time": "2026-06-02 05:00:00",
        "retry_after": "2026-06-02 06:00:00",
    }

    assert schedule_task_due(task, now=dt.datetime(2026, 6, 2, 5, 30).timestamp()) is False
    assert schedule_task_due(task, now=dt.datetime(2026, 6, 2, 6, 0).timestamp()) is True
    assert schedule_task_due({**task, "enabled": False}, now=dt.datetime(2026, 6, 2, 7, 0).timestamp()) is False
    assert schedule_task_due({"enabled": True}, now=dt.datetime(2026, 6, 2, 7, 0).timestamp()) is True


def test_scheduled_task_plan_reason_explains_task_state():
    now = dt.datetime(2026, 6, 2, 5, 0).timestamp()
    supported = lambda task: task.get("task_type") != "unsupported"

    assert scheduled_task_plan_reason({"enabled": False}, False, now=now) == "未启用"
    assert scheduled_task_plan_reason(
        {"enabled": True, "retry_after": "2026-06-02 05:10:00"},
        False,
        now=now,
    ) == "等待重试：2026-06-02 05:10:00"
    assert scheduled_task_plan_reason(
        {"enabled": True, "next_time": "2026-06-02 05:20:00"},
        False,
        now=now,
    ) == "未到时间：2026-06-02 05:20:00"
    assert scheduled_task_plan_reason(
        {"enabled": True, "task_type": "unsupported"},
        True,
        task_supported=supported,
        now=now,
    ) == "尚未纳入当前框架验收"
    assert scheduled_task_plan_reason({"enabled": True}, True, now=now) == "已到期"
    assert scheduled_task_plan_reason({"enabled": True}, False, now=now) == "可手动执行"


def test_scheduled_task_plan_reason_allows_custom_labels_and_keys():
    now = dt.datetime(2026, 6, 2, 5, 0).timestamp()
    task = {"active": True, "retry_at": "2026-06-02 05:10:00"}

    assert scheduled_task_plan_reason(
        task,
        False,
        now=now,
        enabled_key="active",
        retry_after_key="retry_at",
        labels={"retry_wait": "retry until {time}"},
    ) == "retry until 2026-06-02 05:10:00"


def test_build_scheduled_task_plan_marks_due_runnable_and_attaches_facts():
    now = dt.datetime(2026, 6, 2, 5, 0).timestamp()
    tasks = [
        {"id": "future", "task_type": "demo", "label": "Future", "enabled": True, "next_time": "2026-06-02 06:00:00"},
        {"id": "due", "task_type": "demo", "label": "Due", "enabled": True, "next_time": "2026-06-02 04:00:00"},
    ]

    plan = build_scheduled_task_plan(
        tasks,
        now=now,
        task_facts={"due": {"last_result": "success"}},
    )

    assert plan["next_action"] == "run_due"
    assert plan["message"] == "建议执行到期任务：Due"
    assert [item["id"] for item in plan["due_tasks"]] == ["due"]
    assert plan["due_tasks"][0]["runnable"] is True
    assert plan["due_tasks"][0]["fact"] == {"last_result": "success"}
    assert [item["id"] for item in plan["tasks"]] == ["due", "future"]


def test_build_scheduled_task_plan_waits_when_runtime_is_running():
    plan = build_scheduled_task_plan(
        [{"id": "due", "enabled": True}],
        runtime_running=True,
        runtime_task="兑换礼包码",
        now=dt.datetime(2026, 6, 2, 5, 0).timestamp(),
    )

    assert plan["next_action"] == "wait"
    assert plan["message"] == "Runtime 正在运行：兑换礼包码"
    assert plan["due_tasks"][0]["runnable"] is False


def test_build_scheduled_task_plan_reports_blocked_and_idle():
    now = dt.datetime(2026, 6, 2, 5, 0).timestamp()
    blocked = build_scheduled_task_plan(
        [{"id": "unsupported", "enabled": True}],
        task_supported=lambda task: False,
        now=now,
    )
    idle = build_scheduled_task_plan(
        [{"id": "disabled", "enabled": False}],
        now=now,
    )

    assert blocked["next_action"] == "blocked"
    assert blocked["due_tasks"][0]["supported"] is False
    assert blocked["due_tasks"][0]["reason"] == "尚未纳入当前框架验收"
    assert idle["next_action"] == "idle"
    assert idle["due_tasks"] == []


def test_sync_scheduled_tasks_from_facts_updates_mapped_fields_and_checkpoint():
    tasks = [{
        "id": "daily",
        "next_time": None,
        "retry_after": None,
        "last_result": "",
        "checkpoint": None,
    }]
    facts = {
        "daily": {
            "discovered_next_time": "2026-06-02 13:00:00",
            "retry_after": "bad-time",
            "last_result": "success",
            "updated_at": 123,
        }
    }

    changed = sync_scheduled_tasks_from_facts(
        tasks,
        facts,
        time_field_sources={
            "next_time": ("discovered_next_time", "next_time"),
            "retry_after": ("retry_after",),
        },
        text_field_sources={"last_result": ("last_result",)},
        synced_at_key="world_fact_synced_at",
        fact_updated_at_key="world_fact_updated_at",
        synced_at_text="2026-06-02 12:00:00",
    )

    assert changed is True
    assert tasks[0]["next_time"] == "2026-06-02 13:00:00"
    assert tasks[0]["retry_after"] is None
    assert tasks[0]["last_result"] == "success"
    assert tasks[0]["checkpoint"] == {
        "world_fact_synced_at": "2026-06-02 12:00:00",
        "world_fact_updated_at": 123,
    }


def test_sync_scheduled_tasks_from_facts_returns_false_without_changes():
    tasks = [{"id": "daily", "next_time": "2026-06-02 13:00:00"}]

    changed = sync_scheduled_tasks_from_facts(
        tasks,
        {"daily": {"next_time": "2026-06-02 13:00:00", "updated_at": 123}},
        time_field_sources={"next_time": ("next_time",)},
        synced_at_text="sync",
    )

    assert changed is False
    assert "checkpoint" not in tasks[0]


def test_normalize_scheduled_task_record_returns_common_json_shape():
    record = normalize_scheduled_task_record({
        "id": "daily",
        "task_type": "daily_signup",
        "label": "",
        "enabled": 1,
        "interruptible": 0,
        "next_time": "",
        "schedule_times": [5, "06:00"],
        "window": [1, "2", "ignored"],
        "last_run_at": "",
        "last_result": None,
        "retry_after": "",
        "cooldown_seconds": "30",
        "payload": {"ok": True},
        "checkpoint": {"seen": True},
    })

    assert record == {
        "id": "daily",
        "task_type": "daily_signup",
        "label": "daily",
        "source": "manual",
        "schedule_kind": "manual",
        "legacy_name": "",
        "enabled": True,
        "interruptible": False,
        "next_time": None,
        "schedule_times": ["5", "06:00"],
        "window": ["1", "2"],
        "last_run_at": None,
        "last_result": "",
        "retry_after": None,
        "cooldown_seconds": 30,
        "payload": {"ok": True},
        "checkpoint": {"seen": True},
    }


def test_normalize_scheduled_task_record_rejects_missing_identity():
    assert normalize_scheduled_task_record({}) is None
    assert normalize_scheduled_task_record({"id": "daily"}) is None
    assert normalize_scheduled_task_record({"task_type": "daily_signup"}) is None


def test_schedule_task_order_key_uses_kind_then_due_time_then_id():
    daily = {"id": "b", "schedule_kind": "daily", "next_time": "2026-06-02 05:00:00"}
    manual = {"id": "a", "schedule_kind": "manual"}
    dynamic = {"id": "c", "schedule_kind": "dynamic", "schedule_times": ["08:00"]}

    assert isinstance(schedule_task_due_timestamp(dynamic), float)
    assert sorted([manual, dynamic, daily], key=schedule_task_order_key) == [daily, dynamic, manual]


def test_select_due_scheduled_tasks_filters_manual_unsupported_and_orders_results():
    tasks = [
        {"id": "manual", "schedule_kind": "manual", "enabled": True},
        {"id": "unsupported", "schedule_kind": "daily", "enabled": True},
        {"id": "later", "schedule_kind": "daily", "enabled": True, "next_time": "2026-06-02 06:00:00"},
        {"id": "first", "schedule_kind": "daily", "enabled": True, "next_time": "2026-06-02 05:00:00"},
    ]

    due_tasks = select_due_scheduled_tasks(
        tasks,
        task_due=lambda task: task.get("id") != "later",
        task_supported=lambda task: task.get("id") != "unsupported",
    )

    assert [task["id"] for task in due_tasks] == ["first"]


def test_merge_scheduled_task_updates_preserves_runtime_fields_and_initializes_enabled_task():
    current = [{
        "id": "daily",
        "enabled": False,
        "label": "old",
        "last_result": "success",
        "retry_after": None,
        "next_time": None,
        "checkpoint": {"seen": True},
    }]
    incoming = [{"id": "daily", "enabled": True, "label": "new"}]

    merged = merge_scheduled_task_updates(
        current,
        incoming,
        normalizer=lambda item: dict(item) if item.get("id") else None,
        next_time_resolver=lambda task, base_time: f"next:{task['id']}:{base_time}",
        base_time="now",
    )

    assert merged == [{
        "id": "daily",
        "enabled": True,
        "label": "new",
        "last_run_at": None,
        "last_result": "success",
        "retry_after": None,
        "next_time": "next:daily:now",
        "checkpoint": {"seen": True},
    }]


def test_merge_scheduled_task_updates_clears_runtime_time_when_disabled():
    current = [{
        "id": "daily",
        "enabled": True,
        "retry_after": "2026-06-02 06:00:00",
        "next_time": "2026-06-02 05:00:00",
    }]
    incoming = [{"id": "daily", "enabled": False}]

    merged = merge_scheduled_task_updates(
        current,
        incoming,
        normalizer=lambda item: dict(item) if item.get("id") else None,
    )

    assert merged[0]["retry_after"] is None
    assert merged[0]["next_time"] is None


def test_scheduled_task_run_copy_applies_payload_override_without_mutating_source():
    tasks = [{
        "id": "gift-code-weekly",
        "label": "weekly",
        "payload": {"codes": []},
    }]

    run_task = scheduled_task_run_copy(
        tasks,
        "gift-code-weekly",
        {"codes": ["abc"]},
    )

    assert run_task is not None
    assert run_task is not tasks[0]
    assert run_task["payload"]["codes"] == ["abc"]
    assert tasks[0]["payload"]["codes"] == []
    assert scheduled_task_run_copy(tasks, "missing") is None


def test_scheduled_task_payload_with_meta_adds_scheduler_bookkeeping():
    task = {
        "id": "gift-code-weekly",
        "interruptible": False,
        "payload": {"codes": ["abc"]},
    }

    payload = scheduled_task_payload_with_meta(task)

    assert payload == {
        "codes": ["abc"],
        "__scheduler_task_id": "gift-code-weekly",
        "__scheduler_interruptible": False,
    }
    assert task["payload"] == {"codes": ["abc"]}


def test_scheduled_task_state_drops_view_only_fields():
    task = {
        "id": "daily",
        "enabled": True,
        "supported": False,
        "selected": True,
    }

    assert scheduled_task_state(task) == {"id": "daily", "enabled": True, "selected": True}
    assert scheduled_task_state(task, transient_keys=("supported", "selected")) == {"id": "daily", "enabled": True}


def test_repair_orphaned_scheduled_runs_marks_stale_running_task_stopped():
    tasks = [
        {"id": "stale", "last_result": "queued", "last_run_at": "2026-06-02 05:00:00", "retry_after": "2026-06-02 06:00:00"},
        {"id": "pending", "last_result": "running", "last_run_at": "2026-06-02 05:00:00"},
        {"id": "fresh", "last_result": "queued", "last_run_at": "2026-06-02 05:01:30"},
    ]

    changed = repair_orphaned_scheduled_runs(
        tasks,
        pending_task_ids=["pending"],
        now=dt.datetime(2026, 6, 2, 5, 2).timestamp(),
        min_age_seconds=60,
        recovered_at_text="recovered",
    )

    assert changed is True
    assert tasks[0]["last_result"] == "stopped"
    assert tasks[0]["retry_after"] is None
    assert tasks[0]["checkpoint"]["recovered_from_orphaned_run_at"] == "recovered"
    assert tasks[1]["last_result"] == "running"
    assert tasks[2]["last_result"] == "queued"


def test_repair_orphaned_scheduled_runs_skips_when_runtime_is_running():
    tasks = [{"id": "stale", "last_result": "queued"}]

    assert repair_orphaned_scheduled_runs(tasks, runtime_running=True) is False
    assert tasks[0]["last_result"] == "queued"
