import datetime as dt

from pyxllib.prog.schedule_policy import (
    RESULT_FAILURE,
    RESULT_SUCCESS,
    apply_schedule_result,
    compute_next_by_trigger,
    initialize_schedule_state,
    schedule_policy_label,
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
