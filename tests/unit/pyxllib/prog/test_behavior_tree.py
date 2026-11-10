import datetime
import inspect
import json
import os
from pathlib import Path

import pytest

from pyxllib.prog.behavior_tree import (
    Action,
    BehaviorTreeRunner,
    Daily,
    DynamicTime,
    Every,
    IdleUntilNextWake,
    MemorySelector,
    Monthly,
    Once,
    ReactiveSelector,
    Retry,
    Root,
    Sequence,
    Status,
    Timeout,
    Weekly,
    Window,
    WithServices,
)


class FakeClock:
    def __init__(self, value="2026-04-26 00:00:00"):
        self.value = datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")

    def __call__(self):
        return self.value

    def advance(self, **kwargs):
        self.value += datetime.timedelta(**kwargs)


def test_action_supports_generator_and_releases_control(tmp_path):
    events = []

    def task():
        events.append("start")
        yield
        events.append("end")

    runner = BehaviorTreeRunner(Root(Action(task)), tmp_path / "state.json")

    assert runner.run_once() == Status.RUNNING
    assert runner.run_once() == Status.SUCCESS
    assert events == ["start", "end"]


def test_dynamic_time_persists_next_run_while_child_is_running(tmp_path):
    clock = FakeClock("2026-04-26 00:00:00")
    events = []

    def task(ctx):
        events.append("start")
        ctx.next_run_at = ctx.now() + datetime.timedelta(hours=1)
        yield
        events.append("end")
        return ctx.next_time(seconds=7200)

    runner = BehaviorTreeRunner(Root(DynamicTime(Action(task))), tmp_path / "state.json", now_func=clock)

    assert runner.run_once() == Status.RUNNING
    state = next(v for v in runner.state["nodes"].values() if "next_run_at" in v)
    assert state["next_run_at"] == "2026-04-26 01:00:00"

    assert runner.run_once() == Status.SUCCESS
    state = next(v for v in runner.state["nodes"].values() if "next_run_at" in v)
    assert state["next_run_at"] == "2026-04-26 02:00:00"
    assert events == ["start", "end"]


def test_trace_logs_generator_yield_and_return_source_locations(tmp_path):
    lines = {}

    def leaf():
        lines["yield"] = inspect.currentframe().f_lineno + 1
        yield
        lines["return"] = inspect.currentframe().f_lineno + 1
        return "done"

    def wrapper():
        return (yield from leaf())

    log_path = tmp_path / "trace.log"
    runner = BehaviorTreeRunner(Root(Action(wrapper)), tmp_path / "state.json", trace=2, log_path=log_path)
    filename = Path(__file__).name

    assert runner.run_once() == Status.RUNNING
    log_text = log_path.read_text(encoding="utf-8")
    assert f"tree yield: {filename}:{lines['yield']} -> RUNNING" in log_text

    assert runner.run_once() == Status.SUCCESS
    log_text = log_path.read_text(encoding="utf-8")
    assert f"tree return: {filename}:{lines['return']} -> SUCCESS" in log_text
    assert "tree tick:" not in log_text


def test_sequence_and_selector_handle_skip(tmp_path):
    events = []
    clock = FakeClock("2026-04-26 04:00:00")
    tree = Root(
        ReactiveSelector(
            Window("05:00", "05:10", Action(lambda: events.append("window"))),
            Action(lambda: events.append("fallback")),
        )
    )
    runner = BehaviorTreeRunner(tree, tmp_path / "state.json", now_func=clock)

    assert runner.run_once() == Status.SUCCESS
    assert events == ["fallback"]


def test_daily_defaults_to_persistent_next_wake(tmp_path):
    clock = FakeClock("2026-04-26 04:00:00")
    events = []
    state_path = tmp_path / "state.json"

    def task(ctx):
        events.append("run")

    runner = BehaviorTreeRunner(Root(Daily("05:00", Action(task))), state_path, now_func=clock)

    assert runner.run_once() == Status.SUCCESS
    assert events == ["run"]
    daily_state = next(v for k, v in runner.state["nodes"].items() if "Daily" in k)
    assert daily_state["next_run_at"] == "2026-04-26 05:00:00"

    runner2 = BehaviorTreeRunner(Root(Daily("05:00", Action(task))), state_path, now_func=clock)
    assert runner2.run_once() == Status.SKIP
    assert events == ["run"]


def test_daily_start_next_waits_until_next_anchor(tmp_path):
    clock = FakeClock("2026-04-26 04:00:00")
    events = []
    state_path = tmp_path / "state.json"

    runner = BehaviorTreeRunner(
        Root(Action(lambda: events.append("run")).daily("05:00", start="next")),
        state_path,
        now_func=clock,
    )

    assert runner.run_once() == Status.SKIP
    assert events == []
    daily_state = next(v for v in runner.state["nodes"].values() if "next_run_at" in v)
    assert daily_state["next_run_at"] == "2026-04-26 05:00:00"

    clock.advance(hours=1)
    assert runner.run_once() == Status.SUCCESS
    assert events == ["run"]
    assert daily_state["next_run_at"] == "2026-04-27 05:00:00"


def test_monthly_start_next_waits_until_monthly_anchor(tmp_path):
    clock = FakeClock("2026-04-26 04:00:00")
    events = []
    state_path = tmp_path / "state.json"

    runner = BehaviorTreeRunner(
        Root(Action(lambda: events.append("run")).monthly(27, "00:00", start="next")),
        state_path,
        now_func=clock,
    )

    assert runner.run_once() == Status.SKIP
    assert events == []
    monthly_state = next(v for v in runner.state["nodes"].values() if "next_run_at" in v)
    assert monthly_state["next_run_at"] == "2026-04-27 00:00:00"

    clock.advance(hours=20)
    assert runner.run_once() == Status.SUCCESS
    assert events == ["run"]
    assert monthly_state["next_run_at"] == "2026-05-27 00:00:00"


def test_weekly_start_next_waits_until_weekly_anchor(tmp_path):
    clock = FakeClock("2026-08-14 10:00:00")  # Friday
    events = []
    state_path = tmp_path / "state.json"

    runner = BehaviorTreeRunner(
        Root(Action(lambda: events.append("run")).weekly(6, "22:00", start="next")),
        state_path,
        now_func=clock,
    )

    assert runner.run_once() == Status.SKIP
    assert events == []
    weekly_state = next(v for v in runner.state["nodes"].values() if "next_run_at" in v)
    assert weekly_state["next_run_at"] == "2026-08-15 22:00:00"

    clock.advance(days=1, hours=12)
    assert runner.run_once() == Status.SUCCESS
    assert events == ["run"]
    assert weekly_state["next_run_at"] == "2026-08-22 22:00:00"


def test_weekly_class_accepts_positional_child(tmp_path):
    clock = FakeClock("2026-08-14 10:00:00")
    runner = BehaviorTreeRunner(
        Root(Weekly(6, "22:00", Action(lambda: None), start="next")),
        tmp_path / "state.json",
        now_func=clock,
    )

    assert runner.run_once() == Status.SKIP


def test_monthly_skips_month_without_requested_day(tmp_path):
    clock = FakeClock("2026-01-30 04:00:00")
    events = []
    runner = BehaviorTreeRunner(
        Root(Action(lambda: events.append("run")).monthly(31, "00:00", start="next")),
        tmp_path / "state.json",
        now_func=clock,
    )

    assert runner.run_once() == Status.SKIP
    monthly_state = next(v for v in runner.state["nodes"].values() if "next_run_at" in v)
    assert monthly_state["next_run_at"] == "2026-01-31 00:00:00"

    clock.advance(days=1, hours=20)
    assert runner.run_once() == Status.SUCCESS
    assert events == ["run"]
    assert monthly_state["next_run_at"] == "2026-03-31 00:00:00"


def test_monthly_class_accepts_positional_child(tmp_path):
    clock = FakeClock("2026-04-26 04:00:00")
    events = []
    runner = BehaviorTreeRunner(
        Root(Monthly(27, "00:00", Action(lambda: events.append("run")), start="next")),
        tmp_path / "state.json",
        now_func=clock,
    )

    assert runner.run_once() == Status.SKIP
    assert events == []


def test_ctx_can_override_next_monthly_time(tmp_path):
    clock = FakeClock("2026-01-30 04:00:00")
    runner = BehaviorTreeRunner(
        Root(Action(lambda ctx: ctx.next_monthly_time(31, "00:00")).monthly(27, "00:00")),
        tmp_path / "state.json",
        now_func=clock,
    )

    assert runner.run_once() == Status.SUCCESS
    monthly_state = next(v for v in runner.state["nodes"].values() if "next_run_at" in v)
    assert monthly_state["next_run_at"] == "2026-01-31 00:00:00"


def test_daily_start_reset_run_ignores_existing_state(tmp_path):
    clock = FakeClock("2026-04-26 04:00:00")
    events = []
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"nodes": {"Root/task": {"next_run_at": "2026-04-27 05:00:00"}}, "blackboard": {}}),
        encoding="utf-8",
    )

    def task():
        events.append("run")

    runner = BehaviorTreeRunner(Root(Action(task).daily("05:00", start="reset-run")), state_path, now_func=clock)

    assert runner.run_once() == Status.SUCCESS
    assert events == ["run"]
    daily_state = runner.state["nodes"]["Root/task"]
    assert daily_state["next_run_at"] == "2026-04-26 05:00:00"


def test_daily_start_reset_next_ignores_existing_state(tmp_path):
    clock = FakeClock("2026-04-26 04:00:00")
    events = []
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"nodes": {"Root/task": {"next_run_at": "2026-04-27 05:00:00"}}, "blackboard": {}}),
        encoding="utf-8",
    )

    def task():
        events.append("run")

    runner = BehaviorTreeRunner(Root(Action(task).daily("05:00", start="reset-next")), state_path, now_func=clock)

    assert runner.run_once() == Status.SKIP
    assert events == []
    daily_state = runner.state["nodes"]["Root/task"]
    assert daily_state["next_run_at"] == "2026-04-26 05:00:00"


def test_daily_rejects_ambiguous_start_policy(tmp_path):
    with pytest.raises(ValueError, match="invalid start policy"):
        BehaviorTreeRunner(Root(Action(lambda: None).daily("05:00", start="later")), tmp_path / "state.json")

    with pytest.raises(ValueError, match="default_next_time"):
        BehaviorTreeRunner(
            Root(Action(lambda: None).daily("05:00", start="next", default_next_time="2026-04-27 05:00:00")),
            tmp_path / "state.json",
        )


def test_time_node_enabled_false_ignores_without_touching_schedule(tmp_path):
    clock = FakeClock("2026-04-26 04:00:00")
    events = []
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"nodes": {"Root/task": {"next_run_at": "2026-04-27 05:00:00"}}, "blackboard": {}}),
        encoding="utf-8",
    )

    def task():
        events.append("run")

    runner = BehaviorTreeRunner(Root(Action(task).daily("05:00", start="reset-run", enabled=False)), state_path, now_func=clock)

    assert runner.next_wake() is None
    assert runner.run_once() == Status.SKIP
    assert events == []
    assert runner.state["nodes"]["Root/task"]["next_run_at"] == "2026-04-27 05:00:00"


def test_time_node_on_schedule_callback(tmp_path):
    clock = FakeClock("2026-04-26 04:00:00")
    scheduled = []

    runner = BehaviorTreeRunner(
        Root(
            Daily(
                "05:00",
                Action(lambda: None),
                on_schedule=lambda ctx, run_at: scheduled.append(run_at.strftime("%Y-%m-%d %H:%M:%S")),
            )
        ),
        tmp_path / "state.json",
        now_func=clock,
    )

    assert runner.run_once() == Status.SUCCESS
    assert scheduled == ["2026-04-26 04:00:00", "2026-04-26 05:00:00"]


def test_every_defaults_to_memory_state_not_persistent(tmp_path):
    clock = FakeClock("2026-04-26 04:00:00")
    events = []
    state_path = tmp_path / "state.json"

    runner = BehaviorTreeRunner(Root(Every(minutes=5, child=Action(lambda: events.append("run")))), state_path, now_func=clock)
    assert runner.run_once() == Status.SUCCESS
    assert events == ["run"]
    assert not runner.state["nodes"]

    runner2 = BehaviorTreeRunner(Root(Every(minutes=5, child=Action(lambda: events.append("run")))), state_path, now_func=clock)
    assert runner2.run_once() == Status.SUCCESS
    assert events == ["run", "run"]


def test_once_is_only_memory_for_current_runner(tmp_path):
    events = []
    state_path = tmp_path / "state.json"

    runner = BehaviorTreeRunner(Root(Once(Action(lambda: events.append("layout")))), state_path)
    assert runner.run_once() == Status.SUCCESS
    assert runner.run_once() == Status.SKIP
    assert events == ["layout"]
    assert not runner.state["nodes"]

    runner2 = BehaviorTreeRunner(Root(Once(Action(lambda: events.append("layout")))), state_path)
    assert runner2.run_once() == Status.SUCCESS
    assert events == ["layout", "layout"]


def test_idle_waits_for_next_business_wake(monkeypatch, tmp_path):
    clock = FakeClock("2026-04-26 04:00:00")
    events = []
    sleeps = []
    tree = Root(
        ReactiveSelector(
            Daily("05:00", Action(lambda: events.append("task"))),
            Sequence(
                IdleUntilNextWake(ratio=0.8),
                Action(lambda: events.append("layout")),
            ),
        )
    )
    runner = BehaviorTreeRunner(tree, tmp_path / "state.json", now_func=clock)

    monkeypatch.setattr("pyxllib.prog.behavior_tree.time.sleep", lambda seconds: sleeps.append(seconds))

    assert runner.run_once() == Status.SUCCESS
    assert events == ["task"]
    assert runner.run_once() == Status.RUNNING
    assert sleeps == [2880.0]


def test_with_services_do_not_wake_idle_tree_without_business_work(monkeypatch, tmp_path):
    clock = FakeClock("2026-04-26 04:00:00")
    events = []
    sleeps = []
    tree = Root(
        ReactiveSelector(
            WithServices(
                Daily("05:00", Action(lambda: events.append("business"))),
                Every(minutes=1, child=Action(lambda: events.append("guard"))),
            ),
            IdleUntilNextWake(ratio=0.8),
        )
    )
    runner = BehaviorTreeRunner(tree, tmp_path / "state.json", now_func=clock)
    monkeypatch.setattr("pyxllib.prog.behavior_tree.time.sleep", lambda seconds: sleeps.append(seconds))

    assert runner.run_once() == Status.SUCCESS
    assert events == ["guard"]
    assert runner.run_once() == Status.SUCCESS
    assert events == ["guard", "business"]
    assert runner.run_once() == Status.RUNNING
    assert events == ["guard", "business"]
    assert sleeps == [2880.0]

    clock.advance(minutes=60)
    assert runner.run_once() == Status.SUCCESS
    assert events == ["guard", "business", "guard"]


def test_with_services_continues_when_guard_yields_zero(tmp_path):
    events = []

    def guard():
        events.append("guard")
        yield 0

    def business():
        events.append("business")

    runner = BehaviorTreeRunner(
        Root(WithServices(Every(60, child=Action(business)), Action(guard))),
        tmp_path / "state.json",
    )

    assert runner.run_once() == Status.SUCCESS
    assert events == ["guard", "business"]


def test_with_services_stops_tick_when_guard_yields_action_power(tmp_path):
    events = []

    def guard():
        events.append("guard-click")
        yield 1
        events.append("guard-idle")
        yield 0

    def business():
        events.append("business")

    runner = BehaviorTreeRunner(
        Root(WithServices(Every(60, child=Action(business)), Action(guard))),
        tmp_path / "state.json",
    )

    assert runner.run_once() == Status.RUNNING
    assert events == ["guard-click"]
    assert runner.run_once() == Status.SUCCESS
    assert events == ["guard-click", "guard-idle", "business"]


def test_with_services_uses_action_budget_before_business(tmp_path):
    events = []

    def guard():
        events.append("guard-1")
        yield 1
        events.append("guard-2")
        yield 1

    def business():
        events.append("business")

    runner = BehaviorTreeRunner(
        Root(WithServices(Every(60, child=Action(business)), Action(guard))),
        tmp_path / "state.json",
        action_budget=3,
    )

    assert runner.run_once() == Status.SUCCESS
    assert events == ["guard-1", "guard-2", "business"]


def test_memory_selector_continues_running_child(tmp_path):
    events = []

    def first():
        events.append("first")
        yield
        events.append("first-end")

    tree = Root(MemorySelector(Action(first), Action(lambda: events.append("second"))))
    runner = BehaviorTreeRunner(tree, tmp_path / "state.json")

    assert runner.run_once() == Status.RUNNING
    assert runner.run_once() == Status.SUCCESS
    assert events == ["first", "first-end"]


def test_error_without_retry_raises_and_records_runner_state(tmp_path):
    clock = FakeClock("2026-04-26 00:00:00")

    def broken():
        raise RuntimeError("boom")

    runner = BehaviorTreeRunner(Root(Daily("05:00", Action(broken))), tmp_path / "state.json", now_func=clock)

    with pytest.raises(RuntimeError, match="boom"):
        runner.run_once()

    assert runner.state["last_error_at"] == "2026-04-26 00:00:00"
    assert "RuntimeError: boom" in runner.state["last_error"]


def test_error_without_retry_notifies_and_marks_unhandled(tmp_path):
    clock = FakeClock("2026-04-26 00:00:00")
    reports = []

    def broken():
        raise RuntimeError("boom")

    def on_error(err, *, runner, node, handled):
        reports.append((str(err), node, handled, runner.state["last_error_at"]))

    runner = BehaviorTreeRunner(
        Root(Daily("05:00", Action(broken))),
        tmp_path / "state.json",
        now_func=clock,
        on_error=on_error,
    )

    with pytest.raises(RuntimeError, match="boom"):
        runner.run_once()

    assert reports == [("boom", None, False, "2026-04-26 00:00:00")]
    assert runner.state["last_error_handled"] is False
    assert runner.state["last_error_node"] is None


def test_retry_turns_exception_into_next_wake(tmp_path):
    clock = FakeClock("2026-04-26 00:00:00")
    events = []

    def broken():
        events.append("broken")
        raise RuntimeError("boom")

    def other():
        events.append("other")

    tree = Root(
        ReactiveSelector(
            Daily("05:00", Retry(Action(broken), seconds=10)),
            Daily("05:00", Action(other)),
        )
    )
    runner = BehaviorTreeRunner(tree, tmp_path / "state.json", now_func=clock)

    assert runner.run_once() == Status.SUCCESS
    assert runner.run_once() == Status.SUCCESS
    assert events == ["broken", "other"]

    assert any(v.get("next_run_at") == "2026-04-26 00:00:10" for v in runner.state["nodes"].values())
    retry_state = next(v for k, v in runner.memory_state["nodes"].items() if "Retry" in k)
    assert retry_state["retry_at"] == "2026-04-26 00:00:10"


def test_retry_notifies_and_marks_handled_error(tmp_path):
    clock = FakeClock("2026-04-26 00:00:00")
    reports = []

    def broken():
        raise RuntimeError("boom")

    def on_error(err, *, runner, node, handled):
        reports.append((str(err), node.path, handled, runner.state["last_error_at"]))

    tree = Root(Action(broken).daily("05:00").retry(seconds=10))
    runner = BehaviorTreeRunner(tree, tmp_path / "state.json", now_func=clock, on_error=on_error)

    assert runner.run_once() == Status.SUCCESS

    assert reports == [("boom", "Root/broken/Retry[10s]", True, "2026-04-26 00:00:00")]
    assert runner.state["last_error_handled"] is True
    assert runner.state["last_error_node"] == "Root/broken/Retry[10s]"


def test_successful_retry_clears_recovered_runner_error(tmp_path):
    clock = FakeClock("2026-04-26 00:00:00")
    attempts = 0

    def flaky():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("boom")

    runner = BehaviorTreeRunner(
        Root(Retry(Action(flaky), seconds=10)),
        tmp_path / "state.json",
        now_func=clock,
    )

    assert runner.run_once() == Status.SUCCESS
    assert runner.state["last_error_handled"] is True
    assert "RuntimeError: boom" in runner.state["last_error"]

    clock.advance(seconds=10)
    assert runner.run_once() == Status.SUCCESS
    assert attempts == 2
    assert "last_error" not in runner.state
    assert "last_error_at" not in runner.state
    assert "last_error_handled" not in runner.state
    assert "last_error_node" not in runner.state


def test_node_retry_fluent_api(tmp_path):
    clock = FakeClock("2026-04-26 00:00:00")
    events = []

    def broken():
        events.append("broken")
        raise RuntimeError("boom")

    tree = Root(Daily("05:00", Action(broken).retry(seconds=10)))
    runner = BehaviorTreeRunner(tree, tmp_path / "state.json", now_func=clock)

    assert runner.run_once() == Status.SUCCESS
    assert events == ["broken"]
    retry_state = next(v for k, v in runner.memory_state["nodes"].items() if "Retry" in k)
    assert retry_state["retry_at"] == "2026-04-26 00:00:10"


def test_node_daily_retry_fluent_api_keeps_daily_as_outer_node(tmp_path):
    clock = FakeClock("2026-04-26 00:00:00")
    events = []

    def broken():
        events.append("broken")
        raise RuntimeError("boom")

    tree = Root(Action(broken).daily("05:00").retry(seconds=10))
    runner = BehaviorTreeRunner(tree, tmp_path / "state.json", now_func=clock)

    assert runner.run_once() == Status.SUCCESS
    assert events == ["broken"]
    daily_state = next(v for v in runner.state["nodes"].values() if "next_run_at" in v)
    assert daily_state["next_run_at"] == "2026-04-26 00:00:10"


def test_node_retry_defaults_to_immediate_next_tick(tmp_path):
    clock = FakeClock("2026-04-26 00:00:00")
    events = []

    def broken():
        events.append("broken")
        raise RuntimeError("boom")

    tree = Root(Action(broken).daily("05:00").retry())
    runner = BehaviorTreeRunner(tree, tmp_path / "state.json", now_func=clock)

    assert runner.run_once() == Status.SUCCESS
    assert events == ["broken"]

    daily_state = next(v for v in runner.state["nodes"].values() if "next_run_at" in v)
    retry_state = next(v for k, v in runner.memory_state["nodes"].items() if "Retry" in k)
    assert daily_state["next_run_at"] == "2026-04-26 00:00:00"
    assert retry_state["retry_at"] == "2026-04-26 00:00:00"

    assert runner.run_once() == Status.SUCCESS
    assert events == ["broken", "broken"]


def test_node_daily_uses_action_name_as_default_label(tmp_path):
    def task():
        pass

    runner = BehaviorTreeRunner(Root(Action(task).daily("05:00")), tmp_path / "state.json")

    assert runner.root.children[0].path == "Root/task"


def test_save_state_retries_busy_state_file(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    runner = BehaviorTreeRunner(Root(Action(lambda: None)), state_path)
    runner.state["blackboard"]["value"] = 1
    real_replace = os.replace
    attempts = []

    def flaky_replace(src, dst):
        attempts.append((src, dst))
        if len(attempts) < 3:
            raise PermissionError("busy")
        return real_replace(src, dst)

    monkeypatch.setattr("pyxllib.prog.behavior_tree.os.replace", flaky_replace)
    monkeypatch.setattr("pyxllib.prog.behavior_tree.time.sleep", lambda seconds: None)

    runner.save_state()

    assert len(attempts) == 3
    assert json.loads(state_path.read_text(encoding="utf-8"))["blackboard"]["value"] == 1


def test_timeout_is_handled_as_retryable_error(tmp_path):
    clock = FakeClock("2026-04-26 00:00:00")
    events = []

    def slow(ctx):
        events.append("start")
        yield
        events.append("should-not-resume")
        return ctx.next_time(seconds=60)

    tree = Root(Daily("05:00", Retry(Timeout(Action(slow), seconds=10), seconds=30)))
    runner = BehaviorTreeRunner(tree, tmp_path / "state.json", now_func=clock)

    assert runner.run_once() == Status.RUNNING
    clock.advance(seconds=10)
    assert runner.run_once() == Status.SUCCESS

    assert events == ["start"]
    daily_state = next(v for k, v in runner.state["nodes"].items() if "Daily" in k)
    assert daily_state["next_run_at"] == "2026-04-26 00:00:40"
    retry_state = next(v for k, v in runner.memory_state["nodes"].items() if "Retry" in k)
    assert "TimeoutError" in retry_state["last_error"]



def test_log_retention_uses_dates_and_runs_on_monday_startup(tmp_path, monkeypatch):
    import pyxllib.prog.behavior_tree as bt

    class Monday(datetime.date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 9)

    monkeypatch.setattr(bt.datetime, "date", Monday)
    today = datetime.date.today()
    log_path = tmp_path / 'runner.log'
    expired = tmp_path / f'runner.log.{today - datetime.timedelta(days=91)}'
    recent = tmp_path / f'runner.log.{today - datetime.timedelta(days=1)}'
    unrelated = tmp_path / 'runner.log.evidence'
    for path in (expired, recent, unrelated):
        path.write_text('evidence', encoding='utf-8')
    log_path.write_text('existing log\n', encoding='utf-8')
    runner = BehaviorTreeRunner(Root(Action(lambda: None)), trace=1, log_path=log_path)
    try:
        assert not expired.exists()
        assert recent.exists() and unrelated.exists()
        assert log_path.read_text(encoding='utf-8') == 'existing log\n'
    finally:
        runner.close()


def test_runner_logging_is_isolated_and_closed(tmp_path):
    from loguru import logger
    first_path, second_path = tmp_path / 'first.log', tmp_path / 'second.log'
    first = BehaviorTreeRunner(Root(Action(lambda: None)), trace=1, log_path=first_path)
    second = BehaviorTreeRunner(Root(Action(lambda: None)), trace=1, log_path=second_path)
    first.logger.info('first-only')
    second.logger.info('second-only')
    logger.info('unrelated-global-message')
    first.close()
    second.close()
    first.close()
    assert first_path.read_text(encoding='utf-8').count('first-only') == 1
    assert 'second-only' not in first_path.read_text(encoding='utf-8')
    assert 'unrelated-global-message' not in first_path.read_text(encoding='utf-8')
    assert 'first-only' not in second_path.read_text(encoding='utf-8')
    assert not first.logger.handlers and not second.logger.handlers


def test_runner_log_daily_rotation(tmp_path):
    import time
    from logging.handlers import TimedRotatingFileHandler
    path = tmp_path / 'runner.log'
    runner = BehaviorTreeRunner(Root(Action(lambda: None)), trace=1, log_path=path)
    try:
        runner.logger.info('before-midnight')
        handler = next(h for h in runner.logger.handlers if isinstance(h, TimedRotatingFileHandler))
        handler.rolloverAt = int(time.time()) - 1
        runner.logger.info('after-midnight')
        assert 'before-midnight' not in path.read_text(encoding='utf-8')
        assert 'after-midnight' in path.read_text(encoding='utf-8')
        assert any('before-midnight' in p.read_text(encoding='utf-8') for p in tmp_path.glob('runner.log.*'))
    finally:
        runner.close()



def test_log_cleanup_waits_until_monday_and_catches_up(tmp_path, monkeypatch):
    import pyxllib.prog.behavior_tree as bt
    from logging.handlers import TimedRotatingFileHandler

    class ClockDate(datetime.date):
        current = (2026, 11, 10)  # Tuesday

        @classmethod
        def today(cls):
            return cls(*cls.current)

    monkeypatch.setattr(bt.datetime, "date", ClockDate)
    path = tmp_path / "weekly.log"
    expired = tmp_path / "weekly.log.2026-01-01"
    expired.write_text("old", encoding="utf-8")
    runner = BehaviorTreeRunner(Root(Action(lambda: None)), trace=1, log_path=path)
    try:
        handler = next(h for h in runner.logger.handlers if isinstance(h, TimedRotatingFileHandler))
        assert expired.exists()
        ClockDate.current = (2026, 11, 15)  # Sunday
        assert handler.getFilesToDelete() == []
        ClockDate.current = (2026, 11, 17)  # missed Monday: catch up Tuesday
        assert handler.getFilesToDelete() == [str(expired)]
        assert handler.getFilesToDelete() == []  # only once this week
        ClockDate.current = (2026, 11, 23)  # next Monday
        assert handler.getFilesToDelete() == [str(expired)]
    finally:
        runner.close()



def test_manual_cleanup_ignores_weekday_and_preserves_current_log(tmp_path):
    from pyxllib.prog.behavior_tree import cleanup_expired_logs
    today = datetime.date.today()
    current = tmp_path / "manual.log"
    expired = tmp_path / f"manual.log.{today - datetime.timedelta(days=91)}"
    boundary = tmp_path / f"manual.log.{today - datetime.timedelta(days=90)}"
    unrelated = tmp_path / "manual.log.old"
    for path in (current, expired, boundary, unrelated):
        path.write_bytes(b"evidence")
    result = cleanup_expired_logs(current)
    assert result == {"deleted_files": 1, "freed_bytes": 8, "errors": []}
    assert current.exists() and boundary.exists() and unrelated.exists()
    assert not expired.exists()
    assert cleanup_expired_logs(current)["deleted_files"] == 0


def test_legacy_large_log_is_compacted_into_expiring_archive(tmp_path):
    from pyxllib.prog.behavior_tree import migrate_legacy_log, expired_log_files
    path = tmp_path / 'legacy.log'
    with path.open('wb') as f:
        f.write(b'first error evidence\n')
        f.seek(32 * 1024**2)
        f.write(b'last error evidence\n')
    original_size = path.stat().st_size
    result = migrate_legacy_log(path)
    archive = Path(result['archive'])
    assert not path.exists()
    assert archive.stat().st_size < 5 * 1024**2
    text = archive.read_bytes()
    assert b'first error evidence' in text and b'last error evidence' in text
    assert b'NOT a full archive' in text
    assert result['freed_bytes'] == original_size - archive.stat().st_size
    assert archive in expired_log_files(path, today=datetime.date.today() + datetime.timedelta(days=91))


def test_legacy_small_old_log_is_kept_completely(tmp_path):
    from pyxllib.prog.behavior_tree import migrate_legacy_log
    import time
    path = tmp_path / 'legacy.log'
    path.write_bytes(b'complete historical evidence')
    yesterday = time.time() - 86400
    os.utime(path, (yesterday, yesterday))
    result = migrate_legacy_log(path)
    assert Path(result['archive']).read_bytes() == b'complete historical evidence'
    assert result['freed_bytes'] == 0
    assert not path.exists()
