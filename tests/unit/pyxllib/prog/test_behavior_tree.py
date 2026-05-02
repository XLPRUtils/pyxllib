import datetime
import inspect
import json
from pathlib import Path

import pytest

from pyxllib.prog.behavior_tree import (
    Action,
    BehaviorTreeRunner,
    Daily,
    Every,
    IdleUntilNextWake,
    MemorySelector,
    Once,
    ReactiveSelector,
    Retry,
    Root,
    Sequence,
    Status,
    Timeout,
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

