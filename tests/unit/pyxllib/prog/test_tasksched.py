import datetime

import pytest

from pyxllib.prog.tasksched import TaskScheduler


class FakeClock:
    def __init__(self, value="2026-04-25 00:00:00"):
        self.value = datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")

    def __call__(self):
        return self.value

    def advance(self, **kwargs):
        self.value += datetime.timedelta(**kwargs)


def test_no_arg_and_ctx_tasks_can_run(tmp_path):
    clock = FakeClock()
    events = []
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def no_arg_task():
        events.append("no_arg")

    def ctx_task(ctx):
        events.append("ctx")
        return ctx.next_time(seconds=30)

    scheduler.task(no_arg_task).every(minutes=5)
    scheduler.task(ctx_task).every(minutes=5)

    assert scheduler.run_once()
    assert scheduler.run_once()
    assert events == ["no_arg", "ctx"]

    state = scheduler.state["tasks"]["ctx_task"]
    assert state["next_run_at"] == "2026-04-25 00:00:30"


def test_missing_next_run_at_defaults_to_now_then_daily_chooses_nearest_future(tmp_path):
    clock = FakeClock("2026-04-25 04:00:00")
    events = []
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def task():
        events.append("run")

    scheduler.task(task).daily("00:00", "05:00", "12:00")

    state = next(iter(scheduler.state["tasks"].values()))
    assert state["next_run_at"] == "2026-04-25 04:00:00"

    assert scheduler.run_once()
    assert events == ["run"]
    assert state["next_run_at"] == "2026-04-25 05:00:00"


def test_default_next_time_can_override_first_run_time(tmp_path):
    clock = FakeClock("2026-04-25 04:00:00")
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def task():
        pass

    scheduler.task(task, default_next_time="2026-04-25 05:00:00").daily("05:00")

    state = next(iter(scheduler.state["tasks"].values()))
    assert state["next_run_at"] == "2026-04-25 05:00:00"


def test_default_next_time_can_be_callable(tmp_path):
    clock = FakeClock("2026-04-25 04:00:00")
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def task():
        pass

    def default_next_time(sched):
        return sched.next_time(minutes=30)

    scheduler.task(task, default_next_time=default_next_time).daily("05:00")

    state = next(iter(scheduler.state["tasks"].values()))
    assert state["next_run_at"] == "2026-04-25 04:30:00"


def test_existing_next_run_at_is_loaded_from_state_file(tmp_path):
    state_path = tmp_path / "state.json"
    events = []

    def task():
        events.append("run")

    clock1 = FakeClock("2026-04-25 04:00:00")
    scheduler1 = TaskScheduler(state_path, now_func=clock1)
    scheduler1.task(task, default_next_time="2026-04-25 05:00:00").daily("05:00")

    clock2 = FakeClock("2026-04-25 06:00:00")
    scheduler2 = TaskScheduler(state_path, now_func=clock2)
    scheduler2.task(task).daily("05:00")

    state = next(iter(scheduler2.state["tasks"].values()))
    assert state["next_run_at"] == "2026-04-25 05:00:00"

    assert scheduler2.run_once()
    assert events == ["run"]
    assert state["next_run_at"] == "2026-04-26 05:00:00"


def test_generator_yield_allows_upstream_task_to_preempt_and_resume_downstream(tmp_path):
    clock = FakeClock("2026-04-25 00:00:00")
    events = []
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def upstream(ctx):
        events.append("upstream")
        return ctx.next_time("00:01")

    def downstream(ctx):
        events.append("downstream-1")
        yield
        events.append("downstream-2")
        return ctx.next_time(seconds=60)

    scheduler.task(upstream, default_next_time="2026-04-25 00:01:00").daily("00:01")
    scheduler.task(downstream).every(minutes=5)

    assert scheduler.run_once()
    assert events == ["downstream-1"]

    clock.advance(minutes=1)
    assert scheduler.run_once()
    assert events == ["downstream-1", "upstream"]

    assert scheduler.run_once()
    assert events == ["downstream-1", "upstream", "downstream-2"]


def test_trigger_can_reset_downstream_active_job(tmp_path):
    clock = FakeClock("2026-04-25 00:00:00")
    events = []
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def upstream(ctx):
        events.append("upstream")
        return ctx.next_time("00:01")

    def downstream(ctx):
        events.append("downstream-1")
        yield
        events.append("downstream-2")
        return ctx.next_time(seconds=60)

    scheduler.task(
        upstream,
        default_next_time="2026-04-25 00:01:00",
        触发时重置下游=True,
    ).daily("00:01")
    scheduler.task(downstream).every(minutes=5)

    scheduler.run_once()
    clock.advance(minutes=1)
    scheduler.run_once()
    scheduler.run_once()

    assert events == ["downstream-1", "upstream", "downstream-1"]


def test_downstream_task_can_disable_resume_after_upstream_preemption(tmp_path):
    clock = FakeClock("2026-04-25 00:00:00")
    events = []
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def upstream(ctx):
        events.append("upstream")
        return ctx.next_time("00:01")

    def downstream(ctx):
        events.append("downstream-1")
        yield
        events.append("downstream-2")
        return ctx.next_time(seconds=60)

    scheduler.task(upstream, default_next_time="2026-04-25 00:01:00").daily("00:01")
    scheduler.task(downstream, 被上游插队后续跑=False).every(minutes=5)

    scheduler.run_once()
    clock.advance(minutes=1)
    scheduler.run_once()
    scheduler.run_once()

    assert events == ["downstream-1", "upstream", "downstream-1"]


def test_ready_task_without_preempt_waits_for_downstream_job(tmp_path):
    clock = FakeClock("2026-04-25 00:00:00")
    events = []
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def upstream(ctx):
        events.append("upstream")
        return ctx.next_time("00:01")

    def downstream(ctx):
        events.append("downstream-1")
        yield
        events.append("downstream-2")
        return ctx.next_time(seconds=60)

    scheduler.task(
        upstream,
        default_next_time="2026-04-25 00:01:00",
        触发时可插队=False,
    ).daily("00:01")
    scheduler.task(downstream).every(minutes=5)

    scheduler.run_once()
    clock.advance(minutes=1)
    scheduler.run_once()
    scheduler.run_once()

    assert events == ["downstream-1", "downstream-2", "upstream"]


def test_error_without_retry_raises_and_records_state(tmp_path):
    clock = FakeClock()
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def broken():
        raise RuntimeError("boom")

    scheduler.task(broken).every(minutes=5)

    with pytest.raises(RuntimeError, match="boom"):
        scheduler.run_once()

    state = next(iter(scheduler.state["tasks"].values()))
    assert state["last_error_at"] == "2026-04-25 00:00:00"
    assert "RuntimeError: boom" in state["last_error"]


def test_retry_turns_error_into_next_run_time(tmp_path):
    clock = FakeClock()
    events = []
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def broken():
        events.append("broken")
        raise RuntimeError("boom")

    def other():
        events.append("other")

    scheduler.task(broken).every(minutes=5).retry(seconds=10)
    scheduler.task(other).every(minutes=5)

    assert scheduler.run_once()
    assert scheduler.run_once()
    assert events == ["broken", "other"]

    state = scheduler.state["tasks"]["broken"]
    assert state["next_run_at"] == "2026-04-25 00:00:10"
    assert state["error_count"] == 1


def test_same_function_requires_explicit_distinct_names(tmp_path):
    clock = FakeClock()
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def shared():
        pass

    scheduler.task(shared, name="羊驼_游历").daily("05:00")
    scheduler.task(shared, name="驼二_游历").daily("05:00")

    with pytest.raises(ValueError, match="duplicate task name"):
        scheduler.task(shared, name="羊驼_游历").daily("05:00")

    assert set(scheduler.state["tasks"]) == {"羊驼_游历", "驼二_游历"}


def test_repeating_same_function_without_name_raises(tmp_path):
    clock = FakeClock()
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def shared():
        pass

    scheduler.task(shared).daily("05:00")

    with pytest.raises(ValueError, match="explicit name"):
        scheduler.task(shared).daily("05:00")


def test_timeout_is_handled_as_error_and_can_retry(tmp_path):
    clock = FakeClock()
    events = []
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def slow(ctx):
        events.append("start")
        yield
        events.append("should-not-resume")
        return ctx.next_time(seconds=60)

    scheduler.task(slow).every(minutes=5).timeout(seconds=10).retry(seconds=30)

    assert scheduler.run_once()
    clock.advance(seconds=10)
    assert scheduler.run_once()

    assert events == ["start"]
    state = next(iter(scheduler.state["tasks"].values()))
    assert state["next_run_at"] == "2026-04-25 00:00:40"
    assert "TimeoutError" in state["last_error"]


def test_idle_sleep_uses_ratio_before_next_task(tmp_path):
    clock = FakeClock("2026-04-25 00:00:00")
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def task():
        pass

    scheduler.task(task, default_next_time="2026-04-25 00:10:00").daily("00:10")

    assert scheduler._idle_sleep_seconds(tick_seconds=1) == 480


def test_idle_sleep_does_not_oversleep_near_task(tmp_path):
    clock = FakeClock("2026-04-25 00:00:00")
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def task():
        pass

    scheduler.task(task, default_next_time="2026-04-25 00:00:01").daily("00:10")

    assert scheduler._idle_sleep_seconds(tick_seconds=2) == 1


def test_idle_sleep_can_be_capped(tmp_path):
    clock = FakeClock("2026-04-25 00:00:00")
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def task():
        pass

    scheduler.task(task, default_next_time="2026-04-25 01:00:00").daily("01:00")

    assert scheduler._idle_sleep_seconds(tick_seconds=1, max_idle_seconds=60) == 60


def test_run_forever_calls_start_and_wakeup_hooks(monkeypatch, tmp_path):
    clock = FakeClock("2026-04-25 00:00:00")
    events = []
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def task(ctx):
        events.append("task")
        return ctx.next_time("00:10")

    def on_start():
        events.append("start")

    def on_wakeup(sched):
        events.append(f"wakeup:{sched.now():%H:%M:%S}")

    scheduler.on_start(on_start).on_wakeup(on_wakeup)
    scheduler.task(task, default_next_time="2026-04-25 00:00:10").daily("00:10")

    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            clock.advance(seconds=10)
            return
        raise KeyboardInterrupt

    monkeypatch.setattr("pyxllib.prog.tasksched.time.sleep", fake_sleep)

    with pytest.raises(KeyboardInterrupt):
        scheduler.run_forever(tick_seconds=1)

    assert sleeps == [8.0, 1]
    assert events == ["start", "wakeup:00:00:10", "task"]


def test_active_only_task_does_not_wake_idle_scheduler(tmp_path):
    clock = FakeClock("2026-04-25 00:00:00")
    events = []
    scheduler = TaskScheduler(tmp_path / "state.json", now_func=clock)

    def guard(ctx):
        events.append("guard")
        return ctx.next_time(seconds=1)

    def main(ctx):
        events.append("main")
        return ctx.next_time("00:10")

    scheduler.task(guard, priority=200, 仅活跃时运行=True).every(seconds=1)
    scheduler.task(main, default_next_time="2026-04-25 00:10:00").daily("00:10")

    assert not scheduler.run_once()
    assert events == []
    assert scheduler._idle_sleep_seconds(tick_seconds=1) == 480

    clock.advance(minutes=10)
    assert scheduler.run_once()
    assert scheduler.run_once()
    assert events == ["guard", "main"]

    clock.advance(seconds=1)
    assert not scheduler.run_once()
    assert events == ["guard", "main"]
