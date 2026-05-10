#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2026/04/25

"""持久化协作式任务调度器。

这个模块提供一个轻量级调度内核，面向“状态文件 + task 定义 + job 运行实例”的场景。
它不依赖 APScheduler，也不理解具体业务，只负责按时间和 task 顺序推进 callable。
"""

import datetime
import inspect
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import GeneratorType
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "Job",
    "NextRun",
    "TaskContext",
    "TaskScheduler",
    "TaskSpec",
]

_MISSING = object()
_TIME_RE = re.compile(r"^(-)?(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$")
_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
)


def _format_time(value: Optional[datetime.datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _parse_time(value: Any) -> Optional[datetime.datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime.datetime):
        return value.replace(microsecond=0)
    if isinstance(value, datetime.date):
        return datetime.datetime.combine(value, datetime.time())
    if isinstance(value, (int, float)):
        return datetime.datetime.fromtimestamp(value).replace(microsecond=0)

    text = str(value).strip()
    if not text:
        return None
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return datetime.datetime.fromisoformat(text).replace(microsecond=0)
    except ValueError:
        return None


def _duration_seconds(value: Any = _MISSING, *, days=0, hours=0, minutes=0, seconds=0) -> Optional[float]:
    if value is None:
        return None
    if value is not _MISSING:
        seconds += value
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise ValueError("duration must be greater than 0")
    return float(total)


def _resolve_anchor(anchor: str, base_time: datetime.datetime) -> datetime.datetime:
    text = str(anchor).strip()
    match = _TIME_RE.match(text)
    if not match:
        raise ValueError(f"invalid daily anchor: {anchor!r}")

    is_prev, hour, minute, second = match.groups()
    hour, minute = int(hour), int(minute)
    second = int(second or 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(f"invalid daily anchor: {anchor!r}")

    target = base_time.replace(hour=hour, minute=minute, second=second, microsecond=0)
    if is_prev:
        if target > base_time:
            target -= datetime.timedelta(days=1)
    elif target <= base_time:
        target += datetime.timedelta(days=1)
    return target


def _compute_next_time(
    *anchors: str,
    base_time: Optional[datetime.datetime] = None,
    days=0,
    hours=0,
    minutes=0,
    seconds=0,
) -> datetime.datetime:
    current = (base_time or datetime.datetime.now()).replace(microsecond=0)
    delta = datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    if not anchors:
        return current + delta

    targets = [_resolve_anchor(anchor, current) + delta for anchor in anchors]
    return min(targets)


def _default_task_name(func: Callable) -> str:
    base = getattr(func, "func", func)
    return getattr(base, "__name__", base.__class__.__name__)


def _accepts_ctx(func: Callable) -> bool:
    signature = inspect.signature(func)
    params = list(signature.parameters.values())
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required_positional = [p for p in positional if p.default is inspect.Parameter.empty]
    required_keyword_only = [
        p
        for p in params
        if p.kind == inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty
    ]

    if required_keyword_only:
        raise TypeError(f"task function {func!r} has unsupported required keyword-only parameters")
    if len(required_positional) == 0:
        return False
    if len(required_positional) == 1 and len(positional) == 1:
        return True
    raise TypeError(f"task function {func!r} must accept either no required parameter or exactly one ctx parameter")


def _accepts_scheduler_arg(func: Callable) -> bool:
    signature = inspect.signature(func)
    params = list(signature.parameters.values())
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required_positional = [p for p in positional if p.default is inspect.Parameter.empty]
    required_keyword_only = [
        p
        for p in params
        if p.kind == inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty
    ]

    if required_keyword_only:
        raise TypeError(f"hook function {func!r} has unsupported required keyword-only parameters")
    if len(required_positional) == 0:
        return False
    if len(required_positional) == 1 and len(positional) == 1:
        return True
    raise TypeError(f"hook function {func!r} must accept either no required parameter or exactly one scheduler parameter")


@dataclass
class _SchedulerHook:
    func: Callable
    takes_scheduler: bool = False


@dataclass
class NextRun:
    """任务函数返回的下一次运行时间。"""

    run_at: datetime.datetime


class TaskContext:
    """传递给任务函数的最小调度上下文。"""

    def __init__(self, scheduler: "TaskScheduler", task: "TaskSpec", job: "Job"):
        self.scheduler = scheduler
        self.task = task
        self.job = job
        self.next_run_at = None

    def next_time(self, *anchors: str, base_time=None, days=0, hours=0, minutes=0, seconds=0) -> NextRun:
        base = _parse_time(base_time) if base_time is not None else self.scheduler.now()
        run_at = _compute_next_time(
            *anchors,
            base_time=base,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
        )
        self.next_run_at = run_at
        return NextRun(run_at)


@dataclass
class TaskSpec:
    """配置期的任务定义。"""

    scheduler: "TaskScheduler"
    func: Callable
    name: str
    order: int
    priority: int = 100
    can_preempt_on_trigger: bool = True
    invalidate_downstream_on_trigger: bool = False
    resume_after_upstream_preempted: bool = True
    active_only: bool = False
    takes_ctx: bool = False
    schedule_kind: Optional[str] = None
    daily_anchors: Tuple[str, ...] = ()
    every_seconds: Optional[float] = None
    retry_seconds: Optional[float] = None
    timeout_seconds: Optional[float] = None

    @property
    def state(self) -> Dict[str, Any]:
        return self.scheduler._task_state(self.name)

    def daily(self, *anchors: str) -> "TaskSpec":
        if not anchors:
            raise ValueError("daily() requires at least one anchor")
        self.schedule_kind = "daily"
        self.daily_anchors = tuple(str(x) for x in anchors)
        self.scheduler.save_state()
        return self

    def every(self, seconds=_MISSING, *, days=0, hours=0, minutes=0) -> "TaskSpec":
        self.schedule_kind = "every"
        self.every_seconds = _duration_seconds(seconds, days=days, hours=hours, minutes=minutes)
        self.scheduler.save_state()
        return self

    def retry(self, seconds=_MISSING, *, days=0, hours=0, minutes=0) -> "TaskSpec":
        self.retry_seconds = _duration_seconds(seconds, days=days, hours=hours, minutes=minutes)
        return self

    def timeout(self, seconds=_MISSING, *, days=0, hours=0, minutes=0) -> "TaskSpec":
        self.timeout_seconds = _duration_seconds(seconds, days=days, hours=hours, minutes=minutes)
        return self

    def _sort_key(self) -> Tuple[int, int]:
        return -int(self.priority), int(self.order)

    def is_due(self, now: datetime.datetime) -> bool:
        next_run_at = _parse_time(self.state.get("next_run_at"))
        return next_run_at is not None and next_run_at <= now

    def _next_from_schedule(self, base_time: Optional[datetime.datetime] = None) -> Optional[datetime.datetime]:
        base = (base_time or self.scheduler.now()).replace(microsecond=0)
        if self.schedule_kind == "daily":
            return _compute_next_time(*self.daily_anchors, base_time=base)
        if self.schedule_kind == "every":
            return base + datetime.timedelta(seconds=self.every_seconds or 0)
        return None

    def _set_next_run_at(self, value: Optional[datetime.datetime]) -> None:
        self.state["next_run_at"] = _format_time(value)

    def _call(self, ctx: TaskContext) -> Any:
        if self.takes_ctx:
            return self.func(ctx)
        return self.func()


@dataclass
class Job:
    """运行期的一次任务实例。"""

    task: TaskSpec
    started_at: datetime.datetime
    ctx: Optional[TaskContext] = None
    generator: Optional[GeneratorType] = None
    created_from_state: Dict[str, Any] = field(default_factory=dict)

    def elapsed_seconds(self, now: datetime.datetime) -> float:
        return (now - self.started_at).total_seconds()


class TaskScheduler:
    """持久化协作式任务调度器。"""

    def __init__(self, state_path, *, log_path=None, trace=0, now_func=None):
        self.state_path = Path(state_path)
        self.trace = int(trace)
        self.now_func = now_func
        self.tasks = []
        self.task_map = {}
        self._registered_func_ids = set()
        self.active_jobs = {}
        self._start_hooks = []
        self._wakeup_hooks = []
        self._order_counter = 0
        self.state = self.load_state()
        self.logger = self._create_logger(log_path)

    def now(self) -> datetime.datetime:
        return (self.now_func() if self.now_func else datetime.datetime.now()).replace(microsecond=0)

    def next_time(self, *anchors: str, base_time=None, days=0, hours=0, minutes=0, seconds=0) -> datetime.datetime:
        base = _parse_time(base_time) if base_time is not None else self.now()
        return _compute_next_time(
            *anchors,
            base_time=base,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
        )

    def load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
        data.setdefault("tasks", {})
        return data

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_name(f"{self.state_path.name}.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.state_path)

    def task(
        self,
        func: Callable,
        *,
        name: Optional[str] = None,
        default_next_time=None,
        priority: int = 100,
        触发时可插队: bool = True,
        触发时重置下游: bool = False,
        被上游插队后续跑: bool = True,
        仅活跃时运行: bool = False,
    ) -> TaskSpec:
        func_id = id(getattr(func, "func", func))
        if name is None and func_id in self._registered_func_ids:
            raise ValueError("duplicate task function requires explicit name")

        name = name or _default_task_name(func)
        if name in self.task_map:
            raise ValueError(f"duplicate task name: {name}")

        spec = TaskSpec(
            scheduler=self,
            func=func,
            name=name,
            order=self._order_counter,
            priority=priority,
            can_preempt_on_trigger=bool(触发时可插队),
            invalidate_downstream_on_trigger=bool(触发时重置下游),
            resume_after_upstream_preempted=bool(被上游插队后续跑),
            active_only=bool(仅活跃时运行),
            takes_ctx=_accepts_ctx(func),
        )
        self._order_counter += 1
        self.tasks.append(spec)
        self.task_map[name] = spec
        self._registered_func_ids.add(func_id)
        state = self._task_state(name)
        state["name"] = spec.name
        if state.get("next_run_at") is None:
            spec._set_next_run_at(self._resolve_default_next_time(default_next_time))
        self.save_state()
        return spec

    def on_start(self, func: Callable) -> "TaskScheduler":
        """注册 `run_forever` 启动时执行的生命周期钩子。

        :param Callable func: 钩子函数，支持无参数或接收一个 scheduler 参数。
        :return TaskScheduler: 返回调度器本身，便于链式配置。
        """
        self._start_hooks.append(_SchedulerHook(func, _accepts_scheduler_arg(func)))
        return self

    def on_wakeup(self, func: Callable) -> "TaskScheduler":
        """注册空闲等待后重新进入 active 阶段前执行的生命周期钩子。

        :param Callable func: 钩子函数，支持无参数或接收一个 scheduler 参数。
        :return TaskScheduler: 返回调度器本身，便于链式配置。
        """
        self._wakeup_hooks.append(_SchedulerHook(func, _accepts_scheduler_arg(func)))
        return self

    def run_once(self) -> bool:
        candidate = self._select_candidate()
        if candidate is None:
            return False

        if isinstance(candidate, TaskSpec):
            self._prepare_new_job(candidate)
            job = self.active_jobs[candidate.name]
        else:
            job = candidate

        try:
            self._ensure_not_timed_out(job)
            finished, result = self._step_job(job)
            self._ensure_not_timed_out(job)
        except Exception as err:
            self._handle_error(job, err)
            return True

        if finished:
            self._finish_job(job, result)
        return True

    def run_until_idle(self, max_steps=1000) -> int:
        steps = 0
        while steps < max_steps and self.run_once():
            steps += 1
        return steps

    def run_forever(self, tick_seconds=1, *, idle_ratio=0.8, max_idle_seconds=None) -> None:
        self._run_hooks("on_start", self._start_hooks)
        was_idle = False

        while True:
            if was_idle and self._select_candidate() is not None:
                self._run_hooks("on_wakeup", self._wakeup_hooks)
                was_idle = False

            if self.run_once():
                time.sleep(tick_seconds)
            else:
                was_idle = True
                time.sleep(self._idle_sleep_seconds(tick_seconds, idle_ratio=idle_ratio, max_idle_seconds=max_idle_seconds))

    def _task_state(self, name: str) -> Dict[str, Any]:
        return self.state.setdefault("tasks", {}).setdefault(name, {})

    def _resolve_default_next_time(self, default_next_time) -> datetime.datetime:
        if default_next_time is None:
            return self.now()
        if callable(default_next_time):
            signature = inspect.signature(default_next_time)
            required = [
                p
                for p in signature.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ]
            if not required:
                value = default_next_time()
            elif len(required) == 1:
                value = default_next_time(self)
            else:
                raise TypeError("default_next_time callable must accept either no parameter or one scheduler parameter")
        else:
            value = default_next_time
        if isinstance(value, NextRun):
            return value.run_at
        parsed = _parse_time(value)
        if parsed is None:
            raise ValueError(f"invalid default_next_time: {default_next_time!r}")
        return parsed

    def _create_logger(self, log_path) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        if self.trace <= 0:
            logger.addHandler(logging.NullHandler())
            return logger

        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

    def _run_hooks(self, hook_type: str, hooks: List[_SchedulerHook]) -> None:
        for hook in hooks:
            if self.trace >= 1:
                self.logger.info("%s hook: %s", hook_type, _default_task_name(hook.func))
            result = hook.func(self) if hook.takes_scheduler else hook.func()
            if isinstance(result, GeneratorType):
                for _ in result:
                    pass

    def _select_candidate(self) -> Optional[Any]:
        now = self.now()
        has_regular_activity = self._has_regular_activity(now)
        candidates = list(self.active_jobs.values())
        for task in self.tasks:
            if task.name in self.active_jobs:
                continue
            if task.active_only and not has_regular_activity:
                continue
            if not task.is_due(now):
                continue
            if not task.can_preempt_on_trigger and self._has_downstream_active_job(task):
                continue
            candidates.append(task)

        if not candidates:
            return None
        return min(candidates, key=lambda item: item.task._sort_key() if isinstance(item, Job) else item._sort_key())

    def _next_scheduled_time(self) -> Optional[datetime.datetime]:
        times = []
        for task in self.tasks:
            if task.active_only:
                continue
            value = _parse_time(task.state.get("next_run_at"))
            if value is not None:
                times.append(value)
        if not times:
            return None
        return min(times)

    def _idle_sleep_seconds(self, tick_seconds=1, *, idle_ratio=0.8, max_idle_seconds=None) -> float:
        target_time = self._next_scheduled_time()
        if target_time is None:
            return float(tick_seconds)

        remaining = (target_time - self.now()).total_seconds()
        if remaining <= 0:
            return 0.0

        sleep_seconds = min(max(remaining * idle_ratio, float(tick_seconds)), remaining)
        if max_idle_seconds is not None:
            sleep_seconds = min(sleep_seconds, float(max_idle_seconds))
        return sleep_seconds

    def _has_regular_activity(self, now: datetime.datetime) -> bool:
        if any(not job.task.active_only for job in self.active_jobs.values()):
            return True
        return any(not task.active_only and task.is_due(now) for task in self.tasks)

    def _has_downstream_active_job(self, task: TaskSpec) -> bool:
        return any(self._is_before(task, job.task) for job in self.active_jobs.values())

    @staticmethod
    def _is_before(left: TaskSpec, right: TaskSpec) -> bool:
        return left._sort_key() < right._sort_key()

    def _prepare_new_job(self, task: TaskSpec) -> None:
        for key, job in list(self.active_jobs.items()):
            if not self._is_before(task, job.task):
                continue
            if task.invalidate_downstream_on_trigger or not job.task.resume_after_upstream_preempted:
                if self.trace >= 1:
                    self.logger.info("reset downstream job: %s -> %s", task.name, job.task.name)
                del self.active_jobs[key]

        job = Job(task=task, started_at=self.now())
        job.ctx = TaskContext(self, task, job)
        self.active_jobs[task.name] = job
        state = task.state
        state["last_run_at"] = _format_time(job.started_at)
        if self.trace >= 1:
            self.logger.info("job start: %s", task.name)
        self.save_state()

    def _step_job(self, job: Job) -> Tuple[bool, Any]:
        if job.generator is not None:
            try:
                value = next(job.generator)
                self._trace_step(job)
                return False, value
            except StopIteration as err:
                return True, err.value

        result = job.task._call(job.ctx)
        if isinstance(result, GeneratorType):
            job.generator = result
            try:
                value = next(job.generator)
                self._trace_step(job)
                return False, value
            except StopIteration as err:
                return True, err.value
        return True, result

    def _finish_job(self, job: Job, result: Any) -> None:
        next_run_at = self._resolve_next_run_at(job, result)
        state = job.task.state
        state["last_success_at"] = _format_time(self.now())
        state["last_error_at"] = None
        state["last_error"] = None
        state["error_count"] = 0
        job.task._set_next_run_at(next_run_at)
        self.active_jobs.pop(job.task.name, None)
        if self.trace >= 1:
            self.logger.info("job finish: %s, next_run_at=%s", job.task.name, _format_time(next_run_at))
        self.save_state()

    def _resolve_next_run_at(self, job: Job, result: Any) -> Optional[datetime.datetime]:
        if isinstance(result, NextRun):
            return result.run_at
        if job.ctx and job.ctx.next_run_at is not None:
            return job.ctx.next_run_at
        return job.task._next_from_schedule(base_time=self.now())

    def _ensure_not_timed_out(self, job: Job) -> None:
        timeout_seconds = job.task.timeout_seconds
        if timeout_seconds is None:
            return
        if job.elapsed_seconds(self.now()) >= timeout_seconds:
            raise TimeoutError(f"job timeout after {timeout_seconds:g}s: {job.task.name}")

    def _handle_error(self, job: Job, err: Exception) -> None:
        task = job.task
        state = task.state
        state["last_error_at"] = _format_time(self.now())
        state["last_error"] = "".join(traceback.format_exception_only(type(err), err)).strip()
        state["error_count"] = int(state.get("error_count") or 0) + 1
        self.active_jobs.pop(task.name, None)

        if task.retry_seconds is None:
            self.save_state()
            if self.trace >= 1:
                self.logger.exception("job error: %s", task.name)
            raise err

        next_run_at = self.now() + datetime.timedelta(seconds=task.retry_seconds)
        task._set_next_run_at(next_run_at)
        if self.trace >= 1:
            self.logger.warning("job error: %s, retry_at=%s", task.name, _format_time(next_run_at))
        self.save_state()

    def _trace_step(self, job: Job) -> None:
        if self.trace < 2 or job.generator is None:
            return
        frame = job.generator.gi_frame
        if frame is None:
            return
        self.logger.debug("job yield: %s at %s:%s", job.task.name, frame.f_code.co_filename, frame.f_lineno)
