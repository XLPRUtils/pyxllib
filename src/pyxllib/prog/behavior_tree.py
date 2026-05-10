#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2026/04/26

"""behavior_tree.py - 持久化协作式行为树。

本模块提供一个小型行为树内核，面向 GUI 自动化、批处理编排、爬虫等需要
`tick + yield + 状态文件` 的场景。框架只提供通用节点、黑板、时间装饰器和
运行器；账号、课程、窗口、图片识别等业务语义都应留在业务层函数里。

核心用法示例：

>>> events = []
>>> tree = Root(Sequence(Action(lambda: events.append("a")), Action(lambda: events.append("b"))))
>>> runner = BehaviorTreeRunner(tree, state_path=None)
>>> runner.run_once() == Status.SUCCESS
True
>>> events
['a', 'b']
"""

from __future__ import annotations

import datetime
import inspect
import json
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import GeneratorType
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "Action",
    "BehaviorTreeRunner",
    "BTreeRunner",
    "Daily",
    "DynamicTime",
    "Every",
    "IdleUntilNextWake",
    "MemorySelector",
    "MemorySequence",
    "NextWake",
    "Node",
    "Once",
    "ReactiveSelector",
    "Root",
    "Retry",
    "Selector",
    "Sequence",
    "Status",
    "Timeout",
    "TreeContext",
    "Window",
    "WithServices",
]

_MISSING = object()
_TIME_RE = re.compile(r"^(-)?(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$")
_DATETIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")


class Status(Enum):
    """行为树节点的一次 tick 结果。"""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"
    SKIP = "SKIP"


@dataclass
class NextWake:
    """节点返回的下一次唤醒时间。"""

    run_at: datetime.datetime


@dataclass
class _TraceInterrupt:
    kind: str
    location: Optional["_SourceLocation"]
    node_path: str = ""


@dataclass
class _SourceLocation:
    filename: str
    lineno: int
    function: str

    def brief(self) -> str:
        return f"{Path(self.filename).name}:{self.lineno}"


def _format_time(value: Optional[datetime.datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")


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


def _duration_seconds(value: Any = _MISSING, *, days=0, hours=0, minutes=0, seconds=0) -> float:
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
    return min(_resolve_anchor(anchor, current) + delta for anchor in anchors)


def _normalize_daily_anchors(anchors: Tuple[Any, ...]) -> Tuple[str, ...]:
    if len(anchors) == 1 and isinstance(anchors[0], (list, tuple)):
        anchors = tuple(anchors[0])
    if not anchors:
        raise ValueError("Daily requires at least one anchor")
    return tuple(str(x) for x in anchors)


def _default_callable_label(func: Callable) -> str:
    base = getattr(func, "func", func)
    name = getattr(base, "__name__", None)
    if name and name != "<lambda>":
        return name
    return base.__class__.__name__


def _call_accepts_ctx(func: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
    signature = inspect.signature(func)
    try:
        signature.bind(*args, **kwargs)
        return False
    except TypeError:
        pass
    try:
        signature.bind(object(), *args, **kwargs)
        return True
    except TypeError as err:
        raise TypeError(f"action function {func!r} cannot be called with provided args") from err


def _coerce_status(value: Any) -> Status:
    if isinstance(value, Status):
        return value
    return Status.SUCCESS


def _source_location(filename: str, lineno: int, function: str) -> _SourceLocation:
    return _SourceLocation(filename, lineno, function)


def _frame_source_location(frame) -> _SourceLocation:
    return _source_location(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)


def _deepest_generator(generator: GeneratorType) -> GeneratorType:
    current = generator
    while isinstance(current.gi_yieldfrom, GeneratorType):
        current = current.gi_yieldfrom
    return current


def _suspended_generator_location(generator: GeneratorType) -> Optional[_SourceLocation]:
    frame = _deepest_generator(generator).gi_frame
    if frame is None:
        return None
    return _frame_source_location(frame)


def _is_within_time(window_start: str, window_end: str, *, base_time: datetime.datetime) -> bool:
    def parse_clock(value: str) -> int:
        match = _TIME_RE.match(str(value).strip())
        if not match or match.group(1):
            raise ValueError(f"invalid window anchor: {value!r}")
        hour, minute = int(match.group(2)), int(match.group(3))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"invalid window anchor: {value!r}")
        return hour * 60 + minute

    current = base_time.hour * 60 + base_time.minute
    start = parse_clock(window_start)
    end = parse_clock(window_end)
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


class TreeContext:
    """一次 tick 过程中的运行上下文。"""

    def __init__(self, runner: "BehaviorTreeRunner"):
        self.runner = runner
        self.blackboard = runner.state.setdefault("blackboard", {})
        self.next_run_at = None
        self._slept = False
        self.trace_interrupt = None

    def now(self) -> datetime.datetime:
        return self.runner.now()

    def next_time(self, *anchors: str, base_time=None, days=0, hours=0, minutes=0, seconds=0) -> NextWake:
        base = _parse_time(base_time) if base_time is not None else self.now()
        run_at = _compute_next_time(
            *anchors,
            base_time=base,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
        )
        self.next_run_at = run_at
        return NextWake(run_at)

    def node_state(self, node: "Node") -> Dict[str, Any]:
        return self.runner.node_state(node)

    def sleep(self, seconds: float) -> None:
        self._slept = True
        self.runner.sleep(seconds)


class Node:
    """行为树节点基类。"""

    def __init__(self, *children: "Node", label: Optional[str] = None, persist: bool = False):
        self.children = list(children)
        self.label = label
        self.persist = bool(persist)
        self.path = ""
        self.parent = None
        self.last_status = None

    def tick(self, ctx: TreeContext) -> Status:
        raise NotImplementedError

    def reset(self, ctx: TreeContext) -> None:
        self.last_status = None
        for child in self.children:
            child.reset(ctx)

    def state(self, ctx: TreeContext) -> Dict[str, Any]:
        return ctx.node_state(self)

    def default_label(self) -> str:
        return self.__class__.__name__

    def next_wake(self, ctx: TreeContext) -> Optional[datetime.datetime]:
        times = [child.next_wake(ctx) for child in self.children]
        times = [x for x in times if x is not None]
        return min(times) if times else None

    def is_active(self, ctx: TreeContext) -> bool:
        return self.last_status == Status.RUNNING or any(child.is_active(ctx) for child in self.children)

    def retry(
        self,
        seconds=_MISSING,
        *,
        days=0,
        hours=0,
        minutes=0,
        label: Optional[str] = None,
        persist: bool = False,
    ) -> "Retry":
        return Retry(self, seconds, days=days, hours=hours, minutes=minutes, label=label, persist=persist)

    def timeout(
        self,
        seconds=_MISSING,
        *,
        days=0,
        hours=0,
        minutes=0,
        label: Optional[str] = None,
        persist: bool = False,
    ) -> "Timeout":
        return Timeout(self, seconds, days=days, hours=hours, minutes=minutes, label=label, persist=persist)

    def daily(
        self,
        *anchors: str,
        label: Optional[str] = None,
        persist: bool = True,
        default_next_time=None,
        start: str = "run",
        enabled: bool = True,
        on_schedule: Optional[Callable[[TreeContext, datetime.datetime], None]] = None,
    ) -> "Daily":
        return Daily(
            *anchors,
            child=self,
            label=label or self.default_label(),
            persist=persist,
            default_next_time=default_next_time,
            start=start,
            enabled=enabled,
            on_schedule=on_schedule,
        )

    def every(
        self,
        seconds=_MISSING,
        *,
        days=0,
        hours=0,
        minutes=0,
        label: Optional[str] = None,
        persist: bool = False,
        default_next_time=None,
        enabled: bool = True,
        on_schedule: Optional[Callable[[TreeContext, datetime.datetime], None]] = None,
    ) -> "Every":
        if seconds is _MISSING:
            return Every(
                child=self,
                days=days,
                hours=hours,
                minutes=minutes,
                label=label,
                persist=persist,
                default_next_time=default_next_time,
                enabled=enabled,
                on_schedule=on_schedule,
            )
        return Every(
            seconds,
            child=self,
            days=days,
            hours=hours,
            minutes=minutes,
            label=label,
            persist=persist,
            default_next_time=default_next_time,
            enabled=enabled,
            on_schedule=on_schedule,
        )

    def _record(self, status: Status) -> Status:
        self.last_status = status
        return status


class Root(Node):
    """行为树根节点。"""

    def tick(self, ctx: TreeContext) -> Status:
        if not self.children:
            return self._record(Status.SKIP)
        if len(self.children) == 1:
            return self._record(self.children[0].tick(ctx))
        return self._record(Sequence(*self.children).tick(ctx))


class Action(Node):
    """包装普通函数或生成器函数的动作节点。"""

    def __init__(self, func: Callable, *args, label: Optional[str] = None, persist: bool = False, **kwargs):
        super().__init__(label=label, persist=persist)
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.takes_ctx = _call_accepts_ctx(func, args, kwargs)
        self.generator = None

    def default_label(self) -> str:
        return self.label or _default_callable_label(self.func)

    def reset(self, ctx: TreeContext) -> None:
        super().reset(ctx)
        self.generator = None

    def tick(self, ctx: TreeContext) -> Status:
        if self.generator is None:
            result = self.func(ctx, *self.args, **self.kwargs) if self.takes_ctx else self.func(*self.args, **self.kwargs)
            if isinstance(result, NextWake):
                ctx.next_run_at = result.run_at
                return self._record(Status.SUCCESS)
            if not isinstance(result, GeneratorType):
                return self._record(_coerce_status(result))
            self.generator = result

        try:
            ctx.runner._advance_generator(self.generator)
            if ctx.runner.trace >= 2:
                ctx.trace_interrupt = _TraceInterrupt("yield", _suspended_generator_location(self.generator), self.path)
            return self._record(Status.RUNNING)
        except StopIteration as err:
            self.generator = None
            if isinstance(err.value, NextWake):
                ctx.next_run_at = err.value.run_at
            if ctx.runner.trace >= 2:
                ctx.trace_interrupt = _TraceInterrupt("return", ctx.runner._generator_return_location(), self.path)
            return self._record(_coerce_status(err.value))


class Sequence(Node):
    """按顺序执行子节点，遇到 RUNNING、FAILURE 或 SKIP 即返回。"""

    def tick(self, ctx: TreeContext) -> Status:
        for child in self.children:
            status = child.tick(ctx)
            if status != Status.SUCCESS:
                return self._record(status)
        return self._record(Status.SUCCESS)


class MemorySequence(Node):
    """有记忆的顺序节点，RUNNING 时下次从当前子节点继续。"""

    def __init__(self, *children: Node, label: Optional[str] = None, persist: bool = False):
        super().__init__(*children, label=label, persist=persist)
        self.active_index = 0

    def tick(self, ctx: TreeContext) -> Status:
        state = self.state(ctx)
        index = int(state.get("active_index", self.active_index) or 0)
        while index < len(self.children):
            status = self.children[index].tick(ctx)
            if status == Status.SUCCESS:
                index += 1
                self.active_index = index
                state["active_index"] = index
                continue
            if status == Status.RUNNING:
                self.active_index = index
                state["active_index"] = index
            else:
                self.active_index = 0
                state["active_index"] = 0
            return self._record(status)

        self.active_index = 0
        state["active_index"] = 0
        return self._record(Status.SUCCESS)


class Selector(Node):
    """从前往后选择第一个 SUCCESS 或 RUNNING 的子节点。"""

    def tick(self, ctx: TreeContext) -> Status:
        saw_failure = False
        for child in self.children:
            status = child.tick(ctx)
            if status in (Status.SUCCESS, Status.RUNNING):
                return self._record(status)
            if status == Status.FAILURE:
                saw_failure = True
        return self._record(Status.FAILURE if saw_failure else Status.SKIP)


class ReactiveSelector(Selector):
    """每次 tick 都从第一个子节点重新检查的选择节点。"""


class MemorySelector(Node):
    """有记忆的选择节点，子节点 RUNNING 时优先继续该子节点。"""

    def __init__(self, *children: Node, label: Optional[str] = None, persist: bool = False):
        super().__init__(*children, label=label, persist=persist)
        self.active_index = None

    def tick(self, ctx: TreeContext) -> Status:
        state = self.state(ctx)
        stored_index = state.get("active_index", self.active_index)
        if stored_index is not None:
            status = self.children[int(stored_index)].tick(ctx)
            if status == Status.RUNNING:
                self.active_index = int(stored_index)
                state["active_index"] = int(stored_index)
                return self._record(status)
            self.active_index = None
            state["active_index"] = None
            if status == Status.SUCCESS:
                return self._record(status)

        saw_failure = False
        for index, child in enumerate(self.children):
            status = child.tick(ctx)
            if status == Status.RUNNING:
                self.active_index = index
                state["active_index"] = index
                return self._record(status)
            if status == Status.SUCCESS:
                return self._record(status)
            if status == Status.FAILURE:
                saw_failure = True
        return self._record(Status.FAILURE if saw_failure else Status.SKIP)


class Decorator(Node):
    """单子节点装饰器基类。"""

    def __init__(self, child: Node, *, label: Optional[str] = None, persist: bool = False):
        super().__init__(child, label=label, persist=persist)

    @property
    def child(self) -> Node:
        return self.children[0]


class Once(Decorator):
    """当前进程内只执行一次的装饰器，默认不持久化。"""

    def tick(self, ctx: TreeContext) -> Status:
        state = self.state(ctx)
        if state.get("done"):
            return self._record(Status.SKIP)

        status = self.child.tick(ctx)
        if status == Status.SUCCESS:
            state["done"] = True
        return self._record(status)


class Retry(Decorator):
    """把子节点异常转换成一次延后重试，默认只在内存中记录状态。"""

    def __init__(
        self,
        child: Node,
        seconds=_MISSING,
        *,
        days=0,
        hours=0,
        minutes=0,
        label: Optional[str] = None,
        persist: bool = False,
    ):
        super().__init__(child, label=label, persist=persist)
        if seconds is _MISSING and days == hours == minutes == 0:
            self.interval_seconds = 0.0
        else:
            self.interval_seconds = _duration_seconds(seconds, days=days, hours=hours, minutes=minutes)

    def default_label(self) -> str:
        return self.label or f"Retry[{self.interval_seconds:g}s]"

    def next_wake(self, ctx: TreeContext) -> Optional[datetime.datetime]:
        retry_at = _parse_time(self.state(ctx).get("retry_at"))
        if retry_at is not None:
            return retry_at
        return self.child.next_wake(ctx)

    def tick(self, ctx: TreeContext) -> Status:
        retry_at = _parse_time(self.state(ctx).get("retry_at"))
        if retry_at is not None and retry_at > ctx.now():
            return self._record(Status.SKIP)

        try:
            status = self.child.tick(ctx)
        except Exception as err:
            self.child.reset(ctx)
            ctx.runner._record_error(err, node=self, handled=True)
            state = self.state(ctx)
            state["last_error_at"] = _format_time(ctx.now())
            state["last_error"] = "".join(traceback.format_exception(type(err), err, err.__traceback__))
            state["error_count"] = int(state.get("error_count", 0) or 0) + 1
            retry_time = ctx.now() + datetime.timedelta(seconds=self.interval_seconds)
            state["retry_at"] = _format_time(retry_time)
            ctx.next_run_at = retry_time
            return self._record(Status.SUCCESS)

        if status == Status.SUCCESS:
            self.state(ctx)["retry_at"] = None
        return self._record(status)


class Timeout(Decorator):
    """协作式超时保护；子节点长时间不 yield 时无法强制中断。"""

    def __init__(
        self,
        child: Node,
        seconds=_MISSING,
        *,
        days=0,
        hours=0,
        minutes=0,
        label: Optional[str] = None,
        persist: bool = False,
    ):
        super().__init__(child, label=label, persist=persist)
        self.timeout_seconds = _duration_seconds(seconds, days=days, hours=hours, minutes=minutes)

    def default_label(self) -> str:
        return self.label or f"Timeout[{self.timeout_seconds:g}s]"

    def tick(self, ctx: TreeContext) -> Status:
        state = self.state(ctx)
        started_at = _parse_time(state.get("started_at"))
        if started_at is None:
            started_at = ctx.now()
            state["started_at"] = _format_time(started_at)

        if (ctx.now() - started_at).total_seconds() >= self.timeout_seconds:
            self.child.reset(ctx)
            state["started_at"] = None
            raise TimeoutError(f"node timeout after {self.timeout_seconds:g}s: {self.path}")

        status = self.child.tick(ctx)
        if status != Status.RUNNING:
            state["started_at"] = None
        return self._record(status)


class _TimeDecorator(Decorator):
    START_POLICIES = {"run", "next", "reset-run", "reset-next"}

    def __init__(
        self,
        child: Node,
        *,
        label: Optional[str] = None,
        persist: bool = False,
        default_next_time=None,
        start: str = "run",
        enabled: bool = True,
        on_schedule: Optional[Callable[[TreeContext, datetime.datetime], None]] = None,
    ):
        if start not in self.START_POLICIES:
            raise ValueError(f"invalid start policy: {start!r}")
        if default_next_time is not None and start != "run":
            raise ValueError("default_next_time can only be used with start='run'")

        super().__init__(child, label=label, persist=persist)
        self.default_next_time = default_next_time
        self.start = start
        self.enabled = bool(enabled)
        self._start_applied = False
        self.on_schedule = on_schedule

    def _state_next_run_at(self, ctx: TreeContext) -> Optional[datetime.datetime]:
        return _parse_time(self.state(ctx).get("next_run_at"))

    def _set_next_run_at(self, ctx: TreeContext, value: Optional[datetime.datetime]) -> None:
        self.state(ctx)["next_run_at"] = _format_time(value)
        if value is not None and self.on_schedule is not None:
            self.on_schedule(ctx, value)

    def _initial_next_run_at(self, ctx: TreeContext) -> datetime.datetime:
        if self.default_next_time is None:
            return self._default_initial_next_run_at(ctx)
        value = self.default_next_time(ctx.runner) if callable(self.default_next_time) else self.default_next_time
        parsed = _parse_time(value)
        if parsed is None:
            raise ValueError(f"invalid default_next_time: {self.default_next_time!r}")
        return parsed

    def _default_initial_next_run_at(self, ctx: TreeContext) -> datetime.datetime:
        return ctx.now()

    def _ensure_next_run_at(self, ctx: TreeContext) -> datetime.datetime:
        value = self._state_next_run_at(ctx)
        if value is None or (self.start.startswith("reset-") and not self._start_applied):
            value = self._initial_next_run_at(ctx)
            self._set_next_run_at(ctx, value)
        self._start_applied = True
        return value

    def next_wake(self, ctx: TreeContext) -> Optional[datetime.datetime]:
        if not self.enabled:
            return None
        return self._ensure_next_run_at(ctx)

    def retry(
        self,
        seconds=_MISSING,
        *,
        days=0,
        hours=0,
        minutes=0,
        label: Optional[str] = None,
        persist: bool = False,
    ) -> "_TimeDecorator":
        self.children[0] = self.child.retry(
            seconds,
            days=days,
            hours=hours,
            minutes=minutes,
            label=label,
            persist=persist,
        )
        self.children[0].parent = self
        return self

    def timeout(
        self,
        seconds=_MISSING,
        *,
        days=0,
        hours=0,
        minutes=0,
        label: Optional[str] = None,
        persist: bool = False,
    ) -> "_TimeDecorator":
        self.children[0] = self.child.timeout(
            seconds,
            days=days,
            hours=hours,
            minutes=minutes,
            label=label,
            persist=persist,
        )
        self.children[0].parent = self
        return self


class Daily(_TimeDecorator):
    """每日锚点触发的时间装饰器，默认持久化下一次时间。"""

    def __init__(
        self,
        *anchors: Any,
        child: Optional[Node] = None,
        label: Optional[str] = None,
        persist: bool = True,
        default_next_time=None,
        start: str = "run",
        enabled: bool = True,
        on_schedule: Optional[Callable[[TreeContext, datetime.datetime], None]] = None,
    ):
        if child is None:
            if not anchors or not isinstance(anchors[-1], Node):
                raise TypeError("Daily requires a child node")
            *anchors, child = anchors
        self.anchors = _normalize_daily_anchors(tuple(anchors))
        super().__init__(
            child,
            label=label,
            persist=persist,
            default_next_time=default_next_time,
            start=start,
            enabled=enabled,
            on_schedule=on_schedule,
        )

    def default_label(self) -> str:
        return self.label or f"Daily[{','.join(self.anchors)}]"

    def _default_initial_next_run_at(self, ctx: TreeContext) -> datetime.datetime:
        if self.start.endswith("next"):
            return _compute_next_time(*self.anchors, base_time=ctx.now())
        return ctx.now()

    def tick(self, ctx: TreeContext) -> Status:
        if not self.enabled:
            return self._record(Status.SKIP)

        next_run_at = self._ensure_next_run_at(ctx)

        if next_run_at > ctx.now():
            return self._record(Status.SKIP)

        ctx.next_run_at = None
        status = self.child.tick(ctx)
        if status != Status.RUNNING:
            next_run_at = ctx.next_run_at or _compute_next_time(*self.anchors, base_time=ctx.now())
            self._set_next_run_at(ctx, next_run_at)
        return self._record(status)


class Every(_TimeDecorator):
    """固定间隔触发的时间装饰器，默认只在内存中记录下一次时间。"""

    def __init__(
        self,
        *args: Any,
        child: Optional[Node] = None,
        label: Optional[str] = None,
        persist: bool = False,
        default_next_time=None,
        enabled: bool = True,
        on_schedule: Optional[Callable[[TreeContext, datetime.datetime], None]] = None,
        days=0,
        hours=0,
        minutes=0,
        seconds=0,
    ):
        if child is None and args and isinstance(args[-1], Node):
            *args, child = args
        if args:
            seconds += args[0]
        if child is None:
            raise TypeError("Every requires a child node")
        self.interval_seconds = _duration_seconds(days=days, hours=hours, minutes=minutes, seconds=seconds)
        super().__init__(
            child,
            label=label,
            persist=persist,
            default_next_time=default_next_time,
            enabled=enabled,
            on_schedule=on_schedule,
        )

    def default_label(self) -> str:
        return self.label or f"Every[{self.interval_seconds:g}s]"

    def tick(self, ctx: TreeContext) -> Status:
        if not self.enabled:
            return self._record(Status.SKIP)

        if self._state_next_run_at(ctx) is None:
            self._set_next_run_at(ctx, self._initial_next_run_at(ctx))

        if self._state_next_run_at(ctx) > ctx.now():
            return self._record(Status.SKIP)

        ctx.next_run_at = None
        status = self.child.tick(ctx)
        if status != Status.RUNNING:
            self._set_next_run_at(ctx, ctx.next_run_at or (ctx.now() + datetime.timedelta(seconds=self.interval_seconds)))
        return self._record(status)


class DynamicTime(_TimeDecorator):
    """由子节点运行结果动态决定下一次时间的装饰器，默认持久化。"""

    def __init__(
        self,
        child: Node,
        *,
        fallback_seconds: Optional[float] = None,
        label: Optional[str] = None,
        persist: bool = True,
        default_next_time=None,
        enabled: bool = True,
        on_schedule: Optional[Callable[[TreeContext, datetime.datetime], None]] = None,
    ):
        super().__init__(
            child,
            label=label,
            persist=persist,
            default_next_time=default_next_time,
            enabled=enabled,
            on_schedule=on_schedule,
        )
        self.fallback_seconds = fallback_seconds

    def tick(self, ctx: TreeContext) -> Status:
        if not self.enabled:
            return self._record(Status.SKIP)

        if self._state_next_run_at(ctx) is None:
            self._set_next_run_at(ctx, self._initial_next_run_at(ctx))

        if self._state_next_run_at(ctx) > ctx.now():
            return self._record(Status.SKIP)

        ctx.next_run_at = None
        status = self.child.tick(ctx)
        if status != Status.RUNNING:
            if ctx.next_run_at is not None:
                self._set_next_run_at(ctx, ctx.next_run_at)
            elif self.fallback_seconds is not None:
                self._set_next_run_at(ctx, ctx.now() + datetime.timedelta(seconds=self.fallback_seconds))
        return self._record(status)


class Window(Decorator):
    """只在指定时间窗口内运行子节点，窗口外返回 SKIP。"""

    def __init__(self, window_start: str, window_end: str, child: Node, *, label: Optional[str] = None):
        super().__init__(child, label=label)
        self.window_start = window_start
        self.window_end = window_end

    def default_label(self) -> str:
        return self.label or f"Window[{self.window_start}-{self.window_end}]"

    def tick(self, ctx: TreeContext) -> Status:
        if not _is_within_time(self.window_start, self.window_end, base_time=ctx.now()):
            return self._record(Status.SKIP)
        return self._record(self.child.tick(ctx))


class WithServices(Node):
    """给业务子树挂载守护服务，服务只在业务子树活跃或到期时运行。"""

    def __init__(self, child: Node, *services: Node, label: Optional[str] = None):
        super().__init__(child, *services, label=label)

    @property
    def child(self) -> Node:
        return self.children[0]

    @property
    def services(self) -> List[Node]:
        return self.children[1:]

    def next_wake(self, ctx: TreeContext) -> Optional[datetime.datetime]:
        return self.child.next_wake(ctx)

    def tick(self, ctx: TreeContext) -> Status:
        next_wake_at = self.child.next_wake(ctx)
        if not self.child.is_active(ctx) and (next_wake_at is None or next_wake_at > ctx.now()):
            return self._record(Status.SKIP)

        for service in self.services:
            status = service.tick(ctx)
            if status in (Status.SUCCESS, Status.RUNNING, Status.FAILURE):
                return self._record(status)
        return self._record(self.child.tick(ctx))


class IdleUntilNextWake(Node):
    """等待到下一次业务节点唤醒时间。"""

    def __init__(self, *, ratio=0.8, min_seconds=1, max_seconds=None, label: Optional[str] = None):
        super().__init__(label=label)
        self.ratio = float(ratio)
        self.min_seconds = float(min_seconds)
        self.max_seconds = max_seconds

    def tick(self, ctx: TreeContext) -> Status:
        target_time = ctx.runner.next_wake()
        if target_time is None:
            ctx.sleep(self.min_seconds)
            return self._record(Status.RUNNING)

        remaining = (target_time - ctx.now()).total_seconds()
        if remaining <= 0:
            return self._record(Status.SUCCESS)

        sleep_seconds = min(max(remaining * self.ratio, self.min_seconds), remaining)
        if self.max_seconds is not None:
            sleep_seconds = min(sleep_seconds, float(self.max_seconds))
        ctx.sleep(sleep_seconds)
        if (target_time - ctx.now()).total_seconds() <= 0:
            return self._record(Status.SUCCESS)
        return self._record(Status.RUNNING)


class BehaviorTreeRunner:
    """行为树运行器。"""

    def __init__(self, root: Node, state_path=None, *, trace=0, log_path=None, now_func=None, on_error=None):
        self.root = root
        self.state_path = Path(state_path) if state_path is not None else None
        self.trace = int(trace)
        self.now_func = now_func
        self.on_error = on_error
        self.state = self.load_state()
        self.memory_state = {"nodes": {}, "blackboard": {}}
        self.logger = self._create_logger(log_path)
        self._last_generator_return_locations = []
        self.last_trace_interrupt = None
        self.last_trace_signature = None
        self.last_trace_first_seen_at = None
        self.last_trace_last_seen_at = None
        self.last_trace_count = 0
        self._assign_paths()

    def now(self) -> datetime.datetime:
        return (self.now_func() if self.now_func else datetime.datetime.now()).replace(microsecond=0)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def load_state(self) -> Dict[str, Any]:
        if self.state_path is not None and self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
        data.setdefault("nodes", {})
        data.setdefault("blackboard", {})
        return data

    def save_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_name(f"{self.state_path.name}.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.state_path)

    def node_state(self, node: Node) -> Dict[str, Any]:
        state_root = self.state if node.persist else self.memory_state
        return state_root.setdefault("nodes", {}).setdefault(node.path, {})

    def next_wake(self) -> Optional[datetime.datetime]:
        ctx = TreeContext(self)
        return self.root.next_wake(ctx)

    def run_once(self) -> Status:
        ctx = TreeContext(self)
        try:
            status = self.root.tick(ctx)
        except Exception as err:
            self._record_error(err)
            self.save_state()
            raise
        else:
            self.save_state()
            if self.trace >= 2:
                self._log_trace_interrupt(ctx, status)
            return status

    def run_forever(self, tick_seconds=1) -> None:
        while True:
            ctx = TreeContext(self)
            try:
                status = self.root.tick(ctx)
            except Exception as err:
                self._record_error(err)
                self.save_state()
                raise
            else:
                self.save_state()
                if self.trace >= 2:
                    self._log_trace_interrupt(ctx, status)
                if not ctx._slept:
                    time.sleep(tick_seconds)

    def _advance_generator(self, generator: GeneratorType) -> None:
        if self.trace < 2:
            next(generator)
            return

        self._last_generator_return_locations = []
        previous_trace = sys.gettrace()

        def trace_func(frame, event, arg):
            if event == "return" and frame.f_code.co_flags & inspect.CO_GENERATOR:
                self._last_generator_return_locations.append(_frame_source_location(frame))
            return trace_func

        sys.settrace(trace_func)
        try:
            next(generator)
        finally:
            sys.settrace(previous_trace)

    def _generator_return_location(self) -> Optional[_SourceLocation]:
        if not self._last_generator_return_locations:
            return None
        return self._last_generator_return_locations[0]

    def _log_trace_interrupt(self, ctx: TreeContext, status: Status) -> None:
        interrupt = ctx.trace_interrupt
        if interrupt is None:
            return
        self.last_trace_interrupt = interrupt
        location = interrupt.location
        if interrupt.kind == "yield" and location is not None:
            signature = "|".join(
                [
                    str(interrupt.node_path or ""),
                    str(location.filename),
                    str(location.lineno),
                    str(location.function),
                ]
            )
            now = ctx.now()
            if self.last_trace_signature != signature:
                self.last_trace_signature = signature
                self.last_trace_first_seen_at = now
                self.last_trace_count = 1
            else:
                self.last_trace_count += 1
            self.last_trace_last_seen_at = now
        elif interrupt.kind == "return":
            self.last_trace_signature = None
            self.last_trace_first_seen_at = None
            self.last_trace_last_seen_at = None
            self.last_trace_count = 0
        extra = {}
        location_text = "unknown"
        if location is not None:
            location_text = location.brief()
            extra = {
                "source_path": location.filename,
                "source_line": location.lineno,
                "source_function": location.function,
            }
        self.logger.debug("tree %s: %s -> %s", interrupt.kind, location_text, status.value, extra=extra)

    def _create_logger(self, log_path) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        if self.trace <= 0:
            logger.addHandler(logging.NullHandler())
            return logger

        try:
            from loguru import logger as loguru_logger
            
            # 使用 loguru
            class LoguruHandler(logging.Handler):
                def emit(self, record):
                    try:
                        level = loguru_logger.level(record.levelname).name
                    except ValueError:
                        level = record.levelno

                    frame, depth = inspect.currentframe(), 0
                    while frame is not None:
                        if frame.f_code.co_filename == record.pathname and frame.f_code.co_name == record.funcName:
                            break
                        frame = frame.f_back
                        depth += 1

                    if frame is None:
                        depth = 2

                    source = {
                        "source_path": getattr(record, "source_path", record.pathname),
                        "source_line": getattr(record, "source_line", record.lineno),
                        "source_function": getattr(record, "source_function", record.funcName),
                    }
                    loguru_logger.bind(**source).opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
                    
            handler = LoguruHandler()
            logger.addHandler(handler)
            
            if log_path:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                loguru_logger.add(log_path, level="DEBUG", encoding="utf-8")
                
            return logger
        except ImportError:
            pass

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

    def _record_error(self, err: Exception, *, node: Optional[Node] = None, handled: bool = False) -> None:
        self.state["last_error_at"] = _format_time(self.now())
        self.state["last_error"] = "".join(traceback.format_exception(type(err), err, err.__traceback__))
        self.state["last_error_handled"] = bool(handled)
        self.state["last_error_node"] = node.path if node is not None else None
        if self.trace >= 1:
            node_text = f" at {node.path}" if node is not None else ""
            if handled:
                self.logger.exception("tree handled error%s", node_text)
            else:
                self.logger.exception("tree error%s", node_text)
        if self.on_error is not None:
            try:
                self.on_error(err, runner=self, node=node, handled=handled)
            except Exception:
                if self.trace >= 1:
                    self.logger.exception("tree on_error callback failed")

    def _assign_paths(self) -> None:
        def visit(node: Node, parent_path: str = "") -> None:
            base = str(node.label or node.default_label()).replace("/", "|")
            if not parent_path:
                node.path = base
            node.children = [child for child in node.children if child is not None]
            counts = {}
            for child in node.children:
                child.parent = node
                child_base = str(child.label or child.default_label()).replace("/", "|")
                counts[child_base] = counts.get(child_base, 0) + 1
                suffix = "" if counts[child_base] == 1 else f"#{counts[child_base]}"
                child.path = f"{node.path}/{child_base}{suffix}"
                visit(child, child.path)

        visit(self.root)


BTreeRunner = BehaviorTreeRunner

