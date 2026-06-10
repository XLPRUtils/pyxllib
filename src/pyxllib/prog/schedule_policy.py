import calendar
import datetime as _dt
import hashlib
import json
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union


RESULT_SUCCESS = "success"
RESULT_FAILURE = "failure"
RESULT_TIMEOUT = "timeout"


def _now(base_time: Any = None) -> _dt.datetime:
    if base_time is None:
        return _dt.datetime.now().replace(microsecond=0)
    if isinstance(base_time, _dt.datetime):
        return base_time.replace(microsecond=0)
    if isinstance(base_time, (int, float)):
        return _dt.datetime.fromtimestamp(base_time).replace(microsecond=0)
    text = str(base_time).strip()
    if not text:
        return _dt.datetime.now().replace(microsecond=0)
    if text.endswith("Z"):
        text = text[:-1]
    return _dt.datetime.fromisoformat(text).replace(microsecond=0)


def _format_time(value: Optional[_dt.datetime]) -> Optional[str]:
    return value.replace(microsecond=0).isoformat() if value else None


def _parse_time(value: Any) -> Optional[_dt.datetime]:
    if value in (None, ""):
        return None
    try:
        return _now(value)
    except Exception:
        return None


def _duration_seconds(rule: Mapping[str, Any], default: int = 0) -> int:
    has_duration = any(key in rule for key in ("days", "hours", "minutes", "seconds"))
    total = 0
    total += int(rule.get("days") or 0) * 86400
    total += int(rule.get("hours") or 0) * 3600
    total += int(rule.get("minutes") or 0) * 60
    total += int(rule.get("seconds") or 0)
    if not has_duration:
        total = default
    return max(0, total)


def _normalize_result(result: Any) -> str:
    value = getattr(result, "value", result)
    return str(value or "").strip().lower()


def _times(rule: Mapping[str, Any], default: str = "00:00") -> List[str]:
    value = rule.get("times", rule.get("time", default))
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value or [default])
    return [str(item).strip() for item in values if str(item).strip()]


def _parse_clock(value: str) -> Tuple[int, int, int]:
    parts = [int(part) for part in str(value).split(":")]
    if len(parts) == 2:
        hour, minute = parts
        second = 0
    elif len(parts) == 3:
        hour, minute, second = parts
    else:
        raise ValueError(f"invalid clock value: {value!r}")
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(f"invalid clock value: {value!r}")
    return hour, minute, second


def _next_daily(rule: Mapping[str, Any], base: _dt.datetime) -> Optional[_dt.datetime]:
    candidates: List[_dt.datetime] = []
    for day_offset in range(2):
        date = (base + _dt.timedelta(days=day_offset)).date()
        for text in _times(rule):
            hour, minute, second = _parse_clock(text)
            candidate = _dt.datetime.combine(date, _dt.time(hour, minute, second))
            if candidate > base:
                candidates.append(candidate)
    return min(candidates) if candidates else None


def _next_weekly(rule: Mapping[str, Any], base: _dt.datetime) -> Optional[_dt.datetime]:
    weekdays = rule.get("weekdays") or rule.get("days") or []
    normalized = {((int(item) - 1) % 7) for item in weekdays}
    if not normalized:
        return None
    candidates: List[_dt.datetime] = []
    for day_offset in range(8):
        date = (base + _dt.timedelta(days=day_offset)).date()
        if date.weekday() not in normalized:
            continue
        for text in _times(rule):
            hour, minute, second = _parse_clock(text)
            candidate = _dt.datetime.combine(date, _dt.time(hour, minute, second))
            if candidate > base:
                candidates.append(candidate)
    return min(candidates) if candidates else None


MonthDay = Union[int, str]


def _monthly_days(rule: Mapping[str, Any]) -> List[MonthDay]:
    value = rule.get("days", rule.get("day", []))
    if value in (None, ""):
        return []
    values = value if isinstance(value, list) else [value]
    result: List[MonthDay] = []
    for item in values:
        if isinstance(item, str) and item.strip().lower() in {"last", "月末"}:
            result.append("last")
        else:
            result.append(int(item))
    return result


def _resolve_month_day(year: int, month: int, day: MonthDay) -> Optional[int]:
    last_day = calendar.monthrange(year, month)[1]
    if day == "last":
        return last_day
    day = int(day)
    if day < 0:
        resolved = last_day + day + 1
        return resolved if 1 <= resolved <= last_day else None
    if day > last_day:
        return None
    return day if day >= 1 else None


def _next_monthly(rule: Mapping[str, Any], base: _dt.datetime) -> Optional[_dt.datetime]:
    days = _monthly_days(rule)
    if not days:
        return None
    candidates: List[_dt.datetime] = []
    for month_offset in range(14):
        month_index = base.month - 1 + month_offset
        year = base.year + month_index // 12
        month = month_index % 12 + 1
        for day in days:
            resolved_day = _resolve_month_day(year, month, day)
            if resolved_day is None:
                continue
            for text in _times(rule):
                hour, minute, second = _parse_clock(text)
                candidate = _dt.datetime(year, month, resolved_day, hour, minute, second)
                if candidate > base:
                    candidates.append(candidate)
    return min(candidates) if candidates else None


def _next_cron(rule: Mapping[str, Any], base: _dt.datetime) -> Optional[_dt.datetime]:
    expression = str(rule.get("expression") or rule.get("cron") or "").strip()
    if not expression:
        return None
    try:
        from apscheduler.triggers.cron import CronTrigger
    except Exception as exc:  # pragma: no cover - depends on optional runtime.
        raise RuntimeError("cron trigger requires apscheduler") from exc
    trigger = CronTrigger.from_crontab(expression)
    value = trigger.get_next_fire_time(None, base)
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone().replace(tzinfo=None)
    return value.replace(microsecond=0)


def compute_next_by_trigger(
    trigger: Optional[Mapping[str, Any]],
    *,
    base_time: Any = None,
    state: Optional[Mapping[str, Any]] = None,
) -> Optional[_dt.datetime]:
    """Compute the next trigger time from a serializable trigger rule."""

    trigger = dict(trigger or {"type": "manual"})
    state = state or {}
    base = _now(base_time)
    kind = str(trigger.get("type") or "manual").strip().lower()

    if kind in {"manual", "none", "disabled"}:
        return None
    if kind == "once":
        candidate = _parse_time(trigger.get("at") or trigger.get("time"))
        return candidate if candidate and candidate > base else None
    if kind == "interval":
        seconds = _duration_seconds(trigger)
        if seconds <= 0:
            return None
        anchor = str(trigger.get("anchor") or "last_finish").strip().lower()
        if anchor in {"last_trigger", "trigger"}:
            anchor_time = _parse_time(state.get("last_trigger_at"))
        elif anchor in {"last_start", "start"}:
            anchor_time = _parse_time(state.get("last_started_at"))
        elif anchor in {"last_finish", "finish"}:
            anchor_time = _parse_time(state.get("last_finished_at"))
        else:
            anchor_time = None
        candidate = (anchor_time or base) + _dt.timedelta(seconds=seconds)
        while candidate <= base:
            candidate += _dt.timedelta(seconds=seconds)
        return candidate
    if kind == "daily":
        return _next_daily(trigger, base)
    if kind == "weekly":
        return _next_weekly(trigger, base)
    if kind == "monthly":
        return _next_monthly(trigger, base)
    if kind == "cron":
        return _next_cron(trigger, base)
    if kind in {"any", "or"}:
        values = [
            compute_next_by_trigger(rule, base_time=base, state=state)
            for rule in trigger.get("rules") or []
        ]
        values = [value for value in values if value is not None]
        return min(values) if values else None
    raise ValueError(f"unsupported trigger type: {kind}")


def policy_fingerprint(policy: Optional[Mapping[str, Any]]) -> str:
    data = json.dumps(policy or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def compute_next_trigger_at(
    policy: Optional[Mapping[str, Any]],
    *,
    state: Optional[Mapping[str, Any]] = None,
    base_time: Any = None,
    result: Optional[str] = None,
) -> Optional[_dt.datetime]:
    """Compute the next trigger time for a schedule policy.

    Failure/timeout can be redirected to a retry time by the outcome policy;
    otherwise the normal trigger rule is used.
    """

    policy = dict(policy or {})
    if not policy.get("enabled", False):
        return None
    state = state or {}
    base = _now(base_time)
    result = _normalize_result(result)
    outcome = policy.get("outcome") or {}
    if result in {RESULT_FAILURE, RESULT_TIMEOUT}:
        key = "on_timeout" if result == RESULT_TIMEOUT else "on_failure"
        failure_policy = outcome.get(key) or outcome.get("on_failure") or {}
        if str(failure_policy.get("type") or "").lower() == "retry_after":
            max_attempts = failure_policy.get("max_attempts")
            failure_count = int(state.get("failure_count") or 0)
            if max_attempts is None or failure_count <= int(max_attempts):
                seconds = _duration_seconds(failure_policy, default=600)
                if seconds > 0:
                    return base + _dt.timedelta(seconds=seconds)
    return compute_next_by_trigger(policy.get("trigger"), base_time=base, state=state)


def initialize_schedule_state(
    policy: Optional[Mapping[str, Any]],
    state: Optional[Mapping[str, Any]] = None,
    *,
    base_time: Any = None,
    force: bool = False,
) -> Dict[str, Any]:
    state_dict = dict(state or {})
    fingerprint = policy_fingerprint(policy)
    if force or state_dict.get("policy_fingerprint") != fingerprint or "next_trigger_at" not in state_dict:
        state_dict["next_trigger_at"] = _format_time(
            compute_next_trigger_at(policy, state=state_dict, base_time=base_time)
        )
        state_dict["policy_fingerprint"] = fingerprint
    return state_dict


def apply_schedule_result(
    policy: Optional[Mapping[str, Any]],
    state: Optional[Mapping[str, Any]] = None,
    *,
    result: str,
    base_time: Any = None,
) -> Dict[str, Any]:
    base = _now(base_time)
    result = _normalize_result(result)
    state_dict = dict(state or {})
    state_dict["last_finished_at"] = _format_time(base)
    state_dict["last_result"] = result
    if result == RESULT_SUCCESS:
        state_dict["failure_count"] = 0
    else:
        state_dict["failure_count"] = int(state_dict.get("failure_count") or 0) + 1
    state_dict["policy_fingerprint"] = policy_fingerprint(policy)
    state_dict["next_trigger_at"] = _format_time(
        compute_next_trigger_at(policy, state=state_dict, base_time=base, result=result)
    )
    return state_dict


def schedule_policy_label(policy: Optional[Mapping[str, Any]]) -> str:
    policy = dict(policy or {})
    if not policy.get("enabled", False):
        return ""
    trigger = dict(policy.get("trigger") or {})
    kind = str(trigger.get("type") or "manual").lower()
    if kind == "interval":
        seconds = _duration_seconds(trigger)
        if seconds <= 0:
            return ""
        if seconds % 3600 == 0:
            return f"每 {seconds // 3600} 小时"
        if seconds % 60 == 0:
            return f"每 {seconds // 60} 分钟"
        return f"每 {seconds} 秒"
    if kind == "daily":
        return f"每天 {', '.join(_times(trigger))}"
    if kind == "weekly":
        days = ",".join(str(x) for x in (trigger.get("weekdays") or trigger.get("days") or []))
        return f"每周 {days} {', '.join(_times(trigger))}"
    if kind == "monthly":
        days = ",".join(str(x) for x in _monthly_days(trigger))
        return f"每月 {days} 日 {', '.join(_times(trigger))}"
    if kind == "cron":
        return f"Cron {trigger.get('expression') or ''}".strip()
    if kind == "once":
        return f"一次 {trigger.get('at') or trigger.get('time') or ''}".strip()
    if kind in {"any", "or"}:
        return "任一规则触发"
    return ""


def parse_schedule_time(value: Any) -> Optional[float]:
    """Parse timestamp-like schedule values into epoch seconds."""

    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return _dt.datetime.strptime(text[:19], fmt).timestamp()
        except ValueError:
            pass
    return None


def first_valid_schedule_time_text(source: Mapping[str, Any], *keys: str) -> Optional[str]:
    """Return the first non-empty field that can be parsed as a schedule time."""

    for key in keys:
        value = source.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and parse_schedule_time(text) is not None:
            return text
    return None


def parse_daily_clock(value: Any) -> Optional[_dt.time]:
    """Parse HH:MM or HH:MM:SS into a time object."""

    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return _dt.datetime.strptime(text, fmt).time()
        except ValueError:
            pass
    return None


def next_daily_time(
    times: Any,
    *,
    base_time: Optional[_dt.datetime] = None,
    output_format: str = "%Y-%m-%d %H:%M:%S",
) -> Optional[str]:
    """Return the next future time from a list of daily clock strings."""

    values = times if isinstance(times, list) else []
    clocks = [clock for value in values if (clock := parse_daily_clock(value)) is not None]
    if not clocks:
        return None
    base = base_time or _dt.datetime.now()
    candidates: List[_dt.datetime] = []
    for day_offset in (0, 1):
        current_date = base.date() + _dt.timedelta(days=day_offset)
        for clock in clocks:
            candidate = _dt.datetime.combine(current_date, clock)
            if candidate > base:
                candidates.append(candidate)
    if not candidates:
        return None
    return min(candidates).strftime(output_format)


def schedule_task_due(
    task: Mapping[str, Any],
    *,
    now: Optional[float] = None,
    enabled_key: str = "enabled",
    next_time_key: str = "next_time",
    retry_after_key: str = "retry_after",
) -> bool:
    """Check whether a serializable scheduled task is due."""

    if not task.get(enabled_key):
        return False
    next_at = parse_schedule_time(task.get(next_time_key))
    retry_at = parse_schedule_time(task.get(retry_after_key))
    due_at = retry_at if retry_at is not None else next_at
    current_time = _dt.datetime.now().timestamp() if now is None else now
    return due_at is None or due_at <= current_time


def scheduled_task_plan_reason(
    task: Mapping[str, Any],
    due: bool,
    *,
    task_supported: Callable[[Mapping[str, Any]], bool] | None = None,
    now: Optional[float] = None,
    labels: Optional[Mapping[str, str]] = None,
    enabled_key: str = "enabled",
    next_time_key: str = "next_time",
    retry_after_key: str = "retry_after",
) -> str:
    """Explain why a serializable scheduled task is runnable or blocked."""

    text = {
        "disabled": "未启用",
        "retry_wait": "等待重试：{time}",
        "next_wait": "未到时间：{time}",
        "unsupported": "尚未纳入当前框架验收",
        "due": "已到期",
        "manual": "可手动执行",
    }
    text.update(dict(labels or {}))
    if not task.get(enabled_key):
        return text["disabled"]
    current_time = _dt.datetime.now().timestamp() if now is None else now
    retry_after = task.get(retry_after_key)
    next_time = task.get(next_time_key)
    retry_at = parse_schedule_time(retry_after)
    next_at = parse_schedule_time(next_time)
    if retry_at is not None and retry_at > current_time:
        return text["retry_wait"].format(time=retry_after)
    if next_at is not None and next_at > current_time:
        return text["next_wait"].format(time=next_time)
    if task_supported is not None and not task_supported(task):
        return text["unsupported"]
    if due:
        return text["due"]
    return text["manual"]


def build_scheduled_task_plan(
    tasks: list[dict[str, Any]],
    *,
    runtime_running: bool = False,
    runtime_task: str = "",
    task_supported: Callable[[dict[str, Any]], bool] | None = None,
    task_due: Callable[[dict[str, Any]], bool] | None = None,
    task_facts: Mapping[str, Any] | None = None,
    now: Optional[float] = None,
    labels: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Build a generic runnable/due plan for serializable scheduled tasks."""

    text = {
        "runtime_wait": "Runtime 正在运行：{task}",
        "runtime_task": "任务",
        "run_due": "建议执行到期任务：{label}",
        "blocked": "存在到期任务，但当前均不可执行",
        "idle": "没有到期任务",
    }
    text.update(dict(labels or {}))
    current_time = _dt.datetime.now().timestamp() if now is None else now
    facts_by_id = task_facts if isinstance(task_facts, Mapping) else {}
    supported_check = task_supported or (lambda _task: True)
    due_check = task_due or (lambda task: schedule_task_due(task, now=current_time))
    plan_items: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        due = bool(due_check(task))
        supported = bool(supported_check(task))
        runnable = bool(task.get("enabled")) and due and supported and not runtime_running
        fact = facts_by_id.get(task_id) if isinstance(facts_by_id.get(task_id), dict) else {}
        item = {
            "id": task_id,
            "task_type": str(task.get("task_type") or ""),
            "label": str(task.get("label") or task_id),
            "supported": supported,
            "enabled": bool(task.get("enabled")),
            "due": due,
            "runnable": runnable,
            "reason": scheduled_task_plan_reason(
                task,
                due,
                task_supported=supported_check,
                now=current_time,
            ),
            "next_time": task.get("next_time") if task.get("next_time") else None,
            "retry_after": task.get("retry_after") if task.get("retry_after") else None,
            "last_result": str(task.get("last_result") or ""),
            "fact": fact,
        }
        plan_items.append(item)
    plan_items.sort(
        key=lambda item: (
            not item["due"],
            schedule_task_order_key(item),
        )
    )
    due_tasks = [item for item in plan_items if item["due"] and item["enabled"]]
    runnable_tasks = [item for item in due_tasks if item["runnable"]]
    if runtime_running:
        next_action = "wait"
        message = text["runtime_wait"].format(task=runtime_task or text["runtime_task"])
    elif runnable_tasks:
        next_action = "run_due"
        message = text["run_due"].format(label=runnable_tasks[0]["label"])
    elif due_tasks:
        next_action = "blocked"
        message = text["blocked"]
    else:
        next_action = "idle"
        message = text["idle"]
    return {
        "next_action": next_action,
        "message": message,
        "due_tasks": due_tasks,
        "tasks": plan_items,
    }


def select_due_scheduled_tasks(
    tasks: list[dict[str, Any]],
    *,
    task_due: Callable[[dict[str, Any]], bool] | None = None,
    task_supported: Callable[[dict[str, Any]], bool] | None = None,
    excluded_schedule_kinds: Iterable[str] = ("manual",),
    now: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Return due, supported scheduled tasks sorted by scheduler order."""

    current_time = _dt.datetime.now().timestamp() if now is None else now
    due_check = task_due or (lambda task: schedule_task_due(task, now=current_time))
    supported_check = task_supported or (lambda _task: True)
    excluded = {str(kind) for kind in excluded_schedule_kinds}
    return sorted(
        [
            task
            for task in tasks
            if str(task.get("schedule_kind") or "") not in excluded
            and due_check(task)
            and supported_check(task)
        ],
        key=schedule_task_order_key,
    )


def sync_scheduled_tasks_from_facts(
    tasks: list[dict[str, Any]],
    task_facts: Mapping[str, Any],
    *,
    time_field_sources: Mapping[str, tuple[str, ...]] | None = None,
    text_field_sources: Mapping[str, tuple[str, ...]] | None = None,
    task_id_key: str = "id",
    checkpoint_key: str = "checkpoint",
    synced_at_key: str = "fact_synced_at",
    fact_updated_at_key: str = "fact_updated_at",
    synced_at_text: str | None = None,
) -> bool:
    """Sync runtime-owned task fields from fact records keyed by task id."""

    if not isinstance(task_facts, Mapping) or not task_facts:
        return False
    time_sources = dict(time_field_sources or {})
    text_sources = dict(text_field_sources or {})
    changed = False
    for task in tasks:
        task_id = str(task.get(task_id_key) or "")
        fact = task_facts.get(task_id)
        if not isinstance(fact, Mapping):
            continue
        task_changed = False
        for target_key, source_keys in time_sources.items():
            value = first_valid_schedule_time_text(fact, *source_keys)
            if value and task.get(target_key) != value:
                task[target_key] = value
                task_changed = True
                changed = True
        for target_key, source_keys in text_sources.items():
            value = next((str(fact.get(key) or "").strip() for key in source_keys if str(fact.get(key) or "").strip()), "")
            if value and str(task.get(target_key) or "") != value:
                task[target_key] = value
                task_changed = True
                changed = True
        if task_changed:
            checkpoint = task.get(checkpoint_key) if isinstance(task.get(checkpoint_key), dict) else {}
            if synced_at_text is not None:
                checkpoint[synced_at_key] = synced_at_text
            checkpoint[fact_updated_at_key] = fact.get("updated_at")
            task[checkpoint_key] = checkpoint
    return changed


def normalize_scheduled_task_record(
    item: Any,
    *,
    id_key: str = "id",
    task_type_key: str = "task_type",
    default_source: str = "manual",
    default_schedule_kind: str = "manual",
) -> dict[str, Any] | None:
    """Normalize a loose dict into the common JSON scheduled-task shape."""

    if not isinstance(item, Mapping):
        return None
    task_id = str(item.get(id_key) or "").strip()
    task_type = str(item.get(task_type_key) or "").strip()
    if not task_id or not task_type:
        return None
    return {
        id_key: task_id,
        task_type_key: task_type,
        "label": str(item.get("label") or task_id),
        "source": str(item.get("source") or default_source),
        "schedule_kind": str(item.get("schedule_kind") or default_schedule_kind),
        "legacy_name": str(item.get("legacy_name") or ""),
        "enabled": bool(item.get("enabled")),
        "interruptible": bool(item.get("interruptible", True)),
        "next_time": item.get("next_time") if item.get("next_time") else None,
        "schedule_times": [str(value) for value in item.get("schedule_times", [])] if isinstance(item.get("schedule_times"), list) else [],
        "window": [str(value) for value in item.get("window", [])[:2]] if isinstance(item.get("window"), list) else None,
        "last_run_at": item.get("last_run_at") if item.get("last_run_at") else None,
        "last_result": str(item.get("last_result") or ""),
        "retry_after": item.get("retry_after") if item.get("retry_after") else None,
        "cooldown_seconds": int(item.get("cooldown_seconds") or 0),
        "payload": item.get("payload") if isinstance(item.get("payload"), dict) else {},
        "checkpoint": item.get("checkpoint") if isinstance(item.get("checkpoint"), dict) else None,
    }


def schedule_task_due_timestamp(task: Mapping[str, Any]) -> float:
    retry_at = parse_schedule_time(task.get("retry_after"))
    if retry_at is not None:
        return retry_at
    next_at = parse_schedule_time(task.get("next_time"))
    if next_at is not None:
        return next_at
    clocks = [
        value
        for value in task.get("schedule_times", [])
        if str(value or "").strip()
    ] if isinstance(task.get("schedule_times"), list) else []
    if clocks:
        parsed = sorted(str(value) for value in clocks)
        return parse_schedule_time(f"1970-01-01 {parsed[0]}") or 0.0
    return 0.0


def schedule_kind_rank(task: Mapping[str, Any], ranks: Optional[Mapping[str, int]] = None, *, default: int = 90) -> int:
    schedule_kind = str(task.get("schedule_kind") or "").strip()
    return dict(ranks or {"daily": 10, "dynamic": 20, "manual": 30}).get(schedule_kind, default)


def schedule_task_order_key(
    task: Mapping[str, Any],
    *,
    ranks: Optional[Mapping[str, int]] = None,
    default_rank: int = 90,
) -> Tuple[int, float, str]:
    return (
        schedule_kind_rank(task, ranks, default=default_rank),
        schedule_task_due_timestamp(task),
        str(task.get("id") or ""),
    )


TaskNormalizer = Callable[[Any], Optional[dict[str, Any]]]
NextTimeResolver = Callable[[dict[str, Any], Any], Any]


def merge_scheduled_task_updates(
    current_tasks: list[dict[str, Any]],
    incoming_tasks: list[dict[str, Any]],
    *,
    normalizer: TaskNormalizer,
    task_id_key: str = "id",
    enabled_key: str = "enabled",
    next_time_key: str = "next_time",
    retry_after_key: str = "retry_after",
    runtime_keys: tuple[str, ...] = ("last_run_at", "last_result", "retry_after", "next_time", "checkpoint"),
    next_time_resolver: NextTimeResolver | None = None,
    base_time: Any = None,
) -> list[dict[str, Any]]:
    """Merge edited task definitions while preserving runtime-owned fields."""

    current_by_id = {
        str(task.get(task_id_key) or ""): task
        for task in current_tasks
        if str(task.get(task_id_key) or "")
    }
    merged: list[dict[str, Any]] = []
    for incoming in incoming_tasks:
        normalized = normalizer(incoming)
        if normalized is None:
            continue
        current = current_by_id.get(str(normalized.get(task_id_key) or ""))
        if current is None:
            merged.append(normalized)
            continue
        was_enabled = bool(current.get(enabled_key))
        task = {**normalized}
        for key in runtime_keys:
            task[key] = current.get(key)
        if bool(task.get(enabled_key)) and not was_enabled and not task.get(retry_after_key):
            if next_time_resolver is not None:
                task[next_time_key] = next_time_resolver(task, base_time)
        elif not bool(task.get(enabled_key)):
            task[retry_after_key] = None
            task[next_time_key] = None
        merged.append(task)
    return merged


def scheduled_task_run_copy(
    tasks: list[dict[str, Any]],
    task_id: str,
    payload_override: Mapping[str, Any] | None = None,
    *,
    task_id_key: str = "id",
    payload_key: str = "payload",
) -> dict[str, Any] | None:
    """Return a runnable task copy with optional payload overrides."""

    resolved_task_id = str(task_id or "")
    task = next((item for item in tasks if str(item.get(task_id_key) or "") == resolved_task_id), None)
    if task is None:
        return None
    override = dict(payload_override) if isinstance(payload_override, Mapping) else {}
    if not override:
        return task
    original_payload = task.get(payload_key) if isinstance(task.get(payload_key), dict) else {}
    return {**task, payload_key: {**original_payload, **override}}


def scheduled_task_payload_with_meta(
    task: Mapping[str, Any],
    *,
    payload_key: str = "payload",
    task_id_key: str = "id",
    interruptible_key: str = "interruptible",
    meta_task_id_key: str = "__scheduler_task_id",
    meta_interruptible_key: str = "__scheduler_interruptible",
) -> dict[str, Any]:
    """Return task payload with scheduler bookkeeping metadata."""

    payload = task.get(payload_key) if isinstance(task.get(payload_key), dict) else {}
    return {
        **payload,
        meta_task_id_key: str(task.get(task_id_key) or ""),
        meta_interruptible_key: bool(task.get(interruptible_key, True)),
    }


def scheduled_task_state(
    task: Mapping[str, Any],
    *,
    transient_keys: Iterable[str] = ("supported",),
) -> dict[str, Any]:
    """Return the serializable task state without view-only fields."""

    hidden = {str(key) for key in transient_keys}
    return {key: value for key, value in task.items() if str(key) not in hidden}


def repair_orphaned_scheduled_runs(
    tasks: list[dict[str, Any]],
    *,
    pending_task_ids: Iterable[str] = (),
    runtime_running: bool = False,
    now: Optional[float] = None,
    min_age_seconds: float = 60.0,
    running_results: Iterable[str] = ("queued", "running"),
    stopped_result: str = "stopped",
    recovered_at_key: str = "recovered_from_orphaned_run_at",
    recovered_at_text: str | None = None,
) -> bool:
    """Mark stale queued/running scheduled tasks stopped when no runtime owns them."""

    if runtime_running:
        return False
    pending_ids = {str(value) for value in pending_task_ids if str(value)}
    running_result_set = {str(value) for value in running_results}
    current_time = _dt.datetime.now().timestamp() if now is None else now
    recovery_text = recovered_at_text or _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    changed = False
    for task in tasks:
        task_id = str(task.get("id") or "")
        if not task_id or task_id in pending_ids:
            continue
        if str(task.get("last_result") or "") not in running_result_set:
            continue
        last_run_ts = parse_schedule_time(task.get("last_run_at"))
        if last_run_ts is not None and current_time - last_run_ts < min_age_seconds:
            continue
        task["last_result"] = stopped_result
        task["retry_after"] = None
        checkpoint = task.get("checkpoint") if isinstance(task.get("checkpoint"), dict) else {}
        checkpoint[recovered_at_key] = recovery_text
        task["checkpoint"] = checkpoint
        changed = True
    return changed
