import calendar
import datetime as _dt
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


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
