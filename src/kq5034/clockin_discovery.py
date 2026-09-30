"""考勤独立小鹅通日历活动发现；CodeYun 只适配该公共入口。"""
from __future__ import annotations
import json
import time
from typing import Any
from urllib.parse import quote
XIAOE_CLOCKIN_HOME_URL = "https://admin.xiaoe-tech.com/t/clock_admin/index#/punchList/punchHome"
CENTER_CLASSROOM_MARKERS = ("中心教室",)
AUDIT_ONLY_CLASSROOM_MARKERS = ("旁听教室",)

class ClockinLinkDetectionError(RuntimeError):
    pass


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _has_any_marker(text: Any, markers: tuple[str, ...]) -> bool:
    normalized = _normalize_text(text)
    return any(marker in normalized for marker in markers)


def build_xiaoe_diary_list_url(
    *,
    activity_id: str,
    app_id: str,
    mini_middle_url: str = "",
) -> str:
    """Build the admin diaryList URL consumed by KQ5034 clockin export."""

    normalized_activity_id = _normalize_text(activity_id)
    normalized_app_id = _normalize_text(app_id)
    if not normalized_activity_id:
        raise ClockinLinkDetectionError("缺少小鹅通打卡 activity_id")
    if not normalized_app_id:
        raise ClockinLinkDetectionError("缺少小鹅通 app_id")
    resolved_mini_middle_url = _normalize_text(mini_middle_url) or (
        f"https://{normalized_app_id}.h5.xet.pomoho.com/xiaoe_clock/mini_middle"
        f"?activity_id={normalized_activity_id}&app_id={normalized_app_id}"
    )
    return (
        "https://admin.xiaoe-tech.com/t/clock_admin/index#/punchDetail/diaryList"
        f"?activity_id={normalized_activity_id}&markType=&miniMiddleUrl={quote(resolved_mini_middle_url, safe='')}"
    )


def normalize_xiaoe_clockin_activity(raw: dict[str, Any]) -> dict[str, Any]:
    activity_id = _normalize_text(raw.get("id") or raw.get("activity_id"))
    app_id = _normalize_text(raw.get("app_id"))
    mini_middle_url = _normalize_text(raw.get("mini_middle_url"))
    url = ""
    if activity_id and app_id:
        url = build_xiaoe_diary_list_url(
            activity_id=activity_id,
            app_id=app_id,
            mini_middle_url=mini_middle_url,
        )
    return {
        "activity_id": activity_id,
        "app_id": app_id,
        "title": _normalize_text(raw.get("title") or raw.get("name")),
        "url": url,
        "start_date": _normalize_text(raw.get("activity_start_at") or raw.get("start_date")),
        "end_date": _normalize_text(raw.get("activity_stop_at") or raw.get("end_date")),
        "days": int(float(raw.get("task_count") or raw.get("days") or 0)),
        "clockin_user_num": int(float(raw.get("actor_user_count") or raw.get("clockin_user_num") or 0)),
        "total_user_num": int(float(raw.get("clock_count") or raw.get("total_user_num") or 0)),
        "mini_middle_url": mini_middle_url,
        "raw": raw,
    }


def choose_attendance_clockin_activities(
    activities: list[dict[str, Any]],
    *,
    target_keywords: list[str] | None = None,
    prefer_center: bool = True,
    exclude_audit_only: bool = True,
) -> dict[str, Any]:
    """Select clockin activities for attendance.

    If a course has both center/audit-only classroom activities, attendance
    uses only the center classroom. Audit-only activities must not silently
    enter refund statistics.
    """

    keywords = [_normalize_text(item) for item in target_keywords or [] if _normalize_text(item)]
    normalized = [normalize_xiaoe_clockin_activity(item) for item in activities]
    matched = [
        item
        for item in normalized
        if item.get("activity_id") and item.get("url")
        and all(keyword in _normalize_text(item.get("title")) for keyword in keywords)
    ]
    center = [
        item for item in matched
        if _has_any_marker(item.get("title"), CENTER_CLASSROOM_MARKERS)
    ]
    audit_only = [
        item for item in matched
        if _has_any_marker(item.get("title"), AUDIT_ONLY_CLASSROOM_MARKERS)
    ]
    unmarked = [
        item for item in matched
        if item not in center and item not in audit_only
    ]

    warnings: list[str] = []
    if prefer_center and center:
        selected = center
        reason = "matched_center_classroom"
        if audit_only:
            warnings.append("已发现旁听教室打卡活动，按考勤规则排除旁听，仅保留中心教室。")
    elif exclude_audit_only and audit_only and not unmarked:
        selected = []
        reason = "audit_only_without_center"
        warnings.append("只发现旁听教室打卡活动，未自动用于考勤。")
    else:
        selected = unmarked
        reason = "matched_unmarked_activity"

    return {
        "selected": selected,
        "matched": matched,
        "excluded_audit_only": audit_only if selected is not audit_only else [],
        "warnings": warnings,
        "selection_reason": reason,
    }


def _wait_until(tab: Any, predicate: Any, timeout: float = 30.0, interval: float = 0.35) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def _body_text(tab: Any) -> str:
    return _normalize_text(tab.run_js('return document.body && document.body.innerText || ""'))


def _collect_xiaoe_punch_home_activities(tab: Any) -> list[dict[str, Any]]:
    script = """
    function safeClone(obj, depth=0, seen=new WeakSet()) {
      if (obj == null || typeof obj !== 'object') return obj;
      if (seen.has(obj) || depth > 5) return null;
      seen.add(obj);
      if (Array.isArray(obj)) return obj.slice(0, 200).map(x => safeClone(x, depth + 1, seen));
      const out = {};
      for (const k of Object.keys(obj).slice(0, 120)) {
        if (k === '$parent' || k === '$children' || k === '$root') continue;
        const v = obj[k];
        if (typeof v === 'function') continue;
        try { out[k] = safeClone(v, depth + 1, seen); } catch(e) {}
      }
      return out;
    }
    const hits = [];
    function scanVue(vm, seen=new WeakSet()) {
      if (!vm || seen.has(vm)) return;
      seen.add(vm);
      const sources = [vm._data || vm.$data || {}, vm._setupProxy || {}];
      for (const source of sources) {
        for (const k of Object.keys(source || {})) {
          let v;
          try { v = source[k]; } catch(e) { continue; }
          if (Array.isArray(v) && v.some(item => item && item.id && item.title && item.mini_middle_url)) {
            hits.push(...safeClone(v));
          }
        }
      }
      (vm.$children || []).forEach(child => scanVue(child, seen));
    }
    for (const el of document.querySelectorAll('*')) {
      if (el.__vue__) scanVue(el.__vue__);
    }
    const byId = new Map();
    for (const item of hits) {
      if (item && item.id && item.title) byId.set(item.id, item);
    }
    return Array.from(byId.values());
    """
    raw = tab.run_js(script)
    return [dict(item) for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _expand_xiaoe_punch_home_page(tab: Any, *, page_size: int = 100) -> bool:
    """Expand the activity list before matching a monthly course.

    Xiaoe defaults this page to ten rows, while monthly attendance activities
    are often created well ahead of their start date and can therefore fall
    onto page two.  Selecting the largest supported page keeps discovery
    deterministic without teaching callers how to manipulate Xiaoe's UI.
    """

    opened = bool(tab.run_js("""
    const select = document.querySelector('.ss-pagination-select');
    if (!select) return false;
    select.click();
    return true;
    """))
    if not opened:
        return False

    label = f"{int(page_size)} 条"
    choose_script = """
    const label = __LABEL__;
    const option = Array.from(document.querySelectorAll('.ss-select-dropdown__item'))
      .find(el => (el.innerText || el.textContent || '').trim() === label);
    if (!option) return false;
    option.click();
    return true;
    """.replace("__LABEL__", json.dumps(label, ensure_ascii=False))
    return _wait_until(tab, lambda: bool(tab.run_js(choose_script)), timeout=3, interval=0.15)


def _dismiss_xiaoe_clockin_onboarding(tab: Any) -> bool:
    """Dismiss Xiaoe's first-visit clockin guide when it covers the activity list."""

    script = """
    const candidates = Array.from(document.querySelectorAll('button,[role=button],span,a,div'))
      .filter(el => (el.innerText || el.textContent || '').trim() === '跳过')
      .filter(el => {
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        return rect.width > 0 && rect.height > 0
          && style.display !== 'none' && style.visibility !== 'hidden';
      })
      .sort((a, b) => {
        const rank = el => el.tagName === 'BUTTON' ? 0 : el.tagName === 'A' ? 1 : 2;
        return rank(a) - rank(b);
      });
    if (!candidates.length) return false;
    candidates[0].click();
    return true;
    """
    return bool(tab.run_js(script))


def detect_xiaoe_attendance_clockin_activities_browser(
    *,
    target_keywords: list[str],
    shop_name: str = "5034山中薪",
    home_url: str = XIAOE_CLOCKIN_HOME_URL,
    close_browser: bool = False,
) -> dict[str, Any]:
    """Detect Xiaoe punch activities and apply attendance selection rules.

    This is the reusable engineering entrypoint for monthly workbook
    materialization. It intentionally returns both matched and selected
    activities so callers can audit why audit-only classroom activities were
    excluded.
    """

    from kq5034.attendance_api import ensure_attendance_runtime  # type: ignore

    ensure_attendance_runtime()
    from kq5034.attendance_api import _close_kqtools_browser  # type: ignore
    from kq5034.tools import KqTools  # type: ignore

    kqtools = KqTools()
    try:
        tab = kqtools.xe2.switch_shop(shop_name)
        tab.get(home_url)
        _dismiss_xiaoe_clockin_onboarding(tab)
        _wait_until(
            tab,
            lambda: (
                _dismiss_xiaoe_clockin_onboarding(tab)
                or ("打卡名称" in _body_text(tab) and "操作" in _body_text(tab))
            ),
            timeout=45,
        )
        activities: list[dict[str, Any]] = []

        def collect_loaded_activities() -> bool:
            nonlocal activities
            _dismiss_xiaoe_clockin_onboarding(tab)
            activities = _collect_xiaoe_punch_home_activities(tab)
            return bool(activities)

        if not _wait_until(tab, collect_loaded_activities, timeout=30):
            raise ClockinLinkDetectionError(
                "小鹅通打卡首页已打开，但没有读取到任何活动；请检查登录状态、首次引导弹窗或页面结构。"
            )
        initial_activity_count = len(activities)
        if _expand_xiaoe_punch_home_page(tab):
            _wait_until(
                tab,
                lambda: len(_collect_xiaoe_punch_home_activities(tab)) > initial_activity_count,
                timeout=10,
                interval=0.35,
            )
            activities = _collect_xiaoe_punch_home_activities(tab)
        selection = choose_attendance_clockin_activities(
            activities,
            target_keywords=target_keywords,
            prefer_center=True,
            exclude_audit_only=True,
        )
        return {
            "home_url": home_url,
            "shop_name": shop_name,
            "target_keywords": target_keywords,
            "activity_count": len(activities),
            **selection,
        }
    finally:
        if close_browser:
            _close_kqtools_browser(kqtools)
        elif kqtools._xe2 is not None:
            kqtools.xe2.close_if_exceeds_min_tabs(1)

