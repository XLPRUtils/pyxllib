"""Stable service API for KQ5034 attendance integrations.

This module is the boundary used by external services such as codeyun.  It
keeps HTTP/service layers from importing script entrypoints or reaching into
``KqTools`` internals directly.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import socket
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Union


XIAOETONG_SHOP_NAMES = {
    1: "5034山中薪",
    2: "宗门学府",
}


DEFAULT_XL_HOSTS = """[
["host",        "raw_name",     "homedir"],
["codepc_mf",   "",             "D:/home/chenkunze"],
["codepc_mi15", "",             "C:/home/chenkunze"],
["xlpr0,titan1,titan2,tesla1,tesla2,xlpr4,xlpr8,xlpr10,xlpr6",
                "",             "/home/chenkunze"]
]"""


def ensure_attendance_runtime() -> None:
    """Prepare the small amount of runtime environment KQ5034 expects."""

    current_file = Path(__file__).resolve()
    slns_dir = current_file.parents[3] if len(current_file.parents) > 3 else None
    xlproject_src = slns_dir / "xlproject" / "src" if slns_dir is not None else None
    if xlproject_src is not None and xlproject_src.exists():
        path_text = str(xlproject_src)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    env_candidates = [
        Path(os.environ["KQ5034_ATTENDANCE_ENV_PATH"]) if os.environ.get("KQ5034_ATTENDANCE_ENV_PATH") else None,
        slns_dir / "xlproject" / ".env" if slns_dir is not None else None,
    ]
    for env_path in env_candidates:
        if env_path is None or not env_path.exists():
            continue
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path, override=False)
            break
        except Exception:
            pass

    try:
        loadenv = importlib.import_module("xlproject.loadenv")
        importlib.reload(loadenv)
    except Exception:
        pass

    if not os.environ.get("XL_HOSTS"):
        os.environ["XL_HOSTS"] = DEFAULT_XL_HOSTS

    if not os.environ.get("XL_HOMEDIR"):
        hostname = socket.gethostname().replace("-", "_").split(".")[0]
        homedir_by_host = {
            "codepc_mf": "D:/home/chenkunze",
            "codepc_mi15": "C:/home/chenkunze",
        }
        homedir = homedir_by_host.get(hostname)
        if homedir:
            os.environ["XL_HOMEDIR"] = homedir


def get_kqdb() -> Any:
    """Return a configured KQ5034 database handle."""

    ensure_attendance_runtime()
    from .db import get_kqdb as _get_kqdb

    return _get_kqdb()


def lookup_order(order_id: Any, **kwargs: Any) -> Dict[str, Any]:
    """Lookup one WeChat Pay order through the shared KQ5034 implementation."""

    ensure_attendance_runtime()
    from .order_ops import lookup_order as _lookup_order

    return _lookup_order(order_id, **kwargs)


def lookup_registration_user_db(
    names: Any = None,
    phones: Any = None,
    *,
    course_name: str = "",
    course_product_name: str = "",
    shop_id: int = 1,
    return_mode: int = 1,
    kqdb: Any = None,
) -> Any:
    """Lookup one registration user from the local KQ5034 database."""

    ensure_attendance_runtime()
    if kqdb is None:
        kqdb = get_kqdb()
    return kqdb.查找用户(
        _as_text_list(names),
        _as_text_list(phones),
        课程标准名=course_name,
        课程商品名=course_product_name,
        shop_id=shop_id,
        return_mode=return_mode,
    )


def _as_text_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Sequence):
        values = [values]
    return [str(value).strip() for value in values if str(value).strip()]


def _normalize_shop(shop_id: Union[int, str] = 1):
    if isinstance(shop_id, str):
        shop_text = shop_id.strip()
        for candidate_id, candidate_name in XIAOETONG_SHOP_NAMES.items():
            if shop_text == candidate_name:
                return candidate_id, candidate_name
        shop_id = shop_text or 1

    try:
        normalized_shop_id = int(shop_id or 1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid shop_id: {shop_id!r}") from exc

    shop_name = XIAOETONG_SHOP_NAMES.get(normalized_shop_id)
    if not shop_name:
        raise ValueError(f"unsupported shop_id: {normalized_shop_id}")
    return normalized_shop_id, shop_name


def _get_item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _close_kqtools_browser(kqtools: Any) -> None:
    xe2 = getattr(kqtools, "_xe2", None)
    if xe2 is None:
        return
    browser = getattr(xe2, "browser", None)
    for close_target in (
        getattr(xe2, "quit", None),
        getattr(browser, "quit", None),
        getattr(browser, "close", None),
    ):
        if not callable(close_target):
            continue
        try:
            close_target()
            return
        except Exception:
            continue


def lookup_registration_users_browser(
    items: Sequence[Any],
    *,
    course_name: str = "",
    course_product_name: str = "",
    shop_id: Union[int, str] = 1,
    close_browser: bool = True,
) -> List[Dict[str, Any]]:
    """Lookup registration users through Xiaoe web automation.

    Each item may be a mapping or object with ``key``, ``names`` and ``phones``.
    The return value is a list of ``{"key": ..., "user_id": ..., "error": ...}``
    dictionaries so callers can map results back to their own rows.
    """

    ensure_attendance_runtime()
    from .tools import KqTools

    if not items:
        return []

    kqtools = KqTools()
    try:
        _, shop_name = _normalize_shop(shop_id)
        kqtools.xe2.switch_shop(shop_name)

        results: List[Dict[str, Any]] = []
        for item in items:
            key = str(_get_item_value(item, "key", "")).strip()
            names = _as_text_list(_get_item_value(item, "names", []))
            phones = _as_text_list(_get_item_value(item, "phones", []))
            try:
                user_id = kqtools.xe2.查找用户(names, phones, course_name, course_product_name)
                results.append({"key": key, "user_id": str(user_id or "")})
            except Exception as exc:
                results.append({"key": key, "user_id": "", "error": str(exc)})
        return results
    finally:
        if close_browser:
            _close_kqtools_browser(kqtools)


def sync_fanbei_attendance_step1(
    *,
    course_name: str,
    shop_id: Union[int, str] = 1,
    update_lessons: bool = True,
    update_clockins: bool = True,
    clockin_pattern: str = "",
    close_browser: bool = True,
) -> Dict[str, Any]:
    """Run Fanbei step1 through the shared KQ5034 automation layer.

    Step1 is the old ``KqCourse.step1`` data-download part without any WPS
    workbook status mutation: switch Xiaoe shop, update due lesson playback
    data, then update matching clock-in data.  The caller decides scheduling
    and status persistence.
    """

    ensure_attendance_runtime()
    from .tools import KqTools

    normalized_course_name = str(course_name or "").strip()
    if not normalized_course_name:
        raise ValueError("course_name is required")

    normalized_shop_id, shop_name = _normalize_shop(shop_id)
    resolved_clockin_pattern = str(clockin_pattern or "").strip() or f"{normalized_course_name}-*"
    kqtools = KqTools()
    lesson_update_count: int | None = None
    clockin_names: list[str] = []

    try:
        kqtools.xe2.switch_shop(shop_name)

        if update_lessons:
            lesson_update_count = kqtools.update_lesson_data_table(
                shop1=normalized_shop_id == 1,
                shop2=normalized_shop_id == 2,
            )

        if update_clockins:
            clockin_names = [str(name) for name in kqtools._resolve_clockin_names(resolved_clockin_pattern)]
            if clockin_names:
                kqtools.update_clockin(resolved_clockin_pattern)

        return {
            "course_name": normalized_course_name,
            "shop_id": normalized_shop_id,
            "shop_name": shop_name,
            "lesson_update_count": lesson_update_count,
            "clockin_pattern": resolved_clockin_pattern,
            "clockin_names": clockin_names,
            "clockin_update_count": len(clockin_names) if update_clockins else 0,
            "update_lessons": bool(update_lessons),
            "update_clockins": bool(update_clockins),
        }
    finally:
        if close_browser:
            _close_kqtools_browser(kqtools)
        kqdb = getattr(kqtools, "_kqdb", None)
        if kqdb is not None:
            close = getattr(kqdb, "close", None)
            if callable(close):
                close()


def build_fanbei_attendance_step2_data(
    *,
    course_name: str,
    user_ids: Sequence[Any],
    clockin_names: Sequence[str] | None = None,
    clockin_titles: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Build Fanbei step2 attendance data from the local KQ5034 database."""

    ensure_attendance_runtime()
    from .tools import KqTools

    normalized_course_name = str(course_name or "").strip()
    if not normalized_course_name:
        raise ValueError("course_name is required")

    normalized_user_ids = [str(value or "").strip() for value in user_ids]
    normalized_clockin_names = [str(value or "").strip() for value in (clockin_names or ["打卡数"]) if str(value or "").strip()]
    normalized_clockin_titles = [
        str(value or "").strip()
        for value in (
            clockin_titles
            if clockin_titles is not None
            else [f"学修日志{i:02}" for i in range(1, 12)]
        )
        if str(value or "").strip()
    ]

    kqtools = KqTools()
    try:
        df_clockin = kqtools.kqdb.browser_clockin_data(
            f"{normalized_course_name}-",
            normalized_clockin_names,
            user_id2s=normalized_user_ids,
            titles=normalized_clockin_titles,
        )
        df_lesson = kqtools.browser_lesson_data(
            normalized_course_name,
            user_id2s=normalized_user_ids,
        )
        df_result = kqtools.拼接打卡视频数据(normalized_user_ids, df_clockin, df_lesson)
        df_result = df_result.replace([float("inf"), float("-inf")], None)
        df_result = df_result.where(df_result.notnull(), "")

        return {
            "course_name": normalized_course_name,
            "columns": [str(column) for column in df_result.columns],
            "rows": df_result.values.tolist(),
            "user_count": len(normalized_user_ids),
            "clockin_names": normalized_clockin_names,
            "clockin_titles": normalized_clockin_titles,
        }
    finally:
        kqdb = getattr(kqtools, "_kqdb", None)
        if kqdb is not None:
            close = getattr(kqdb, "close", None)
            if callable(close):
                close()


def inspect_fanbei_attendance_step2_source(
    *,
    course_name: str,
    user_ids: Sequence[Any] | None = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Inspect source DB rows used by Fanbei step2."""

    ensure_attendance_runtime()
    from .tools import KqTools

    normalized_course_name = str(course_name or "").strip()
    if not normalized_course_name:
        raise ValueError("course_name is required")

    normalized_user_ids = [str(value or "").strip() for value in (user_ids or []) if str(value or "").strip()]
    safe_course_name = normalized_course_name.replace("'", "''")
    safe_limit = max(1, min(int(limit or 10), 50))

    kqtools = KqTools()
    try:
        lessons = kqtools.kqdb.exec2dict(
            "SELECT lesson_id, lesson_name, lesson_id2, shop_id, start_date, next_update, "
            "video_duration, end_date FROM lesson_table "
            f"WHERE lesson_name LIKE '%{safe_course_name}%' ORDER BY lesson_id"
        ).fetchall()

        user_in_clause = ""
        if normalized_user_ids:
            escaped = ",".join("'" + item.replace("'", "''") + "'" for item in normalized_user_ids)
            user_in_clause = f" AND user_id2 IN ({escaped})"
        lesson_summaries = []
        for lesson in lessons:
            lesson_id = lesson["lesson_id"]
            total_count = kqtools.kqdb.exec2one(
                f"SELECT COUNT(*) FROM lesson_data_table WHERE lesson_id={lesson_id}"
            )
            matched_count = kqtools.kqdb.exec2one(
                f"SELECT COUNT(DISTINCT user_id2) FROM lesson_data_table WHERE lesson_id={lesson_id}{user_in_clause}"
            ) if user_in_clause else None
            sample_rows = kqtools.kqdb.exec2dict(
                "SELECT user_id2, progress, study_state, cum_seconds, finish_time, last_play_time "
                f"FROM lesson_data_table WHERE lesson_id={lesson_id}{user_in_clause} "
                f"ORDER BY lesson_data_id LIMIT {safe_limit}"
            ).fetchall()
            if not sample_rows:
                sample_rows = kqtools.kqdb.exec2dict(
                    "SELECT user_id2, progress, study_state, cum_seconds, finish_time, last_play_time "
                    f"FROM lesson_data_table WHERE lesson_id={lesson_id} "
                    f"ORDER BY lesson_data_id LIMIT {safe_limit}"
                ).fetchall()
            lesson_summaries.append({
                **dict(lesson),
                "lesson_data_count": int(total_count or 0),
                "matched_user_count": None if matched_count is None else int(matched_count or 0),
                "sample_rows": [dict(row) for row in sample_rows],
            })

        return {
            "course_name": normalized_course_name,
            "input_user_count": len(normalized_user_ids),
            "lesson_count": len(lesson_summaries),
            "lessons": lesson_summaries,
        }
    finally:
        kqdb = getattr(kqtools, "_kqdb", None)
        if kqdb is not None:
            close = getattr(kqdb, "close", None)
            if callable(close):
                close()


def inspect_fanbei_lesson_export_page(
    *,
    course_name: str,
    lesson_number: int = 1,
    shop_id: Union[int, str] = 1,
) -> Dict[str, Any]:
    """Open one Fanbei lesson export page and inspect visible export controls."""

    ensure_attendance_runtime()
    from .tools import KqTools

    normalized_course_name = str(course_name or "").strip()
    if not normalized_course_name:
        raise ValueError("course_name is required")
    _, shop_name = _normalize_shop(shop_id)

    lesson_label = f"第{int(lesson_number):02}课"
    safe_course_name = normalized_course_name.replace("'", "''")
    safe_lesson_label = lesson_label.replace("'", "''")

    kqtools = KqTools()
    try:
        lesson = kqtools.kqdb.exec2dict(
            "SELECT * FROM lesson_table "
            f"WHERE lesson_name LIKE '%{safe_course_name}%' AND lesson_name LIKE '%{safe_lesson_label}%' "
            "ORDER BY lesson_id LIMIT 1"
        ).fetchone()
        if not lesson:
            raise ValueError(f"lesson not found: {normalized_course_name} {lesson_label}")

        kqtools.xe2.switch_shop(shop_name)
        lesson_id2 = str(lesson.get("lesson_id2") or "").strip()
        work_url = lesson_id2
        wait_seconds = 3
        if not lesson_id2.startswith("https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page") \
                and not lesson_id2.startswith("https://admin.xiaoe-tech.com/t/course/camp_pro/course_detail_page"):
            work_url = f"https://admin.xiaoe-tech.com/t/live_management#/userOperation?id={lesson_id2}&tabName=UserManage"
            wait_seconds = 5
        elif lesson_id2.startswith("https://admin.xiaoe-tech.com/t/course/camp_pro/course_detail_page"):
            wait_seconds = 5

        with kqtools.xe2.临时工作标签页(url=work_url, wait_seconds=wait_seconds) as tab:
            def count(locator: str) -> int:
                try:
                    return len(tab.eles(locator, timeout=1))
                except Exception:
                    return -1

            body_text = ""
            try:
                body_text = tab.run_js("return document.body.innerText || ''") or ""
            except Exception as exc:
                body_text = f"<body text error: {exc}>"

            rows_preview = []
            for locator in ("t:table@@class:ss-table__body", "t:div@@class:ss-table__body-wrapper", "tag:tbody"):
                try:
                    holder = tab(locator, timeout=1)
                    if holder:
                        rows_preview = [(row.text or "")[:300] for row in holder.eles("t:tr", timeout=1)[:5]]
                        if rows_preview:
                            break
                except Exception:
                    continue

            return {
                "lesson": dict(lesson),
                "work_url": work_url,
                "current_url": str(getattr(tab, "url", "")),
                "body_text": body_text[:3000],
                "selector_counts": {
                    "table.ss-table__body": count("t:table@@class:ss-table__body"),
                    "div.ss-table__body-wrapper": count("t:div@@class:ss-table__body-wrapper"),
                    "tbody": count("tag:tbody"),
                    "export_list_buttons": count("tag:button@@text()=导出列表"),
                    "export_buttons": count("tag:button@@text():导出"),
                    "empty_text": count("text:暂无数据"),
                },
                "rows_preview": rows_preview,
            }
    finally:
        if getattr(kqtools, "_xe2", None) is not None:
            _close_kqtools_browser(kqtools)
        kqdb = getattr(kqtools, "_kqdb", None)
        if kqdb is not None:
            close = getattr(kqdb, "close", None)
            if callable(close):
                close()
