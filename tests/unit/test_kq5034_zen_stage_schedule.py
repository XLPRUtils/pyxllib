from __future__ import annotations

import datetime

from kq5034.tools import KqTools


def _row(next_update: datetime.datetime, *, lesson_name: str = "d260712禅宗13期一阶-第2周=佛教概观2"):
    return {
        "lesson_name": lesson_name,
        "start_date": datetime.datetime(2026, 7, 19),
        "next_update": next_update,
        "end_date": datetime.datetime(2026, 11, 22),
        "video_duration": 0,
    }


def test_zen_stage_dirty_saturday_update_aligns_to_sunday():
    row = _row(datetime.datetime(2026, 7, 25))

    assert KqTools._对齐禅宗课次更新时间(row) == datetime.datetime(2026, 7, 26)
    assert KqTools._计算课次下一次需要更新的时间点(
        row,
        now=datetime.datetime(2026, 7, 22),
    ) == datetime.datetime(2026, 7, 26)


def test_zen_stage_exact_boundary_advances_one_week():
    row = _row(datetime.datetime(2026, 7, 26))

    assert KqTools._计算课次下一次需要更新的时间点(
        row,
        now=datetime.datetime(2026, 7, 26),
    ) == datetime.datetime(2026, 8, 2)


def test_repair_class_name_is_also_treated_as_zen_stage():
    row = _row(datetime.datetime(2026, 7, 25), lesson_name="d260517修道班7期5阶-第10周=坛经14")

    assert KqTools._是禅宗修道班课次(row)
    assert KqTools._对齐禅宗课次更新时间(row) == datetime.datetime(2026, 7, 26)
