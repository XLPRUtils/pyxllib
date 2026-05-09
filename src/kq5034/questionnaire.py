#!/usr/bin/env python3

"""新版问卷体系的提醒逻辑。"""

from __future__ import annotations

import datetime
import json
import os
import re
from pathlib import Path

import pandas as pd
import requests

问卷滞留提醒头部 = '【问卷滞留问题】 https://code4101.com/sheet/5'
问卷滞留提醒分组 = {
    '中台': ('中台', re.compile('念住|觉观|闯关')),
    '梵呗': ('梵呗', None),
    '禅宗': ('禅宗', None),
    '未分组': (None, None),
}
问卷滞留提醒发送目标 = {
    '中台': '考勤中台',
    '梵呗': '本体音艺考勤班委群',
    '禅宗': '禅宗修道考勤管理',
}
问卷滞留提醒去重分组 = {'禅宗'}
CodeYun问卷数据接口 = os.getenv(
    'KQ5034_CODEYUN_QUESTIONNAIRE_DATA_URL',
    'https://code4101.com/api/attendance/wjx-data',
)
CodeYun问卷数据页大小 = 100


def _默认问卷提醒状态文件():
    """问卷提醒的本地持久化状态文件。"""

    from pyxllib.prog.xlenv import get_xl_homedir

    return Path(os.fspath(get_xl_homedir())) / 'data/m2112kq5034/questionnaire/reminder_state.json'


def _读取问卷提醒状态(state_path):
    path = Path(os.fspath(state_path))
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}
    return data


def _写入问卷提醒状态(state_path, state):
    path = Path(os.fspath(state_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    tmp_path.replace(path)


def _规范问卷提醒日期(today):
    if isinstance(today, datetime.datetime):
        return today.date()
    return today or datetime.date.today()


def _是否发送问卷提醒(group_name, items, state, today):
    if not items:
        return False
    if group_name not in 问卷滞留提醒去重分组:
        return True

    last_state = state.get(group_name) or {}
    last_seen_date = last_state.get('last_seen_date')
    today_text = today.isoformat()
    yesterday_text = (today - datetime.timedelta(days=1)).isoformat()
    same_items = last_state.get('items') == items

    if today.isoweekday() == 7:
        return not (same_items and last_seen_date == today_text)

    return not (same_items and last_seen_date in {today_text, yesterday_text})


def _更新问卷提醒状态(state, group_name, items, today, *, sent):
    last_state = state.get(group_name) or {}
    record = {
        'last_seen_date': today.isoformat(),
        'items': list(items),
    }
    if sent:
        record['last_sent_date'] = today.isoformat()
    elif 'last_sent_date' in last_state:
        record['last_sent_date'] = last_state['last_sent_date']
    state[group_name] = record


def _fill_text_na(df: pd.DataFrame, value='') -> pd.DataFrame:
    """Only fill text-like columns so pandas 3 keeps numeric dtypes intact."""
    for col in df.columns:
        series = df[col]
        if (pd.api.types.is_numeric_dtype(series.dtype)
                or pd.api.types.is_bool_dtype(series.dtype)
                or pd.api.types.is_datetime64_any_dtype(series.dtype)):
            continue
        df[col] = series.fillna(value)
    return df


def _转整数(value, default=0):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _转文本(value):
    if value is None or pd.isna(value):
        return ''
    return str(value).strip()


def _读取CodeYun问卷提醒数据(api_url=CodeYun问卷数据接口, page_size=CodeYun问卷数据页大小):
    """从 CodeYun 接口分页读取问卷提醒所需字段。"""

    rows = []
    page = 1

    while True:
        resp = requests.get(
            api_url,
            params={'page': page, 'page_size': page_size, 'process_status': '__empty__'},
            timeout=(5, 20),
        )
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get('items') or []

        for item in items:
            seq = _转整数(item.get('seq'), 0)
            if not seq:
                continue
            rows.append({
                '序号': seq,
                '1、所属课程': _转文本(item.get('course_name')),
                '处理状态': _转文本(item.get('process_status')),
            })

        if len(items) < page_size:
            break

        page += 1
        if page > 100:
            raise RuntimeError('CodeYun 问卷提醒分页超过 100 页，疑似接口异常')

    if not rows:
        return pd.DataFrame(columns=['序号', '1、所属课程', '处理状态'])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['序号'], keep='last')
    _fill_text_na(df, '')
    return df


def 分析问卷滞留记录(df: pd.DataFrame, *, header=问卷滞留提醒头部):
    """按课程分组整理未处理问卷，供任意壳层决定是否发送提醒。"""

    groups = {name: [] for name in 问卷滞留提醒分组}
    if len(df) == 0:
        return {
            name: {'items': [], 'message': '【当前无问卷滞留问题】'}
            for name in groups
        }

    df2 = df.copy()
    if '处理状态' in df2.columns:
        df2['处理状态'] = df2['处理状态'].fillna('')
        df2 = df2[df2['处理状态'] == ''].copy()

    if len(df2) == 0:
        return {
            name: {'items': [], 'message': '【当前无问卷滞留问题】'}
            for name in groups
        }

    df2['序号'] = pd.to_numeric(df2['序号'], errors='coerce')
    df2 = df2[df2['序号'].notna()].copy()
    if len(df2) == 0:
        return {
            name: {'items': [], 'message': '【当前无问卷滞留问题】'}
            for name in groups
        }

    df2['序号'] = df2['序号'].astype(int)
    df2['1、所属课程'] = df2['1、所属课程'].fillna('')
    df2 = df2.sort_values(by='序号')
    grouped = df2.groupby('1、所属课程')['序号'].apply(lambda x: ','.join(map(str, x)))

    for course, seq in grouped.items():
        target_group = '未分组'
        for name, (keyword, pattern) in 问卷滞留提醒分组.items():
            if name == '未分组':
                continue
            if keyword and keyword in course:
                target_group = name
                break
            if pattern and pattern.search(course):
                target_group = name
                break
        groups[target_group].append(f'{course}：{seq}')

    result = {}
    for name, items in groups.items():
        formatted_items = items
        if len(items) > 1:
            formatted_items = [f'{i + 1}. {item}' for i, item in enumerate(items)]
        if formatted_items:
            message = '\n'.join([header] + formatted_items)
        else:
            message = '【当前无问卷滞留问题】'
        result[name] = {'items': items, 'message': message}

    return result


def 提醒问卷数据(api_url=CodeYun问卷数据接口, *, state_path=None, today=None):
    """从 CodeYun 读取滞留问卷数据并发送微信提醒。"""

    from .common import wechat_lock_send

    df = _读取CodeYun问卷提醒数据(api_url=api_url)
    groups = 分析问卷滞留记录(df)
    today = _规范问卷提醒日期(today)
    state_path = state_path or _默认问卷提醒状态文件()
    state = _读取问卷提醒状态(state_path)
    sent_groups = {}

    for group_name, dst in 问卷滞留提醒发送目标.items():
        payload = groups[group_name]
        items = payload['items']
        should_send = _是否发送问卷提醒(group_name, items, state, today)
        if should_send:
            wechat_lock_send(dst, payload['message'])
        sent_groups[group_name] = should_send

    for group_name in 问卷滞留提醒去重分组:
        if group_name in groups:
            _更新问卷提醒状态(
                state,
                group_name,
                groups[group_name]['items'],
                today,
                sent=sent_groups.get(group_name, False),
            )

    _写入问卷提醒状态(state_path, state)

    return groups


__all__ = [
    'CodeYun问卷数据接口',
    '_读取CodeYun问卷提醒数据',
    '分析问卷滞留记录',
    '提醒问卷数据',
]
