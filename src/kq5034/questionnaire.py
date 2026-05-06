#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""新版问卷体系的提醒逻辑。"""

from __future__ import annotations

import os
import re

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
CodeYun问卷数据接口 = os.getenv(
    'KQ5034_CODEYUN_QUESTIONNAIRE_DATA_URL',
    'https://code4101.com/api/attendance/wjx-data',
)
CodeYun问卷数据页大小 = 100


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


def 提醒问卷数据(api_url=CodeYun问卷数据接口):
    """从 CodeYun 读取滞留问卷数据并发送微信提醒。"""

    from .common import wechat_lock_send

    df = _读取CodeYun问卷提醒数据(api_url=api_url)
    groups = 分析问卷滞留记录(df)

    for group_name, dst in 问卷滞留提醒发送目标.items():
        payload = groups[group_name]
        if payload['items']:
            wechat_lock_send(dst, payload['message'])

    return groups


__all__ = [
    'CodeYun问卷数据接口',
    '_读取CodeYun问卷提醒数据',
    '分析问卷滞留记录',
    '提醒问卷数据',
]
