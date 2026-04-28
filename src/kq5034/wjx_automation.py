#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2026/02/28

""" 问卷星自动化脚本 """

import os
import re
import json
import logging
import time
import pandas as pd
import requests

from pyxllib.ext.wjxlib import WjxWeb, 清理问卷星干扰弹窗

默认问卷activity_id = 264266843
课程题目标题 = '所属课程'


class _LazyKqLogger:
    """Prefer the KQ5034 loguru logger when the legacy runtime is available."""

    def __getattr__(self, name):
        try:
            from .common import logger as kq_logger
        except Exception:
            return getattr(logging.getLogger(__name__), name)
        return getattr(kq_logger, name)


logger = _LazyKqLogger()


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


问卷星结果字段 = [
    '序号',
    '提交答卷时间',
    '所用时间',
    '来源',
    '来源详情',
    '来自IP',
    '1、所属课程',
    '2、学号',
    '3、姓名',
    '4、修正需求',
    '5、其他补充说明',
]
问卷滞留提醒头部 = '【问卷滞留问题】 https://code4101.com/attendance/questionnaire/data'
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
    'KQ5034_CODEYUN_WJX_DATA_URL',
    'https://code4101.com/api/attendance/wjx-data',
)
CodeYun问卷数据页大小 = 100


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


def _转整数或原值(value):
    text = _转文本(value)
    if not text:
        return ''
    if re.fullmatch(r'-?\d+(?:\.0+)?', text):
        return int(float(text))
    return text


def 标准化问卷星记录(df: pd.DataFrame) -> pd.DataFrame:
    """提取固定字段并统一列名，方便不同壳层共用。"""

    data = {}
    row_count = len(df.index)
    for col in 问卷星结果字段:
        matched = next((actual_col for actual_col in df.columns if actual_col.startswith(col)), None)
        if matched:
            data[col] = df[matched]
        else:
            data[col] = pd.Series([pd.NA] * row_count)

    df2 = pd.DataFrame(data)
    if '序号' in df2.columns:
        df2['序号'] = pd.to_numeric(df2['序号'], errors='coerce')
        df2 = df2[df2['序号'].notna()].copy()
        if len(df2):
            df2['序号'] = df2['序号'].astype(int)

    _fill_text_na(df2, '')
    return df2


def 筛选问卷星增量记录(df: pd.DataFrame, exist_max_id=0) -> pd.DataFrame:
    """按序号筛选新增记录，返回升序结果。"""

    if len(df) == 0:
        return df.copy()

    exist_max_id = _转整数(exist_max_id, 0)
    df2 = df.copy()
    df2['序号'] = pd.to_numeric(df2['序号'], errors='coerce')
    df2 = df2[df2['序号'].notna()].copy()
    if len(df2):
        df2['序号'] = df2['序号'].astype(int)
    df2 = df2[df2['序号'] > exist_max_id].copy()
    if len(df2):
        df2 = df2.sort_values(by='序号')
    return df2


def 获取问卷设计页链接(activity_id):
    return f'https://www.wjx.cn/wjx/design/designstart.aspx?activity={activity_id}'


def 获取问卷结果汇总页链接(activity_id):
    return f'https://www.wjx.cn/wjx/activitystat/viewstatsummary.aspx?activity={activity_id}&sat=1&op=1'


def 打开问卷设计页(activity_id=264266843, wjx=None):
    """ 打开指定问卷的设计页，并清理干扰弹窗 """
    wjx = wjx or 自动登录问卷星()
    target = 获取问卷设计页链接(activity_id)
    if wjx.tab.url != target:
        try:
            wjx.tab.handle_alert(accept=True, next_one=True)
        except Exception:
            pass
        wjx.tab.get(target)
        try:
            wjx.tab.handle_alert(accept=True, timeout=1)
        except Exception:
            pass
    清理多余问卷星标签页(wjx.tab)
    清理问卷星干扰弹窗(wjx.tab)
    return wjx


def _获取问卷运行按钮(tab):
    return tab('#ctl02_ContentPlaceHolder1_btnRun', timeout=5)


def _获取元素文本(ele):
    return (getattr(ele, 'value', None) or ele.text or '').strip()


def _按顺序去重(items):
    seen = set()
    result = []
    for item in items or []:
        item = (item or '').strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _等待直到(checker, timeout=10, interval=0.2, desc='目标状态'):
    end_time = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < end_time:
        try:
            value = checker()
            if value:
                return value
        except Exception as e:
            last_error = e
        time.sleep(interval)

    if last_error:
        raise TimeoutError(f'等待{desc}超时，最后错误：{last_error}')
    raise TimeoutError(f'等待{desc}超时')


def _等待元素(tab, selectors, timeout=10, desc='元素'):
    if isinstance(selectors, str):
        selectors = [selectors]

    def checker():
        for loc in selectors:
            try:
                ele = tab.ele(loc, timeout=1)
            except Exception:
                continue
            if ele:
                return ele

    return _等待直到(checker, timeout=timeout, desc=desc)


def _等待文本元素(tab, texts, timeout=10, desc='按钮'):
    selectors = []
    for text in texts:
        selectors.extend([
            f'xpath://a[contains(normalize-space(.), "{text}")]',
            f'xpath://button[contains(normalize-space(.), "{text}")]',
            f'xpath://span[contains(normalize-space(.), "{text}")]',
            f'xpath://input[contains(@value, "{text}")]',
        ])
    return _等待元素(tab, selectors, timeout=timeout, desc=desc)


def _等待网址包含(tab, keyword, timeout=15, desc='页面跳转'):
    return _等待直到(lambda: keyword in tab.url and tab.url, timeout=timeout, desc=desc)


def _运行js_json(tab, script):
    res = tab.run_js(script)
    return json.loads(res) if res else None


def 清理多余问卷星标签页(tab):
    """ 只保留当前问卷星标签，避免堆积太多 wjx.cn 页面 """
    try:
        tabs = tab.browser.get_tabs(url='wjx.cn')
    except Exception:
        return 0

    current_id = getattr(tab, 'tab_id', None)
    closed = 0
    for tab2 in tabs:
        if current_id is not None and getattr(tab2, 'tab_id', None) == current_id:
            continue
        try:
            tab2.close()
            closed += 1
        except Exception:
            pass

    if closed:
        logger.info(f'已清理多余问卷星标签页：{closed}个')
    return closed


def _获取问卷运行状态(tab):
    btn = _获取问卷运行按钮(tab)
    text = _获取元素文本(btn)
    if text == '恢复运行':
        return 'paused'
    if text == '暂停接收答卷':
        return 'running'
    raise RuntimeError(f'无法识别问卷运行按钮文案：{text!r}')


def _确认问卷弹窗(tab):
    selectors = [
        'tag:input@@value:确认',
        'tag:button@@text():确认',
        'tag:input@@value:确定',
        'tag:button@@text():确定',
        'tag:a@@text():确认',
        'tag:a@@text():确定',
        'tag:span@@text():确认',
        'tag:span@@text():确定',
    ]
    for loc in selectors:
        try:
            ele = tab(loc, timeout=2)
            if ele:
                ele.click(by_js=True)
                return True
        except Exception:
            pass
    return False


def 暂停接收答卷(activity_id=264266843, wjx=None):
    """ 暂停问卷接收答卷，并校验按钮变为“恢复运行” """
    wjx = 打开问卷设计页(activity_id, wjx=wjx)
    tab = wjx.tab
    before_text = _获取元素文本(_获取问卷运行按钮(tab))

    if before_text == '恢复运行':
        logger.info('问卷当前已暂停，看到“恢复运行”按钮，跳过重复操作')
        return wjx

    if before_text != '暂停接收答卷':
        raise RuntimeError(f'未找到预期的“暂停接收答卷”按钮，当前按钮文案：{before_text!r}')

    _获取问卷运行按钮(tab).click(by_js=True)
    time.sleep(0.5)
    _确认问卷弹窗(tab)

    after_text = _等待直到(
        lambda: _获取元素文本(_获取问卷运行按钮(tab)) == '恢复运行' and '恢复运行',
        timeout=10,
        desc='暂停后出现恢复运行按钮',
    )
    if after_text != '恢复运行':
        raise RuntimeError(f'暂停后校验失败，未看到“恢复运行”按钮，当前按钮文案：{after_text!r}')

    logger.info('已暂停接收答卷，并校验看到“恢复运行”按钮')
    return wjx


def 恢复接收答卷(activity_id=默认问卷activity_id, wjx=None):
    """ 恢复问卷运行，并校验按钮变为“暂停接收答卷” """
    wjx = 打开问卷设计页(activity_id, wjx=wjx)
    tab = wjx.tab
    before_text = _获取元素文本(_获取问卷运行按钮(tab))

    if before_text == '暂停接收答卷':
        logger.info('问卷当前已恢复运行，看到“暂停接收答卷”按钮，跳过重复操作')
        return wjx

    if before_text != '恢复运行':
        raise RuntimeError(f'未找到预期的“恢复运行”按钮，当前按钮文案：{before_text!r}')

    _获取问卷运行按钮(tab).click(by_js=True)
    time.sleep(0.5)
    _确认问卷弹窗(tab)

    after_text = _等待直到(
        lambda: _获取元素文本(_获取问卷运行按钮(tab)) == '暂停接收答卷' and '暂停接收答卷',
        timeout=10,
        desc='恢复后出现暂停接收答卷按钮',
    )
    if after_text != '暂停接收答卷':
        raise RuntimeError(f'恢复后校验失败，未看到“暂停接收答卷”按钮，当前按钮文案：{after_text!r}')

    logger.info('已恢复接收答卷，并校验看到“暂停接收答卷”按钮')
    return wjx


def 打开问卷编辑页(activity_id=默认问卷activity_id, wjx=None):
    """ 打开正式编辑页 designnew.aspx """
    wjx = wjx or 自动登录问卷星()
    tab = wjx.tab
    清理多余问卷星标签页(tab)

    if 'designnew.aspx' in tab.url and f'curid={activity_id}' in tab.url:
        return _等待问卷编辑页就绪(wjx)

    if not ('editquestionnaire.aspx' in tab.url and f'activity={activity_id}' in tab.url):
        打开问卷设计页(activity_id, wjx=wjx)
        edit_selectors = [
            'css:#ctl02_ContentPlaceHolder1_hrefEdit',
            f'xpath://a[contains(@href, "/newwjx/design/editquestionnaire.aspx?activity={activity_id}")]',
            f'xpath://a[contains(@href, "/newwjx/design/editquestionnaire.aspx") and contains(normalize-space(.), "编辑问卷")]',
        ]
        _等待元素(tab, edit_selectors, timeout=10, desc='编辑问卷入口').click(by_js=True)
        _等待直到(lambda: ('editquestionnaire.aspx' in tab.url or 'designnew.aspx' in tab.url) and tab.url,
                timeout=15, desc='进入问卷编辑模式')

    if 'editquestionnaire.aspx' in tab.url:
        _等待文本元素(tab, ['下一步'], timeout=10, desc='修改模式下一步按钮').click(by_js=True)

    _等待网址包含(tab, 'designnew.aspx', timeout=20, desc='正式问卷编辑页')
    return _等待问卷编辑页就绪(wjx)


def _等待问卷编辑页就绪(wjx):
    tab = wjx.tab

    def checker():
        if 'designnew.aspx' not in tab.url:
            return False
        info = _运行js_json(tab, f"""
return JSON.stringify((() => {{
    const holders = window.questionHolder || [];
    const title = {json.dumps(课程题目标题, ensure_ascii=False)};
    for (let i = 0; i < holders.length; i++) {{
        const dn = holders[i] && holders[i].dataNode;
        if (dn && dn._title === title) {{
            return {{
                ready: true,
                topic: dn._topic || i + 1,
                title: dn._title,
                select_len: (dn._select || []).length
            }};
        }}
    }}
    return {{ready: false}};
}})())
""")
        return info if info and info.get('ready') else False

    _等待直到(checker, timeout=20, desc='问卷编辑页就绪')
    清理问卷星干扰弹窗(tab)
    return wjx


def _获取所属课程题目信息(tab):
    info = _运行js_json(tab, f"""
return JSON.stringify((() => {{
    const holders = window.questionHolder || [];
    const title = {json.dumps(课程题目标题, ensure_ascii=False)};
    for (let i = 0; i < holders.length; i++) {{
        const dn = holders[i] && holders[i].dataNode;
        if (dn && dn._title === title) {{
            return {{
                topic: dn._topic || i + 1,
                title: dn._title,
                type: dn._type,
                select_len: (dn._select || []).length
            }};
        }}
    }}
    return null;
}})())
    """)
    if not info:
        raise RuntimeError(f'未找到题目“{课程题目标题}”')
    info['topic'] = int(info['topic'])
    if info.get('select_len') is not None:
        info['select_len'] = int(info['select_len'])
    return info


def 进入所属课程编辑态(wjx):
    """ 激活第1题所属课程，并展开其选项编辑区 """
    tab = wjx.tab
    info = _获取所属课程题目信息(tab)
    res = _运行js_json(tab, f"""
return JSON.stringify((() => {{
    const txt = el => ((el && (el.innerText || el.textContent)) || '').trim();
    const title = {json.dumps(课程题目标题, ensure_ascii=False)};
    const q = typeof getDivByTopic === 'function' ? getDivByTopic({info['topic']}) : null;
    if (!q) return {{ok: false, error: 'question_not_found'}};
    q.scrollIntoView({{block: 'center'}});
    q.click();
    let edit = [...q.querySelectorAll('span, a, button')].find(el => txt(el) === '编辑');
    if (!edit) {{
        edit = [...document.querySelectorAll('.div_question.qactive span, .div_question.qactive a, .div_question.qactive button')]
            .find(el => txt(el) === '编辑');
    }}
    const alreadyEditing = !!(window.cur && window.cur.dataNode && window.cur.dataNode._title === title
        && document.querySelectorAll('.item_title').length > 0);
    if (edit) edit.click();
    return {{ok: !!edit || alreadyEditing, text: txt(q).slice(0, 120), alreadyEditing}};
}})())
""")
    if not res or not res.get('ok'):
        raise RuntimeError(f'进入“{课程题目标题}”编辑态失败：{res}')

    _等待直到(
        lambda: len(读取所属课程选项(wjx, 仅非隐藏=False)['全部课程']) > 0,
        timeout=10,
        desc='所属课程选项加载',
    )
    return wjx


def 读取所属课程选项(wjx, 仅非隐藏=False):
    """ 读取第1题所属课程清单，默认返回全部课程及隐藏状态 """
    tab = wjx.tab
    info = _获取所属课程题目信息(tab)
    data = _运行js_json(tab, f"""
return JSON.stringify((() => {{
    const dn = typeof getDataNodeByTopic === 'function'
        ? getDataNodeByTopic({info['topic']})
        : (window.questionHolder && window.questionHolder[{info['topic'] - 1}] ? window.questionHolder[{info['topic'] - 1}].dataNode : null);
    if (!dn) return null;
    const items = [];
    const selects = dn._select || [];
    for (let i = 1; i < selects.length; i++) {{
        const item = selects[i];
        if (!item) continue;
        const name = (item._item_title || '').trim();
        if (!name) continue;
        items.push({{
            顺序: i,
            名称: name,
            已隐藏: item._item_relation === '-1',
            关联值: item._item_relation || '',
            选项值: item._item_value || ''
        }});
    }}
    return {{
        题目topic: dn._topic || {info['topic']},
        全部课程: items,
        非隐藏课程: items.filter(x => !x['已隐藏']).map(x => x['名称'])
    }};
}})())
""")
    if not data:
        raise RuntimeError(f'读取“{课程题目标题}”选项失败')

    if 仅非隐藏:
        return data['非隐藏课程']
    return data


def _校验新增课程(add, existing_names):
    if not add:
        return

    dup_in_add = []
    seen = set()
    for name in add:
        if name in seen and name not in dup_in_add:
            dup_in_add.append(name)
        seen.add(name)
    if dup_in_add:
        raise ValueError(f'新增课程列表里有重复项：{dup_in_add}')

    dup_existing = [name for name in add if name in existing_names]
    if dup_existing:
        raise ValueError(f'新增课程与已有选项重名，不能重复新增：{dup_existing}')


def _批量修改所属课程(wjx, hide=None, add=None):
    """ 在已进入所属课程编辑态的前提下，批量隐藏/新增课程 """
    tab = wjx.tab
    hide = _按顺序去重(hide)
    add = _按顺序去重(add)
    payload = json.dumps({'hide': hide, 'add': add, 'title': 课程题目标题}, ensure_ascii=False)

    result = _运行js_json(tab, f"""
return JSON.stringify((() => {{
    const payload = {payload};
    const title = payload.title;
    const holders = window.questionHolder || [];
    let dn = null;
    for (let i = 0; i < holders.length; i++) {{
        const node = holders[i] && holders[i].dataNode;
        if (node && node._title === title) {{
            dn = node;
            break;
        }}
    }}
    if (!dn) throw new Error('未找到所属课程题目');
    if (!window.cur) throw new Error('当前未激活所属课程编辑态');

    const result = {{
        hidden_applied: [],
        hidden_skipped: [],
        hidden_missing: [],
        added_applied: []
    }};
    const indexByName = {{}};
    const selects = dn._select || [];
    for (let i = 1; i < selects.length; i++) {{
        const item = selects[i];
        if (item && item._item_title) indexByName[item._item_title.trim()] = i;
    }}

    for (const name of payload.hide || []) {{
        const idx = indexByName[name];
        if (!idx) {{
            result.hidden_missing.push(name);
            continue;
        }}
        const item = selects[idx];
        if (item._item_relation === '-1') {{
            result.hidden_skipped.push(name);
            continue;
        }}
        if (item._item_relation && item._item_relation !== '-1') {{
            throw new Error(`课程“${{name}}”已设置选项关联，不能直接自动隐藏`);
        }}
        item._item_relation = '-1';
        result.hidden_applied.push(name);
    }}

    const oldConfirm = window.confirm;
    window.confirm = () => true;
    try {{
        for (const name of payload.add || []) {{
            window.cur.addNewItem();
            const idx = dn._select.length - 1;
            const item = dn._select[idx];
            item._item_title = name;
            item._item_relation = '';
            item._item_value = item._item_value || String(idx);
            const titleEl = window.cur.option_radio[idx] && window.cur.option_radio[idx].get_item_title
                ? window.cur.option_radio[idx].get_item_title()
                : null;
            if (titleEl) {{
                titleEl.innerHTML = name;
                titleEl.textContent = name;
                titleEl.value = name;
            }}
            result.added_applied.push(name);
        }}
    }} finally {{
        window.confirm = oldConfirm;
    }}

    if (window.cur.updateItem) window.cur.updateItem();
    if (window.cur.updateReferQ) window.cur.updateReferQ();
    if (window.cur.setRandomText) window.cur.setRandomText();
    if (window.cur.checkItemTitle && !window.cur.checkItemTitle()) {{
        throw new Error('所属课程选项校验未通过');
    }}

    return result;
}})())
""")
    if not result:
        raise RuntimeError('批量修改所属课程失败')
    return result


def 完成问卷编辑(activity_id=默认问卷activity_id, wjx=None):
    """ 点击“完成编辑”并等待返回设计向导页 """
    wjx = 打开问卷编辑页(activity_id, wjx=wjx)
    tab = wjx.tab
    _等待元素(tab, 'css:#hrefFiQ', timeout=10, desc='完成编辑按钮').click()
    _等待直到(
        lambda: 'designstart.aspx' in tab.url and f'activity={activity_id}' in tab.url and tab.url,
        timeout=30,
        desc='完成编辑后返回设计向导页',
    )
    清理问卷星干扰弹窗(tab)
    return wjx


def 获取问卷课程清单(activity_id=默认问卷activity_id, 仅非隐藏=True, wjx=None):
    """ 只读获取当前问卷课程清单；不暂停问卷、不保存修改 """
    wjx = wjx or 自动登录问卷星()
    打开问卷设计页(activity_id, wjx=wjx)
    running_before = _获取问卷运行状态(wjx.tab) == 'running'
    paused_by_me = False

    try:
        if running_before:
            logger.info('问卷当前正在运行，读取课程清单前先临时暂停，读取完成后再恢复')
            暂停接收答卷(activity_id, wjx=wjx)
            paused_by_me = True

        打开问卷编辑页(activity_id, wjx=wjx)

        data = 读取所属课程选项(wjx, 仅非隐藏=False)
        打开问卷设计页(activity_id, wjx=wjx)
        return data['非隐藏课程'] if 仅非隐藏 else data
    finally:
        if paused_by_me:
            try:
                恢复接收答卷(activity_id, wjx=wjx)
            except Exception as e:
                logger.warning(f'读取课程清单后恢复问卷运行失败：{e}')


def 同步问卷课程选项(activity_id=默认问卷activity_id, hide=None, add=None, 恢复运行=None, wjx=None):
    """ 统一处理所属课程的检查、隐藏、新增与提交

    规则：
    1、hide 只隐藏，不删除；已隐藏项静默跳过
    2、add 严格追加到末尾；和现有任一项重名直接报错
    3、当 hide/add 都为空时，只做读取，不暂停、不保存
    """
    hide = _按顺序去重(hide)
    add = _按顺序去重(add)
    overlap = [name for name in add if name in set(hide)]
    if overlap:
        raise ValueError(f'同一课程不能同时出现在 hide 和 add 中：{overlap}')

    wjx = wjx or 自动登录问卷星()
    tab = wjx.tab
    清理多余问卷星标签页(tab)

    if not hide and not add:
        data = 获取问卷课程清单(activity_id, 仅非隐藏=False, wjx=wjx)
        return {
            'activity_id': activity_id,
            '只读': True,
            '已保存': False,
            '已恢复运行': False,
            '全部课程': data['全部课程'],
            '非隐藏课程': data['非隐藏课程'],
        }

    打开问卷设计页(activity_id, wjx=wjx)
    running_before = _获取问卷运行状态(tab) == 'running'
    should_resume = running_before if 恢复运行 is None else bool(恢复运行)
    paused_by_me = False
    saved = False
    resumed = False

    try:
        暂停接收答卷(activity_id, wjx=wjx)
        paused_by_me = True

        打开问卷编辑页(activity_id, wjx=wjx)
        进入所属课程编辑态(wjx)
        before = 读取所属课程选项(wjx, 仅非隐藏=False)

        existing_names = {row['名称'] for row in before['全部课程']}
        _校验新增课程(add, existing_names)

        changed = _批量修改所属课程(wjx, hide=hide, add=add)
        after = 读取所属课程选项(wjx, 仅非隐藏=False)

        完成问卷编辑(activity_id, wjx=wjx)
        saved = True

        if should_resume:
            恢复接收答卷(activity_id, wjx=wjx)
            resumed = True

        return {
            'activity_id': activity_id,
            '只读': False,
            '运行前状态': 'running' if running_before else 'paused',
            '已保存': True,
            '已恢复运行': resumed,
            'hidden_applied': changed['hidden_applied'],
            'hidden_skipped': changed['hidden_skipped'],
            'hidden_missing': changed['hidden_missing'],
            'added_applied': changed['added_applied'],
            '全部课程_前': before['全部课程'],
            '全部课程_后': after['全部课程'],
            '非隐藏课程_前': before['非隐藏课程'],
            '非隐藏课程_后': after['非隐藏课程'],
        }
    finally:
        if paused_by_me and not saved:
            try:
                打开问卷设计页(activity_id, wjx=wjx)
            except Exception:
                pass
            if should_resume:
                try:
                    恢复接收答卷(activity_id, wjx=wjx)
                except Exception as e:
                    logger.warning(f'问卷课程选项流程异常后恢复运行失败：{e}')


def 获得问卷星数据(
    all_pages=False,
    num_of_page=100,
    activity_id=默认问卷activity_id,
    *,
    login_username=None,
    password=None,
):
    """
    访问问卷星网页并获取数据
    """
    # 1 访问网页拿到数据
    # 这里使用 WjxWeb，它会自动调用 login() 进行登录
    wjx = WjxWeb(
        获取问卷结果汇总页链接(activity_id),
        login_username=login_username,
        password=password,
    )
    清理问卷星干扰弹窗(wjx.tab)
    wjx.set_num_of_page(num_of_page)
    time.sleep(3)
    清理问卷星干扰弹窗(wjx.tab)
    df = wjx.get_df(all_pages)
    wjx.tab.close()

    # 2 针对考勤情况定制数据格式功能
    return 标准化问卷星记录(df)


def 获取问卷星增量记录(
    exist_max_id=0,
    *,
    activity_id=默认问卷activity_id,
    num_of_page=100,
    login_username=None,
    password=None,
    fetcher=None,
):
    """
    获取增量问卷记录，不依赖 WPS。

    返回:
        {
            'df': DataFrame,
            'exist_max_id': int,
            'latest_max_id': int,
            'recent_count': int,
            'fetched_count': int,
            'incremental_count': int,
            'used_all_pages': bool,
        }
    """

    exist_max_id = _转整数(exist_max_id, 0)

    if fetcher is None:
        def fetcher(*, all_pages=False):
            return 获得问卷星数据(
                all_pages=all_pages,
                num_of_page=num_of_page,
                activity_id=activity_id,
                login_username=login_username,
                password=password,
            )

    recent_df = 标准化问卷星记录(fetcher(all_pages=False))
    source_df = recent_df
    used_all_pages = False

    if len(recent_df) and int(recent_df['序号'].min()) > exist_max_id:
        source_df = 标准化问卷星记录(fetcher(all_pages=True))
        used_all_pages = True

    incremental_df = 筛选问卷星增量记录(source_df, exist_max_id=exist_max_id)
    latest_max_id = int(source_df['序号'].max()) if len(source_df) else exist_max_id

    return {
        'df': incremental_df,
        'exist_max_id': exist_max_id,
        'latest_max_id': latest_max_id,
        'recent_count': len(recent_df),
        'fetched_count': len(source_df),
        'incremental_count': len(incremental_df),
        'used_all_pages': used_all_pages,
    }


def 更新问卷数据(
    exist_max_id=None,
    *,
    activity_id=默认问卷activity_id,
    写回WPS=True,
    login_username=None,
    password=None,
):
    """
    更新 WPS 考勤表中的问卷数据
    """
    # 1 获取已有数据的最大序号
    if exist_max_id is None:
        if not 写回WPS:
            raise ValueError('不写回 WPS 时，必须显式提供 exist_max_id')

        from .db import KqBook

        wb = KqBook()
        exist_max_id = wb.run_func('获取问卷数据最大编号')
        logger.info(f'已存在最大序号：{exist_max_id}')

    # 2 在 Python 侧获得增量数据
    result = 获取问卷星增量记录(
        exist_max_id=exist_max_id,
        activity_id=activity_id,
        login_username=login_username,
        password=password,
    )
    df = result['df']
    if len(df) == 0 or not 写回WPS:
        return result

    df = df.astype(object).where(df.notna(), None)  # nan值改为none
    data = df.to_dict(orient='split')
    data_json = json.dumps(data, ensure_ascii=False)

    # 3 写回 WPS 并后续处理
    from .db import KqBook

    wb = KqBook()
    wb.run_func('_更新问卷数据', data_json)
    return result


def _读取CodeYun问卷提醒数据(api_url=CodeYun问卷数据接口, page_size=CodeYun问卷数据页大小):
    """从 CodeYun 接口分页读取问卷提醒所需字段。"""
    rows = []
    page = 1
    total = None

    while True:
        resp = requests.get(
            api_url,
            params={'page': page, 'page_size': page_size},
            timeout=(5, 20),
        )
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get('items') or []
        total = _转整数(payload.get('total'), total or 0)

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
        if total and len(rows) >= total:
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


def 分析问卷星滞留记录(df: pd.DataFrame, *, header=问卷滞留提醒头部):
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
    groups = 分析问卷星滞留记录(df)

    for group_name, dst in 问卷滞留提醒发送目标.items():
        payload = groups[group_name]
        if payload['items']:
            wechat_lock_send(dst, payload['message'])

    return groups


def 匹配问卷星用户候选(df: pd.DataFrame, 填写学号, 填写姓名):
    """在任意候选表中执行问卷用户匹配，不依赖 WPS 链接。"""

    from pyxllib.text.levenshtein import get_levenshtein_similar

    if len(df) == 0:
        return {
            'message': '没有可匹配的候选数据',
            'similarity': 0,
            'student_id': '',
            'display_name': '',
            'user_id': '',
            'row_index': '',
            'top_candidates': [],
        }

    df2 = df.copy()
    for col in ['学号', '姓名', '微信昵称', '用户ID']:
        if col not in df2.columns:
            df2[col] = ''
    _fill_text_na(df2, '')

    填写学号文本 = _转文本(填写学号)
    填写姓名文本 = _转文本(填写姓名)

    sim = []
    for idx, row in df2.iterrows():
        row_student_id = _转文本(row['学号'])
        row_index = _转文本(idx)

        if row_student_id and row_student_id == 填写学号文本:
            a = 1
        elif row_index and row_index == 填写学号文本:
            a = 0.7
        else:
            a = 0.5

        b = max(
            get_levenshtein_similar(填写姓名文本, _转文本(row['姓名'])),
            0.9 * get_levenshtein_similar(填写姓名文本, _转文本(row['微信昵称'])),
        )
        sim.append(round(a * b, 4))

    df2['sim'] = sim
    df2.sort_values('sim', ascending=False, inplace=True)
    row = df2.iloc[0]

    姓名相似度 = get_levenshtein_similar(_转文本(row['姓名']), 填写姓名文本)
    昵称相似度 = get_levenshtein_similar(_转文本(row['微信昵称']), 填写姓名文本)
    display_name = row['姓名'] if 姓名相似度 > 昵称相似度 else row['微信昵称']

    top_candidates = []
    for idx, candidate in df2.head(5).iterrows():
        top_candidates.append({
            'row_index': idx,
            'student_id': _转整数或原值(candidate['学号']),
            'name': _转文本(candidate['姓名']),
            'nickname': _转文本(candidate['微信昵称']),
            'user_id': _转文本(candidate['用户ID']),
            'similarity': float(candidate['sim']),
        })

    return {
        'message': '',
        'similarity': float(row['sim']),
        'student_id': _转整数或原值(row['学号']),
        'display_name': _转文本(display_name),
        'user_id': _转文本(row['用户ID']),
        'row_index': row.name,
        'top_candidates': top_candidates,
    }


def 提取修正需求课次标签(修正需求=''):
    from pyxllib.text.convert import chinese2digits

    m = re.search(r'(\d+)(课|堂|次|天)', chinese2digits(_转文本(修正需求)))
    if not m:
        return ''
    return f'第{int(m.group(1)):02}课'


def 问卷星用户数据匹配(考勤表链接, 填写学号, 填写姓名, 修正需求=''):
    """
    在考勤表中匹配问卷星填写的用户数据
    """
    # 1 分析链接
    if not 考勤表链接:
        return ['没有考勤表链接呢', '', '', '', '', '', '']

    if 'kdocs.cn' not in 考勤表链接:
        return ['非wps表格目前不支持匹配', '', '', '', '', '', '']

    表格id = 考勤表链接.split('/')[-1]

    # 2 获得考勤表数据
    from pyxllib.ext.wpsapi import WpsOnlineBook

    wb = WpsOnlineBook(表格id)
    df = wb.get_df('考勤表', ['学号', '姓名', '微信昵称', '用户ID'], 4)
    df.index += 4

    # 3 计算匹配结果
    matched = 匹配问卷星用户候选(df, 填写学号, 填写姓名)
    if matched['message']:
        return {'data': [matched['message'], '', '', '', '', '', '']}

    # 4 尝试找对应课次，及数据
    对应课次, 当前数据 = '', ''
    tag = 提取修正需求课次标签(修正需求)
    if tag:
        jscode = f"""const ur = Sheets('考勤表').UsedRange
const cel2 = ur.Rows('2:2').Find('{tag}')
if (cel2) {{
    const row = findRow('{matched['user_id']}', ur)
    const cel3 = ur.Cells(row, cel2.Column)
    return [[cel2.Text, cel2.Hyperlinks.Item(1).Address], [cel3.Text, cel3.Interior.Color]]
}}"""
        vals = wb.run_airscript(jscode)
        if vals:
            对应课次, 当前数据 = vals

    return {
        'data': [
            '',
            matched['similarity'],
            matched['student_id'],
            matched['display_name'],
            matched['user_id'],
            对应课次,
            当前数据,
        ]
    }


def 自动登录问卷星():
    """
    自动登录问卷星并保持会话
    """
    logger.info("正在尝试自动登录问卷星...")
    logger.info("如果浏览器停留在登录页，请手动完成验证码或额外验证。")
    wjx = WjxWeb(WjxWeb.LOGIN_URL)
    清理问卷星干扰弹窗(wjx.tab)
    logger.info("登录完成！")
    return wjx


if __name__ == '__main__':
    import fire

    fire.Fire()
