import sys
import types

import pandas as pd

from kq5034.wjx_automation import (
    _读取CodeYun问卷提醒数据,
    分析问卷星滞留记录,
    匹配问卷星用户候选,
    提取修正需求课次标签,
    提醒问卷数据,
    获取问卷星增量记录,
    标准化问卷星记录,
)


def test_标准化问卷星记录保持固定字段顺序():
    df = pd.DataFrame(
        {
            '序号(自动)': ['1', '2'],
            '1、所属课程[单选题]': ['第45届觉观', None],
            '3、姓名': ['张三', None],
        }
    )

    result = 标准化问卷星记录(df)

    assert result['序号'].tolist() == [1, 2]
    assert result['1、所属课程'].tolist() == ['第45届觉观', '']
    assert result['3、姓名'].tolist() == ['张三', '']
    assert list(result.columns) == [
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


def test_获取问卷星增量记录默认只抓最近页():
    calls = []
    recent_df = pd.DataFrame({'序号': [8, 9, 10, 11], '1、所属课程': ['A', 'B', 'C', 'D']})

    def fake_fetcher(*, all_pages=False):
        calls.append(all_pages)
        return recent_df

    result = 获取问卷星增量记录(9, fetcher=fake_fetcher)

    assert calls == [False]
    assert result['used_all_pages'] is False
    assert result['incremental_count'] == 2
    assert result['df']['序号'].tolist() == [10, 11]


def test_获取问卷星增量记录在最近页不够时回退全量():
    calls = []
    recent_df = pd.DataFrame({'序号': [20, 21], '1、所属课程': ['A', 'B']})
    all_df = pd.DataFrame({'序号': [9, 10, 11, 12], '1、所属课程': ['A', 'B', 'C', 'D']})

    def fake_fetcher(*, all_pages=False):
        calls.append(all_pages)
        return all_df if all_pages else recent_df

    result = 获取问卷星增量记录(10, fetcher=fake_fetcher)

    assert calls == [False, True]
    assert result['used_all_pages'] is True
    assert result['fetched_count'] == 4
    assert result['df']['序号'].tolist() == [11, 12]


def test_分析问卷星滞留记录支持分组和未分组():
    df = pd.DataFrame(
        {
            '序号': [2, 1, 4, 3, 5],
            '1、所属课程': [
                '20260408第39届念住',
                '20260401第45届觉观',
                '20260422未知课程',
                '20260415梵呗初阶',
                '20260429禅宗5阶',
            ],
            '处理状态': ['', '', '', '', '已处理'],
        }
    )

    result = 分析问卷星滞留记录(df)

    assert result['中台']['items'] == ['20260401第45届觉观：1', '20260408第39届念住：2']
    assert '1. 20260401第45届觉观：1' in result['中台']['message']
    assert result['梵呗']['items'] == ['20260415梵呗初阶：3']
    assert result['未分组']['items'] == ['20260422未知课程：4']
    assert result['禅宗']['items'] == []


def test_读取CodeYun问卷提醒数据支持分页(monkeypatch):
    payloads = {
        1: {
            'total': 3,
            'items': [
                {'seq': 12, 'course_name': '20260415梵呗初阶', 'process_status': ''},
                {'seq': 11, 'course_name': '20260408第39届念住', 'process_status': ''},
            ],
        },
        2: {
            'total': 3,
            'items': [
                {'seq': 10, 'course_name': '20260401第45届觉观', 'process_status': ''},
            ],
        },
    }

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, *, params=None, timeout=None):
        assert url == 'https://code4101.com/api/attendance/wjx-data'
        assert timeout == (5, 20)
        return FakeResponse(payloads[params['page']])

    monkeypatch.setattr('kq5034.wjx_automation.requests.get', fake_get)

    df = _读取CodeYun问卷提醒数据(
        api_url='https://code4101.com/api/attendance/wjx-data',
        page_size=2,
    )

    assert df['序号'].tolist() == [12, 11, 10]
    assert df['1、所属课程'].tolist() == ['20260415梵呗初阶', '20260408第39届念住', '20260401第45届觉观']
    assert df['处理状态'].tolist() == ['', '', '']


def test_提醒问卷数据支持传入CodeYun接口(monkeypatch):
    api_url = 'http://192.168.31.63:5173/api/attendance/wjx-data'

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                'total': 1,
                'items': [
                    {'seq': 12, 'course_name': '20260415梵呗初阶', 'process_status': ''},
                ],
            }

    def fake_get(url, *, params=None, timeout=None):
        assert url == api_url
        assert params == {'page': 1, 'page_size': 100}
        assert timeout == (5, 20)
        return FakeResponse()

    sent = []
    fake_common = types.ModuleType('kq5034.common')
    fake_common.wechat_lock_send = lambda dst, message: sent.append((dst, message))
    monkeypatch.setitem(sys.modules, 'kq5034.common', fake_common)
    monkeypatch.setattr('kq5034.wjx_automation.requests.get', fake_get)

    result = 提醒问卷数据(api_url=api_url)

    assert result['梵呗']['items'] == ['20260415梵呗初阶：12']
    assert sent == [('本体音艺考勤班委群', result['梵呗']['message'])]


def test_匹配问卷星用户候选复用相似度算法():
    df = pd.DataFrame(
        [
            {'学号': 1, '姓名': '张三', '微信昵称': '三三', '用户ID': 'u1'},
            {'学号': 2, '姓名': '李四', '微信昵称': '阿四', '用户ID': 'u2'},
        ],
        index=[4, 5],
    )

    result = 匹配问卷星用户候选(df, '1', '张三')

    assert result['message'] == ''
    assert result['user_id'] == 'u1'
    assert result['student_id'] == 1
    assert result['display_name'] == '张三'
    assert result['top_candidates'][0]['user_id'] == 'u1'
    assert result['similarity'] > result['top_candidates'][1]['similarity']


def test_提取修正需求课次标签():
    assert 提取修正需求课次标签('第3课没有记录') == '第03课'
    assert 提取修正需求课次标签('我补第十二次') == '第12课'
    assert 提取修正需求课次标签('只说了没打卡') == ''
