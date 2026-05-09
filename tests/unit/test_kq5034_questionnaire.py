import datetime
import sys
import types

import pandas as pd

from kq5034.questionnaire import (
    _读取CodeYun问卷提醒数据,
    分析问卷滞留记录,
    提醒问卷数据,
)


def test_分析问卷滞留记录支持分组和未分组():
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

    result = 分析问卷滞留记录(df)

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
        assert params['process_status'] == '__empty__'
        assert timeout == (5, 20)
        return FakeResponse(payloads[params['page']])

    monkeypatch.setattr('kq5034.questionnaire.requests.get', fake_get)

    df = _读取CodeYun问卷提醒数据(
        api_url='https://code4101.com/api/attendance/wjx-data',
        page_size=2,
    )

    assert df['序号'].tolist() == [12, 11, 10]
    assert df['1、所属课程'].tolist() == ['20260415梵呗初阶', '20260408第39届念住', '20260401第45届觉观']
    assert df['处理状态'].tolist() == ['', '', '']


def test_提醒问卷数据支持传入CodeYun接口(monkeypatch, tmp_path):
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
        assert params == {'page': 1, 'page_size': 100, 'process_status': '__empty__'}
        assert timeout == (5, 20)
        return FakeResponse()

    sent = []
    fake_common = types.ModuleType('kq5034.common')
    fake_common.wechat_lock_send = lambda dst, message: sent.append((dst, message))
    monkeypatch.setitem(sys.modules, 'kq5034.common', fake_common)
    monkeypatch.setattr('kq5034.questionnaire.requests.get', fake_get)

    result = 提醒问卷数据(api_url=api_url, state_path=tmp_path / 'reminder_state.json')

    assert result['梵呗']['items'] == ['20260415梵呗初阶：12']
    assert sent == [('本体音艺考勤班委群', result['梵呗']['message'])]


def test_提醒问卷数据禅宗同清单非周日不重复发送(monkeypatch, tmp_path):
    api_url = 'http://192.168.31.63:5173/api/attendance/wjx-data'

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                'total': 2,
                'items': [
                    {'seq': 652, 'course_name': '202605禅宗五阶', 'process_status': ''},
                    {'seq': 653, 'course_name': '202605禅宗五阶', 'process_status': ''},
                ],
            }

    monkeypatch.setattr('kq5034.questionnaire.requests.get', lambda *args, **kwargs: FakeResponse())
    sent = []
    fake_common = types.ModuleType('kq5034.common')
    fake_common.wechat_lock_send = lambda dst, message: sent.append((dst, message))
    monkeypatch.setitem(sys.modules, 'kq5034.common', fake_common)

    state_path = tmp_path / 'reminder_state.json'
    提醒问卷数据(api_url=api_url, state_path=state_path, today=datetime.datetime(2026, 5, 8, 9))
    提醒问卷数据(api_url=api_url, state_path=state_path, today=datetime.datetime(2026, 5, 9, 9))

    assert len(sent) == 1
    assert sent[0][0] == '禅宗修道考勤管理'


def test_提醒问卷数据禅宗周日同清单仍发送(monkeypatch, tmp_path):
    api_url = 'http://192.168.31.63:5173/api/attendance/wjx-data'

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                'total': 2,
                'items': [
                    {'seq': 652, 'course_name': '202605禅宗五阶', 'process_status': ''},
                    {'seq': 653, 'course_name': '202605禅宗五阶', 'process_status': ''},
                ],
            }

    monkeypatch.setattr('kq5034.questionnaire.requests.get', lambda *args, **kwargs: FakeResponse())
    sent = []
    fake_common = types.ModuleType('kq5034.common')
    fake_common.wechat_lock_send = lambda dst, message: sent.append((dst, message))
    monkeypatch.setitem(sys.modules, 'kq5034.common', fake_common)

    state_path = tmp_path / 'reminder_state.json'
    提醒问卷数据(api_url=api_url, state_path=state_path, today=datetime.date(2026, 5, 9))
    提醒问卷数据(api_url=api_url, state_path=state_path, today=datetime.date(2026, 5, 10))

    assert len(sent) == 2
    assert [x[0] for x in sent] == ['禅宗修道考勤管理', '禅宗修道考勤管理']
