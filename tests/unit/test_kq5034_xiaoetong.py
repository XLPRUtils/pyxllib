from pathlib import Path

import pytest

from kq5034.xiaoetong import XiaoetongWeb


class _FakeButton:
    def click(self, *_args, **_kwargs):
        return None

    def input(self, *_args, **_kwargs):
        return None


class _FakeLabel:
    def __call__(self, *_args, **_kwargs):
        return _FakeButton()


class _FakeTab:
    url = 'https://admin.xiaoe-tech.com/t/live_management#/userOperation?id=123&tabName=UserManage'

    def get(self, *_args, **_kwargs):
        return None

    def __call__(self, *_args, **_kwargs):
        return _FakeButton()

    def eles(self, locator):
        if locator == 'tag:label@@class=el-checkbox':
            return [_FakeLabel(), _FakeLabel()]
        return []

    def wait(self, *_args, **_kwargs):
        return None


def test_assert_shop_rejects_wrong_visible_shop(monkeypatch):
    web = XiaoetongWeb.__new__(XiaoetongWeb)
    web.tab = _FakeTab()
    web.cur_shop_id = 2
    monkeypatch.setattr(web, '_当前店铺名', lambda timeout=0.8: '5034山中薪')

    with pytest.raises(RuntimeError, match='target=宗门学府.*current_shop=5034山中薪'):
        web.assert_shop(2)


def test_switch_shop_navigates_with_timeout_and_confirms_visible_shop(monkeypatch):
    class FakeSwitchTab(_FakeTab):
        def __init__(self):
            self.url = 'https://admin.xiaoe-tech.com/t/merchant/index'
            self.get_calls = []

        def get(self, url, **kwargs):
            self.get_calls.append((url, kwargs))
            self.url = url
            return True

        def eles(self, _locator):
            return []

    tab = FakeSwitchTab()
    web = XiaoetongWeb.__new__(XiaoetongWeb)
    web.tab = tab
    web.cur_shop_id = 1
    visible_shop = {'name': '5034山中薪'}
    monkeypatch.setattr(web, '_重连当前标签页', lambda: tab)
    monkeypatch.setattr(web, '_当前店铺名', lambda timeout=0.8: visible_shop['name'])

    def click_shop(shop):
        visible_shop['name'] = shop
        return True

    monkeypatch.setattr(web, '_在选店页点击店铺', click_shop)

    result = web.switch_shop(2)

    assert result is tab
    assert web.cur_shop_id == 2
    assert tab.get_calls == [(
        XiaoetongWeb._choose_shop_url,
        {
            'retry': 1,
            'interval': 1,
            'timeout': XiaoetongWeb._switch_shop_page_timeout_seconds,
        },
    )]


def test_iter_export_user_list_falls_back_when_download_center_task_name_changes(monkeypatch):
    web = XiaoetongWeb.__new__(XiaoetongWeb)
    web.tab = _FakeTab()

    def fake_download_records(keywords=None):
        names = ['旧导出A'] if keywords else ['旧导出A', '旧导出B']
        return [
            {
                'name': name,
                'status': '已完成',
                'action_text': '下载',
                'can_download': True,
                'apply_time': '',
            }
            for name in names
        ]

    monkeypatch.setattr(web, '_列出下载中心任务记录', fake_download_records)

    calls = []

    def fake_iter_download_last_file(match_keywords=None, exclude_task_names=None, **kwargs):
        calls.append((match_keywords, list(exclude_task_names or []), kwargs.get('max_wait_seconds')))
        if len(calls) == 1:
            if False:
                yield None
            raise RuntimeError("下载中心等待超时，未找到匹配任务：keywords=['用户列表导出'] url=test")
        if False:
            yield None
        return Path('C:/tmp/用户列表导出-最新.csv')

    monkeypatch.setattr(web, 'iter_download_last_file', fake_iter_download_last_file)

    result = XiaoetongWeb._取生成器返回值(web.iter_export_user_list())

    assert result == Path('C:/tmp/用户列表导出-最新.csv')
    assert calls == [
        (list(XiaoetongWeb.用户列表导出关键词), ['旧导出A'], 20 * 60),
        (None, ['旧导出A', '旧导出B'], 10 * 60),
    ]


def test_export_lesson_data_caps_single_lesson_wait(monkeypatch):
    web = XiaoetongWeb.__new__(XiaoetongWeb)
    web.tab = _FakeTab()
    web.exist_files = set()

    class _FakeTempTab:
        def __enter__(self):
            return web.tab

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(web, '_make_runtime_cache_key', lambda *args, **kwargs: 'lesson-cache-key')
    monkeypatch.setattr(web, '_restore_runtime_cached_file', lambda *args, **kwargs: XiaoetongWeb._CACHE_MISS)
    monkeypatch.setattr(web, '_store_runtime_cached_file', lambda _key, file: file)
    monkeypatch.setattr(web, '临时工作标签页', lambda *args, **kwargs: _FakeTempTab())
    monkeypatch.setattr(web, '_等待直播课用户导出按钮', lambda *args, **kwargs: _FakeButton())
    monkeypatch.setattr(web, '_直播课用户列表为空', lambda *args, **kwargs: False)

    calls = []

    def fake_download_last_file(*args, **kwargs):
        calls.append((args, kwargs))
        return Path('C:/tmp/lesson-export.csv')

    monkeypatch.setattr(web, 'download_last_file', fake_download_last_file)

    result = web.export_lesson_data({'lesson_id2': '28969108', 'lesson_name': '测试课次'})

    assert result == Path('C:/tmp/lesson-export.csv')
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ()
    assert kwargs['exclude_task_names'] == []
    assert 1 <= kwargs['max_wait_seconds'] <= XiaoetongWeb._resource_download_center_wait_seconds
    assert 1 <= kwargs['download_wait_seconds'] <= XiaoetongWeb._resource_download_wait_seconds
    assert XiaoetongWeb._lesson_resource_export_timeout_seconds == 5 * 60


def test_export_clockin_data_prefers_page_name_and_falls_back_to_new_task(monkeypatch):
    web = XiaoetongWeb.__new__(XiaoetongWeb)

    class _ClockinElement:
        states = type('States', (), {'is_clickable': True})()
        scroll = type('Scroll', (), {'to_see': lambda self: None})()

        class _Wait:
            def __init__(self, element):
                self.element = element

            def clickable(self):
                return self.element

        def __init__(self):
            self.wait = self._Wait(self)

        def __call__(self, *_args, **_kwargs):
            return self

        def click(self, *_args, **_kwargs):
            return None

    class _ClockinTab:
        url = 'https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?component_name=clock_task_data'
        action_type = staticmethod(lambda *_args, **_kwargs: None)

        def __call__(self, *_args, **_kwargs):
            return _ClockinElement()

        def get2(self, *_args, **_kwargs):
            return self

        def run_js(self, *_args, **_kwargs):
            return '13期一阶忏悔门打卡\n任务数据'

        def wait(self, *_args, **_kwargs):
            return None

    web.tab = _ClockinTab()
    monkeypatch.setattr(web, '_make_runtime_cache_key', lambda *args, **kwargs: 'clockin-cache-key')
    monkeypatch.setattr(web, '_restore_runtime_cached_file', lambda *args, **kwargs: XiaoetongWeb._CACHE_MISS)
    monkeypatch.setattr(web, '_store_runtime_cached_file', lambda _key, file: file)
    monkeypatch.setattr(
        web,
        '_查找本地下载文件',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('打卡导出不能按跨课程重复的文件名复用本地文件')
        ),
    )
    monkeypatch.setattr(web, '_等待禅宗打卡导出按钮', lambda *args, **kwargs: (_ClockinElement(), _ClockinElement()))
    monkeypatch.setattr(web, '_提取已生成下载文件名', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(web, '_列出下载中心任务名', lambda keywords=None: ['旧任务'])

    calls = []

    def fake_download_last_file(match_keywords=None, exclude_task_names=None, **kwargs):
        calls.append((match_keywords, list(exclude_task_names or []), kwargs.get('max_wait_seconds')))
        return Path('C:/tmp/13期一阶忏悔门打卡.csv')

    monkeypatch.setattr(web, 'download_last_file', fake_download_last_file)

    result = web.export_clockin_data(
        web.tab.url,
        expected_download_name='d260712禅宗13期一阶-共修打卡-忏悔门',
        start_date='2026-07-11',
        end_date='2026-09-11',
        exclude_existing_download_tasks=False,
    )

    assert result == Path('C:/tmp/13期一阶忏悔门打卡.csv')
    assert len(calls) == 1
    assert calls[0][:2] == (None, ['旧任务'])
    assert 1 <= calls[0][2] <= XiaoetongWeb._resource_download_center_wait_seconds


def test_export_diary_clockin_ignores_local_empty_text_when_export_button_exists(monkeypatch):
    web = XiaoetongWeb.__new__(XiaoetongWeb)

    class _RecordingButton:
        def __init__(self):
            self.clicked = False

        def click(self, *_args, **_kwargs):
            self.clicked = True
            return None

    class _Dialog:
        def __call__(self, *_args, **_kwargs):
            return _FakeButton()

    export_button = _RecordingButton()

    class _DiaryTab:
        url = 'https://admin.xiaoe-tech.com/t/clock_admin/index#/punchDetail/diaryList?activity_id=ac_center'

        def __call__(self, locator, *_args, **_kwargs):
            if locator == 'tag:button@@text():导出动态':
                return export_button
            if locator == 'tag:div@@role=dialog@@aria-label=导出数据':
                return _Dialog()
            return _FakeButton()

        def get2(self, *_args, **_kwargs):
            return self

        def run_js(self, *_args, **_kwargs):
            return '第48届觉观技术公益网课【中心教室】\n#【打卡】中心教室-21\n暂无内容'

        def wait(self, *_args, **_kwargs):
            return None

    web.tab = _DiaryTab()
    monkeypatch.setattr(web, '_make_runtime_cache_key', lambda *args, **kwargs: 'diary-clockin-cache-key')
    monkeypatch.setattr(web, '_restore_runtime_cached_file', lambda *args, **kwargs: XiaoetongWeb._CACHE_MISS)
    monkeypatch.setattr(web, '_store_runtime_cached_file', lambda _key, file: file)
    monkeypatch.setattr(web, '_查找本地下载文件', lambda *args, **kwargs: None)
    monkeypatch.setattr(web, '_列出下载中心任务名', lambda keywords=None: ['旧导出'] if keywords else ['旧任务'])

    calls = []

    def fake_download_last_file(match_keywords=None, exclude_task_names=None, **kwargs):
        calls.append((match_keywords, list(exclude_task_names or []), kwargs.get('max_wait_seconds')))
        return Path('C:/tmp/第48届觉观-打卡数.csv')

    monkeypatch.setattr(web, 'download_last_file', fake_download_last_file)

    result = web.export_clockin_data(
        web.tab.url,
        expected_download_name='第48届觉观-打卡数',
    )

    assert export_button.clicked
    assert result == Path('C:/tmp/第48届觉观-打卡数.csv')
    assert len(calls) == 1
    assert calls[0][:2] == (None, ['旧任务'])
    assert 1 <= calls[0][2] <= XiaoetongWeb._resource_download_center_wait_seconds
    assert XiaoetongWeb._clockin_resource_export_timeout_seconds == 10 * 60


def test_export_diary_clockin_confirmation_wait_is_bounded(monkeypatch):
    web = XiaoetongWeb.__new__(XiaoetongWeb)

    class _MissingConfirmDialog:
        def __call__(self, *_args, **_kwargs):
            raise RuntimeError('确认按钮未出现')

    class _DiaryTab:
        url = 'https://admin.xiaoe-tech.com/t/clock_admin/index#/punchDetail/diaryList?activity_id=timeout'

        def __call__(self, locator, *_args, **_kwargs):
            if locator == 'tag:button@@text():导出动态':
                return _FakeButton()
            if locator == 'tag:div@@role=dialog@@aria-label=导出数据':
                return _MissingConfirmDialog()
            return _FakeButton()

        def get2(self, *_args, **_kwargs):
            return self

        def run_js(self, *_args, **_kwargs):
            return '打卡动态'

        def wait(self, *_args, **_kwargs):
            return None

    web.tab = _DiaryTab()
    monkeypatch.setattr(web, '_make_runtime_cache_key', lambda *args, **kwargs: 'clockin-timeout-key')
    monkeypatch.setattr(web, '_restore_runtime_cached_file', lambda *args, **kwargs: XiaoetongWeb._CACHE_MISS)
    monkeypatch.setattr(web, '_查找本地下载文件', lambda *args, **kwargs: None)
    monkeypatch.setattr(web, '_列出下载中心任务名', lambda *args, **kwargs: [])

    with pytest.raises(RuntimeError, match='日历打卡导出确认等待超时'):
        web.export_clockin_data(web.tab.url)


def test_search_lesson_links_closes_detail_tab_before_yield(monkeypatch):
    web = XiaoetongWeb.__new__(XiaoetongWeb)

    class _DetailTab:
        url = 'https://admin.xiaoe-tech.com/t/live#/detail?id=lesson_123&tab=playbackSettings'

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    detail_tab = _DetailTab()

    class _Click:
        def for_new_tab(self, **_kwargs):
            return detail_tab

    class _Button:
        click = _Click()

    class _Title:
        text = '8月梵呗初阶第1课'

    class _Row:
        def __call__(self, locator):
            if locator == '.title title-hover ss-popover__reference':
                return _Title()
            if locator == 't:button@@text():管理':
                return _Button()
            raise AssertionError(f'意外定位器：{locator}')

    class _Tbody:
        def __call__(self, locator):
            assert locator == '没有相应的数据'
            return False

        def eles(self, locator):
            assert locator == 'tag:tr'
            return [_Row()]

    class _ListTab:
        def get(self, _url):
            return None

        def __call__(self, locator):
            assert locator == 'tag:tbody'
            return _Tbody()

    monkeypatch.setattr(web, 'switch_shop', lambda: _ListTab())

    row = next(web.search_lesson_links('8月梵呗初阶', maxn=1))

    assert detail_tab.closed is True
    assert row == {
        'lesson_name': '8月梵呗初阶第1课',
        'lesson_id': 'lesson_123',
        'lesson_id2': 'lesson_123',
    }
