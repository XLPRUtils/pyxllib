from pathlib import Path

from kq5034.xiaoetong import XiaoetongWeb


class _FakeButton:
    def click(self):
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


def test_iter_export_user_list_falls_back_when_download_center_task_name_changes(monkeypatch):
    web = XiaoetongWeb.__new__(XiaoetongWeb)
    web.tab = _FakeTab()

    monkeypatch.setattr(web, '_列出下载中心任务名', lambda keywords=None: ['旧导出A'] if keywords else ['旧导出A', '旧导出B'])
    monkeypatch.setattr(web, '_列出下载中心任务记录', lambda keywords=None: [])

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


def test_export_lesson_data_uses_longer_download_wait_for_large_exports(monkeypatch):
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
    assert calls == [(
        (),
        {
            'exclude_task_names': [],
            'max_wait_seconds': 20 * 60,
            'download_wait_seconds': XiaoetongWeb._lesson_export_download_wait_seconds,
        },
    )]
