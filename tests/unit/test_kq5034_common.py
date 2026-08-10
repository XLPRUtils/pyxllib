from kq5034 import common


class _FakeBrowser:
    def __init__(self):
        self.closed = []

    def _run_cdp(self, method, **kwargs):
        if method == 'Target.closeTarget':
            self.closed.append(kwargs['targetId'])
            return {}
        assert method == 'Target.getTargets'
        pages = [
            {'type': 'page', 'targetId': 'keep', 'url': 'https://admin.xiaoe-tech.com/t/merchant/index'},
            {'type': 'page', 'targetId': 'duplicate', 'url': 'https://admin.xiaoe-tech.com/t/account/choose'},
            {'type': 'page', 'targetId': 'other', 'url': 'https://www.bilibili.com/'},
        ]
        return {'targetInfos': [item for item in pages if item['targetId'] not in self.closed]}


def test_cleanup_attaches_default_browser_and_verifies_duplicate_closed(monkeypatch):
    browser = _FakeBrowser()
    monkeypatch.setattr(common, 'Chromium', lambda: browser)

    result = common.尝试关闭重复页面(
        reason='test cleanup',
        keep_tab_ids=['keep'],
        timeout=0.2,
    )

    assert result is True
    assert browser.closed == ['duplicate']
