import threading

import pytest

from pyxllib.ext.drissionlib import DpWebBase


class _Browser:
    def __init__(self, *, block=False, error=None):
        self.block = block
        self.error = error
        self.release = threading.Event()

    def new_tab(self, url=None):
        if self.block:
            self.release.wait(5)
        if self.error:
            raise self.error
        return f'tab:{url}'


def _web(browser, timeout):
    web = DpWebBase.__new__(DpWebBase)
    web.browser = browser
    web.new_tab_timeout = timeout
    return web


def test_new_tab_without_timeout_keeps_direct_call():
    assert _web(_Browser(), None)._new_tab('https://a') == 'tab:https://a'


def test_new_tab_timeout_fails_instead_of_hanging():
    browser = _Browser(block=True)
    try:
        with pytest.raises(TimeoutError, match='新建标签超时'):
            _web(browser, 0.2)._new_tab('https://a')
    finally:
        browser.release.set()


def test_new_tab_timeout_reraises_worker_error():
    with pytest.raises(ConnectionError, match='gone'):
        _web(_Browser(error=ConnectionError('gone')), 1)._new_tab('https://a')
    assert _web(_Browser(), 1)._new_tab('https://b') == 'tab:https://b'
