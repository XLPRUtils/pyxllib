from contextlib import nullcontext
from pathlib import Path

import pytest

from kq5034 import weipay
from kq5034.weipay import Weipay, _weipay_login_max_attempts


def test_weipay_login_retry_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv('KQ_WEIPAY_LOGIN_ALLOW_RETRY', raising=False)
    monkeypatch.setenv('KQ_WEIPAY_LOGIN_MAX_ATTEMPTS', '5')

    assert _weipay_login_max_attempts() == 1


def test_weipay_login_retry_requires_explicit_allow_flag(monkeypatch):
    monkeypatch.setenv('KQ_WEIPAY_LOGIN_ALLOW_RETRY', '1')
    monkeypatch.setenv('KQ_WEIPAY_LOGIN_MAX_ATTEMPTS', '3')

    assert _weipay_login_max_attempts() == 3


class _FakeQrImage:
    def save(self, *_args, **_kwargs):
        return Path('C:/tmp/qrcode.png')


class _FakeQrDiv:
    def __call__(self, locator, *_, **__):
        if locator == 'tag:img':
            return _FakeQrImage()
        return None


class _FakeTab:
    url = 'https://pay.weixin.qq.com'

    def __call__(self, locator, *_, **__):
        if locator == 'tag:div@@id=IDQrcodeImg':
            return _FakeQrDiv()
        if locator == 'tag:div@@class=qrcode-img':
            return _FakeQrDiv()
        return None

    def get(self, url):
        self.url = url

    def refresh(self):
        return None


def test_weipay_login_sends_only_one_qrcode_when_scan_step_fails(monkeypatch):
    monkeypatch.delenv('KQ_WEIPAY_LOGIN_ALLOW_RETRY', raising=False)
    monkeypatch.setenv('KQ_WEIPAY_QRCODE_LIFETIME_SECONDS', '20')
    monkeypatch.setenv('KQ_WEIPAY_LOGIN_TIMEOUT_SECONDS', '20')

    instance = Weipay.__new__(Weipay)
    instance.tab = _FakeTab()
    instance.get_recive = lambda *_args, **_kwargs: ''

    reset_calls = []
    sent_messages = []
    scan_calls = []

    monkeypatch.setattr(Weipay, '_reset_wechat_qrcode_windows_for_login', lambda *a, **k: reset_calls.append((a, k)) or {'status': 'ok'})
    monkeypatch.setattr(weipay, 'wechat_lock_send', lambda *a, **k: sent_messages.append((a, k)))
    monkeypatch.setattr(weipay, 'get_autogui_lock', lambda *a, **k: nullcontext())

    def fail_scan(*args, **kwargs):
        scan_calls.append((args, kwargs))
        raise TimeoutError('scan failed')

    monkeypatch.setattr(weipay.KqWechat, '扫码登录微信支付', fail_scan)

    with pytest.raises(TimeoutError, match='已重试 1 轮'):
        instance.login(users=['考勤后台'])

    assert len(sent_messages) == 1
    assert len(scan_calls) == 1
    assert len(reset_calls) == 2
