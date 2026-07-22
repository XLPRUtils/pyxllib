import pytest

from kq5034 import wechat_runtime


class _FakeRect:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom


class _FakeControl:
    BoundingRectangle = _FakeRect(100, 200, 511, 948)

    def __init__(self):
        self.activated = False


class _FakeUiCtrlNode:
    def __init__(self, ctrl, build_depth=1):
        self.ctrl = ctrl
        self.build_depth = build_depth

    def activate(self):
        self.ctrl.activated = True


def test_click_wechat_pay_merchant_row_prefers_semantic_control(monkeypatch):
    ctrl = _FakeControl()
    semantic_calls = []
    coordinate_calls = []

    def fake_click_matching(root, keywords, **kwargs):
        semantic_calls.append((root, keywords, kwargs))
        return True

    monkeypatch.setattr(wechat_runtime, '_点击匹配控件', fake_click_matching)
    monkeypatch.setattr(wechat_runtime, '_点击微信支付商户行OCR', lambda _ctrl: False)
    monkeypatch.setattr(wechat_runtime.pyautogui, 'click', lambda *args: coordinate_calls.append(args))

    assert wechat_runtime._点击微信支付商户行(ctrl) is True
    assert semantic_calls == [(ctrl, ['1599622041', '武陵禅寺客堂'], {})]
    assert coordinate_calls == []


def test_click_wechat_pay_merchant_row_uses_ocr_before_coordinate_fallback(monkeypatch):
    ctrl = _FakeControl()
    ocr_calls = []
    coordinate_calls = []

    monkeypatch.setattr(wechat_runtime, '_点击匹配控件', lambda *args, **kwargs: False)
    monkeypatch.setattr(wechat_runtime, '_点击微信支付商户行OCR', lambda c: ocr_calls.append(c) or True)
    monkeypatch.setattr(wechat_runtime.pyautogui, 'click', lambda *args: coordinate_calls.append(args))

    assert wechat_runtime._点击微信支付商户行(ctrl) is True
    assert ocr_calls == [ctrl]
    assert coordinate_calls == []


def test_click_wechat_pay_merchant_row_clicks_ocr_row_right_side(monkeypatch):
    ctrl = _FakeControl()
    clicks = []

    monkeypatch.setattr(
        wechat_runtime,
        '_微信支付商户行OCR文本框',
        lambda _ctrl: {
            'window': [100, 200, 511, 948],
            'width': 411,
            'height': 748,
            'text': '武陵禅寺客堂',
            'box': [50, 240, 165, 266],
            'score': 3,
        },
    )
    monkeypatch.setattr(wechat_runtime.pyautogui, 'click', lambda x, y: clicks.append((x, y)))

    assert wechat_runtime._点击微信支付商户行OCR(ctrl) is True
    assert clicks == [(pytest.approx(100 + 411 * 0.86), pytest.approx(200 + (240 + 266) / 2))]


def test_wechat_pay_ocr_rows_support_common_ocr_labelme_document():
    rows = wechat_runtime._微信支付OCR文本框列表({
        'shapes': [
            {
                'label': {'text': '武陵禅寺客堂', 'score': 0.99},
                'points': [[47, 239], [155, 239], [155, 259], [47, 259]],
            },
            {
                'label': '{"text": "1599622041"}',
                'points': [[50, 271], [131, 271], [131, 285], [50, 285]],
            },
        ],
    })

    assert rows == [
        {'text': '武陵禅寺客堂', 'box': [47.0, 239.0, 155.0, 259.0]},
        {'text': '1599622041', 'box': [50.0, 271.0, 131.0, 285.0]},
    ]


def test_wechat_pay_helper_screenshot_prefers_anlib_multiscreen(monkeypatch):
    calls = []

    monkeypatch.setattr(wechat_runtime.pyautogui, 'screenshot', lambda **kwargs: 'pyautogui')
    monkeypatch.setitem(
        __import__('sys').modules,
        'pyxllib.autogui.anlib',
        type('FakeAnlib', (), {'_screenshot_region': staticmethod(lambda region: calls.append(region) or 'anlib')}),
    )

    assert wechat_runtime._微信支付商家助手截图(100, 200, 411, 748) == 'anlib'
    assert calls == [[100, 200, 411, 748]]


def test_click_wechat_pay_merchant_row_uses_explicit_coordinate_fallback_when_ocr_misses(monkeypatch):
    ctrl = _FakeControl()
    clicks = []

    monkeypatch.setattr(wechat_runtime, '_点击匹配控件', lambda *args, **kwargs: False)
    monkeypatch.setattr(wechat_runtime, '_点击微信支付商户行OCR', lambda _ctrl: False)
    monkeypatch.setattr(wechat_runtime, 'UiCtrlNode', _FakeUiCtrlNode)
    monkeypatch.setattr(wechat_runtime.pyautogui, 'click', lambda x, y: clicks.append((x, y)))

    assert wechat_runtime._点击微信支付商户行(ctrl) is True

    assert ctrl.activated is True
    assert len(clicks) == 1
    x, y = clicks[0]
    assert x == pytest.approx(100 + (511 - 100) * 0.28)
    assert y == pytest.approx(200 + (948 - 200) * 0.345)


def test_normalize_wechat_pay_helper_activates_visible_window(monkeypatch):
    ctrl = _FakeControl()

    monkeypatch.setattr(wechat_runtime, 'UiCtrlNode', _FakeUiCtrlNode)
    monkeypatch.setattr(wechat_runtime.pyautogui, 'size', lambda: (1920, 1080))

    assert wechat_runtime._规范化微信支付商家助手窗口(ctrl) is ctrl
    assert ctrl.activated is True


def test_wait_after_merchant_selection_is_not_skipped(monkeypatch):
    sleeps = []

    monkeypatch.delenv('KQ_WECHAT_PAY_MERCHANT_SELECT_SETTLE_SECONDS', raising=False)
    monkeypatch.setattr(wechat_runtime.time, 'sleep', lambda seconds: sleeps.append(seconds))

    wechat_runtime._等待微信支付商户选择生效()

    assert sleeps == [2.5]
