from types import SimpleNamespace

import pytest

from kq5034 import attendance_api
from kq5034 import tools


@pytest.mark.parametrize('switch_during_query', [False, True])
def test_live_lookup_discards_results_if_shop_changes(monkeypatch, switch_during_query):
    events = []
    state = {'shop': ''}

    def switch(shop):
        state['shop'] = shop
        events.append(('switch', shop))

    def check(shop):
        events.append(('check', shop))
        if state['shop'] != shop:
            raise RuntimeError('shop changed')

    def lookup(*args):
        events.append(('lookup', args))
        if switch_during_query:
            state['shop'] = '宗门学府'
        return 'u_verified'

    browser = SimpleNamespace(switch_shop=switch, assert_shop=check, 查找用户=lookup)
    monkeypatch.setattr(attendance_api, 'ensure_attendance_runtime', lambda: None)
    monkeypatch.setattr(tools, 'KqTools', lambda: SimpleNamespace(xe2=browser))
    result = attendance_api.lookup_registration_users_browser(
        [{'key': '51', 'names': ['学员'], 'phones': ['13800000000']}],
        shop_id=1, close_browser=False,
    )
    assert [e[0] for e in events] == ['switch', 'check', 'lookup', 'check']
    assert all(e[1] == '5034山中薪' for e in events if e[0] != 'lookup')
    assert result[0]['user_id'] == ('' if switch_during_query else 'u_verified')
    if switch_during_query:
        assert result[0]['error'] == 'shop changed'


@pytest.mark.parametrize('tab_count,close_fails', [(1, False), (2, False), (2, True)])
def test_adapter_cleanup_preserves_shared_browser(tab_count, close_fails):
    events = []

    def close():
        events.append('close_owned_tab')
        if close_fails:
            raise RuntimeError('disconnected')

    owned = SimpleNamespace(tab_id='owned', close=close)
    tabs = [owned] + ([SimpleNamespace(tab_id='other')] if tab_count == 2 else [])
    browser = SimpleNamespace(
        get_tabs=lambda: tabs,
        quit=lambda: events.append('quit_browser'),
        close=lambda: events.append('close_browser'),
    )
    adapter = SimpleNamespace(browser=browser, tab=owned, quit=lambda: events.append('quit_adapter'))
    attendance_api._close_kqtools_browser(SimpleNamespace(_xe2=adapter))
    assert events == (['close_owned_tab'] if tab_count > 1 else [])
