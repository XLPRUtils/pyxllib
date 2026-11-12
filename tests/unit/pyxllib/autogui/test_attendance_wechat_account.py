from types import SimpleNamespace
import pytest
from kq5034 import wechat
from pyxllib.autogui import wxautolib


def test_attendance_sender_is_fixed(monkeypatch, tmp_path):
    monkeypatch.setenv("KQ_WECHAT_ROUTING_PATH", str(tmp_path / "routing.json"))
    calls = []
    monkeypatch.setattr("pyxllib.autogui.weixin4_instrumentation.send_text", lambda *args, **kwargs: calls.append(kwargs))
    wechat.wechat_lock_send("考勤中台", "mock only")
    assert calls == [{"sender_account_id": wechat.ATTENDANCE_WECHAT_ACCOUNT_ID}]
    with pytest.raises(ValueError):
        wechat.wechat_lock_send("考勤中台", "mock only", sender_account_id="main")


def test_attendance_logger_keeps_account_and_general_logger_keeps_default(monkeypatch):
    calls = []
    monkeypatch.setattr(wxautolib, "wechat_lock_send", lambda *args, **kwargs: calls.append(kwargs))
    wxautolib.wechat_handler(SimpleNamespace(record={"extra": {"wechat_user": "filehelper", "wechat_sender_account_id": wechat.ATTENDANCE_WECHAT_ACCOUNT_ID}}))
    wxautolib.wechat_handler(SimpleNamespace(record={"extra": {"wechat_user": "filehelper"}}))
    assert calls == [{"sender_account_id": wechat.ATTENDANCE_WECHAT_ACCOUNT_ID}, {}]


def test_temporary_route_applies_to_send_and_logger_and_can_restore(monkeypatch, tmp_path):
    monkeypatch.setenv('KQ_WECHAT_ROUTING_PATH', str(tmp_path / 'routing.json'))
    checks, calls = [], []
    monkeypatch.setattr('pyxllib.autogui.wechat_accounts.check_account', lambda a, r: checks.append((a,r)))
    monkeypatch.setattr(wechat, '_send', lambda *a, **k: calls.append((a,k)))
    new, owner = 'wxid_temporary', 'wxid_owner'
    wechat.configure_transport(sender_account_id=new,
        recipient_overrides={'考勤中台': owner}, reason='temporary outage')
    wechat.wechat_lock_send('考勤中台', '日报')
    assert calls[0][0][:2] == (owner, '日报')
    assert calls[0][1]['sender_account_id'] == new
    record = {'extra': {'wechat_user':'考勤中台'}}
    wechat._route_log(record)
    assert record['extra'] == {'wechat_user':owner,'wechat_sender_account_id':new}
    with pytest.raises(ValueError):
        wechat.wechat_lock_send('考勤中台','x',sender_account_id=wechat.ATTENDANCE_WECHAT_ACCOUNT_ID)
    wechat.configure_transport(sender_account_id=wechat.ATTENDANCE_WECHAT_ACCOUNT_ID,
        recipient_overrides={}, reason='restored')
    wechat.wechat_lock_send('考勤中台', 'normal')
    assert calls[-1][0][0] == '考勤中台'
    assert checks[0] == (new,[owner])


def test_failed_identity_check_does_not_replace_route(monkeypatch, tmp_path):
    monkeypatch.setenv('KQ_WECHAT_ROUTING_PATH', str(tmp_path/'routing.json'))
    def unavailable(*args):
        raise RuntimeError('offline')
    monkeypatch.setattr('pyxllib.autogui.wechat_accounts.check_account', unavailable)
    with pytest.raises(RuntimeError, match='offline'):
        wechat.configure_transport(sender_account_id='wxid_new', recipient_overrides={},reason='test')
    assert wechat.get_transport_config()['sender_account_id'] == wechat.ATTENDANCE_WECHAT_ACCOUNT_ID


@pytest.mark.parametrize('original', ['其他课程群', '123456@chatroom', 'wxid_another', '考勤中台'])
def test_global_redirect_covers_unlisted_recipients_and_labels_source(monkeypatch, tmp_path, original):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    monkeypatch.setenv('KQ_WECHAT_ROUTING_PATH', str(tmp_path/'routing.json'))
    monkeypatch.setattr('pyxllib.autogui.wechat_accounts.check_account', lambda *a: {})
    calls=[]
    monkeypatch.setattr(wechat, '_send', lambda *a, **kw: calls.append((a,kw)))
    wechat.configure_transport(sender_account_id='wxid_temp',recipient_overrides={}, reason='today only',
        redirect_all_to='wxid_owner',redirect_until=(datetime.now(ZoneInfo('Asia/Shanghai'))+timedelta(hours=1)).isoformat())
    wechat.wechat_lock_send(original,'待处理消息')
    assert calls[0][0][0]=='wxid_owner'
    assert calls[0][0][1]==f'[原收件人：{original}]\n待处理消息'
    record={'extra':{'wechat_user':original},'message':'日志'}
    wechat._route_log(record)
    assert record['extra']['wechat_user']=='wxid_owner'
    assert record['message']==f'[原收件人：{original}]\n日志'


def test_expired_redirect_blocks_instead_of_releasing_to_group(monkeypatch):
    config={'sender_account_id':'wxid_temp','recipient_overrides':{},
            'redirect_all_to':'wxid_owner','redirect_until':'2000-01-01T00:00:00+08:00'}
    monkeypatch.setattr(wechat,'get_transport_config',lambda:config)
    calls=[]
    monkeypatch.setattr(wechat,'_send',lambda *a,**k:calls.append(a))
    with pytest.raises(ValueError,match='到期'):
        wechat.wechat_lock_send('课程群','日报')
    with pytest.raises(ValueError,match='到期'):
        wechat._route_log({'extra':{'wechat_user':'课程群'},'message':'日志'})
    assert not calls
