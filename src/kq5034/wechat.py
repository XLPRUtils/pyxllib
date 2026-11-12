"""Attendance sender and temporary notification routing, read on every send."""
import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from filelock import FileLock
from pyxllib.autogui.wxautolib import wechat_lock_send as _send, wechat_logger as _logger

ATTENDANCE_WECHAT_ACCOUNT_ID = 'wxid_gxgjjgft1oj722'


def routing_path():
    from pyxllib.autogui.weixin4_instrumentation import DEFAULT_CONTACT_DB
    return Path(os.environ.get('KQ_WECHAT_ROUTING_PATH') or
                DEFAULT_CONTACT_DB.parents[3] / 'attendance-wechat-routing.json')


def get_transport_config():
    """Missing override means 4102; malformed config fails closed."""
    path = routing_path()
    value = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    return {'sender_account_id': ATTENDANCE_WECHAT_ACCOUNT_ID, 'recipient_overrides': {}, **value}


def configure_transport(*, sender_account_id, recipient_overrides, reason,
                        redirect_all_to=None, redirect_until=None):
    """Persist explicit routing after checking sender/archive/recipients; never send.

    Restore via this API with the original sender and empty overrides.
    """
    from pyxllib.autogui.wechat_accounts import check_account
    if not reason.strip() or not isinstance(recipient_overrides, dict):
        raise ValueError('Explicit reason and recipient mapping required')
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
           for k, v in recipient_overrides.items()):
        raise ValueError('Recipient mapping must contain nonempty strings')
    if bool(redirect_all_to) != bool(redirect_until):
        raise ValueError('Global redirect requires recipient and expiry together')
    if redirect_all_to:
        expiry = datetime.fromisoformat(redirect_until)
        if expiry.tzinfo is None or expiry <= datetime.now(ZoneInfo('Asia/Shanghai')):
            raise ValueError('Redirect expiry must be a future timezone-aware timestamp')
    check_account(sender_account_id, list(set(recipient_overrides.values()) | ({redirect_all_to} if redirect_all_to else set())))
    value = {'sender_account_id': sender_account_id, 'recipient_overrides': recipient_overrides,
             'reason': reason}
    if redirect_all_to:
        value.update(redirect_all_to=redirect_all_to, redirect_until=redirect_until)
    path = routing_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + '.lock'):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    return get_transport_config()


def route_recipient(user, config):
    """One routing decision for reports and logger. Expiry never releases group sends."""
    if target := config.get('redirect_all_to'):
        if datetime.now(ZoneInfo('Asia/Shanghai')) >= datetime.fromisoformat(config['redirect_until']):
            raise ValueError('临时考勤转发已到期，请明确恢复或续期，消息尚未发送')
        return target
    return config['recipient_overrides'].get(user, user)


def _route_log(record):
    config = get_transport_config()
    extra = record['extra']
    extra['wechat_sender_account_id'] = config['sender_account_id']
    user = extra.get('wechat_user')
    if user:
        extra['wechat_user'] = route_recipient(user, config)
        if config.get('redirect_all_to') and user != extra['wechat_user']:
            record['message'] = f'[原收件人：{user}]\n' + record['message']


wechat_logger = _logger.patch(_route_log)


def wechat_lock_send(user, text=None, files=None, url=None, *, timeout=-1, **kwargs):
    """Route notifications through the explicitly configured sender."""
    config = get_transport_config()
    sender = kwargs.pop('sender_account_id', config['sender_account_id'])
    if sender != config['sender_account_id']:
        raise ValueError('Attendance sender differs from configured account')
    recipient = route_recipient(user, config)
    if config.get('redirect_all_to') and user != recipient and text:
        text = f'[原收件人：{user}]\n{text}'
    return _send(recipient, text, files, url, timeout=timeout, sender_account_id=sender, **kwargs)
