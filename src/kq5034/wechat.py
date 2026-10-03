"""考勤微信统一使用 code4102 / 代号4102；账号身份以 wxid 为准。"""
from pyxllib.autogui.wxautolib import wechat_lock_send as _send, wechat_logger as _logger

ATTENDANCE_WECHAT_ACCOUNT_ID = "wxid_gxgjjgft1oj722"
wechat_logger = _logger.bind(wechat_sender_account_id=ATTENDANCE_WECHAT_ACCOUNT_ID)


def wechat_lock_send(user, text=None, files=None, url=None, *, timeout=-1, **kwargs):
    """考勤日报、问卷、登录提醒统一固定业务账号，调用方只指定收件人。"""
    sender = kwargs.pop("sender_account_id", ATTENDANCE_WECHAT_ACCOUNT_ID)
    if sender != ATTENDANCE_WECHAT_ACCOUNT_ID:
        raise ValueError("考勤微信只能使用代号4102账号 (wxid_gxgjjgft1oj722)")
    return _send(user, text, files, url, timeout=timeout,
                 sender_account_id=ATTENDANCE_WECHAT_ACCOUNT_ID, **kwargs)
