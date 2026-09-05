#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/12/16

import sys

from pyxllib.prog.lazyimport import lazy_import

try:
    from loguru import logger
except ModuleNotFoundError:
    logger = lazy_import('from loguru import logger')

if sys.platform == 'win32':
    try:
        from wxauto4 import WeChat, WxParam
    except ModuleNotFoundError:
        WeChat = lazy_import('from wxauto4 import WeChat', 'wxauto4')
        WxParam = lazy_import('from wxauto4 import WxParam', 'wxauto4')

from pyxllib.prog.filelock import get_autogui_lock


class WeChatSingletonLock:
    """ 基于 get_autogui_lock 的微信全局唯一单例控制器，确保同一时间仅有一个微信自动化程序在操作 """

    def __init__(self, lock_timeout=-1, *, init=True):
        # 初始化全局锁
        self.lock = get_autogui_lock(timeout=lock_timeout)
        self.init = init
        self.wx = None

    def __enter__(self):
        # 获取锁并激活微信窗口
        self.lock.acquire()
        if self.init and self.wx is None:
            try:
                self.wx = WeChat()
            except Exception as exc:
                # 微信 4.x 已不再暴露旧版 WeChatMainWndForPC 控件树。
                # 只替换底层文本传输，上层 Step 6、通知对象和文案保持原样。
                from pyxllib.autogui.weixin4 import Weixin4TextClient

                self.wx = Weixin4TextClient()
        if self.wx:
            self.wx._show()
            return self.wx

    def __exit__(self, exc_type, exc_value, traceback):
        # 释放锁
        self.lock.release()


def wechat_lock_send(user, text=None, files=None, url=None, *, timeout=-1, **kwargs):
    """通过进程内 API 发送微信消息，绝不激活或操作微信 GUI。

    旧调用名为保持业务兼容而保留。API 尚未支持的消息类型必须失败关闭，
    不能以软件升级、版本不匹配或能力缺失为由降级到桌面自动化。
    """
    del timeout
    if files or url or kwargs.get('at') or not text:
        raise NotImplementedError("微信 API 当前只支持不带 @ 的纯文本消息，禁止降级 GUI")

    from pyxllib.autogui.weixin4_instrumentation import send_text

    return send_text(user, str(text))


def wechat_handler(message):
    # 获取群名，如果没有指定，不使用此微信发送功能
    user = message.record["extra"].get("wechat_user")
    if user:
        wechat_lock_send(user, message)


if sys.platform == 'win32':
    # 创建专用的微信日志记录器，不绑定默认群名
    wechat_logger = logger.bind(wechat_user='文件传输助手')

    # 添加专用的微信处理器
    wechat_logger.add(wechat_handler,
                      format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}")

    """ 往微信发送特殊的日志格式报告
    用法：wechat_logger.bind(wechat_user='文件传输助手').info(message)

    或者：
    # 先做好默认群名绑定
    wechat_logger = wechat_logger.bind(wechat_user='考勤管理')
    # 然后就能普通logger用法发送了
    wechat_logger.info('测试')
    """
else:
    # 降级为普通logger
    wechat_logger = logger
