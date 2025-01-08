#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/12/16

import sys

from loguru import logger

if sys.platform == 'win32':
    try:  # 尝试加载VIP版
        from wxautox import WeChat
        from wxautox.elements import WxParam  # WxParam.DEFALUT_SAVEPATH 可以用来配置数据自动保存位置
    except ModuleNotFoundError:  # 否则用贫民版
        from wxauto import WeChat
        from wxauto.elements import WxParam

from pyxllib.prog.filelock import get_autoui_lock


class WeChatSingletonLock:
    """ 基于 get_autoui_lock 的微信全局唯一单例控制器，确保同一时间仅有一个微信自动化程序在操作 """

    def __init__(self, lock_timeout=-1, *, init=True):
        # 初始化全局锁
        self.lock = get_autoui_lock(timeout=lock_timeout)
        self.wx = WeChat() if init else None

    def __enter__(self):
        # 获取锁并激活微信窗口
        self.lock.acquire()
        if self.wx:
            self.wx._show()
            return self.wx

    def __exit__(self, exc_type, exc_value, traceback):
        # 释放锁
        self.lock.release()


def wechat_lock_send(user, text=None, files=None, *, timeout=-1):
    """ 使用全局唯一单例锁，确保同一时间仅有一个微信自动化程序在操作 """
    with WeChatSingletonLock(timeout) as we:
        # 241223周一12:27，今天可被这个默认2秒坑惨了，往错误群一直发骚扰消息
        # 22:07，但我复测，感觉不可能找不到啊，为什么会找到禅宗考勤管理群呢，太离谱了
        status = we.ChatWith(user, timeout=5)

        if status != user:
            raise ValueError(f'无法找到用户：{user}')

        if text:
            we.SendMsg(text, user)
        if files:
            we.SendFiles(files, user)


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
