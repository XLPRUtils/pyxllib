#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/09/07 10:21


import win32com.client as win32
import pythoncom


def get_win32_app(name, visible=False):
    """ 启动可支持pywin32自动化处理的应用

    Args:
        str name: 应用名称，不区分大小写，比如word, excel, powerpoint, onenote
            不带'.'的情况下，会自动添加'.Application'的后缀
        visible: 应用是否可见

    Returns: app

    """
    # 1 name
    name = name.lower()
    if '.' not in name:
        name += '.application'

    # 2 app
    # 这里可能还有些问题，不同的应用，机制不太一样，后面再细化完善吧
    try:
        app = win32.GetActiveObject(f'{name}')  # 不能关联到普通方式打开的应用。但代码打开的应用都能找得到。
    except pythoncom.com_error:
        app = win32.gencache.EnsureDispatch(f'{name}')
        # 还有种常见的初始化方法，是 win32com.client.Dispatch和win32com.client.dynamic.Dispatch
        # from win32com.client.dynamic import Disypatch

    if visible is not None:
        app.Visible = visible

    return app
