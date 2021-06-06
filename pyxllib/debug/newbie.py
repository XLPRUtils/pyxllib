#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

def funcmsg(func):
    """输出函数func所在的文件、函数名、函数起始行"""
    # showdir(func)
    if not hasattr(func, '__name__'):  # 没有__name__属性表示这很可能是一个装饰器去处理原函数了
        if hasattr(func, 'func'):  # 我的装饰器常用func成员存储原函数对象
            func = func.func
        else:
            return f'装饰器：{type(func)}，无法定位'
    return f'函数名：{func.__name__}，来自文件：{func.__code__.co_filename}，所在行号={func.__code__.co_firstlineno}'
