#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

import re


def gettag_name(s):
    """ 获取tag名称

    >>> gettag_name('%<page>(1)')
    'page'
    >>> gettag_name('%<page/>')
    'page'
    >>> gettag_name('%<page type=1/>')
    'page'
    """
    m = re.match(r'%<([a-zA-Z0-9_]+)', s)
    if m:
        return m.group(1)
    return None


def settag_name(s, name):
    """ 修改tag名称

    >>> settag_name('%<page>(1)', 'image')
    '%<image>(1)'
    """
    m = re.match(r'(%<)([a-zA-Z0-9_]+)', s)
    if m:
        return m.group(1) + name + s[m.end():]
    return s


def gettag_attr(s, attr, default=None):
    """ 获取tag属性值

    >>> gettag_attr('%<page type=1>(1)', 'type')
    '1'
    >>> gettag_attr('%<page type="1">(1)', 'type')
    '1'
    >>> gettag_attr('%<page>(1)', 'type') is None
    True
    """
    # 简单的正则匹配，不支持太复杂的属性格式
    p = re.compile(r'\s' + attr + r'=["\']?([^"\'>\s]+)["\']?')
    m = p.search(s)
    if m:
        return m.group(1)
    return default


def settag_attr(s, attr, value):
    """ 设置tag属性值

    >>> settag_attr('%<page>(1)', 'type', 1)
    '%<page type="1">(1)'
    >>> settag_attr('%<page type=2>(1)', 'type', 1)
    '%<page type="1">(1)'
    """
    if value is None:  # 删除属性
        return re.sub(r'\s' + attr + r'=["\']?[^"\'>\s]+["\']?', '', s)

    # 查找是否有该属性
    p = re.compile(r'(\s' + attr + r'=)(["\']?[^"\'>\s]+["\']?)')
    m = p.search(s)
    if m:
        # 替换现有属性
        return s[:m.start(2)] + f'"{value}"' + s[m.end(2):]
    else:
        # 新增属性，添加到tag名后面
        m = re.match(r'(%<[a-zA-Z0-9_]+)', s)
        if m:
            return m.group(1) + f' {attr}="{value}"' + s[m.end():]
    return s


def brieftexstr(s, *, max_len=200):
    """ 生成简略的tex字符串，用于日志显示等

    :param s: 原始字符串
    :param max_len: 最大长度
    """
    s = str(s)
    s = re.sub(r'\s+', ' ', s)
    if len(s) > max_len:
        s = s[:max_len] + '...'
    return s
