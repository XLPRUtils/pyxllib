#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

import os
import re
import textwrap


def shorten(s, width=200, placeholder='...'):
    """
    :param width: 这个长度是上限，即使用placeholder时的字符串总长度也在这个范围内

    >>> shorten('aaa', 10)
    'aaa'
    >>> shorten('hell world! 0123456789 0123456789', 11)
    'hell wor...'
    >>> shorten("Hello  world!", width=12)
    'Hello world!'
    >>> shorten("Hello  world!", width=11)
    'Hello wo...'
    >>> shorten('0123456789 0123456789', 2, 'xyz')  # 自己写的shorten
    'xy'

    注意textwrap.shorten的缩略只针对空格隔开的单词有效，我这里的功能与其不太一样
    >>> textwrap.shorten('0123456789 0123456789', 11)  # 全部字符都被折叠了
    '[...]'
    >>> shorten('0123456789 0123456789', 11)  # 自己写的shorten
    '01234567...'
    """
    s = re.sub(r'\s+', ' ', str(s))
    n, m = len(s), len(placeholder)
    if n > width:
        s = s[:max(width - m, 0)] + placeholder
    return s[:width]  # 加了placeholder在特殊情况下也会超，再做个截断最保险

    # return textwrap.shorten(str(s), width)


class StrDecorator:
    """将函数的返回值字符串化，仅调用朴素的str字符串化

    装饰器开发可参考： https://mp.weixin.qq.com/s/Om98PpncG52Ba1ZQ8NIjLA
    """

    def __init__(self, func):
        self.func = func  # 使用self.func可以索引回原始函数名称
        self.last_raw_res = None  # last raw result，上一次执行函数的原始结果

    def __call__(self, *args, **kwargs):
        self.last_raw_res = self.func(*args, **kwargs)
        return str(self.last_raw_res)


class PrintDecorator:
    """将函数返回结果直接输出"""

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        s = self.func(*args, **kwargs)
        print(s)
        return s  # 输出后仍然会返回原函数运行值


def binary_cut_str(s, fmt='0'):
    """180801坤泽：“二分”切割字符串
    :param s: 要截取的全字符串
    :param fmt: 截取格式，本来是想只支持0、1的，后来想想支持23456789也行
        0：左边一半
        1：右边的1/2
        2：右边的1/3
        3：右边的1/4
        ...
        9：右边的1/10
    :return: 截取后的字符串

    >>> binary_cut_str('1234', '0')
    '12'
    >>> binary_cut_str('1234', '1')
    '34'
    >>> binary_cut_str('1234', '10')
    '3'
    >>> binary_cut_str('123456789', '20')
    '7'
    >>> binary_cut_str('123456789', '210')  # 向下取整，'21'获得了9，然后'0'取到空字符串
    ''
    """
    for t in fmt:
        t = int(t)
        n = len(s) // (1 + max(1, t))
        if t == 0:
            s = s[:n]
        else:
            s = s[(len(s) - n):]
    return s


def endswith(s, tags):
    """除了模拟str.endswith方法，输入的tag也可以是可迭代对象

    >>> endswith('a.dvi', ('.log', '.aux', '.dvi', 'busy'))
    True
    """
    if isinstance(tags, str):
        return s.endswith(tags)
    elif isinstance(tags, (list, tuple)):
        for t in tags:
            if s.endswith(t):
                return True
    else:
        raise TypeError
    return False


def refine_digits_set(digits):
    """美化连续数字的输出效果

    >>> refine_digits_set([210, 207, 207, 208, 211, 212])
    '207,208,210-212'
    """
    arr = sorted(list(set(digits)))  # 去重
    n = len(arr)
    res = ''
    i = 0
    while i < n:
        j = i + 2
        if j < n and arr[i] + 2 == arr[j]:
            while j < n and arr[j] - arr[i] == j - i:
                j += 1
            j = j if j < n else n - 1
            res += str(arr[i]) + '-' + str(arr[j]) + ','
            i = j + 1
        else:
            res += str(arr[i]) + ','
            i += 1
    return res[:-1]  # -1是去掉最后一个','


def del_tail_newline(s):
    """删除末尾的换行"""
    if len(s) > 1 and s[-1] == '\n':
        s = s[:-1]
    return s


def latexstrip(s):
    """latex版的strip"""
    return s.strip('\t\n ~')


def add_quote(s):
    return f'"{s}"'


def remove_prefix(original_string, prefix):
    if original_string.startswith(prefix):
        return original_string[len(prefix):]
    return original_string


def remove_suffix(original_string, suffix):
    if original_string.endswith(suffix):
        return original_string[:-len(suffix)]
    return original_string


def printoneline(s):
    """将输出控制在单行，适应终端大小"""
    try:
        columns = os.get_terminal_size().columns - 3  # 获取终端的窗口宽度
    except OSError:  # 如果没和终端相连，会抛出异常
        # 这应该就是在PyCharm，直接来个大值吧
        columns = 500
    s = shorten(s, columns)
    print(s)
