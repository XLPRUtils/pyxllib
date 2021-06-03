#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/12/08 15:33

""" 一些高级的文本处理功能 """

import re


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


def strwidth(s):
    """ string width

    中英字符串实际宽度
    >>> strwidth('ab')
    2
    >>> strwidth('a⑪中⑩')
    7

    ⑩等字符的宽度还是跟字体有关的，不过在大部分地方好像都是域宽2，目前算法问题不大
    """
    try:
        res = len(s.encode('gbk'))
    except UnicodeEncodeError:
        count = len(s)
        for x in s:
            if ord(x) > 127:
                count += 1
        res = count
    return res
