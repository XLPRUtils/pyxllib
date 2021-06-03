#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 23:04


def int2excel_col_name(d):
    """
    >>> int2excel_col_name(1)
    'A'
    >>> int2excel_col_name(28)
    'AB'
    >>> int2excel_col_name(100)
    'CV'
    """
    s = []
    while d:
        t = (d - 1) % 26
        s.append(chr(65 + t))
        d = (d - 1) // 26
    return ''.join(reversed(s))


def excel_col_name2int(s):
    """
    >>> excel_col_name2int('A')
    1
    >>> excel_col_name2int('AA')
    27
    >>> excel_col_name2int('AB')
    28
    """
    d = 0
    for ch in s:
        d = d * 26 + (ord(ch) - 64)
    return d
