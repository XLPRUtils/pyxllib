#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

from pyxllib.prog.newbie import round_int


def vector_compare(x, y):
    """
    :param x: 一个数值向量
    :param y: 一个数值向量，理论上长度要跟x相同
    :return: 返回<、=、>、?
        =，各个位置的数值都相同
        <，x各个位置的值都≤y，且至少有一处值<y
        >，x各个位置的值都≥，且至少有一处值>y
        ?，其他情况，x、y各有优劣

    >>> vector_compare([1, 2, 3], [1, 3, 3])
    '<'
    >>> vector_compare([1, 2, 3], [1, 1, 4])
    '?'
    >>> vector_compare({'a': 1, 'b': 2}, {'a': 1, 'b': 2, 'c': 0})
    '='
    >>> vector_compare({'b': 3, 'a': 1}, {'a': 1, 'b': 2, 'c': 0})
    '>'
    >>> vector_compare({'b': 3, 'a': 1}, {'a': 1, 'b': 2, 'c': 1})
    '?'
    """
    # 1 不同数据类型处理
    if isinstance(x, dict) and isinstance(y, dict):
        # 字典比较比较特别，要先做个映射
        keys = set(x.keys()) | set(y.keys())
        a = [x.get(k, 0) - y.get(k, 0) for k in keys]
    else:
        a = [i - j for i, j in zip(x, y)]

    # 2 算法
    n, m = max(a), min(a)
    if n == m == 0:
        return '='
    elif n <= 0:
        return '<'
    elif m >= 0:
        return '>'
    else:
        return '?'


def round_unit(x, unit):
    """ 按特定单位量对x取倍率

    round_int偏向于工程代码简化，round_unit偏向算法，功能不太一样，所以分组不同

    Args:
        x: 原值
        unit: 单位量

    Returns: 新值，是unit的整数倍

    >>> round_unit(1.2, 0.5)
    1.0
    >>> round_unit(1.6, 0.5)
    1.5
    >>> round_unit(7, 5)
    5
    >>> round_unit(13, 5)
    15
    """
    return round_int(x / unit) * unit


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
        d = d * 26 + (ord(ch.upper()) - 64)
    return d


def gentuple(n, tag):
    """ 有点类似range函数，但生成的数列更加灵活

    :param n:
        数组长度
    :param tag:
        int类型，从指定数字开始编号
            0，从0开始编号
            1，从1开始编号
        'A'，用Excel的形式编号
        tuple，按枚举值循环显示
            ('A', 'B')：循环使用A、B编号

    >>> gentuple(4, 'A')
    ('A', 'B', 'C', 'D')
    """
    a = [''] * n
    if isinstance(tag, int):
        for i in range(n):
            a[i] = i + tag
    elif tag == 'A':
        a = tuple(map(lambda x: int2excel_col_name(x + 1), range(n)))
    elif isinstance(tag, (list, tuple)):
        k = len(tag)
        a = tuple(map(lambda x: tag[x % k], range(n)))
    return a




