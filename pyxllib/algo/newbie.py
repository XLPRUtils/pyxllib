#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51


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
