#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/06/02 11:09


import pandas as pd

from pyxllib.debug.strlib import natural_sort_key


def dict2list(d: dict, *, nsort=False):
    """字典转n*2的list
    :param d: 字典
    :param nsort:
        True: 对key使用自然排序
        False: 使用d默认的遍历顺序
    :return:
    """
    ls = list(d.items())
    if nsort:
        ls = sorted(ls, key=lambda x: natural_sort_key(str(x[0])))
    return ls


def dict2df(d):
    """dict类型转DataFrame类型"""
    name = type(d)
    li = dict2list(d, nsort=True)
    return pd.DataFrame.from_records(li, columns=(f'{name}-key', f'{name}-value'))


def list2df(li, **kwargs):
    if li and isinstance(li[0], (list, tuple)):  # 有两维时按表格显示
        df = pd.DataFrame.from_records(li, **kwargs)
    else:  # 只有一维时按一列显示
        df = pd.DataFrame(pd.Series(li), **kwargs)
    return df


def try2df(arg):
    """尝试将各种不同的类型转成dataframe"""
    if isinstance(arg, dict):
        df = dict2df(arg)
    elif isinstance(arg, (list, tuple)):
        df = list2df(arg)
    elif isinstance(arg, pd.Series):
        df = pd.DataFrame(arg)
    else:
        df = arg
    return df
