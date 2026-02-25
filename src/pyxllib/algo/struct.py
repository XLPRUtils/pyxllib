#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 14:22

from pyxllib.algo.convert import int2excel_col_name
from pyxllib.algo.sort import natural_sort_key


def make_index_function(li, *, start=0, nan=None):
    """ 返回一个函数，输入值，返回对应下标，找不到时返回 not_found

    :param li: 列表数据
    :param start: 起始下标
    :param nan: 找不到对应元素时的返回值
        注意这里找不到默认不是-1，而是li的长度，这样用于排序时，找不到的默认会排在尾巴

    >>> func = make_index_function(['少儿', '小学', '初中', '高中'])
    >>> sorted(['初中', '小学', '高中'], key=func)
    ['小学', '初中', '高中']

    # 不在枚举项目里的，会统一列在最后面
    >>> sorted(['初中', '小学', '高中', '幼儿'], key=func)
    ['小学', '初中', '高中', '幼儿']
    """
    data = {x: i for i, x in enumerate(li, start=start)}
    if nan is None:
        nan = len(li)

    def warpper(x, default=None):
        if default is None:
            default = nan
        return data.get(x, default)

    return warpper


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


def intersection_split(a, b):
    """ 输入两个对象a,b，可以是dict或set类型，list等

    会分析出二者共有的元素值关系
    返回值是 ls1, ls2, ls3, ls4，大部分是list类型，但也有可能遵循原始情况是set类型
        ls1：a中，与b共有key的元素值
        ls2：a中，独有key的元素值
        ls3：b中，与a共有key的元素值
        ls4：b中，独有key的元素值
    """
    # 1 获得集合的key关系
    keys1 = set(a)
    keys2 = set(b)
    keys0 = keys1 & keys2  # 两个集合共有的元素

    # TODO 如果是字典，希望能保序

    # 2 组合出ls1、ls2、ls3、ls4

    def split(t, s, ks):
        """原始元素为t，集合化的值为s，共有key是ks"""
        if isinstance(t, (set, list, tuple)):
            return ks, s - ks
        elif isinstance(t, dict):
            ls1 = sorted(map(lambda x: (x, t[x]), ks), key=lambda x: natural_sort_key(x[0]))
            ls2 = sorted(map(lambda x: (x, t[x]), s - ks), key=lambda x: natural_sort_key(x[0]))
            return ls1, ls2
        else:
            # dprint(type(s))  # s不是可以用来进行集合规律分析的类型
            raise ValueError(f'{type(s)}不是可以用来进行集合规律分析的类型')

    ls1, ls2 = split(a, keys1, keys0)
    ls3, ls4 = split(b, keys2, keys0)
    return ls1, ls2, ls3, ls4
