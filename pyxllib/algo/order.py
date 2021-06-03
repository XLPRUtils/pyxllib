#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 14:22

""" 排序相关 """

import itertools
import re

from deprecated import deprecated


def natural_sort_key(key):
    """
    >>> natural_sort_key('0.0.43') < natural_sort_key('0.0.43.1')
    True

    >>> natural_sort_key('0.0.2') < natural_sort_key('0.0.12')
    True
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    return [convert(c) for c in re.split('([0-9]+)', str(key))]


def natural_sort(ls, only_use_digits=False):
    """ 自然排序

    :param only_use_digits: 正常会用数字作为分隔，切割每一部分进行比较
        如果只想比较数值部分，可以only_use_digits=True

    >>> natural_sort(['0.1.12', '0.0.10', '0.0.23'])
    ['0.0.10', '0.0.23', '0.1.12']
    """
    if only_use_digits:
        def func(key):
            return [int(c) for c in re.split('([0-9]+)', str(key)) if c.isdigit()]
    else:
        func = natural_sort_key
    return sorted(ls, key=func)


@deprecated(reason='这个实现方式不佳，请参考 make_index_function')
def sort_by_given_list(a, b):
    r""" 本函数一般用在数据透视表中，分组中元素名为中文，没有按指定规律排序的情况

    :param a: 需要排序的对象
    :param b: 参照的排序数组
    :return: 排序后的a

    >>> sort_by_given_list(['初中', '小学', '高中'], ['少儿', '小学', '初中', '高中'])
    ['小学', '初中', '高中']

    # 不在枚举项目里的，会统一列在最后面
    >>> sort_by_given_list(['初中', '小学', '高中', '幼儿'], ['少儿', '小学', '初中', '高中'])
    ['小学', '初中', '高中', '幼儿']
    """
    # 1 从b数组构造一个d字典，d[k]=i，值为k的元素在第i位
    d = dict()
    for i, bb in enumerate(b): d[bb] = i
    # 2 a数组分两部分，可以通过d排序的a1，和不能通过d排序的a2
    a1, a2 = [], []
    for aa in a:
        if aa in d:
            a1.append(aa)
        else:
            a2.append(aa)
    # 3 用不同的规则排序a1、a2后合并
    a1 = sorted(a1, key=lambda x: d[x])
    a2 = sorted(a2)
    return a1 + a2


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


def product(*iterables, order=None, repeat=1):
    """ 对 itertools 的product扩展orders参数的更高级的product迭代器

    :param order: 假设iterables有n=3个迭代器，则默认 orders=[1, 2, 3] （起始编号1）
        即标准的product，是按顺序对每个迭代器进行重置、遍历的
        但是我扩展的这个接口，允许调整每个维度的更新顺序
        例如设置为 [-2, 1, 3]，表示先对第2维降序，然后按第1、3维的方式排序获得各个坐标点
        注：可以只输入[-2]，默认会自动补充维[1, 3]

        不从0开始编号，是因为0没法记录正负排序情况

    for x in product('ab', 'cd', 'ef', order=[3, -2, 1]):
        print(x)

    ['a', 'd', 'e']
    ['b', 'd', 'e']
    ['a', 'c', 'e']
    ['b', 'c', 'e']
    ['a', 'd', 'f']
    ['b', 'd', 'f']
    ['a', 'c', 'f']
    ['b', 'c', 'f']

    TODO 我在想numpy这么牛逼，会不会有等价的功能接口可以实现，我不用重复造轮子？
    """
    import numpy as np

    # 一、标准调用方式
    if order is None:
        for x in itertools.product(*iterables, repeat=repeat):
            yield x
        return

    # 二、输入orders参数的调用方式
    # 1 补全orders参数长度
    n = len(iterables)
    for i in range(1, n + 1):
        if not (i in order or -i in order):
            order.append(i)
    if len(order) != n: return ValueError(f'orders参数值有问题 {order}')

    # 2 生成新的迭代器组
    new_iterables = [(iterables[i - 1] if i > 0 else reversed(iterables[-i - 1])) for i in order]
    idx = np.argsort([abs(i) - 1 for i in order])
    for y in itertools.product(*new_iterables, repeat=repeat):
        yield [y[i] for i in idx]
