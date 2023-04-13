#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 14:26

"""
并查集相关功能
"""

from tqdm import tqdm

from pyxllib.prog.pupil import check_install_package

check_install_package('disjoint_set', 'disjoint-set==0.6.3')

from itertools import combinations

from disjoint_set import DisjointSet


def disjoint_set(items, join_checker, print_mode=False):
    """ 按照一定的相连规则分组

    :param items: 项目清单
    :param join_checker: 检查任意两个对象是否相连，进行分组
    :return:

    算法：因为会转成下标，按照下标进行分组合并，所以支持items里有重复值，或者unhashable对象

    >>> disjoint_set([-1, -2, 2, 0, 0, 1], lambda x, y: x*y>0)
    [[-1, -2], [2, 1], [0], [0]]
    """

    # 1 添加元素
    ds = DisjointSet()
    items = tuple(items)
    n = len(items)
    for i in range(n):
        ds.find(i)

    # 2 连接、分组
    for i, j in tqdm(combinations(range(n), 2), disable=not print_mode):
        if join_checker(items[i], items[j]):
            ds.union(i, j)

    # 3 返回分组信息
    res = []
    for group in ds.itersets():
        group_elements = [items[g] for g in group]
        res.append(group_elements)
    return res
