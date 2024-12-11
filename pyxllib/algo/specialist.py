#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 11:16

import copy
import itertools

import numpy as np
import pandas as pd

from pyxllib.prog.pupil import DictTool
# from pyxllib.prog.deprecatedlib import deprecated
from deprecated import deprecated


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


class MatchPairs:
    """ 匹配类，对X,Y两组数据中的x,y等对象按照cmp_func的规则进行相似度配对

    MatchBase(ys, cmp_func).matches(xs, least_score)
    """

    def __init__(self, ys, cmp_func):
        self.ys = list(ys)
        self.cmp_func = cmp_func

    def __getitem__(self, idx):
        return self.ys[idx]

    # def __del__(self, idx):
    #     del self.ys[idx]

    def __len__(self):
        return len(self.ys)

    def match(self, x, k=1):
        """ 匹配一个对象

        :param x: 待匹配的一个对象
        :param k: 返回次优的几个结果
        :return:
            当 k = 1 时，返回 (idx, score)
            当 k > 1 时，返回 [(idx1, score1), (idx2, score2), ...]
        """
        scores = [self.cmp_func(x, y) for y in self.ys]
        if k == 1:
            score = max(scores)
            idx = scores.index(score)
            return idx, score
        else:
            # 按权重从大到小排序
            idxs = np.argsort(scores)
            idxs = idxs[::-1][:k]
            return [(idx, scores[idx]) for idx in idxs]

    def matches(self, xs):
        """ 对xs中每个元素都找到一个最佳匹配对象

        注意：这个功能是支持ys中的元素被重复匹配的，而且哪怕相似度很低，也会返回一个最佳匹配结果
            如果想限定相似度，或者不支持重复匹配，请到隔壁使用 matchpairs

        :param xs: 要匹配的一组对象
        :return: 为每个x找到一个最佳的匹配y，存储其下标和对应的分值
            [(idx0, score0), (idx1, score1), ...]  长度 = len(xs)

        >>> m = MatchPairs([1, 5, 8, 9, 2], lambda x,y: 1-abs(x-y)/max(x,y))
        >>> m.matches([4, 6])  # 这里第1个值是下标，所以分别是对应5、8
        [(1, 0.8), (1, 0.8333333333333334)]

        # 要匹配的对象多于实际ys，则只会返回前len(ys)个结果
        # 这种情况建议用matchpairs功能实现，或者实在想用就对调xs、ys
        >>> m.matches([4, 6, 1, 2, 9, 4, 5])
        [(1, 0.8), (1, 0.8333333333333334), (0, 1.0), (4, 1.0), (3, 1.0), (1, 0.8), (1, 1.0)]
        """
        return [self.match(x) for x in xs]


def get_ndim(coords):
    # 注意 np.array(coords[:1])，只需要取第一个元素就可以判断出ndim
    coords = coords if isinstance(coords, np.ndarray) else np.array(coords[:1])
    return coords.ndim


class DictCmper:
    """ 字典结构比较工具

    集合是字典的特殊情况，也支持集合间的对比
    """

    def __init__(self, dicts):
        """
        :param Dict[str, Dict|set] dicts: Dicts[dictname, dict_]
            如果是set，会升级为dict，value默认为1
        """
        self.dicts = {}
        for k, v in dicts.items():
            if isinstance(v, set):
                self.dicts[k] = {k: 1 for k in v}
            else:
                self.dicts[k] = v

    def details(self):
        """ 返回详细的分析表

        按行罗列所有键，按列罗列所有字典，中间显示各字典键值
        可以把结果保存到excel，然后详细筛选分析

        >>> dc = DictCmper({'d1': {'a': 1, 'b': 2}, 'd2': {'b': 3, 'e': 5}, 'd3': {'d': 4}})
        >>> dc.details()
            d1   d2   d3
        a  1.0  NaN  NaN
        b  2.0  3.0  NaN
        e  NaN  5.0  NaN
        d  NaN  NaN  4.0
        """
        # 1 获得所有键
        #   集合无法保存元素顺序，所以用合并字典来代替
        #   还有个三方库orderedset，不想安装。就这样简便解决就好。
        keys = DictTool.or_(*self.dicts.values()).keys()

        # 2 取出所有字典值
        ls = []
        for k in keys:
            ls.append([(d[k] if k in d else np.nan) for d in self.dicts.values()])

        # 3 转为df表格
        df = pd.DataFrame.from_records(ls, columns=self.dicts.keys())
        df.index = keys
        return df

    def pair_summary(self, func=lambda x, y: len(x.keys() & y.keys())):
        r""" 两两对比表

        :param func: 默认是计算两个字典共有的键数量
        :return: df
            df对角线存储的是每个集合自身大小，df第i行第j列是第i个集合减去第j个集合的剩余元素数

        >>> s1 = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        >>> s2 = {1, 3, 5, 7, 8}
        >>> s3 = {2, 3, 5, 8}
        >>> dc = DictCmper({'s1': s1, 's2': s2, 's3': s3})
        >>> dc.pair_summary()
            s1  s2  s3
        s1   9   5   4
        s2   5   5   3
        s3   4   3   4
        >>> dc.pair_summary(lambda x, y: len(x.keys() - y.keys()))
            s1  s2  s3
        s1   0   4   5
        s2   0   0   2
        s3   0   1   0
        """
        dictnames = list(self.dicts.keys())
        n = len(dictnames)
        rows = []
        for i, name1 in enumerate(dictnames):
            row = [0] * n
            for j, name2 in enumerate(dictnames):
                row[j] = func(self.dicts[name1], self.dicts[name2])
            rows.append(row)
        df = pd.DataFrame.from_records(rows, columns=dictnames)
        df.index = dictnames
        return df
