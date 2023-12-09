#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 14:22

import textwrap
from collections import defaultdict, Counter
import math
import re
import sys

from pyxllib.prog.newbie import typename
from pyxllib.text.pupil import listalign, int2myalphaenum


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


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


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


class ValuesStat:
    """ 一串数值的相关统计分析 """

    def __init__(self, values):
        from statistics import pstdev, mean
        self.values = values
        self.n = len(values)
        self.sum = sum(values)
        if self.n:
            self.mean = mean(self.values)
            self.std = pstdev(self.values)
            self.min, self.max = min(values), max(values)
        else:
            self.mean = self.std = self.min = self.max = float('nan')

    def __len__(self):
        return self.n

    def summary(self, valfmt='g'):
        """ 输出性能分析报告，data是每次运行得到的时间数组

        :param valfmt: 数值显示的格式
            g是比较智能的一种模式
            也可以用 '.3f'表示保留3位小数
            可以是一个函数，该函数接收一个数值作为输入，返回格式化后的字符串
            注意可以写None表示删除特定位的显示

            也可以传入长度5的格式清单，表示 [和、均值、标准差、最小值、最大值] 一次展示的格式
        """
        if isinstance(valfmt, str) or callable(valfmt):
            valfmt = [valfmt] * 5

        ls = []

        def format_value(value, format_spec):
            """ 根据指定的格式来格式化值 """
            if callable(format_spec):
                return format_spec(value)
            else:
                return f"{value:{format_spec}}"

        if self.n > 1:
            ls.append(f'总数: {self.n}')  # 注意输出其实完整是6个值，还有个总数不用控制格式
            if valfmt[0]:
                ls.append(f'总和: {format_value(self.sum, valfmt[0])}')
            if valfmt[1] or valfmt[2]:
                mean_str = format_value(self.mean, valfmt[1]) if valfmt[1] else ''
                std_str = format_value(self.std, valfmt[2]) if valfmt[2] else ''
                if mean_str and std_str:
                    ls.append(f'均值标准差: {mean_str}±{std_str}')
                elif mean_str:
                    ls.append(f'均值: {mean_str}')
                elif std_str:
                    ls.append(f'标准差: {std_str}')
            if valfmt[3]:
                ls.append(f'最小值: {format_value(self.min, valfmt[3])}')
            if valfmt[4]:
                ls.append(f'最大值: {format_value(self.max, valfmt[4])}')
            return '\t'.join(ls)
        elif self.n == 1:
            return format_value(self.sum, valfmt[0] or 'g')
        else:
            raise ValueError("无效的数据数量")


class Groups:
    def __init__(self, data):
        """ 分组

        :param data: 输入字典结构直接赋值
            或者其他结构，会自动按相同项聚合

        TODO 显示一些数值统计信息，甚至图表
        TODO 转文本表达，方便bc比较
        """
        if not isinstance(data, dict):
            new_data = dict()
            # 否要要转字典类型，自动从1~n编组
            for k, v in enumerate(data, start=1):
                new_data[k] = v
            data = new_data
        self.data = data  # 字典存原数据
        self.ctr = Counter({k: len(x) for k, x in self.data.items()})  # 计数
        self.stat = ValuesStat(self.ctr.values())  # 综合统计数据

    def __repr__(self):
        ls = []
        for i, (k, v) in enumerate(self.data.items(), start=1):
            ls.append(f'{i}, {k}：{v}')
        return '\n'.join(ls)

    @classmethod
    def groupby(cls, ls, key, ykey=None):
        """
        :param ls: 可迭代等数组类型
        :param key: 映射规则，ls中每个元素都会被归到映射的key组上
            Callable[Any, 不可变类型]
            None，未输入时，默认输入的ls已经是分好组的数据
        :param ykey: 是否对分组后存储的内容y，也做一个函数映射
        :return: dict
        """
        data = defaultdict(list)
        for x in ls:
            k = key(x)
            if ykey:
                x = ykey(x)
            data[k].append(x)
        return cls(data)


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


def matchpairs(xs, ys, cmp_func, least_score=sys.float_info.epsilon, *,
               key=None, index=False):
    r""" 匹配两组数据

    :param xs: 第一组数据
    :param ys: 第二组数据
    :param cmp_func: 所用的比较函数，值越大表示两个对象相似度越高
    :param least_score: 允许匹配的最低分，默认必须要大于0
    :param key: 是否需要对xs, ys进行映射后再传入 cmp_func 操作
    :param index: 返回的不是原值，而是下标
    :return: 返回结构[(x1, y1, score1), (x2, y2, score2), ...]，注意长度肯定不会超过min(len(xs), len(ys))

    注意：这里的功能①不支持重复匹配，②任何一个x,y都有可能没有匹配到
        如果每个x必须都要有一个匹配，或者支持重复配对，请到隔壁使用 MatchPairs

    TODO 这里很多中间步骤结果都是很有分析价值的，能改成类，然后支持分析中间结果？
    TODO 这样全量两两比较是很耗性能的，可以加个参数草算，不用精确计算的功能？

    >>> xs, ys = [4, 6, 1, 2, 9, 4, 5], [1, 5, 8, 9, 2]
    >>> cmp_func = lambda x,y: 1-abs(x-y)/max(x,y)
    >>> matchpairs(xs, ys, cmp_func)
    [(1, 1, 1.0), (2, 2, 1.0), (9, 9, 1.0), (5, 5, 1.0), (6, 8, 0.75)]
    >>> matchpairs(ys, xs, cmp_func)
    [(1, 1, 1.0), (5, 5, 1.0), (9, 9, 1.0), (2, 2, 1.0), (8, 6, 0.75)]
    >>> matchpairs(xs, ys, cmp_func, 0.9)
    [(1, 1, 1.0), (2, 2, 1.0), (9, 9, 1.0), (5, 5, 1.0)]
    >>> matchpairs(xs, ys, cmp_func, 0.9, index=True)
    [(2, 0, 1.0), (3, 4, 1.0), (4, 3, 1.0), (6, 1, 1.0)]
    """
    # 0 实际计算使用的是 xs_, ys_
    if key:
        xs_ = [key(x) for x in xs]
        ys_ = [key(y) for y in ys]
    else:
        xs_, ys_ = xs, ys

    # 1 计算所有两两相似度
    n, m = len(xs), len(ys)
    all_pairs = []
    for i in range(n):
        for j in range(m):
            score = cmp_func(xs_[i], ys_[j])
            if score >= least_score:
                all_pairs.append([i, j, score])
    # 按分数权重排序，如果分数有很多相似并列，就只能按先来后到排序啦
    all_pairs = sorted(all_pairs, key=lambda v: (-v[2], v[0], v[1]))

    # 2 过滤出最终结果
    pairs = []
    x_used, y_used = set(), set()
    for p in all_pairs:
        i, j, score = p
        if i not in x_used and j not in y_used:
            if index:
                pairs.append((i, j, score))
            else:
                pairs.append((xs[i], ys[j], score))
            x_used.add(i)
            y_used.add(j)

    return pairs


def get_number_width(n):
    """ 判断数值n的长度

    >>> get_number_width(0)
    Traceback (most recent call last):
    AssertionError
    >>> get_number_width(9)
    1
    >>> get_number_width(10)
    2
    >>> get_number_width(97)
    2
    """
    assert n > 0
    return math.ceil(math.log10(n + 1))


class SearchBase:
    """ 一个dfs、bfs模板类 """

    def __init__(self, root):
        """
        Args:
            root: 根节点
        """
        self.root = root

    def get_neighbors(self, node):
        """ 获得邻接节点，必须要用yield实现，方便同时支持dfs、bfs的使用

        对于树结构而言，相当于获取直接子结点

        这里默认是bs4中Tag规则；不同业务需求，可以重定义该函数
        例如对图结构、board类型，可以在self存储图访问状态，在这里实现遍历四周的功能
        """
        try:
            for node in node.children:
                yield node
        except AttributeError:
            pass

    def dfs_nodes(self, node=None, depth=0):
        """ 返回深度优先搜索得到的结点清单

        :param node: 起始结点，默认是root根节点
        :param depth: 当前node深度
        :return: list，[(node1, depth1), (node2, depth2), ...]
        """
        if not node:
            node = self.root

        ls = [(node, depth)]
        for t in self.get_neighbors(node):
            ls += self.dfs_nodes(t, depth + 1)
        return ls

    def bfs_nodes(self, node=None, depth=0):
        if not node:
            node = self.root

        ls = [(node, depth)]
        i = 0

        while i < len(ls):
            x, d = ls[i]
            nodes = self.get_neighbors(x)
            ls += [(t, d + 1) for t in nodes]
            i += 1

        return ls

    def fmt_node(self, node, depth, *, prefix='    ', show_node_type=False):
        """ node格式化显示 """
        s1 = prefix * depth
        s2 = typename(node) + '，' if show_node_type else ''
        s3 = textwrap.shorten(str(node), 200)
        return s1 + s2 + s3

    def fmt_nodes(self, *, nodes=None, select_depth=None, linenum=False,
                  msghead=True, show_node_type=False, prefix='    '):
        """ 结点清单格式化输出

        :param nodes: 默认用dfs获得结点，也可以手动指定结点
        :param prefix: 缩进格式，默认用4个空格
        :param select_depth: 要显示的深度
            单个数字：获得指定层
            Sequences： 两个整数，取出这个闭区间内的层级内容
        :param linenum：节点从1开始编号
            行号后面，默认会跟一个类似Excel列名的字母，表示层级深度
        :param msghead: 第1行输出一些统计信息
        :param show_node_type:

        Requires
            textwrap：用到shorten
            align.listalign：生成列编号时对齐
        """
        # 1 生成结点清单
        ls = nodes if nodes else self.dfs_nodes()
        total_node = len(ls)
        total_depth = max(map(lambda x: x[1], ls))
        head = f'总节点数：1~{total_node}，总深度：0~{total_depth}'

        # 2 过滤与重新整理ls（select_depth）
        logo = True
        cnt = 0
        tree_num = 0
        if isinstance(select_depth, int):

            for i in range(total_node):
                if ls[i][1] == select_depth:
                    ls[i][1] = 0
                    cnt += 1
                    logo = True
                elif ls[i][1] < select_depth and logo:  # 遇到第1个父节点添加一个空行
                    ls[i] = ''
                    tree_num += 1
                    logo = False
                else:  # 删除该节点，不做任何显示
                    ls[i] = None
            head += f'；挑选出的节点数：{cnt}，所选深度：{select_depth}，树数量：{tree_num}'

        elif hasattr(select_depth, '__getitem__'):
            for i in range(total_node):
                if select_depth[0] <= ls[i][1] <= select_depth[1]:
                    ls[i][1] -= select_depth[0]
                    cnt += 1
                    logo = True
                elif ls[i][1] < select_depth[0] and logo:  # 遇到第1个父节点添加一个空行
                    ls[i] = ''
                    tree_num += 1
                    logo = False
                else:  # 删除该节点，不做任何显示
                    ls[i] = None
            head += f'；挑选出的节点数：{cnt}，所选深度：{select_depth[0]}~{select_depth[1]}，树数量：{tree_num}'
        """注意此时ls[i]的状态，有3种类型
            (node, depth)：tuple类型，第0个元素是node对象，第1个元素是该元素所处层级
            None：已删除元素，但为了后续编号方便，没有真正的移出，而是用None作为标记
            ''：已删除元素，但这里涉及父节点的删除，建议此处留一个空行
        """

        # 3 格式处理
        def mystr(item):
            return self.fmt_node(item[0], item[1], prefix=prefix, show_node_type=show_node_type)

        line_num = listalign(range(1, total_node + 1))
        res = []
        for i in range(total_node):
            if ls[i] is not None:
                if isinstance(ls[i], str):  # 已经指定该行要显示什么
                    res.append(ls[i])
                else:
                    if linenum:  # 增加了一个能显示层级的int2excel_col_name
                        res.append(line_num[i] + int2myalphaenum(ls[i][1]) + ' ' + mystr(ls[i]))
                    else:
                        res.append(mystr(ls[i]))

        s = '\n'.join(res)

        # 是否要添加信息头
        if msghead:
            s = head + '\n' + s

        return s
