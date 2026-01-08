#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2019/12/04 11:16


""" 区间类

关于区间类，可以参考： https://github.com/AlexandreDecan/python-intervals
但其跟我的业务场景有区别，不太适用，所以这里还是开发了自己的功能库

文档： https://histudy.yuque.com/docs/share/365f3a75-28d0-4595-bc80-5e9d6ab36f71#
"""

import collections
import itertools
import math
import re


class Interval:
    """
    这个类要考虑跟正则的 Match 对象兼容，所以有些细节比较“诡异”
        主要是区间的记录，是用了特殊的 regs 格式
        即虽然说是一个区间，但这个区间可以标记许多子区间
        对应正则里的group(0)，和众多具体的子区间group(1)、group(2)等
        正则里是用regs存储这些区间，所以这里的Interval跟正则Match的regs概念相同
    这里的形式统一为左闭右开
    """
    __slots__ = ('regs',)

    def __init__(self, arg1=None, arg2=None):
        if isinstance(arg1, int) and isinstance(arg2, int):
            # 正常的构建方式
            self.regs = ((arg1, arg2),)
        elif getattr(arg1, 'regs', None):
            # 有regs成员变量，则直接取用（一般是re的Match对象传过来转区间类的）
            self.regs = arg1.regs
        elif arg2 is None and arg1 and len(arg1) == 2 and isinstance(arg1[0], int):
            self.regs = (tuple(arg1),)
        elif arg1:
            # 直接传入区间集
            self.regs = tuple(arg1)
        else:
            # 空区间
            self.regs = None
        self.update()

    def update(self):
        """ 将空区间统一标记为None

        >>> Interval(5, 5).regs  # 返回 None
        >>> Interval(6, 5).regs
        """
        if self.regs and self.regs[0][0] >= self.regs[0][1]:
            self.regs = None

    def start(self, idx=0):
        return self.regs[idx][0] if self.regs else math.inf

    def end(self, idx=0):
        return self.regs[idx][1] if self.regs else -math.inf

    def __bool__(self):
        """
        >>> bool(Interval(5, 6))
        True
        >>> bool(Interval())
        False
        >>> bool(Interval(5, 5))  # 由于标记是左闭右开
        False
        >>> bool(Interval(6, 5))
        False
        """
        if self.regs:  # self.regs存在还不够，区间必须要有长度
            return self.end() > self.start()
        else:
            return False

    def __repr__(self):
        """ 所有区间都是左闭右开！ 第一组(x~y)是主区间，后面跟的是子区间

        >>> Interval()  # 空区间
        []
        >>> Interval(6, 5)  # 空区间
        []
        >>> Interval(4, 8)  # 会将底层的左闭右开区间值，改成左闭右闭的区间值显示，更直观
        [4~7]
        >>> Interval(5, 6)  # 只有单个值的，就不写5~5了，而是简洁性写成5
        [5]
        >>> Interval(((4, 8), (4, 6), (7, 8)))  # 如果有子区间，会在主区间冒号后显示
        [4~7: 4~5 7]
        """
        li = []
        if self.regs:
            li = [(f'{a}~{b - 1}' if b - a > 1 else str(a)) for a, b in self.regs]
            if len(li) > 1: li[0] += ':'
        return '[' + ' '.join(li) + ']'

    def __eq__(self, other):
        """只要主区间标记范围一致就是相等的
        >>> Interval(5, 7) == Interval([(5, 7), (4, 6)])
        True
        >>> Interval(5, 5) == Interval()
        True
        >>> Interval(5, 6) == Interval(5, 7)
        False
        """
        a = self.regs and self.regs[0]
        b = other.regs and other.regs[0]
        return a == b

    def __lt__(self, other):
        """两个区间比大小，只考虑regs[0]，先比第1个值，如果相同再比第2个值

        >>> Interval(4, 6) < Interval(7, 8)
        True
        >>> Interval(4, 5) < Interval(4, 6)
        True
        >>> Interval(4, 6) < Interval(5, 6)
        True
        >>> Interval(4, 6) > Interval(7, 8)  # 虽然只写了<，但python对>也能智能对称操作
        False
        """
        return (self.end() < other.end()) if self.start() == other.start() else (self.start() < other.start())

    def __and__(self, other):
        """
        >>> Interval(4, 7) & Interval(5, 8)
        [5~6]
        >>> Interval(4, 6) & Interval(5, 8)
        [5]
        >>> Interval(4, 6) & Interval(6, 8)
        []

        # 可以和区间集对象运算，会把区间集当成一系列的子区间，将其上下限范围作为主区间来分析
        #    注意和实际的线段点集求并的结果相区别：Intervals([(2, 4), [9, 11]]) & Interval(0, 10)
        >>> Interval(0, 10) & Intervals([(2, 4), [9, 11]])
        [2~9]
        """
        # 如果左右值不合理，类初始化里自带的update会自动变为None
        return Interval(max(self.start(), other.start()), min(self.end(), other.end()))

    def __contains__(self, other):
        """
        :param other: 另一个Interval对象
        :return: self.regs[0] 是否包含了 other.regs[0]

        >>> Interval(6, 8) in Interval(4, 10)
        True
        >>> Interval(2, 7) in Interval(4, 10)
        False
        >>> Intervals([(1,3), (2,4)]) in Interval(1, 4)  # 可以和Intervals混合使用
        True
        >>> Interval(2, 7) not in Interval(4, 10)
        True
        """
        return self.start() <= other.start() and self.end() >= other.end()

    def __or__(self, other):
        """两个regs[0]共有的部分
        :param other: 另一个Interval对象 或 Intervals 区间集对象
        :return: 返回other对象的类型（注意，会丢失所有子区间！），
            如果不存在则regs的值为None
        >>> m1, m2, m3 = Interval(4, 6), Interval(5, 8), Interval(10, 15)
        >>> m1 | m2  # 注意两个子区间会按顺序排
        [4~7: 4~5 5~7]
        >>> m1 | m3
        [4~14: 4~5 10~14]
        >>> m1 | Intervals([(2, 4), (7, 9)])
        [2~8: 2~8 4~5]
        """
        left, right = min(self.start(), other.start()), max(self.end(), other.end())
        a, b = sorted([self.regs[0], (other.start(), other.end())])  # 小的排左边
        return Interval(((left, right), a, b))

    def __add__(self, other):
        """
        >>> Interval(10, 15) + 6
        [16~20]
        """
        if isinstance(other, int):
            regs = [(t[0] + other, t[1] + other) for t in self.regs]
            return Interval(regs)
        else:
            return self | other

    def __sub__(self, other):
        """
        :param other: 另一个Interval对象 或者 Intervals对象
        :return: self.regs[0] 减去 other.regs[0] 后剩余的区间（会丢失子空间集）

        区间减区间：,
        >>> Interval(4, 6) - Interval(5, 8)
        [4]

        # 这种特殊情况会返回含有两个子区间的Interval对象
        >>> Interval(0, 10) - Interval(4, 6)
        [0~9: 0~3 6~9]

        # 这里后者实际区间值并不包含前者，
        #   但实际是按后者的start、end界定的范围作为一个Interval来减的
        >>> Interval(4, 7) - Intervals([(2, 3), (7, 8)])
        []
        """
        if isinstance(other, Intervals):
            other = Interval(other.start(), other.end())
        a, a1, a2 = self, self.start(), self.end()
        b, b1, b2 = other, other.start(), other.end()
        if a1 >= b2 or a2 <= b1:  # a 与 b 不相交
            # 这里不能直接返回a，如果a有子区间，会混淆return值类型
            return Interval(a1, a2)
        else:
            c1, c2 = Interval(a1, b1), Interval(b2, a2)
            if c1 and not c2:
                return c1
            elif c2 and not c1:
                return c2
            elif not c1 and not c2:
                return Interval()
            else:
                return Interval(((a1, a2), (c1.start(), c1.end()), (c2.start(), c2.end())))


class ReMatch(Interval):
    """
    1、伪re._sre.SRE_Match类
        真Match类在re.py的第223行
        有什么办法嘞，标准Match不支持修改成员变量，不支持自定义spes
    2、这个类同时还可以作为“区间”类使用
        有配套的Intervals区间集类，有很刁的各种区间运算功能
    """
    __slots__ = ('regs', 'string', 'pos', 'endpos', 'lastindex', 'lastgroup', 're')

    def __init__(self, regs=None, string=None, pos=0, endpos=None, lastindex=None, lastgroup=None, re=None):
        """Create a new match object.

        :param regs: 区间值
        :param string: 原始的完整字符串内容
        :param pos: 匹配范围开始的位置，一般就是0
        :param endpos: 匹配范围的结束位置，一般就是原字符串长度
        :param lastindex: int，表示有多少个子分组
        :param lastgroup: NoneType，None，The name of the last matched capturing group,
            or None if the group didn’t have a name, or if no group was matched at all.
        :param re: 使用的原正则匹配模式
        """
        if getattr(regs, 'regs', None):
            # 从一个类match对象来初始化
            m = regs
            self.pos = getattr(m, 'pos', None)
            self.endpos = getattr(m, 'endpos', None)
            self.lastindex = getattr(m, 'lastindex', None)
            self.lastgroup = getattr(m, 'lastgroup', None)
            self.re = getattr(m, 're', None)
            self.string = getattr(m, 'string', None)
            self.regs = getattr(m, 'regs', None)
        else:
            self.regs = regs
            self.string = string
            self.pos = pos
            self.endpos = endpos
            self.lastindex = lastindex
            if not self.lastindex and len(self.regs) > 1: self.lastindex = len(self.regs) - 1
            self.lastgroup = lastgroup
            self.re = re
        self.update()

    def group(self, idx=0):
        return self.string[self.regs[idx][0]:self.regs[idx][1]]

    def expand(self, template):
        """Return the string obtained by doing backslash substitution on the
        template string template.

        好像是个输入'\1'可以返回匹配的第1组类似这样的功能

        :type template: T
        :rtype: T
        """
        raise NotImplementedError

    def groups(self, default=None):
        """Return a tuple containing all the subgroups of the match, from 1 up
        to however many groups are in the pattern.

        :rtype: tuple
        """
        return tuple(map(lambda x: self.string[x[0]:x[1]], self.regs[1:]))

    def groupdict(self, default=None):
        """Return a dictionary containing all the named subgroups of the match,
        keyed by the subgroup name.

        :rtype: dict[bytes | unicode, T]
        """
        raise NotImplementedError

    def span(self, group=0):
        """Return a 2-tuple (start, end) for the substring matched by group.

        :type group: int | bytes | unicode
        :rtype: (int, int)
        """
        return self.regs[group][0], self.regs[group][1]


class Intervals:

    def __init__(self, li=None):
        """
        :param li: 若干interval对象
        """
        # 1 matches支持list等类型初始化
        if hasattr(li, 'intervals'):
            li = li.intervals
        if isinstance(li, Intervals):
            self.__dict__ = li.__dict__
        else:
            self.li = []
            if li is None: li = []
            for m in li:
                if not isinstance(m, Interval):
                    m = Interval(m)
                if m: self.li.append(m)  # 只加入非空区间
            self.li.sort()  # 按顺序排
            # 2 生成成员变量
            # self._start = min([m.start() for m in self.li], default=math.inf)
            self._start = min([m.start() for m in self.li[:1]], default=math.inf)
            self._end = max([m.end() for m in self.li], default=-math.inf)

    def start(self):
        """和Interval操作方法尽量对称，头尾也用函数来取，不要用成员变量取"""
        return self._start

    def end(self):
        return self._end

    def _merge_intersect_interval(self, adjacent=True):
        if not self: return []
        li = [self[0]]
        for m in self[1:]:
            if li[-1].end() > m.start() or (adjacent and li[-1].end() == m.start()):
                li[-1] = m | li[-1]  # 如果跟上一个相交，则合并过去
            else:
                li.append(m)  # 否则新建一个区间
        return li

    def merge_intersect_interval(self, adjacent=False):
        """ 将存在相交的区域进行合并

        :param adjacent: 如果相邻紧接，也进行拼接

        # 注意(1,3)和(2,4)合并为一个区间了
        >>> Intervals([(1, 3), (2, 4), (5, 6)]).merge_intersect_interval(True)
        {[1~3: 1~2 2~3], [5]}
        >>> Intervals([(1, 2), (2, 3)]).merge_intersect_interval(True)
        {[1~2: 1 2]}
        >>> Intervals([(1, 2), (2, 3)]).merge_intersect_interval(adjacent=False)
        {[1], [2]}
        """
        # 因为可能经常被调用，所以要变成static存储
        if adjacent:
            self._li1 = getattr(self, '_li1', None)
            if self._li1 is None:
                self._li1 = Intervals(self._merge_intersect_interval(adjacent))
            return self._li1
        else:
            self._li2 = getattr(self, '_li2', None)
            if self._li2 is None:
                self._li2 = Intervals(self._merge_intersect_interval(adjacent))
            return self._li2

    def true_intersect_subinterval(self, other):
        """判断改区间集，与other区间集，是否存在相交的子区间（真相交，不含子集关系）

        如果一个区间只是self或other独有，或者一个子区间被另一方的一个子区间完全包含，是存在很多优良性质，方便做很多自动化的
        否则，如果存在相交的子区间，就麻烦了

        其实就是想把两个区间完全相同的子区间去掉，然后求交就好了~~

        >>> a, b = Intervals([(1,3), (7,9), (10,12)]), Intervals([(2,4), (7,9), (10, 15)])
        >>> a.true_intersect_subinterval(b)  # 在集合交的基础上，去掉完全包含的情况
        {[2]}
        """
        # 0 区间 转 区间集
        if isinstance(other, Interval):
            other = Intervals([other])

        # 1 区间集和区间集做相交运算，生成一个新的区间集
        """假设a、b都是从左到右按顺序取得，所以可以对b的遍历进行一定过滤简化"""
        A = self.merge_intersect_interval()
        B = other.merge_intersect_interval()
        li, k = [], 0
        for a in A:
            for j in range(k, len(B)):
                b = B[j]
                x1, y1, x2, y2 = a.start(), a.end(), b.start(), b.end()
                if y2 <= x1:
                    # B[0~j]都在a前面，在后续的a中，可以直接从B[j]开始找
                    k = j
                elif x2 >= y1:  # b已经到a右边，后面的b不用再找了，不会有相交
                    break
                elif (x2 < x1 < y2 < y1) or (x1 < x2 < y1 < y2):  # 严格相交，非子集关系
                    li.append(a & b)
        return Intervals(li)

    def sub(self, s, repl, *, out_repl=None, adjacent=False) -> str:
        r"""
        :param repl: 替换的规则函数
            暂不支持和正则等价的字符串替换规则表达
                这个得找技巧，用re现成的功能代码，不可能自己暴力解析
        :param out_repl: 对范围外若有处理需要，可以自定义处理函数
        :param s: 要处理的文本串
        原版 re.sub 还有 count 和 flags 参数，这里难开发，暂时先不做这个接口
        :return:

        >>> s = '0123456789'
        >>> inters = Intervals([(2, 5), (7, 8)])
        >>> inters.sub(s, lambda m: 'b')
        '01b56b89'
        >>> inters.sub(s, lambda m: 'b', out_repl=lambda m: 'a')
        'ababa'
        >>> inters.sub(s, 'b')
        '01b56b89'
        >>> inters.sub(s, 'b', out_repl='a')
        'ababa'
        >>> inters.sub(s, lambda m: ' ' + ''.join(reversed(m.group())) + ' ')
        '01 432 56 7 89'
        >>> inters.sub(s, lambda m: ' ' + ''.join(reversed(m.group())) + ' ', out_repl=lambda m: str(len(m.group())))
        '2 432 2 7 2'
        """
        res, idx = [], 0

        def str2func(a):
            # TODO，如果是str类型，应该要处理字符串标记中的编组和转义等信息的
            return (lambda s: a) if isinstance(a, str) else a

        repl, out_repl = str2func(repl), str2func(out_repl)

        def func1(regs):
            return repl(ReMatch(regs, s, 0, len(s)))  # 构造伪match类并传入

        def func2(start_, end_):
            if out_repl:
                return out_repl(ReMatch(((start_, end_),), s, 0, len(s)))
            else:
                return s[start_:end_]

        for inter in self.merge_intersect_interval(adjacent=adjacent):
            # 匹配范围外的文本处理
            if inter.start() > idx:
                res.append(func2(idx, inter.start()))
            # 匹配范围内的处理
            res.append(func1(inter.regs))
            idx = inter.end()
        if idx < len(s): res.append(func2(idx, len(s)))
        return ''.join(res)

    def replace(self, s, arg1, arg2=None, *, out_repl=lambda s: s, adjacent=False) -> str:
        r"""类似sub函数，但是对两个自定义函数传入的是普通字符串类型，而不是match对象

        :param arg1: 可以输入一个自定义函数
        :param arg2: 可以配合arg1使用，功能同str.replace(arg1, arg2)
        :param adjacent: 替换的时候，为了避免混乱出错，是先要合并重叠的区间集的
            这里有个adjacent参数，True表示邻接的区间会合并，反之则不会合并临接区间

        >>> s = '0123456789'
        >>> inters = Intervals([(2, 5), (7, 8)])
        >>> inters.replace(s, lambda s: 'b')
        '01b56b89'
        >>> inters.replace(s, lambda s: 'b', out_repl=lambda s: 'a')
        'ababa'
        >>> inters.replace(s, 'b')
        '01b56b89'
        >>> inters.replace(s, '2', 'b')
        '01b3456789'
        >>> inters.replace(s, lambda s: ' ' + ''.join(reversed(s)) + ' ', out_repl=lambda s: str(len(s)))
        '2 432 2 7 2'
        """
        res, idx = [], 0

        def str2func(a):
            return (lambda s: a) if isinstance(a, str) else a

        repl, out_repl = str2func(arg1), str2func(out_repl)
        if arg2:
            repl = lambda a: a.replace(arg1, arg2)

        for inter in self.merge_intersect_interval(adjacent=adjacent):
            # 匹配范围外的文本处理
            if inter.start() >= idx:
                res.append(out_repl(s[idx:inter.start()]))
                idx = inter.end()
            # 匹配范围内的处理
            res.append(repl(s[inter.start():inter.end()]))
        if idx < len(s): res.append(out_repl(s[idx:]))
        return ''.join(res)

    def __bool__(self):
        """
        >>> bool(Intervals())
        False
        >>> bool(Intervals([(1, 2), (4, 5)]))
        True
        >>> bool(Intervals([(2, 1), (5, 4)]))
        False
        """
        return bool(self.li)

    def __getitem__(self, item):
        return self.li[item]

    def __iter__(self):
        for m in self.li:
            yield m

    def __len__(self):
        return len(self.li)

    def __repr__(self):
        return '{' + ', '.join([str(m) for m in self.li]) + '}'

    def __eq__(self, other):
        """"数量相等，且每个Interval也相等

        即只考虑强相等，不考虑“弱相等”。
        例如两个区间集虽然数量不同，但使用merge_intersect_interval后，再比较可能就是一样的。

        >>> Intervals([(1,2), (3,5)]) == Intervals([(1,2), (3,5)])
        True
        >>> Intervals([(1,2), (3,5)]) == Intervals([(1,2), (3,4), (4,5)])
        False
        >>> Intervals([(1,2), (3,5)]) == Intervals([(1,2), (3,4), (4,5)]).merge_intersect_interval(True)
        True
        """
        if len(self) != len(other): return False
        for i in range(len(self)):
            if self[i] != other[i]:
                return False
        return True

    def __invert__(self, maxn=None):
        """取反区间集的补集
        注意这样会丢失所有区间的子区间标记

        >>> ~Intervals([(1, 3), (4, 6), (8, 10)])  # 区间取反操作
        {[0], [3], [6~7]}
        >>> ~Intervals([])  # 区间取反操作
        {}
        """
        # 1 要先把有相交的区间合并了
        itvs = self.merge_intersect_interval()

        # 2 辅助变量
        li = []
        if maxn is None: maxn = itvs.end()  # 计算出坐标上限

        # 3 第1个区间是否从0开始
        if len(itvs) and itvs[0].start() == 0:
            idx = itvs[0].end()
            i = 1
        else:
            i = idx = 0

        # 4 循环取得新的区间值
        for m in itvs[i:]:
            li.append(Interval(idx, m.start()))
            idx = m.end()

        # 5 最后一个区间特判
        if idx != maxn: li.append(Interval(idx, maxn))
        res = Intervals(li)
        return res

    def invert(self, maxn=None):
        """
        >>> Intervals([(1, 3), (4, 6), (8, 10)]).invert(20)
        {[0], [3], [6~7], [10~19]}
        """
        return self.__invert__(maxn)

    def __and__(self, other):
        r"""
        # 区间集和单个区间的相交运算：
        >>> Intervals([(2, 4), (9, 11)]) & Interval(0, 10)
        {[2~3], [9]}

        # 区间集和区间集的相交运算：
        >>> Intervals([(1, 5), (6, 8)]) & Intervals([(2, 7), (7, 9)])
        {[2~4], [6], [7]}

        >>> Intervals([(2, 11)]) & Intervals()
        {}
        >>> Intervals() & Intervals([(2, 11)])
        {}
        """
        # 0 区间 转 区间集
        if isinstance(other, Interval):
            other = Intervals([other])

        # 1 区间集和区间集做相交运算，生成一个新的区间集
        """假设a、b都是从左到右按顺序取得，所以可以对b的遍历进行一定过滤简化"""
        A = self.merge_intersect_interval()
        B = other.merge_intersect_interval()
        li, k = [], 0
        for a in A:
            for j in range(k, len(B)):
                b = B[j]
                if b.end() <= a.start():
                    # B[0~j]都在a前面，在后续的a中，可以直接从B[j]开始找
                    k = j
                elif b.start() >= a.end():  # b已经到a右边，后面的b不用再找了，不会有相交
                    break
                else:  # 可能有相交
                    li.append(a & b)
        return Intervals(li)

    def is_adjacent_and(self, other):
        """ __and__运算的变形，两区间邻接时也认为相交

        >>> Intervals([(2, 4), (9, 11)]).is_adjacent_and(Interval(0, 10))
        True
        >>> Intervals([(1, 5), (6, 8)]).is_adjacent_and(Intervals([(2, 7), (7, 9)]))
        True
        >>> Intervals([(2, 11)]).is_adjacent_and(Intervals())
        False
        >>> Intervals().is_adjacent_and(Intervals([(2, 11)]))
        False
        >>> Intervals([(2, 11)]).is_adjacent_and(Interval(11, 13))
        True
        """
        # 0 区间 转 区间集
        if isinstance(other, Interval):
            other = Intervals([other])

        # 1 区间集和区间集做相交运算，生成一个新的区间集
        """假设a、b都是从左到右按顺序取得，所以可以对b的遍历进行一定过滤简化"""
        A = self.merge_intersect_interval()
        B = other.merge_intersect_interval()
        li, k = [], 0
        for a in A:
            for j in range(k, len(B)):
                b = B[j]
                if b.end() < a.start():
                    # B[0~j]都在a前面，在后续的a中，可以直接从B[j]开始找
                    k = j
                elif b.start() > a.end():  # b已经到a右边，后面的b不用再找了，不会有相交
                    break
                else:  # 可能有相交
                    return True
        return False

    def __contains__(self, other):
        r"""
        >>> Interval(3, 5) in Intervals([(2, 6)])
        True
        >>> Interval(3, 5) in Intervals([(0, 4)])
        False
        >>> Intervals([(1, 2), (3, 4)]) in Intervals([(0, 3), (3, 5)])
        True
        >>> Interval(3, 5) not in Intervals([(2, 6)])
        False

        这里具体实现，可以双循环暴力，但考虑区间集顺序性，其实只要双指针同时往前找就好了，
            设几个条件去对循环进行优化，跳出，能大大提高效率
        """
        # 1 区间集 是否包含 区间，转为 区间集 是否包含 区间集 处理
        if isinstance(other, (Interval, list, tuple)):
            other = Intervals([other])

        # 2 合并相交区域
        A = self.merge_intersect_interval()
        B = other.merge_intersect_interval()

        # 3 看是否每个b，都能在A中找到一个a包含它
        i = 0
        for b in B:
            for j in range(i, len(A)):
                if b in A[j]:
                    # A[j-1]前面都不能包含b，但是A[j]能包含b的，后面的b，A[j-1]也一定包含不到
                    i = j
                    break
                elif A[j].start() > b.end():
                    # 后续的a左边都已经比b的右边更大，就不用再找了，肯定都不会有相交的了
                    return False
            else:  # 找到一个b，在A中不包含它
                return False
        return True

    def __or__(self, other):
        r"""区间集相加运算，合成一个新的区间集对象（会丢失所有子区间）

        出现相交的元素会合成一个新的元素，避免区间集中存在相交的两个元素，在sub时会出bug
        >>> Intervals([(2, 4), (5, 7)]) | Interval(1, 3)
        {[1~3: 1~2 2~3], [5~6]}
        >>> Intervals([(2, 4), (5, 7)]) | Intervals([(1, 3), (6, 9)])
        {[1~3: 1~2 2~3], [5~8: 5~6 6~8]}

        >>> Intervals([(1, 3), (6, 9)]) + 3
        {[4~5], [9~11]}
        """
        if isinstance(other, Interval):
            other = Intervals([other])
        else:
            other = Intervals(other)
        return Intervals(self.li + other.li).merge_intersect_interval()

    def __add__(self, other):
        if isinstance(other, int):
            li = [x + other for x in self.li]
            return Intervals(li)
        else:
            return self | other

    def __sub__(self, other):
        """区间集减法操作（注意跟Interval减法操作是有区别的）
        对于任意的 a ∈ self，更新 a = a - {b | b ∈ other}

        >>> Intervals([(0, 10)]) - Interval(4, 6)
        {[0~3], [6~9]}
        >>> Intervals([(0, 10)]) - Interval(8, 12)
        {[0~7]}

        >>> Intervals([(0, 10), (20, 30)]) - Intervals([(0, 5), (15, 25)])
        {[5~9], [25~29]}
        >>> Intervals([(0, 10), (20, 30)]) - Intervals([(2, 5), (7, 12), (25, 27)])
        {[0~1], [5~6], [20~24], [27~29]}
        """
        # 1 
        if isinstance(other, Interval):
            other = Intervals([other])

        # a - b，一个a可能会拆成a1,a2两段，此时左边的a1可以继续处理，
        #   但是a2要加到堆栈A，留作下一轮处理，所以A要用栈结构处理
        A = list(reversed(self.merge_intersect_interval().li))

        B = other.merge_intersect_interval()

        # 2 
        li, k = [], 0
        while A:
            a = A.pop()
            for j in range(k, len(B)):
                b = B[j]
                if b.end() < a.start():
                    k = j
                elif a.end() < a.start():
                    break
                else:
                    c = a - b
                    if not c:  # a已经被减光，就直接跳出循环了
                        a = Interval()
                        break
                    elif len(c.regs) == 1:
                        a = c
                    else:  # 如果 a - c 变成了两段，则左边a1继续处理，右边a2加入A下轮处理
                        a = Interval(c.regs[1])
                        A.append(Interval(c.regs[2]))
            if a: li.append(a)

        return Intervals(li).merge_intersect_interval()


def iter_intervals(arg):
    """从多种类区间类型来构造Interval对象，返回值可能有多组"""

    def judge_range(t):
        return hasattr(t, '__len__') and len(t) == 2 and isinstance(t[0], int) and isinstance(t[1], int)

    if hasattr(arg, 'regs'):
        yield Interval(arg)
    elif judge_range(arg):
        yield Interval(arg)
    elif isinstance(arg, Interval):
        yield arg
    elif isinstance(arg, Intervals):
        for i in range(len(arg)):
            yield arg[i]
    elif hasattr(arg, '__len__') and len(arg) and judge_range(arg[0]):
        for i in range(len(arg)):
            yield Interval(arg[i])
    elif isinstance(arg, collections.Iterable):
        for t in list(arg):
            yield t


def highlight_intervals(content, intervals, colors=None, background=True,
                        use_mathjax=False,
                        only_body=False,
                        title='highlight_intervals',
                        set_pre='<pre class="prettyprint nocode linenums" style="white-space: pre-wrap;">'):
    """文本匹配可视化
    获得高亮显示的匹配区间的html代码

    :param content：需要展示的文本内容
    :param intervals: 输入一个数组，数组的每个元素支持单区间或区间集相关类
        Interval、re正则的Match对象、(4, 10)
        Intervals、[(2,4), (6,8)]

        请自行保证区间嵌套语法正确性，本函数不检查处理嵌套混乱错误问题
    :param set_pre: 设置<pre>显示格式。
        标准 不自动换行： '<pre class="prettyprint nocode linenums">'
        比如常见的，对于太长的文本行，可以自动断行：
            set_pre='<pre class="prettyprint nocode linenums" style="white-space: pre-wrap;">'
    :param colors: 一个数组，和intervals一一对应，轮询使用的颜色
        默认值为： ['red']
    :param background:
        True，使用背景色
        False，不使用背景色，而是字体颜色
    :param use_mathjax:
        True，渲染公式
        False，不渲染公式，只以文本展示
    :param only_body: 不返回完整的html页面内容，只有body主体内容
    """
    # 1 存储要插入的html样式
    from collections import defaultdict
    import html
    from pyxllib.text.xmllib import get_jinja_template

    d = defaultdict(str)

    # 2 其他所有子组从颜色列表取颜色清单，每组一个颜色
    if colors is None:
        colors = ('red',)
    elif isinstance(colors, str):
        colors = (colors,)
    n = len(colors)
    for i, arg in enumerate(intervals):
        color = colors[i % n]
        for interval in iter_intervals(arg):
            l, r = interval.start(), interval.end()
            if background:
                d[l] = d[l] + f'<span style="background-color: {color}">'
                d[r] = '</span>' + d[r]
            else:
                d[l] = d[l] + f'<font color={color}>'
                d[r] = '</font>' + d[r]

    # 3 拼接最终的html代码
    res = [set_pre]
    s = content
    idxs = sorted(d.keys())  # 按顺序取需要插入的下标

    # （3）拼接
    if idxs: res.append(s[:idxs[0]])
    for i in range(1, len(idxs)):
        res.append(d[idxs[i - 1]])
        res.append(html.escape(s[idxs[i - 1]:idxs[i]]))
    if idxs:  # 最后一个标记
        res.append(d[idxs[-1]])
        res.append(s[idxs[-1]:])
    if not idxs:
        res.append(s)
    res.append('</pre>')

    if only_body:
        return ''.join(res)
    else:
        return get_jinja_template('highlight_code.html').render(title=title, body=''.join(res), use_mathjax=use_mathjax)


class StrIdxBack:
    r"""字符串删除部分干扰字符后，对新字符串匹配并回溯找原字符串的下标

    >>> ob = StrIdxBack('bxx  ax xbxax')
    >>> ob.delchars(r'[ x]+')
    >>> ob  # 删除空格、删除字符x
    baba
    >>> print(ob.idx)  # keystr中与原字符串对应位置：(0, 5, 9, 11)
    (0, 5, 9, 11)
    >>> m = re.match(r'b(ab)', ob.keystr)
    >>> m = ob.matchback(m)
    >>> m.group(1)
    'ax xb'
    >>> ob.search('ab')  # 找出原字符串中内容：'ax xb'
    'ax xb'
    """

    def __init__(self, s):
        self.oristr = s
        self.idx = tuple(range(len(s)))  # 存储还保留着内容的下标
        self.keystr = s

    def delchars(self, pattern, flags=0):
        r""" 模仿正则的替换语法
        但是不用输入替换目标s，以及目标格式，因为都是删除操作

        利用正则可以知道被删除的是哪个区间范围
        >>> ob = StrIdxBack('abc123df4a'); ob.delchars(r'\d+'); str(ob)
        'abcdfa'
        >>> ob.idx
        (0, 1, 2, 6, 7, 9)
        """
        k = 0
        idxs = []

        def repl(m):
            nonlocal k, idxs
            idxs.append(self.idx[k:m.start(0)])
            k = m.end(0)
            return ''

        self.keystr = re.sub(pattern, repl, self.keystr, flags=flags)
        idxs.append(self.idx[k:])
        self.idx = tuple(itertools.chain(*idxs))

    def compare_newstr(self, limit=300):
        r"""比较直观的比较字符串前后变化

        newstr相对于oldnew作展开，比较直观的显示字符串前后变化差异
        >>> ob = StrIdxBack('abab'); ob.delchars('b'); ob.compare_newstr()
        'a a '
        """
        s1 = self.oristr
        dd = set(self.idx)

        s2 = []
        k = 0
        for i in range(min(len(s1), limit)):
            if i in dd:
                s2.append(s1[i])
                k += 1
            else:
                if ord(s1[i]) < 128:
                    if s1[i] == ' ':  # 原来是空格的，删除后要用_表示
                        s2.append('_')
                    else:  # 原始不是空格的，可以用空格表示已被删除
                        s2.append(' ')
                else:  # 中文字符要用两个空格表示才能对齐
                    s2.append('  ')
        s2 = ''.join(s2)
        s2 = s2.replace('\n', r'\n')

        return s2

    def compare(self, limit=300):
        """比较直观的比较字符串前后变化"""
        s1 = self.oristr

        s1 = s1.replace('\n', r'\n')[:limit]
        s2 = self.compare_newstr(limit)

        return s1 + '\n' + s2 + '\n'

    def matchback(self, m):
        """输入一个keystr匹配的match对象，将其映射回oristr的match对象"""
        regs = []
        for rs in getattr(m, 'regs'):
            regs.append((self.idx[rs[0]], self.idx[rs[1] - 1] + 1))  # 注意右边界的处理有细节
        return ReMatch(regs, self.oristr, m.pos, len(self.oristr), m.lastindex, m.lastgroup, m.re)

    def search(self, pattern):
        """在新字符串上查找模式，但是返回的是原字符串的相关下标数据"""
        m = re.search(pattern, self.keystr)
        if m:
            m = self.matchback(m)  # pycharm这里会提示m没有regs的成员变量，其实是正常的，没问题
            return m.group()
        else:
            return ''

    def __repr__(self):
        """返回处理后当前的新字符串"""
        return self.keystr
