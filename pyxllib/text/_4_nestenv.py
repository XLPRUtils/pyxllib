#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : chenkz@histudy.com
# @Date   : 2019/02/20 10:03


import bisect

from pyxllib.text._1_base import *
from pyxllib.text._3_interval import *


def pqmove(s, p, q):
    """在s[p:q]定位基础上，再类似strip去掉两边空白向内缩"""
    if p == q:
        return p, q
    while s[p] in ' \t':
        p += 1
    if s[p] == '\n':
        p += 1

    if q == -1: q = len(s)  # 没找到tail匹配，就以字符串末尾作为q
    while s[q - 1] in ' \t':
        q -= 1
    if s[q - 1] == '\n':
        q -= 1

    return p, q


def substr_intervals(s, head, tail=None, invert=False, inner=False):
    """ 旧模块，不推荐使用，建议使用新版的NestEnv接口直接处理

    :param s: 内容
    :param head: 头
        TODO 含正则和不含正则的，可以分子函数来实现，不要都写在这个函数
    :param tail: 尾
        TODO 支持普通字符串和正则对象的头尾搭配
    :param invert: 是否取反
    :param inner:  TODO 注意目前很多匹配功能还不支持inner模式
        False，定位内部时，含标签
        True，不含标签
    :return:

    TODO 考虑tabular嵌套tabular这种的正常定位？
    TODO 支持同时定位topic和sub_topic？
    """
    from pyxllib.text import NestEnv, LatexNestEnv

    def infer_headtail(head, tail=None):
        """输入简化的head、tail命令，返回智能推导出的完整的head、tail值"""
        if isinstance(head, str) and tail is None:
            if re.match(r'\$+$', head):  # 公式
                tail = head
            elif re.match(r'\\(chapter|section|subsection){', head):
                pass  # 这种情况 tail 不用改，就是用 None 来代表不确定性结尾标记
            elif head[-1] in '[{(<':  # 配对括号
                tail = {'[': ']', '{': '}', '(': ')', '<': '>'}[head[-1]]
            elif head.startswith('%<'):
                tail = '%/'
            elif head[0] == '<':
                tail = 'xmltag'
            elif re.match(r'\\begin{[a-zA-Z]+}', head):  # latex类的环境匹配
                m = re.match(r'\\begin({[a-zA-Z]+})', head)
                tail = r'\end' + m.group(1)
            else:  # 没有推导出来
                tail = None
        return head, tail

    head, tail = infer_headtail(head, tail)

    pos1, parts = 0, []
    # 1 括号匹配：head最后一个字符和tail第一个字符是匹配括号 # TODO 其实可以考虑tail的匹配括号不在头尾而在内容中间的情况
    if head[-1] in '[{(<' and tail and len(tail) and tail[0] == ']})>'['[{(<'.index(head[-1])]:
        parts = NestEnv(s).bracket(head, tail, inner).intervals
    # 2 第2种括号匹配: head第一个字符与tail最后一个字符是匹配括号
    elif head[0] in '[{(<' and tail and len(tail) and tail[-1] == ']})>'['[{(<'.index(head[0])]:
        parts = NestEnv(s).bracket2(head, tail, inner).intervals
    # 3 公式匹配
    elif head == tail == '$':
        parts = LatexNestEnv(s).formula(inner).intervals
    # 4 百分注结构 %<xxx a='yy'> ... %</xxx> 的格式匹配
    elif re.match(r'%<[a-zA-Z\-_]+', head) and tail == '%/':
        parts = LatexNestEnv(s).pxmltag(head[2:], 'inner').intervals
    # 5 latex的 章、节、子节 匹配
    elif re.match(r'\\(chapter|section|subsection)', head) and not tail:  # TODO 支持inner功能
        parts = LatexNestEnv(s).latexpart(head[1:], inner=inner)
    elif head == r'\item':
        parts = LatexNestEnv(s).item().intervals
    # 7 latex类的环境匹配
    elif re.match(r'\\begin{([a-zA-Z]+)}', head):
        m1 = re.match(r'\\begin{([a-zA-Z]+)}', head)
        m2 = re.match(r'\\end{([a-zA-Z]+)}', tail)
        if m2 and m1.group(1) == m2.group(1):
            parts = LatexNestEnv(s).latexenv(head, tail, inner).intervals
        else:
            parts = LatexNestEnv(s).find2(head, tail, inner).intervals
    # 8 抓取latex中所有插图命令
    elif head == r'\includegraphics' and tail is None:
        parts = LatexNestEnv(s).includegraphics('inner').intervals
    # 9 lewis电子式匹配
    elif head == r'\lewis' and tail is None:
        parts = LatexNestEnv(s).lewis(inner=inner).intervals
    # 10 xml标签结点匹配
    elif head[0] == '<' and tail == 'xmltag':
        parts = NestEnv(s).xmltag(head[1:], inner).intervals
    # +、普通匹配
    elif isinstance(head, str) and isinstance(tail, str):
        parts = NestEnv(s).find2(head, tail, inner).intervals
    elif isinstance(head, str) and not isinstance(tail, str):
        parts = NestEnv(s).find(head).intervals

    t = Intervals(parts)
    if invert: t = t.invert(len(s))
    return t


def substrfunc(s, head, tail, *, func1=lambda x: x, func2=lambda x: x):
    r"""对字符串s，查找里面的代码块
    代码块有head、tail组成，例如
        “$”，“$”：能扩展支持对双美元符的定位； TODO:使用$...$时，也能智能识别\(、\)
            目前也支持 r'$\begin{array}', r'\end{array}$' 的定位了
        第1种括号匹配：head最后一个字符与tail最后一个字符是匹配括号
            “\ce{”、“}”
            “\text{”、“}”
        第2种括号匹配：head第一个字符与tail最后一个字符是匹配括号
            '{\centerline', '}'
        “\includegraph”

    如果head的最后一个字符是：[{(、且tail的第一个字符是对应的]})，则会进行智能括号匹配
    反向选择=True时：对锁定外的区域进行操作

    对找到的每个子字符串，调用func1进行操作；对反向内容，调用func2进行操作

    >>> substrfunc('aa\\itemQ%\naabb\nccdd\n\n\\test\n{A}\n{B}\n\\item\nabc', r'\item', '', func1= lambda x: 'X' + x + 'Y')
    'aaX\\itemQ%\naabb\nccddY\n\nX\\test\n{A}\n{B}Y\nX\\item\nabcY'
    >>> substrfunc(r'aa\verb|bb|cc', r'\verb|', '|', func2 = lambda x: '')
    '\\verb|bb|'
    """
    intervals = substr_intervals(s, head, tail)
    return intervals.replace(s, func1, out_repl=func2)


class __NestEnvBase:
    __slots__ = ('s', 'intervals')

    def __init__(self, s, intervals=None):
        self.s = s
        if intervals is None: intervals = Intervals([[0, len(s)]])
        self.intervals = Intervals(intervals)

    def inner(self, head, tail=None):
        r""" 0、匹配标记里，不含head、tail标记

        >>> NestEnv(r'01\ce{H2O\ce{2}}01\ce{1\ce{3}5}').inner(r'\ce{').inner(r'\ce{').replace('x')
        '01\\ce{H2O\\ce{x}}01\\ce{1\\ce{x}5}'
        >>> NestEnv(r'01\ce{H2O\ce{2}}01\ce{1\ce{3}5}').inner(r'\cc{').string()
        >>> NestEnv(r'01\ce{H2O\ce{2}}01\ce{1\ce{3}5}').outside(r'\cc{').string()
        '01\\ce{H2O\\ce{2}}01\\ce{1\\ce{3}5}'

        TODO 注意 topic、analysis 这类定位 该函数目前还不支持，会有bug
        TODO 0的标记其实不好，不方便功能组合，1和2是互斥的，但是0和2不是互斥的，是可以组合的，即范围外含标签的内容，4会更合适，但现在改也挺别扭的，就先记录着，以后再看
        """
        li = []
        for reg in self.intervals:
            left, right = reg.start(), reg.end()
            li.extend(substr_intervals(self.s[left:right], head, tail, inner=True) + left)
        return NestEnv(self.s, Intervals(li))

    def inside(self, head, tail=None):
        r""" 1、匹配标记里

        >>> NestEnv(r'01\ce{H2O\ce{2}}01\ce{1\ce{3}5}').inside(r'\ce{').replace('x')
        '01x01x'
        """
        li = []
        for reg in self.intervals:
            left, right = reg.start(), reg.end()
            li.extend(substr_intervals(self.s[left:right], head, tail) + left)
        return NestEnv(self.s, Intervals(li))

    def outside(self, head, tail=None):
        r""" 2、匹配标记外

        >>> NestEnv(r'01\ce{H2O\ce{2}}01\ce{1\ce{3}5}').outside(r'\ce{').replace(lambda s: 'x')
        'x\\ce{H2O\\ce{2}}x\\ce{1\\ce{3}5}'
        """
        li = []
        for reg in self.intervals:
            left, right = reg.start(), reg.end()
            li.extend(substr_intervals(self.s[left:right], head, tail, invert=True) + left)
        return NestEnv(self.s, Intervals(li))

    def expand(self, ne):
        r""" 在现有区间上，判断是否有被其他区间包含，有则进行延展
        可以输入head、tail配对规则，也可以输入现成的区间

        >>> ne = NestEnv(r'aa$cc\ce{a}dd$bb\ce{d}h$h$')
        >>> ne.latexcmd1(r'ce').expand(ne.formula()).strings()
        ['$cc\\ce{a}dd$', '\\ce{d}']

        TODO 扩展临接也能延展的功能？
        """
        if isinstance(ne, NestEnv):
            b = ne.intervals
        elif isinstance(ne, Intervals):
            b = ne
        else:
            raise TypeError
        c = self.intervals + Intervals([x for x in b if (self.intervals & x)])
        return NestEnv(self.s, c)

    def filter(self, func):
        r""" 传入一个自定义函数func，会将每个区间的s传入，只保留func(s)为True的区间

        >>> NestEnv('aa$bbb$ccc$d$eee$fff$g').formula().filter(lambda s: len(s) > 4).strings()
        ['$bbb$', '$fff$']
        """
        li = list(filter(lambda x: func(self.s[x.start():x.end()]), self.intervals))
        return NestEnv(self.s, li)

    def _parse_tags(self, tags):
        if not isinstance(tags[0], (list, tuple)):
            # 旧单维数组输入，要先转成二维结构
            n = len(tags) // 3
            assert n and n * 3 == len(tags)
            tags = [tags[3 * i:3 * (i + 1)] for i in range(n)]
        return tags

    def any(self, tags):
        r""" 区间集求并

        :param tags: 同nestenv的tags参数规则

        >>> NestEnv(r'12$34$56\ce{78}90').any(['$', '$', 1, r'\ce{', '}', 1]).replace(lambda s: 'x')
        '12x56x90'
        """
        tags, li = self._parse_tags(tags), []
        for tag in tags:
            head, tail, t = tag
            for reg in self.intervals:
                left, right = reg.start(), reg.end()
                li.extend(substr_intervals(self.s[left:right], head, tail, invert=(t == 2), inner=(t == 0)) + left)
        return NestEnv(self.s, Intervals(li))

    def all(self, tags):
        r""" 区间集求交

        :param tags: 同nestenv的tags参数规则

        # 删除即在公式里，也在ce里的内容
        >>> NestEnv(r'12$34$56\ce{78$x$}90').all([r'\ce{', '}', 1, '$', '$', 1]).replace(lambda s: '')
        '12$34$56\\ce{78}90'

        >>> NestEnv(r'12$34$56\ce{78$x$}90').all(['$', '$', 1, r'\ce{', '}', 1]).replace(lambda s: '')
        '12$34$56\\ce{78}90'
        """
        tags, intervals = self._parse_tags(tags), self.intervals
        for tag in tags:
            head, tail, t = tag
            li = []
            for reg in self.intervals:
                left, right = reg.start(), reg.end()
                li.extend(substr_intervals(self.s[left:right], head, tail, invert=(t == 2), inner=(t == 0)) + left)
            intervals &= Intervals(li)
        return NestEnv(self.s, intervals)

    def __repr__(self):
        """不在定位范围内的非换行字符，全部替换为空格"""
        t = self.intervals.replace(self.s, lambda s: s, lambda s: re.sub(r'[^\n]', ' ', s))
        return t

    def __bool__(self):
        """NestEnv类的布尔逻辑由区间集的逻辑确定"""
        return bool(self.intervals)

    def string(self, idx=0):
        """第一个区间匹配的值

        >>> NestEnv('11a22b33a44bcc').find2('a', 'b').string()
        'a22b'
        """
        if self.intervals and idx < len(self.intervals):
            r = self.intervals.li[idx]
            return self.s[r.start():r.end()]
        else:
            return None

    def strings(self):
        """所有区间匹配的值"""
        if self.intervals:
            return [self.s[r.start():r.end()] for r in self.intervals]
        else:
            return []

    def startlines(self, unique=False):
        r""" 每个匹配到的区间处于原内容s的第几行

        >>> NestEnv('{}\naa\n{}\n{}{}a\nb').inside('{', '}').startlines()
        [1, 3, 4, 4]
        """
        if not self.intervals: return []
        # 1 辅助变量
        linepos = [m.start() for m in re.finditer(r'\n', self.s)]
        n = len(self.s)
        if n and (not linepos or linepos[-1] != n): linepos.append(n)
        # 2 每个子区间起始行号
        lines = [bisect.bisect_right(linepos, x.start() - 1) + 1 for x in self.intervals]
        if unique: lines = sorted(set(lines))
        return lines

    def group(self, idx=0):
        """第一个匹配区间，以match格式返回"""
        if self.intervals and idx < len(self.intervals):
            r = self.intervals.li[idx]
            return ReMatch(r.regs, self.s, 0, len(self.s))
        else:
            return None

    def groups(self):
        """所有匹配区间，以match格式返回"""
        if self.intervals:
            return [ReMatch(r.regs, self.s, 0, len(self.s)) for r in self.intervals]
        else:
            return []

    # TODO def gettag、settag、gettags、settags  特殊的inside操作
    # TODO def getattr、setattr、getattrs、setattrs

    def sub(self, infunc=lambda m: m.group(), *, outfunc=lambda m: m.group(), adjacent=False) -> str:
        """类似re.sub正则模式的替换"""
        return self.intervals.sub(self.s, infunc, out_repl=outfunc, adjacent=adjacent)

    def replace(self, arg1, arg2=None, *, outfunc=lambda s: s, adjacent=False) -> str:
        """ 类似字符串replace模式的替换

        arg1可以输入自定义替换函数，也可以像str.replace(arg1, arg2)这样传入参数
        """
        return self.intervals.replace(self.s, arg1, arg2, out_repl=outfunc, adjacent=adjacent)

    def __invert__(self):
        r"""
        >>> (~NestEnv('aa$b$cc').find2('$', '$')).strings()
        ['aa', 'cc']
        """
        return NestEnv(self.s, self.intervals.invert(len(self.s)))

    def invert(self):
        r"""
        >>> NestEnv('aa$b$cc').find2('$', '$').invert().strings()
        ['aa', 'cc']
        """
        return ~self

    def __and__(self, other):
        r""" 区间集求并运算

        >>> s = 'aa$b$ccc$dd$eee'
        >>> (NestEnv(s).find2('$', '$') & NestEnv(s).inside('a', 'd')).strings()
        ['$b$', '$d']
        >>> (NestEnv(s).find2('$', '$') & re.finditer(r'a.*?d', s)).strings()
        ['$b$', '$d']
        """
        if isinstance(other, Intervals):
            return NestEnv(self.s, self.intervals & other)
        elif isinstance(other, NestEnv):
            if self.s != other.s:  # 两个不是同个文本内容的话是不能合并的
                raise ValueError('两个NestEnv的主文本内容不相同，不能求子区间集的交')
            return NestEnv(self.s, self.intervals & other.intervals)
        else:  # 其他一律转Intervals对象处理
            # raise TypeError(rf'NestEnv不能和{type(other)}类型做区间集交运算')
            return NestEnv(self.s, self.intervals & Intervals(other))

    def __or__(self, other):
        """ 区间集相加运算

        >>> s = 'aa$b$ccc$dd$eee'
        >>> (NestEnv(s).find2('$', '$') | NestEnv(s).inside('a', 'd')).strings()
        ['aa$b$ccc$dd$']
        >>> (NestEnv(s).find2('$', '$') | re.finditer(r'a.*?d', s)).strings()
        ['aa$b$ccc$dd$']
        """
        if isinstance(other, Intervals):
            return NestEnv(self.s, self.intervals | other)
        elif isinstance(other, NestEnv):
            if self.s != other.s:
                raise ValueError('两个NestEnv的主文本内容不相同，不能求子区间集的并')
            return NestEnv(self.s, self.intervals | other.intervals)
        else:  # 其他一律转Intervals对象处理
            return NestEnv(self.s, self.intervals | Intervals(other))

    def __add__(self, other):
        return self | other

    def __sub__(self, other):
        """ 区间集减法运算

        >>> s = 'aa$b$ccc$dd$eee'
        >>> (NestEnv(s).find2('$', '$') - NestEnv(s).inside('a', 'd')).strings()
        ['d$']
        >>> (NestEnv(s).find2('$', '$') - re.finditer(r'a.*?d', s)).strings()
        ['d$']
        """
        if isinstance(other, Intervals):
            return NestEnv(self.s, self.intervals - other)
        elif isinstance(other, NestEnv):
            if self.s != other.s:
                raise ValueError('两个NestEnv的主文本内容不相同，子区间集不能相减')
            return NestEnv(self.s, self.intervals - other.intervals)
        else:  # 其他一律转Intervals对象处理
            return NestEnv(self.s, self.intervals - Intervals(other))

    def nest(self, func, invert=False):
        """ 对每个子区间进行一层嵌套定位

        :param func: 输入一个函数，模式为 func(s)
            支持输入一个字符串，返回一个类区间集
        :param invert: 是否对最终的结果再做一次取反
        :return: 返回一个新的NestEnv对象
        """
        li = []
        for reg in self.intervals:
            left, right = reg.start(), reg.end()
            t = self.s[left:right]
            res = Intervals(func(t))
            if invert: res = res.invert(len(t))
            li.extend(res + left)
        return NestEnv(self.s, Intervals(li))


class NestEnv(__NestEnvBase):
    """可以在该类扩展特定字符串匹配功能
        实现方法可以参照find、find2
            核心是要实现一个core函数
            支持输入一个字符串s，能计算出需要定位的子区间集位置
    """

    def find(self, head, invert=False):
        r"""没有tail，仅查找head获得区间集的算法

        >>> ne = NestEnv('111222333')
        >>> ne.find('2').strings()
        ['2', '2', '2']
        >>> ne.find('2', invert=True).strings()
        ['111', '333']
        >>> ne.find('4').strings()
        []
        >>> ne.find('4', invert=True).strings()
        ['111222333']
        >>> ne.find('22').find('2').strings()
        ['2', '2']
        """

        def core(s):
            pos1, parts = 0, []
            while True:
                pos2 = s.find(head, pos1)
                if pos2 == -1: break
                pos1 = pos2 + len(head)
                parts.append([pos2, pos1])
            return parts

        return self.nest(core, invert)

    def find2(self, head, tail, inner=False, invert=False):
        r""" 配对字符串匹配

        >>> ne = NestEnv('111222333')
        >>> ne.find2('1', '3').strings()
        ['1112223']
        """

        def core(s):
            pos1, parts = 0, []
            while True:
                pos2 = s.find(head, pos1)
                if pos2 == -1: break
                t = s.find(tail, pos2 + len(head))
                if t == -1:
                    break  # 有头无尾，不处理，跳过
                    # dprint(s, head, tail)
                    # raise ValueError
                pos1 = t + len(tail)

                if inner:
                    parts.append(pqmove(s, pos2 + len(head), pos1 - len(tail)))
                else:
                    parts.append([pos2, pos1])

            return parts

        return self.nest(core, invert)

    def search(self, pattern, flags=0, group=0, invert=False):
        r""" 正则模式匹配

        :param group: 可以指定返回的编组内容，默认第0组

        >>> NestEnv(r'xx\teste{aa}{bb}').search(r'\\test[a-z]*').strings()
        ['\\teste']

        TODO 如果需要用iner可以用命名组 (?P<inner>.*?)，含inner组名时，inner默认值为True
        """

        def core(s):
            return [m.span(group) for m in re.finditer(pattern, s, flags)]

        return self.nest(core, invert)

    def search2(self, pattern1, pattern2, *, flags1=0, flags2=0, invert=False):
        """ 配对正则匹配
        TODO 实现应该可以参考find2
        """
        raise NotImplementedError

    def bracket(self, head, tail=None, inner=False, *, latexenv=False, invert=False):
        r""" (尾)括号匹配
        head最后一个字符是参考匹配括号
        tail可以自定义，甚至可以长度不为1，但长度超过1时，算法是有bug的，只是不会抛出异常而已

        :param latexenv: latex环境下的括号匹配，需要忽略注释，以及\{等转义的影响
            目前并未细致优化该功能，只是暂时考虑了\{的影响

        >>> NestEnv('__{_}_[_]_{[_]+[_]}__').bracket('{', '}').bracket('[', ']', inner=True).replace('1')
        '__{_}_[_]_{[1]+[1]}__'

        >>> NestEnv('__{_}_[_]_{[_]+[_]}__').bracket('{', '}', inner=True).bracket('[', ']', invert=True).replace('1')
        '__{1}_[_]_{[_]1[_]}__'

        >>> NestEnv(r'xx\ce{b{c}d}b').bracket(r'\ce{').strings()
        ['\\ce{b{c}d}']

        >>> NestEnv(r'01\ce{H2O\ce{2}}01\ce{1\ce{3}5}').bracket(r'\ce{', inner=True).bracket(r'\ce{', inner=True).replace(lambda s: 'x')
        '01\\ce{H2O\\ce{x}}01\\ce{1\\ce{x}5}'
        """

        def core(s):
            # # 1 对常用的latex命令括号做加速  （做了实验发现加速差不到哪里去，还不如不加速）
            # if re.match(r'\\[a-zA-Z]+\{$', head) and tail == '}':
            #     i = 'inner' if inner else 0
            #     brace5 = r'{(?P<inner>(?:[^{}]|{(?:[^{}]|{(?:[^{}]|{(?:[^{}]|{(?:[^{}]|{[^{}]*})*})*})*})*})*)}'
            #     return [m.span(i) for m in re.finditer(r'\\' + head[1:-1] + brace5, s)]

            # 2 原版正常功能
            pos1, parts = 0, []
            # TODO grp_bracket(5, head[-1])，使用 grp_bracket 正则实现方式提速（需要测试性能差别，和一定测试）
            pos2 = s.find(head, pos1)
            while pos2 >= 0:
                # 找tail位置，目标区段
                p = bracket_match2(s, pos2 + len(head) - 1)
                if not p:
                    s += ' '
                    p = len(s)
                pos1 = p + 1

                if inner:
                    parts.append(pqmove(s, pos2 + len(head), pos1 - len(tail)))
                else:
                    parts.append([pos2, pos1])

                pos2 = s.find(head, pos1)
                if pos2 < pos1: break
            return parts

        # 自动推导 tail 的取值
        if not tail and head[-1] in '[{(<':  # 配对括号
            tail = {'[': ']', '{': '}', '(': ')', '<': '>'}[head[-1]]

        return self.nest(core, invert)

    def bracket2(self, head, tail=None, inner=False, *, latexenv=False, invert=False):
        r""" (头)括号匹配
        >>> NestEnv(r'xx{\centerline{aa}b}yy').bracket2(r'{\centerline').strings()
        ['{\\centerline{aa}b}']
        """

        def core(s):
            pos1, parts = 0, []
            # TODO 考虑用正则实现提速？
            pos2 = s.find(head, pos1)
            while pos2 >= 0:
                # 找tail位置，目标区段
                p = bracket_match2(s, pos2)
                if not p:
                    s += ' '
                    p = len(s)
                pos1 = p + 1

                if inner:
                    parts.append(pqmove(s, pos2 + len(head), pos1 - len(tail)))
                else:
                    parts.append([pos2, pos1])

                pos2 = s.find(head, pos1)
                if pos2 < pos1: break
            return parts

        # 自动推导 tail 的取值
        if not tail and head[0] in '[{(<':  # 配对括号
            tail = {'[': ']', '{': '}', '(': ')', '<': '>'}[head[0]]

        return self.nest(core, invert)

    def xmltag(self, head, inner=False, invert=False):
        r"""
        # >>> s = 'a\n<p class="clearfix">\nbb\n</p>c<p a="2"/>cc'
        # >>> NestEnv(s).inside('<p').replace('x')
        # 'a\nxc'

        陷阱备忘：
            1、百分注就那么几种格式，所以写的稍微不太严谨也么关系的，比如不会出现自闭合标签
            2、但要用正则做通用的xml格式解析，就很难去保证严谨性能，很容易出bug
            3、但用标准的xml解析也不行，因为很多文本场合并不是严格意义上的xml文档
            4、似乎只有用编译原理的理念一个个字符去解析文本才能真正确保准确性了。。。
                但这不切实际，实际可行方案还是得用正则，虽然不严谨有风险
        """

        def core(s):
            i = 'inner' if inner else 0
            pattern = fr'<({head})(?:>|\s.*?>)\s*(?P<inner>.*?)\s*</\1>'
            res = [m.span(i) for m in re.finditer(pattern, s, flags=re.DOTALL + re.MULTILINE)]

            # if not inner:  # 如果没开inner模式，还要再加上纯标签情况
            #     pattern = fr'<({name})(?:/>|\s[^>]*?/>)'
            # TODO 该函数应急使用，但算法本身非常不严谨，只要出现嵌套、自闭合等等特殊情况，就会有问题
            return res

        return self.nest(core, invert)

    def attr(self, name, part=0, prefix=r'(?<![a-zA-Z])', suffix=r'\s*=\s*', invert=False):
        r"""
        :param name: 正则规则的属性名
        :param prefix: 前向切割断言
        :param suffix: 后缀与值之间的间隔
        :param part:
            'value', 纯的属性值，不含引号
            'rawvalue'，如果属性被引号包裹，则只返回包括引号本身的内容
            'name'，仅属性名
            'name-op', 属性名和设置的=内容
            0，（默认）整串内容
        :return:

        >>> NestEnv('a b=12 3 c').attr('b', 'value').string()
        '12'
        >>> NestEnv('a b=123 c').attr('a', 'value').string()
        >>> NestEnv('a b="123" c').attr('b', 'rawvalue').string()  # 匹配含引号本身的值
        '"123"'
        >>> NestEnv("a b='123' c").attr('b', 'rawvalue').string()
        "'123'"
        >>> NestEnv('a b="123" c').attr('b', 'value').string()  # 仅匹配值
        '123'
        >>> NestEnv('a b="123" c').attr('b').string()  # 匹配整串
        'b="123"'
        >>> NestEnv('a=ab b="123" c').attr(r'(a|b)', 'rawvalue').strings()  # 正则匹配属性名
        ['ab', '"123"']
        >>> NestEnv('a=ab b="123" c').attr(r'(a|b)', 'name').strings()
        ['a', 'b']
        >>> NestEnv('a= ab b ="123" c').attr(r'(a|b)', 'name-op').strings()
        ['a= ', 'b =']
        """

        def core(s):
            p, parts = 0, []
            while True:
                m0 = re.search(pattern0, s[p:])
                if not m0: break

                q = p + m0.end()
                ch = s[q] if q < len(s) else ''
                if ch in '"\'':  # 下一个字符是双引号或者单引号
                    m1 = re.search(fr'{ch}[^ch]*{ch}', s[q:])
                    inner_left, inner_right = m1.start() + 1, m1.end() - 1
                else:  # 没有引号
                    m1 = re.search(r'\S*', s[q:])
                    inner_left, inner_right = m1.span()

                if part == 0:
                    parts.append([p + m0.start(), q + m1.end()])
                elif part == 'value':
                    parts.append([q + inner_left, q + inner_right])
                elif part == 'rawvalue':
                    parts.append([q + m1.start(), q + m1.end()])
                elif part == 'name':
                    parts.append([p + m0.start(), p + m0.start('op')])
                elif part == 'name-op':
                    parts.append([p + m0.start(), p + m0.end()])
                else:
                    raise ValueError(f'part名称不对{part}')

                p = q + m1.end()
            return parts

        pattern0 = fr'{prefix}{name}(?P<op>{suffix})'
        return self.nest(core, invert)

    def pathstem(self):
        r""" TODO 路径相关的规则匹配
        例如pathstem可以和includegraphics('inner')配合
        但这个接口最后不一定是这样设计的，可能会写个通用的path处理接口
        """
        raise NotImplementedError

    def paragraph(self, linefeed=1, invert=False):
        """ 定位段落
        :param linefeed: 每个段落间至少间隔换行符数量
        """

        def core(s):
            return list(filter(lambda m: m.group().count('\n') >= linefeed, re.finditer(r'\s+', s)))

        # 由于这个算法核心是要定位分隔符，最后实际效果是否invert是要取反的
        return self.nest(core, not invert)
