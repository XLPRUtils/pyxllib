#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : chenkz@histudy.com
# @Date   : 2019/02/20 10:03

import bisect
import re

from pyxllib.algo.intervals import Intervals, ReMatch
from pyxllib.text.newbie import bracket_match2
from pyxllib.text.pupil import grp_bracket, strfind, findspan, substr_count


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

    def expand(self, ne, adjacent=False):
        r""" 在现有区间上，判断是否有被其他区间包含，有则进行延展
        可以输入head、tail配对规则，也可以输入现成的区间

        :param adjacent: 是否支持邻接的区间扩展

        >>> ne = LatexNestEnv(r'aa$cc\ce{a}dd$bb\ce{d}h$h$')
        >>> ne.latexcmd1(r'ce').expand(ne.formula()).strings()
        ['$cc\\ce{a}dd$', '\\ce{d}']
        """
        if isinstance(ne, NestEnv):
            b = ne.intervals
        elif isinstance(ne, Intervals):
            b = ne
        else:
            raise TypeError
        if adjacent:
            c = self.intervals + Intervals([x for x in b if (self.intervals.is_adjacent_and(x))])
            c = Intervals(c.merge_intersect_interval(adjacent=True))
        else:
            c = self.intervals + Intervals([x for x in b if (self.intervals & x)])
        return NestEnv(self.s, c)

    def filter(self, func):
        r""" 传入一个自定义函数func，会将每个区间的s传入，只保留func(s)为True的区间

        >>> LatexNestEnv('aa$bbb$ccc$d$eee$fff$g').formula().filter(lambda s: len(s) > 4).strings()
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

    # def any(self, tags):
    #     r""" 区间集求并
    #
    #     :param tags: 同nestenv的tags参数规则
    #
    #     >>> NestEnv(r'12$34$56\ce{78}90').any(['$', '$', 1, r'\ce{', '}', 1]).replace(lambda s: 'x')
    #     '12x56x90'
    #     """
    #     tags, li = self._parse_tags(tags), []
    #     for tag in tags:
    #         head, tail, t = tag
    #         for reg in self.intervals:
    #             left, right = reg.start(), reg.end()
    #             li.extend(substr_intervals(self.s[left:right], head, tail, invert=(t == 2), inner=(t == 0)) + left)
    #     return NestEnv(self.s, Intervals(li))

    # def all(self, tags):
    #     r""" 区间集求交
    #
    #     :param tags: 同nestenv的tags参数规则
    #
    #     # 删除即在公式里，也在ce里的内容
    #     >>> NestEnv(r'12$34$56\ce{78$x$}90').all([r'\ce{', '}', 1, '$', '$', 1]).replace(lambda s: '')
    #     '12$34$56\\ce{78}90'
    #
    #     >>> NestEnv(r'12$34$56\ce{78$x$}90').all(['$', '$', 1, r'\ce{', '}', 1]).replace(lambda s: '')
    #     '12$34$56\\ce{78}90'
    #     """
    #     tags, intervals = self._parse_tags(tags), self.intervals
    #     for tag in tags:
    #         head, tail, t = tag
    #         li = []
    #         for reg in self.intervals:
    #             left, right = reg.start(), reg.end()
    #             li.extend(substr_intervals(self.s[left:right], head, tail, invert=(t == 2), inner=(t == 0)) + left)
    #         intervals &= Intervals(li)
    #     return NestEnv(self.s, intervals)

    def __repr__(self):
        """不在定位范围内的非换行字符，全部替换为空格"""
        t = self.intervals.replace(self.s, lambda s: s, out_repl=lambda s: re.sub(r'[^\n]', ' ', s))
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
            支持输入一个字符串，返回一个"区间集like"对象
        :param invert: 是否对最终的结果再做一次取反
        :return: 返回一个新的NestEnv对象

        注意所有的定位功能，基本都要基于这个模式开发。
        因为不是要对self.s整串匹配，而是要嵌套处理，只处理self.intervals标记的区间。
        """
        li = []
        for reg in self.intervals:
            left, right = reg.start(), reg.end()
            t = self.s[left:right]
            res = Intervals(func(t))
            if invert: res = res.invert(len(t))
            li.extend(res + left)
        return type(self)(self.s, Intervals(li))


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

    def find2(self, head, tail, *, inner=False, invert=False, symmetry=False):
        r""" 配对字符串匹配

        :param head: 默认字符串匹配，也支持输入re.compile的正则
        :param tail: 同head
        :param symmetry: 要求匹配到的head和tail数量对称
            这个算法会慢非常多，如无必要不用开

        >>> ne = NestEnv('111222333')
        >>> ne.find2('1', '3').strings()
        ['1112223']
        """

        def core(s):
            pos1, parts = 0, []
            while True:
                # 找到第1个head
                pos2, pos2end = findspan(s, head, pos1)
                if pos2 == -1:
                    break

                # 找到上一个head后，最近出现的tail
                pos3, pos1 = findspan(s, tail, pos2end)

                if symmetry:
                    while True:
                        substr = s[pos2:pos1]
                        cnt1, cnt2 = substr_count(substr, head), substr_count(substr, tail)

                        if pos3 == -1 or cnt1 == cnt2:
                            break
                        else:
                            pos3, pos1 = findspan(s, tail, pos1)

                if pos3 == -1:
                    # 有头无尾，不处理，跳过
                    break

                # 坐标计算、存储
                if inner:
                    parts.append(pqmove(s, pos2end, pos3))
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
            return [m.span(group) for m in re.finditer(pattern, s, flags=flags)]

        return self.nest(core, invert)

    def search2(self, pattern1, pattern2, *, inner=False, flags1=0, flags2=0, invert=False):
        """ 配对正则匹配
        TODO 实现应该可以参考find2
        """
        head = re.compile(pattern1, flags=flags1)
        tail = re.compile(pattern2, flags=flags2)
        return self.find2(head, tail, inner=inner, invert=invert)

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

    def xmltag(self, head, inner=False, invert=False, symmetry=True):
        r"""
        >>> s = 'a\n<p class="clearfix">\nbb\n</p>c<p a="2"/>cc'
        >>> NestEnv(s).xmltag('p').replace('x')
        'a\nxc<p a="2"/>cc'

        陷阱备忘：
            1、百分注就那么几种格式，所以写的稍微不太严谨也么关系的，比如不会出现自闭合标签
            2、但要用正则做通用的xml格式解析，就很难去保证严谨性能，很容易出bug
            3、但用标准的xml解析也不行，因为很多文本场合并不是严格意义上的xml文档
            4、似乎只有用编译原理的理念一个个字符去解析文本才能真正确保准确性了。。。
                但这不切实际，实际可行方案还是得用正则，虽然不严谨有风险
        """
        # 暂不考虑自关闭 <a/>的情况
        h = re.compile(rf'<({head})(?:>|\s.*?>)', flags=re.DOTALL)
        t = re.compile(f'</{head}>')
        return self.find2(h, t, inner=inner, invert=invert, symmetry=symmetry)

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


class LatexNestEnv(NestEnv):
    def includegraphics(self, part=0, invert=False):
        r"""能抓取各种插图命令，使用inner可以只获得图片文件名

        :param part:
            0           全内容
            cmd         命令名
            optional    可选参数的内容
            inner       花括号里的参数内容
            stem        inner里不含路径、扩展名的纯文件名

        >>> s = r'阳离子：\underline{\includegraphics{18pH-g1=8-8.eps}\qquad \figt[9pt]{18pH-g1=8-9.eps}}'
        >>> LatexNestEnv(s).includegraphics().strings()
        ['\\includegraphics{18pH-g1=8-8.eps}', '\\figt[9pt]{18pH-g1=8-9.eps}']
        >>> LatexNestEnv(s).includegraphics('inner').strings()
        ['18pH-g1=8-8.eps', '18pH-g1=8-9.eps']
        """

        def core(s):
            grp_bracket3 = '{(?P<inner>(?:[^{}]|{(?:[^{}]|{(?:[^{}]|{[^{}]*})*})*})*)}'
            pattern = r'\\(?P<cmd>includegraphics|figt|figc|figr|fig)(?P<optional>.*?)' + grp_bracket3
            return [m.span(part) for m in re.finditer(pattern, s, flags=re.DOTALL + re.MULTILINE)]

        if part == 'stem': raise NotImplementedError
        return self.nest(core, invert)

    def lewis(self, inner=False, invert=False):
        r"""电子式的匹配
        这个本身是有个命令的并不难，难的是实际情况中，往往两边会有拓展

        >>> LatexNestEnv(r'aa H\Lewis{0:2:4:6:,N}H bb').inside(r'\lewis').strings()
        ['H\\Lewis{0:2:4:6:,N}H']
        >>> LatexNestEnv(r'aa H\Lewis{0:2:4:6:,N}H bb').lewis().strings()
        ['H\\Lewis{0:2:4:6:,N}H']
        """

        def core(s):
            if inner:  # 只取\lewis{}花括号里内容的定位
                raise ValueError(r"lewis模式没有inner模式，如果需要可以使用NestEnv(s).inner(r'\lewis{')")

            lewis = r'\\(l|L)ewis' + grp_bracket(5, inner=True)  # 基本匹配模式
            ms = re.finditer(rf'(H?~*{lewis}\s*|~*H)*(~*{lewis}|~*H)', s)  # 有一定的延展
            return [m.span(0) for m in ms if 'lewis' in m.group().lower()]

        return self.nest(core, invert)

    def item(self, invert=False):
        r""" 主要用于word转latex中，对不含百分注，但是有基本的\itemQ、\test、itemKey等的切分定位，找出每个item的区间

        # TODO 支持inner功能
        """

        def core(s):
            pos1, parts = 0, []
            pos2 = strfind(s, (r'\item', r'\test'), start=pos1)
            while pos2 >= 0:
                # 找tail位置，目标区段
                p = strfind(s, (r'\item', r'\test'), start=pos2 + 1)
                if p == -1:
                    s += ' '
                    p = len(s)
                # else:
                #     while s[p-1] in ' \t\n': p -= 1
                if p < len(s) and s[p] == '\n': p += 1
                while s[p - 1] in ' \t\n': p -= 1
                pos1 = p
                parts.append([pos2, pos1])
                pos2 = strfind(s, (r'\item', r'\test'), start=pos1)
                if pos2 < pos1: break
            return parts

        return self.nest(core, invert)

    def latexcmd(self, name=r'[a-zA-Z]+', *, part=0, star=True, optional=True,
                 min_bracket=0, max_bracket=float('inf'), brackets=None,
                 linefeed=1, invert=False):
        r"""匹配latex命令区间

        :param part: TODO 功能待开发~~
            0，整块内容
            name，仅命令名，如果有star会含*
            optional，可选参数部分
            optional-value，可选参数里的值

            rawvalues，含花括号的整组匹配内容，例如  {...}{...}{...}
            rawvalue1，仅每个命令第一对花括号内容
            rawvalue2，仅每个命令第二对花括号内容
            ...rawvalue

            values，仅花括号里的值，如果bracket不只一个，会拆成多个子区间，即虽然匹配到2个命令，但有可能得到6个子区间
            value1，仅每个命令的第一个花括号里的值
            value2，仅每个命令的第二个花括号里的值
            ...valuex

        :param name: 命令名，需要使用正则模式指明匹配规则，不用写前缀\\和后缀(?![a-zA-Z])
        :param star: 命令后是否支持跟一个*，默认支持。可以设置star=False强制不支持
        :param optional: 是否支持 [...] 的可选参数模式，默认支持
        :param min_bracket: 最少要有多少对花括号，默认可以没有
        :param max_bracket: 最多匹配多少对花括号，默认值inf表示不设上限
        :param brackets: 特定匹配多少对花括号
            使用该参数时，min_bracket、max_bracket重置为brackets设定的值
        :param linefeed: 各项内容之间最多只能有几个换行
        :return:

        TODO 后面有余力可以考虑下怎么扩展inner参数

        算法：由于情况的复杂性，基本思路是只能一步一步往前search
        注意：这里暂不支持multirow那种可以在第2个参数写*的情形，那种情况可以另外再开发multirow匹配函数

        >>> LatexNestEnv('\n\\ssb{有关概念及其相互关系}\n\n{\\includegraphics{19pS-g4=5-1.png}}').latexcmd().replace('')
        '\n\n\n{}'
        """

        def core(s):
            right, parts = 0, []
            while True:
                m0 = re.search(r'\\(' + name + r')(?![a-zA-Z])', s[right:])
                if not m0: break
                left, right = m0.start() + right, m0.end() + right

                if star:
                    m1 = re.match(r'(\s*)(\*)', s[right:])
                    if m1 and m1.group(1).count('\n') <= linefeed and m1.group(2):
                        right += m1.end()

                if optional:
                    m2 = re.match(r'(\s*)(' + grp_bracket(5, '[') + ')', s[right:])
                    if m2 and m2.group(1).count('\n') <= linefeed and m2.group(2):
                        right += m2.end()

                cur_cnt, pattern = 0, r'(\s*)(' + grp_bracket(5) + ')'
                max_bracket_ = max_bracket
                if max_bracket == float('inf'):
                    if m0.group(1) in ('begin', 'end'): max_bracket_ = 1  # 有些命令只能匹配一个花括号
                    if m0.group(1) in ('hfil', 'hfill'): max_bracket_ = 0  # 有些命令不能匹配花括号
                while cur_cnt < max_bracket_:
                    m3 = re.match(pattern, s[right:])
                    if m3 and m3.group(1).count('\n') <= linefeed and m3.group(2):
                        right += m3.end()
                        cur_cnt += 1
                    else:
                        break

                if cur_cnt >= min_bracket:
                    parts.append([left, right])

            return parts

        if brackets:
            min_bracket = max_bracket = brackets

        return self.nest(core, invert)

    def latexcmd0(self, name=r'[a-zA-Z]+', *, part=0, star=False, optional=False,
                  min_bracket=0, max_bracket=0, brackets=None,
                  linefeed=1, invert=False):
        r""" 只匹配命令本身，不含star、optional、brackets
        """
        return self.latexcmd(name, part=part, star=star, optional=optional,
                             min_bracket=min_bracket, max_bracket=max_bracket, brackets=brackets,
                             linefeed=linefeed, invert=invert)

    def latexcmd1(self, name=r'[a-zA-Z]+', *, part=0, star=True, optional=True,
                  min_bracket=1, max_bracket=1, brackets=None,
                  linefeed=1, invert=False):
        r""" 只有一个花括号的latex命令匹配
        """
        return self.latexcmd(name, part=part, star=star, optional=optional,
                             min_bracket=min_bracket, max_bracket=max_bracket, brackets=brackets,
                             linefeed=linefeed, invert=invert)

    def latexenv(self, head, tail=None, inner=False, invert=False):
        r"""latex的\begin、\end环境匹配，支持嵌套定位

        >>> s = r"\begin{center}\begin{tabular}\begin{tabular}\end{tabular}\end{tabular}\end{center}"
        >>> LatexNestEnv(s).latexenv('tabular').replace('x')
        '\\begin{center}x\\end{center}'

        TODO 因为存在自嵌套情况，暂时还不好对head扩展支持正则匹配模式
        """

        def core(s):
            pos1, parts = 0, []
            # 最外层的head支持有额外杂质（tail暂不支持杂质），但是内部的h、t不考虑杂质，但最好不要遇到、用到这么危险的小概率功能
            h, t = re.match(r'\\begin{[a-zA-Z]+}', head).group(), re.match(r'\\end{[a-zA-Z]+}', tail).group()
            while True:
                pos2 = s.find(head, pos1)
                if pos2 == -1: break
                cnt1, cnt2, pos1 = 1, 0, pos2 + len(head)
                while cnt1 != cnt2:
                    pos1 = s.find(t, pos1)
                    if pos1 == -1:
                        break
                    else:
                        pos1 += len(t)
                    cnt1, cnt2 = s[:pos1].count(h), s[:pos1].count(t)
                if pos1 != -1:
                    if inner:
                        parts.append(pqmove(s, pos2 + len(head), pos1 - len(tail)))
                    else:
                        parts.append([pos2, pos1])
            return parts

        # 参数推算
        if re.match(r'[a-zA-Z]+$', head):
            head = r'\begin{' + head + '}'
        if not tail and re.match(r'\\begin{[a-zA-Z]+}', head):  # latex类的环境匹配
            m = re.match(r'\\begin({[a-zA-Z]+})', head)
            tail = r'\end' + m.group(1)

        return self.nest(core, invert)

    def latexcomment(self, include_pxmltag=False, invert=False):
        """ latex 的注释性代码
        :param include_pxmltag: 是否包含进百分注，默认不包含
        """
        if include_pxmltag:
            pattern = r'(?<!\\)%.*'
        else:
            pattern = r'(?<!\\)%(?!<).*'
        return self.search(pattern, invert=invert)

    def formula(self, inner=False, invert=False):
        r"""公式匹配
        >>> LatexNestEnv(r'aa$bb$cc').formula().strings()
        ['$bb$']

        TODO 遇到 "$$ x xx $xx$"，前面的$$是中间漏了内容的，应该要有异常处理机制，至少报个错误位置吧
        """

        def core(s):
            if r'\\$' in s:
                raise ValueError(r'内容中含有\\$，请先跑「refine_formula」加上空格')
            i = 'inner' if inner else 0
            # 线上才要考虑转义情况，线下也有可能\\后面跟$是不用处理的
            li1 = [m.span(i) for m in re.finditer(r'(?<!\\)(\$\$?)(?P<inner>.*?)(?<!\\)\1', s, flags=re.DOTALL)]
            li2 = [m.span(i) for m in
                   re.finditer(r'\$\s*\\begin{array}\s*(?P<inner>.*?)\s*\\end{array}\s*\$', s, flags=re.DOTALL)]
            return Intervals(li1) + Intervals(li2)

        return self.nest(core, invert)

    def pxmltag(self, name, part=0, invert=False):
        r""" 百分注结构匹配
            p（百分号 percent 的缩写） + xml + tag

        :param part:
            0               完整的全内容匹配
            head            百分注开标签前的内容，含边界的换行等空白字符
            inner_head      百分注开标签前的内容，不含边界的换行等空白字符
            open            开标签的内容，%<...>
            open_name       开标签的名称
            open_attrs      开标签的属性
            inner_open      不含 %<、> 的边界，但如果有左右空白还是带入
            body            开关标签间的内容，含边界
            inner           开关标签间的内容，不含边界
            close           关标签的内容，%</...>
            close_name      同inner_close
            inner_close     不含 %</、> 的边界，但如果有左右空白还是会带入
            tail            关标签后的内容，含边界，注意before和tail在定位topic等时可能会有重叠
            inner_tail      关标签后的内容，不含边界

        >>> s = 'a\n%<topic>\nbb\n%</topic>c'
        >>> LatexNestEnv(s).pxmltag('topic').replace('x')
        'a\nxc'

        head支持正则模式，例如：stem|answer
        TODO 怎么避免属性值中含有 > 的干扰？
        """

        def head(s):
            ms = re.finditer(r'(?P<head>\s*(?P<inner_head>.*?)\s*)(?P<tail>%<.*?>)', s)
            return [m.span(part) for m in ms if re.search(fr'%<({name})', m.group('tail'))]

        def open(s):
            pattern = fr'(?<!\\)%<(?P<open_name>{name})(>|\s[^\n]*?>)'
            if part == 'open':
                return [m.span() for m in re.finditer(pattern, s)]
            elif part == 'inner_open':
                return [[m.start() + 2, m.end() - 1] for m in re.finditer(pattern, s)]
            elif part == 'open_name':
                return [m.span('open_name') for m in re.finditer(pattern, s)]
            elif part == 'open_attrs':
                return [m.span(2) for m in re.finditer(fr'(?<!\\)%<({name})\s+(.+?)>', s)]
            else:
                raise ValueError(f'part名称不对{part}')

        def core(s):
            pattern = fr'(?<!\\)%<({name})(?:>|\s[^\n]*?>)(?P<body>\s*(?P<inner>.*?)\s*)%</\1>'
            return [m.span(part) for m in re.finditer(pattern, s, flags=re.DOTALL + re.MULTILINE)]

        def close(s):
            pattern = fr'(?<!\\)%</(?P<close_name>{name})>'
            res = [m.span() for m in re.finditer(pattern, s)]
            if part in ('inner_close', 'close_name'):
                res = [[x[0] + 3, x[1] - 1] for x in res]
            return res

        def tail(s):
            pattern = fr'(?<!\\)%</({name})>(?P<tail>\s*(?P<inner_tail>.*?)\s*)(?=%<|$)'
            return [m.span(part) for m in re.finditer(pattern, s, flags=re.DOTALL)]

        if name.startswith('%<'):
            raise ValueError

        if part in (0, 'body', 'inner'):
            return self.nest(core, invert)
        if part in ('head', 'inner_head'):
            return self.nest(head, invert)
        elif part in ('tail', 'inner_tail'):
            return self.nest(tail, invert)
        elif part in ('open', 'inner_open', 'open_name', 'open_attrs'):
            return self.nest(open, invert)
        elif part in ('close', 'inner_close'):
            return self.nest(close, invert)
        else:
            raise ValueError(f'part名称不对{part}')

    def latexpart(self, head, inner=False, invert=False):
        # TexPos比较特殊，暂时不迁移

        def core(s):
            parts = []
            n = TexPos(s).get(f'{head}Cnt')
            for i in range(n):
                p1, p2 = texpos(s, f'{head}{i}')
                if inner:  # inner应该要排除掉整个\chapter{...}内容
                    inter = (NestEnv(s[p1:p2]).latexcmd1(head, invert=True) + p1).intervals[0]
                    p1, p2 = inter.start(), inter.end()
                parts.append([p1, p2])
            return parts

        return self.nest(core, invert)

    def latexparagraph(self, linefeed=2):
        """
        latex的段落不能简单从这个函数继承，latex需要考虑注释的影响！
        """
        return self.paragraph(linefeed=linefeed).search(r'\n?(?<!\\)%<.+\n?', invert=True)


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


class CppNestEnv(NestEnv):
    def comments(self, *, invert=False):
        """ 这个实现是不太严谨的，如果急用，可以先凑合吧 """

        def core(s):
            ne = NestEnv(s)
            ne2 = ne.find2('/*', '*/') + ne.search(r'//.+')  # 找出c++的注释块
            return ne2

        return self.nest(core, invert)


class PyNestEnv(NestEnv):
    def imports(self, *, invert=False):
        """ 定位所有的from、import，默认每个import是分开的 """

        def core(s):
            # 捕捉连续的以'from ', 'import '开头的行
            ne = NestEnv(s)
            ne2 = ne.search(r'^(import|from)\s.+\n?', flags=re.MULTILINE) \
                  + ne.search(r'^from\s.+\([^\)]+\)[ \t]*\n?', flags=re.MULTILINE)
            return ne2

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

    def identifier(self, name, flags=0, group=0, *, invert=False):
        """ 定位特定的标识符位置
        这东西比较简单，可以在正则头尾加\b即可，也可以用普通正则实现

        :param name: 支持正则格式

        """

        def core(s):
            return [m.span(group) for m in re.finditer(rf'\b{name}\b', s, flags)]

        return self.nest(core, invert)
