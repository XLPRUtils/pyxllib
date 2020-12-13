#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/12/08 17:02

from pyxllib.text._4_nestenv import *


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

            lewis = r'\\(l|L)ewis' + BRACE5  # 基本匹配模式
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
