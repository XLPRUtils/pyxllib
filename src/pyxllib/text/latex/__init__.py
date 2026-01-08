#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/10/08 22:26

import re

from pyxllib.prog.specialist import browser
from pyxllib.text.pupil import grp_bracket, continuous_zero


class TexTabular:
    @classmethod
    def parse_multirow(cls, s, brace_text_only=True):
        r"""

        :param brace_text_only: 只取花括号里面的内容
            如果为False，会把multirow外部的内容做拼接

        multirow 和 multicolumn 的不同是，第1、2个花括号后面可以有可选参数。
        第2个花括号如果内容是*，可以省略。
        两个[]的内容先省略，不做分析处理

        注意：这里会取出前后缀内容！业务需要，防止bug，不过这种概率很小

        >>> TexTabular.parse_multirow(r'\multirow{2}*{特点}')
        (2, None, '*', None, '特点')
        >>> TexTabular.parse_multirow(r'\multirow{2}{*}{特点}')
        (2, None, '*', None, '特点')
        >>> TexTabular.parse_multirow(r'aa\multirow{2}[bla1]{*}[bla2]{特点}bb', brace_text_only=False)
        (2, 'bla1', '*', 'bla2', 'aa特点bb')

        TODO multirow第一个数字是可以负值的，代表向上合并单元格数，
        """
        square = r'(?:\[(.*?)\])?'  # 可选参数
        m = re.search(r'\\multirow' + grp_bracket(3, inner=True) + square +
                      r'(?:{(.*?)}|(\*))' + square + grp_bracket(5, inner=True), s)
        if not m: return None
        n, bigstructs, width1, width2, fixup, text = m.groups()
        width = width1 or width2
        if not brace_text_only: text = s[:m.start()] + text + s[m.end():]
        # if re.match(r'\d+$', text): text = int(text)  # 如果可以，直接识别为数字

        n = int(n)
        if -1 <= n <= 1:
            n = 1
        elif n > 1:
            pass
        else:
            raise ValueError(f'{s} 不支持解析multirow第一个值为负数，向上合并单元格的情况')

        return n, bigstructs, width, fixup, text

    @classmethod
    def parse_multicolumn(cls, s):
        r"""找出s中第一次出现的满足模式的multicolumn，返回3个关键值

        :returns:
            第1个参数是该合并单元格的尺寸，固定格式： (行数, 列数)，只有一行是也会写'1'

        >>> TexTabular.parse_multicolumn(r'\multicolumn{2}{|c|}{aa\multirow{3}*{特点}bb}')
        ((3, 2), '|c|', 'aa特点bb')
        """
        # 1 基本的模式匹配抓取
        m = re.search(r'\\multicolumn' + grp_bracket(3, inner=True) * 2
                      + grp_bracket(5, inner=True), s)  # 最后层多套下，我怕不够用
        if not m: return None

        # 2 取出参数值
        m, col_align, text = m.groups()
        m = int(m)

        # 3 如果有 multirow
        if 'multirow' in text:
            n, bigstructs, width, fixup, text = cls.parse_multirow(text, brace_text_only=False)
        else:
            n = 1
        # if isinstance(text, str) and re.match(r'\d+$', text): text = int(text)  # 如果可以，直接识别为数字
        return (n, m), col_align, text

    @classmethod
    def parse_align(cls, s):
        r"""解析latex表头的列对齐格式

        latex表头的规则很复杂，这里目前只处理一些较常用的功能点

        :param s: 内容文本
        :return: 不考虑竖线和一些高级对齐格式，暂时返回一个str
            长度是表格列数，每个元素是一个字母存储对齐信息（后续可以扩展更细致的对齐格式信息）

        >>> TexTabular.parse_align('{|c|c|c|c|c|c|c|c|c<{}|c|}')
        'cccccccccc'
        >>> TexTabular.parse_align('{|c|w{6em}|w{23mm}|w{47mm}|w{22mm}|}')
        'cwwww'
        >>> TexTabular.parse_align('cc*{8}{l}')
        'ccllllllll'
        >>> TexTabular.parse_align('|c|')
        'c'
        >>> TexTabular.parse_align('|c|*{2}{m{38mm}<{\\centering}|}')
        'cmm'
        """
        # 展开 *{n}{列格式} 模式
        s = re.sub(r'\*(\d+)', r'*{\1}', s)  # 给*数字加上花括号，不然我的匹配会错
        s = re.sub(r'\*{(\d+)}' + grp_bracket(3, inner=True), lambda m: m.group(2) * int(m.group(1)), s)
        # 删除其他干扰字符
        if s[0] == '{' and s[-1] == '}': s = s[1:-1]  # 去掉头尾 { }
        s = re.sub(r'{.*?}', '', s)
        for char in '|<>!':
            s = s.replace(char, '')
        return s

    @classmethod
    def create_cline(cls, merge_count):
        r"""
        :param merge_count: 一个长度等于表格列数的list，第i位的值存储了第i列累计到当前被合并的格子数
            假设一个3*4的表格，第1、3、4列正常，第2列被合并了
            那么遍历到第二行时，merge_count为 [0, 2, 0, 0]
            遍历到第三行时，merge_count 为 [0, 1, 0, 0]
        :return:

        >>> TexTabular.create_cline([0, 1, 0, 0])
        '\\cline{1-1} \\cline{3-4}'
        """

        s = ''.join([('1' if v else '0') for v in merge_count])
        if s.count('1') == 0: return '\\hline'  # 没有间断，直接用hline命令

        spans = continuous_zero(s)  # 注意返回的区间是从0开始编号，左闭右开的
        li = [f'\\cline{{{span[0] + 1}-{span[1]}}}' for span in spans]
        return ' '.join(li)

    @classmethod
    def create_formats(cls, format_count):
        """ 获得latex表头格式 """

        def count(s):
            """列对齐格式统计，返回最终去用的对齐格式"""
            if not s: return 'l'  # 默认左对齐
            l, c, r = s.count('l'), s.count('c'), s.count('r')
            if l >= c and l >= r:
                return 'l'
            elif c >= l and c >= r:
                return 'c'
            else:
                return 'r'

        formats = [count(x) for x in format_count]
        return '{|' + '|'.join(formats) + '|}'


def browser_latex(text='请输入...'):
    from html import escape
    from pyxllib.text.xmllib import get_jinja_template

    # 致谢：感谢奕本在晓波做的工具基础上，做出的这个简洁版的latex渲染器
    content = get_jinja_template('latex_editor.html').render(text=escape(text))
    browser.html(content)
