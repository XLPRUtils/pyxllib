#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

import math
import unicodedata

from pyxllib.prog.basic import GrowingList
from pyxllib.prog.matrix import len_in_dim2
from pyxllib.prog.run import run_once
from pyxllib.text.convert import fullwidth2halfwidth


def east_asian_len(s, ambiguous_as_wide=False):
    """ 计算字符串的显示宽度

    :param s: 输入字符串
    :param ambiguous_as_wide:
        False，默认，将ambiguous的字符视为宽1
        True，将ambiguous的字符视为宽2，例如在win32上
    :return: 字符串显示宽度
    """
    n = 0
    for c in s:
        w = unicodedata.east_asian_width(c)
        if w in ('W', 'F'):
            n += 2
        elif w == 'A':
            n += 2 if ambiguous_as_wide else 1
        else:
            n += 1
    return n


def east_asian_shorten(s, width, placeholder='...', ambiguous_as_wide=False):
    """ 按照东亚字符宽度截断字符串

    :param s: 原始字符串
    :param width: 限制的最大显示宽度
    :param placeholder: 截断后追加的省略号，这个占位符的长度也会计算在width内
    """
    s = str(s)
    n = east_asian_len(s, ambiguous_as_wide)
    if n <= width:
        return s

    # 需要截断
    w_placeholder = east_asian_len(placeholder, ambiguous_as_wide)
    target_width = width - w_placeholder
    if target_width <= 0:  # 连省略号都放不下
        return placeholder[:width]  # 尽量返回省略号的一部分

    # 寻找截断位置
    current_width = 0
    cut_idx = 0
    for i, c in enumerate(s):
        w = unicodedata.east_asian_width(c)
        cw = 2 if w in ('W', 'F') or (ambiguous_as_wide and w == 'A') else 1
        if current_width + cw > target_width:
            cut_idx = i
            break
        current_width += cw
    else:
        cut_idx = len(s)

    return s[:cut_idx] + placeholder


@run_once('ignore,str')
def strwidth_proc(s, ambiguous_as_wide=False):
    """
    :param ambiguous_as_wide:
        False，默认，将ambiguous的字符视为宽1
        True，将ambiguous的字符视为宽2，例如在win32上
    """
    if not isinstance(s, str):
        s = str(s)
    return east_asian_len(s, ambiguous_as_wide)


def strwidth(s, ambiguous_as_wide=False):
    """
    :param s: 字符
    :param ambiguous_as_wide: 像①这种字符，在linux、pycharm中是宽1，但在windows终端是宽2
        默认False，取宽1
        设为True，取宽2
    """
    return strwidth_proc(s, ambiguous_as_wide)


def get_strwidth(ambiguous_as_wide=False):
    return lambda s: strwidth(s, ambiguous_as_wide)


def realign(text, tabsize=4):
    """去除一段文本的左侧缩进，并清除首尾空行
    该函数跟textwrap.dedent的区别是：
        dedent只去除左侧缩进
        realign还会去除首尾空行，并且可以重设tabsize（默认将tab转为4个空格处理）
    """
    text = text.strip('\n')
    if not text: return ''
    if tabsize:
        text = text.replace('\t', ' ' * tabsize)
    lines = text.splitlines()
    # 1 找到第一行非空行的缩进
    indent = ''
    for line in lines:
        if line.strip():
            indent = re.match(r'\s*', line).group()
            break
    # 2 去除所有行的缩进
    n = len(indent)
    for i in range(len(lines)):
        if lines[i].startswith(indent):
            lines[i] = lines[i][n:]
    return '\n'.join(lines)


def listalign(arr, alignment='l',
              ambiguous_as_wide=False, max_width=None,
              return_col_width=False):
    """将一个list里的元素，对齐长度后输出
    :param arr: 列表，如果不是str，会先str(x)转为字符串
    :param alignment:
        'l': 左对齐
        'r': 右对齐
        'c': 居中对齐
    :param ambiguous_as_wide: 参考 strwidth 的参数
    :param max_width: 限制每列最大宽度
    :param return_col_width: 是否返回列宽

    >>> listalign(['a', 'bb', 'ccc'])
    ['a  ', 'bb ', 'ccc']
    >>> listalign(['a', 'bb', 'ccc'], 'r')
    ['  a', ' bb', 'ccc']
    """
    # 1 计算最大宽度
    width_func = get_strwidth(ambiguous_as_wide)
    # col_width = max([width_func(str(x)) for x in arr]) if arr else 0
    # 上面这种写法在空列表时会报错，改成下面这样
    col_width = 0
    for x in arr:
        w = width_func(str(x))
        if w > col_width:
            col_width = w

    if max_width and col_width > max_width:
        col_width = max_width

    # 2 对齐处理
    res = []
    for x in arr:
        s = str(x)
        w = width_func(s)
        if w < col_width:
            pad = col_width - w
            if alignment == 'l':
                s = s + ' ' * pad
            elif alignment == 'r':
                s = ' ' * pad + s
            elif alignment == 'c':
                left = pad // 2
                right = pad - left
                s = ' ' * left + s + ' ' * right
        elif max_width and w > max_width:
            # 需要截断，这里简单处理，可能需要更复杂的截断逻辑
            s = east_asian_shorten(s, max_width, ambiguous_as_wide=ambiguous_as_wide)

        res.append(s)

    if return_col_width:
        return res, col_width
    return res


def arr_hangclear(arr):
    """二维数组行清理，把全空的行去掉"""
    return [x for x in arr if any(map(str, x))]


def arr2table(arr, alignment='l',
              ambiguous_as_wide=False, max_width=None):
    """ 二维数组转表格字符串

    :param arr: 二维数组
    :param alignment: 对齐方式，可以是一个字符统一设置，也可以是字符串分别设置每列
    :param ambiguous_as_wide: 参考 strwidth 的参数
    :param max_width: 限制每列最大宽度，可以是单个整数，也可以是整数列表
    """
    if not arr: return ''
    n_cols = len_in_dim2(arr)
    if n_cols == 0: return ''

    # 1 补全数组
    # 使用GrowingList来处理可能得缺省值
    # 但是这里为了性能和逻辑简单，先转成标准的list of list
    data = []
    for row in arr:
        if isinstance(row, (list, tuple)):
            data.append(list(row) + [''] * (n_cols - len(row)))
        else:
            data.append([row] + [''] * (n_cols - 1))

    # 2 计算每列宽度
    col_widths = [0] * n_cols
    width_func = get_strwidth(ambiguous_as_wide)
    for row in data:
        for j, val in enumerate(row):
            w = width_func(str(val))
            if w > col_widths[j]:
                col_widths[j] = w

    # 3 处理最大宽度限制
    if max_width:
        if isinstance(max_width, int):
            max_widths = [max_width] * n_cols
        else:
            max_widths = list(max_width) + [None] * (n_cols - len(max_width))
        for j in range(n_cols):
            if max_widths[j] and col_widths[j] > max_widths[j]:
                col_widths[j] = max_widths[j]

    # 4 处理对齐方式
    if len(alignment) == 1:
        alignments = [alignment] * n_cols
    else:
        alignments = list(alignment) + ['l'] * (n_cols - len(alignment))

    # 5 生成表格
    res_lines = []
    for row in data:
        line_parts = []
        for j, val in enumerate(row):
            s = str(val)
            w = width_func(s)
            cw = col_widths[j]
            align = alignments[j]

            if w < cw:
                pad = cw - w
                if align == 'l':
                    s = s + ' ' * pad
                elif align == 'r':
                    s = ' ' * pad + s
                elif align == 'c':
                    left = pad // 2
                    right = pad - left
                    s = ' ' * left + s + ' ' * right
            elif w > cw:
                s = east_asian_shorten(s, cw, ambiguous_as_wide=ambiguous_as_wide)

            line_parts.append(s)
        res_lines.append(' '.join(line_parts))

    return '\n'.join(res_lines)


def xldictstr(d, *, key_width=None, value_width=None):
    """将字典格式化为对齐的字符串"""
    if not d: return '{}'
    keys = [str(k) for k in d.keys()]
    values = [str(v) for v in d.values()]

    if key_width is None:
        keys_aligned, kw = listalign(keys, return_col_width=True)
    else:
        keys_aligned = listalign(keys, max_width=key_width)

    if value_width is None:
        # values不需要对齐，只需要处理换行等
        pass
    else:
        values = [east_asian_shorten(v, value_width) for v in values]

    res = []
    for k, v in zip(keys_aligned, values):
        res.append(f'{k}: {v}')
    return '\n'.join(res)


def fold_dict(d, n=0):
    """ 将字典显示折叠

    :param n: 缩进格数
    """
    s = str(d)
    if len(s) < 100:
        return s
    # 否则按键值对显示
    res = []
    indent = ' ' * n
    for k, v in d.items():
        res.append(f'{indent}{k}: {v}')
    return '\n'.join(res)


class ListingFormat:
    """ 列表编号格式化工具 """

    def __init__(self, fmt='{}. '):
        self.fmt = fmt
        self.idx = 0

    def __call__(self, s):
        self.idx += 1
        return self.fmt.format(self.idx) + str(s)


class BookContents:
    """ 书本目录类 """

    def __init__(self):
        self.contents = []  # 目录条目按顺序保存在list中

    def add(self, level, title, page=None):
        """
        Args:
            level:
            title:
            page: 不一定要放整数的页数，也可以放其他一些比例之类的数值

        Returns:

        """
        self.contents.append([level, title, page])

    def format_numbers(self, number='normal', *, indent='', start_level=1, jump=False):
        """ 每级目录的编号

        :param number: 编号格式，目前有默认方式，以后有需要可以扩展其他模式
        :param start_level: 开始展示的层级（高层级也会展示，只是不带编号和缩进）
            可以设为负数，表示自动推算，比如-1
        :param jump: 支持跳级，比如2级"3"，跳到4级本来是"3.0.1"，但开启该参数则会优化为"3.1"
        :return: list，跟contents等长，表示每个标题的编号，可能为空''
        """
        # 1
        if start_level == -1:
            # 自动推算合适的开始编号
            # -1模式，表示第一个不只一项的level
            levels = [x[0] for x in self.contents]
            levels_cnt = collections.Counter(levels)
            for i in range(min(levels), max(levels) + 1):
                if levels_cnt[i] > 1:
                    start_level = i
                    break

        # 2
        ls = []
        ct = collections.defaultdict(int)
        for x in self.contents:
            # print(x)
            level = x[0]
            sign = indent * (level - start_level)

            # 处理计数器
            ct[level] += 1
            for k, v in ct.items():
                if k > level:
                    ct[k] = 0

            # 当前编号
            if number == 'normal':
                numbers = [ct[i] for i in range(start_level, level + 1)]
                if jump:  # 过滤0
                    numbers = [x for x in numbers if x]
                sign += '.'.join(map(str, numbers))
            else:
                pass

            ls.append(sign)

        return ls

    def format_str(self, indent='\t', *, number='normal', page=False, start_level=1, jump=False):
        """ 转文本展示

        :param indent: 每级展示的缩进量
        :param page: 是否展示页码
        """
        numbers = self.format_numbers(number, indent=indent, start_level=start_level, jump=jump)

        # 2
        ls = []
        for num, x in zip(numbers, self.contents):
            level, title, page_ = x

            # 标题
            if level < start_level:
                sign = title
            else:
                sign = ' '.join([num, x[1]])

            # 加后缀
            if page:
                sign += f'，{page_}'

            ls.append(sign)

        return '\n'.join(ls)
