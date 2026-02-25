#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 11:00

import io
import math
import pprint
from collections import Counter

from pyxllib.prog.basic import typename, safe_div


def round_int(x, *, ndim=0):
    """ 先四舍五入，再取整

    :param x: 一个数值，或者多维数组
    :param ndim: x是数值是默认0，否则指定数组维度，批量处理
        比如ndim=1是一维数组
        ndim=2是二维数组

    >>> round_int(1.5)
    2
    >>> round_int(1.4)
    1
    >>> round_int([2.3, 1.42], ndim=1)
    [2, 1]
    >>> round_int([[2.3, 1.42], [3.6]], ndim=2)
    [[2, 1], [4]]
    """
    if ndim:
        return [round_int(a, ndim=ndim - 1) for a in x]
    else:
        return int(round(x, 0))


def human_readable_size(n):
    """ 我个人习惯常用的size显示方式 """
    for u in [' B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f'{round_int(n)}{u}'
        else:
            n /= 1024
    else:
        return f'{round_int(n)}TB'


def human_readable_number(value, base_type='K', precision=4):
    """ 数字美化输出函数

    :param float|int value: 要转换的数值
    :param int precision: 有效数字的长度
    :param str base_type: 进制类型，'K'为1000进制, 'KB'为1024进制（KiB同理）, '万'为中文万进制
    :return: 美化后的字符串
    """
    if value is None:
        return ''

    if abs(value) < 1:
        return f'{value:.{precision}g}'

    # 设置不同进制的单位和基数
    units, base = {
        'K': (['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'], 1000),
        'KB': (['', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'], 1024),
        'KiB': (['', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'], 1024),
        '万': (['', '万', '亿', '万亿', '亿亿'], 10000),
        '秒': (['秒', [60, '分'], [60, '时'], [24, '天'], [7, '周'], [4.345, '月'], [12, '年']], 60),
    }.get(base_type, ([''], 1))  # 默认为空单位和基数1

    x, i = abs(value), 0
    while x >= base and i < len(units) - 1:
        x /= base
        i += 1
        if isinstance(units[i], list):
            base = units[i][0]
            units[i] = units[i][1]

    x = f'{x:.{precision}g}'  # 四舍五入到指定精度
    prefix = '-' if value < 0 else ''  # 负数处理
    return f"{prefix}{x}{units[i]}"


def xl_format_g(x, p=3):
    """ 普通format的g模式不太满足我的需求，做了点修改

    注：g是比较方便的一种数值格式化方法，会比较智能地判断是否整数显示，或者普通显示、科学计数法显示

    :param x: 数值x
    """
    s = f'{x:g}'
    if 'e' in s:
        # 如果变成了科学计数法，明确保留3位有效数字
        return '{:.{}g}'.format(x, p=3)
    else:
        # 否则返回默认的g格式
        return s


def print2string(*args, **kwargs):
    """https://stackoverflow.com/questions/39823303/python3-print-to-string"""
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


def prettifystr(s):
    """对一个对象用更友好的方式字符串化

    :param s: 输入类型不做限制，会将其以友好的形式格式化
    :return: 格式化后的字符串
    """
    title = ''
    if isinstance(s, str):
        pass
    elif isinstance(s, Counter):  # Counter要按照出现频率显示
        li = s.most_common()
        title = f'collections.Counter长度：{len(s)}\n'
        # 不使用复杂的pd库，先简单用pprint即可
        # df = pd.DataFrame.from_records(s, columns=['value', 'count'])
        # s = dataframe_str(df)
        s = pprint.pformat(li)
    elif isinstance(s, (list, tuple)):
        title = f'{typename(s)}长度：{len(s)}\n'
        s = pprint.pformat(s)
    elif isinstance(s, (dict, set)):
        title = f'{typename(s)}长度：{len(s)}\n'
        s = pprint.pformat(s)
    else:  # 其他的采用默认的pformat
        s = pprint.pformat(s)
    return title + s


class PrettifyStrDecorator:
    """将函数的返回值字符串化（调用 prettifystr 美化）"""

    def __init__(self, func):
        self.func = func  # 使用self.func可以索引回原始函数名称
        self.last_raw_res = None  # last raw result，上一次执行函数的原始结果

    def __call__(self, *args, **kwargs):
        self.last_raw_res = self.func(*args, **kwargs)
        return prettifystr(self.last_raw_res)


def get_number_width(n):
    """ 判断数值n的长度

    参考资料：https://jstrieb.github.io/posts/digit-length/

    >>> get_number_width(0)
    1
    >>> get_number_width(9)
    1
    >>> get_number_width(10)
    2
    >>> get_number_width(97)
    2
    """
    # assert n > 0
    # return math.ceil(math.log10(n + 1))

    return 1 if n == 0 else (math.floor(math.log10(n)) + 1)


def aligned_range(start, stop=None, step=1):
    """ 返回按照域宽对齐的数字序列 """
    if stop is None:
        start, stop = 0, start

    max_width = get_number_width(stop - step)
    format_str = '{:0' + str(max_width) + 'd}'

    return [format_str.format(i) for i in range(start, stop, step)]


def percentage_and_value(numbers, precision=2, *, total=None, sep='.'):
    """ 对输入的一串数值，转换成一种特殊的表达格式 "百分比.次数"

    :param list numbers: 数值列表
    :param int precision: 百分比的精度（小数点后的位数），默认为 2
    :param int total: 总数，如果不输入，则默认为输入数值的和
    :param str sep: 分隔符
    :return: 整数部分是比例，小数部分是原始数值的整数部分
    """
    if total is None:
        total = sum(numbers)
    width = get_number_width(total)
    result = []
    for num in numbers:
        percent = safe_div(num, total) * 10 ** precision
        result.append(f"{percent:.0f}{sep}{num:0{width}d}")
    return result
