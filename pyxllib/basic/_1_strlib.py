#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/06/01


"""
TODO 这个文件结构还有点乱，需要梳理
"""

import copy
import io
import logging
import math
import pprint
import re
import socket
import sys
import textwrap

from disjoint_set import DisjointSet

HOSTNAME = socket.getfqdn()

____str_funcs = """
字符串类的一些辅助函数
"""


def shorten(s, width=200, placeholder='...'):
    """
    :param width: 这个长度是上限，即使用placeholder时的字符串总长度也在这个范围内

    >>> shorten('aaa', 10)
    'aaa'
    >>> shorten('hell world! 0123456789 0123456789', 11)
    'hell wor...'
    >>> shorten("Hello  world!", width=12)
    'Hello world!'
    >>> shorten("Hello  world!", width=11)
    'Hello wo...'
    >>> shorten('0123456789 0123456789', 2, 'xyz')  # 自己写的shorten
    'xy'

    注意textwrap.shorten的缩略只针对空格隔开的单词有效，我这里的功能与其不太一样
    >>> textwrap.shorten('0123456789 0123456789', 11)  # 全部字符都被折叠了
    '[...]'
    >>> shorten('0123456789 0123456789', 11)  # 自己写的shorten
    '01234567...'
    """
    s = re.sub(r'\s+', ' ', str(s))
    n, m = len(s), len(placeholder)
    if n > width:
        s = s[:max(width - m, 0)] + placeholder
    return s[:width]  # 加了placeholder在特殊情况下也会超，再做个截断最保险

    # return textwrap.shorten(str(s), width)


def natural_sort_key(key):
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


def strwidth(s):
    """ string width

    中英字符串实际宽度
    >>> strwidth('ab')
    2
    >>> strwidth('a⑪中⑩')
    7

    ⑩等字符的宽度还是跟字体有关的，不过在大部分地方好像都是域宽2，目前算法问题不大
    """
    try:
        res = len(s.encode('gbk'))
    except UnicodeEncodeError:
        count = len(s)
        for x in s:
            if ord(x) > 127:
                count += 1
        res = count
    return res


def strwidth_proc(s, fmt='r', chinese_char_width=1.8):
    """ 此函数可以用于每个汉字域宽是w=1.8等奇怪的情况

    为了让字符串域宽为一个整数，需要补充中文空格，会对原始字符串进行修改。
    故返回值有2个，第1个是修正后的字符串s，第2个是实际宽度w。

    :param s: 一个字符串
    :param fmt: 目标对齐格式
    :param chinese_char_width: 每个汉字字符宽度
    :return: (s, w)
        s: 修正后的字符串值s
        w: 修正后字符串的实际宽度

    >>> strwidth_proc('哈哈a', chinese_char_width=1.8)
    ('　　　哈哈a', 10)
    """
    # 1 计算一些参数值
    s = str(s)  # 确保是字符串类型
    l1 = len(s)
    l2 = strwidth(s)
    y = l2 - l1  # 中文字符数
    x = l1 - y  # 英文字符数
    ch = chr(12288)  # 中文空格
    w = x + y * chinese_char_width  # 当前字符串宽度
    # 2 计算需要补充t个中文空格
    error = 0.05  # 允许误差范围
    t = 0  # 需要补充中文字符数
    while error < w % 1 < 1 - error:  # 小数部分超过误差
        t += 1
        w += chinese_char_width
    # 3 补充中文字符
    if t:
        if fmt == 'r':
            s = ch * t + s
        elif fmt == 'l':
            s = s + ch * t
        else:
            s = ch * (t - t // 2) + s + ch * (t // 2)
    return s, int(w)


def strfind(fullstr, objstr, *, start=None, times=0, overlap=False):
    r"""进行强大功能扩展的的字符串查找函数

    TODO 性能有待优化

    :param fullstr: 原始完整字符串
    >>> strfind('aabbaabb', 'bb')  # 函数基本用法
    2

    :param objstr: 需要查找的目标字符串，可以是一个list或tuple
    TODO 有空看下AC自动机，看这里是否可以优化提速，或者找现成的库接口
    >>> strfind('bbaaaabb', 'bb') # 查找第1次出现的位置
    0
    >>> strfind('aabbaabb', 'bb', times=1) # 查找第2次出现的位置
    6
    >>> strfind('aabbaabb', 'cc') # 不存在时返回-1
    -1
    >>> strfind('aabbaabb', ['aa', 'bb'], times=2)
    4

    :param start: 起始查找位置。默认值为0，当times<0时start的默认值为-1。
    >>> strfind('aabbaabb', 'bb', start=2) # 恰好在起始位置
    2
    >>> strfind('aabbaabb', 'bb', start=3)
    6
    >>> strfind('aabbaabb', ['aa', 'bb'], start=5)
    6

    :param times: 定位第几次出现的位置，默认值为0，即从前往后第1次出现的位置。
        如果是负数，则反向查找，并返回的是目标字符串的起始位置。
    >>> strfind('aabbaabb', 'aa', times=-1)
    4
    >>> strfind('aabbaabb', 'aa', start=5, times=-1)
    4
    >>> strfind('aabbaabb', 'aa', start=3, times=-1)
    0
    >>> strfind('aabbaabb', 'bb', start=7, times=-1)
    6

    :param overlap: 重叠情况是否重复计数
    >>> strfind('aaaa', 'aa', times=1)  # 默认不计算重叠部分
    2
    >>> strfind('aaaa', 'aa', times=1, overlap=True)
    1

    >>> strfind(r'\item=\item+', (r'\item', r'\test'), start=1)
    6
    """

    def nonnegative_min_value(*arr):
        """计算出最小非负整数，如果没有非负数，则返回-1"""
        arr = tuple(filter(lambda x: x >= 0, arr))
        return min(arr) if arr else -1

    def nonnegative_max_value(*arr):
        """计算出最大非负整数，如果没有非负数，则返回-1"""
        arr = tuple(filter(lambda x: x >= 0, arr))
        return max(arr) if arr else -1

    # 1 根据times不同，start的初始默认值设置方式也不同
    if times < 0 and start is None:
        start = len(fullstr) - 1  # 反向查找start设到末尾字符-1
    if start is None:
        start = 0  # 正向查找start设为0
    p = -1  # 记录答案位置，默认找不到

    # 2 单串匹配
    if isinstance(objstr, str):  # 单串匹配
        offset = 1 if overlap else len(objstr)  # overlap影响每次偏移量

        # A、正向查找
        if times >= 0:
            p = start - offset
            for _ in range(times + 1):
                p = fullstr.find(objstr, p + offset)
                if p == -1:
                    return -1

        # B、反向查找
        else:
            p = start + offset + 1
            for _ in range(-times):
                p = fullstr.rfind(objstr, 0, p - offset)
                if p == -1:
                    return -1

    # 3 多模式匹配（递归调用，依赖单串匹配功能）
    else:
        # A、正向查找
        if times >= 0:
            p = start - 1
            for _ in range(times + 1):
                # 把每个目标串都找一遍下一次出现的位置，取最近的一个
                #   因为只找第一次出现的位置，所以overlap参数传不传都没有影响
                # TODO 需要进行性能对比分析，有必要的话后续可以改AC自动机实现多模式匹配
                ls = tuple(map(lambda x: strfind(fullstr, x, start=p + 1, overlap=overlap), objstr))
                p = nonnegative_min_value(*ls)
                if p == -1:
                    return -1

        # B、反向查找
        else:
            p = start + 1
            for _ in range(-times):  # 需要循环处理的次数
                # 使用map对每个要查找的目标调用strfind
                ls = tuple(map(lambda x: strfind(fullstr, x, start=p - 1, times=-1, overlap=overlap), objstr))
                p = nonnegative_max_value(*ls)
                if p == -1:
                    return -1

    return p


def typename(c):
    """ 简化输出的type类型

    >>> typename(123)
    'int'
    """
    return str(type(c))[8:-2]


____str = """
文本处理相关功能
"""


class StrDecorator:
    """将函数的返回值字符串化，仅调用朴素的str字符串化

    装饰器开发可参考： https://mp.weixin.qq.com/s/Om98PpncG52Ba1ZQ8NIjLA
    """

    def __init__(self, func):
        self.func = func  # 使用self.func可以索引回原始函数名称
        self.last_raw_res = None  # last raw result，上一次执行函数的原始结果

    def __call__(self, *args, **kwargs):
        self.last_raw_res = self.func(*args, **kwargs)
        return str(self.last_raw_res)


class PrintDecorator:
    """将函数返回结果直接输出"""

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        s = self.func(*args, **kwargs)
        print(s)
        return s  # 输出后仍然会返回原函数运行值


def realign(text, least_blank=4, tab2blank=4, support_chinese=False, sep=None):
    r"""
    :param text: 一段文本
        支持每行列数不同
    :param least_blank: 每列最少间距空格数
    :param tab2blank:
    :param support_chinese: 支持中文域宽计算
    :param sep: 每列分隔符，默认为least_blank个空格
    :return: 对齐美化的一段文本

    >>> realign('  Aget      keep      hold         show\nmaking    selling    giving    collecting')
    'Aget      keep       hold      show\nmaking    selling    giving    collecting'
    """
    # 1 预处理
    s = text.replace('\t', ' ' * tab2blank)
    s = re.sub(' {' + str(least_blank) + ',}', r'\t', s)  # 统一用\t作为分隔符
    lenfunc = strwidth if support_chinese else len
    if sep is None: sep = ' ' * least_blank

    # 2 计算出每一列的最大宽度
    lines = s.splitlines()
    n = len(lines)
    max_width = GrowingList()  # 因为不知道有多少列，用自增长的list来存储每一列的最大宽度
    for i, line in enumerate(lines):
        line = line.strip().split('\t')
        m = len(line)
        for j in range(m): max_width[j] = max(max_width[j] or 0, lenfunc(line[j]))
        lines[i] = line
    if len(max_width) == 1: return '\n'.join(map(lambda x: x[0], lines))

    # 3 重组内容
    for i, line in enumerate(lines):
        for j in range(len(line) - 1): line[j] += ' ' * (max_width[j] - lenfunc(line[j]))  # 注意最后一列就不用加空格了
        lines[i] = sep.join(line)
    return '\n'.join(lines)


class Stdout:
    """重定向标准输出流，切换print标准输出位置

    使用with语法调用
    """

    def __init__(self, path=None, mode='w'):
        """
        :param path: 可选参数
            如果是一个合法的文件名，在__exit__时，会将结果写入文件
            如果不合法不报错，只是没有功能效果
        :param mode: 写入模式
            'w': 默认模式，直接覆盖写入
            'a': 追加写入
        """
        self.origin_stdout = sys.stdout
        self._path = path
        self._mode = mode
        self.strout = io.StringIO()
        self.result = None

    def __enter__(self):
        sys.stdout = self.strout
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.origin_stdout
        self.result = str(self)

        # 如果输入的是一个合法的文件名，则将中间结果写入
        if not self._path:
            return

        try:
            with open(self._path, self._mode) as f:
                f.write(self.result)
        except TypeError as e:
            logging.exception(e)
        except FileNotFoundError as e:
            logging.exception(e)

        self.strout.close()

    def __str__(self):
        """在这个期间获得的文本内容"""
        if self.result:
            return self.result
        else:
            return self.strout.getvalue()


def listalign(ls, fmt='r', *, width=None, fillchar=' ', prefix='', suffix='', chinese_char_width=2):
    """文档： https://blog.csdn.net/code4101/article/details/80985218（不过文档有些过时了）

    listalign列表对齐
    py3中str的len是计算字符数量，例如len('ab') --> 2， len('a中b') --> 3。
    但在对齐等操作中，是需要将每个汉字当成宽度2来处理，计算字符串实际宽度的。
    所以我们需要开发一个strwidth函数，效果： strwidth('ab') --> 2，strwidth('a中b') --> 4。

    :param ls:
        要处理的列表，会对所有元素调用str处理，确保全部转为string类型
            且会将换行符转为\n显示
    :param fmt: （format）
        l: left，左对齐
        c: center，居中
        r: right，右对齐
        多个字符: 扩展fmt长度跟ls一样，每一个元素单独设置对齐格式。如果fmt长度小于ls，则扩展的格式按照fmt[-1]设置
    :param width:
        None或者设置值小于最长字符串: 不设域宽，直接按照最长的字符串为准
    :param fillchar: 填充字符
    :param prefix: 添加前缀
    :param suffix: 添加后缀
    :param chinese_char_width: 每个汉字字符宽度

    :return:
        对齐后的数组ls，每个元素会转为str类型

    >>> listalign(['a', '哈哈', 'ccd'])
    ['   a', '哈哈', ' ccd']
    >>> listalign(['a', '哈哈', 'ccd'], chinese_char_width=1.8)
    ['        a', '　　　哈哈', '      ccd']
    """
    # 1 处理fmt数组
    if len(fmt) == 1:
        fmt = [fmt] * len(ls)
    elif len(fmt) < len(ls):
        fmt = list(fmt) + [fmt[-1]] * (len(ls) - len(fmt))

    # 2 算出需要域宽
    if chinese_char_width == 2:
        strs = [str(x).replace('\n', r'\n') for x in ls]  # 存储转成字符串的元素
        lens = [strwidth(x) for x in strs]  # 存储每个元素的实际域宽
    else:
        strs = []  # 存储转成字符串的元素
        lens = []  # 存储每个元素的实际域宽
        for i, t in enumerate(ls):
            t, n = strwidth_proc(t, fmt[i], chinese_char_width)
            strs.append(t)
            lens.append(n)
    w = max(lens)
    if width and isinstance(width, int) and width > w:
        w = width

    # 3 对齐操作
    for i, s in enumerate(strs):
        if fmt[i] == 'r':
            strs[i] = fillchar * (w - lens[i]) + strs[i]
        elif fmt[i] == 'l':
            strs[i] = strs[i] + fillchar * (w - lens[i])
        elif fmt[i] == 'c':
            t = w - lens[i]
            strs[i] = fillchar * (t - t // 2) + strs[i] + fillchar * (t // 2)
        strs[i] = prefix + strs[i] + suffix
    return strs


def len_in_dim2_min(arr):
    """ 计算类List结构在第2维上的最小长度

    >>> len_in_dim2([[1,1], [2], [3,3,3]])
    3

    >>> len_in_dim2([1, 2, 3])  # TODO 是不是应该改成0合理？但不知道牵涉到哪些功能影响
    1
    """
    if not isinstance(arr, (list, tuple)):
        raise TypeError('类型错误，不是list构成的二维数组')

    # 找出元素最多的列
    column_num = math.inf
    for i, item in enumerate(arr):
        if isinstance(item, (list, tuple)):  # 该行是一个一维数组
            column_num = min(column_num, len(item))
        else:  # 如果不是数组，是指单个元素，当成1列处理
            column_num = min(column_num, 1)
            break  # 只要有个1，最小长度就一定是1了

    return column_num


def len_in_dim2(arr):
    """ 计算类List结构在第2维上的最大长度

    >>> len_in_dim2([[1,1], [2], [3,3,3]])
    3

    >>> len_in_dim2([1, 2, 3])  # TODO 是不是应该改成0合理？但不知道牵涉到哪些功能影响
    1
    """
    if not isinstance(arr, (list, tuple)):
        raise TypeError('类型错误，不是list构成的二维数组')

    # 找出元素最多的列
    column_num = 0
    for i, item in enumerate(arr):
        if isinstance(item, (list, tuple)):  # 该行是一个一维数组
            column_num = max(column_num, len(item))
        else:  # 如果不是数组，是指单个元素，当成1列处理
            column_num = max(column_num, 1)

    return column_num


def ensure_array(arr, default_value=''):
    """对一个由list、tuple组成的二维数组，确保所有第二维的列数都相同

    >>> ensure_array([[1,1], [2], [3,3,3]])
    [[1, 1, ''], [2, '', ''], [3, 3, 3]]
    """
    max_cols = len_in_dim2(arr)
    if max_cols == 1:
        return arr
    dv = str(default_value)
    a = [[]] * len(arr)
    for i, ls in enumerate(arr):
        if isinstance(ls, (list, tuple)):
            t = list(arr[i])
        else:
            t = [ls]  # 如果不是数组，是指单个元素，当成1列处理
        a[i] = t + [dv] * (max_cols - len(t))  # 左边的写list，是防止有的情况是tuple，要强制转list后拼接
    return a


def swap_rowcol(a, *, ensure_arr=False, default_value=''):
    """矩阵行列互换

    注：如果列数是不均匀的，则会以最小列数作为行数

    >>> swap_rowcol([[1,2,3], [4,5,6]])
    [[1, 4], [2, 5], [3, 6]]
    """
    if ensure_arr:
        a = ensure_array(a, default_value)
    # 这是非常有教学意义的行列互换实现代码
    return list(map(list, zip(*a)))


def int2excel_col_name(d):
    """
    >>> int2excel_col_name(1)
    'A'
    >>> int2excel_col_name(28)
    'AB'
    >>> int2excel_col_name(100)
    'CV'
    """
    s = []
    while d:
        t = (d - 1) % 26
        s.append(chr(65 + t))
        d = (d - 1) // 26
    return ''.join(reversed(s))


def excel_col_name2int(s):
    """
    >>> excel_col_name2int('A')
    1
    >>> excel_col_name2int('AA')
    27
    >>> excel_col_name2int('AB')
    28
    """
    d = 0
    for ch in s:
        d = d * 26 + (ord(ch) - 64)
    return d


def int2myalphaenum(n):
    """
    :param n: 0~52的数字
    """
    if 0 <= n <= 52:
        return '_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[n]
    else:
        print('警告：不在处理范围内的数值', n)
        raise ValueError


def gentuple(n, tag):
    """有点类似range函数，但生成的数列更加灵活
    :param n:
        数组长度
    :param tag:
        整数，从指定整数开始编号
        int类型，从指定数字开始编号
            0，从0开始编号
            1，从1开始编号
        'A'，用Excel的形式编号
        tuple，按枚举值循环显示
            ('A', 'B')：循环使用A、B编号

    >>> gentuple(4, 'A')
    ('A', 'B', 'C', 'D')
    """
    a = [''] * n
    if isinstance(tag, int):
        for i in range(n):
            a[i] = i + tag
    elif tag == 'A':
        a = tuple(map(lambda x: int2excel_col_name(x + 1), range(n)))
    elif isinstance(tag, (list, tuple)):
        k = len(tag)
        a = tuple(map(lambda x: tag[x % k], range(n)))
    return a


def ensure_gbk(s):
    """检查一个字符串的所有内容是否能正常转为gbk，
    如果不能则ignore掉不能转换的部分"""
    try:
        s.encode('gbk')
    except UnicodeEncodeError:
        origin_s = s
        s = s.encode('gbk', errors='ignore').decode('gbk')
        print('警告：字符串存在无法转为gbk的字符', origin_s, s)
    return s


def funcmsg(func):
    """输出函数func所在的文件、函数名、函数起始行"""
    # showdir(func)
    if not hasattr(func, '__name__'):  # 没有__name__属性表示这很可能是一个装饰器去处理原函数了
        if hasattr(func, 'func'):  # 我的装饰器常用func成员存储原函数对象
            func = func.func
        else:
            return f'装饰器：{type(func)}，无法定位'
    return f'函数名：{func.__name__}，来自文件：{func.__code__.co_filename}，所在行号={func.__code__.co_firstlineno}'


class GrowingList(list):
    """可变长list"""

    def __init__(self, default_value=None):
        super().__init__(self)
        self.default_value = default_value

    def __getitem__(self, index):
        if index >= len(self):
            self.extend([self.default_value] * (index + 1 - len(self)))
        return list.__getitem__(self, index)

    def __setitem__(self, index, value):
        if index >= len(self):
            self.extend([self.default_value] * (index + 1 - len(self)))
        list.__setitem__(self, index, value)


def arr_hangclear(arr, depth=None):
    """ 清除连续相同值，简化表格内容
    >> arr_hangclear(arr, depth=2)
    原表格：
        A  B  D
        A  B  E
        A  C  E
        A  C  E
    新表格：
        A  B  D
              E
           C  E
              E

    :param arr: 二维数组
    :param depth: 处理列上限
        例如depth=1，则只处理第一层
        depth=None，则处理所有列

    >>> arr_hangclear([[1, 2, 4], [1, 2, 5], [1, 3, 5], [1, 3, 5]])
    [[1, 2, 4], ['', '', 5], ['', 3, 5], ['', '', 5]]
    >>> arr_hangclear([[1, 2, 4], [1, 2, 5], [2, 2, 5], [1, 2, 5]])
    [[1, 2, 4], ['', '', 5], [2, 2, 5], [1, 2, 5]]
    """
    m = depth or len_in_dim2(arr) - 1
    a = copy.deepcopy(arr)

    # 算法原理：从下到上，从右到左判断与上一行重叠了几列数据
    for i in range(len(arr) - 1, 0, -1):
        for j in range(m):
            if a[i][j] == a[i - 1][j]:
                a[i][j] = ''
            else:
                break
    return a


def arr2table(arr, rowmerge=False):
    """数组转html表格代码
    :param arr:  需要处理的数组
    :param rowmerge: 行单元格合并
    :return: html文本格式的<table>

    这个arr2table是用来画合并单元格的
    >> chrome(arr2table([['A', 1, 'a'], ['', 2, 'b'], ['B', 3, 'c'], ['', '', 'd'], ['', 5, 'e']], True), 'a.html')
    效果图：http://i1.fuimg.com/582188/c452f40b5a072f8d.png
    """
    n = len(arr)
    m = len_in_dim2(arr)
    res = ['<table border="1"><tbody>']
    for i, line in enumerate(arr):
        res.append('<tr>')
        for j, ele in enumerate(line):
            if rowmerge:
                if ele != '':
                    cnt = 1
                    while i + cnt < n and arr[i + cnt][j] == '':
                        for k in range(j - 1, -1, -1):
                            if arr[i + cnt][k] != '':
                                break
                        else:
                            cnt += 1
                            continue
                        break
                    if cnt > 1:
                        res.append(f'<td rowspan="{cnt}">{ele}</td>')
                    else:
                        res.append(f'<td>{ele}</td>')
                elif j == m - 1:
                    res.append(f'<td>{ele}</td>')
            else:
                res.append(f'<td>{ele}</td>')
        res.append('</tr>')
    res.append('</tbody></table>')
    return ''.join(res)


def digit2weektag(d):
    """输入数字1~7，转为“周一~周日”

    >>> digit2weektag(1)
    '周一'
    >>> digit2weektag('7')
    '周日'
    """
    d = int(d)
    if 1 <= d <= 7:
        return '周' + '一二三四五六日'[d - 1]
    else:
        raise ValueError


def print2string(*args, **kwargs):
    """https://stackoverflow.com/questions/39823303/python3-print-to-string"""
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


def fullwidth2halfwidth(ustring):
    """ 把字符串全角转半角

    python3环境下的全角与半角转换代码和测试_大数据挖掘SparkExpert的博客-CSDN博客:
    https://blog.csdn.net/sparkexpert/article/details/82749207

    >>> fullwidth2halfwidth("你好ｐｙｔｈｏｎａｂｄａｌｄｕｉｚｘｃｖｂｎｍ")
    '你好pythonabdalduizxcvbnm'
    """
    ss = []
    for s in ustring:
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            ss.append(chr(inside_code))
    return ''.join(ss)


def fullwidth2halfwidth2(ustring):
    """ 不处理标点符号的版本

    >>> fullwidth2halfwidth2("你好ｐｙｔｈｏｎａｂｄａ，ｌｄｕｉｚｘｃｖｂｎｍ")
    '你好pythonabda，lduizxcvbnm'
    """
    ss = []
    for s in ustring:
        for uchar in s:
            if uchar in '：；！（），？＂．':
                ss.append(uchar)
            else:
                inside_code = ord(uchar)
                if inside_code == 12288:  # 全角空格直接转换
                    inside_code = 32
                elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
                    inside_code -= 65248
                ss.append(chr(inside_code))
    return ''.join(ss)


def halfwidth2fullwidth(ustring):
    """ 把字符串全角转半角

    >>> halfwidth2fullwidth("你好ｐｙｔｈｏｎａｂｄａｌｄｕｉｚｘｃｖｂｎｍ")
    '你好ｐｙｔｈｏｎａｂｄａｌｄｕｉｚｘｃｖｂｎｍ'
    """
    ss = []
    for s in ustring:
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 32:  # 全角空格直接转换
                inside_code = 12288
            elif 33 <= inside_code <= 126:  # 全角字符（除空格）根据关系转化
                inside_code += 65248
            ss.append(chr(inside_code))
    return ''.join(ss)


def sort_by_given_list(a, b):
    r"""本函数一般用在数据透视表中，分组中元素名为中文，没有按指定规律排序的情况
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


____disjoint_set = """
并查集相关功能
"""


def disjoint_set(items, join_checker):
    """ 按照一定的相连规则分组

    :param items: 项目清单
    :param join_checker: 检查任意两个对象是否相连，进行分组
    :return:

    算法：因为会转成下标，按照下标进行分组合并，所以支持items里有重复值，或者unhashable对象

    >>> disjoint_set([-1, -2, 2, 0, 0, 1], lambda x, y: x*y>0)
    [[-1, -2], [2, 1], [0], [0]]
    """
    from itertools import combinations

    # 1 添加元素
    ds = DisjointSet()
    items = tuple(items)
    n = len(items)
    for i in range(n):
        ds.find(i)

    # 2 连接、分组
    for i, j in combinations(range(n), 2):
        if join_checker(items[i], items[j]):
            ds.union(i, j)

    # 3 返回分组信息
    res = []
    for group in ds.itersets():
        group_elements = [items[g] for g in group]
        res.append(group_elements)
    return res


____other = """
"""
