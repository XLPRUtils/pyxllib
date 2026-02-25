#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

import collections
import re


def bracket_match(s, idx):
    """括号匹配位置
    这里以{、}为例，注意也要适用于'[]', '()'
    >>> bracket_match('{123}', 0)
    4
    >>> bracket_match('0{23{5}}89', 1)
    7
    >>> bracket_match('0{23{5}}89', 7)
    1
    >>> bracket_match('0{23{5}78', 1) is None
    True
    >>> bracket_match('0{23{5}78', 20) is None
    True
    >>> bracket_match('0[2[4]{7}]01', 9)
    1
    >>> bracket_match('0{[34{6}89}', -4)
    5
    """
    key = '{[(<>)]}'
    try:
        if idx < 0:
            idx += len(s)
        ch1 = s[idx]
        idx1 = key.index(ch1)
    except ValueError:  # 找不到ch1
        return None
    except IndexError:  # 下标越界，表示没有匹配到右括号
        return None
    idx2 = len(key) - idx1 - 1
    ch2 = key[idx2]
    step = 1 if idx2 > idx1 else -1
    cnt = 1
    i = idx + step
    if i < 0:
        i += len(s)
    while 0 <= i < len(s):
        if s[i] == ch1:
            cnt += 1
        elif s[i] == ch2:
            cnt -= 1
        if cnt == 0:
            return i
        i += step
    return None


def bracket_match2(s, idx):
    r"""与“bracket_match”相比，会考虑"\{"转义字符的影响

    >>> bracket_match2('a{b{}b}c', 1)
    6
    >>> bracket_match2('a{b{\}b}c}d', 1)
    9
    """
    key = '{[(<>)]}'
    try:
        if idx < 0:
            idx += len(s)
        ch1 = s[idx]
        idx1 = key.index(ch1)
    except ValueError:  # 找不到ch1
        return None
    except IndexError:  # 下标越界，表示没有匹配到右括号
        return None
    idx2 = len(key) - idx1 - 1
    ch2 = key[idx2]
    step = 1 if idx2 > idx1 else -1
    cnt = 1
    i = idx + step
    if i < 0:
        i += len(s)
    while 0 <= i < len(s):
        if i and s[i - 1] == '\\':
            pass
        elif s[i] == ch1:
            cnt += 1
        elif s[i] == ch2:
            cnt -= 1
        if cnt == 0:
            return i
        i += step
    return None


def findnth(haystack, needle, n):
    """https://stackoverflow.com/questions/1883980/find-the-nth-occurrence-of-substring-in-a-string"""
    if n < 0:
        n += haystack.count(needle)
    if n < 0:
        return -1

    parts = haystack.split(needle, n + 1)
    if len(parts) <= n + 1:
        return -1
    return len(haystack) - len(parts[-1]) - len(needle)


def strfind(fullstr, objstr, *, start=None, times=0, overlap=False):
    r""" 进行强大功能扩展的的字符串查找函数

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


def findspan(src, sub, start=0, end=None):
    """ str.find的封装

    :param sub:
        str，普通的字符串查找
        re.Pattern，正则模式的查找
    :return: (start, end)
        找不到的时候返回 (-1, -1)
        否则返回区间的左开右闭位置
    """
    if end is None:
        end = len(src)

    if isinstance(sub, str):
        pos = src.find(sub, start, end)
    elif isinstance(sub, re.Pattern):
        pattern = sub
        m = pattern.search(src[start:end])
        if m:
            pos = m.start() + start
            sub = m.group()
        else:
            pos = -1
    else:
        raise TypeError

    if pos == -1:
        return -1, -1
    else:
        return pos, pos + len(sub)


def substr_count(src, sub, overlape=False):
    """ 判断字符串src中符合pattern的字串有几个 """
    if overlape:
        raise NotImplementedError
    else:
        if isinstance(sub, str):
            cnt = src.count(sub)
        elif isinstance(sub, re.Pattern):
            cnt = len(sub.findall(src))
        else:
            raise TypeError

    return cnt


def count_word(s, *patterns):
    """ 统计一串文本中，各种规律串出现的次数

    :param s: 文本内容
    :param patterns: （正则规则）
        匹配的多个目标模式list
        按优先级一个一个往后处理，被处理掉的部分会用\x00代替
    :return: Counter.most_common() 对象
    """
    s = str(s)

    if not patterns:  # 不写参数的时候，默认统计所有单个字符
        return collections.Counter(list(s)).most_common()

    ls = []
    for t in patterns:
        ls += re.findall(t, s)
        s = re.sub(t, '\x00', s)
        # s = re.sub(r'\x00+', '\x00', s)  # 将连续的特殊删除设为1，减短字符串长度，还未试验这段代码精确度与效率
    ct = collections.Counter(ls)

    ls = ct.most_common()
    for i in range(len(ls)):
        ls[i] = (ls[i][1], repr(ls[i][0])[1:-1])
    return ls


def continuous_zero(s):
    """ 返回一个字符串中连续0的位置

    :param s: 一个字符串

    做html转latex表格中，合并单元格的处理要用到这个函数计算cline

    >>> continuous_zero('0100')  # 从0开始编号，左闭右开区间
    [(0, 1), (2, 4)]
    """
    return [m.span() for m in re.finditer(r'0+', s)]
