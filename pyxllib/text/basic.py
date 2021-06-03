#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 22:56

import io
import logging
import re
import sys


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


def int2myalphaenum(n):
    """
    :param n: 0~52的数字
    """
    if 0 <= n <= 52:
        return '_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[n]
    else:
        print('警告：不在处理范围内的数值', n)
        raise ValueError


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


def digit2weektag(d):
    """ 输入数字1~7，转为“周一~周日”

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


class ContentPartSpliter:
    """ 文本内容分块处理 """

    @classmethod
    def multi_blank_lines(cls, content, leastlines=2):
        """ 用多个空行隔开的情况

        :param leastlines: 最少2个空行隔开，为新的一块内容
        """
        fmt = r'\n{' + str(leastlines) + ',}'
        parts = [x.strip() for x in re.split(fmt, content)]
        parts = list(filter(bool, parts))  # 删除空行
        return parts
