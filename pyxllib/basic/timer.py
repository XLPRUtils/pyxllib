#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/31


import math
import re
import textwrap
import timeit


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


def parse_perf(data):
    """ 输出性能分析报告，data是每次运行得到的时间数组
    """
    n, sum_ = len(data), sum(data)

    if n > 1:  # 有多轮，则应该输出些参考统计指标
        # np有标准差等公式，但这是basic底层库，不想依赖太多第三方库，所以手动实现
        mean = sum_ / n
        std = math.sqrt((sum([(x - mean) ** 2 for x in data]) / n))
        li = [f'总耗时: {sum_:.3f}s', f'均值标准差: {mean:.3f}±{std:.3f}s',
              f'总数: {n}', f'最小值: {min(data):.3f}s', f'最大值: {max(data):.3f}s']
        return '\t'.join(li)
    elif n == 1:  # 只有一轮，则简单地输出耗时即可
        sum_ = sum(data)
        return f'用时: {sum_:.3f}s'
    else:
        raise ValueError


def performit(title, stmt="pass", setup="pass", repeat=1, number=1, globals=None, res_width=None):
    """ 在timeit.repeat的基础上，做了层封装，能更好的展示运行效果
    :param title: 测试标题、名称功能
    :param res_width: 运行结果内容展示的字符上限数
    :return: 返回原函数单次执行结果
    """
    # 1 性能报告
    data = timeit.repeat(stmt=stmt, setup=setup,
                         repeat=repeat, number=number, globals=globals)

    # 2 原函数运行结果
    if callable(stmt):
        res = stmt()
    else:
        res = eval(stmt, globals)

    if res_width is None:
        # 如果性能报告比较短，只有一次测试，那res_width默认长度可以高一点
        res_width = 50 if len(data) > 1 else 200
    print(title, parse_perf(data), '运行结果：' + shorten(str(res), res_width))

    return res


class Timer:
    """分析性能用的计时器类，支持with语法调用
    必须显示地指明每一轮的start()和end()，否则会报错
    """

    def __init__(self, title=''):
        """
        :param title: 计时器名称
        """
        # 不同的平台应该使用的计时器不同，这个直接用timeit中的配置最好
        self.default_timer = timeit.default_timer
        # 标题
        self.title = title
        self.data = []
        self.start_clock = float('nan')

    def start(self):
        self.start_clock = self.default_timer()

    def stop(self):
        self.data.append(self.default_timer() - self.start_clock)

    def report(self, msg=''):
        """ 报告目前性能统计情况
        """
        msg = f'{self.title} {msg}'
        n = len(self.data)

        if n >= 1:
            print(msg, parse_perf(self.data))
        elif n == 1:
            sum_ = sum(self.data)
            print(f'{msg} 用时: {sum_:.3f}s')
        else:  # 没有统计数据，则补充执行一次stop后汇报
            print(f'{msg} 暂无计时信息')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.report()
