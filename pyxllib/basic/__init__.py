#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/08/14 21:52


"""
最基础常用的一些功能

basic中依赖的三方库有直接写到 requirements.txt 中
（其他debug、image等库的依赖项都是等使用到了才加入）

且basic依赖的三方库，都确保是体积小
    能快速pip install
    及pyinstaller -F打包生成的exe也不大的库
"""

# 1 时间相关工具
from .pytictoc import TicToc
from .timer import *
from .arrow_ import Datetime

# 2 调试1
from .dprint import *

# 3 文本
from .strlib import *

# 4 文件、目录工具
from .judge import *
from .chardet_ import *
from .qiniu_ import *
from .pathlib_ import Path
from .dirlib import *


# 5 其他一些好用的基础功能

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


class RunOnlyOnce:
    """ 被装饰的函数，不同的参数输入形式，只会被执行一次，

    重复执行时会从内存直接调用上次相同参数调用下的运行的结果
    可以使用reset成员函数重置，下一次调用该函数时则会重新执行

    文档：https://www.yuque.com/xlpr/pyxllib/RunOnlyOnce

    使用好该装饰器，可以让很多动态规划dp问题、搜索问题变得异常简洁
    """

    def __init__(self, func, distinct_args=True):
        """
        :param func: 封装的函数
        :param distinct_args: 默认不同输入参数形式，都会保存一个结果
            设为False，则不管何种参数形式，函数就真的只会保存第一次运行的结果
        """
        self.func = func
        self.distinct_args = distinct_args
        self.results = {}

    def __call__(self, *args, **kwargs):
        tag = f'{args}{kwargs}' if self.distinct_args else ''
        # TODO 思考更严谨，考虑了值类型的tag标记
        #   目前的tag规则，是必要不充分条件。还可以使用id，则是充分不必要条件
        #   能找到充要条件来做是最好的，不行的话，也应该用更严谨的充分条件来做
        # TODO kwargs的顺序应该是没影响的，要去掉顺序干扰
        if tag not in self.results:
            self.results[tag] = self.func(*args, **kwargs)
        return self.results[tag]

    def reset(self):
        self.results = {}
