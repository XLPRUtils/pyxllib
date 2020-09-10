#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/06/02 15:48

"""调试功能，通用底层功能

文中所有前缀4个下划线_的是模块划分标记，且
    前4个下划线_是一级结构
    前8个下划线_是二级结构

相关文档： https://blog.csdn.net/code4101/article/details/83269101
"""

____main = """
这里会加载好主要功能，但并不会加载所有功能
"""

from .installer import *
from .typelib import *
from .chrome import *
from .showdir import *
from .bcompare import *

____other = """
"""


class SingletonForEveryInitArgs(type):
    """Python单例模式(Singleton)的N种实现 - 知乎: https://zhuanlan.zhihu.com/p/37534850

    注意！注意！注意！重要的事说三遍！
    我的单例类不是传统意义上的单例类。
    传统意义的单例类，不管用怎样不同的初始化参数创建对象，永远都只有最初的那个对象类。
    但是我的单例类，为每种不同的参数创建形式，都构造了一个对象。
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        tag = f'{cls}{args}{kwargs}'  # id加上所有参数的影响来控制单例类
        # 其实转字符串来判断是不太严谨的，例如数字1和字符串'1'并不是等价的输入参数
        # dprint(tag)
        if tag not in cls._instances:
            cls._instances[tag] = super(SingletonForEveryInitArgs, cls).__call__(*args, **kwargs)
        return cls._instances[tag]


class EmptyPoolExecutor:
    """伪造一个类似concurrent.futures.ThreadPoolExecutor、ProcessPoolExecutor的接口类
        用来检查多线程、多进程中的错误

    即并行中不会直接报出每个线程的错误，只能串行执行才好检查
        但是两种版本代码来回修改很麻烦，故设计此类，只需把
            concurrent.futures.ThreadPoolExecutor 暂时改为 EmptyPoolExecutor 进行调试即可
    """

    def __init__(self, *args, **kwargs):
        """参数并不需要实际处理，并没有真正并行，而是串行执行"""
        pass

    def submit(self, func, *args, **kwargs):
        """执行函数"""
        func(*args, **kwargs)

    def shutdown(self):
        print('并行执行结束')


def render_echart(ob, name, show=False):
    """ 渲染显示echart图表

    https://www.yuque.com/xlpr/pyxllib/render_echart

    :param ob: 一个echart图表对象
    :param name: 存储的文件名
    :param show: 是否要立即在浏览器显示
    :return: 存储的文件路径
    """
    # 如果没有设置页面标题，则默认采用文件名作为标题
    if not ob.page_title or ob.page_title == 'Awesome-pyecharts':
        ob.page_title = name
    f = ob.render(Path(f'{name}.html', root=Path.TEMP).fullpath)
    if show: chrome(f)
    return f


if __name__ == '__main__':
    pass
