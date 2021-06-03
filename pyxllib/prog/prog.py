#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 14:14

"""
一些简单的工程开发组件
"""


class SingletonForEveryClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        tag = f'{cls}'
        # 其实转字符串来判断是不太严谨的，有些类型字符串后的显示效果是一模一样的
        # dprint(tag)
        if tag not in cls._instances:
            cls._instances[tag] = super(SingletonForEveryClass, cls).__call__(*args, **kwargs)
        return cls._instances[tag]


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
        # 其实转字符串来判断是不太严谨的，有些类型字符串后的显示效果是一模一样的
        # dprint(tag)
        if tag not in cls._instances:
            cls._instances[tag] = super(SingletonForEveryInitArgs, cls).__call__(*args, **kwargs)
        return cls._instances[tag]


def xlbool(v):
    """ 有些类型不能直接判断真假，例如具有多值的np.array，df等

    这些有歧义的情况，在我的mybool里暂时都判断为True，如果有需要，需要精细化判断，可以扩展自己的npbool、dfbool
    """
    try:
        return bool(v)
    except ValueError:
        return True
