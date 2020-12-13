#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/08/14 21:52


"""
最基础常用的一些功能

basic中依赖的三方库有直接写到 requirements.txt 中
（其他debug、cv等库的依赖项都是等使用到了才加入）

且basic依赖的三方库，都确保是体积小
    能快速pip install
    及pyinstaller -F打包生成的exe也不大的库
"""

# 1 文本处理等一些基础杂项功能
from pyxllib.basic._1_strlib import *
# 2 时间相关工具
from pyxllib.basic._2_timelib import *
# 3 文件、路径工具
from pyxllib.basic._3_filelib import *
# 4 调试工具，Iterate等一些高级通用功能
from pyxllib.basic._4_loglib import *
# 5 目录工具
from pyxllib.basic._5_dirlib import *

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
        # 其实转字符串来判断是不太严谨的，有些类型字符串后的显示效果是一模一样的
        # dprint(tag)
        if tag not in cls._instances:
            cls._instances[tag] = super(SingletonForEveryInitArgs, cls).__call__(*args, **kwargs)
        return cls._instances[tag]


if __name__ == '__main__':
    pass
