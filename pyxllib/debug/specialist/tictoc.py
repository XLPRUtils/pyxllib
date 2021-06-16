#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/09/20


import time
import timeit

from humanfriendly import format_timespan

from pyxllib.text.pupil import shorten, listalign
from pyxllib.algo.pupil import natural_sort, ValuesStat

__tictoc = """
基于 pytictoc 代码，做了些自定义扩展

原版备注：
Module with class TicToc to replicate the functionality of MATLAB's tic and toc.
Documentation: https://pypi.python.org/pypi/pytictoc
__author__       = 'Eric Fields'
__version__      = '1.4.0'
__version_date__ = '29 April 2017'
"""


class TicToc:
    """
    Replicate the functionality of MATLAB's tic and toc.

    #Methods
    TicToc.tic()       #start or re-start the timer
    TicToc.toc()       #print elapsed time since timer start
    TicToc.tocvalue()  #return floating point value of elapsed time since timer start

    #Attributes
    TicToc.start     #Time from timeit.default_timer() when t.tic() was last called
    TicToc.end       #Time from timeit.default_timer() when t.toc() or t.tocvalue() was last called
    TicToc.elapsed   #t.end - t.start; i.e., time elapsed from t.start when t.toc() or t.tocvalue() was last called
    """

    def __init__(self, title='', *, disable=False):
        """Create instance of TicToc class."""
        self.start = timeit.default_timer()
        self.end = float('nan')
        self.elapsed = float('nan')
        self.title = title
        self.disable = disable

    def tic(self):
        """Start the timer."""
        self.start = timeit.default_timer()

    def toc(self, msg='', restart=False):
        """
        Report time elapsed since last call to tic().

        Optional arguments:
            msg     - String to replace default message of 'Elapsed time is'
            restart - Boolean specifying whether to restart the timer
        """
        self.end = timeit.default_timer()
        self.elapsed = self.end - self.start
        if not self.disable:
            # print(f'{self.title} {msg} {self.elapsed:.3f} 秒.')
            print(f'{self.title} {msg} elapsed {format_timespan(self.elapsed)}.')
        if restart:
            self.start = timeit.default_timer()

    def tocvalue(self, restart=False):
        """
        Return time elapsed since last call to tic().

        Optional argument:
            restart - Boolean specifying whether to restart the timer
        """
        self.end = timeit.default_timer()
        self.elapsed = self.end - self.start
        if restart:
            self.start = timeit.default_timer()
        return self.elapsed

    @staticmethod
    def process_time(msg='time.process_time():'):
        """计算从python程序启动到目前为止总用时"""
        print(f'{msg} {format_timespan(time.process_time())}.')

    def __enter__(self):
        """Start the timer when using TicToc in a context manager."""
        from pyxllib.debug.specialist import get_xllog

        if self.title == '__main__' and not self.disable:
            get_xllog().info(f'time.process_time(): {format_timespan(time.process_time())}.')
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """On exit, print time elapsed since entering context manager."""
        from pyxllib.debug.specialist import get_xllog
        
        elapsed = self.tocvalue()
        xllog = get_xllog()

        if exc_tb is None:
            if not self.disable:
                xllog.info(f'{self.title} finished in {format_timespan(elapsed)}.')
        else:
            xllog.info(f'{self.title} interrupt in {format_timespan(elapsed)},')


__timer = """

"""


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
            print(msg, '用时(秒) ' + ValuesStat(self.data).summary(valfmt='.3f'))
        elif n == 1:
            sum_ = sum(self.data)
            print(f'{msg} 用时: {sum_:.3f}s')
        else:  # 没有统计数据，则补充执行一次stop后汇报
            print(f'{msg} 暂无计时信息')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.report()


def performit(title, stmt="pass", setup="pass", repeat=1, number=1, globals=None):
    """ 在timeit.repeat的基础上，做了层封装

    200920周日15:33，简化函数，该函数不再获得执行结果，避免重复运行

    :param title: 测试标题、名称功能
    :return: 返回原函数单次执行结果
    """
    data = timeit.repeat(stmt=stmt, setup=setup, repeat=repeat, number=number, globals=globals)
    print(title, '用时(秒) ' + ValuesStat(data).summary(valfmt='.3f'))
    return data


def perftest(title, stmt="pass", repeat=1, number=1, globals=None, res_width=None, print_=True):
    """ 与performit的区别是，自己手动循环，记录程序运行结果

    :param title: 测试标题、名称功能
    :param res_width: 运行结果内容展示的字符上限数
    :param print_: 输出报告
    :return: 返回原函数单次执行结果

    这里为了同时获得表达式返回值，就没有用标注你的timeit.repeat实现了
    """
    # 1 确保stmt是可调用对象
    if callable(stmt):
        func = stmt
    else:
        code = compile(stmt, '', 'eval')

        def func():
            return eval(code, globals)

    # 2 原函数运行结果（这里要先重载stdout）
    data = []
    res = ''
    for i in range(repeat):
        start = time.clock()
        for j in range(number):
            res = func()
        data.append(time.clock() - start)

    # 3 报告格式
    if res_width is None:
        # 如果性能报告比较短，只有一次测试，那res_width默认长度可以高一点
        res_width = 50 if len(data) > 1 else 200
    if res is None:
        res = ''
    else:
        res = '运行结果：' + shorten(str(res), res_width)
    if print_:
        print(title, '用时(秒) ' + ValuesStat(data).summary(valfmt='.3f'), res)

    return data


class PerfTest:
    """ 这里模仿了unittest的机制

    v0.0.38 重要改动，将number等参数移到perf操作，而不是类初始化中操作，继承使用上会更简单
    """

    def perf(self, number=1, repeat=1, globals=None):
        """

        :param number: 有些代码运算过快，可以多次运行迭代为一个单元
        :param number: 对单元重复执行次数，最后会计算平均值、标准差
        """
        # 1 找到所有perf_为前缀，且callable的函数方法
        funcnames = []
        for k in dir(self):
            if k.startswith('perf_'):
                if callable(getattr(self, k)):
                    funcnames.append(k)

        # 2 自然排序
        funcnames = natural_sort(funcnames)
        funcnames2 = listalign([fn[5:] for fn in funcnames], 'r')
        for i, funcname in enumerate(funcnames):
            perftest(funcnames2[i], getattr(self, funcname),
                     number=number, repeat=repeat, globals=globals)
