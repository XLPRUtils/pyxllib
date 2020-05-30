#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/30 18:55


"""
基于 pytictoc 代码，做了些自定义扩展

原版备注：
Module with class TicToc to replicate the functionality of MATLAB's tic and toc.
Documentation: https://pypi.python.org/pypi/pytictoc
__author__       = 'Eric Fields'
__version__      = '1.4.0'
__version_date__ = '29 April 2017'
"""


import time
import timeit


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

    def __init__(self, title=''):
        """Create instance of TicToc class."""
        self.start = timeit.default_timer()
        self.end = float('nan')
        self.elapsed = float('nan')
        self.title = title

    def tic(self):
        """Start the timer."""
        self.start = timeit.default_timer()

    def toc(self, msg='用时', restart=False):
        """
        Report time elapsed since last call to tic().

        Optional arguments:
            msg     - String to replace default message of 'Elapsed time is'
            restart - Boolean specifying whether to restart the timer
        """
        self.end = timeit.default_timer()
        self.elapsed = self.end - self.start
        print(f'{self.title} {msg} {self.elapsed:.3f} 秒.')
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
    def process_time(msg='程序已启动'):
        """计算从python程序启动到目前为止总用时"""
        print(f'{msg} {time.process_time():.3f} 秒')

    def __enter__(self):
        """Start the timer when using TicToc in a context manager."""
        self.start = timeit.default_timer()

    def __exit__(self, *args):
        """On exit, print time elapsed since entering context manager."""
        self.end = timeit.default_timer()
        self.elapsed = self.end - self.start
        print(f'{self.title} {self.elapsed:.3f} 秒.')
