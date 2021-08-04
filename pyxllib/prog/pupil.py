#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 23:21


""" 封装一些代码开发中常用的功能，工程组件 """

from urllib.parse import urlparse
import io
import json
import math
import os
import queue
import socket
import sys
import time


def is_url(arg):
    """输入是一个字符串，且值是一个合法的url"""
    if not isinstance(arg, str): return False
    try:
        result = urlparse(arg)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def is_file(arg, exists=True):
    """相较于标准库的os.path.isfile，对各种其他错误类型也会判False

    :param exists: arg不仅需要是一个合法的文件名，还要求其实际存在
        设为False，则只判断文件名合法性，不要求其一定要存在
    """
    if not isinstance(arg, str): return False
    if not exists:
        raise NotImplementedError
    return os.path.isfile(arg)


def get_hostname():
    return socket.getfqdn()


def get_username():
    return os.path.split(os.path.expanduser('~'))[-1]


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


def print2string(*args, **kwargs):
    """https://stackoverflow.com/questions/39823303/python3-print-to-string"""
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


class EmptyPoolExecutor:
    """伪造一个类似concurrent.futures.ThreadPoolExecutor、ProcessPoolExecutor的接口类
        用来检查多线程、多进程中的错误

    即并行中不会直接报出每个线程的错误，只能串行执行才好检查
        但是两种版本代码来回修改很麻烦，故设计此类，只需把
            concurrent.futures.ThreadPoolExecutor 暂时改为 EmptyPoolExecutor 进行调试即可
    """

    def __init__(self, *args, **kwargs):
        """参数并不需要实际处理，并没有真正并行，而是串行执行"""
        self._work_queue = queue.Queue()

    def submit(self, func, *args, **kwargs):
        """执行函数"""
        func(*args, **kwargs)

    def shutdown(self):
        # print('并行执行结束')
        pass


def xlwait(func, condition=bool, *, limit=None, interval=1):
    """ 不断重复执行func，直到得到满足condition条件的期望值

    :param condition: 退出等待的条件，默认为bool真值
    :param limit: 重复执行的上限时间（单位 秒），默认一直等待
    :param interval: 重复执行间隔 （单位 秒）

    """
    t = time.time()
    while True:
        res = func()
        if condition(res):
            return res
        elif limit and (time.time() - t > limit):
            return res  # 超时也返回目前得到的结果
        time.sleep(interval)


class DictTool:
    @classmethod
    def json_loads(cls, label, default=None):
        """ 尝试从一段字符串解析为字典

        :param default: 如果不是字典时的处理策略
            None，不作任何处理
            str，将原label作为defualt这个键的值来存储
        :return: s为非字典结构时返回空字典

        >>> DictTool.json_loads('123', 'label')
        {'label': '123'}
        >>> DictTool.json_loads('[123, 456]', 'label')
        {'label': '[123, 456]'}
        >>> DictTool.json_loads('{"a": 123}', 'label')
        {'a': 123}
        """
        labelattr = dict()
        try:
            data = json.loads(label)
            if isinstance(data, dict):
                labelattr = data
        except json.decoder.JSONDecodeError:
            pass
        if not labelattr and isinstance(default, str):
            labelattr[default] = label
        return labelattr

    @classmethod
    def or_(cls, *args):
        """ 合并到新字典

        左边字典有的key，优先取左边，右边不会覆盖。
        如果要覆盖效果，直接用 d1.update(d2)功能即可。

        :return: args[0] | args[1] | ... | args[-1].
        """
        res = {}
        cls.ior(res, *args)
        return res

    @classmethod
    def ior(cls, dict_, *args):
        """ 合并到第1个字典

        :return: dict_ |= (args[0] | args[1] | ... | args[-1]).
        """
        if sys.version_info.major == 3 and sys.version_info.minor >= 9:
            for x in args:
                dict_ |= x
        else:  # 旧版本py手动实现一个兼容功能
            for x in args:
                for k, v in x.items():
                    if k not in dict_:
                        dict_[k] = v

    @classmethod
    def sub(cls, dict_, keys):
        """ 删除指定键值（不存在的跳过，不报错）

        inplace subtraction

        keys可以输入另一个字典，也可以输入一个列表表示要删除的键值清单

        :return: dict_ -= keys
        """
        if isinstance(keys, dict):
            keys = keys.keys()

        return {k: v for k, v in dict_.items() if k not in keys}

    @classmethod
    def isub(cls, dict_, keys):
        """ 删除指定键值（不存在的跳过，不报错）

        inplace subtraction

        keys可以输入另一个字典，也可以输入一个列表表示要删除的键值清单

        :return: dict_ -= keys
        """
        if isinstance(keys, dict):
            keys = keys.keys()

        for k in keys:
            if k in dict_:
                del dict_[k]