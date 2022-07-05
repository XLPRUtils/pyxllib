#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 23:21


""" 封装一些代码开发中常用的功能，工程组件 """
import datetime
import functools
import io
import itertools
import json
import logging
import math
import os
import queue
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlparse

from pyxllib.prog.newbie import classproperty


def system_information():
    """主要是测试一些系统变量值，顺便再演示一次Timer用法"""

    def pc_messages():
        """演示如何获取当前操作系统的PC环境数据"""
        # fqdn：fully qualified domain name
        print('1、socket.getfqdn() :', socket.getfqdn())  # 完全限定域名，可以理解成pcname，计算机名
        # 注意py的很多标准库功能本来就已经处理了不同平台的问题，尽量用标准库而不是自己用sys.platform作分支处理
        print('2、sys.platform     :', sys.platform)  # 运行平台，一般是win32和linux
        # li = os.getenv('PATH').split(os.path.pathsep)  # 环境变量名PATH，win中不区分大小写，linux中区分大小写必须写成PATH
        # print("3、os.getenv('PATH'):", f'数量={len(li)},', pprint.pformat(li, 4))

    def executable_messages():
        """演示如何获取被执行程序相关的数据"""
        print('1、sys.executable   :', sys.executable)  # 当前被执行脚本位置
        print('2、sys.version      :', sys.version)  # python的版本
        print('3、os.getcwd()      :', os.getcwd())  # 获得当前工作目录
        print('4、gettempdir()     :', tempfile.gettempdir())  # 临时文件夹位置
        # print('5、sys.path       :', f'数量={len(sys.path)},', pprint.pformat(sys.path, 4))  # import绝对位置包的搜索路径

    print('【pc_messages】')
    pc_messages()
    print('【executable_messages】')
    executable_messages()


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
    if len(arg) > 500: return False
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

        >>> DictTool.json_loads('123', 'text')
        {'text': '123'}
        >>> DictTool.json_loads('[123, 456]', 'text')
        {'text': '[123, 456]'}
        >>> DictTool.json_loads('{"a": 123}', 'text')
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

        220601周三15:45，默认已有对应key的话，值是不覆盖的，如果要覆盖，直接用update就行了，不需要这个接口
            所以把3.9的|=功能关掉
        """
        # if sys.version_info.major == 3 and sys.version_info.minor >= 9:
        #     for x in args:
        #         dict_ |= x
        # else:  # 旧版本py手动实现一个兼容功能
        for x in args:
            for k, v in x.items():
                # if k not in dict_:
                #     dict_[k] = v
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


def check_install_package(package, speccal_install_name=None, *, user=False):
    """ https://stackoverflow.com/questions/12332975/installing-python-module-within-code

    :param speccal_install_name: 注意有些包使用名和安装名不同，比如pip install python-opencv，使用时是import cv2，
        此时应该写 check_install_package('cv2', 'python-opencv')

    TODO 不知道频繁调用这个，会不会太影响性能，可以想想怎么提速优化？
    注意不要加@RunOnlyOnce，亲测速度会更慢三倍

    警告: 不要在频繁调用的底层函数里使用 check_install_package
        如果是module级别的还好，调几次其实性能影响微乎其微
        但在频繁调用的函数里使用，每百万次还是要额外的0.5秒开销的
    """
    try:
        __import__(package)
    except ModuleNotFoundError:
        cmds = [sys.executable, "-m", "pip", "install"]
        if user: cmds.append('--user')
        cmds.append(speccal_install_name if speccal_install_name else package)
        subprocess.check_call(cmds)


def run_once(distinct_mode=0, *, limit=1):
    """
    :param int|str distinct_mode:
        0，默认False，不区分输入的参数值（包括cls、self），强制装饰的函数只运行一次
        'str'，设为True或1时，仅以字符串化的差异判断是否是重复调用，参数不同，会判断为不同的调用，每种调用限制最多执行limit次
        'id,str'，在'str'的基础上，第一个参数使用id代替。一般用于类方法、对象方法的装饰。
            不考虑类、对象本身的内容改变，只要还是这个类或对象，视为重复调用。
        'ignore,str'，首参数忽略，第2个开始的参数使用str格式化
            用于父类某个方法，但是子类继承传入cls，原本id不同会重复执行
            使用该模式，首参数会ignore忽略，只比较第2个开始之后的参数
        func等callable类型的对象也行，是使用run_once装饰器的简化写法
    :param limit: 默认只会执行一次，该参数可以提高限定的执行次数，一般用不到，用于兼容旧的 limit_call_number 装饰器
    Returns: 返回decorator
    """
    if callable(distinct_mode):
        # @run_once，没写括号的时候去装饰一个函数，distinct_mode传入的是一个函数func
        # 使用run_once本身的默认值
        return run_once()(distinct_mode)

    def get_tag(args, kwargs):
        if not distinct_mode:
            ls = tuple()
        elif distinct_mode == 'str':
            ls = (str(args), str(kwargs))
        elif distinct_mode == 'id,str':
            ls = (id(args[0]), str(args[1:]), str(kwargs))
        elif distinct_mode == 'ignore,str':
            ls = (str(args[1:]), str(kwargs))
        else:
            raise ValueError
        return ls

    def decorator(func):
        counter = {}  # 映射到一个[cnt, last_result]

        def wrapper(*args, **kwargs):
            tag = get_tag(args, kwargs)
            if tag not in counter:
                counter[tag] = [0, None]
            x = counter[tag]
            if x[0] < limit:
                res = func(*args, **kwargs)
                x = counter[tag] = [x[0] + 1, res]
            return x[1]

        return wrapper

    return decorator


def set_default_args(*d_args, **d_kwargs):
    """ 增设默认参数

    有时候需要调试一个函数，试跑一些参数结果，
    但这些参数又不适合定为标准化的接口值，可以用这个函数设置

    参数加载、覆盖顺序（越后面的优先级越高）
    1、函数定义阶段设置的默认值
    2、装饰器定义的参数 d_args、d_kwargs
    3、运行阶段明确指定的参数，即传入的f_args、f_kwargs
    """

    def decorator(func):
        def wrapper(*f_args, **f_kwargs):
            args = f_args + d_args
            d_kwargs.update(f_kwargs)  # 优先使用外部传参传入的值，再用装饰器里扩展的默认值
            return func(*args, **d_kwargs)

        return wrapper

    return decorator


def utc_now(offset_hours=8):
    """ 有的机器可能本地时间设成了utc0，可以用这个方式，获得准确的utc8时间 """
    return datetime.datetime.utcnow() + datetime.timedelta(hours=offset_hours)


def utc_timestamp(offset_hours=8):
    """ mysql等数据库支持的日期格式 """
    return utc_now(offset_hours).isoformat(timespec='seconds')


class Timeout:
    """ 对函数等待执行的功能，限制运行时间

    【实现思路】
    1、最简单的方式是用signal.SIGALRM实现
        https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
        但是这个不支持windows系统~~
    2、那windows和linux通用的做法，就是把原执行函数变成一个子线程来运行
        https://stackoverflow.com/questions/21827874/timeout-a-function-windows
        但是，又在onenote的win32com发现，有些功能没办法丢到子线程里，会出问题
        而且使用子线程，也没法做出支持with上下文语法的功能了
    3、于是就有了当前我自己搞出的一套机制
        是用一个Timer计时器子线程计时，当timeout超时，使用信号机制给主线程抛出一个异常
            ① 注意，不能子线程直接抛出异常，这样影响不了主线程
            ② 也不能直接抛出错误signal，这样会强制直接中断程序。应该抛出TimeoutError，让后续程序进行超时逻辑的处理
    """

    def __init__(self, seconds):
        self.seconds = seconds
        self.alarm = None

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1 如果超时，主线程收到信号会执行的功能
            def overtime(signum, frame):
                raise TimeoutError(f'function [{func.__name__}] timeout [{self.seconds} seconds] exceeded!')

            signal.signal(signal.SIGABRT, overtime)

            # 2 开一个子线程计时器，超时的时候发送信号
            def send_signal():
                signal.raise_signal(signal.SIGABRT)

            alarm = threading.Timer(self.seconds, send_signal)
            alarm.start()

            # 3 执行主线程功能
            res = func(*args, **kwargs)
            alarm.cancel()  # 正常执行完则关闭计时器

            return res

        return wrapper

    def __enter__(self):
        def overtime(signum, frame):
            raise TimeoutError(f'with 上下文代码块运行超时 > [{self.seconds} 秒]')

        signal.signal(signal.SIGABRT, overtime)

        def send_signal():
            signal.raise_signal(signal.SIGABRT)

        self.alarm = threading.Timer(self.seconds, send_signal)
        self.alarm.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.alarm.cancel()


@run_once('str')
def inject_members(from_obj, to_obj, member_list=None, *,
                   check=False, ignore_case=False,
                   white_list=None, black_list=None):
    """ 将from_obj的方法注入到to_obj中

    一般用于类继承中，将子类from_obj的新增的成员方法，添加回父类to_obj中
        反经合道：这样看似很违反常理，父类就会莫名其妙多出一些可操作的成员方法。
            但在某些时候能保证面向对象思想的情况下，大大简化工程代码开发量。
    也可以用于模块等方法的添加

    :param from_obj: 一般是一个类用于反向继承的方法来源，但也可以是模块等任意对象。
        注意py一切皆类，一个class定义的类本质也是type类定义出的一个对象
        所以这里概念上称为obj才是准确的，反而是如果叫from_cls不太准确，虽然这个方法主要确实都是用于class类
    :param to_obj: 同from_obj，要被注入方法的对象
    :param Sequence[str] member_list: 手动指定的成员方法名，可以不指定，自动生成
    :param check: 检查重名方法
    :param ignore_case: 忽略方法的大小写情况，一般用于win32com接口
    :param Sequence[str] white_list: 白名单。无论是否重名，这里列出的方法都会被添加
    :param Sequence[str] black_list: 黑名单。这里列出的方法不会被添加
    """
    # 1 整理需要注入的方法清单
    dst = set(dir(to_obj))
    if ignore_case:
        dst = {x.lower() for x in dst}

    if member_list:
        src = set(member_list)
    else:
        if ignore_case:
            src = {x for x in dir(from_obj) if (x.lower() not in dst)}
        else:
            src = set(dir(from_obj)) - dst

    # 2 微调
    if white_list:
        src |= set(white_list)
    if black_list:
        src -= set(black_list)

    # 3 注入方法
    for x in src:
        setattr(to_obj, x, getattr(from_obj, x))
        if check and (x in dst or (ignore_case and x.lower() in dst)):
            logging.warning(f'Conflict of the same name! {to_obj}.{x}')
