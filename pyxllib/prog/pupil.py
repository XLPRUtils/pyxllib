#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 23:21


""" 封装一些代码开发中常用的功能，工程组件 """
import builtins
from collections import Counter
import ctypes
import datetime
import functools
import inspect
import io
import itertools
import json
import logging
import math
import os
import pprint
import queue
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from urllib.parse import urlparse

from pyxllib.prog.newbie import classproperty, typename


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
                # 220729周五21:21，又切换成dict_有的不做替换
                if k not in dict_:
                    dict_[k] = v
                # dict_[k] = v

    @classmethod
    def sub(cls, dict_, keys):
        """ 删除指定键值（不存在的跳过，不报错）

        inplace subtraction

        :param keys: 可以输入另一个字典，也可以输入一个列表表示要删除的键值清单

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


class EnchantCvt:
    """ 把类_cls的功能绑定到类cls里
    根源_cls里的实现类型不同，到cls需要呈现的接口形式不同，有很多种不同的转换形式
    每个分支里，随附了getattr目标函数的一般默认定义模板
    用_self、_cls表示dst_cls，区别原cls类的self、cls标记
    """

    @staticmethod
    def staticmethod2objectmethod(cls, _cls, x):
        # 目前用的最多的转换形式
        # @staticmethod
        # def func1(_self, *args, **kwargs): ...
        setattr(_cls, x, getattr(cls, x))

    @staticmethod
    def staticmethod2property(cls, _cls, x):
        # @staticmethod
        # def func2(_self): ...
        setattr(_cls, x, property(getattr(cls, x)))

    @staticmethod
    def staticmethod2classmethod(cls, _cls, x):
        # @staticmethod
        # def func3(_cls, *args, **kwargs): ...
        setattr(_cls, x, classmethod(getattr(cls, x)))

    @staticmethod
    def staticmethod2classproperty(cls, _cls, x):
        # @staticmethod
        # def func4(_cls): ...
        setattr(_cls, x, classproperty(getattr(cls, x)))

    @staticmethod
    def classmethod2objectmethod(cls, _cls, x):
        # @classmethod
        # def func5(cls, _self, *args, **kwargs): ...
        setattr(_cls, x, lambda *args, **kwargs: getattr(cls, x)(*args, **kwargs))

    @staticmethod
    def classmethod2property(cls, _cls, x):
        # @classmethod
        # def func6(cls, _self): ...
        setattr(_cls, x, lambda *args, **kwargs: property(getattr(cls, x)(*args, **kwargs)))

    @staticmethod
    def classmethod2classmethod(cls, _cls, x):
        # @classmethod
        # def func7(cls, _cls, *args, **kwargs): ...
        setattr(_cls, x, lambda *args, **kwargs: classmethod(getattr(cls, x)(*args, **kwargs)))

    @staticmethod
    def classmethod2classproperty(cls, _cls, x):
        # @classmethod
        # def func8(cls, _cls): ...
        setattr(_cls, x, lambda *args, **kwargs: classproperty(getattr(cls, x)(*args, **kwargs)))

    @staticmethod
    def staticmethod2modulefunc(cls, _cls, x):
        # @staticmethod
        # def func9(*args, **kwargs): ...
        setattr(_cls, x, getattr(cls, x))

    @staticmethod
    def classmethod2modulefunc(cls, _cls, x):
        # @classmethod
        # def func10(cls, *args, **kwargs): ...
        setattr(_cls, x, lambda *args, **kwargs: getattr(cls, x)(*args, **kwargs))

    @staticmethod
    def to_moduleproperty(cls, _cls, x):
        # 理论上还有'to_moduleproperty'的转换模式
        #   但这个很容易引起歧义，是应该存一个数值，还是动态计算？
        #   如果是动态计算，可以使用modulefunc的机制显式执行，更不容易引起混乱。
        #   从这个分析来看，是不需要实现'2moduleproperty'的绑定体系的。py标准语法本来也就没有module @property的概念。
        raise NotImplementedError


class EnchantBase:
    """
    一些三方库的类可能功能有限，我们想做一些扩展。
    常见扩展方式，是另外写一些工具函数，但这样就不“面向对象”了。
    如果要“面向对象”，需要继承已有的类写新类，但如果组件特别多，开发难度是很大的。
        比如excel就有单元格、工作表、工作薄的概念。
        如果自定义了新的单元格，那是不是也要自定义新的工作表、工作薄，才能默认引用到自己的单元格类。
        这个看着很理想，其实并没有实际开发可能性。
    所以我想到一个机制，把额外函数形式的扩展功能，绑定到原有类上。
        这样原来的功能还能照常使用，但多了很多我额外扩展的成员方法，并且也没有侵入原三方库的源码
        这样一种设计模式，简称“绑定”。换个逼格高点的说法，就是“强化、附魔”的过程，所以称为Enchant。
        这个功能应用在cv2、pillow、fitz、openpyxl，并在win32com中也有及其重要的应用。
    """

    @classmethod
    def check_enchant_names(cls, classes, names=None, *, white_list=None, ignore_case=False):
        """
        :param list classes: 不能跟这里列出的模块、类的成员重复
        :param list|str|tuple names: 要检查的名称清单
        :param white_list: 白名单，这里面的名称不警告
            在明确要替换三方库标准功能的时候，可以使用
        :param ignore_case: 忽略大小写
        """
        exist_names = {x.__name__: set(dir(x)) for x in classes}
        if names is None:
            names = {x for x in dir(cls) if x[:2] != '__'} \
                    - {'check_enchant_names', '_enchant', 'enchant'}

        white_list = set(white_list) if white_list else {}

        if ignore_case:
            names = {x.lower() for x in names}
            for k, values in exist_names.items():
                exist_names[k] = {x.lower() for x in exist_names[k]}
            white_list = {x.lower() for x in white_list}

        for name, k in itertools.product(names, exist_names):
            if name in exist_names[k] and name not in white_list:
                print(f'警告！同名冲突！ {k}.{name}')

        return set(names)

    @classmethod
    def _enchant(cls, _cls, names, cvt=EnchantCvt.staticmethod2objectmethod):
        """ 这个框架是支持classmethod形式的转换的，但推荐最好还是用staticmethod，可以减少函数嵌套层数，提高效率 """
        for name in set(names):
            cvt(cls, _cls, name)

    @classmethod
    def enchant(cls):
        raise NotImplementedError


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
    """ mysql等数据库支持的日期格式
    """
    return utc_now(offset_hours).isoformat(' ', timespec='seconds')


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

    # 把XlDocxTable的成员方法绑定到docx.table.Table里
    >> inject_members(XlDocxTable, docx.table.Table)

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


def find_free_ports(count=1):
    """ 随机获得可用端口

    :param count: 需要的端口数量（会保证给出的端口号不重复）
    :return: list
    """
    ports = set()
    while len(ports) < count:
        with socketserver.TCPServer(("localhost", 0), None) as s:
            ports.add(s.server_address[1])

    return list(ports)


def __debug系列():
    pass


def func_input_message(depth=2) -> dict:
    """假设调用了这个函数的函数叫做f，这个函数会获得
        调用f的时候输入的参数信息，返回一个dict，键值对为
            fullfilename：完整文件名
            filename：文件名
            funcname：所在函数名
            lineno：代码所在行号
            comment：尾巴的注释
            depth：深度
            funcnames：整个调用过程的函数名，用/隔开，例如...

            argnames：变量名（list），这里的变量名也有可能是一个表达式
            types：变量类型（list），如果是表达式，类型指表达式的运算结果类型
            argvals：变量值（list）

        这样以后要加新的键值对也很方便

        :param depth: 需要分析的层级
            0，当前func_input_message函数的参数输入情况
            1，调用func_input_message的函数 f 参数输入情况
            2，调用 f 的函数 g ，g的参数输入情况

        参考： func_input_message 的具体使用方法可以参考 dformat 函数
        细节：inspect可以获得函数签名，也可以获得一个函数各个参数的输入值，但我想要展现的是原始表达式，
            例如func(a)，以func(1+2)调用，inpect只能获得“a=3”，但我想要的是“1+2=3”的效果
    """
    res = {}
    # 1 找出调用函数的代码
    ss = inspect.stack()
    frameinfo = ss[depth]
    arginfo = inspect.getargvalues(ss[depth - 1][0])
    if arginfo.varargs:
        origin_args = arginfo.locals[arginfo.varargs]
    else:
        origin_args = list(map(lambda x: arginfo.locals[x], arginfo.args))

    res['fullfilename'] = frameinfo.filename
    res['filename'] = os.path.basename(frameinfo.filename)
    res['funcname'] = frameinfo.function
    res['lineno'] = frameinfo.lineno
    res['depth'] = len(ss)
    ls_ = list(map(lambda x: x.function, ss))
    # ls.reverse()
    res['funcnames'] = '/'.join(ls_)

    if frameinfo.code_context:
        code_line = frameinfo.code_context[0].strip()
    else:  # 命令模式无法获得代码，是一个None对象
        code_line = ''

    funcname = ss[depth - 1].function  # 调用的函数名
    # 这一行代码不一定是从“funcname(”开始，所以要用find找到开始位置
    code = code_line[code_line.find(funcname + '(') + len(funcname):]

    # 2 先找到函数的()中参数列表，需要以')'作为分隔符分析
    # TODO 可以考虑用ast重实现
    ls = code.split(')')
    logo, i = True, 1
    while logo and i <= len(ls):
        # 先将'='做特殊处理，防止字典类参数导致的语法错误
        s = ')'.join(ls[:i]).replace('=', '+') + ')'
        try:
            compile(s, '<string>', 'single')
        except SyntaxError:
            i += 1
        else:  # 正常情况
            logo = False
    code = ')'.join(ls[:i])[1:]

    # 3 获得注释
    # 这个注释实现的不是很完美，不过影响应该不大，还没有想到比较完美的解决方案
    t = ')'.join(ls[i:])
    comment = t[t.find('#'):] if '#' in t else ''
    res['comment'] = comment

    # 4 获得变量名
    ls = code.split(',')
    n = len(ls)
    argnames = list()
    i, j = 0, 1
    while j <= n:
        s = ','.join(ls[i:j])
        try:
            compile(s.lstrip(), '<string>', 'single')
        except SyntaxError:
            j += 1
        else:  # 没有错误的时候执行
            argnames.append(s.strip())
            i = j
            j = i + 1

    # 5 获得变量值和类型
    res['argvals'] = origin_args
    res['types'] = list(map(typename, origin_args))

    if not argnames:  # 如果在命令行环境下调用，argnames会有空，需要根据argvals长度置空名称
        argnames = [''] * len(res['argvals'])
    res['argnames'] = argnames

    return res


def dformat(*args, depth=2,
            delimiter=' ' * 4,
            strfunc=repr,
            fmt='[{depth:02}]{filename}/{lineno}: {argmsg}',
            subfmt='{name}<{tp}>={val}'):
    r"""
    :param args:  需要检查的表达式
        这里看似没有调用，其实在func_input_message用inspect会提取到args的信息
    :param depth: 处理对象
        默认值2，即处理dformat本身
        2以下值没意义
        2以上的值，可以不传入args参数
    :param delimiter: 每个变量值展示之间的分界
    :param strfunc: 对每个变量值的文本化方法，常见的有repr、str
    :param fmt: 展示格式，除了func_input_message中的关键字，新增
        argmsg：所有的「变量名=变量值」，或所有的「变量名<变量类型>=变量值」，或自定义格式，采用delimiter作为分界符
        旧版还用过这种格式： '{filename}/{funcname}/{lineno}: {argmsg}    {comment}'
    :param subfmt: 自定义每个变量值对的显示形式
        name，变量名
        val，变量值
        tp，变量类型
    :return: 返回格式化好的文本字符串
    """
    res = func_input_message(depth)
    ls = [subfmt.format(name=name, val=strfunc(val), tp=tp)
          for name, val, tp in zip(res['argnames'], res['argvals'], res['types'])]
    res['argmsg'] = delimiter.join(ls)
    return fmt.format(**res)


def dprint(*args, **kwargs):
    r"""
    # 故意写的特别复杂，测试在极端情况下是否能正确解析出表达式
    >> a, b = 1, 2
    >> re.sub(str(dprint(1, b, a, "aa" + "bb)", "a[,ba\nbb""b", [2, 3])), '', '##')  # 注释 # 注
    [08]<doctest debuglib.dprint[1]>/1: 1<int>=1    b<int>=2    a<int>=1    "aa" + "bb)"<str>='aabb)'    "a[,ba\nbb""b"<str>='a[,ba\nbbb'    [2, 3]<list>=[2, 3]    ##')  # 注释 # 注
    '##'
    """
    print(dformat(depth=3, **kwargs))


# dprint会被注册进builtins，可以在任意地方直接使用
setattr(builtins, 'dprint', dprint)


class DPrint:
    """ 用来存储上下文相关变量，进行全局性地调试

    TODO 需要跟logging库一样，可以获取不同名称的配置
        可以进行很多扩展，比如输出到stderr还是stdout
    """

    watch = {}

    @classmethod
    def reset(cls):
        cls.watch = {}

    @classmethod
    def format(cls, watch2, show_type=False, sep=' '):
        """
        :param watch2: 必须也是字典类型
        :param show_type: 是否显示每个数值的类型
        :param sep: 每部分的间隔符
        """
        msg = []
        input_msg = func_input_message(2)
        filename, lineno = input_msg['filename'], input_msg['lineno']
        msg.append(f'{filename}/{lineno}')

        watch3 = cls.watch.copy()
        watch3.update(watch2)
        for k, v in watch3.items():
            if k.startswith('$'):
                # 用 $ 修饰的不显示变量名，直接显示值
                msg.append(f'{v}')
            else:
                if show_type:
                    msg.append(f'{k}<{typename(v)}>={repr(v)}')
                else:
                    msg.append(f'{k}={repr(v)}')

        return sep.join(msg)


def format_exception(e):
    return ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))


def prettifystr(s):
    """对一个对象用更友好的方式字符串化

    :param s: 输入类型不做限制，会将其以友好的形式格式化
    :return: 格式化后的字符串
    """
    title = ''
    if isinstance(s, str):
        pass
    elif isinstance(s, Counter):  # Counter要按照出现频率显示
        li = s.most_common()
        title = f'collections.Counter长度：{len(s)}\n'
        # 不使用复杂的pd库，先简单用pprint即可
        # df = pd.DataFrame.from_records(s, columns=['value', 'count'])
        # s = dataframe_str(df)
        s = pprint.pformat(li)
    elif isinstance(s, (list, tuple)):
        title = f'{typename(s)}长度：{len(s)}\n'
        s = pprint.pformat(s)
    elif isinstance(s, (dict, set)):
        title = f'{typename(s)}长度：{len(s)}\n'
        s = pprint.pformat(s)
    else:  # 其他的采用默认的pformat
        s = pprint.pformat(s)
    return title + s


class PrettifyStrDecorator:
    """将函数的返回值字符串化（调用 prettifystr 美化）"""

    def __init__(self, func):
        self.func = func  # 使用self.func可以索引回原始函数名称
        self.last_raw_res = None  # last raw result，上一次执行函数的原始结果

    def __call__(self, *args, **kwargs):
        self.last_raw_res = self.func(*args, **kwargs)
        return prettifystr(self.last_raw_res)


def hide_console_window():
    """ 隐藏命令行窗口 """
    import ctypes
    kernel32 = ctypes.WinDLL('kernel32')
    user32 = ctypes.WinDLL('user32')
    SW_HIDE = 0
    hWnd = kernel32.GetConsoleWindow()
    user32.ShowWindow(hWnd, SW_HIDE)


def get_installed_packages():
    """ 使用pip list获取当前环境安装了哪些包 """
    output = subprocess.check_output(["pip", "list"], universal_newlines=True)
    packages = [line.split()[0] for line in output.split("\n")[2:] if line]
    return packages


class OutputLogger(logging.Logger):
    """
    我在jupyter写代码，经常要print输出一些中间结果。
    但是当结果很多的时候，又要转存到文件里保存起来查看。
    要保存到文件时，和普通的print写法是不一样的，一般要新建立一个ls = []的变量。
    然后print改成ls.append操作，会很麻烦。

    就想着能不能自己定义一个类，支持.print方法，不仅能实现正常的输出控制台的功能。
    也能在需要的时候指定文件路径，会自动将结果存储到文件中。
    """

    def __init__(self, name='OutputLogger', log_file=None, output_to_console=True):
        """
        :param str name: 记录器的名称。默认为 'OutputLogger'。
        :param log_file: 日志文件的路径。默认为 None，表示不输出到文件。
        :param bool output_to_console: 是否输出到命令行。默认为 True。
        """
        super().__init__(name)

        self.output_to_console = output_to_console
        self.log_file = log_file

        self.setLevel(logging.INFO)
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s/%(lineno)d - %(message)s',
                                      '%Y-%m-%d %H:%M:%S')

        # 提前重置为空文件
        with open(log_file, 'w') as f:
            f.write('')

        # 创建文件日志处理器
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)

        # 创建命令行日志处理器
        if output_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.addHandler(console_handler)

    def print(self, *args, **kwargs):
        msg = print2string(*args, **kwargs)

        if self.output_to_console:
            print(msg, end='')

        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(msg)

        return msg
