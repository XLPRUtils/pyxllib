#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 23:21


""" 封装一些代码开发中常用的功能，工程组件 """
from collections import defaultdict
import datetime
import io
import itertools
import json
import math
import os
import queue
import socket
import subprocess
import sys
import tempfile
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


def run_once(distinct_mode=False, *, limit=1):
    """
    Args:
        distinct_mode:
            0，默认False，不区分输入的参数值（包括cls、self），强制装饰的函数只运行一次
            'str'，设为True或1时，仅以字符串化的差异判断是否是重复调用，参数不同，会判断为不同的调用，每种调用限制最多执行limit次
            'id,str'，在'str'的基础上，第一个参数使用id代替。一般用于类方法、对象方法的装饰。
                不考虑类、对象本身的内容改变，只要还是这个类或对象，视为重复调用。
            'ignore,str'，首参数忽略，第2个开始的参数使用str格式化
                用于父类某个方法，但是子类继承传入cls，原本id不同会重复执行
                使用该模式，首参数会ignore忽略，只比较第2个开始之后的参数
            func等callable类型的对象也行，是使用run_once装饰器的简化写法
        limit: 默认只会执行一次，该参数可以提高限定的执行次数，一般用不到，用于兼容旧的 limit_call_number 装饰器
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
