#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 11:00

import concurrent.futures
import functools
import logging
import queue
import signal
import threading
import time


class RunOnlyOnce:
    """ 被装饰的函数，不同的参数输入形式，只会被执行一次，

    重复执行时会从内存直接调用上次相同参数调用下的运行的结果
    可以使用reset成员函数重置，下一次调用该函数时则会重新执行

    文档：https://www.yuque.com/xlpr/pyxllib/RunOnlyOnce

    使用好该装饰器，可能让一些动态规划dp、搜索问题变得更简洁，
    以及一些配置文件操作，可以做到只读一遍
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

    @classmethod
    def decorator(cls, distinct_args=True):
        """ 作为装饰器的时候，如果要设置参数，要用这个接口 """

        def wrap(func):
            return cls(func, distinct_args)

        return wrap

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


def run_once(distinct_mode=0, *, limit=1, debug=False):
    """ 装饰器，装饰的函数在一次程序里其实只会运行一次

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
    returns: 返回decorator
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


class Timeout:
    """ 对函数等待执行的功能，限制运行时间

    【实现思路】
    1、最简单的方式是用signal.SIGALRM实现（包括三方库timeout-decorator也有这个局限）
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
            ③ 这里是让子线程抛出信号，主线程收到信号后，再抛出TimeoutError

    注意：这个函数似乎不支持多线程
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
        if self.seconds == 0:  # 可以设置0来关闭超时功能
            return

        def overtime(signum, frame):
            raise TimeoutError(f'with 上下文代码块运行超时 > [{self.seconds} 秒]')

        signal.signal(signal.SIGABRT, overtime)

        def send_signal():
            signal.raise_signal(signal.SIGABRT)

        # 挂起一个警告器，如果"没人"管它，self.seconds就会抛出错误
        self.alarm = threading.Timer(self.seconds, send_signal)
        self.alarm.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seconds == 0:
            return

        # with已经运行完了，马上关闭警告器
        self.alarm.cancel()


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


class XlThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.futures = []

    def submit(self, *args, **kwargs):
        future = super().submit(*args, **kwargs)
        self.futures.append(future)
        return future

    def yield_result(self, timeout=None):
        for future in self.futures:
            yield future.result(timeout=timeout)


def xlwait(func, condition=bool, *, timeout=None, interval=1):
    """ 不断重复执行func，直到得到满足condition条件的期望值

    :param condition: 退出等待的条件，默认为bool真值
    :param timeout: 重复执行的上限时间（单位 秒），默认一直等待
    :param interval: 重复执行间隔 （单位 秒）

    """
    t = time.time()
    while True:
        res = func()
        if condition(res):
            return res
        elif timeout and (time.time() - t > timeout):
            return res  # 超时也返回目前得到的结果
        if interval > 0:
            time.sleep(interval)


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

    240826周一，其他可参考学习的三方现成工具库：from fastcore.foundation import patch
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


def inplace_decorate(parent, func_name, wrapper):
    """ 将指定的函数替换为装饰器版本
    允许在运行时动态地将一个函数或方法替换为其装饰版本。通常用于添加日志、性能测试、事务处理等。
    （既然可以写成装饰版本，相当于其实要完全替换成另外的函数也是可以的）

    当然，因为py一切皆对象，这里处理的不是函数，而是其他变量等对象也是可以的

    这个功能跟直接把原代码替换修改了还是有区别的，如果原函数在被这个装饰之前，已经被其他地方调用，或者被装饰器补充，
        太晚使用这个装饰，并不会改变前面已经运行、被捕捉的情况
        遇到这种情况，也可以考虑在原函数定义后，直接紧接着加上这个函数重置

    对于类成员方法，直接用这个设置可能也不行，只能去改源码了
        比如要给函数加计算时间的部分，可以考虑使用 get_global_var 等来夸作用域记录时间数据
    """

    if hasattr(parent, func_name):
        if callable(wrapper):  # 对函数的封装
            original_func = getattr(parent, func_name)

            @functools.wraps(original_func)
            def decorated_func(*args, **kwargs):
                return wrapper(original_func, *args, **kwargs)

            setattr(parent, func_name, decorated_func)
        else:  # 对数值的封装，不过如果是数值，其实也没必要调用这个函数，直接赋值就好了
            return setattr(parent, func_name, wrapper)
    else:  # 否则按照字典的模式来处理

        original_func = parent[func_name]

        @functools.wraps(original_func)
        def decorated_func(*args, **kwargs):
            return wrapper(original_func, *args, **kwargs)

        parent[func_name] = decorated_func
