#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

import hashlib
import itertools
import sys


def typename(c):
    """ 简化输出的type类型

    >>> typename(123)
    'int'
    """
    return str(type(c))[8:-2]


class SingletonForEveryClass(type):
    """ 注意如果A是单例类，B从A继承，那么实际有且仅有A、B两个不同的实例对象 """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        tag = f'{cls}'
        # 其实转字符串来判断是不太严谨的，有些类型字符串后的显示效果是一模一样的
        # dprint(tag)
        if tag not in cls._instances:
            cls._instances[tag] = super(SingletonForEveryClass, cls).__call__(*args, **kwargs)
        return cls._instances[tag]


class SingletonForEveryInitArgs(type):
    """ Python单例模式(Singleton)的N种实现 - 知乎: https://zhuanlan.zhihu.com/p/37534850

    注意！注意！注意！重要的事说三遍！
    这里的单例类不是传统意义上的单例类。
    传统意义的单例类，不管用怎样不同的初始化参数创建对象，永远都只有最初的那个对象类。
    但是这个单例类，为每种不同的参数创建形式，都构造了一个对象。
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


def document(func):
    """文档函数装饰器
    用该装饰器器时，表明一个函数是用伪代码在表示一系列的操作逻辑，不能直接拿来执行的
    很可能是一套半自动化工具
    """

    def wrapper(*args):
        raise RuntimeError(f'函数:{func.__name__} 是一个伪代码流程示例文档，不能直接运行')

    return wrapper


class CvtType:
    """ 这些系列的转换函数，转换失败统一报错为ValueError

    返回异常会比较好，否则返回None、False之类的，
        有时候可能就是要转换的值呢，会有歧义
    """

    @classmethod
    def str2list(cls, arg):
        try:
            res = eval(arg)
        except SyntaxError:
            raise ValueError

        if not isinstance(res, list):
            raise ValueError

        return res

    @classmethod
    def str2dict(cls, arg):
        try:
            res = eval(arg)
        except SyntaxError:
            raise ValueError

        if not isinstance(res, dict):
            raise ValueError

        return res

    @classmethod
    def factory(cls, name):
        """ 从字符串名称，映射到对应的转换函数 """
        return {'int': int,
                'float': float,
                'str': str,
                'list': cls.str2list,
                'dict': cls.str2dict}.get(name, None)


def mod_minabs(x, m):
    """
    >>> mod_minabs(5, 8)
    -3
    >>> mod_minabs(3, 8)
    3
    """
    a = x % m
    return a if a < m / 2 else a - m


class classproperty(property):
    """ python - Using property() on classmethods - Stack Overflow
        https://stackoverflow.com/questions/128573/using-property-on-classmethods
    """

    def __get__(self, obj, objtype=None):
        return super(classproperty, self).__get__(objtype)

    def __set__(self, obj, value):
        super(classproperty, self).__set__(type(obj), value)

    def __delete__(self, obj):
        super(classproperty, self).__delete__(type(obj))


def decode_bitflags(n, flags, return_type=dict):
    """ 解析一个位标记的功能

    :param int n: 一个整数标记
    :param list|tuple flags: 一个对应明文数组
        flags[0]对应2**0的明文，flags[1]对应2**1的明文，以此类推
    :param type return_type: 返回的数据类型
        默认dict，记录所有key的bool结果
        可以设set，只返回真的标记结果

    >>> decode_bitflags(18, ('superscript', 'italic', 'serifed', 'monospaced', 'bold'))
    {'superscript': 2, 'italic': 0, 'serifed': 0, 'monospaced': 16, 'bold': 0}
    >>> decode_bitflags(18, ('superscript', 'italic', 'serifed', 'monospaced', 'bold'), set)
    {'superscript', 'monospaced'}
    """
    if return_type == dict:
        return {x: n & (2 << i) for i, x in enumerate(flags)}
    elif return_type == set:
        return {x for i, x in enumerate(flags) if (n & (2 << i))}
    else:
        raise ValueError


class EmptyWith:
    """ 空上下文类
    """

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


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


def safe_div(a, b):
    """ 安全除法，避免除数为0的情况

    :param a: 被除数
    :param b: 除数
    :return: a/b，如果b为0，返回0
    """
    if b == 0:
        return a / sys.float_info.epsilon
    else:
        return a / b


def xlmd5(content):
    if isinstance(content, str):
        content = content.encode('utf-8')
    elif not isinstance(content, bytes):
        content = str(content).encode('utf-8')

    if len(content) <= 32:  # 32位以下的字符串，直接返回
        return content
    else:
        return hashlib.md5(content).hexdigest()


def generate_int_hash_from_str(s):
    """ 对字符串使用md5编码，然后转出一个数值哈希，一般是用来进行随机分组
    比如获得一个整数后，对3取余，就是按照余数为0、1、2的情况分3组
    """
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def get_groupid_from_string(s, n_groups):
    """ 通过计算一个字符串的哈希值来对其进行分组，需要提前知道总组别数n_groups """
    hash_value = generate_int_hash_from_str(s)
    return hash_value % n_groups


class GrowingList(list):
    """可变长list"""

    def __init__(self, default_value=None):
        super().__init__(self)
        self.default_value = default_value

    def __getitem__(self, index):
        if index >= len(self):
            self.extend([self.default_value] * (index + 1 - len(self)))
        return list.__getitem__(self, index)

    def __setitem__(self, index, value):
        if index >= len(self):
            self.extend([self.default_value] * (index + 1 - len(self)))
        list.__setitem__(self, index, value)
