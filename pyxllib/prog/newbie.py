#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

from pyxllib.excel.newbie import int2excel_col_name


def typename(c):
    """ 简化输出的type类型

    >>> typename(123)
    'int'
    """
    return str(type(c))[8:-2]


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


def document(func):
    """文档函数装饰器
    用该装饰器器时，表明一个函数是用伪代码在表示一系列的操作逻辑，不能直接拿来执行的
    很可能是一套半自动化工具
    """

    def wrapper(*args):
        raise RuntimeError(f'函数:{func.__name__} 是一个伪代码流程示例文档，不能直接运行')

    return wrapper


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


def len_in_dim2(arr):
    """ 计算类List结构在第2维上的最大长度

    >>> len_in_dim2([[1,1], [2], [3,3,3]])
    3

    >>> len_in_dim2([1, 2, 3])  # TODO 是不是应该改成0合理？但不知道牵涉到哪些功能影响
    1
    """
    if not isinstance(arr, (list, tuple)):
        raise TypeError('类型错误，不是list构成的二维数组')

    # 找出元素最多的列
    column_num = 0
    for i, item in enumerate(arr):
        if isinstance(item, (list, tuple)):  # 该行是一个一维数组
            column_num = max(column_num, len(item))
        else:  # 如果不是数组，是指单个元素，当成1列处理
            column_num = max(column_num, 1)

    return column_num


def ensure_array(arr, default_value=''):
    """对一个由list、tuple组成的二维数组，确保所有第二维的列数都相同

    >>> ensure_array([[1,1], [2], [3,3,3]])
    [[1, 1, ''], [2, '', ''], [3, 3, 3]]
    """
    max_cols = len_in_dim2(arr)
    if max_cols == 1:
        return arr
    dv = str(default_value)
    a = [[]] * len(arr)
    for i, ls in enumerate(arr):
        if isinstance(ls, (list, tuple)):
            t = list(arr[i])
        else:
            t = [ls]  # 如果不是数组，是指单个元素，当成1列处理
        a[i] = t + [dv] * (max_cols - len(t))  # 左边的写list，是防止有的情况是tuple，要强制转list后拼接
    return a


def swap_rowcol(a, *, ensure_arr=False, default_value=''):
    """矩阵行列互换

    注：如果列数是不均匀的，则会以最小列数作为行数

    >>> swap_rowcol([[1,2,3], [4,5,6]])
    [[1, 4], [2, 5], [3, 6]]
    """
    if ensure_arr:
        a = ensure_array(a, default_value)
    # 这是非常有教学意义的行列互换实现代码
    return list(map(list, zip(*a)))


def gentuple(n, tag):
    """ 有点类似range函数，但生成的数列更加灵活

    :param n:
        数组长度
    :param tag:
        int类型，从指定数字开始编号
            0，从0开始编号
            1，从1开始编号
        'A'，用Excel的形式编号
        tuple，按枚举值循环显示
            ('A', 'B')：循环使用A、B编号

    >>> gentuple(4, 'A')
    ('A', 'B', 'C', 'D')
    """
    a = [''] * n
    if isinstance(tag, int):
        for i in range(n):
            a[i] = i + tag
    elif tag == 'A':
        a = tuple(map(lambda x: int2excel_col_name(x + 1), range(n)))
    elif isinstance(tag, (list, tuple)):
        k = len(tag)
        a = tuple(map(lambda x: tag[x % k], range(n)))
    return a


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


class GenFunction:
    """ 一般用来生成高阶函数的函数对象

    这个名字可能还不是很精确，后面有想法再改
    """

    @classmethod
    def ensure_func(cls, x, default):
        """ 确保x是callable对象，否则用default初始化 """
        if callable(x):
            return x
        else:
            return default


def first_nonnone(args, judge=None):
    """ 返回第1个满足条件的值

    :param args: 参数清单
    :param judge: 判断器，默认返回第一个非None值，也可以自定义判定函数
    """
    judge = GenFunction.ensure_func(judge, lambda x: x is not None)
    for x in args:
        if judge(x):
            return x


def round_int(x):
    """ 先四舍五入，再取整 """
    return int(round(x, 0))


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
