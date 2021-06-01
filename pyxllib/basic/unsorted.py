import os
import re
import struct

from disjoint_set import DisjointSet


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


____disjoint_set = """
并查集相关功能
"""


def disjoint_set(items, join_checker):
    """ 按照一定的相连规则分组

    :param items: 项目清单
    :param join_checker: 检查任意两个对象是否相连，进行分组
    :return:

    算法：因为会转成下标，按照下标进行分组合并，所以支持items里有重复值，或者unhashable对象

    >>> disjoint_set([-1, -2, 2, 0, 0, 1], lambda x, y: x*y>0)
    [[-1, -2], [2, 1], [0], [0]]
    """
    from itertools import combinations

    # 1 添加元素
    ds = DisjointSet()
    items = tuple(items)
    n = len(items)
    for i in range(n):
        ds.find(i)

    # 2 连接、分组
    for i, j in combinations(range(n), 2):
        if join_checker(items[i], items[j]):
            ds.union(i, j)

    # 3 返回分组信息
    res = []
    for group in ds.itersets():
        group_elements = [items[g] for g in group]
        res.append(group_elements)
    return res


____other = """
"""


def xlbool(v):
    """ 有些类型不能直接判断真假，例如具有多值的np.array，df等

    这些有歧义的情况，在我的mybool里暂时都判断为True，如果有需要，需要精细化判断，可以扩展自己的npbool、dfbool
    """
    try:
        return bool(v)
    except ValueError:
        return True


def struct_unpack(f, fmt):
    r""" 类似np.fromfile的功能，读取并解析二进制数据

    :param f:
        如果带有read方法，则用read方法读取指定字节数
        如果bytes对象则直接处理
    :param fmt: 格式
        默认按小端解析(2, 1, 0, 0) -> 258，如果需要大端，可以加前缀'>'
        字节：c=char, b=signed char, B=unsigned char, ?=bool
        2字节整数：h=short, H=unsigned short（后文同理，大写只是变成unsigned模式，不在累述）
        4字节整数：i, I, l, L
        8字节整数：q, Q
        浮点数：e=2字节，f=4字节，d=8字节

    >>> b = struct.pack('B', 127)
    >>> b
    b'\x7f'
    >>> struct_unpack(b, 'c')
    b'\x7f'
    >>> struct_unpack(b, 'B')
    127

    >>> b = struct.pack('I', 258)
    >>> b
    b'\x02\x01\x00\x00'
    >>> struct_unpack(b, 'I')  # 默认都是按小端打包、解析
    258
    >>> struct_unpack(b, '>I') # 错误示范，按大端解析的值
    33619968
    >>> struct_unpack(b, 'H'*2)  # 解析两个值，fmt*2即可
    (258, 0)

    >>> f = io.BytesIO(b'\x02\x01\x03\x04')
    >>> struct_unpack(f, 'B'*3)  # 取前3个值，等价于np.fromfile(f, dtype='uint8', count=3)
    (2, 1, 3)
    >>> struct_unpack(f, 'B')  # 取出第4个值
    4
    """
    # 1 取数据
    size_ = struct.calcsize(fmt)
    if hasattr(f, 'read'):
        data = f.read(size_)
        if len(data) < size_:
            raise ValueError(f'剩余数据长度 {len(data)} 小于 fmt 需要的长度 {size_}')
    else:  # 对于bytes等矩阵，可以多输入，但是只解析前面一部分
        data = f[:size_]

    # 2 解析
    res = struct.unpack(fmt, data)
    if len(res) == 1:  # 解析结果恰好只有一个的时候，返回值本身
        return res[0]
    else:
        return res


class ContentPartSpliter:
    """ 文本内容分块处理 """

    @classmethod
    def multi_blank_lines(cls, content, leastlines=2):
        """ 用多个空行隔开的情况

        :param leastlines: 最少2个空行隔开，为新的一块内容
        """
        fmt = r'\n{' + str(leastlines) + ',}'
        parts = [x.strip() for x in re.split(fmt, content)]
        parts = list(filter(bool, parts))  # 删除空行
        return parts


def get_username():
    return os.path.split(os.path.expanduser('~'))[-1]


def linux_path_fmt(p):
    p = str(p)
    p = p.replace('\\', '/')
    return p


def product(*iterables, order=None, repeat=1):
    """ 对 itertools 的product扩展orders参数的更高级的product迭代器

    :param order: 假设iterables有n=3个迭代器，则默认 orders=[1, 2, 3] （起始编号1）
        即标准的product，是按顺序对每个迭代器进行重置、遍历的
        但是我扩展的这个接口，允许调整每个维度的更新顺序
        例如设置为 [-2, 1, 3]，表示先对第2维降序，然后按第1、3维的方式排序获得各个坐标点
        注：可以只输入[-2]，默认会自动补充维[1, 3]

        不从0开始编号，是因为0没法记录正负排序情况

    for x in product('ab', 'cd', 'ef', order=[3, -2, 1]):
        print(x)

    ['a', 'd', 'e']
    ['b', 'd', 'e']
    ['a', 'c', 'e']
    ['b', 'c', 'e']
    ['a', 'd', 'f']
    ['b', 'd', 'f']
    ['a', 'c', 'f']
    ['b', 'c', 'f']

    TODO 我在想numpy这么牛逼，会不会有等价的功能接口可以实现，我不用重复造轮子？
    """
    import itertools, numpy

    # 一、标准调用方式
    if order is None:
        for x in itertools.product(*iterables, repeat=repeat):
            yield x
        return

    # 二、输入orders参数的调用方式
    # 1 补全orders参数长度
    n = len(iterables)
    for i in range(1, n + 1):
        if not (i in order or -i in order):
            order.append(i)
    if len(order) != n: return ValueError(f'orders参数值有问题 {order}')

    # 2 生成新的迭代器组
    new_iterables = [(iterables[i - 1] if i > 0 else reversed(iterables[-i - 1])) for i in order]
    idx = numpy.argsort([abs(i) - 1 for i in order])
    for y in itertools.product(*new_iterables, repeat=repeat):
        yield [y[i] for i in idx]
