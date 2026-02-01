"""类型互转工具"""

from functools import singledispatch

from pyxllib.algo.pupil import natural_sort_key


@singledispatch
def to_tuple(obj, *args, **kwargs):
    """通用的转tuple函数"""
    return tuple(obj)


@singledispatch
def to_list(obj, *args, **kwargs):
    """通用的转list函数"""
    return list(obj)


@to_list.register(dict)
def _(d: dict, *, nsort=False):
    """字典转n*2的list

    :param d: 字典
    :param nsort:
        True: 对key使用自然排序
        False: 使用d默认的遍历顺序
    :return:
    """
    ls = list(d.items())
    if nsort:
        ls = sorted(ls, key=lambda x: natural_sort_key(str(x[0])))
    return ls


@singledispatch
def to_dict(obj, *args, **kwargs):
    """通用的转dict函数"""
    return dict(obj)
