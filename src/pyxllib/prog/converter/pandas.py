"""pandas的df类型相关的互转"""

from functools import singledispatch
from collections import Counter

import pandas as pd

from pyxllib.prog.newbie import typename
from pyxllib.prog.converter import to_list


@singledispatch
def to_df(obj, *args, **kwargs):
    """通用的转DataFrame函数

    :param obj: 待转换的对象
    :return: pd.DataFrame
    """
    return obj


@to_df.register(list)
@to_df.register(tuple)
def _(li, *args, **kwargs):
    if li and isinstance(li[0], (list, tuple)):  # 有两维时按表格显示
        df = pd.DataFrame.from_records(li)
    else:  # 只有一维时按一列显示
        df = pd.DataFrame(pd.Series(li), columns=(typename(li),))
    return df


@to_df.register(dict)
def _(d, *args, **kwargs):
    name = typename(d)
    li = to_list(d, nsort=True)
    return pd.DataFrame.from_records(li, columns=(f"{name}-key", f"{name}-value"))


@to_df.register(Counter)
def _(d, *args, **kwargs):
    name = typename(d)
    li = d.most_common()
    return pd.DataFrame.from_records(li, columns=(f"{name}-key", f"{name}-value"))


@to_df.register(pd.Series)
def _(s, *args, **kwargs):
    return pd.DataFrame(s)


@to_df.register(pd.DataFrame)
def _(df, *args, **kwargs):
    return df
