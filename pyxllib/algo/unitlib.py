#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/09/26 09:47

""" 单位功能库 """

from pyxllib.prog.lazyimport import lazy_import

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import('pandas')

try:
    import pint
except ModuleNotFoundError:
    pint = lazy_import('pint')

from pyxllib.prog.newbie import xl_format_g
from pyxllib.prog.pupil import run_once


@run_once
def get_ureg():
    """ 如果不想重复生成，可以用这个函数，只定义一个 """
    ureg = pint.UnitRegistry()
    return ureg


def compare_quantity(units, base_unit, scale=None):
    """ 同一物理维度的数值量级大小对比

    比如一套天文长度、距离数据，对比它们之间的量级大小

    :param List[str, str] units: 一组要对比的单位数值
        第1个str是描述，第2个str是数值和单位
    :param base_unit: 基础单位值，比如长度是m，也可以用km等自己喜欢的基础单位值
    :param scale: 对数值进行缩放
    """
    ureg = get_ureg()

    # 统一到单位：米
    for x in units:
        x[1] = ureg(x[1]).to(base_unit)

    # 从小到大排
    units.sort(key=lambda x: x[1])

    # 格式化输出r
    unit_list = []
    if scale:
        columns = ['name', f'基础单位：{base_unit}', 'scale', '与上一项倍率']
    else:
        columns = ['name', f'基础单位：{base_unit}', 'scale', '与上一项倍率']
    last_value = units[0][1]
    for i, x in enumerate(units, start=1):
        x = x.copy()
        # x.append('{:.3g}'.format(round(float(x[1] / last_value), 2)))
        if scale:
            x.append(f'{x[1] * scale:.2e}')
        x.append(xl_format_g(round(float(x[1] / last_value), 2)))
        last_value = x[1]
        x[1] = f'{x[1]:.2e}'
        # print(f'{i:>4}', '\t'.join(x))
        unit_list.append(x)

    df = pd.DataFrame.from_records(unit_list, columns=columns)
    return df
