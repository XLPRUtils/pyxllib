#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/09/26 09:47

""" 单位功能库 """

import pint

from pyxllib.prog.newbie import RunOnlyOnce


@RunOnlyOnce
def get_ureg():
    """ 如果不想重复生成，可以用这个函数，只定义一个 """
    ureg = pint.UnitRegistry()
    return ureg
