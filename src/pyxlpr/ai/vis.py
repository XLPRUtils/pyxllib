#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/11/07 14:04

import numpy as np


def show_feature_map(feature_map, show=True, *, pading=5):
    """ 显示特征图 """
    from pyxllib.xlcv import xlcv

    a = np.array(feature_map)
    a = a - a.min()
    m = a.max()
    if m:
        a = (a / m) * 255
    a = a.astype('uint8')

    if a.ndim == 3:
        a = xlcv.concat(list(a), pad=pading)
    elif a.ndim == 4:
        a = xlcv.concat([list(x) for x in a], pad=pading)

    if show:
        xlcv.show(a)

    return a
