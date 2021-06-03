#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/11/15 10:16

"""几何、数学运算"""

import numpy as np


def get_ndim(coords):
    # 注意 np.array(coords[:1])，只需要取第一个元素就可以判断出ndim
    coords = coords if isinstance(coords, np.ndarray) else np.array(coords[:1])
    return coords.ndim


def xywh2ltrb(p):
    return [p[0], p[1], p[0] + p[2], p[1] + p[3]]


def ltrb2xywh(p):
    return [p[0], p[1], p[2] - p[0], p[3] - p[1]]
