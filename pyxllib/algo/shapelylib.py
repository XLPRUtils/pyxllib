#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/23 11:26

import re

from pyxllib.prog.lazyimport import lazy_import

try:
    import numpy as np
except ModuleNotFoundError:
    np = lazy_import('numpy')

try:
    import shapely
    from shapely.geometry import Polygon
except ModuleNotFoundError:
    shapely = lazy_import('shapely')
    Polygon = lazy_import('from shapely.geometry import Polygon')

from pyxllib.algo.geo import rect2polygon


class ShapelyPolygon:
    @classmethod
    def gen(cls, x):
        """ 转成shapely的Polygon对象

        :param x: 支持多种格式，详见代码
        :return: Polygon

        >>> print(ShapelyPolygon.gen([[0, 0], [10, 20]]))  # list
        POLYGON ((0 0, 10 0, 10 20, 0 20, 0 0))
        >>> print(ShapelyPolygon.gen({'shape_type': 'polygon', 'points': [[0, 0], [10, 0], [10, 20], [0, 20]]}))  # labelme shape
        POLYGON ((0 0, 10 0, 10 20, 0 20, 0 0))
        >>> print(ShapelyPolygon.gen('107,247,2358,209,2358,297,107,335'))  # 字符串格式
        POLYGON ((107 247, 2358 209, 2358 297, 107 335, 107 247))
        >>> print(ShapelyPolygon.gen('107 247.5, 2358 209.2, 2358 297, 107.5 335'))  # 字符串格式
        POLYGON ((107 247.5, 2358 209.2, 2358 297, 107.5 335, 107 247.5))
        """
        if isinstance(x, Polygon):
            return x
        elif isinstance(x, dict) and 'points' in x:
            if x['shape_type'] in ('rectangle', 'polygon'):
                # 目前这种情况一般是输入了labelme的shape格式
                return ShapelyPolygon.gen(x['points'])
            else:
                raise ValueError('无法转成多边形的类型')
        elif isinstance(x, str):
            coords = re.findall(r'[\d\.]+', x)
            return ShapelyPolygon.gen(coords)
        else:
            x = np.array(x).reshape((-1, 2))
            if x.shape[0] == 2:
                x = rect2polygon(x)
                x = np.array(x)
            if x.shape[0] >= 3:
                return Polygon(x)
            else:
                raise ValueError

    @classmethod
    def to_ndarray(cls, p, dtype=None):
        return np.array(p.exterior.coords, dtype=dtype)
