#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/08/13 14:53


import numpy as np
from shapely.geometry import Polygon
import cv2


def ndim(coords):
    coords = coords if isinstance(coords, np.ndarray) else np.array(coords)
    return coords.ndim


def coords1d(coords, dtype=None):
    """ 转成一维点数据

    [(x1, y1), (x2, y2), ...] --> [x1, y1, x2, y2, ...]
    会尽量遵循原始的array、list等结构返回

    >>> coords1d([(1, 2), (3, 4)])
    [1, 2, 3, 4]
    >>> coords1d(np.array([[1, 2], [3, 4]]))
    array([1, 2, 3, 4])
    >>> coords1d([1, 2, 3, 4])
    [1, 2, 3, 4]

    >>> coords1d([[1.5, 2], [3.5, 4]])
    [1.5, 2.0, 3.5, 4.0]
    >>> coords1d([1, 2, [3, 4], [5, 6, 7]])  # 这种情况，[3,4]、[5,6,7]都是一个整体
    [1, 2, [3, 4], [5, 6, 7]]
    >>> coords1d([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    """

    if isinstance(coords, (list, tuple)):
        return np.array(coords, dtype=dtype).reshape(-1).tolist()
    elif isinstance(coords, np.ndarray):
        return np.array(coords, dtype=dtype).reshape(-1)
    else:
        raise TypeError(f'未知类型 {coords}')


def coords2d(coords, m=2, dtype=None):
    """ 一维的点数据转成二维点数据

    :param m: 转成行列结构后，每列元素数，默认2个

    [x1, y1, x2, y2, ...] --> [(x1, y1), (x2, y2), ...]
    会尽量遵循原始的array、list等结构返回

    >>> coords2d([1, 2, 3, 4])
    [[1, 2], [3, 4]]
    >>> coords2d(np.array([1, 2, 3, 4]))
    array([[1, 2],
           [3, 4]])
    >>> coords2d([[1, 2], [3, 4]])
    [[1, 2], [3, 4]]

    >>> coords2d([1.5, 2, 3.5, 4])
    [[1.5, 2.0], [3.5, 4.0]]
    >>> coords2d([1.5, 2, 3.5, 4], dtype=int)  # 数据类型转换
    [[1, 2], [3, 4]]
    """
    if isinstance(coords, (list, tuple)):
        return np.array(coords, dtype=dtype).reshape((-1, m)).tolist()
    elif isinstance(coords, np.ndarray):
        return np.array(coords, dtype=dtype).reshape((-1, m))
    else:
        raise TypeError(f'未知类型 {coords}')


def divide_quadrangle(coords, r1=0.5, r2=None):
    """ 切分一个四边形为两个四边形
    :param coords: 1*8的坐标，或者4*2的坐标
    :param r1: 第一个切分比例，0.5相当于中点（即第一个四边形右边位置）
    :param r2: 第二个切分比例，即第二个四边形左边位置
    :return: 返回切割后所有的四边形

    一般用在改标注结果中，把一个框拆成两个框
    TODO 把接口改成切分一个四边形为任意多个四边形？即把r1、r2等整合为一个list参数输入
    """

    # 1 计算分割点工具
    def segment_point(pt1, pt2, rate=0.5):
        """ 两点间的分割点
        :param rate: 默认0.5是二分点，rate为0时即pt1，rate为1时为pt2，取值可以小于0、大于-1
        :return:
        """
        x1, y1 = pt1
        x2, y2 = pt2
        x, y = x1 + rate * (x2 - x1), y1 + rate * (y2 - y1)
        return int(x), int(y)

    # 2 优化参数值
    coords = coords2d(coords)
    if not r2: r2 = 1 - r1

    # 3 计算切分后的四边形坐标
    pt1, pt2, pt3, pt4 = coords
    pt5, pt6 = segment_point(pt1, pt2, r1), segment_point(pt4, pt3, r1)
    pt7, pt8 = segment_point(pt1, pt2, r2), segment_point(pt4, pt3, r2)
    return [pt1, pt5, pt6, pt4], [pt7, pt2, pt3, pt8]


def rect_bounds(coords, dtype=int):
    """ 多边形的最大外接矩形
    :param coords: 任意多边形的一维值[x1, y1, x2, y2, ...]，或者二维结构[(x1, y1), (x2, y2), ...]
    :param dtype: 默认存储的数值类型
    :return: rect的两个点坐标
    """
    p = Polygon(coords2d(coords)).bounds
    x1, y1, x2, y2 = [dtype(v) for v in p]
    return [[x1, y1], [x2, y2]]


def rect_bounds1d(coords, dtype=int):
    """ 多边形的最大外接矩形
    :param coords: 任意多边形的一维值[x1, y1, x2, y2, ...]，或者二维结构[(x1, y1), (x2, y2), ...]
    :param dtype: 默认存储的数值类型
    :return: rect的两个点坐标
    """
    p = Polygon(coords2d(coords)).bounds
    x1, y1, x2, y2 = [dtype(v) for v in p]
    return [x1, y1, x2, y2]


def rect2polygon(x1, y1, x2, y2):
    """
    TODO 这个可能要用类似多进制的方式，做多种格式间的来回转换
    """
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


____warp_points = """
仿射、透视变换相关功能
"""


def get_warp_mat(src, dst):
    """
    :param src: 原点集，支持多种格式输入
    :param dst: 变换后的点集
    :return: np.ndarray，3*3的变换矩阵
    """

    def cvt_data(pts):
        # opencv的透视变换，输入的点集有类型限制，必须使用float32
        return np.array(pts, dtype='float32').reshape((-1, 2))

    src, dst = cvt_data(src), cvt_data(dst)
    n = src.shape[0]
    if n == 3:
        # 只有3个点，则使用仿射变换
        warp_mat = cv2.getAffineTransform(src, dst)
        warp_mat = np.concatenate([warp_mat, [[0, 0, 1]]], axis=0)
    elif n == 4:
        # 有4个点，则使用透视变换
        warp_mat = cv2.getPerspectiveTransform(src, dst)
    else:
        raise ValueError('点集数量过多')
    return warp_mat


def warp_points(pts, warp_mat):
    """ 透视等点集坐标转换

    :param pts: 支持list、tuple、np.ndarray等结构，支持1d、2d的维度
        其实这个坐标变换就是一个简单的矩阵乘法，只是pts的数据结构往往比较特殊，
        并不是一个n*3的矩阵结构，所以需要进行一些简单的格式转换
        例如 [x1, y1, x2, y2, x3, y3] --> [[x1, x2, x3], [y1, y2, y3], [1, 1, 1]]
    :param warp_mat: 变换矩阵，一般是个3*3的矩阵，但是只输入2*3的矩阵也行，因为第3行并用不到
    :return: 会遵循原始的 pts 数据类型、维度结构返回

    >>> warp_mat = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]  # 对换x、y
    >>> warp_points([[1, 2], [11, 22]], warp_mat)  # 处理两个点
    [[2, 1], [22, 11]]
    >>> warp_points([[1, 2], [11, 22]], [[0, 1, 0], [1, 0, 0]])  # 输入2*3的变换矩阵也可以
    [[2, 1], [22, 11]]
    >>> warp_points([1, 2, 11, 22], warp_mat)  # 也可以用一维的结构来输入点集
    [2, 1, 22, 11]
    >>> warp_points([1, 2, 11, 22, 111, 222], warp_mat)  # 点的数量任意，返回的结构同输入的结构形式
    [2, 1, 22, 11, 222, 111]
    >>> warp_points(np.array([1, 2, 11, 22, 111, 222]), warp_mat)  # 也可以用np.ndarray等结构
    array([  2,   1,  22,  11, 222, 111])
    """
    pts1 = np.array(pts).reshape(-1, 2).T
    pts1 = np.concatenate([pts1, [[1] * pts1.shape[1]]], axis=0)
    pts2 = np.dot(warp_mat[:2], pts1)
    pts2 = pts2.T
    if ndim(pts) == 1:
        pts2 = pts2.reshape(-1)
    if not isinstance(pts, np.ndarray):
        pts2 = pts2.tolist()
    return pts2


def warp_image(img, warp_mat, dsize=None):
    """ 对图像进行透视变换

    :param img: np.ndarray的图像数据
    :param warp_mat: 变换矩阵
    :param dsize: 目标图片尺寸，默认同原图（注意是先输入列，在输入行）
    """
    if dsize is None:
        dsize = (img.shape[1], img.shape[0])
    dst = cv2.warpPerspective(img, warp_mat, dsize)
    return dst
