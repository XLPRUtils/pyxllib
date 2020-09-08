#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/08/13 14:53

import copy

import numpy as np
from shapely.geometry import Polygon
import cv2

from pyxllib.debug import dprint


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


def rect_bounds1d(coords, dtype=int):
    """ 多边形的最大外接矩形
    :param coords: 任意多边形的一维值[x1, y1, x2, y2, ...]，或者二维结构[(x1, y1), (x2, y2), ...]
    :param dtype: 默认存储的数值类型
    :return: rect的两个点坐标，同时也是 [left, top, right, bottom]
    """
    pts = coords2d(coords)
    if len(pts) > 2:
        p = Polygon(pts).bounds
    else:
        pts = coords1d(pts)
        p = [min(pts[::2]), min(pts[1::2]), max(pts[::2]), max(pts[1::2])]
    return [dtype(v) for v in p]


def rect_bounds(coords, dtype=int):
    """ 多边形的最大外接矩形
    :param coords: 任意多边形的一维值[x1, y1, x2, y2, ...]，或者二维结构[(x1, y1), (x2, y2), ...]
    :param dtype: 默认存储的数值类型
    :return: rect的两个点坐标
    """
    x1, y1, x2, y2 = rect_bounds1d(coords, dtype=dtype)
    return [[x1, y1], [x2, y2]]


def rect2polygon(x1, y1, x2, y2):
    """
    TODO 这个可能要用类似多进制的方式，做多种格式间的来回转换
    """
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


____warp_perspective = """
仿射、透视变换相关功能

https://www.yuque.com/xlpr/pyxllib/warpperspective
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


def retain_pts_struct(pts, ref_pts):
    """ 参考ref_pts的数据结构，设置pts的结构
    :return: 更新后的pts
    """
    # 1 确保都是np.ndarray，好分析问题
    if not isinstance(pts, np.ndarray):
        pts = np.array(pts)
    if not isinstance(ref_pts, np.ndarray):
        ref_pts2 = np.array(ref_pts)
    else:
        ref_pts2 = ref_pts

    # 2 参照ref_pts的结构
    if ref_pts2.ndim == 1:
        pts = pts.reshape(-1)
    elif ref_pts2.ndim == 2:
        pts = pts.reshape((-1, ref_pts2.shape[1]))
    if not isinstance(ref_pts, np.ndarray):
        pts = pts.tolist()
    return pts


def warp_points(pts, warp_mat):
    """ 透视等点集坐标转换

    :param pts: 支持list、tuple、np.ndarray等结构，支持1d、2d的维度
        其实这个坐标变换就是一个简单的矩阵乘法，只是pts的数据结构往往比较特殊，
        并不是一个n*3的矩阵结构，所以需要进行一些简单的格式转换
        例如 [x1, y1, x2, y2, x3, y3] --> [[x1, x2, x3], [y1, y2, y3], [1, 1, 1]]
    :param warp_mat: 变换矩阵，一般是个3*3的矩阵，但是只输入2*3的矩阵也行，因为第3行并用不到（点集只要取前两个维度X'Y'的结果值）
        TODO 不过这里我有个点也没想明白，如果不用第3行，本质上不是又变回仿射变换了，如何达到透视变换效果？第三维的深度信息能完全舍弃？
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
    return retain_pts_struct(pts2, pts)


def warp_image(img, warp_mat, dsize=None, *, view_rate=False, max_zoom=1):
    """ 对图像进行透视变换

    :param img: np.ndarray的图像数据
        TODO 支持PIL.Image格式？
    :param warp_mat: 变换矩阵
    :param dsize: 目标图片尺寸
        没有任何输入时，同原图
        如果有指定，则会决定最终的图片大小
        如果使用了view_rate、max_zoom，会改变变换矩阵所展示的内容
    :param view_rate: 视野比例，默认不开启，当输入非0正数时，几个数值功能效果如下
        1，关注原图四个角点位置在变换后的位置，确保新的4个点依然在目标图中
            为了达到该效果，会增加【平移】变换，以及自动控制dsize
        2，将原图依中心面积放到至2倍，记录新的4个角点变换后的位置，确保变换后的4个点依然在目标图中
        0.5，同理，只是只关注原图局部的一半位置
    :param max_zoom: 默认1倍，当设置时（只在开启view_rate时有用），会增加【缩小】变换，限制view_rate扩展的上限
    """
    from math import sqrt

    # 1 得到3*3的变换矩阵
    if not isinstance(warp_mat, np.ndarray):
        warp_mat = np.array(warp_mat)
    if warp_mat.shape[0] == 2:
        warp_mat = np.concatenate([warp_mat, [[0, 0, 1]]], axis=0)

    # 2 view_rate，视野比例改变导致的变换矩阵规则变化
    if view_rate:
        # 2.1 视野变化后的四个角点
        h, w = img.shape[:2]
        y, x = h / 2, w / 2  # 图片中心点坐标
        h1, w1 = view_rate * h / 2, view_rate * w / 2
        l, t, r, b = [-w1 + x, -h1 + y, w1 + x, h1 + y]
        pts1 = np.array([[l, t], [r, t], [r, b], [l, b]])
        # 2.2 变换后角点位置产生的外接矩形
        left, top, right, bottom = rect_bounds1d(warp_points(pts1, warp_mat))
        # 2.3 增加平移变换确保左上角在原点
        warp_mat = np.dot(np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]]), warp_mat)
        # 2.4 控制面积变化率
        h2, w2 = (bottom - top, right - left)
        if max_zoom:
            rate = w2 * h2 / w / h  # 目标面积比原面积
            if rate > max_zoom:
                r = 1 / sqrt(rate / max_zoom)
                warp_mat = np.dot(np.array([[r, 0, 0], [0, r, 0], [0, 0, 1]]), warp_mat)
                h2, w2 = round(h2 * r), round(w2 * r)
        if not dsize:
            dsize = (w2, h2)

    # 3 标准操作，不做额外处理，按照原图默认的图片尺寸展示
    if dsize is None:
        dsize = (img.shape[1], img.shape[0])
    dst = cv2.warpPerspective(img, warp_mat, dsize)
    return dst


def quad_warp_wh(pts, method='average'):
    """ 四边形转为矩形的宽、高
    :param pts: 四个点坐标
        TODO 暂时认为pts是按点集顺时针顺序输入的
        TODO 暂时认为pts[0]就是第一个坐标点
    :param method:
        记四条边分别为w1, h1, w2, h2
        average: 平均宽、高
        max: 最大宽、高
        min: 最小宽、高
    :return: (w, h) 变换后的矩形宽、高
    """
    # 1 计算四边长
    from math import hypot
    pts = coords2d(pts)
    lens = [0] * 4
    for i in range(4):
        pt1, pt2 = pts[i], pts[(i + 1) % 4]
        lens[i] = hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    # 2 目标宽、高
    if method == 'average':
        w, h = (lens[0] + lens[2]) / 2, (lens[1] + lens[3]) / 2
    elif method == 'max':
        w, h = max(lens[0], lens[2]), max(lens[1], lens[3])
    elif method == 'min':
        w, h = min(lens[0], lens[2]), min(lens[1], lens[3])
    else:
        raise ValueError(f'不支持的方法 {method}')
    # 这个主要是用于图像变换的，而图像一般像素坐标要用整数，所以就取整运算了
    return round(w), round(h)


def warp_quad_pts(pts, method='average'):
    """ 将不规则四边形转为矩形
    :param pts: 不规则四边形的四个点坐标
    :param method: 计算矩形宽、高的算法
    :return: 规则矩形的四个点坐标

    >>> warp_quad_pts([[89, 424], [931, 424], [399, 290], [621, 290]])
    [[0, 0], [532, 0], [532, 549], [0, 549]]
    >>> warp_quad_pts([89, 424, 931, 424, 399, 290, 621, 290])
    [0, 0, 532, 0, 532, 549, 0, 549]
    """
    w, h = quad_warp_wh(pts, method)
    pts2 = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    return retain_pts_struct(pts2, pts)


def get_sub_image(src_image, pts, warp_quad=False):
    """ 从src_image取一个子图

    :param src_image: 原图
        可以是图片路径、np.ndarray、PIL.Image对象
        TODO 目前先只支持np.ndarray格式
    :param pts: 子图位置信息
        只有两个点，认为是矩形的两个角点
        只有四个点，认为是任意四边形
        同理，其他点数量，默认为
    :param warp_quad: 变形的四边形
        默认是截图pts的外接四边形区域，使用该参数
            且当pts为四个点时，是否强行扭转为矩形
    :return: 子图
        文件、np.ndarray --> np.ndarray
        PIL.Image --> PIL.Image
    """
    pts = coords2d(pts)
    if not warp_quad or len(pts) != 4:
        x1, y1, x2, y2 = rect_bounds1d(pts)
        dst = src_image[y1:y2, x1:x2]
    else:
        w, h = quad_warp_wh(pts, method=warp_quad)
        pts2 = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        warp_mat = get_warp_mat(pts, pts2)
        dst = warp_image(src_image, warp_mat, (w, h))
    return dst
