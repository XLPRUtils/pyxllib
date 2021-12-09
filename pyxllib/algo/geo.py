#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/11/15 10:16

""" 几何、数学运算

specialist级别
"""

import copy

import numpy as np
import cv2

from pyxllib.algo.intervals import Intervals

____base = """

"""


def xywh2ltrb(p):
    return [p[0], p[1], p[0] + p[2], p[1] + p[3]]


def ltrb2xywh(p):
    return [p[0], p[1], p[2] - p[0], p[3] - p[1]]


def rect2polygon(src_pts):
    """ 矩形对角线两个点，转成四边形四个点的模式来表达
    （输入左上、右下两个顶点坐标）

    :param list|np.ndarray src_pts: size 2*2
    :rtype: list

    >>> rect2polygon([[0, 0], [10, 20]])
    [[0, 0], [10, 0], [10, 20], [0, 20]]
    >>> rect2polygon(np.array([[0, 0], [10, 20]]))
    [[0, 0], [10, 0], [10, 20], [0, 20]]
    >>> rect2polygon([[10, 0], [0, 20]])
    [[0, 0], [10, 0], [10, 20], [0, 20]]
    """
    [[x1, y1], [x2, y2]] = src_pts
    dst_pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    dst_pts = resort_quad_points(dst_pts)
    return dst_pts


def reshape_coords(coords, m, dtype=None):
    """ 重置坐标点的维度

    :param list coords: 这个函数主要还是封装了对list情况的处理
        其实np.ndarray结构也行，但这种情况直接用np接口操作就行，不需要引用该函数
    :rtype: list

    # 转成 n*1 的矩阵

    >>> reshape_coords([(1, 2), (3, 4)], 1)
    [1, 2, 3, 4]
    >>> reshape_coords(np.array([[1, 2], [3, 4]]), 1)
    [1, 2, 3, 4]
    >>> reshape_coords([1, 2, 3, 4], 1)
    [1, 2, 3, 4]

    >>> reshape_coords([[1.5, 2], [3.5, 4]], 1)
    [1.5, 2.0, 3.5, 4.0]

    # 这种情况，[3,4]、[5,6,7]都是一个整体
    # VisibleDeprecationWarning
    >>> reshape_coords([1, 2, [3, 4], [5, 6, 7]], 1)
    [1, 2, [3, 4], [5, 6, 7]]

    >>> reshape_coords([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], 1)
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # 变成 n*2 的矩阵

    >>> reshape_coords([1, 2, 3, 4], 2)
    [[1, 2], [3, 4]]
    >>> reshape_coords(np.array([1, 2, 3, 4]), 2)
    [[1, 2], [3, 4]]
    >>> reshape_coords([[1, 2], [3, 4]], 2)
    [[1, 2], [3, 4]]
    >>> reshape_coords([1.5, 2, 3.5, 4], 2)
    [[1.5, 2.0], [3.5, 4.0]]
    >>> reshape_coords([1.5, 2, 3.5, 4], 2, dtype=int)  # 数据类型转换
    [[1, 2], [3, 4]]
    """
    if m == 1:
        return np.array(coords, dtype=dtype).reshape(-1).tolist()
    else:
        return np.array(coords, dtype=dtype).reshape((-1, m)).tolist()


def rect_bounds(coords):
    """ 多边形的最大外接矩形

    :param coords: 支持list、np等类型，支持1d、2d两种维度表达方式
    :return: rect的两个点坐标，同时也是 [left, top, right, bottom]
    """
    pts = np.array(coords).reshape(-1).tolist()  # tolist不能删，不然int类型就变了。比如int64不能json.dump
    p = [min(pts[::2]), min(pts[1::2]), max(pts[::2]), max(pts[1::2])]
    return [v for v in p]


def resort_quad_points(src_pts):
    """ 重置四边形点集顺序，确保以左上角为起点，顺时针罗列点集

    算法：先确保pt1、pt2在上面，然后再确保pt1在pt2左边

    :param list|tuple|np.ndarray src_pts: 点集
    :rtype: list|np.ndarray

    >>> pts = [[100, 50], [200, 0], [100, 0], [0, 50]]
    >>> resort_quad_points(pts)
    [[100, 0], [200, 0], [100, 50], [0, 50]]
    >>> pts  # 原来的点不会被修改
    [[100, 50], [200, 0], [100, 0], [0, 50]]

    >>> pts = np.array([[100, 50], [200, 0], [100, 0], [0, 50]])
    >>> resort_quad_points(pts)
    array([[100,   0],
           [200,   0],
           [100,   0],
           [  0,  50]])
    >>> pts  # 原来的点不会被修改
    array([[100,  50],
           [200,   0],
           [100,   0],
           [  0,  50]])
    """
    pts = copy.copy(src_pts)
    if pts[0][1] > pts[2][1]:
        pts[0], pts[2] = pts[2], pts[0]
    if pts[1][1] > pts[3][1]:
        pts[1], pts[3] = pts[3], pts[1]
    if pts[0][0] > pts[1][0]:
        pts[0], pts[1] = pts[1], pts[0]
        pts[2], pts[3] = pts[3], pts[2]
    return pts


def ltrb_border(ltrb, border, size=None):
    """ 给原来的ltrb定位扩展border像素

    Args:
        ltrb:
        border: 可以一个数字，表示统一添加的像素值
            也可以四个数字，表示每个维度分别加的像素值
        size:
            原图的 (width, height)，防止越界
            可以不填，默认不考虑越界问题
    Returns: 新的ltrb坐标
    """
    if isinstance(border, int):
        border = [border] * 4

    l = max(0, ltrb[0] - border[0])
    t = max(0, ltrb[1] - border[1])
    r = min(size[0], ltrb[2] + border[2])
    b = min(size[1], ltrb[3] + border[3])

    return [l, t, r, b]


____warp_perspective = """
仿射、透视变换相关功能

https://www.yuque.com/xlpr/pyxllib/warpperspective
"""


def warp_points(pts, warp_mat):
    """ 透视等点集坐标转换

    :param list|tuple|np.ndarray pts: 支持1d、2d的维度
        其实这个坐标变换就是一个简单的矩阵乘法，只是pts的数据结构往往比较特殊，
        并不是一个n*3的矩阵结构，所以需要进行一些简单的格式转换
        例如 [x1, y1, x2, y2, x3, y3] --> [[x1, x2, x3], [y1, y2, y3], [1, 1, 1]]
    :param list|tuple|np.ndarray warp_mat: 变换矩阵，一般是个3*3的矩阵，但是只输入2*3的矩阵也行，因为第3行并用不到（点集只要取前两个维度X'Y'的结果值）
        TODO 不过这里我有个点也没想明白，如果不用第3行，本质上不是又变回仿射变换了，如何达到透视变换效果？第三维的深度信息能完全舍弃？
    :rtype: np.ndarray

    >>> warp_mat = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]  # 对换x、y
    >>> warp_points([[1, 2], [11, 22]], warp_mat)  # 处理两个点
    array([[ 2,  1],
           [22, 11]])
    >>> warp_points([[1, 2], [11, 22]], [[0, 1, 0], [1, 0, 0]])  # 输入2*3的变换矩阵也可以
    array([[ 2,  1],
           [22, 11]])
    >>> warp_points([1, 2, 11, 22], warp_mat)  # 也可以用一维的结构来输入点集
    array([[ 2,  1],
           [22, 11]])
    >>> warp_points([1, 2, 11, 22, 111, 222], warp_mat)  # 点的数量任意，返回的结构同输入的结构形式
    array([[  2,   1],
           [ 22,  11],
           [222, 111]])
    >>> warp_points(np.array([1, 2, 11, 22, 111, 222]), warp_mat)  # 也可以用np.ndarray等结构
    array([[  2,   1],
           [ 22,  11],
           [222, 111]])
    >>> warp_points([1, 2, 11, 22], warp_mat)  # 也可以用一维的结构来输入点集
    array([[ 2,  1],
           [22, 11]])
    """
    pts1 = np.array(pts).reshape(-1, 2).T
    pts1 = np.concatenate([pts1, [[1] * pts1.shape[1]]], axis=0)
    pts2 = np.dot(warp_mat[:2], pts1)
    pts2 = pts2.T
    return pts2


def get_warp_mat(src, dst):
    """ 从前后点集计算仿射变换矩阵

    :param src: 原点集，支持多种格式输入
    :param dst: 变换后的点集
    :return np.ndarray: 3*3的变换矩阵
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
    # pts = ReshapeCoords.list_2d(pts)
    lens = [0] * 4
    for i in range(4):
        pt1, pt2 = pts[i], pts[(i + 1) % 4]
        lens[i] = hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    # 2 目标宽、高
    if method is True:
        method = 'average'
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
    :return: 返回时，仍然用四个点的坐标表达，规则矩形的四个点坐标

    >>> warp_quad_pts([[89, 424], [931, 424], [399, 290], [621, 290]])
    [[0, 0], [532, 0], [532, 549], [0, 549]]
    """
    w, h = quad_warp_wh(pts, method)
    return rect2polygon([[0, 0], [w, h]])


____polygon = """
"""


class ComputeIou:
    """ 两个多边形的交并比 Intersection Over Union """

    @classmethod
    def ltrb(cls, pts1, pts2):
        """ https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        x_a = max(pts1[0], pts2[0])
        y_a = max(pts1[1], pts2[1])
        x_b = min(pts1[2], pts2[2])
        y_b = min(pts1[3], pts2[3])

        # compute the area of intersection rectangle
        inter_area = abs(max((x_b - x_a, 0)) * max((y_b - y_a), 0))
        if inter_area == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        box_a_area = abs((pts1[2] - pts1[0]) * (pts1[3] - pts1[1]))
        box_b_area = abs((pts2[2] - pts2[0]) * (pts2[3] - pts2[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        # return the intersection over union value
        return iou

    @classmethod
    def polygon(cls, pts1, pts2):
        inter_area = pts1.intersection(pts2).area
        if inter_area:
            union_area = pts1.area + pts2.area - inter_area
            return (inter_area / union_area) if union_area else 0
        else:
            return 0

    @classmethod
    def polygon2(cls, pts1, pts2):
        """ 会强制转为polygon对象再处理

        >>> ComputeIou.polygon2([[0, 0], [10, 10]], [[5, 5], [15, 15]])
        0.14285714285714285
        """
        from pyxllib.algo.shapelylib import ShapelyPolygon
        polygon1, polygon2 = ShapelyPolygon.gen(pts1), ShapelyPolygon.gen(pts2)
        return cls.polygon(polygon1, polygon2)

    @classmethod
    def nms_basic(cls, boxes, func, iou=0.5, *, key=None, index=False):
        """ 假设boxes已经按权重从大到小排过序

        :param boxes: 支持输入一组box列表 [box1, box2, box3, ...]
        :param key: 将框映射为可计算对象
        :param index: 返回不是原始框，而是对应的下标 [i1, i2, i3, ...]
        """
        # 1 映射到items来操作
        if callable(key):
            items = list(enumerate([key(b) for b in boxes]))
        else:
            items = list(enumerate(boxes))

        # 2 正常nms功能
        idxs = []
        while items:
            # 1 加入权值大的框
            i, b = items[0]
            idxs.append(i)
            # 2 抑制其他框
            left_items = []
            for j in range(1, len(items)):
                if func(b, items[j][1]) < iou:
                    left_items.append(items[j])
            items = left_items

        # 3 返回值
        if index:
            return idxs
        else:
            return [boxes[i] for i in idxs]

    @classmethod
    def nms_ltrb(cls, boxes, iou=0.5, *, key=None, index=False):
        return cls.nms_basic(boxes, cls.ltrb, iou, key=key, index=index)

    @classmethod
    def nms_xywh(cls, boxes, iou=0.5, *, key=None, index=False):
        if callable(key):
            func = lambda x: xywh2ltrb(key(x))
        else:
            func = xywh2ltrb
        return cls.nms_ltrb(boxes, iou, key=func, index=index)

    @classmethod
    def nms_polygon(cls, boxes, iou=0.5, *, key=None, index=False):
        # ShapelyPolygon.gen
        return cls.nms_basic(boxes, cls.polygon, iou, key=key, index=index)


____other = """
"""


def divide_quadrangle(coords, r1=0.5, r2=None):
    """ 切分一个四边形为两个四边形

    :param coords: 4*2的坐标
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
    # coords = ReshapeCoords.list_2d(coords)
    if not r2: r2 = 1 - r1

    # 3 计算切分后的四边形坐标
    pt1, pt2, pt3, pt4 = coords
    pt5, pt6 = segment_point(pt1, pt2, r1), segment_point(pt4, pt3, r1)
    pt7, pt8 = segment_point(pt1, pt2, r2), segment_point(pt4, pt3, r2)
    return [pt1, pt5, pt6, pt4], [pt7, pt2, pt3, pt8]


def split_vector_interval(vec, maxsplit=None, minwidth=3):
    """
    :param vec: 一个一维向量，需要对这个向量进行切割
        需要前置工作先处理好数值
            使得背景在非正数，背景概率越大，负值绝对值越大
            前景在正值，前景概率越大，数值越大
        要得到能量最大（数值最大、前景内容）的几个区域
        但是因为有噪声的原因，该算法要有一定的抗干扰能力

        一般情况下
            用 0 代表背景
            用 <1 的正值表示这一列黑点所占比例（np.mean）
            用 np.sum 传入整数暂时也行，但考虑以后功能扩展性，用比例会更好
            传入负数，表示特殊背景，该背景可以抵消掉的minwidth宽度数
    :param maxsplit: 最大切分数量，即最多得到几个子区间
        没设置的时候，会对所有满足条件的情况进行切割
    :param minwidth: 每个切分位置最小具有的宽度
    :return: [(l, r), (l, r), ...]  每一段文本的左右区间
    """
    # 1 裁剪左边、右边
    n_vec = len(vec)
    left, right = 0, n_vec
    while left < right and vec[left] <= 0:
        left += 1
    while right > left and vec[right - 1] <= 0:
        right -= 1
    # 左右空白至少也要达到minwidth才去除
    # if left < minwidth: left = 0
    # if n_vec - right + 1 < minwidth: right = n_vec

    vec = vec[left:right]
    width = len(vec)
    if width == 0:
        return []  # 没有内容，返回空list

    # 2 找切分位置
    #   统计每一段连续的背景长度，并且对其数值求和，作为这段是背景的置信度
    bg_probs, bg_start, cnt = [], 0, 0

    def update_fg():
        """ 遇到前景内容，或者循环结束，更新一下 """
        nonlocal cnt
        prob = vec[bg_start:bg_start + cnt].sum()
        # print(cnt, prob)
        if cnt >= (minwidth + prob):  # 负值可以减小minwidth限定
            itv = [bg_start, bg_start + cnt]
            bg_probs.append([itv, prob])
        cnt = 0

    for i in range(width):
        if vec[i] <= 0:
            if not cnt:
                bg_start = i
            cnt += 1
        else:
            update_fg()
    else:
        update_fg()

    # 3 取置信度最大的几个分割点
    if maxsplit:
        bg_probs = sorted(bg_probs, key=lambda x: x[1])[:(maxsplit - 1)]
    bg_probs = sorted(bg_probs, key=lambda x: x[0])  # 从左到右排序

    # 4 返回文本区间（反向计算）
    res = []
    intervals = Intervals([itv for itv, prob in bg_probs]).invert(width) + left
    # print(intervals)
    for interval in intervals:
        res.append([interval.start(), interval.end()])
    return res


def bound_scale(bound, scale):
    """ 一个矩形，以中心为原点，缩放面积为原来scale的新矩形

    :param bound: [x1, y1, x2, y2]
    :param scale: 比例，例如0.5，就是缩放一半
    """
    x1, y1, x2, y2 = bound
    x0 = (x2 - x1) / 2
    y0 = (y2 - y1) / 2
    r = 1 - scale ** 0.5

    x1 += r * abs(x0 - x1)
    y1 += r * abs(y0 - y1)
    x2 -= r * abs(x0 - x2)
    y2 -= r * abs(y0 - y2)
    return x1, y1, x2, y2
