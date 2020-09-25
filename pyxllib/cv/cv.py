#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/08/13 14:53


"""
以后这里cv功能多了，可以再拆子文件夹
"""

import copy
import re

import numpy as np
import cv2
import PIL.Image
from shapely.geometry import Polygon

from pyxllib.basic import Path
from pyxllib.debug import dprint, showdir

____ensure_array_type = """
数组方面的类型转换，相关类型有

list (tuple)  1d, 2d
np.ndarray    1d, 2d
PIL.Image.Image
Polygon

这里封装的目的，是尽量减少不必要的数据重复拷贝，需要做些底层的特定判断优化
"""


def np_array(x, dtype=None, shape=None):
    """确保数据是np.ndarray结构，如果不是则做一个转换

    如果x已经是np.ndarray，尽量减小数据的拷贝，提高效率

    :param dtype: 还可以顺便指定数据类型，可以修改值的存储类型

    TODO 增加一些字符串初始化方法，例如类似matlab这样的 [1 2 3 4]。虽然从性能角度不推荐，但是工程化应该提供尽可能完善全面的功能。
    """
    if isinstance(x, np.ndarray):
        if x.dtype == np.dtype(dtype):
            y = x
        else:
            y = np.array(x, dtype=dtype)
    elif isinstance(x, PIL.Image.Image):
        # PIL RGB图像数据转 np的 BGR数据
        y = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
        if dtype: y = np.array(x, dtype=dtype)
    elif isinstance(x, Polygon):
        y = np.array(x.exterior.coords, dtype=dtype)
    else:
        y = np.array(x, dtype=dtype)

    if shape:
        y = y.reshape(shape)

    return y


def to_list(x, dtype=None, shape=None):
    """
    :param x:
    :param shape: 输入格式如：-1, (-1, ), (-1, 2)
    :return: list、tuple嵌套结构
    """
    if isinstance(x, (list, tuple)):
        # 1 尽量不要用这两个参数，否则一定会使用到np矩阵作为中转
        if dtype or shape:
            y = np_array(x, dtype, shape)
        else:
            y = x
    else:
        y = np_array(x, dtype, shape).tolist()
    return y


def pil_image(x):
    if isinstance(x, PIL.Image.Image):
        y = x
    else:
        y = np_array(x)
        y = PIL.Image.fromarray(cv2.cvtColor(y, cv2.COLOR_BGR2RGB)) if y.size else None
    return y


def shapely_polygon(x):
    """ 转成shapely的Polygon对象

    :param x: 支持多种格式，详见代码
    :return: Polygon

    >>> print(shapely_polygon([[0, 0], [10, 20]]))  # list
    POLYGON ((0 0, 10 0, 10 20, 0 20, 0 0))
    >>> print(shapely_polygon({'shape_type': 'polygon', 'points': [[0, 0], [10, 0], [10, 20], [0, 20]]}))  # labelme shape
    POLYGON ((0 0, 10 0, 10 20, 0 20, 0 0))
    >>> print(shapely_polygon('107,247,2358,209,2358,297,107,335'))  # 字符串格式
    POLYGON ((107 247, 2358 209, 2358 297, 107 335, 107 247))
    >>> print(shapely_polygon('107 247.5, 2358 209.2, 2358 297, 107.5 335'))  # 字符串格式
    POLYGON ((107 247.5, 2358 209.2, 2358 297, 107.5 335, 107 247.5))
    """
    from shapely.geometry import Polygon

    if isinstance(x, Polygon):
        return x
    elif isinstance(x, dict) and 'points' in x:
        if x['shape_type'] in ('rectangle', 'polygon'):
            # 目前这种情况一般是输入了labelme的shape格式
            return shapely_polygon(x['points'])
        else:
            raise ValueError('无法转成多边形的类型')
    elif isinstance(x, str):
        coords = re.findall(r'[\d\.]+', x)
        return shapely_polygon(coords)
    else:
        x = np_array(x, shape=(-1, 2))
        if x.shape[0] == 2:
            x = rect2polygon(x)
            x = np.array(x)
        if x.shape[0] >= 3:
            return Polygon(x)
        else:
            raise ValueError


def ensure_array_type(src_data, target_type):
    """ 参考 target 的数据结构，重设src的结构

    目前支持的数据结构
        PIL.Image.Image，涉及到图像格式转换的，为了与opencv兼容，一律以BGR为准，除了Image自身默认用RGB
        np.ndarray
        list
        shapely polygon

    :param src_data: 原数据
    :param target_type: 正常是指定一个type类型
    :return: 重置结构后的src数据

    >>> ensure_array_type([1, 2, 3, 4], type([[1, 1], [2, 2]]))
    [1, 2, 3, 4]
    >>> ensure_array_type([1, 2, 3, 4], type(np.array([[1, 1], [2, 2]])))
    array([1, 2, 3, 4])
    >>> ensure_array_type([1, 2, 3, 4], np.ndarray)
    array([1, 2, 3, 4])
    >>> ensure_array_type(np.array([[1, 2], [3, 4]]), type([10, 20, 30, 40]))
    [[1, 2], [3, 4]]
    >>> ensure_array_type(np.array([]), type([10, 20, 30, 40]))
    []

    其他测试：
    # np矩阵转PIL图像
    img = cv2.imread(r'textline.jpg')
    dst = reset_arr_struct(img, PIL.Image.Image)
    print(type(dst))  # <class 'PIL.Image.Image'>

    # PIL图像转np矩阵
    img = Image.open('textline.jpg')
    dst = reset_arr_struct(img, np.ndarray)
    print(type(dst))  # <class 'numpy.ndarray'>
    """
    # 根据不同的目标数据类型，进行格式转换
    if target_type == np.ndarray:
        return np_array(src_data)
    elif target_type in (list, tuple):
        return np_array(src_data).tolist()
    elif target_type == PIL.Image.Image:
        return pil_image(src_data)
    elif target_type == Polygon:
        return shapely_polygon(src_data)
    else:
        raise TypeError(f'未知目标类型 {target_type}')


____base = """

"""


def get_ndim(coords):
    # 注意 np.array(coords[:1])，只需要取第一个元素就可以判断出ndim
    coords = coords if isinstance(coords, np.ndarray) else np.array(coords[:1])
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
        return np_array(coords, dtype=dtype).reshape(-1).tolist()
    elif isinstance(coords, np.ndarray):
        return np_array(coords, dtype=dtype).reshape(-1)
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
        return np_array(coords, dtype=dtype).reshape((-1, m)).tolist()
    elif isinstance(coords, np.ndarray):
        return np_array(coords, dtype=dtype).reshape((-1, m))
    else:
        raise TypeError(f'未知类型 {coords}')


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


def rect2polygon(src):
    """ 矩形转成四边形结构来表达存储

    >>> rect2polygon([[0, 0], [10, 20]])
    array([[ 0,  0],
           [10,  0],
           [10, 20],
           [ 0, 20]])
    >>> rect2polygon(np.array([0, 0, 10, 20]))
    array([[ 0,  0],
           [10,  0],
           [10, 20],
           [ 0, 20]])
    """
    x1, y1, x2, y2 = np_array(src).reshape(-1)
    dst = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return dst


def resort_quad_points(src_pts):
    """ 重置四边形点集顺序，确保以左上角为起点，顺时针罗列点集

    算法：先确保pt1、pt2在上面，然后再确保pt1在pt2左边

    >>> pts = [[100, 50], [200, 0], [100, 0], [0, 50]]
    >>> resort_quad_points(pts)
    [[100, 0], [200, 0], [100, 50], [0, 50]]
    >>> pts  # 原来的点不会被修改
    [[100, 50], [200, 0], [100, 0], [0, 50]]
    """
    # numpy的交换会有问题！必须要转为list结构
    src_type = type(src_pts)
    pts = to_list(src_pts)
    if src_type == list:
        # list的时候比较特别，要拷贝、不能引用数据
        pts = copy.copy(pts)
    if pts[0][1] > pts[2][1]:
        pts[0], pts[2] = pts[2], pts[0]
    if pts[1][1] > pts[3][1]:
        pts[1], pts[3] = pts[3], pts[1]
    if pts[0][0] > pts[1][0]:
        pts[0], pts[1] = pts[1], pts[0]
        pts[2], pts[3] = pts[3], pts[2]
    return ensure_array_type(pts, src_type)


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


def warp_points(pts, warp_mat, reserve_struct=True):
    """ 透视等点集坐标转换

    :param pts: 支持list、tuple、np.ndarray等结构，支持1d、2d的维度
        其实这个坐标变换就是一个简单的矩阵乘法，只是pts的数据结构往往比较特殊，
        并不是一个n*3的矩阵结构，所以需要进行一些简单的格式转换
        例如 [x1, y1, x2, y2, x3, y3] --> [[x1, x2, x3], [y1, y2, y3], [1, 1, 1]]
    :param warp_mat: 变换矩阵，一般是个3*3的矩阵，但是只输入2*3的矩阵也行，因为第3行并用不到（点集只要取前两个维度X'Y'的结果值）
        TODO 不过这里我有个点也没想明白，如果不用第3行，本质上不是又变回仿射变换了，如何达到透视变换效果？第三维的深度信息能完全舍弃？
    :param reserve_struct: 是否保留原来pts的结构返回，默认True
        关掉该功能可以提高性能，此时返回结果统一为 n*2 的np矩阵
    :return: 会遵循原始的 pts 数据类型、维度结构返回

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
    >>> warp_points([1, 2, 11, 22], warp_mat, reserve_struct=False)  # 也可以用一维的结构来输入点集
    array([[ 2,  1],
           [22, 11]])
    """
    pts1 = np_array(pts).reshape(-1, 2).T
    pts1 = np.concatenate([pts1, [[1] * pts1.shape[1]]], axis=0)
    pts2 = np.dot(warp_mat[:2], pts1)
    pts2 = pts2.T
    return pts2


def warp_image(img, warp_mat, dsize=None, *, view_rate=False, max_zoom=1, reserve_struct=True):
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
    :param reserve_struct: 是否保留原来img的数据类型返回，默认True
        关掉该功能可以提高性能，此时返回结果统一为 np 矩阵
    :return: 见 reserve_struct
    """
    from math import sqrt

    # 0 参数整理
    img0 = img
    if not isinstance(img, np.ndarray):
        img = np_array(img)

    # 1 得到3*3的变换矩阵
    warp_mat = np_array(warp_mat)
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
        warp_mat = np.dot([[1, 0, -left], [0, 1, -top], [0, 0, 1]], warp_mat)
        # 2.4 控制面积变化率
        h2, w2 = (bottom - top, right - left)
        if max_zoom:
            rate = w2 * h2 / w / h  # 目标面积比原面积
            if rate > max_zoom:
                r = 1 / sqrt(rate / max_zoom)
                warp_mat = np.dot([[r, 0, 0], [0, r, 0], [0, 0, 1]], warp_mat)
                h2, w2 = round(h2 * r), round(w2 * r)
        if not dsize:
            dsize = (w2, h2)

    # 3 标准操作，不做额外处理，按照原图默认的图片尺寸展示
    if dsize is None:
        dsize = (img.shape[1], img.shape[0])
    dst = cv2.warpPerspective(img, warp_mat, dsize)

    # 4 返回值
    return dst


____get_sub_image = """
"""


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
    array([[  0,   0],
           [532,   0],
           [532, 549],
           [  0, 549]])
    """
    w, h = quad_warp_wh(pts, method)
    return rect2polygon([0, 0, w, h])


def get_sub_image(src_image, pts, warp_quad=False):
    """ 从src_image取一个子图

    :param src_image: 原图
        可以是图片路径、np.ndarray、PIL.Image对象
        TODO 目前先只支持np.ndarray格式
    :param pts: 子图位置信息
        只有两个点，认为是矩形的两个对角点
        只有四个点，认为是任意四边形
        同理，其他点数量，默认为
    :param warp_quad: 变形的四边形
        默认是截图pts的外接四边形区域，使用该参数
            且当pts为四个点时，是否强行扭转为矩形
    :return: 子图
        文件、np.ndarray --> np.ndarray
        PIL.Image --> PIL.Image
    """
    src_img = ensure_array_type(src_image, np.ndarray)
    pts = coords2d(pts)
    if not warp_quad or len(pts) != 4:
        x1, y1, x2, y2 = rect_bounds1d(pts)
        dst = src_img[y1:y2, x1:x2]  # 这里越界不会报错，只是越界的那个维度shape为0
    else:
        w, h = quad_warp_wh(pts, method=warp_quad)
        warp_mat = get_warp_mat(pts, rect2polygon([0, 0, w, h]))
        dst = warp_image(src_img, warp_mat, (w, h))
    return dst


____opencv = """
对opencv相关功能的一些优化、 封装
"""


def imread_v1(path, flags=1):
    """ opencv 源生的 imread不支持中文路径，所以要用PIL先读取，然后再转np.ndarray

    :param flags:
        0，转成 GRAY
        1，转成 BGR

    TODO 不知道新版的imread会不会出什么问题~~
        所以这个版本的代码暂时先留着
    """
    src = PIL.Image.open(str(path))
    src = np.array(src)  # 如果原图是灰度图，获得的可能是单通道的结果

    if flags == 0:
        if src.ndim == 3:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    elif flags == 1:
        if src.ndim == 3:
            src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        else:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(flags)
    return src


def imread(path, flags=1):
    """ https://www.yuque.com/xlpr/pyxllib/imread

    cv2.imread的flags默认参数相当于是1

    """
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def imwrite(path, img, if_exists='replace'):
    """
    TODO 200922周二16:21，如何更好与Path融合？能直接Path('a.jpg').write(img)？
    """
    if not isinstance(path, Path):
        path = Path(path)
    data = cv2.imencode(ext=path.suffix, img=img)[1]
    return path.write(data.tobytes(), if_exists=if_exists)


def imshow(mat, winname=None, flags=0):
    """ 展示窗口

    :param mat:
    :param winname: 未输入时，则按test1、test2依次生成窗口
    :param flags:
        cv2.WINDOW_NORMAL，0，输入2等偶数值好像也等价于输入0
        cv2.WINDOW_AUTOSIZE，1，输入3等奇数值好像等价于1
        cv2.WINDOW_OPENGL，4096
    :return:
    """
    if winname is None:
        imshow.num = getattr(imshow, 'num', 0) + 1
        winname = f'test{imshow.num}'
    cv2.namedWindow(winname, flags)
    cv2.imshow(winname, mat)


def plot_lines(src, lines, color=None, thickness=1, line_type=cv2.LINE_AA, shift=None):
    """ 在src图像上画系列线段
    """
    # 1 判断 lines 参数内容
    if lines is None:
        return src
    if not isinstance(lines, np.ndarray):
        lines = np.array(lines)
    if not lines.any():
        return src

    # 2 预备
    dst = np.array(src)  # 拷贝一张图来修改
    if color is None:
        if src.ndim == 3:
            color = (0, 0, 255)  # TODO 可以根据背景色智能推导画线用的颜色，目前是固定红色
        elif src.ndim == 2:
            color = [255]  # 灰度图，默认先填白色

    # 3 画线
    if lines.any():
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = line
            cv2.line(dst, (x1, y1), (x2, y2), color, thickness, line_type, shift)
    return dst


class TrackbarTool:
    """ 滑动条控件组
    """

    def __init__(self, winname, img, flags=0):
        if not isinstance(img, np.ndarray):
            img = imread(str(img))
        cv2.namedWindow(winname, flags)
        cv2.imshow(winname, img)
        self.winname = winname
        self.img = img
        self.trackbar_names = {}

    def imshow(self, img=None):
        """ 刷新显示的图片 """
        if img is None:
            img = self.img
        cv2.imshow(self.winname, img)

    def default_run(self, x):
        """ 默认执行器，这个在类继承后，基本都是要自定义成自己的功能的

        TODO 从1滑到20，会运行20次，可以研究一个机制，来只运行一次
        """
        kwargs = {}
        for k in self.trackbar_names.keys():
            kwargs[k] = self[k]
        print(kwargs)

    def create_trackbar(self, trackbar_name, count, value=0, on_change=None):
        """ 创建一个滑动条
        :param trackbar_name: 滑动条名称
        :param count: 上限值
        :param on_change: 回调函数
        :param value: 初始值
        :return:
        """
        if on_change is None:
            on_change = self.default_run
        cv2.createTrackbar(trackbar_name, self.winname, value, count, on_change)

    def __getitem__(self, item):
        """ 可以通过 Trackbars 来获取滑动条当前值
        :param item: 滑动条名称
        :return: 当前取值
        """
        return cv2.getTrackbarPos(item, self.winname)


def get_background_color(src_img, edge_size=5, binary_img=None):
    """ 智能判断图片背景色

    对全图二值化后，考虑最外一层宽度未edge_size的环中，0、1分布最多的作为背景色
        然后取全部背景色的平均值返回

    :param src_img: 支持黑白图、彩图
    :param edge_size: 边缘宽度，宽度越高一般越准确，但也越耗性能
    :param binary_img: 运算中需要用二值图，如果外部已经计算了，可以直接传入进来，避免重复运算
    :return: color

    TODO 可以写个获得前景色，道理类似，只是最后再图片中心去取平均值
    """
    from itertools import chain

    # 1 获得二值图，区分前背景
    if binary_img is None:
        gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY) if src_img.ndim == 3 else src_img
        _, binary_img = cv2.threshold(gray_img, np.mean(gray_img), 255, cv2.THRESH_BINARY)

    # 2 分别存储点集
    n, m = src_img.shape[:2]
    colors0, colors1 = [], []
    for i in range(n):
        if i < edge_size or i >= n - edge_size:
            js = range(m)
        else:
            js = chain(range(edge_size), range(m - edge_size, m))
        for j in js:
            if binary_img[i, j]:
                colors1.append(src_img[i, j])
            else:
                colors0.append(src_img[i, j])

    # 3 计算平均像素
    # 以数量多的作为背景像素
    colors = colors0 if len(colors0) > len(colors1) else colors1
    return np.mean(np.array(colors), axis=0, dtype='int').tolist()


____polygon = """
"""


def intersection_over_union(pts1, pts2):
    """ 两个多边形的交并比 Intersection Over Union
    :param pts1: 可以转成polygon的数据类型
    :param pts2:可以转成polygon的数据类型
    :return: 交并比

    >>> intersection_over_union([[0, 0], [10, 10]], [[5, 5], [15, 15]])
    0.14285714285714285

    TODO 其实，如果有大量的双循环两两对比，每次判断shapely类型是比较浪费性能的
        此时可以考虑直接在业务层用三行代码计算，不必调用该函数
        同时，我也要注意一些密集型的底层计算函数，应该尽量避免这种工程性类型判断的泛用操作，会影响性能
    """
    polygon1, polygon2 = shapely_polygon(pts1), shapely_polygon(pts2)
    inter_area = polygon1.intersection(polygon2).area
    union_area = polygon1.area + polygon2.area - inter_area
    return inter_area / union_area


def non_maximun_suppression():
    raise NotImplementedError


____other = """
"""


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
