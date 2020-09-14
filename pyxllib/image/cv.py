#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/08/13 14:53


"""
以后这里cv功能多了，可以再拆子文件夹
"""

import copy

import numpy as np
import cv2
import PIL.Image
from shapely.geometry import Polygon

from pyxllib.debug import dprint

____base = """

"""


def ensure_nparr(x, dtype=None):
    """确保数据是np.ndarray结构，如果不是则做一个转换

    如果x已经是np.ndarray，尽量减小数据的拷贝，提高效率

    :param dtype: 如果指明了数据类型，表示要做转换，则必然会进行拷贝

    """
    return x if (dtype is None and isinstance(x, np.ndarray)) else np.array(x, dtype=dtype)


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
        return ensure_nparr(coords, dtype=dtype).reshape(-1).tolist()
    elif isinstance(coords, np.ndarray):
        return ensure_nparr(coords, dtype=dtype).reshape(-1)
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
        return ensure_nparr(coords, dtype=dtype).reshape((-1, m)).tolist()
    elif isinstance(coords, np.ndarray):
        return ensure_nparr(coords, dtype=dtype).reshape((-1, m))
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


def rect2polygon(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def reset_arr_struct(src, target):
    """ 参考 target 的数据结构，重设src的结构

    目前支持的数据结构
        PIL.Image.Image，涉及到图像格式转换的，为了与opencv兼容，一律以BGR为准，除了Image自身默认用RGB
        np.ndarray
        list
    对后两者，还会进一步判断维度信息 （目前维度信息还考虑比较简单，如果超过2维等场合，可能会有些未知、意外效果）
        1d，例如[x1 ,y1, x2, y2]
        2d，例如[[x1, y1], [x2, y2]]

    本库默认策略，图片是存np.ndarray，点集是存成np.ndarray 2d结构：[[x1, y1], [x2, y2]]
    如果希望一些函数接口功能，能输入什么类型，就返回原来什么类型，就需要这个函数来实现格式判断、转换

    本函数实现上，还是为了效率考虑，分支比较多，能不进行中间类型转换的，尽量去避免了，所以代码比较冗长
    否则直接用np结构中转，实现代码还是比较简洁的

    :param src: 原数据
    :param target: 正常是指定一个type类型，
        但是也可以输入一个【实例对象】作为参考，可以分析到更多细节信息
    :return: 重置结构后的src数据

    >>> reset_arr_struct([1, 2, 3, 4], [[1, 1], [2, 2]])
    [[1, 2], [3, 4]]
    >>> reset_arr_struct([1, 2, 3, 4], np.array([[1, 1], [2, 2]]))
    array([[1, 2],
           [3, 4]])
    >>> reset_arr_struct([1, 2, 3, 4], np.ndarray)
    array([1, 2, 3, 4])
    >>> reset_arr_struct(np.array([[1, 2], [3, 4]]), [10, 20, 30, 40])
    [1, 2, 3, 4]
    >>> reset_arr_struct(np.array([]), [10, 20, 30, 40])
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
    # 1 判断输入数据原始的类型，和目标类型
    type1 = type(src)
    if isinstance(target, type):
        type2 = target
        target = None
    else:
        type2 = type(target)

    # 2 一些辅助功能
    def np2np(a, b):
        """np.ndarray 维度细节信息的统一

        很多转换，底层都会变成一个np和np矩阵的对比问题
        """
        a = ensure_nparr(a)
        if b is None: return a
        # b 只需要取出第1个维度转np.ndarray就行，只是作为一个维度参考，不需要全解析
        if not isinstance(b, np.ndarray): b = np.array(b[:1])
        if (a.ndim == b.ndim) or (b.ndim > 2):
            return a
        if b.ndim == 1:
            return a.reshape(-1)
        elif b.ndim == 2:
            return a.reshape((-1, b.shape[1]))
        else:
            raise ValueError

    # 3 根据不同的目标数据类型，进行格式转换
    if type2 == np.ndarray:
        # 3.1 需要转成 np.ndarray
        return np2np(src, target)
    elif type2 in (list, tuple):
        # 3.2 需要转成lis嵌套list
        if type1 == PIL.Image.Image:
            src = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR)
        return np2np(src, target).tolist()
    elif isinstance(target, PIL.Image.Image):
        # 3.3 需要转成 Image 结构
        if type1 in (list, tuple):
            src = np.ndarray(src)
        if isinstance(src, np.ndarray):
            src = PIL.Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)) if src.size else None
        return src
    else:
        # TODO 可以增加Polygon的类型判断、转换；暂时还没有这需求，所以就没做
        raise TypeError(f'未知类型 {type2}')


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
    [[2, 1], [22, 11]]
    >>> warp_points([[1, 2], [11, 22]], [[0, 1, 0], [1, 0, 0]])  # 输入2*3的变换矩阵也可以
    [[2, 1], [22, 11]]
    >>> warp_points([1, 2, 11, 22], warp_mat)  # 也可以用一维的结构来输入点集
    [2, 1, 22, 11]
    >>> warp_points([1, 2, 11, 22, 111, 222], warp_mat)  # 点的数量任意，返回的结构同输入的结构形式
    [2, 1, 22, 11, 222, 111]
    >>> warp_points(np.array([1, 2, 11, 22, 111, 222]), warp_mat)  # 也可以用np.ndarray等结构
    array([  2,   1,  22,  11, 222, 111])
    >>> warp_points([1, 2, 11, 22], warp_mat, reserve_struct=False)  # 也可以用一维的结构来输入点集
    array([[ 2,  1],
           [22, 11]])
    """
    pts1 = ensure_nparr(pts).reshape(-1, 2).T
    pts1 = np.concatenate([pts1, [[1] * pts1.shape[1]]], axis=0)
    pts2 = np.dot(warp_mat[:2], pts1)
    pts2 = pts2.T
    if reserve_struct:
        pts2 = reset_arr_struct(pts2, pts)
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
        img = reset_arr_struct(img, np.ndarray)

    # 1 得到3*3的变换矩阵
    warp_mat = ensure_nparr(warp_mat)
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
    if reserve_struct:
        dst = reset_arr_struct(dst, img0)

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
    [[0, 0], [532, 0], [532, 549], [0, 549]]
    >>> warp_quad_pts([89, 424, 931, 424, 399, 290, 621, 290])
    [0, 0, 532, 0, 532, 549, 0, 549]
    """
    w, h = quad_warp_wh(pts, method)
    return reset_arr_struct(rect2polygon(0, 0, w, h), pts)


def get_sub_image(src_image, pts, warp_quad=False, reserve_struct=True):
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
    src_img = reset_arr_struct(src_image, np.ndarray)
    pts = coords2d(pts)
    if not warp_quad or len(pts) != 4:
        x1, y1, x2, y2 = rect_bounds1d(pts)
        dst = src_img[y1:y2, x1:x2]  # 这里越界不会报错，只是越界的那个维度shape为0
    else:
        w, h = quad_warp_wh(pts, method=warp_quad)
        warp_mat = get_warp_mat(pts, rect2polygon(0, 0, w, h))
        dst = warp_image(src_img, warp_mat, (w, h))
    if reserve_struct:
        dst = reset_arr_struct(dst, src_image)
    return dst


____opencv = """
对opencv相关功能的一些优化、 封装
"""


def imread(path, flags=1):
    """ opencv 源生的 imread不支持中文路径，所以要用PIL先读取，然后再转np.ndarray

    :param flags:
        0，转成 GRAY
        1，转成 BGR
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
    dst = np.array(src)  # 拷贝一张图来修改
    if color is None:
        if src.ndim == 3:
            color = (0, 0, 255)  # TODO 可以根据背景色智能推导画线用的颜色，目前是固定红色
        elif src.ndim == 2:
            color = [255]  # 灰度图，默认先填白色
    if not isinstance(lines, np.ndarray): lines = np.array(lines)
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = line
        cv2.line(dst, (x1, y1), (x2, y2), color, thickness, line_type, shift)
    return dst


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
