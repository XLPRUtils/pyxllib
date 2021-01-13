#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/11/17 15:21

from pyxllib.cv._3_pilprcs import *

____XxImg = """
对CvPrcs、PilPrcs的类层级接口封装
"""


class CvImg:
    prcs = CvPrcs
    __slots__ = ('img',)

    def __init__(self, img, flags=None, **kwargs):
        if isinstance(img, type(self)):
            img = img.img
        else:
            img = self.prcs.read(img, flags, **kwargs)
        self.img = img

    def cvt_channel(self, flags):
        _t = type(self)
        return _t(self.prcs.cvt_channel(self.img, flags))

    def write(self, path, if_exists='delete', **kwargs):
        return self.prcs.write(self.img, path, if_exists, **kwargs)

    @property
    def size(self):
        return self.prcs.size(self.img)

    @property
    def n_channels(self):
        return self.prcs.n_channels(self.img)

    def resize(self, size, **kwargs):
        _t = type(self)
        return _t(self.prcs.resize(self.img, size, **kwargs))

    def show(self, winname=None, flags=0):
        return self.prcs.show(self.img, winname, flags)

    def reduce_by_area(self, area):
        _t = type(self)
        return _t(self.prcs.reduce_by_area(self.img, area))


class PilImg(CvImg):
    """
    注意这样继承实现虽然简单，但是如果是CvPrcs有，但PilPrcs没有的功能，运行是会报错的
    """
    prcs = PilPrcs

    # 一些比较特别的接口微调下即可
    def random_direction(self):
        """ 这个功能可以考虑关掉，否则PilImg返回是普通img也有点奇怪~~ """
        # _t = type(self)
        img, label = self.prcs.random_direction(self.img)
        return img, label


____alias = """
对CvPrcs中一些常用功能的名称简化
"""

imread = CvPrcs.read
imwrite = CvPrcs.write
imshow = CvPrcs.show

____get_sub_image = """

TODO 这里很多功能，都要抽时间整理进CvImg、PilImg

"""


def warp_image(img, warp_mat, dsize=None, *, view_rate=False, max_zoom=1, reserve_struct=False):
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
    img = imread(img)

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


def pad_image(im, pad_size, constant_values=0, mode='constant', **kwargs):
    r""" 拓宽图片上下左右尺寸

    基于np.pad，定制、简化了针对图片类型数据的操作
    https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    :pad_size: 输入单个整数值，对四周pad相同尺寸，或者输入四个值的list，表示对上、下、左、右的扩展尺寸

    >>> a = np.ones([2, 2])
    >>> b = np.ones([2, 2, 3])
    >>> pad_image(a, 1).shape  # 上下左右各填充1行/列
    (4, 4)
    >>> pad_image(b, 1).shape
    (4, 4, 3)
    >>> pad_image(a, [1, 2, 3, 0]).shape  # 上填充1，下填充2，左填充3，右不填充
    (5, 5)
    >>> pad_image(b, [1, 2, 3, 0]).shape
    (5, 5, 3)
    >>> pad_image(a, [1, 2, 3, 0])
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 1.],
           [0., 0., 0., 1., 1.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])
    >>> pad_image(a, [1, 2, 3, 0], 255)
    array([[255., 255., 255., 255., 255.],
           [255., 255., 255.,   1.,   1.],
           [255., 255., 255.,   1.,   1.],
           [255., 255., 255., 255., 255.],
           [255., 255., 255., 255., 255.]])
    """
    # 0 参数检查
    if im.ndim < 2 or im.ndim > 3:
        raise ValueError

    if isinstance(pad_size, int):
        ltrb = [pad_size] * 4
    else:
        ltrb = pad_size

    # 1 pad_size转成np.pad的格式
    pad_width = [(ltrb[0], ltrb[1]), (ltrb[2], ltrb[3])]
    if im.ndim == 3:
        pad_width.append((0, 0))

    dst = np.pad(im, pad_width, mode, constant_values=constant_values, **kwargs)

    return dst


def _get_subrect_image(src_img, pts, fill=0):
    """
    :return:
        dst_img 按外接四边形截取的子图
        new_pts 新的变换后的点坐标
    """
    # 1 计算需要pad的宽度
    x1, y1, x2, y2 = rect_bounds1d(pts)
    h, w = src_img.shape[:2]
    pad = [-y1, y2 - h, -x1, x2 - w]  # 各个维度要补充的宽度
    pad = [max(0, v) for v in pad]  # 负数宽度不用补充，改为0

    # 2 pad并定位rect局部图
    tmp_img = pad_image(src_img, pad, fill) if max(pad) > 0 else src_img
    dst_img = tmp_img[y1 + pad[0]:y2, x1 + pad[2]:x2]  # 这里越界不会报错，只是越界的那个维度shape为0
    new_pts = [(pt[0] - x1, pt[1] - y1) for pt in pts]
    return dst_img, new_pts


def get_sub_image(src_image, pts, *, fill=0, warp_quad=False):
    """ 从src_image取一个子图

    :param src_image: 原图
        可以是图片路径、np.ndarray、PIL.Image对象
        TODO 目前只支持np.ndarray、pil图片输入，返回统一是np.ndarray
    :param pts: 子图位置信息
        只有两个点，认为是矩形的两个对角点
        只有四个点，认为是任意四边形
        同理，其他点数量，默认为
    :param fill: 支持pts越界选取，此时可以设置fill自动填充的颜色值
    :param warp_quad: 变形的四边形
        默认是截图pts的外接四边形区域，使用该参数
            且当pts为四个点时，是否强行扭转为矩形
        一般写 'average'，也可以写'max'、'min'，详见 quad_warp_wh()
    :return: 子图
        文件、np.ndarray --> np.ndarray
        PIL.Image --> PIL.Image
    """
    src_img = imread(src_image)
    pts = coords2d(pts)
    dst, pts = _get_subrect_image(src_img, pts)
    if len(pts) == 4 and warp_quad:
        w, h = quad_warp_wh(pts, method=warp_quad)
        warp_mat = get_warp_mat(pts, rect2polygon([0, 0, w, h]))
        dst = warp_image(dst, warp_mat, (w, h))
    return dst


____other = """
"""


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


def debug_images(dir_, func, *, save=None, show=False):
    """
    :param dir_: 选中的文件清单
    :param func: 对每张图片执行的功能，函数应该只有一个图片路径参数  new_img = func(img)
        当韩式有个参数时，可以用lambda函数技巧： lambda im: func(im, arg1=..., arg2=...)
    :param save: 如果输入一个目录，会将debug结果图存储到对应的目录里
    :param show: 如果该参数为True，则每处理一张会imshow显示处理效果
        此时弹出的窗口里，每按任意键则显示下一张，按ESC退出
    :return:

    TODO 显示原图、处理后图的对比效果
    TODO 支持同时显示多张图处理效果
    """
    if save:
        save = File(save)

    for f in dir_.subfiles():
        im1 = imread(f)
        im2 = func(im1)

        if save:
            imwrite(im2, File(save / f.name, dir_))

        if show:
            imshow(im2)
            key = cv2.waitKey()
            if key == '0x1B':  # ESC 键
                break
