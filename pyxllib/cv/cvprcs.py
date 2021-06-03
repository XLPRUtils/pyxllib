#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/11/15 10:09

from pyxllib.file import File
from pyxllib.algo.geo import *

import cv2
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None

__functional = """
torchvision.transforms搬过来的功能

这个库太大了，底层又依赖tensor，
"""


def is_pil_image(im):
    if accimage is not None:
        return isinstance(im, (Image.Image, accimage.Image))
    else:
        return isinstance(im, Image.Image)


def is_numpy(im):
    return isinstance(im, np.ndarray)


def is_numpy_image(im):
    return is_numpy(im) and im.ndim in {2, 3}


def cv2pil(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image. （删除了tensor的转换功能）

    See :class:`~torchvision.transforms.ToPILImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if pic.ndim not in {2, 3}:
        raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))
    if pic.ndim == 2:
        # if 2D image, add channel dimension (HWC)
        pic = np.expand_dims(pic, 2)

    npim = pic

    if npim.shape[2] == 1:
        expected_mode = None
        npim = npim[:, :, 0]
        if npim.dtype == np.uint8:
            expected_mode = 'L'
        elif npim.dtype == np.int16:
            expected_mode = 'I;16'
        elif npim.dtype == np.int32:
            expected_mode = 'I'
        elif npim.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npim.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError("Only modes {} are supported for 2D inputs".format(permitted_2_channel_modes))

        if mode is None and npim.dtype == np.uint8:
            mode = 'LA'

    elif npim.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npim.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npim.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npim.dtype))

    return Image.fromarray(npim, mode=mode)


__other_func = """
"""


def pil2cv(im):
    """ pil图片转np图片 """
    x = im
    y = np_array(x)
    y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB) if y.size else None
    return y


__opencv = """
opencv-python文档： https://opencv-python-tutroals.readthedocs.io/en/latest/
pillow文档： https://pillow.readthedocs.io/en/stable/reference/


TODO 201115周日20:18
1、很多功能是新写的，可能有bug，多用多优化~~
2、画图功能
    ① CvPlot有待整合进Cvim
    ② Pilim增加对应的画图功能
    ③ 整合、统一接口形式、名称
3、旧的仿射变换等功能
    ① 需要整合进Cvim
    ② Pilim需要对应的实现
    ③ 整合，统一接口
"""


class CvPlot:
    @classmethod
    def get_plot_color(cls, src):
        """ 获得比较适合的作画颜色

        TODO 可以根据背景色智能推导画线用的颜色，目前是固定红色
        """
        if src.ndim == 3:
            return 0, 0, 255
        elif src.ndim == 2:
            return 255  # 灰度图，默认先填白色

    @classmethod
    def get_plot_args(cls, src, color=None):
        # 1 作图颜色
        if not color:
            color = cls.get_plot_color(src)

        # 2 画布
        if len(color) >= 3 and src.ndim <= 2:
            dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        else:
            dst = np.array(src)

        return dst, color

    @classmethod
    def lines(cls, src, lines, color=None, thickness=1, line_type=cv2.LINE_AA, shift=None):
        """ 在src图像上画系列线段
        """
        # 1 判断 lines 参数内容
        lines = np_array(lines).reshape(-1, 4)
        if not lines.size:
            return src

        # 2 参数
        dst, color = cls.get_plot_args(src, color)

        # 3 画线
        if lines.any():
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(dst, (x1, y1), (x2, y2), color, thickness, line_type)
        return dst

    @classmethod
    def circles(cls, src, circles, color=None, thickness=1, center=False):
        """ 在图片上画圆形

        :param src: 要作画的图
        :param circles: 要画的圆形参数 (x, y, 半径 r)
        :param color: 画笔颜色
        :param center: 是否画出圆心
        """
        # 1 圆 参数
        circles = np_array(circles, dtype=int).reshape(-1, 3)
        if not circles.size:
            return src

        # 2 参数
        dst, color = cls.get_plot_args(src, color)

        # 3 作画
        for x in circles:
            cv2.circle(dst, (x[0], x[1]), x[2], color, thickness)
            if center:
                cv2.circle(dst, (x[0], x[1]), 2, color, thickness)

        return dst


class _CvPrcsBase:
    """ opencv和pil中，虽然实现方式不同，但彼此都有自己一套实现的功能 """
    _show_win_num = 0

    @classmethod
    def read(cls, file, flags=1, **kwargs):
        """
        :param file: 支持非文件路径参数，会做类型转换
            因为这个接口的灵活性，要判断file参数类型等，速度会慢一点点
            如果需要效率，可以显式使用imread、Image.open等明确操作类型
        :param flags:
            -1，按照图像原样读取，保留Alpha通道（第4通道）
            0，将图像转成单通道灰度图像后读取
            1，将图像转换成3通道BGR彩色图像
        """
        if is_numpy_image(file):
            im = file
        elif File.safe_init(file):
            # https://www.yuque.com/xlpr/pyxllib/imread
            im = cv2.imdecode(np.fromfile(str(file), dtype=np.uint8), flags)
        elif is_pil_image(file):
            im = pil2cv(file)
        else:
            raise TypeError(f'类型错误或文件不存在：{type(file)} {file}')
        return cls.cvt_channel(im, flags)

    @classmethod
    def cvt_channel(cls, im, flags=None):
        """ 确保图片目前是flags指示的通道情况 """
        if flags is None: return im
        n_c = cls.n_channels(im)
        if flags == 0 and n_c > 1:
            if n_c == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            elif n_c == 4:
                im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
        elif flags == 1 and n_c != 3:
            if n_c == 1:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif n_c == 4:
                im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        return im

    @classmethod
    def write(cls, im, file, if_exists='delete', **kwargs):
        if not isinstance(file, File):
            file = File(file)
        data = cv2.imencode(ext=file.suffix, img=im)[1]
        return file.write(data.tobytes(), if_exists=if_exists)

    @classmethod
    def size(cls, im):
        """ 图片尺寸，统一返回(height, width)，不含通道 """
        return im.shape[:2]

    @classmethod
    def resize(cls, im, size, **kwargs):
        """
        :param size: (h, w)
        :param kwargs:
            interpolation=cv2.INTER_CUBIC
        """
        # if 'interpolation' not in kwargs:
        #     kwargs['interpolation'] = cv2.INTER_CUBIC
        return cv2.resize(im, size[::-1], **kwargs)

    @classmethod
    def n_channels(cls, im):
        """ 通道数 """
        if im.ndim == 3:
            return im.shape[2]
        else:
            return 1

    @classmethod
    def show(cls, im, winname=None, flags=1):
        """ 展示窗口

        :param winname: 未输入时，则按test1、test2依次生成窗口
        :param flags:
            cv2.WINDOW_NORMAL，0，输入2等偶数值好像也等价于输入0，可以自动拉伸窗口大小
            cv2.WINDOW_AUTOSIZE，1，输入3等奇数值好像等价于1
            cv2.WINDOW_OPENGL，4096
        :return:
        """
        if winname is None:
            n = cls._show_win_num + 1
            winname = f'test{n}'
        cv2.namedWindow(winname, flags)
        cv2.imshow(winname, im)


class CvPrcsBase(_CvPrcsBase):
    """ 同时适用于opencv和pil的算法 """

    @classmethod
    def reduce_area(cls, im, area):
        """ 根据面积上限缩小图片

        即图片面积超过area时，按照等比例缩小到面积为area的图片
        """
        h, w = cls.size(im)
        s = h * w
        if s > area:
            r = (area / s) ** 0.5
            size = int(r * h), int(r * w)
            im = cls.resize(im, size)
        else:
            im = cls
        return im


class CvPrcs(CvPrcsBase):
    """ opencv当前独有的功能 """

    @classmethod
    def warp(cls, im, warp_mat, dsize=None, *, view_rate=False, max_zoom=1, reserve_struct=False):
        """ 对图像进行透视变换

        :param im: np.ndarray的图像数据
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
        :param reserve_struct: 是否保留原来im的数据类型返回，默认True
            关掉该功能可以提高性能，此时返回结果统一为 np 矩阵
        :return: 见 reserve_struct
        """
        from math import sqrt

        # 1 得到3*3的变换矩阵
        warp_mat = np_array(warp_mat)
        if warp_mat.shape[0] == 2:
            warp_mat = np.concatenate([warp_mat, [[0, 0, 1]]], axis=0)

        # 2 view_rate，视野比例改变导致的变换矩阵规则变化
        if view_rate:
            # 2.1 视野变化后的四个角点
            h, w = im.shape[:2]
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
            dsize = (im.shape[1], im.shape[0])
        dst = cv2.warpPerspective(im, warp_mat, dsize)

        # 4 返回值
        return dst

    @classmethod
    def bg_color(cls, src_im, edge_size=5, binary_img=None):
        """ 智能判断图片背景色

        对全图二值化后，考虑最外一层宽度为edge_size的环中，0、1分布最多的作为背景色
            然后取全部背景色的平均值返回

        :param src_im: 支持黑白图、彩图
        :param edge_size: 边缘宽度，宽度越高一般越准确，但也越耗性能
        :param binary_img: 运算中需要用二值图，如果外部已经计算了，可以直接传入进来，避免重复运算
        :return: color

        TODO 可以写个获得前景色，道理类似，只是最后在图片中心去取平均值
        """
        from itertools import chain

        # 1 获得二值图，区分前背景
        if binary_img is None:
            gray_img = cv2.cvtColor(src_im, cv2.COLOR_BGR2GRAY) if src_im.ndim == 3 else src_im
            _, binary_img = cv2.threshold(gray_img, np.mean(gray_img), 255, cv2.THRESH_BINARY)

        # 2 分别存储点集
        n, m = src_im.shape[:2]
        colors0, colors1 = [], []
        for i in range(n):
            if i < edge_size or i >= n - edge_size:
                js = range(m)
            else:
                js = chain(range(edge_size), range(m - edge_size, m))
            for j in js:
                if binary_img[i, j]:
                    colors1.append(src_im[i, j])
                else:
                    colors0.append(src_im[i, j])

        # 3 计算平均像素
        # 以数量多的作为背景像素
        colors = colors0 if len(colors0) > len(colors1) else colors1
        return np.mean(np.array(colors), axis=0, dtype='int').tolist()

    @classmethod
    def pad(cls, im, pad_size, constant_values=0, mode='constant', **kwargs):
        r""" 拓宽图片上下左右尺寸

        基于np.pad，定制、简化了针对图片类型数据的操作
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html

        :pad_size: 输入单个整数值，对四周pad相同尺寸，或者输入四个值的list，表示对上、下、左、右的扩展尺寸

        >>> a = np.ones([2, 2])
        >>> b = np.ones([2, 2, 3])
        >>> CvPrcs.pad(a, 1).shape  # 上下左右各填充1行/列
        (4, 4)
        >>> CvPrcs.pad(b, 1).shape
        (4, 4, 3)
        >>> CvPrcs.pad(a, [1, 2, 3, 0]).shape  # 上填充1，下填充2，左填充3，右不填充
        (5, 5)
        >>> CvPrcs.pad(b, [1, 2, 3, 0]).shape
        (5, 5, 3)
        >>> CvPrcs.pad(a, [1, 2, 3, 0])
        array([[0., 0., 0., 0., 0.],
               [0., 0., 0., 1., 1.],
               [0., 0., 0., 1., 1.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])
        >>> CvPrcs.pad(a, [1, 2, 3, 0], 255)
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

    @classmethod
    def _get_subrect_image(cls, src_im, pts, fill=0):
        """
        :return:
            dst_img 按外接四边形截取的子图
            new_pts 新的变换后的点坐标
        """
        # 1 计算需要pad的宽度
        x1, y1, x2, y2 = rect_bounds1d(pts)
        h, w = src_im.shape[:2]
        pad = [-y1, y2 - h, -x1, x2 - w]  # 各个维度要补充的宽度
        pad = [max(0, v) for v in pad]  # 负数宽度不用补充，改为0

        # 2 pad并定位rect局部图
        tmp_img = cls.pad(src_im, pad, fill) if max(pad) > 0 else src_im
        dst_img = tmp_img[y1 + pad[0]:y2, x1 + pad[2]:x2]  # 这里越界不会报错，只是越界的那个维度shape为0
        new_pts = [(pt[0] - x1, pt[1] - y1) for pt in pts]
        return dst_img, new_pts

    @classmethod
    def get_sub(cls, src_im, pts, *, fill=0, warp_quad=False):
        """ 从src_im取一个子图

        :param src_im: 原图
            可以是图片路径、np.ndarray、PIL.Image对象
            TODO 目前只支持np.ndarray、pil图片输入，返回统一是np.ndarray
        :param pts: 子图位置信息
            只有两个点，认为是矩形的两个对角点
            只有四个点，认为是任意四边形
            同理，其他点数量，默认为
        :param fill: 支持pts越界选取，此时可以设置fill自动填充的颜色值
            TODO fill填充一个rgb颜色的时候应该会不兼容报错，还要想办法优化
        :param warp_quad: 变形的四边形
            默认是截图pts的外接四边形区域，使用该参数
                且当pts为四个点时，是否强行扭转为矩形
            一般写 'average'，也可以写'max'、'min'，详见 quad_warp_wh()
        :return: 子图
            文件、np.ndarray --> np.ndarray
            PIL.Image --> PIL.Image
        """
        dst, pts = cls._get_subrect_image(cls.read(src_im), coords2d(pts), fill)
        if len(pts) == 4 and warp_quad:
            w, h = quad_warp_wh(pts, method=warp_quad)
            warp_mat = get_warp_mat(pts, rect2polygon([0, 0, w, h]))
            dst = cls.warp(dst, warp_mat, (w, h))
        return dst
