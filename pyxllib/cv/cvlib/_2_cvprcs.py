#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/11/15 10:09

from pyxllib.basic import File
from pyxllib.cv.cvlib._1_geo import *

import cv2
import PIL.Image
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None
from collections.abc import Sequence, Iterable
from abc import ABC

____functional = """
torchvision.transforms搬过来的功能

这个库太大了，底层又依赖tensor，
"""


def is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def is_numpy(img):
    return isinstance(img, np.ndarray)


def is_numpy_image(img):
    return is_numpy(img) and img.ndim in {2, 3}


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

    npimg = pic

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError("Only modes {} are supported for 2D inputs".format(permitted_2_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


____other_func = """
"""


def pil2cv(img):
    """ pil图片转np图片 """
    x = img
    y = np_array(x)
    y = PIL.Image.fromarray(cv2.cvtColor(y, cv2.COLOR_BGR2RGB)) if y.size else None
    return y


____opencv = """
opencv-python文档： https://opencv-python-tutroals.readthedocs.io/en/latest/
pillow文档： https://pillow.readthedocs.io/en/stable/reference/


TODO 201115周日20:18
1、很多功能是新写的，可能有bug，多用多优化~~
2、画图功能
    ① CvPlot有待整合进CvImg
    ② PilImg增加对应的画图功能
    ③ 整合、统一接口形式、名称
3、旧的仿射变换等功能
    ① 需要整合进CvImg
    ② PilImg需要对应的实现
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


class CvPrcs:
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
            img = file
        elif File(file).is_file():
            # https://www.yuque.com/xlpr/pyxllib/imread
            img = cv2.imdecode(np.fromfile(str(file), dtype=np.uint8), flags)
        elif is_pil_image(file):
            img = pil2cv(file)
        else:
            raise TypeError(f'类型错误：{type(file)} {file}')
        return cls.cvt_channel(img, flags)

    @classmethod
    def cvt_channel(cls, img, flags):
        """ 确保图片目前是flags指示的通道情况 """
        n_c = cls.n_channels(img)
        if flags == 0 and n_c > 1:
            if n_c == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif n_c == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif flags == 1 and n_c != 3:
            if n_c == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif n_c == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    @classmethod
    def write(cls, img, file, if_exists='replace', **kwargs):
        if not isinstance(file, File):
            file = File(file)
        data = cv2.imencode(ext=file.suffix, img=img)[1]
        return file.write(data.tobytes(), if_exists=if_exists)

    @classmethod
    def size(cls, img):
        """ 图片尺寸，统一返回(height, width)，不含通道 """
        return img.shape[:2]

    @classmethod
    def n_channels(cls, img):
        """ 通道数 """
        if img.ndim == 3:
            return img.shape[2]
        else:
            return 1

    @classmethod
    def resize(cls, img, size, interpolation=cv2.INTER_CUBIC, **kwargs):
        """
        :param size: (h, w)
        """
        return cv2.resize(img, size[::-1], interpolation, **kwargs)

    @classmethod
    def show(cls, img, winname=None, flags=0):
        """ 展示窗口

        :param winname: 未输入时，则按test1、test2依次生成窗口
        :param flags:
            cv2.WINDOW_NORMAL，0，输入2等偶数值好像也等价于输入0
            cv2.WINDOW_AUTOSIZE，1，输入3等奇数值好像等价于1
            cv2.WINDOW_OPENGL，4096
        :return:
        """
        if winname is None:
            n = cls._show_win_num + 1
            winname = f'test{n}'
        cv2.namedWindow(winname, flags)
        cv2.imshow(winname, img)

    @classmethod
    def reduce_by_area(cls, img, area):
        """ 根据面积上限缩小图片

        即图片面积超过area时，按照等比例缩小到面积为area的图片
        """
        h, w = cls.size(img)
        s = h * w
        if s > area:
            r = (area / s) ** 0.5
            size = int(r * h), int(r * w)
            img = cls.resize(img, size)
        else:
            img = cls
        return img
