#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/11/15 10:09

from pyxllib.basic import is_file, Path
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


____opencv = """
对opencv相关功能的一些优化、 封装
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


____process = """

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


class ImgProcesser(ABC):
    """ 无论是np图片，还是pil图片，共有的一些功能、处理逻辑

     写有 NotImplementedError 的是np和pil机制有区别的底层功能，必须要实现的接口
     """
    __slots__ = ('img',)

    @classmethod
    def read(cls, file, flag=1, **kwargs):
        """
        :param file: 支持非文件路径参数，会做类型转换
            因为这个接口的灵活性，要判断file参数类型等，速度会慢一点点
            如果需要效率，可以显式使用imread、Image.open等明确操作类型
        :param flag:
            -1，按照图像原样读取，保留Alpha通道（第4通道）
            0，将图像转成单通道灰度图像后读取
            1，将图像转换成3通道BGR彩色图像
        """
        raise NotImplementedError

    def cvt_channel(self, flags):
        """ 确保图片目前是flags指示的通道情况 """
        raise NotImplementedError

    def write(self, path, if_exists='replace', **kwargs):
        raise NotImplementedError

    @property
    def size(self):
        """ 图片尺寸，统一返回(height, width)，不含通道 """
        raise NotImplementedError

    @property
    def n_channels(self):
        """ 通道数 """
        raise NotImplementedError

    def resize(self, size, interpolation=None, **kwargs):
        """
        :param size: (h, w)
        """
        raise NotImplementedError

    def reduce_by_area(self, area):
        """ 根据面积上限缩小图片

        即图片面积超过area时，按照等比例缩小到面积为area的图片
        """
        h, w = self.size
        s = h * w
        if s > area:
            r = (area / s) ** 0.5
            size = int(r * h), int(r * w)
            img = self.resize(size)
        else:
            img = self
        return img


class CvImg(ImgProcesser):
    def __init__(self, img):
        """ 注意源生的初始化不做任何类型判断，直接赋值 """
        self.img = img

    _show_win_num = 0

    @classmethod
    def read(cls, file, flags=1, **kwargs):
        if is_numpy_image(file):
            img = file
        elif Path(file).is_file():
            # https://www.yuque.com/xlpr/pyxllib/imread
            img = cv2.imdecode(np.fromfile(str(file), dtype=np.uint8), flags)
        elif is_pil_image(file):
            img = cls.pil2np(file)
        else:
            raise TypeError(f'类型错误：{type(file)}')
        return CvImg(img).cvt_channel(flags)

    @classmethod
    def pil2np(cls, img):
        """ pil图片转np图片 """
        x = img
        y = np_array(x)
        y = PIL.Image.fromarray(cv2.cvtColor(y, cv2.COLOR_BGR2RGB)) if y.size else None
        return y

    def cvt_channel(self, flags):
        img = self.img
        n_c = self.n_channels
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
        return CvImg(img)

    def write(self, path, if_exists='replace', **kwargs):
        img = self.img
        if not isinstance(path, Path):
            path = Path(path)
        data = cv2.imencode(ext=path.suffix, img=img)[1]
        return path.write(data.tobytes(), if_exists=if_exists)

    @property
    def size(self):
        return self.img.shape[:2]

    @property
    def n_channels(self):
        """ 通道数 """
        img = self.img
        if img.ndim == 3:
            return img.shape[2]
        else:
            return 1

    def resize(self, size, interpolation=cv2.INTER_CUBIC, **kwargs):
        """
        >>> im = CvImg(np.zeros([100, 200], dtype='uint8'))
        >>> im.size
        (100, 200)
        >>> im2 = im.reduce_by_area(50*50)
        >>> im2.size
        (35, 70)
        """
        img = cv2.resize(self.img, size[::-1], interpolation, **kwargs)
        return CvImg(img)

    def show(self, winname=None, flags=1):
        """ 展示窗口

        :param winname: 未输入时，则按test1、test2依次生成窗口
        :param flags:
            cv2.WINDOW_NORMAL，0，输入2等偶数值好像也等价于输入0
            cv2.WINDOW_AUTOSIZE，1，输入3等奇数值好像等价于1
            cv2.WINDOW_OPENGL，4096
        :return:
        """
        if winname is None:
            n = CvImg._show_win_num + 1
            winname = f'test{n}'
        cv2.namedWindow(winname, flags)
        cv2.imshow(winname, self.img)


class PilImg(ImgProcesser):
    def __init__(self, img):
        """ 注意源生的初始化不做任何类型判断，直接赋值 """
        self.img = img

    @classmethod
    def read(cls, file, flags=1, **kwargs):
        if is_pil_image(file):
            img = file
        elif is_numpy_image(file):
            img = cls.np2pil(file)
        elif Path(file).is_file():
            img = Image.open(file, **kwargs)
        else:
            raise TypeError(f'类型错误：{type(file)}')
        return PilImg(img).cvt_channel(flags)

    @classmethod
    def np2pil(cls, pic, mode=None):
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

    def cvt_channel(self, flags):
        img = self.img
        n_c = self.n_channels
        if flags == 0 and n_c > 1:
            img = img.convert('L')
        elif flags == 1 and n_c != 3:
            img = img.convert('RGB')
        return PilImg(img)

    def write(self, path, if_exists='replace', **kwargs):
        p = Path(path)
        if p.preprocess(if_exists):
            p.ensure_dir('file')
            self.img.save(str(p), **kwargs)

    def resize(self, size, interpolation=Image.BILINEAR, **kwargs):
        """
        >>> im = PilImg.read(np.zeros([100, 200], dtype='uint8'), 0)
        >>> im.size
        (100, 200)
        >>> im2 = im.reduce_by_area(50*50)
        >>> im2.size
        (35, 70)
        """
        # 注意pil图像尺寸接口都是[w,h]，跟标准的[h,w]相反
        return PilImg(self.img.resize(size[::-1], interpolation))

    @property
    def size(self):
        w, h = self.img.size
        return h, w

    @property
    def n_channels(self):
        """ 通道数 """
        return len(self.img.getbands())

    def show(self, title=None, command=None):
        return self.img.show(title, command)

    def random_direction(self):
        """ 假设原图片是未旋转的状态0

        顺时针转90度是label=1，顺时针转180度是label2 ...
        """
        img = self.img
        label = np.random.randint(4)
        if label == 1:
            # PIL的旋转角度，是指逆时针角度；但是我这里编号是顺时针
            img = img.transpose(PIL.Image.ROTATE_270)
        elif label == 2:
            img = img.transpose(PIL.Image.ROTATE_180)
        elif label == 3:
            img = img.transpose(PIL.Image.ROTATE_90)
        return img, label


____get_sub_image = """

TODO 这里很多功能，都要抽时间整理进CvImg、PilImg

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
    img = CvImg.read(img).img

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


def get_sub_image(src_image, pts, warp_quad=False):
    """ 从src_image取一个子图

    :param src_image: 原图
        可以是图片路径、np.ndarray、PIL.Image对象
        TODO 目前只支持np.ndarray、pil图片输入，返回统一是np.ndarray
    :param pts: 子图位置信息
        只有两个点，认为是矩形的两个对角点
        只有四个点，认为是任意四边形
        同理，其他点数量，默认为
    :param warp_quad: 变形的四边形
        默认是截图pts的外接四边形区域，使用该参数
            且当pts为四个点时，是否强行扭转为矩形
        一般写 'average'，也可以写'max'、'min'，详见 quad_warp_wh()
    :return: 子图
        文件、np.ndarray --> np.ndarray
        PIL.Image --> PIL.Image
    """
    src_img = CvImg.read(src_image)
    pts = coords2d(pts)
    if not warp_quad or len(pts) != 4:
        x1, y1, x2, y2 = rect_bounds1d(pts)
        dst = src_img[y1:y2, x1:x2]  # 这里越界不会报错，只是越界的那个维度shape为0
    else:
        w, h = quad_warp_wh(pts, method=warp_quad)
        warp_mat = get_warp_mat(pts, rect2polygon([0, 0, w, h]))
        dst = warp_image(src_img, warp_mat, (w, h))
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
