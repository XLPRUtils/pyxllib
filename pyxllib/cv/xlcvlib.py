#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/25 15:57

import base64
import itertools

import cv2
import humanfriendly
import numpy as np
import PIL.Image
import requests

from pyxllib.prog.newbie import round_int, RunOnlyOnce
from pyxllib.prog.pupil import EnchantBase, EnchantCvt, is_url
from pyxllib.algo.geo import rect_bounds, warp_points, reshape_coords, quad_warp_wh, get_warp_mat, rect2polygon
from pyxllib.file.specialist import File

_show_win_num = 0


class xlcv(EnchantBase):

    @classmethod
    @RunOnlyOnce
    def enchant(cls):
        """ 把xlcv的功能嵌入cv2中

        不太推荐使用该类，可以使用CvImg类更好地解决问题。
        """
        # 虽然只绑定cv2，但其他相关的几个库的方法上，最好也不要重名
        names = cls.check_enchant_names([np.ndarray, PIL.Image, PIL.Image.Image])
        names -= {'concat'}
        cls._enchant(cv2, names, EnchantCvt.staticmethod2modulefunc)

    @staticmethod
    def __1_read():
        pass

    @staticmethod
    def read(file, flags=None, **kwargs):
        """
        :param file: 支持非文件路径参数，会做类型转换
            因为这个接口的灵活性，要判断file参数类型等，速度会慢一点点
            如果需要效率，可以显式使用imread、Image.open等明确操作类型
        :param flags:
            -1，按照图像原样读取，保留Alpha通道（第4通道）
            0，将图像转成单通道灰度图像后读取
            1，将图像转换成3通道BGR彩色图像

        220426周二14:20，注，有些图位深不是24而是48，读到的不是uint8而是uint16
            目前这个接口没做适配，需要下游再除以256后arr.astype('uint8')
        """
        from pyxllib.cv.xlpillib import xlpil

        if xlcv.is_cv2_image(file):
            im = file
        elif File.safe_init(file):
            # https://www.yuque.com/xlpr/pyxllib/imread
            # + np.frombuffer
            im = cv2.imdecode(np.fromfile(str(file), dtype=np.uint8), -1 if flags is None else flags)
        elif xlpil.is_pil_image(file):
            im = xlpil.to_cv2_image(file)
        else:
            raise TypeError(f'类型错误或文件不存在：{type(file)} {file}')
        return xlcv.cvt_channel(im, flags)

    @staticmethod
    def read_from_buffer(buffer, flags=None, *, b64decode=False):
        """ 从二进制流读取图片
        这个二进制流指，图片以png、jpg等某种格式存储为文件时，其对应的文件编码
    
        :param b64decode: 是否需要先进行base64解码
        """
        if b64decode:
            buffer = base64.b64decode(buffer)
        buffer = np.frombuffer(buffer, dtype=np.uint8)
        im = cv2.imdecode(buffer, -1 if flags is None else flags)
        return xlcv.cvt_channel(im, flags)

    @staticmethod
    def read_from_url(url, flags=None, *, b64decode=False):
        """ 从url直接获取图片到内存中
        """
        content = requests.get(url).content
        return xlcv.read_from_buffer(content, flags, b64decode=b64decode)

    @staticmethod
    def __2_attrs():
        pass

    @staticmethod
    def imsize(im):
        """ 图片尺寸，统一返回(height, width)，不含通道 """
        return im.shape[:2]

    @staticmethod
    def n_channels(im):
        """ 通道数 """
        if im.ndim == 3:
            return im.shape[2]
        else:
            return 1

    @staticmethod
    def height(im):
        """ 注意PIL.Image.Image本来就有height、width属性，所以不用自定义这两个方法 """
        return im.shape[0]

    @staticmethod
    def width(im):
        return im.shape[1]

    @staticmethod
    def __3_write__(self):
        pass

    @staticmethod
    def to_pil_image(pic, mode=None):
        """ 我也不懂torch里这个实现为啥这么复杂，先直接哪来用，没深究细节
    
        Convert a tensor or an ndarray to PIL Image. （删除了tensor的转换功能）
    
        See :class:`~torchvision.transforms.ToPILImage` for more details.
    
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
            mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
    
        .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    
        Returns:
            PIL Image: Image converted to PIL Image.
        """
        # 需要先做个通道转换。这里的cvt是比较万精油的，支持pic是灰度图、RGBA等多种场景情况。
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

        # 以下是原版实现代码
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

        return PIL.Image.fromarray(npim, mode=mode)

    @staticmethod
    def is_cv2_image(im):
        return isinstance(im, np.ndarray) and im.ndim in {2, 3}

    @staticmethod
    def cvt_channel(im, flags=None):
        """ 确保图片目前是flags指示的通道情况

        1. 通道互转功能适用于RGBA情况
        2. RGBA转RGB默认是黑底填充。如果需要白底填充，可以使用xlpil.rgba2rgb等专门的特殊处理策略。
        """
        if flags is None: return im
        n_c = xlcv.n_channels(im)
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

    @staticmethod
    def write(im, file, if_exists='replace'):
        if not isinstance(file, File):
            file = File(file)
        data = cv2.imencode(ext=file.suffix, img=im)[1]
        return file.write(data.tobytes(), if_exists=if_exists)

    @staticmethod
    def show(im):
        """ 类似Image.show，可以用计算机本地软件打开查看图片 """
        xlcv.to_pil_image(im).show()

    @staticmethod
    def to_buffer(im, ext='.jpg', *, b64encode=False):
        flag, buffer = cv2.imencode(ext, im)
        buffer = bytes(buffer)
        if b64encode:
            buffer = base64.b64encode(buffer)
        return buffer

    @staticmethod
    def to_buffer2(in_, flags=1, *, b64decode=True, b64encode=False,
                   min_length=None, max_length=None,
                   limit_b64buffer_size=None):
        """ 获取in_代表的图片的二进制数据，一般是用在网络图片传输，API调用等场景

        :param in_: 可以是本地文件，也可以是图片url地址，也可以是Image对象
            注意这个函数，输入是url，也会获取重置图片数据上传
            如果为了效率明确只传url，可以用aip.AipOcr原生的相关url函数
        :param b64decode: 如果输入是bytes类型，是否要用b64解码，默认需要
        :param b64encode: 返回的结果，是否需要b64编码，默认不需要
        :return: 返回图片文件二进制值的buffer, 缩放系数(小余1是缩小，大于1是放大)
        """
        # 1 取不同来源的数据
        # 下面应该是比较通用的一套操作，如果有特殊接口，可以另外处理，不一定要通过该接口处理图片
        if isinstance(in_, bytes):
            im = xlcv.read_from_buffer(in_, flags, b64decode=b64decode)
        elif is_url(in_):
            im = xlcv.read_from_url(in_, flags)
        else:
            im = xlcv.read(in_, flags)
        origin_height = im.shape[0]

        # 2 图片尺寸不符合要求，要缩放
        if min_length or max_length:
            im = xlcv.adjust_shape(im, min_length, max_length)

        # 3 图片文件不能过大，要调整
        if limit_b64buffer_size:
            # b64后大小固定会变4/3，所以留给原文件的大小是要缩水，只有0.75；再以防万一总不能卡得刚刚好，所以设为0.74
            im = xlcv.reduce_filesize(im, limit_b64buffer_size * 0.74)
        current_height = im.shape[0]

        buffer = xlcv.to_buffer(im, b64encode=b64encode)
        ratio = current_height / origin_height
        return buffer, ratio

    @staticmethod
    def imshow2(im, winname=None, flags=1):
        """ 展示窗口
    
        :param winname: 未输入时，则按test1、test2依次生成窗口
        :param flags:
            cv2.WINDOW_NORMAL，0，输入2等偶数值好像也等价于输入0，可以自动拉伸窗口大小
            cv2.WINDOW_AUTOSIZE，1，输入3等奇数值好像等价于1
            cv2.WINDOW_OPENGL，4096
        :return:
        """
        if winname is None:
            n = _show_win_num + 1
            winname = f'test{n}'
        cv2.namedWindow(winname, flags)
        cv2.imshow(winname, im)

    @staticmethod
    def display(im):
        """ 在jupyter中展示 """
        try:
            from IPython.display import display
            display(xlcv.to_pil_image(im))
        except ModuleNotFoundError:
            pass

    @staticmethod
    def __4_plot():
        pass

    @staticmethod
    def bg_color(im, edge_size=5, binary_img=None):
        """ 智能判断图片背景色
    
        对全图二值化后，考虑最外一层宽度为edge_size的环中，0、1分布最多的作为背景色
            然后取全部背景色的平均值返回
    
        :param im: 支持黑白图、彩图
        :param edge_size: 边缘宽度，宽度越高一般越准确，但也越耗性能
        :param binary_img: 运算中需要用二值图，如果外部已经计算了，可以直接传入进来，避免重复运算
        :return: color
    
        TODO 可以写个获得前景色，道理类似，只是最后在图片中心去取平均值
        """
        from itertools import chain

        # 1 获得二值图，区分前背景
        if binary_img is None:
            gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
            _, binary_img = cv2.threshold(gray_img, np.mean(gray_img), 255, cv2.THRESH_BINARY)

        # 2 分别存储点集
        n, m = im.shape[:2]
        colors0, colors1 = [], []
        for i in range(n):
            if i < edge_size or i >= n - edge_size:
                js = range(m)
            else:
                js = chain(range(edge_size), range(m - edge_size, m))
            for j in js:
                if binary_img[i, j]:
                    colors1.append(im[i, j])
                else:
                    colors0.append(im[i, j])

        # 3 计算平均像素
        # 以数量多的作为背景像素
        colors = colors0 if len(colors0) > len(colors1) else colors1
        return np.mean(np.array(colors), axis=0, dtype='int').tolist()

    @staticmethod
    def get_plot_color(im):
        """ 获得比较适合的作画颜色
    
        TODO 可以根据背景色智能推导画线用的颜色，目前是固定红色
        """
        if im.ndim == 3:
            return 0, 0, 255
        elif im.ndim == 2:
            return 255  # 灰度图，默认先填白色

    @staticmethod
    def get_plot_args(im, color=None):
        # 1 作图颜色
        if not color:
            color = xlcv.get_plot_color(im)

        # 2 画布
        if len(color) >= 3 and im.ndim <= 2:
            dst = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        else:
            dst = np.array(im)

        return dst, color

    @staticmethod
    def lines(im, lines, color=None, thickness=1, line_type=cv2.LINE_AA, shift=None):
        """ 在src图像上画系列线段
        """
        # 1 判断 lines 参数内容
        lines = np.array(lines).reshape(-1, 4)
        if not lines.size:
            return im

        # 2 参数
        dst, color = xlcv.get_plot_args(im, color)

        # 3 画线
        if lines.any():
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(dst, (x1, y1), (x2, y2), color, thickness, line_type)
        return dst

    @staticmethod
    def circles(im, circles, color=None, thickness=1, center=False):
        """ 在图片上画圆形
    
        :param im: 要作画的图
        :param circles: 要画的圆形参数 (x, y, 半径 r)
        :param color: 画笔颜色
        :param center: 是否画出圆心
        """
        # 1 圆 参数
        circles = np.array(circles, dtype=int).reshape(-1, 3)
        if not circles.size:
            return im

        # 2 参数
        dst, color = xlcv.get_plot_args(im, color)

        # 3 作画
        for x in circles:
            cv2.circle(dst, (x[0], x[1]), x[2], color, thickness)
            if center:
                cv2.circle(dst, (x[0], x[1]), 2, color, thickness)

        return dst

    __5_resize = """
    """

    @staticmethod
    def reduce_area(im, area):
        """ 根据面积上限缩小图片
    
        即图片面积超过area时，按照等比例缩小到面积为area的图片
        """
        h, w = xlcv.imsize(im)
        s = h * w
        if s > area:
            r = (area / s) ** 0.5
            size = int(r * h), int(r * w)
            im = xlcv.resize2(im, size)
        return im

    @staticmethod
    def adjust_shape(im, min_length=None, max_length=None):
        """ 限制图片的最小边，最长边

        >>> a = np.zeros((100, 200,3), np.uint8)
        >>> xlcv.adjust_shape(a, 101).shape
        (101, 202, 3)
        >>> xlcv.adjust_shape(a, max_length=150).shape
        (75, 150, 3)
        """
        # 1 参数预计算
        h, w = im.shape[:2]
        x, y = min(h, w), max(h, w)  # 短边记为x, 长边记为y
        a, b = min_length, max_length  # 小阈值记为a, 大阈值记为b
        r = 1  # 需要进行的缩放比例

        # 2 判断缩放系数r
        if a and b:  # 同时考虑放大缩小，不可能真的放大和缩小的，要计算逻辑，调整a、b值
            if y * a > x * b:
                raise ValueError(f'无法满足缩放要求 {x}x{y} limit {a} {b}')

        if a and x < a:
            r = a / x  # 我在想有没可能有四舍五入误差，导致最终resize后获得的边长小了？
        elif b and y > b:
            r = b / y

        # 3 缩放图片
        if r != 1:
            im = cv2.resize(im, None, fx=r, fy=r)

        return im

    @staticmethod
    def resize2(im, dsize, **kwargs):
        """
        :param dsize: (h, w)
        :param kwargs:
            interpolation=cv2.INTER_CUBIC
        """
        # if 'interpolation' not in kwargs:
        #     kwargs['interpolation'] = cv2.INTER_CUBIC
        return cv2.resize(im, dsize[::-1], **kwargs)

    @staticmethod
    def reduce_filesize(im, filesize=None, suffix='.jpg'):
        """ 按照保存后的文件大小来压缩im

        :param filesize: 单位Bytes
            可以用 300*1024 来表示 300KB
            可以不输入，默认读取后按原尺寸返回，这样看似没变化，其实图片一读一写，是会对手机拍照的很多大图进行压缩的
        :param suffix: 使用的图片类型

        >> reduce_filesize(im, 300*1024, 'jpg')
        """

        # 1 工具
        def get_file_size(im):
            success, buffer = cv2.imencode(suffix, im)
            return len(buffer)

        # 2 然后开始循环处理
        while filesize:
            r = get_file_size(im) / filesize
            if r <= 1:
                break

            # 假设图片面积和文件大小成正比，如果r=4，表示长宽要各减小至1/(r**0.5)才能到目标文件大小
            rate = min(1 / (r ** 0.5), 0.95)  # 并且限制每轮至少要缩小至95%，避免可能会迭代太多轮
            im = cv2.resize(im, (int(im.shape[0] * rate), int(im.shape[1] * rate)))
        return im

    @staticmethod
    def __5_warp():
        pass

    @staticmethod
    def warp(im, warp_mat, dsize=None, *, view_rate=False, max_zoom=1, reserve_struct=False):
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
        warp_mat = np.array(warp_mat)
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
            left, top, right, bottom = rect_bounds(warp_points(pts1, warp_mat))
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

    @staticmethod
    def pad(im, pad_size, constant_values=0, mode='constant', **kwargs):
        r""" 拓宽图片上下左右尺寸
    
        基于np.pad，定制、简化了针对图片类型数据的操作
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    
        :pad_size: 输入单个整数值，对四周pad相同尺寸，或者输入四个值的list，表示对上、下、左、右的扩展尺寸
    
        >>> a = np.ones([2, 2])
        >>> b = np.ones([2, 2, 3])
        >>> xlcv.pad(a, 1).shape  # 上下左右各填充1行/列
        (4, 4)
        >>> xlcv.pad(b, 1).shape
        (4, 4, 3)
        >>> xlcv.pad(a, [1, 2, 3, 0]).shape  # 上填充1，下填充2，左填充3，右不填充
        (5, 5)
        >>> xlcv.pad(b, [1, 2, 3, 0]).shape
        (5, 5, 3)
        >>> xlcv.pad(a, [1, 2, 3, 0])
        array([[0., 0., 0., 0., 0.],
               [0., 0., 0., 1., 1.],
               [0., 0., 0., 1., 1.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])
        >>> xlcv.pad(a, [1, 2, 3, 0], 255)
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

    @staticmethod
    def _get_subrect_image(im, pts, fill=0):
        """
        :return:
            dst_img 按外接四边形截取的子图
            new_pts 新的变换后的点坐标
        """
        # 1 计算需要pad的宽度
        x1, y1, x2, y2 = [round_int(v) for v in rect_bounds(pts)]
        h, w = im.shape[:2]
        pad = [-y1, y2 - h, -x1, x2 - w]  # 各个维度要补充的宽度
        pad = [max(0, v) for v in pad]  # 负数宽度不用补充，改为0

        # 2 pad并定位rect局部图
        tmp_img = xlcv.pad(im, pad, fill) if max(pad) > 0 else im
        dst_img = tmp_img[y1 + pad[0]:y2, x1 + pad[2]:x2]  # 这里越界不会报错，只是越界的那个维度shape为0
        new_pts = [(pt[0] - x1, pt[1] - y1) for pt in pts]
        return dst_img, new_pts

    @staticmethod
    def get_sub(im, pts, *, fill=0, warp_quad=False):
        """ 从src_im取一个子图
    
        :param im: 原图
            可以是图片路径、np.ndarray、PIL.Image对象
            TODO 目前只支持np.ndarray、pil图片输入，返回统一是np.ndarray
        :param pts: 子图位置信息
            只有两个点，认为是矩形的两个对角点
            只有四个点，认为是任意四边形
            同理，其他点数量，默认为多边形的点集
        :param fill: 支持pts越界选取，此时可以设置fill自动填充的颜色值
            TODO fill填充一个rgb颜色的时候应该会不兼容报错，还要想办法优化
        :param warp_quad: 当pts为四个点时，是否进行仿射变换矫正
            默认是截图pts的外接四边形区域
            一般写 True、'average'，也可以写'max'、'min'，详见 quad_warp_wh()
        :return: 子图
            文件、np.ndarray --> np.ndarray
            PIL.Image --> PIL.Image
        """
        dst, pts = xlcv._get_subrect_image(xlcv.read(im), reshape_coords(pts, 2), fill)
        if len(pts) == 4 and warp_quad:
            w, h = quad_warp_wh(pts, method=warp_quad)
            warp_mat = get_warp_mat(pts, rect2polygon([[0, 0], [w, h]]))
            dst = xlcv.warp(dst, warp_mat, (w, h))
        return dst

    @staticmethod
    def trim(im, *, border=0, color=None):
        """ 如果想用cv2实现，可以参考： https://stackoverflow.com/questions/49907382/how-to-remove-whitespace-from-an-image-in-opencv
        目前为了偷懒节省代码量，就直接调用pil的版本了
        """
        from pyxllib.cv.xlpillib import xlpil
        im = xlcv.to_pil_image(im)
        im = xlpil.trim(im, border=border, color=color)
        return xlpil.to_cv2_image(im)

    @staticmethod
    def keep_subtitles(im, judge_func=None, trim_color=(255, 255, 255)):
        """ 保留（白色）字幕，去除背景，并会裁剪图片缩小尺寸

        是比较业务级的一个功能，主要是这段代码挺有学习价值的，有其他变形需求，
        可以修改judge_func，也可以另外写函数，这个仅供参考
        """

        def fore_pixel(rgb):
            """ 把图片变成白底黑字

            判断像素是不是前景，是返回0，不是返回255
            """
            if sum(rgb > 170) == 3 and rgb.max() - rgb.min() < 30:
                return [0, 0, 0]
            else:
                return [255, 255, 255]

        if judge_func is None:
            judge_func = fore_pixel

        im2 = np.apply_along_axis(judge_func, 2, im).astype('uint8')
        if trim_color:
            im2 = xlcv.trim(im2, color=trim_color)
        return im2

    @classmethod
    def concat(cls, images, *, pad=5, pad_color=None):
        """ 拼接输入的多张图为一张图，请自行确保尺寸匹配

        :param images: 一维或二维list数组存储的np.array矩阵
            一维的时候，默认按行拼接，变成 n*1 的图片。如果要显式按列拼接，请再套一层list，输入 [1, n] 的list
        :param pad: 图片之间的间隔，也可以输入 [5, 10] 表示左右间隔5像素，上下间隔10像素
        :param pad_color: 填充颜色，默认白色

        # TODO 下标编号
        """
        if pad_color is None:
            pad_color = 255

        if isinstance(pad, int):
            pad = [pad, pad]

        def hstack(imgs):
            """ 水平拼接图片 """
            patches = []

            max_length = max([im.shape[0] for im in imgs])
            max_channel = max([xlcv.n_channels(im) for im in imgs])
            if pad[1]:
                board = np.ones([max_length, pad[1]], dtype='uint8') * pad_color
                if max_channel == 3:
                    board = xlcv.cvt_channel(board, 1)
            else:
                board = None

            for im in imgs:
                h = im.shape[0]
                if h != max_length:
                    im = xlcv.pad(im, [0, max_length - h, 0, 0])
                if max_channel == 3:
                    im = xlcv.cvt_channel(im, 1)
                patches += [im]
                if board is not None:
                    patches += [board]
            if board is None:
                return np.hstack(patches)
            else:
                return np.hstack(patches[:-1])

        def vstack(imgs):
            patches = []

            max_length = max([im.shape[1] for im in imgs])
            max_channel = max([xlcv.n_channels(im) for im in imgs])
            if pad[0]:
                board = np.ones([pad[0], max_length], dtype='uint8') * pad_color
                if max_channel == 3:
                    board = xlcv.cvt_channel(board, 1)
            else:
                board = None

            for im in imgs:
                w = im.shape[1]
                if w != max_length:
                    im = xlcv.pad(im, [0, 0, 0, max_length - w])
                if max_channel == 3:
                    im = xlcv.cvt_channel(im, 1)
                patches += [im]
                if board is not None:
                    patches += [board]

            if board is None:
                return np.vstack(patches)
            else:
                return np.vstack(patches[:-1])

        if isinstance(images[0], list):
            images = [hstack(imgs) for imgs in images]
        return vstack(images)

    def __6_other(self):
        pass

    @staticmethod
    def count_pixels(im):
        """ 统计image中各种rgb出现的次数 """
        colors, counts = np.unique(im.reshape(-1, 3), axis=0, return_counts=True)
        colors = [[tuple(c), cnt] for c, cnt in zip(colors, counts)]
        colors.sort(key=lambda x: -x[1])
        return colors

    @staticmethod
    def color_desc(im, color_num=10):
        """ 描述一张图的颜色分布，这个速度还特别慢，没有优化 """
        from collections import Counter
        from pyxllib.cv.rgbfmt import RgbFormatter

        colors = xlcv.count_pixels(im)
        total = sum([cnt for _, cnt in colors])
        colors2 = Counter()
        for c, cnt in colors[:10000]:
            c0 = RgbFormatter(*c).find_similar_std_color()
            colors2[c0.to_tuple()] += cnt

        for c, cnt in colors2.most_common(color_num):
            desc = RgbFormatter(*c).relative_color_desc()
            print(desc, f'{cnt / total:.2%}')


class CvImg(np.ndarray):
    def __new__(cls, input_array, info=None):
        """ 从np.ndarray继承的固定写法
        https://numpy.org/doc/stable/user/basics.subclassing.html

        该类使用中完全等价np.ndarray，但额外增加了xlcv中的功能
        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    @classmethod
    def read(cls, file, flags=None, **kwargs):
        return cls(xlcv.read(file, flags, **kwargs))

    @classmethod
    def read_from_buffer(cls, buffer, flags=None, *, b64decode=False):
        return cls(xlcv.read_from_buffer(buffer, flags, b64decode=b64decode))

    @classmethod
    def read_from_url(cls, url, flags=None, *, b64decode=False):
        return cls(xlcv.read_from_url(url, flags, b64decode=b64decode))

    @property
    def imsize(self):
        # 这里几个属性本来可以直接调用xlcv的实现，但为了性能，这里复写一遍
        return self.shape[:2]

    @property
    def n_channels(self):
        if self.ndim == 3:
            return self.shape[2]
        else:
            return 1

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    def __getattr__(self, item):
        """ 对cv2、xlcv的类层级接口封装

        这里使用了较高级的实现方法
        好处：从而每次开发只需要在xlcv写一遍
        坏处：没有代码接口提示...

        注意，同名方法，比如size，会优先使用np.ndarray的版本
        所以为了区分度，xlcv有imsize来代表xlcv的size版本

        并且无法直接使用 cv2.resize， 因为np.ndarray已经有了resize

        这里没有做任何安全性检查，请开发使用者自行分析使用合理性
        """

        def warp_func(*args, **kwargs):
            func = getattr(cv2, item, getattr(xlcv, item, None))  # 先在cv2找方法，再在xlcv找方法
            if func is None:
                raise ValueError(f'不存在的方法名 {item}')
            res = func(self, *args, **kwargs)
            if isinstance(res, np.ndarray):  # 返回是原始图片格式，打包后返回
                return type(self)(res)  # 自定义方法必须自己再转成CvImage格式，否则会变成np.ndarray
            elif isinstance(res, tuple):  # 如果是tuple类型，则里面的np.ndarray类型也要处理
                res2 = []
                for x in res:
                    if isinstance(x, np.ndarray):
                        res2.append(type(self)(x))
                    else:
                        res2.append(x)
                return tuple(res2)
            return res

        return warp_func


if __name__ == '__main__':
    im = xlcv.read(r"C:\home\chenkunze\data\aipocr_test\01通用\accurate\3e99d47940bf9fae942c733dcdf5dbe7.jpg")
    im = xlcv.reduce_filesize(im, 20 * 1024)
    xlcv.write(im, 'code4101.jpg')
    a = xlcv.to_buffer(im)
    b = xlcv.to_buffer(im, b64encode=True)
    print(humanfriendly.format_size(len(a)), humanfriendly.format_size(len(b)), len(b) / len(a))
