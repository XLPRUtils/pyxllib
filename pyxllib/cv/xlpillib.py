#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/25 15:57

"""
图方便的时候，xlpil和xlcv可以进行图片类型转换，然后互相引用彼此已有的一个实现版本
为了性能的时候，则尽量减少各种绕弯，使用最源生的代码来实现
"""

import base64
import io
import itertools

import cv2
import numpy as np
from PIL import Image
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import requests

try:
    import accimage
except ImportError:
    accimage = None

from pyxllib.prog.newbie import RunOnlyOnce
from pyxllib.prog.pupil import EnchantBase
from pyxllib.file.specialist import File
from pyxllib.cv.xlcvlib import xlcv


class xlpil(EnchantBase):
    @classmethod
    @RunOnlyOnce
    def enchant(cls):
        """ 把xlpil的功能嵌入到PIL.Image.Image类中，作为成员函数直接使用
        即 im = Image.open('test.jpg')
        im.to_buffer() 等价于使用 xlpil.to_buffer(im)

        pil相比cv，由于无法类似CvImg这样新建一个和np.ndarray等效的类
        所以还是比较支持嵌入到Image中直接操作
        """
        names = cls.check_enchant_names([cv2, np.ndarray, PIL.Image, PIL.Image.Image])

        # 1 绑定到模块下的方法
        pil_names = 'read read_from_buffer read_from_url'.split()
        cls._enchant(PIL.Image, pil_names, 'staticmethod2modulefunc')

        # 2 绑定到PIL.Image.Image下的方法
        # 2.1 属性类
        propertys = 'imsize n_channels'.split()
        cls._enchant(PIL.Image.Image, propertys, 'staticmethod2property')

        # 2.2 其他均为方法类
        cls._enchant(PIL.Image.Image, names - pil_names - propertys, 'staticmethod2objectmethod')

    @staticmethod
    def __1_read():
        pass

    @staticmethod
    def read(file, flags=None, **kwargs):
        if xlpil.is_pil_image(file):
            im = file
        elif xlcv.is_cv2_image(file):
            im = xlcv.to_pil_image(file)
        elif File.safe_init(file):
            im = Image.open(str(file), **kwargs)
        else:
            raise TypeError(f'类型错误或文件不存在：{type(file)} {file}')
        return xlpil.cvt_channel(im, flags)

    @staticmethod
    def read_from_buffer(buffer, flags=None, *, b64decode=False):
        """ 先用opencv实现，以后可以再研究PIL.Image.frombuffer是否有更快处理策略 """
        if b64decode:
            buffer = base64.b64decode(buffer)
        im = Image.open(io.BytesIO(buffer))
        return xlpil.cvt_channel(im, flags)

    @staticmethod
    def read_from_url(url, flags=None, *, b64decode=False):
        content = requests.get(url).content
        return xlpil.read_from_buffer(content, flags, b64decode=b64decode)

    @staticmethod
    def __2_attrs():
        pass

    @staticmethod
    def imsize(im):
        return im.size[::-1]

    @staticmethod
    def n_channels(im):
        """ 通道数 """
        return len(im.getbands())

    @staticmethod
    def __3_write():
        pass

    @staticmethod
    def to_cv2_image(im):
        """ pil图片转np图片 """
        y = np.array(im)
        y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB) if y.size else None
        return y

    @staticmethod
    def is_pil_image(im):
        if accimage is not None:
            return isinstance(im, (Image.Image, accimage.Image))
        else:
            return isinstance(im, Image.Image)

    @staticmethod
    def write(im, path, if_exists=None, **kwargs):
        p = File(path)
        if p.exist_preprcs(if_exists):
            p.ensure_parent()
            im.save(str(p), **kwargs)

    @staticmethod
    def cvt_channel(im, flags=None):
        if flags is None: return im
        n_c = xlpil.n_channels(im)
        if flags == 0 and n_c > 1:
            im = im.convert('L')
        elif flags == 1 and n_c != 3:
            im = im.convert('RGB')
        return im

    @staticmethod
    def to_buffer(im, ext='.jpg', *, b64encode=False):
        # 主要是偷懒，不想重写一遍，就直接去调用cv版本的实现了
        return xlcv.to_buffer(xlpil.to_cv2_image(im), ext, b64encode=b64encode)

    @staticmethod
    def display(im):
        """ 在jupyter中展示 """
        try:
            from IPython.display import display
            display(im)
        except ModuleNotFoundError:
            pass

    @staticmethod
    def __4_plot():
        pass

    @staticmethod
    def plot_border(im, border=1, fill='black'):
        """ 给图片加上边框

        Args:
            im:
            border: 边框的厚度
            fill: 边框颜色

        Returns: 一张新图
        """
        from PIL import ImageOps
        im2 = ImageOps.expand(im, border=border, fill=fill)
        return im2

    @staticmethod
    def __5_resize():
        pass

    @staticmethod
    def reduce_area(im, area):
        """ 根据面积上限缩小图片

        即图片面积超过area时，按照等比例缩小到面积为area的图片
        """
        h, w = xlpil.imsize(im)
        s = h * w
        if s > area:
            r = (area / s) ** 0.5
            size = int(r * h), int(r * w)
            im = xlpil.resize2(im, size)
        return im

    @staticmethod
    def resize2(im, size, **kwargs):
        """
        :param size: 默认是 (w, h)， 这里我倒过来 (h, w)
            但计算机领域，确实经常都是用 (w, h) 的格式，毕竟横轴是x，纵轴才是y
        :param kwargs:
            resample=3，插值算法；有PIL.Image.NEAREST, ~BOX, ~BILINEAR, ~HAMMING, ~BICUBIC, ~LANCZOS等
                默认是 PIL.Image.BICUBIC；如果mode是"1"或"P"模式，则总是 PIL.Image.NEAREST

        >>> im = read(np.zeros([100, 200], dtype='uint8'), 0)
        >>> im.size
        (100, 200)
        >>> im2 = im.reduce_area(50*50, **kwargs)
        >>> im2.size
        (35, 70)
        """
        # 注意pil图像尺寸接口都是[w,h]，跟标准的[h,w]相反
        return im.resize(size[::-1], **kwargs)

    @staticmethod
    def reduce_filesize(im, filesize=None, suffix='jpeg'):
        """ 按照保存后的文件大小来压缩im

        :param filesize: 单位Bytes
            可以用 300*1024 来表示 300KB
            可以不输入，默认读取后按原尺寸返回，这样看似没变化，其实图片一读一写，是会对手机拍照的很多大图进行压缩的
        :param suffix: 使用的图片类型

        >> reduce_filesize(im, 300*1024, 'jpg')
        """
        # 1 工具
        # save接口不支持jpg参数
        if suffix == 'jpg':
            suffix = 'jpeg'

        def get_file_size(im):
            file = io.BytesIO()
            im.save(file, suffix)
            return len(file.getvalue())

        # 2 然后开始循环处理
        while filesize:
            r = get_file_size(im) / filesize
            if r <= 1:
                break

            # 假设图片面积和文件大小成正比，如果r=4，表示长宽要各减小至1/(r**0.5)才能到目标文件大小
            rate = min(1 / (r ** 0.5), 0.95)  # 并且限制每轮至少要缩小至95%，避免可能会迭代太多轮
            im = im.resize((int(im.size[0] * rate), int(im.size[1] * rate)))
        return im

    @staticmethod
    def trim(im, *, border=0, color=(255, 255, 255)):
        """ 默认裁剪掉白色边缘，可以配合 get_backgroup_color 裁剪掉背景色

        :param border: 上下左右保留多少边缘
            输入一个整数，表示统一留边
            也可以输入[int, int, int, int]，分别表示left top right bottom留白
        :param percent-background: TODO 控制裁剪百分比上限
        """
        from PIL import Image, ImageChops
        from pyxllib.algo.geo import xywh2ltrb, ltrb2xywh, ltrb_border

        bg = Image.new(im.mode, im.size, color)
        diff = ImageChops.difference(im, bg)
        bbox = diff.getbbox()  # 如果im跟bg一样，也就是裁"消失"了，此时bbox值为None
        if bbox:
            if border:
                ltrb = xywh2ltrb(bbox)
                ltrb = ltrb_border(ltrb, border, im.size)
                bbox = ltrb2xywh(ltrb)
            im = im.crop(bbox)
        return im

    @staticmethod
    def __6_warp():
        pass

    @staticmethod
    def __x_other():
        pass

    @staticmethod
    def random_direction(im):
        """ 假设原图片是未旋转的状态0

        顺时针转90度是label=1，顺时针转180度是label2 ...
        """
        label = np.random.randint(4)
        if label == 1:
            # PIL的旋转角度，是指逆时针角度；但是我这里编号是顺时针
            im = im.transpose(PIL.Image.ROTATE_270)
        elif label == 2:
            im = im.transpose(PIL.Image.ROTATE_180)
        elif label == 3:
            im = im.transpose(PIL.Image.ROTATE_90)
        return im, label

    @staticmethod
    def apply_exif_orientation(im):
        """ 摆正图片角度

        Image.open读取图片时，是手机严格正放时拍到的图片效果，
        但手机拍照时是会记录旋转位置的，即可以判断是物理空间中，实际朝上、朝下的方向，
        从而识别出正拍（代号1），顺时针旋转90度拍摄（代号8），顺时针180度拍摄（代号3）,顺时针270度拍摄（代号6）。
        windows上的图片查阅软件能识别方向代号后正确摆放；
        为了让python处理图片的时候能增加这个属性的考虑，这个函数能修正识别角度返回新的图片。

        我自己写过个版本，后来发现 labelme.utils.image 写过功能更强的，抄了过来~~
        """
        try:
            exif = im._getexif()
        except AttributeError:
            exif = None

        if exif is None:
            return im

        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in exif.items()
            if k in PIL.ExifTags.TAGS
        }

        orientation = exif.get("Orientation", None)

        if orientation == 1:
            # do nothing
            return im
        elif orientation == 2:
            # left-to-right mirror
            return PIL.ImageOps.mirror(im)
        elif orientation == 3:
            # rotate 180
            return im.transpose(PIL.Image.ROTATE_180)
        elif orientation == 4:
            # top-to-bottom mirror
            return PIL.ImageOps.flip(im)
        elif orientation == 5:
            # top-to-left mirror
            return PIL.ImageOps.mirror(im.transpose(PIL.Image.ROTATE_270))
        elif orientation == 6:
            # rotate 270
            return im.transpose(PIL.Image.ROTATE_270)
        elif orientation == 7:
            # top-to-right mirror
            return PIL.ImageOps.mirror(im.transpose(PIL.Image.ROTATE_90))
        elif orientation == 8:
            # rotate 90
            return im.transpose(PIL.Image.ROTATE_90)
        else:
            return im

    @staticmethod
    def get_exif(im):
        """ 旧函数名：查看图片的Exif信息 """
        exif_data = im._getexif()
        if exif_data:
            exif = {PIL.ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in PIL.ExifTags.TAGS}
        else:
            exif = None
        return exif

    @staticmethod
    def rgba2rgb(im):
        if im.mode in ('RGBA', 'P'):
            # 判断图片mode模式，如果是RGBA或P等可能有透明底，则和一个白底图片合成去除透明底
            background = Image.new('RGBA', im.size, (255, 255, 255))
            # composite是合成的意思。将右图的alpha替换为左图内容
            im = Image.alpha_composite(background, im.convert('RGBA')).convert('RGB')
        return im
