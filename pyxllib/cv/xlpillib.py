#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/25 15:57

"""
图方便的时候，PilImg和CvImg可以进行图片类型转换，然后互相引用彼此已有的一个实现版本
为了性能的时候，则尽量减少各种绕弯，使用最源生的代码来实现
"""

import base64
import io
import os
import random

import cv2
import numpy as np
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import requests

try:
    import accimage
except ImportError:
    accimage = None

from pyxllib.prog.pupil import inject_members
from pyxllib.file.specialist import XlPath, get_font_file
from pyxllib.cv.xlcvlib import xlcv


class PilImg(PIL.Image.Image):

    def __1_read(self):
        pass

    @classmethod
    def read(cls, file, flags=None, *, apply_exif_orientation=False, **kwargs) -> 'PilImg':
        if PilImg.is_pil_image(file):
            im = file
        elif xlcv.is_cv2_image(file):
            im = xlcv.to_pil_image(file)
        elif XlPath.safe_init(file):
            im = PIL.Image.open(str(file), **kwargs)
        else:
            raise TypeError(f'类型错误或文件不存在：{type(file)} {file}')
        if apply_exif_orientation:
            im = PilImg.apply_exif_orientation(im)
        return PilImg.cvt_channel(im, flags)

    @classmethod
    def read_from_buffer(cls, buffer, flags=None, *, b64decode=False):
        """ 先用opencv实现，以后可以再研究PIL.Image.frombuffer是否有更快处理策略 """
        if b64decode:
            buffer = base64.b64decode(buffer)
        im = PIL.Image.open(io.BytesIO(buffer))
        return PilImg.cvt_channel(im, flags)

    @classmethod
    def read_from_url(cls, url, flags=None, *, b64decode=False):
        content = requests.get(url).content
        return PilImg.read_from_buffer(content, flags, b64decode=b64decode)

    def __2_attrs(self):
        pass

    def imsize(self):
        return self.size[::-1]

    def n_channels(self):
        """ 通道数 """
        return len(self.getbands())

    def __3_write(self):
        pass

    def to_cv2_image(self):
        """ pil图片转np图片 """
        y = np.array(self)
        y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB) if y.size else None
        return y

    def is_pil_image(self):
        if accimage is not None:
            return isinstance(self, (PIL.Image.Image, accimage.Image))
        else:
            return isinstance(self, PIL.Image.Image)

    def write(self, path, *, if_exists=None, **kwargs):
        p = XlPath(path)
        if p.exist_preprcs(if_exists):
            os.makedirs(p.parent, exist_ok=True)
            suffix = p.suffix[1:]
            if suffix.lower() == 'jpg':
                suffix = 'jpeg'
            if self.mode in ('RGBA', 'P') and suffix == 'jpeg':
                im = self.convert('RGB')
            else:
                im = self
            im.save(str(p), suffix, **kwargs)

    def cvt_channel(self, flags=None):
        im = self
        if flags is None or flags == -1: return im
        n_c = im.n_channels
        if flags == 0 and n_c > 1:
            im = im.convert('L')
        elif flags == 1 and n_c != 3:
            im = im.convert('RGB')
        return im

    def to_buffer(self, ext='.jpg', *, b64encode=False):
        # 主要是偷懒，不想重写一遍，就直接去调用cv版本的实现了
        return xlcv.to_buffer(self.to_cv2_image(), ext, b64encode=b64encode)

    def display(self):
        """ 在jupyter中展示 """
        try:
            from IPython.display import display
            display(self)
        except ModuleNotFoundError:
            pass

    def __4_plot(self):
        pass

    def plot_border(self, border=1, fill='black'):
        """ 给图片加上边框

        Args:
            self:
            border: 边框的厚度
            fill: 边框颜色

        Returns: 一张新图
        """
        from PIL import ImageOps
        im2 = ImageOps.expand(self, border=border, fill=fill)
        return im2

    def plot_text(self, text, xy=None, font_size=10, font_type='simfang.ttf', **kwargs):
        """
        :param xy: 写入文本的起始坐标，没写入则自动写在垂直居中位置
        """
        from PIL import ImageFont, ImageDraw
        font_file = get_font_file(font_type)
        font = ImageFont.truetype(font=str(font_file), size=font_size, encoding="utf-8")
        draw = ImageDraw.Draw(self)
        if xy is None:
            w, h = font_getsize(font, text)
            xy = ((self.size[0] - w) / 2, (self.size[1] - h) / 2)
        draw.text(xy, text, font=font, **kwargs)
        return self

    def __5_resize(self):
        pass

    def reduce_area(self, area):
        """ 根据面积上限缩小图片

        即图片面积超过area时，按照等比例缩小到面积为area的图片
        """
        im = self
        h, w = PilImg.imsize(im)
        s = h * w
        if s > area:
            r = (area / s) ** 0.5
            size = int(r * h), int(r * w)
            im = PilImg.resize2(im, size)
        return im

    def resize2(self, size, **kwargs):
        """
        :param size: 默认是 (w, h)， 这里我倒过来 (h, w)
            但计算机领域，确实经常都是用 (w, h) 的格式，毕竟横轴是x，纵轴才是y
        :param kwargs:
            resample=3，插值算法；有PIL.Image.NEAREST, ~BOX, ~BILINEAR, ~HAMMING, ~BICUBIC, ~LANCZOS等
                默认是 PIL.Image.BICUBIC；如果mode是"1"或"P"模式，则总是 PIL.Image.NEAREST

        >>> im = read(np.zeros([100, 200], dtype='uint8'), 0)
        >>> self.size
        (100, 200)
        >>> im2 = im.reduce_area(50*50, **kwargs)
        >>> im2.size
        (35, 70)
        """
        # 注意pil图像尺寸接口都是[w,h]，跟标准的[h,w]相反
        return self.resize(size[::-1], **kwargs)

    def evaluate_image_file_size(self, suffix='.jpeg'):
        """ 评估图像存成文件后的大小

        :param suffix: 使用的图片类型
        :return int: 存储后的文件大小，单位为字节
        """
        im = self

        # save接口不支持jpg参数
        if suffix[0] == '.':
            suffix = suffix[1:]
        if suffix.lower() == 'jpg':
            suffix = 'jpeg'

        file = io.BytesIO()
        if im.mode in ('RGBA', 'P') and suffix == 'jpeg':
            im = im.convert('RGB')
        im.save(file, suffix)
        return len(file.getvalue())

    def reduce_filesize(self, filesize=None, suffix='.jpeg'):
        """ 按照保存后的文件大小来压缩im

        :param filesize: 单位Bytes
            int, 可以用 300*1024 来表示 300KB
            None, 可以不输入，默认读取后按原尺寸返回，这样看似没变化，其实图片一读一写，是会对手机拍照的很多大图进行压缩的

        >> reduce_filesize(im, 300*1024, 'jpg')
        """
        im = self
        # 循环处理
        while filesize:
            r = xlpil.evaluate_image_file_size(im, suffix) / filesize
            if r <= 1:
                break

            # 假设图片面积和文件大小成正比，如果r=4，表示长宽要各减小至1/(r**0.5)才能到目标文件大小
            rate = min(1 / (r ** 0.5), 0.95)  # 并且限制每轮至少要缩小至95%，避免可能会迭代太多轮
            im = im.resize((int(im.size[0] * rate), int(im.size[1] * rate)))
        return im

    def trim(self, *, border=0, color=None):
        """ 默认裁剪掉白色边缘，可以配合 get_backgroup_color 裁剪掉背景色

        :param border: 上下左右保留多少边缘
            输入一个整数，表示统一留边
            也可以输入[int, int, int, int]，分别表示left top right bottom留白
        :param color: 要裁剪的颜色，这是精确值，没有误差，如果需要模糊，可以提前预处理成精确值
            这个默认值设成None，本来是想说支持灰度图，此时输入一个255的值就好
            但是pil的灰度图机制好像有些不太一样，总之目前默认值还是自动会设成 (255, 255, 255)
        :param percent-background: TODO 控制裁剪百分比上限
        """
        from PIL import Image, ImageChops
        from pyxllib.algo.geo import xywh2ltrb, ltrb2xywh, ltrb_border

        if color is None:
            color = (255, 255, 255)
        else:
            color = tuple(color)

        # 如果图片通道数跟预设的color不同，要断言
        assert self.n_channels() == len(color), f'图片通道数{self.n_channels}跟预设的color{color}不同'

        im = self
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

    def __6_warp(self):
        pass

    def __x_other(self):
        pass

    def random_direction(self):
        """ 假设原图片是未旋转的状态0

        顺时针转90度是label=1，顺时针转180度是label2 ...
        """
        im = self
        label = np.random.randint(4)
        if label == 1:
            # PIL的旋转角度，是指逆时针角度；但是我这里编号是顺时针
            im = im.transpose(PIL.Image.ROTATE_270)
        elif label == 2:
            im = im.transpose(PIL.Image.ROTATE_180)
        elif label == 3:
            im = im.transpose(PIL.Image.ROTATE_90)
        return im, label

    def flip_direction(self, direction):
        """
        :param direction: 顺时针旋转几个90度
            标记现在图片是哪个方向：0是正常，1是向右翻转，2是向下翻转，3是向左翻转
        """
        im = self
        direction = direction % 4
        if direction:
            im = im.transpose({1: PIL.Image.ROTATE_270,
                               2: PIL.Image.ROTATE_180,
                               3: PIL.Image.ROTATE_90}[direction])
        return im

    def apply_exif_orientation(self):
        """ 摆正图片角度

        Image.open读取图片时，是手机严格正放时拍到的图片效果，
        但手机拍照时是会记录旋转位置的，即可以判断是物理空间中，实际朝上、朝下的方向，
        从而识别出正拍（代号1），顺时针旋转90度拍摄（代号8），顺时针180度拍摄（代号3）,顺时针270度拍摄（代号6）。
        windows上的图片查阅软件能识别方向代号后正确摆放；
        为了让python处理图片的时候能增加这个属性的考虑，这个函数能修正识别角度返回新的图片。

        我自己写过个版本，后来发现 labelme.utils.image 写过功能更强的，抄了过来~~
        """
        im = self
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

    def get_exif(self):
        """ 旧函数名：查看图片的Exif信息 """
        exif_data = self._getexif()
        if exif_data:
            exif = {PIL.ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in PIL.ExifTags.TAGS}
        else:
            exif = None
        return exif

    def rgba2rgb(self):
        if self.mode in ('RGBA', 'P'):
            # 判断图片mode模式，如果是RGBA或P等可能有透明底，则和一个白底图片合成去除透明底
            background = PIL.Image.new('RGBA', self.size, (255, 255, 255))
            # composite是合成的意思。将右图的alpha替换为左图内容
            self = PIL.Image.alpha_composite(background, self.convert('RGBA')).convert('RGB')
        return self

    def keep_subtitles(self, judge_func=None, trim_color=(255, 255, 255)):
        im = self.to_cv2_image()
        im = im.keep_subtitles(judge_func=judge_func, trim_color=trim_color)
        return im.to_pil_image()


# pil相比cv，由于无法类似CvImg这样新建一个和np.ndarray等效的类，所以还是比较支持嵌入到Image中直接操作
inject_members(PilImg, PIL.Image.Image)

xlpil = PilImg  # 与xlcv对称


def font_getsize(font, text):
    """ 官方自带的font.getsize遇到换行符的text，计算不准确
    """
    texts = text.split('\n')
    sizes = [font.getsize(t) for t in texts]
    w = max([w for w, h in sizes])
    h = sum([h for w, h in sizes])
    return w, h


def create_text_image(text, size=None, *, xy=None, font_size=14, bg_color=None, text_color=None, **kwargs):
    """ 生成文字图片

    :param size: 注意我这里顺序是 (height, width)
        默认None，根据写入的文字动态生成图片大小
    :param bg_color: 背景图颜色，如 (0, 0, 0)
        默认None，随机颜色
    :param text_color: 文本颜色，如 (255, 255, 255)
        默认None，随机颜色
    """
    if size is None:
        from PIL import ImageFont, ImageDraw
        font_file = get_font_file(kwargs.get('font_type', 'simfang.ttf'))
        font = ImageFont.truetype(font=str(font_file), size=font_size, encoding="utf-8")
        w, h = font_getsize(font, text)
        size = (h + 10, w + 10)
        xy = (5, 0)  # 自动生成尺寸的情况下，文本从左上角开始写

    if bg_color is None:
        bg_color = tuple([random.randint(0, 255) for i in range(3)])
    if text_color is None:
        text_color = tuple([random.randint(0, 255) for i in range(3)])

    h, w = size
    im: PilImg = PIL.Image.new('RGB', (w, h), tuple(bg_color))
    im2 = im.plot_text(text, xy=xy, font_size=font_size, fill=tuple(text_color), **kwargs)

    return im2
