#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/11/17 15:13

import io

from PIL import Image
import PIL.ExifTags
import numpy as np

from pyxllib.file.specialist import File
from pyxllib.cv.specialist.cvprcs import CvPrcsBase, is_pil_image, is_numpy_image, cv2pil


class PilPrcsBase(CvPrcsBase):
    """ 相同功能，但pil要另外实现的算法 """

    @classmethod
    def read(cls, file, flags=None, **kwargs):
        if is_pil_image(file):
            im = file
        elif is_numpy_image(file):
            im = cv2pil(file)
        elif File(file):
            im = Image.open(str(file), **kwargs)
        else:
            raise TypeError(f'类型错误或文件不存在：{type(file)} {file}')
        return cls.cvt_channel(im, flags)

    @classmethod
    def cvt_channel(cls, im, flags=None):
        if flags is None: return im
        n_c = cls.n_channels(im)
        if flags == 0 and n_c > 1:
            im = im.convert('L')
        elif flags == 1 and n_c != 3:
            im = im.convert('RGB')
        return im

    @classmethod
    def write(cls, im, path, if_exists=None, **kwargs):
        p = File(path)
        if p.exist_preprcs(if_exists):
            p.ensure_parent()
            im.save(str(p), **kwargs)

    @classmethod
    def resize(cls, im, size, **kwargs):
        """

        :param kwargs:
            resample=3，插值算法；有PIL.Image.NEAREST, ~BOX, ~BILINEAR, ~HAMMING, ~BICUBIC, ~LANCZOS等
                默认是 PIL.Image.BICUBIC；如果mode是"1"或"P"模式，则总是 PIL.Image.NEAREST

        >>> im = PilPrcs.read(np.zeros([100, 200], dtype='uint8'), 0)
        >>> im.size
        (100, 200)
        >>> im2 = im.reduce_area(50*50, **kwargs)
        >>> im2.size
        (35, 70)
        """
        # 注意pil图像尺寸接口都是[w,h]，跟标准的[h,w]相反
        return im.resize(size[::-1])

    @classmethod
    def size(cls, im):
        w, h = im.size
        return h, w

    @classmethod
    def n_channels(cls, im):
        """ 通道数 """
        return len(im.getbands())

    @classmethod
    def show(cls, im, title=None, command=None):
        return cls.show(im, title, command)


class PilPrcs(PilPrcsBase):
    """ pil暂时独有的功能 """

    @classmethod
    def random_direction(cls, im):
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

    @classmethod
    def reset_orientation(cls, im):
        """Image.open读取图片时，是手机严格正放时拍到的图片效果，
        但手机拍照时是会记录旋转位置的，即可以判断是物理空间中，实际朝上、朝下的方向，
        从而识别出正拍（代号1），顺时针旋转90度拍摄（代号8），顺时针180度拍摄（代号3）,顺时针270度拍摄（代号6）。
        windows上的图片查阅软件能识别方向代号后正确摆放；
        为了让python处理图片的时候能增加这个属性的考虑，这个函数能修正识别角度返回新的图片。

        旧函数名：图像实际视图
        """
        exif_data = im._getexif()
        if exif_data:
            exif = {
                PIL.ExifTags.TAGS[k]: v
                for k, v in exif_data.items()
                if k in PIL.ExifTags.TAGS
            }
            orient = exif['Orientation']
            if orient == 8:
                im = im.transpose(PIL.Image.ROTATE_90)
            elif orient == 3:
                im = im.transpose(PIL.Image.ROTATE_180)
            elif orient == 6:
                im = im.transpose(PIL.Image.ROTATE_270)
        return im

    @classmethod
    def get_exif(cls, im):
        """ 旧函数名：查看图片的Exif信息 """
        exif_data = im._getexif()
        if exif_data:
            exif = {PIL.ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in PIL.ExifTags.TAGS}
        else:
            exif = None
        return exif

    @classmethod
    def rgba2rgb(cls, im):
        if im.mode in ('RGBA', 'P'):
            # 判断图片mode模式，如果是RGBA或P等可能有透明底，则和一个白底图片合成去除透明底
            background = Image.new('RGBA', im.size, (255, 255, 255))
            # composite是合成的意思。将右图的alpha替换为左图内容
            im = Image.alpha_composite(background, im.convert('RGBA')).convert('RGB')
        return im

    @classmethod
    def reduce_filesize(cls, im, filesize=None, suffix='jpeg'):
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
