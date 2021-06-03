#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/11/17 15:21


from functools import partial
from pyxllib.cv.pilprcs import *
from pyxllib.file.file import File

____XxImg = """
对CvPrcs、PilPrcs的类层级接口封装

这里使用了较高级的实现方法
好处：从而每次开发只需要在CvPrcs和PilPrcs写一遍
坏处：没有代码接口提示...
"""


class CvImg:
    prcs = CvPrcs
    imtype = np.ndarray
    __slots__ = ('im',)

    def __init__(self, im, flags=1, **kwargs):
        if isinstance(im, type(self)):
            im = im.im
        else:
            im = self.prcs.read(im, flags, **kwargs)
        self.im = im

    def __getattr__(self, item):
        if item == 'im':
            return self.im

        def warp_func(*args, **kwargs):
            res = getattr(self.prcs, item)(self.im, *args, **kwargs)
            if isinstance(res, self.imtype):  # 返回是原始图片格式，打包后返回
                return type(self)(res)
            else:  # 不在预期类型内，返回原值
                return res

        return warp_func


class PilImg(CvImg):
    """
    注意这样继承实现虽然简单，但如果是CvPrcs有，但PilPrcs没有的功能，运行是会报错的
    """
    prcs = PilPrcs
    imtype = PIL.Image.Image


____alias = """
对CvPrcs中一些常用功能的名称简化

有些功能是为了兼容旧版代码，可以逐步取消别名
"""

imread = CvPrcs.read
imwrite = CvPrcs.write
imshow = CvPrcs.show

warp_image = CvPrcs.warp
get_background_color = CvPrcs.bg_color
pad_image = CvPrcs.pad
get_sub_image = CvPrcs.get_sub

____other = """
"""


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
