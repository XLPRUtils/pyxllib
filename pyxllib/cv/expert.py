#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/25 15:57

from functools import reduce

import cv2
from shapely.geometry import MultiPolygon

from pyxllib.algo.geo import bound_scale, rect2polygon
from pyxllib.algo.shapelylib import ShapelyPolygon
from pyxllib.algo.disjoint import disjoint_set
from pyxllib.cv.xlcvlib import xlcv
from pyxllib.cv.xlpillib import xlpil  # noqa
from pyxllib.file.specialist import File


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
        im1 = xlcv.read(f)
        im2 = func(im1)

        if save:
            xlcv.write(im2, File(save / f.name, dir_))

        if show:
            xlcv.imshow2(im2)
            key = cv2.waitKey()
            if key == '0x1B':  # ESC 键
                break

