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


class TextlineShape:
    """ 一个文本行标注对象

    这里的基础功能主要是几何关系计算，可以继承类后扩展其他功能
    """

    def __init__(self, box, *, shrink_bound=False):
        """
        :param box: 可以转成Polygon的数据类型
        :param shrink_bound: 倾斜度过大的文本框，需要特殊处理，把外接矩形缩小会更准确些。
        """
        self.polygon = ShapelyPolygon.gen(box)
        self.bounds = self.polygon.bounds
        if shrink_bound:
            b = self.bounds
            total_area = (b[2] - b[0]) * (b[3] - b[1])
            # 缩放比例
            self.bounds = bound_scale(self.bounds, self.polygon.area / total_area)

        self.minx, self.maxx = self.bounds[0], self.bounds[2]
        self.width = self.maxx - self.minx
        self.miny, self.maxy = self.bounds[1], self.bounds[3]
        self.height = self.maxy - self.miny
        self.centroid = self.polygon.centroid

    def in_the_same_line(self, other):
        """ 两个框在同一个文本行 """
        if other.miny < self.centroid.y < other.maxy:
            return True
        elif self.miny < other.centroid.y < self.maxy:
            return True
        else:
            return False

    def is_lr_intersect(self, other, gap=5):
        """ 左右相交
        """
        if other.minx - gap <= self.minx <= other.maxx + gap:
            return True
        elif other.minx - gap <= self.maxx <= other.maxx + gap:
            return True
        else:
            return False

    def is_tb_intersect(self, other, gap=5):
        """ 上下相交
        """
        # 这个 gap 规则是不动产的，不能放在通用规则里
        # gap = min(50, self.height / 2, other.height / 2)  # 允许的最大间距，默认按照最小的高，但还要再设置一个50的上限
        if other.miny - gap <= self.miny <= other.maxy + gap:
            return True
        elif other.miny - gap <= self.maxy <= other.maxy + gap:
            return True
        else:
            return False

    def is_intersect(self, other):
        return self.polygon.intersects(other)

    def __add__(self, other):
        """ 合并两个文本行 """
        box = rect2polygon(MultiPolygon([self.polygon, other.polygon]).bounds)
        return TextlineShape(box)

    def __lt__(self, other):
        """ 框的排序准则 """
        if self.in_the_same_line(other):
            return self.centroid.x < other.centroid.x
        else:
            return self.centroid.y < other.centroid.y

    @classmethod
    def merge(cls, shapes):
        """ 将同张图片里的多个shape进行合并 """
        # 1 对文本框分组
        shape_groups = disjoint_set(shapes, lambda x, y: x.is_intersect(y))

        # 2 合并文本内容
        new_shapes = []
        for group in shape_groups:
            shape = reduce(lambda x, y: x + y, sorted(group))
            new_shapes.append(shape)
        return new_shapes


def get_font_file(name):
    """ 获得指定名称的字体文件

    :param name: 记得要写后缀，例如 "simfang.ttf"
        simfang.ttf，仿宋
        msyh.ttf，微软雅黑
    """
    from pyxllib.file.specialist import ensure_localfile, XlPath

    # 1 windows直接找系统的字体目录
    font_file = XlPath(f'C:/Windows/Fonts/{name}')
    if font_file.is_file():
        return font_file

    # 2 否则下载到.xlpr/fonts
    # 注意不能下载到C:/Windows/Fonts，会遇到权限问题，报错
    font_file = XlPath.userdir() / f'.xlpr/fonts/{name}'
    # 去github上paddleocr项目下载
    # TODO 不过paddleocr字体数量有限，而且github有时候可能会卡下载不了~~本来要弄到码云，但是码云现在要登陆才能下载了
    from_url = f'https://raw.githubusercontent.com/code4101/data1/main/fonts/{name}'
    try:
        ensure_localfile(font_file, from_url)
    except TimeoutError as e:
        raise TimeoutError(f'{font_file} 下载失败：{from_url}')

    return font_file
