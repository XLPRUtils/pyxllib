#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/25 15:57

import cv2
import numpy as np
from PIL import Image
import PIL.ExifTags
import PIL.Image

from pyxllib.prog.newbie import RunOnlyOnce
from pyxllib.file.specialist import File
from pyxllib.debug.specialist import get_xllog

from pyxllib.cv import xlcv, xlpil


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


def check_exist_method_name(exist_names, name):
    for k in exist_names:
        if name in exist_names[k]:
            print(f'警告！同名冲突！ {k}.{name}')


@RunOnlyOnce
def pil_binding_xlpil():
    """ 把xlpil的功能嵌入到PIL.Image.Image类中，作为成员函数直接使用
    即 im = Image.open('test.jpg')
    im.to_buffer() 等价于使用 xlpil.to_buffer(im)

    pil相比cv，由于无法类似CvImg这样新建一个和np.ndarray等效的类
    所以还是比较支持嵌入到Image中直接操作
    """
    # 0 已有的方法名，任何新功能接口都不能跟已有的重名，避免歧义，误导性太强
    exist_names = {'cv2': set(dir(cv2)),
                   'np.ndarray': set(dir(np.ndarray)),
                   'PIL.Image': set(dir(PIL.Image)),
                   'PIL.Image.Image': set(dir(PIL.Image.Image))}

    # 1 绑定到模块下的方法
    pil_names = set('read read_from_buffer read_from_url'.split())
    for name in pil_names:
        check_exist_method_name(exist_names, name)
        setattr(PIL.Image, name, getattr(xlpil, name))

    # 2 绑定到PIL.Image.Image下的方法
    # 2.1 属性类
    attrs_names = set('imsize n_channels'.split())
    for name in attrs_names:
        check_exist_method_name(exist_names, name)
        setattr(PIL.Image.Image, name, property(getattr(xlpil, name)))

    # 2.2 其他均为方法类
    all_names = {x for x in dir(xlpil) if (x[:2] != '__' or x[-2:] != '__')}
    all_names -= set('base64 io cv2 np Image requests accimage File xlcv'.split())
    for name in (all_names - pil_names - attrs_names):
        check_exist_method_name(exist_names, name)
        setattr(PIL.Image.Image, name, getattr(xlpil, name))


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
        im1 = xlcv.read(f)
        im2 = func(im1)

        if save:
            xlcv.write(im2, File(save / f.name, dir_))

        if show:
            xlcv.imshow2(im2)
            key = cv2.waitKey()
            if key == '0x1B':  # ESC 键
                break
