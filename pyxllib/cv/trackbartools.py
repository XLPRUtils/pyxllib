#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/09/30 10:45

"""
这个文件默认不加载到cv里，需要使用的时候再导入

from pyxlib.cv.debugtools import *
"""

import math
import time

import cv2
import numpy as np

from pyxllib.debug.specialist import TicToc
from pyxllib.cv.expert import xlcv


class TrackbarTool:
    """ 滑动条控件组
    """

    def __init__(self, winname=None, *, imgproc=None, img=None, flags=1, verbose=0, delay=100):
        """
        Args:
            winname: 窗口名可以不输入，可以从func获取，或者有默认窗口名
            imgproc: 有时候明确要对指定的函数进行可视化分析
            img: 图片可以不输入，可以内置一个图片
            flags: 0 窗口可以调节大小，1 窗口按照图片大小固定
            verbose: 0 静默模式，1 显示每次运行时间，2 显示每次运行时间及参数值
            delay: 有时候改动滑动条，并不想立即生效，可以设置延时，这里单位是ms
                即如果delay时间段内已经有执行过回调函数，会避免重复执行
        """

        if winname is None:
            if imgproc:
                winname = imgproc.__name__
            else:
                winname = 'TrackbarTool'

        if img is None:
            img = np.zeros([500, 100, 3], dtype='uint8')
        else:
            img = xlcv.read(img)

        self.winname = winname
        self.img = img
        cv2.namedWindow(winname, flags)
        cv2.imshow(winname, self.img)

        self.imgproc = imgproc
        self.trackbar_names = {}

        self.verbose = verbose

        self.delay = delay
        self.last_run_time = 0

    @classmethod
    def from_album(cls, obj, **kwargs):
        """ albumentations的类功能 """

        def imgproc(img, **kws):
            f = obj(**kws, p=1)  # p是概率，这里做可视化，是百分百执行
            return f(image=img)['image']

        return cls(**kwargs, imgproc=imgproc, winname=obj.__name__)

    def imshow(self, img=None):
        """ 刷新显示的图片 """
        if img is None:
            img = self.img
        cv2.imshow(self.winname, img)

    def create_trackbar(self, name, count, value=0, *, vmaps=None, on_change=None):
        """ 创建一个滑动条

        :param name: 滑动条名称
        :param int count: 上限值
        :param int value: 初始值
        :param list[func(int)] vmaps: 滑动条只能设置整数值，但有些参数不一定是整数，可以写个自定义的映射函数
        :param on_change: 自定义，额外扩展的回调函数
        :return:
        """
        if on_change is None:
            on_change = self._on_change
        self.trackbar_names[name] = {'count': count, 'value': value, 'vmaps': vmaps}
        cv2.createTrackbar(name, self.winname, value, count, on_change)

    def default_run(self, **kwargs):
        """ 默认执行器，这个在类继承后，基本都是要自定义成自己的功能的

        kwargs里存储了当前所有滑动条的取值
        如果有设置vmap，
        """
        pass

    def _on_change(self, x):
        """ 默认的回调函数，x是对应滑动条当前的取值

        如果需要对每个特定滑动条做特殊定制操作，可以create_trackbar传入特殊on_change
        这里的_on_change是所有滑动条统一回调版本，不区分是哪个滑动条改动了
        """
        # 1 记录运行时间
        cur_time = time.time() * 1000
        if cur_time - self.last_run_time < self.delay:
            return
        else:
            self.last_run_time = cur_time

        # 2 生成、解析滑动条的参数值
        kwargs = {}
        for k, v in self.trackbar_names.items():
            x = cv2.getTrackbarPos(k, self.winname)
            if v['vmaps']:
                for kk, func in v['vmaps'].items():
                    kwargs[kk] = func(x)
            else:
                kwargs[k] = x

        # 3 执行绑定的功能
        tt = TicToc()
        if self.imgproc:
            try:
                dst = self.imgproc(self.img, **kwargs)
                self.imshow(dst)
            except Exception as e:
                if 'missing required argument' in e.args[0]:
                    pass
                else:
                    raise e
        else:
            self.default_run(**kwargs)

        # 4 是否显示运行信息
        if self.verbose == 1:
            tt.toc()
        elif self.verbose == 2:
            tt.toc(kwargs)

    def __getitem__(self, item):
        """ 可以通过 Trackbars 来获取滑动条当前值

        :param item: 滑动条名称
            扩展：也支持获取vmaps后对应的名称及值
        :return: 当前取值
        """
        if item in self.trackbar_names:
            return cv2.getTrackbarPos(item, self.winname)
        else:
            for k, v in self.trackbar_names.items():
                if v['vmaps'] and item in v['vmaps'].keys():
                    x = cv2.getTrackbarPos(k, self.winname)
                    return v['vmaps']['item'](x)


class ErodeTool(TrackbarTool):
    r""" 腐蚀 erode

    HoughLinesPTool('handwriting/C100001448573-002.png')
    """

    def __init__(self, img, winname='ErodeTool', flags=1, *, verbose=2):
        super().__init__(winname, img=img, flags=flags, verbose=verbose)
        self.create_trackbar('size_x', 100, 15)
        self.create_trackbar('size_y', 100, 15)

    def default_run(self, **kwargs):
        """ 默认执行器
        """
        size = kwargs['size_x'], kwargs['size_y']
        element = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        dst = cv2.erode(self.img, element)
        self.imshow(dst)


class HoughLinesPTool(TrackbarTool):
    r""" 霍夫线段检测

    HoughLinesPTool('handwriting/C100001448573-002.png')
    """

    def __init__(self, img, winname='HoughLinesPTool', flags=1, *, verbose=2):
        super().__init__(winname, img=img, flags=flags, verbose=verbose)

        # 1 增加控件
        # 以像素为单位的距离精度
        self.create_trackbar('rho*10', 100, 10, vmaps={'rho': lambda x: x / 10})
        # 以弧度为单位的角度精度
        self.create_trackbar('theta*pi/360', 10, 2, vmaps={'theta': lambda x: x * math.pi / 360})
        # 累加平面的阈值参数
        self.create_trackbar('threshold', 200, 80)
        # 最低线段长度
        self.create_trackbar('minLineLength', 200, 50)  # 如果要显示全，这里名称最好只有10个字符，也可能是跟图片尺寸有关，有待进一步观察
        # 允许将同一行点与点之间连接起来的最大的距离
        self.create_trackbar('maxLineGap', 100, 10)

        # 2 图片预处理，这里暂时用自适应二值化，后面可以把该部分也变成可调试项
        if self.img.ndim == 3:
            gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = self.img
        self.binary_img = cv2.adaptiveThreshold(gray_img, 255, 0, 1, 11, 3)

    def default_run(self, **kwargs):
        """ 默认执行器
        """
        lines = cv2.HoughLinesP(self.binary_img, kwargs['rho'], kwargs['theta'],
                                kwargs['threshold'], kwargs['minLineLength'], kwargs['maxLineGap'])
        if lines is None: lines = np.array([])
        # 处理用二值化，显示用原图
        self.imshow(xlcv.lines(self.img, lines))


def test_adaptiveThreshold(img=None):
    """ 滑动条工具使用方法的示例 """
    # 1 图片可以传入路径，这里做一个随机图片
    if img is None:
        img = np.random.randint(0, 255, [500, 200], dtype='uint8')
    else:
        img = xlcv.read(img, 0)  # 确保是单通道灰度图

    # 2 指定要分析的函数
    t = TrackbarTool(imgproc=cv2.adaptiveThreshold, img=img, verbose=2, delay=100)
    t.create_trackbar('maxValue', 255, 255)  # 添加滑动条：名称，最大值，默认值
    # 创建滑动条过程中，可能会报错，因为参数还不全，指定的imgproc还无法运行，可以先不用管
    t.create_trackbar('adaptiveMethod', 10, 0)
    t.create_trackbar('thresholdType', 10, 1)
    t.create_trackbar('blockSize', 100, 11)
    t.create_trackbar('C', 20, 3)
    # t.imshow(img2)  # 临时显式其他图片处理效果。如果需要持久修改，可以更改self.img的值
    cv2.waitKey()


def test_CoarseDropout(img=None):
    """ 做albumentations库里类功能的示例写法 """
    import albumentations as A

    if img is None:
        img = np.random.randint(0, 255, [500, 200], dtype='uint8')

    t = TrackbarTool.from_album(A.CoarseDropout, img=img, verbose=2)
    t.create_trackbar('max_holes', 50, 8)
    t.create_trackbar('max_height', 50, 8)
    t.create_trackbar('max_width', 50, 8)
    t.create_trackbar('fill_value', 255, 0)
    cv2.waitKey()
