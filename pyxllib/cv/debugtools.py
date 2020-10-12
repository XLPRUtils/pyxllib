#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/09/30 10:45

"""
这个文件默认不加载到cv里，需要使用的时候再导入

from pyxlib.cv.debugtools import *
"""

from pyxllib.cv import *


class ErodeTool(TrackbarTool):
    r""" 腐蚀 erode

    HoughLinesPTool('handwriting/C100001448573-002.png')
    """
    def __init__(self, img, winname='ErodeTool', flags=1):
        super().__init__(winname, img, flags)

        # 1 增加控件
        self.create_trackbar('size_x', 100, 15)
        self.create_trackbar('size_y', 100, 15)

        # 2 初始化执行一遍
        self.default_run(0)

    def default_run(self, x):
        """ 默认执行器
        """
        tt = TicToc()
        size = self['size_x'], self['size_y']
        element = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        dst = cv2.erode(self.img, element)
        self.imshow(dst)
        tt.toc(f'element_size={size}')


class HoughLinesPTool(TrackbarTool):
    r""" 霍夫线段检测

    HoughLinesPTool('handwriting/C100001448573-002.png')
    """
    def __init__(self, img, winname='HoughLinesPTool', flags=1):
        super().__init__(winname, img, flags)

        # 1 增加控件
        # 以像素为单位的距离精度
        self.create_trackbar('rho*10', 100, 10)
        # 以弧度为单位的角度精度
        self.create_trackbar('theta*pi/360', 10, 2)
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

        # 3 初始化执行一遍
        self.default_run(0)

    def default_run(self, x):
        """ 默认执行器
        """
        tt = TicToc()
        lines = cv2.HoughLinesP(self.binary_img,
                                self['rho*10'] / 10, self['theta*pi/360'] * math.pi / 360,
                                self['threshold'], self['minLineLength'], self['maxLineGap'])
        if lines is None: lines = np.array([])
        # 处理用二值化，显示用原图
        self.imshow(CvPlot.lines(self.img, lines))
        tt.toc(f'x={x} lines.shape={lines.squeeze().shape}')
