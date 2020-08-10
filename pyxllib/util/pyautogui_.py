#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/06/06


import os
import re
import subprocess
import time

try:
    import pyautogui
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'pyautogui'])
    import pyautogui

try:
    import keyboard
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'keyboard'])
    import keyboard

import pyscreeze

from pyxllib.debug.qiniu_ import get_etag
from pyxllib.debug.pathlib_ import Path
from pyxllib.debug.dprint import dprint


class AutoGui:
    """ 跟GUI有关的操作自动化功能

    每帧处理要0.15秒左右，仅适用于一般的办公、回合制手游，不适合fps类游戏外挂
    """
    def __init__(self, figspath, grayscale=False, confidence=0.999):
        """
        :param figspath: 图片素材所在目录
        :param grayscalse: 匹配时是否仅考虑灰度图
        :param confidence: 匹配时允许的误差，使用该功能需要OpenCV库的支持
            confidence参数暂未实装
        """
        self.figspath = str(figspath)
        self.grayscale = grayscale
        self.confidence = confidence
        self.use_opencv = pyscreeze.useOpenCV

    def find(self, name, grayscale=None, confidence=None):
        """
        :param name: 判断当前屏幕中，是否存在图库中名为name的图
        :param grayscale: 这里如果设置了，会覆盖类中的默认设置
        :return: 如果存在，返回其中心坐标，否则返回None
        """
        grayscale = self.grayscale if grayscale is None else grayscale
        confidence = self.confidence if confidence is None else confidence
        file_path = os.path.join(self.figspath, name)
        # 如果使用了cv2，是不能直接处理含中文的路径的；有的话，就要先拷贝到一个英文路径来处理
        # 出现这种情况，均速会由0.168秒变慢18%到0.198秒。
        if self.use_opencv and re.search(r'[\u4e00-\u9fa5，。；？（）【】、①-⑨]', file_path):
            p1 = Path(file_path)
            p2 = Path(get_etag(file_path), Path(file_path).suffix, root=Path.TEMP)
            p1.copy(p2, if_exists='ignore')  # 使用etag去重，出现相同etag则不用拷贝了
            file_path = p2.fullpath
            # print(file_path)
        pos = pyautogui.locateCenterOnScreen(file_path, grayscale=grayscale, confidence=confidence)
        # pyautogui.screenshot().save('debug.jpg')
        # dprint(name, pos)
        return pos

    def move_to(self, name, grayscale=None):
        """找到则返回坐标，否则返回None"""
        pos = self.find(name, grayscale)
        if pos: pyautogui.moveTo(*pos)
        return pos

    def click(self, name, grayscale=None, confidence=None, wait=True, back=False):
        """找到则返回坐标，否则返回None
        :param wait: 是否等待图标出现运行成功后再继续
        """
        pos0 = pyautogui.position()
        while True:
            if isinstance(name, (tuple, list)) and len(name) == 2:  # 直接输入坐标
                pos = name
            else:
                pos = self.find(name, grayscale, confidence)
            if pos: pyautogui.click(*pos)
            if pos or not wait: break
        if back: pyautogui.moveTo(*pos0)  # 恢复鼠标原位置
        return pos

    def try_click(self, name, grayscale=None, confidence=None, back=False):
        return self.click(name, grayscale=grayscale, confidence=confidence, wait=False, back=back)


class PosTran:
    """ 坐标位置变换

    应用场景： 原来在A窗口下的点p和区域r，
        在窗口位置、大小改成B后，p和r的坐标
    """
    def __init__(self, w1, w2):
        """
        :param w1: 原窗口位置 (x, y, w, h)
        :param w2: 新窗口位置
        """
        self.w1 = w1
        self.w2 = w2

    @classmethod
    def point2point(cls, w1, p1, w2):
        """ 窗口w1切到w2，原点p1变换到坐标p2
        """
        x1, y1, w1, h1 = w1
        x2, y2, w2, h2 = w2
        x = x2 + (p1[0] - x1) * w2 / w1
        y = y2 + (p1[1] - y1) * h2 / h1
        return round(x), round(y)

    def w1point(self, w2point):
        return self.point2point(self.w2, w2point, self.w1)

    def w2point(self, w1point):
        """旧窗口中的点坐标转回新窗口点坐标"""
        return self.point2point(self.w1, w1point, self.w2)

    @classmethod
    def dx2dx(cls, w1, dx1, w2):
        """ 宽度偏移量的变化
        """
        return round(dx1 * w2[2] / w1[2])

    def w1dx(self, dx2):
        return self.dx2dx(self.w2, dx2, self.w1)

    def w2dx(self, dx1):
        return self.dx2dx(self.w1, dx1, self.w2)

    @classmethod
    def dy2dy(cls, w1, dy1, w2):
        return round(dy1 * w2[3] / w1[3])

    def w1dy(self, dy2):
        return self.dy2dy(self.w2, dy2, self.w1)

    def w2dy(self, dy1):
        return self.dy2dy(self.w1, dy1, self.w2)

    @classmethod
    def region2region(cls, w1, r1, w2):
        """ 窗口w1切到w2，原区域r1变换到坐标r2
        """
        x, y = cls.point2point(w1, r1[:2], w2)
        w = round(r1[2] * w2[2] / w1[2])
        h = round(r1[3] * w2[3] / w1[3])
        return x, y, w, h

    def w1region(self, w2region):
        return self.region2region(self.w2, w2region, self.w1)

    def w2region(self, w1region):
        return self.region2region(self.w1, w1region, self.w2)


def lookup_mouse_position():
    """查看鼠标位置的工具
    """
    from keyboard import read_key

    left_top_point = None
    while True:
        k = read_key()
        if k == 'ctrl':
            # 定位当前鼠标坐标位置
            # 也可以用来定位区域时，先确定区域的左上角点
            left_top_point = pyautogui.position()
            print('坐标：', *left_top_point)
        elif k == 'alt':
            # 定位区域的右下角点，并输出区域
            p = pyautogui.position()
            print('区域(x y w h)：', left_top_point.x, left_top_point.y, p.x-left_top_point.x, p.y-left_top_point.y)
        elif k == 'esc':
            break
        # keyboard的监控太快了，需要暂停一下
        time.sleep(0.4)


def lookup_mouse_position2(w1, w2, reverse=False):
    """ 涉及到窗口位置、大小调整时的坐标计算
        当前在窗口w1的坐标，切换到w2时的坐标
    :param reverse: 当前窗口实际是变换后的w2，输出的时候需要按照 w1 -> w2 的格式展示
        代码实现的时候，其实只要把输出的内容对调即可

    该函数是lookup_mouse_postion的进阶版，两个函数还是先不做合并，便于理解维护
    """
    import keyboard
    postran = PosTran(w1, w2)

    left_top_point = None
    while True:
        k = keyboard.read_key()
        if k == 'ctrl':
            # 定位当前鼠标坐标位置
            # 也可以用来定位区域时，先确定区域的左上角点
            left_top_point = pyautogui.position()
            p1, p2 = left_top_point, postran.w2point(left_top_point)
            if reverse: p1, p2 = p2, p1
            print('坐标：', *p1, '-->', *p2)
        elif k == 'alt':
            # 定位区域的右下角点，并输出区域
            p = pyautogui.position()
            r1 = [left_top_point.x, left_top_point.y, p.x-left_top_point.x, p.y-left_top_point.y]
            r2 = postran.w2region(r1)
            if reverse: r1, r2 = r2, r1
            print('区域(x y w h)：', *r1, '-->', *r2)
        elif k == 'esc':
            break
        time.sleep(0.4)
