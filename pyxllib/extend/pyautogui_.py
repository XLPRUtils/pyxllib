#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/21 15:59


import subprocess


try:
    import pyautogui
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'pyautogui'])
    import pyautogui


class AutoGui:
    """ 跟GUI有关的操作自动化功能
    """
    def __init__(self, figspath, grayscale=False, confidence=None):
        """
        :param figspath: 图片素材所在目录
        :param grayscalse: 匹配时是否仅考虑灰度图
        :param confidence: 匹配时允许的误差，使用该功能需要OpenCV库的支持
            confidence参数暂未实装
        """
        self.figspath = str(figspath)
        self.grayscale = grayscale
        self.confidence = confidence

    def find(self, name, grayscale=None):
        """
        :param name: 判断当前屏幕中，是否存在图库中名为name的图
        :param grayscale: 这里如果设置了，会覆盖类中的默认设置
        :return: 如果存在，返回其中心坐标，否则返回None
        """
        grayscale = self.grayscale if grayscale is None else grayscale
        return pyautogui.locateCenterOnScreen(os.path.join(self.figspath, name),
                                              grayscale=grayscale)

    def move_to(self, name, grayscale=None):
        """找到则返回坐标，否则返回None"""
        pos = self.find(name, grayscale)
        if pos: pyautogui.moveTo(*pos)
        return pos

    def click(self, name, grayscale=None, wait=True, back=False):
        """找到则返回坐标，否则返回None
        :param wait: 是否等待图标出现运行成功后再继续
        """
        pos0 = pyautogui.position()
        while True:
            if isinstance(name, tuple) and len(name) == 2:  # 直接输入坐标
                pos = name
            else:
                pos = self.find(name, grayscale)
            if pos: pyautogui.click(*pos)
            if pos or not wait: break
        if back: pyautogui.moveTo(*pos0)  # 恢复鼠标原位置
        return pos

    def try_click(self, name, grayscale=None, back=False):
        return self.click(name, grayscale=None, wait=False, back=back)
