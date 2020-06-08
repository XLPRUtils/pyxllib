#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/06/06


import os
import re
import subprocess


try:
    import pyautogui
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'pyautogui'])
    import pyautogui

import pyscreeze


from pyxllib.debug.qiniu_ import get_etag
from pyxllib.debug.pathlib_ import Path


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
        return pyautogui.locateCenterOnScreen(file_path, grayscale=grayscale, confidence=confidence)

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
            if isinstance(name, tuple) and len(name) == 2:  # 直接输入坐标
                pos = name
            else:
                pos = self.find(name, grayscale, confidence)
            if pos: pyautogui.click(*pos)
            if pos or not wait: break
        if back: pyautogui.moveTo(*pos0)  # 恢复鼠标原位置
        return pos

    def try_click(self, name, grayscale=None, confidence=None, back=False):
        return self.click(name, grayscale=grayscale, confidence=confidence, wait=False, back=back)
