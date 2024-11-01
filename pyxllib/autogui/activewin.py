#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/10/29

import platform
import sys

import cv2
import psutil
import numpy as np

if sys.platform == 'win32':
    import win32gui
    import win32process
    # from pywinauto import Desktop
    import uiautomation

from pyxllib.prog.cachetools import xlcache
from pyxllib.cv.xlcvlib import CvImg
from pyxlpr.ai.clientlib import XlAiClient

# 根据平台扩展相应的类
if platform.system().lower() == "darwin":
    from mss.darwin import MSS as OriginalMSS
elif platform.system().lower() == "linux":
    from mss.linux import MSS as OriginalMSS
elif platform.system().lower() == "windows":
    from mss.windows import MSS as OriginalMSS
else:
    raise RuntimeError("Unsupported platform")


def adjust_monitors(monitors, scale_factor):
    # 调整所有物理显示器的尺寸和位置
    for i in range(1, len(monitors)):
        monitors[i]['width'] = int(monitors[i]['width'] / scale_factor)
        monitors[i]['height'] = int(monitors[i]['height'] / scale_factor)
        monitors[i]['top'] = int(monitors[i]['top'] / scale_factor)
        monitors[i]['left'] = int(monitors[i]['left'] / scale_factor)

    # 计算虚拟桌面的边界
    left_bound = min(monitor['left'] for monitor in monitors[1:])
    top_bound = min(monitor['top'] for monitor in monitors[1:])
    right_bound = max(monitor['left'] + monitor['width'] for monitor in monitors[1:])
    bottom_bound = max(monitor['top'] + monitor['height'] for monitor in monitors[1:])

    # 设置虚拟桌面的尺寸和位置
    monitors[0]['width'] = right_bound - left_bound
    monitors[0]['height'] = bottom_bound - top_bound
    monitors[0]['top'] = top_bound
    monitors[0]['left'] = left_bound

    return monitors


class ActiveWindowCapture(OriginalMSS):
    """
    基于`MSS`库的一个扩展，用于实现跨平台的多屏幕截图，并添加了以下独特功能：
    1. 实时窗口捕获：能够获取当前激活的窗口信息，并关联到对应的进程对象。
        这一特性使得该类在需要实时监控、分析特定应用窗口时尤为实用。
    2. 缓存机制：该类采用了缓存策略，通过`xlcache`装饰器缓存截取的屏幕和窗口数据，避免频繁重复捕获，提升性能。
        缓存机制同时确保在多窗口环境中，当前获取的窗口信息是实际的活动窗口，防止数据滞后或错位。
    3. 多屏幕拼接与坐标调整：支持多屏幕拼接功能，能够调整不同显示器的坐标和尺寸，以生成全屏拼接的截图画布。
        对于分辨率缩放和多屏偏移，也可以进行自适应调整。
    4. 标记窗口功能：在截屏时，可以选择性地标记当前激活窗口的位置，支持自定义标记的颜色、厚度和偏移量，便于在截图中直观地显示当前活跃窗口。

    适合需要频繁截图、实时监控活动窗口和对应进程的场景，尤其在多屏和分辨率适配方面提供了便捷的解决方案。
    """

    def __1_动态成员变量(self):
        pass

    def __init__(self, *, scale=1):
        """
        :param scale: 可适当兼容主屏幕做了缩放处理的情景，但太复杂的情况下也不一定对
        """
        super().__init__()
        self.scale = scale

    @xlcache(property=True)
    def monitors2(self):
        """ 修正后的各个屏幕坐标 """
        monitors = self.monitors
        if self.scale != 1:
            monitors = adjust_monitors(monitors, self.scale)
        return monitors

    @xlcache(property=True)
    def window(self):
        """ 当前激活窗口 """
        win = uiautomation.GetForegroundControl()
        return win

    @xlcache(property=True)
    def process(self):
        """ 当前激活窗口 """
        hwnd = self.window.NativeWindowHandle
        thread_id, process_id = win32process.GetWindowThreadProcessId(hwnd)
        return psutil.Process(process_id)

    @xlcache(property=True)
    def xlapi(self):
        return XlAiClient()

    def __2_截图数据(self):
        pass

    @xlcache()
    def _capture_single_monitor(self, order):
        """ 缓存单个屏幕的截图 """
        if order == 0:
            # order=0时，拼接所有屏幕图像。虽然mss其实可以直接提供，但是这样就没有每个屏幕缓存的图片了，还是我自己处理一遍更好。

            # 计算坐标偏移量，将所有坐标调整至非负
            min_left = min(monitor['left'] for monitor in self.monitors2)
            min_top = min(monitor['top'] for monitor in self.monitors2)

            # 获取总画布大小（根据所有屏幕的范围计算）
            max_right = max(monitor['left'] + monitor['width'] for monitor in self.monitors2)
            max_bottom = max(monitor['top'] + monitor['height'] for monitor in self.monitors2)
            canvas_width = max_right - min_left
            canvas_height = max_bottom - min_top

            # 使用numpy创建空白画布 (BGR格式)
            img_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            # 获取并缓存每个子屏幕截图，按调整后的坐标拼接到总画布
            for i in range(1, len(self.monitors2)):
                monitor = self.monitors2[i]
                single_img = self._capture_single_monitor(i)  # 获取单屏截图 (cv2格式)
                # 将子屏幕截图粘贴到总画布的指定位置，应用坐标偏移
                x_offset = monitor['left'] - min_left
                y_offset = monitor['top'] - min_top
                img_canvas[y_offset:y_offset + single_img.shape[0],
                x_offset:x_offset + single_img.shape[1]] = single_img

            img_cv2 = img_canvas
        else:
            if order == -1:
                # 截取当前活动窗口的截图
                active_window = self.window
                rect = active_window.BoundingRectangle
                monitor = {
                    'left': int(rect.left),
                    'top': int(rect.top),
                    'width': int(rect.right - rect.left),
                    'height': int(rect.bottom - rect.top),
                }
                sct_img = self.grab(monitor)
            else:
                sct_img = self.grab(self.monitors2[order])

            # 将截图的原始数据转换成numpy数组
            img_np = np.frombuffer(sct_img.bgra, dtype=np.uint8)
            img_np = img_np.reshape((sct_img.height, sct_img.width, 4))  # 高、宽、4通道
            # 转换成BGR通道顺序
            img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

        return img_cv2

    def _mark_window_on_screenshot(self, screenshot, window_rect, order=0, mark_params=None):
        """
        在截图中标记窗口，并根据 order 参数自动调整坐标偏移。

        :param screenshot: 要标记的截图图像
        :param window_rect: 当前激活窗口的原始坐标 (left, top, right, bottom)
        :param order: 屏幕索引 (0 表示拼接所有屏幕，全图模式)
        :param mark_params: 标记参数字典，包含 color, base, thickness 等
        :return: 标记后的截图
        """
        # 1 设定默认标记参数，若传入 mark_params 则进行更新
        default_params = {
            "color": (0, 255, 0),  # 默认绿色
            "base": 10,  # 默认缩进 10 像素
            "thickness": 3  # 默认厚度 3 像素
        }
        mark_params = mark_params if isinstance(mark_params, dict) else None
        if mark_params:
            default_params.update(mark_params)

        color = default_params["color"]
        base = default_params["base"]
        thickness = default_params["thickness"]

        # 2 获取对应 order 的屏幕原点坐标进行偏差修正
        if order == 0:
            origin_x = self.monitors2[0]['left']
            origin_y = self.monitors2[0]['top']
        else:
            origin_x = self.monitors2[order]['left']
            origin_y = self.monitors2[order]['top']

        # 计算相对于指定屏幕或全图的窗口坐标
        adjusted_coords = (
            int(window_rect.left) - origin_x,
            int(window_rect.top) - origin_y,
            int(window_rect.right) - origin_x,
            int(window_rect.bottom) - origin_y,
        )

        # 3 绘制矩形标记
        cv2.rectangle(
            screenshot,
            (adjusted_coords[0] + base, adjusted_coords[1] + base),
            (adjusted_coords[2] - base, adjusted_coords[3] - base),
            color,  # 使用自定义颜色
            thickness  # 使用自定义厚度
        )

        return screenshot

    @xlcache()
    def screenshot(self, order=0, *, to_pil=False, mark_active_window=False):
        """ 截屏

        :param int order:
            -1, 截取当前激活窗口
            0，默认值，截取全部屏幕
            1，截取第1个屏幕
            2，截取第2个屏幕
            ...
        :param to_pil: 是否转换成 PIL 格式
        :param mark_active_window: 是否标记当前激活窗口，若为字典可设置颜色、厚度等参数
        :return: np矩阵或PIL图片，取决于to_cv2参数
        """
        assert order < len(self.monitors), '屏幕下标越界'
        img = self._capture_single_monitor(order)  # 调用子函数处理所有屏幕拼接

        # 若需要标记当前激活窗口
        if mark_active_window:
            img = self._mark_window_on_screenshot(img, self.window.BoundingRectangle, order,
                                                  mark_params=mark_active_window)

        if to_pil:
            img = CvImg.read(img).to_pil_image()  # 转成pil矩阵格式

        return img
