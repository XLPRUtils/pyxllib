#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2025/06/05

""" 这是一套基于labelme标注来进行 """

import copy
import sys
from collections import defaultdict
import json
import os
import time
import random
import re
import datetime
from collections import UserDict

from pyxllib.prog.lazyimport import lazy_import

try:
    from loguru import logger
except ModuleNotFoundError:
    logger = lazy_import('from loguru import logger')

try:
    import numpy as np
except ModuleNotFoundError:
    np = lazy_import('numpy')

try:
    import win32com
    import win32gui
except ModuleNotFoundError:
    win32com = lazy_import('win32com', 'pywin32')
    win32gui = lazy_import('win32gui', 'pywin32')

try:
    import pyautogui
except ModuleNotFoundError:
    pyautogui = lazy_import('pyautogui')

try:
    import pyscreeze
except ModuleNotFoundError:
    pyscreeze = lazy_import('pyscreeze')

from pyxllib.prog.newbie import first_nonnone, round_int
from pyxllib.prog.pupil import xlwait, DictTool, run_once
from pyxllib.prog.filelock import get_autogui_lock
from pyxllib.algo.geo import ComputeIou, ltrb2xywh, xywh2ltrb
from pyxllib.file.specialist import XlPath
from pyxllib.cv.expert import xlcv, xlpil
from pyxlpr.data.labelme import LabelmeDict
from pyxllib.autogui.uiautolib import uia, find_ctrl, UiCtrlNode

from pyxlpr.ai.clientlib import XlAiClient

"""
桌面自动化操作，Anto GUI，简称An

相关概念，名称定义如下

从空间位置理解：
monitor：实际存在的若干物理屏幕
screen：多个monitor组合在一起的整个逻辑屏幕
region：screen中的某个局部区域
window: 特指某个软件所处的区域

从状态(时间)理解：
view: 同一个窗口位置上，可能有不同的各种功能、菜单变化，称为不同的视图
shape: view里的各个位置对象标记内容。上述所有类其实都是shape类
"""


@run_once()
def get_xlapi():
    xlapi = XlAiClient(auto_login=False, check=False)
    # logger.info(f"{os.getenv('XL_API_PRIU_TOKEN')}, {os.getenv('MAIN_WEBSITE')}")
    xlapi.login_priu(os.getenv('XL_API_PRIU_TOKEN'), os.getenv('MAIN_WEBSITE'))
    return xlapi


class ImageTools:

    @classmethod
    def pixel_distance(cls, pixel1, pixel2):
        """ 返回最大的像素值差，如果相同返回0 """
        return max([abs(x - y) for x, y in zip(pixel1, pixel2)])

    @classmethod
    def img_distance(cls, img1, img2, *, grayscale=None, color_tolerance=10):
        """ 返回距离：不相同像素占的百分比，相同返回0

        :param grayscale: 是否转换为灰度图进行比较，默认为None，即不转换
        :param color_tolerance: 每个像素允许的颜色差值。RGB情况计算的是3个值分别的差的和。
        """
        img1 = np.array(img1, dtype=int)
        img2 = np.array(img2, dtype=int)

        if grayscale:
            # 如果是灰度图，将图像转换为单通道
            if len(img1.shape) == 3:
                img1 = np.mean(img1, axis=2, dtype=int)
            if len(img2.shape) == 3:
                img2 = np.mean(img2, axis=2, dtype=int)
            cmp = np.array(abs(img1 - img2) > color_tolerance)
        else:
            # 彩色图情况，计算每个通道的差值
            cmp = np.array(abs(img1 - img2) > color_tolerance)
            # 只要有一个通道的差值超过阈值，就认为该像素不同
            cmp = np.any(cmp, axis=-1)

        return cmp.sum() / cmp.size

    @classmethod
    def base_find_img(cls, img, haystack=None, *, grayscale=None, confidence=None, sort_by_confidence=False):
        """ 根据预存的img数据，匹配出多个内容对应的所在的rect位置
        这函数不能取同名find_img，否则IDE不好自动识别跳转

        :param img: 要查找的目标的子图
        :param haystack: 整张大图
        :param grayscale: 是否使用灰度图像进行匹配
        :param confidence: 匹配置信度阈值
        :param sort_by_confidence: 是否按照置信度排序返回结果
        :return: 会返回多个匹配结果的坐标
        """
        # 若不需要按置信度排序，使用原始pyautogui方法
        if not sort_by_confidence:
            return cls._find_with_pyautogui(img, haystack, grayscale, confidence)
        else:
            return cls._find_with_opencv(img, haystack, grayscale, confidence)

    @classmethod
    def _find_with_pyautogui(cls, img, haystack, grayscale, confidence):
        """使用pyautogui原始方法查找匹配"""
        try:
            boxes = pyautogui.locateAll(img, haystack, grayscale=grayscale, confidence=confidence)

            try:
                boxes = list(boxes)
            except pyscreeze.ImageNotFoundException:
                return []

            # 过滤掉重叠超过一半面积的框
            rects = ComputeIou.nms_xywh(boxes)

            # rects里会有numpy.int64类型的数据，需要做个转换
            rects2 = []
            for rect in rects:
                rects2.append([int(x) for x in rect])
            return rects2
        except pyautogui.ImageNotFoundException:
            # 捕获图像未找到异常，返回空列表
            return []

    @classmethod
    def _find_with_opencv(cls, img, haystack, grayscale, confidence):
        """使用OpenCV查找匹配并按置信度排序"""
        try:
            # 转换图像格式
            template, search_img = cls._prepare_images(img, haystack, grayscale)

            # 获取匹配结果及置信度
            matches_with_conf = cls._match_template_with_confidence(template, search_img, confidence)

            # 按置信度排序
            matches_with_conf.sort(key=lambda x: x[4], reverse=True)

            # 提取位置信息
            boxes = [(x, y, w, h) for x, y, w, h, _ in matches_with_conf]

            # 应用NMS
            if boxes:
                rects = ComputeIou.nms_xywh(boxes)

                # 转换为整数类型
                rects2 = []
                for rect in rects:
                    rects2.append([int(x) for x in rect])
                return rects2
            return []
        except Exception as e:
            # 捕获异常，返回空列表
            print(f"Error in OpenCV template matching: {e}")
            return []

    @classmethod
    def _prepare_images(cls, img, haystack, grayscale):
        """准备图像用于OpenCV模板匹配"""
        import cv2
        import numpy as np

        # 如果是文件路径，则读取图像
        if isinstance(img, str):
            template = cv2.imread(img, cv2.IMREAD_COLOR)
        elif isinstance(img, np.ndarray):
            template = img
        else:
            # 如果是PIL图像，转换为OpenCV格式
            template = np.array(img)
            template = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)

        if isinstance(haystack, str):
            search_img = cv2.imread(haystack, cv2.IMREAD_COLOR)
        elif isinstance(haystack, np.ndarray):
            search_img = haystack
        else:
            # 如果是PIL图像，转换为OpenCV格式
            search_img = np.array(haystack)
            search_img = cv2.cvtColor(search_img, cv2.COLOR_RGB2BGR)

        # 如果需要灰度图像
        if grayscale:
            if len(template.shape) > 2:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            if len(search_img.shape) > 2:
                search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)

        return template, search_img

    @classmethod
    def _match_template_with_confidence(cls, template, search_img, confidence):
        """使用OpenCV进行模板匹配并返回带置信度的结果"""
        import cv2
        import numpy as np

        # 确保置信度设置
        if confidence is None:
            confidence = 0.8

        # 执行模板匹配
        result = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
        w, h = template.shape[1], template.shape[0]

        # 找出所有超过阈值的点
        locations = np.where(result >= confidence)
        matches = []

        # 收集坐标和置信度
        for pt in zip(*locations[::-1]):  # 转换为(x, y)格式
            x, y = pt
            conf = float(result[y, x])  # 获取该位置的置信度值
            matches.append((x, y, w, h, conf))

        return matches


class _AnShapeBasic(UserDict):
    """ 基本你的初始化、保存等模块 """

    def __1_构建(self):
        pass

    def __init__(self, initialdata=None, parent=None):
        """
        :param initialdata: 这个字典基础key字段和含义如下
            text，原本label标注的纯文本信息
            xywh，相对parent所处的外接矩形坐标位置
            center，中心点坐标

            （使用图片初始化的时候才有以下字段）
            img，cv格式的图片数据（这是特指labelme中存储的静态数据）
            pixel，中心点的像素值

            shot，这是特指程序运行中实时截图获取的动态快照图片
        :param parent:
        """
        super().__init__(initialdata)
        self.parent = parent or None

    def update_data(self, *, shot=False):
        """ 更新衍生字段数据 """
        # 1 补充 points
        if 'xywh' in self.data and 'points' not in self.data:
            self.data['shape_type'] = 'rectangle'
            l, t, r, b = xywh2ltrb(self.data['xywh'])
            self.data['points'] = [[l, t], [r, b]]
        # 2 外接矩形 xywh
        elif 'points' in self.data:
            pts = self.data['points']
            shape_type = self.data['shape_type'] or 'rectangle'
            if shape_type in ('rectangle', 'polygon'):
                pts_array = np.array(pts)
                left = int(pts_array[:, 0].min())
                top = int(pts_array[:, 1].min())
                right = int(pts_array[:, 0].max())
                bottom = int(pts_array[:, 1].max())
                ltrb = [left, top, right, bottom]
            elif shape_type == 'circle':
                x, y = pts[0]
                r = ((x - pts[1][0]) ** 2 + (y - pts[1][1]) ** 2) ** 0.5
                ltrb = [round_int(v) for v in [x - r, y - r, x + r, y + r]]
            # 把ltrb转换为xywh
            self.data['xywh'] = ltrb2xywh(ltrb)

        # 3 中心点
        if 'xywh' in self.data:
            xywh = self['xywh']
            self.data['center'] = [xywh[0] + xywh[2] // 2, xywh[1] + xywh[3] // 2]

        # 4 图片数据
        if shot:
            self.data['img'] = self.shot()
        elif 'xywh' in self.data and self.parent and self.parent.get('img') is not None:
            ltrb = xywh2ltrb(self.data['xywh'])
            self.data['img'] = xlcv.get_sub(self.parent['img'], ltrb)

        # 5 像素数据
        if 'center' in self.data and self.parent and self.parent.get('img') is not None:
            # 更新pixel数据
            w, h = self.data['xywh'][2:]
            self.data['pixel'] = tuple(self.data['img'][h // 2, w // 2].tolist()[::-1])

    def move(self, dx, dy, *, shot=True):
        """ 移动当前位置 """
        if 'points' in self:
            # 调整points中所有点的坐标
            for point in self['points']:
                point[0] += dx
                point[1] += dy
        elif 'xywh' in self:
            # 调整xywh坐标
            self['xywh'][0] += dx
            self['xywh'][1] += dy

        # 移动后更新图像数据
        self.update_data(shot=shot)

    def read_lmshape(self, lmshape):
        """
        将labelme格式的shape字典改为该任务特有的字典结构
        """
        # 1 解析原label为字典
        # 如果是来自普通的label，会把原文本自动转换为text字段。如果是来自xllabelme，则一般默认就是字典值，带text字段。
        if isinstance(lmshape['label'], str):
            anshape = DictTool.json_loads(lmshape['label'], 'text')
        else:
            anshape = lmshape['label']
        anshape.update(DictTool.sub(lmshape, ['label']))  # 因为原本的label被展开成字典了，这里要删掉label

        # 2 更新图像相关数据
        self.update(anshape)
        self.update_data(shot=False)

    def __2_转换与保存(self):
        pass

    def convert_to_view(self, mode='inner', *, shot=False):
        """ 将这个shape升级为一个view对象

        :param mode: 初始化shapes所用的模式
            inner，默认，将parent中几何关系被self包含的都归到子shapes中
            empty, 空shapes
            ocr, 使用ocr来生成初始的shapes
        """
        view = AnView(parent=self.parent)
        view.data = self.data
        view.update_data(shot=shot)

        if mode == 'inner':
            view.add_inner_shapes()
        elif mode == 'ocr':
            view.add_ocr_shapes()

        return view

    @classmethod
    def raw_save_image(cls, img, region_folder, view_name=None, timetag=None):
        timetag_ = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        if not view_name:  # 未输入则用时间戳代替
            view_name = timetag_
        elif timetag:  # 如果指定了时间戳参数，则强制加上时间戳前缀
            view_name = timetag_ + view_name

        impath = XlPath(region_folder) / f'{view_name}.jpg'
        os.makedirs(os.path.dirname(impath), exist_ok=True)
        xlcv.write(img, impath)

    def save_image(self, region_folder, view_name=None, timetag=None):
        """ 保存当前快照图片 """
        self.raw_save_image(self.shot(), region_folder, view_name, timetag)

    @classmethod
    def _save_view(cls, view, region_folder, view_name=None, timetag=None):
        timetag_ = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        if not view_name:  # 未输入则用时间戳代替
            view_name = timetag_
        elif timetag:  # 如果指定了时间戳参数，则强制加上时间戳前缀
            view_name = timetag_ + view_name

        impath = XlPath(region_folder) / f'{view_name}.jpg'
        view.save_labelme_file(impath)

    def save_view(self, region_folder, view_name=None, timetag=None):
        """ 保存当前视图（即保存图片和对应的标注数据） """
        view = self if isinstance(self, AnView) else self.convert_to_view('ocr', shot=True)
        self._save_view(view, region_folder, view_name, timetag)

    def __3_基础功能(self):
        pass

    def get_parent_argv(self, arg_name, cur_value=None):
        """ 查找arg_name的配置值
        从当前节点往父节点找，找到第1个非None值作为配置
        """
        cur_shape = self
        while cur_value is None:
            cur_value = cur_shape.get(arg_name)
            if cur_value is not None:
                break
            elif cur_shape.parent is None:
                break
            else:
                cur_shape = cur_shape.parent
        return cur_value

    def get_abs_point(self, point):
        """
        将相对坐标转换为绝对坐标

        :param point: 相对于当前节点的坐标，默认为None（使用自身中心点）
        :return: 绝对坐标 [x, y]
        """
        point = point.copy()

        # 向上遍历所有父节点，累加偏移量
        current = self
        while current.parent and current.parent.get('xywh'):
            # 父节点的xywh格式为[x, y, width, height]
            x, y = current.parent['xywh'][:2]
            # 将父节点的偏移量添加到绝对坐标
            point[0] += x
            point[1] += y
            # 移动到上一级父节点
            current = current.parent

        return point

    def total_name(self):
        """ 按父节点 text1/text2 的形式组织的完整text内容 """
        names = []
        cur = self
        while cur:
            names.append(cur.get('text', ''))
            cur = cur.parent
        return '/'.join(names[::-1])

    def __3_获取图像(self):
        pass

    def shot(self):
        """ 截取当前的快照画面 """
        # 计算绝对坐标下的xywh
        x, y = self.get_abs_point(self['xywh'][:2])  # 左上角相对坐标转绝对坐标
        w, h = self['xywh'][2:]  # 宽度和高度
        # logger.info(f'{x} {y} {w} {h}')
        pil_img = pyautogui.screenshot(region=[x, y, w, h])
        return xlpil.to_cv2_image(pil_img)

    def get_pixel(self):
        """ 获得当前中心点的像素值

         官方pyautogui.pixel版本，在windows有时候会出bug
         OSError: windll.user32.ReleaseDC failed : return 0
        """
        w, h = self['xywh'][2:]
        shot = self.shot()
        return tuple(shot[h // 2, w // 2].tolist()[::-1])


class _AnShapePupil(_AnShapeBasic):
    """ 基础操作功能 """

    def __1_ocr识别(self):
        pass

    def ocr_text(self):
        text = get_xlapi().rec_singleline(self.shot())
        return text

    def ocr_value(self):
        vals = self.ocr_values()
        return vals[0] if vals else 0

    def ocr_values(self):
        def parse_val(v):
            return float(v) if '.' in v else int(v)

        text = self.ocr_text()
        vals = re.findall(r'\d+', text) or []
        vals = [parse_val(v) for v in vals]
        return vals

    def __2_基础操作(self):
        pass

    def move_to(self, *, random_bias=None):
        """ 移动鼠标到该形状的中心位置，支持随机偏移

        :param int random_bias: 允许在一定范围内随机偏移目标坐标
        """
        # 0 检索配置参数
        random_bias = self.get_parent_argv('random_bias', random_bias) or 0

        # 1 计算出相对根节点的目标point位置
        point = self.get_abs_point(self['center'])

        # 2 添加随机偏移
        if random_bias:
            point[0] += random.randint(-random_bias, random_bias)
            point[1] += random.randint(-random_bias, random_bias)

        # 3 执行鼠标移动
        with get_autogui_lock():  # 全局ui锁，避免跟微信同时操作等冲突问题
            pyautogui.moveTo(*point)
        return point

    def click(self, x_bias=0, y_bias=0, *, random_bias=None, back=None, wait_change=None, wait_seconds=None):
        """ 点击这个shape

        :param x_bias: 原本中心点击位置，增设一个偏移量
        :param y_bias: y轴偏移量
        :param int random_bias: 允许在一定范围内随机点击点
        :param bool back: 是否点击后鼠标移回原坐标
        :param wait_change: 点击后，等待画面产生变化，才结束函数
            注意这个只是比较局部区域图片变化，不是全图的变化
        :param wait_seconds: 点击后要继续等待若干秒再退出函数
        """
        # 0 检索配置参数
        random_bias = self.get_parent_argv('random_bias', random_bias) or 0
        back = self.get_parent_argv('back', back) or False
        wait_change = self.get_parent_argv('wait_change', wait_change) or False
        wait_seconds = self.get_parent_argv('wait_seconds', wait_seconds) or 0

        # 1 计算出相对根节点的目标point位置
        point = self.get_abs_point(self['center'])
        if x_bias or y_bias:
            point[0] += x_bias
            point[1] += y_bias

        # 2 添加随机偏移
        if random_bias:
            point[0] += random.randint(-random_bias, random_bias)
            point[1] += random.randint(-random_bias, random_bias)

        # 3 鼠标移动
        with get_autogui_lock():
            origin_point = pyautogui.position()
            if wait_change:
                before_click_shot = self.shot()
            pyautogui.click(*point)
            if back:
                pyautogui.moveTo(*origin_point)  # 恢复鼠标原位置

        # 4 等待
        if wait_change:
            color_tolerance = self.get_parent_argv('color_tolerance', None) or 10
            confidence = self.get_parent_argv('confidence', None) or 0.95
            func = lambda: ImageTools.img_distance(before_click_shot, self.shot(),
                                                   color_tolerance=color_tolerance) > confidence
            xlwait(func)
        if wait_seconds:
            time.sleep(wait_seconds)

        return point

    def drag_to(self, percent=50, direction='up', duration=None, **kwargs):
        """ 在当前shape里拖拽

        :param int percent: 拖拽比例，范围0-100，表示滚动区域的百分比
        :param str direction: 滚动方向，可选 'up', 'down', 'left', 'right'
        :param float duration: 拖拽动作持续时间(秒)
        """
        # 0 检索配置参数
        drag_duration = self.get_parent_argv('drag_duration', duration) or 1

        # 1 计算起点和终点位置
        x, y = self.get_abs_point(self['center'])
        width, height = self['xywh'][2:]

        # 限制百分比在0-100之间
        percent = max(0, min(100, percent))
        offset = percent / 2  # 从中心点偏移的百分比

        # 根据方向和百分比计算起点和终点
        if direction == 'up':
            # 从下往上拖
            start_point = [x, y + int(height * offset / 100)]
            end_point = [x, y - int(height * offset / 100)]
        elif direction == 'down':
            # 从上往下拖
            start_point = [x, y - int(height * offset / 100)]
            end_point = [x, y + int(height * offset / 100)]
        elif direction == 'left':
            # 从右往左拖
            start_point = [x + int(width * offset / 100), y]
            end_point = [x - int(width * offset / 100), y]
        elif direction == 'right':
            # 从左往右拖
            start_point = [x - int(width * offset / 100), y]
            end_point = [x + int(width * offset / 100), y]
        else:
            raise ValueError("方向必须是 'up', 'down', 'left', 'right' 之一")

        # 2 执行拖拽操作
        with get_autogui_lock():
            pyautogui.moveTo(*start_point)
            pyautogui.dragTo(*end_point, drag_duration, **kwargs)

        # 本来有想过返回拖拽前后图片是否有变化，但是这个会较影响性能，还是另外设计更合理


class AnShape(_AnShapePupil):
    """ 高级交互功能 """

    def __1_图像类(self):
        pass

    def find_img(self,
                 dst=None,
                 *,
                 part=None,
                 drag=0,
                 drag_percent=50,
                 drag_direction='up',
                 drag_duration=None,
                 color_tolerance=None,
                 grayscale=None,
                 confidence=None,
                 sort_by_confidence=False,
                 ):
        """ 查找图像匹配

        :param dst: 匹配目标
            None，默认自匹配，使用self['img']
        :param part:
            False, 全匹配模式，比如图片就是指整张图匹配，文本指整张图的文本
            True, 局部匹配模式，比如图片是指局部匹配到图片，文本指局部文本行匹配
            None, 根据上下文智能判断类型。
        :param int drag: 是否支持在该区域滚动检索，以及支持的拖拽上限次数
        :param color_tolerance: 每个像素允许的差值
            注意以下3个图像配参数，不要在函数里设默认值，函数里优先级是高于父节点设置参数的
            默认值只能在代码里get_parent_argv后再or默认值
        :param grayscale: 是否转成灰度图对比，默认不转。
        :param confidence: 置信度，距离。默认95%即要求95%区域的相同性。
        :return:
            全匹配模式返回None或self
            局部匹配返回匹配的shapes列表
        """
        # 1 匹配目标
        default_text = ''
        if dst is None:
            dst, part = self['img'], False
            default_text = self['text']
        elif isinstance(dst, str):  # XlPath不是str类型，利用这个特性可以输入图片路径用xlcv读取
            """ 如果输入字符串，表示这是一个相对当前self的view路径名 """
            dst = self[str]

        if isinstance(dst, AnShape):
            default_text = dst['text']
            img = dst['img']
            img_wh = dst['xywh'][:2]
            if part is None:
                part = self['xywh'][2:] != dst['xywh'][:2]
        else:
            # 否则当作输入自定义图片了，用xlcv读取
            img = xlcv.read(dst)
            img_wh = list(img.shape[1::-1])

        if part is None:
            # 根据wh判断part
            part = self['xywh'][2:] != img_wh

        # 配置参数
        color_tolerance = self.get_parent_argv('color_tolerance', color_tolerance) or 10
        grayscale = self.get_parent_argv('grayscale', grayscale) or False
        confidence = self.get_parent_argv('confidence', confidence) or 0.95

        # 2 完全匹配场景（该场景不支持drag参数）
        if not part:
            # 这种情况不用pyautogui的接口，用我自带的函数够了
            conf = 1 - ImageTools.img_distance(self.shot(), img, color_tolerance=color_tolerance)
            logger.info(f'{self.total_name()} {round(conf, 4)}')
            return self if conf > confidence else None

        # 3 局部匹配场景
        def find_subimg():
            # 单帧检索
            rects = ImageTools.base_find_img(img, self.shot(), grayscale=grayscale,
                                             confidence=confidence, sort_by_confidence=sort_by_confidence)
            # 把rects转成AnShape类型，每个rect都是[x, y, w, h]的list，不过其有numpy格式，要转成int类型
            shapes = []
            for rect in rects:
                sp = AnShape({'text': default_text, 'xywh': [int(x) for x in rect]}, parent=self)
                sp.update_data(shot=True)
                shapes.append(sp)
            return shapes

        # 有无drag都走这套逻辑
        k, shapes = 0, find_subimg()
        while not shapes and k < drag:
            self.drag_to(drag_percent, drag_direction, drag_duration)
            shapes = find_subimg()
            k += 1

        return shapes

    def wait_img(self, dst=None, *, limit=None, interval=None, **kwargs):
        """ 等到目标匹配图片出现 """
        interval = self.get_parent_argv('wait_interval', interval) or 1
        limit = self.get_parent_argv('limit', limit)
        return xlwait(lambda: self.find_img(dst, **kwargs), limit=limit, interval=interval)

    def waitleave_img(self, dst=None, *, limit=None, interval=None, **kwargs):
        """ 等到目标匹配图片离开（不再出现） """
        interval = self.get_parent_argv('wait_interval', interval) or 1
        limit = self.get_parent_argv('limit', limit)
        return xlwait(lambda: not self.find_img(dst, **kwargs), limit=limit, interval=interval)

    def __2_文本类(self):
        """ find_系列统一返回shapes匹配列表 """
        pass

    def find_text(self,
                  dst=None,
                  *,
                  part=True,
                  drag=0,
                  drag_percent=50,
                  drag_direction='up',
                  drag_duration=None,
                  ):
        """ 查找文本匹配

        :param dst: 匹配目标
            None，默认自匹配，使用self['img']
        :param part:
            False, 全匹配模式，比如图片就是指整张图匹配，文本指整张图的文本
            True, 局部匹配模式，比如图片是指局部匹配到图片，文本指局部文本行匹配
        :param int drag: 是否支持在该区域滚动检索，以及支持的拖拽上限次数
        :return:
            全匹配模式返回None或self
            局部匹配返回匹配的shapes列表
        """
        # 1 匹配目标
        if dst is None:
            pattern = self['text']
        elif isinstance(dst, AnShape):
            pattern = dst['text']
        else:
            pattern = dst

        # 2 全图匹配
        if not part:
            return self if re.search(pattern, self.ocr_text()) else None

        # 3 局部匹配场景
        def find_subtext():
            shapes = []
            sub_view = self.convert_to_view('ocr', shot=True)
            for sp in sub_view.shapes:
                if re.search(pattern, sp['text']):
                    shapes.append(sp)
            return shapes

        # 有无drag都走这套逻辑
        k, shapes = 0, find_subtext()
        while not shapes and k < drag:
            self.drag_to(drag_percent, drag_direction, drag_duration)
            shapes = find_subtext()
            k += 1

        return shapes

    def wait_text(self, dst=None, *, limit=None, interval=None, **kwargs):
        """ 等到目标匹配文本出现 """
        interval = self.get_parent_argv('wait_interval', interval) or 1
        return xlwait(lambda: self.find_text(dst, **kwargs), limit=limit, interval=interval)

    def waitleave_text(self, dst=None, *, limit=None, interval=None, **kwargs):
        """ 等到目标匹配文本离开（不再出现） """
        interval = self.get_parent_argv('wait_interval', interval) or 1
        return xlwait(lambda: not self.find_text(dst, **kwargs), limit=limit, interval=interval)

    def __3_查找点击功能(self):
        """ 一些常用情况的简化名称，使用起来更快捷 """

    @classmethod
    def _split_kwargs(self, kwargs):
        # 定义需要传递给 click 方法的参数
        click_params = ['x_bias', 'y_bias', 'random_bias', 'back', 'wait_change', 'wait_seconds']
        # 分离参数，click_kwargs 存储给 click 用的参数，find_img_kwargs 存储给 find_img 用的参数
        find_img_kwargs = {k: v for k, v in kwargs.items() if k not in click_params}
        click_kwargs = {k: v for k, v in kwargs.items() if k in click_params}
        return find_img_kwargs, click_kwargs

    @classmethod
    def _try_click(cls, shapes, **kwargs):
        if shapes:
            sp = shapes[0] if isinstance(shapes, list) else shapes
            sp.click(**kwargs)
            return sp

    def find_img_click(self, dst=None, **kwargs):
        find_img_kwargs, click_kwargs = self._split_kwargs(kwargs)
        shapes = self.find_img(dst, **find_img_kwargs)
        return self._try_click(shapes, **click_kwargs)

    def wait_img_click(self, dst=None, **kwargs):
        wait_img_kwargs, click_kwargs = self._split_kwargs(kwargs)
        shapes = self.wait_img(dst, **wait_img_kwargs)
        return self._try_click(shapes, **click_kwargs)

    def find_text_click(self, dst=None, **kwargs):
        find_text_kwargs, click_kwargs = self._split_kwargs(kwargs)
        shapes = self.find_text(dst, **find_text_kwargs)
        return self._try_click(shapes, **click_kwargs)

    def wait_text_click(self, dst=None, **kwargs):
        wait_text_kwargs, click_kwargs = self._split_kwargs(kwargs)
        shapes = self.wait_text(dst, **wait_text_kwargs)
        return self._try_click(shapes, **click_kwargs)

    def __4_其他高级功能(self):
        pass

    def wait_img_notchange(self, *, limit=None, interval=2, **kwargs):
        """ 一直等待到图片不再变化，以当前shot为原图，类似find_img的基础匹配 """
        interval = self.get_parent_argv('wait_interval', interval) or 1
        interval *= 2  # 这个等待可以久一点

        # 获取颜色容差参数，默认值为20
        color_tolerance = self.get_parent_argv('color_tolerance', kwargs.get('color_tolerance')) or 10
        # 获取置信度参数，默认值为0.05，表示允许5%的变化
        confidence = self.get_parent_argv('confidence', kwargs.get('confidence')) or 0.95

        # 初始截图
        prev_shot = self.shot()

        # 定义判断图片是否未变化的函数
        def check_not_change():
            nonlocal prev_shot
            current_shot = self.shot()
            # 计算图片距离
            conf = 1 - ImageTools.img_distance(prev_shot, current_shot, color_tolerance=color_tolerance)
            prev_shot = current_shot
            return conf >= confidence

        # 使用xlwait等待图片不再变化
        return xlwait(check_not_change, limit=limit, interval=interval)


class AnView(AnShape):

    def __1_构建(self):
        pass

    def __init__(self, initialdata=None, parent=None):
        """ 这个类基本都要用预设的几个特殊的init接口去初始化，实际使用中一般不直接用这个raw init

        :param initialdata: 原始UserDict的字典初始化数据
        """
        super().__init__(initialdata, parent)
        self['text'] = 'view'

        self.impath = None  # 如果是从文件读取来的数据，存储原始图片路径，在save重写回去的时候就可以缺省保存参数
        self.shapes = []  # 以列表的形式顺序存储的anshapes

    def read_lmshapes(self, lmshapes):
        self.shapes = []
        for lmsp in lmshapes:
            sp = AnShape(parent=self)
            sp.read_lmshape(lmsp)
            self.shapes.append(sp)

    @classmethod
    def init_from_labelme_json_file(cls, lmfile):
        """ 最常用的初始化方式，从一个labelme json标注文件来初始化 """
        lmfile = XlPath(lmfile)
        lmdict = lmfile.read_json()

        view = AnView()
        view.impath = lmfile.with_name(lmdict['imagePath'])
        if not view.impath.exists():
            view.impath = lmfile.with_suffix('.jpg')

        view['img'] = xlcv.read(view.impath)
        view.read_lmshapes(lmdict['shapes'])
        view['xywh'] = [0, 0, view['img'].shape[1], view['img'].shape[0]]
        view.update_data(shot=False)

        return view

    def move(self, dx, dy, *, shot=True):
        super().move(dx, dy, shot=shot)
        # 只要view移动坐标就行，子shape其实只是做一次图片更新
        for sp in self.shapes:
            sp.move(0, 0, shot=shot)

    def add_inner_shapes(self):
        # 1 找其最近的一个是AnView类型的父节点，如果没有则退出
        parent_view = self.parent
        dx, dy = 0, 0
        while parent_view is not None and not isinstance(parent_view, AnView):
            if parent_view.get('xywh'):
                dx += parent_view['xywh'][0]  # 累加x偏移量
                dy += parent_view['xywh'][1]  # 累加y偏移量
            parent_view = parent_view.parent
        if parent_view is None:
            return

        # 2 算出self相对parent_view的ltrb
        self_ltrb = xywh2ltrb(self['xywh'])
        self_ltrb[0] += dx  # 加上父视图的x偏移量
        self_ltrb[1] += dy  # 加上父视图的y偏移量
        self_ltrb[2] += dx
        self_ltrb[3] += dy

        # dx += self['xywh'][0]
        # dy += self['xywh'][1]

        # 3 遍历parent_view.shapes，把符合inner规则的shapes添加进来
        for sp in parent_view.shapes:
            sp_ltrb = xywh2ltrb(sp['xywh'])
            if (sp_ltrb[0] >= self_ltrb[0] and sp_ltrb[1] >= self_ltrb[1] and
                    sp_ltrb[2] <= self_ltrb[2] and sp_ltrb[3] <= self_ltrb[3]):
                new_sp = AnShape({'text': sp['text']}, parent=self)
                new_sp['xywh'] = [sp['xywh'][0] - dx - self['xywh'][0],
                                  sp['xywh'][1] - dy - self['xywh'][1],
                                  sp['xywh'][2],
                                  sp['xywh'][3]]
                new_sp.update_data(shot=True)
                self.shapes.append(new_sp)

    def add_ocr_shapes(self):
        xlapi = get_xlapi()
        lmdict = xlapi.common_ocr(self['img'])
        self.read_lmshapes(lmdict['shapes'])

    @classmethod
    def init_from_image(cls, image, ocr=True):
        """ 从一张图片初始化
        这里默认会调用xlapi来生成labelme标注文件
        """
        # 1 读取图片
        view = AnView()
        view['img'] = xlcv.read(image)
        if isinstance(image, (str, XlPath)):
            view.impath = image
        view['xywh'] = [0, 0, view['img'].shape[1], view['img'].shape[0]]
        view.update_data(shot=False)

        # 2 调用ocr识别
        if ocr:
            view.add_ocr_shapes()

        return view

    def init_from_shot(self):
        """ 已有xywh等形状参数的情况下，可以用这个从shot初始化 """
        self['img'] = self.shot()
        xlapi = get_xlapi()
        lmdict = xlapi.common_ocr(self['img'])
        self.read_lmshapes(lmdict['shapes'])

    def save_view(self, region_folder, view_name=None, timetag=None):
        self._save_view(self, region_folder, view_name, timetag)

    def __2_检索功能(self):
        pass

    def find_shapes_by_text(self, text, limit=None):
        """ 通过text内容来匹配检索shapes

        :param text: 要匹配的文本内容
        :param limit: 限制识别的数量上限
        """
        shapes = []
        for sp in self.shapes:
            if sp['text'] == text:
                shapes.append(sp)
                # 若达到数量限制则提前终止循环
                if limit is not None and len(shapes) >= limit:
                    break
        return shapes

    def find_shape_by_text(self, text):
        shapes = self.find_shapes_by_text(text, limit=1)
        return shapes[0] if shapes else None

    def loc(self, item):
        if isinstance(item, int):
            return self.shapes[item]
        elif isinstance(item, str):
            return self.find_shape_by_text(item)
        elif isinstance(item, slice):
            # 处理slice对象，返回切片后的子列表
            start, stop, step = item.indices(len(self.shapes))
            return [self.shapes[i] for i in range(start, stop, step)]
        else:
            raise TypeError

    def search(self, pattern, flags=0):
        """ 正则的search语法匹配，在shapes的sp['text']查找返回第一个的sp """
        for sp in self.shapes:
            if re.search(pattern, sp['text'], flags=flags):
                return sp

    def __getitem__(self, item):
        """
        :return:
            整数访问shapes[i]，更准确的说，是支持slice模式
            字符串则是返回第一个匹配的shape

        注意这个函数并没有__setitem__版本，如果有需要修改数据，请直接修改self.shapes
        """
        if item in self.data:
            return self.data[item]
        else:
            return self.loc(item)

    def __x_保存(self):
        pass

    def save_labelme_file(self, impath=None, save_image=True):
        """ 将数据保存为labelme标注文件

        anshape中有存储了原始的lmshape信息。

        :param impath: 要保存的路径，只要输入图片路径就行，json路径是同位置同名的，只有后缀区别
            如果已经有 self.impath ，这里可以不输入
        :param save_image: 是否覆盖原有的图片文件
            todo 注意，如果原本没有图片，这里又不进行覆盖，生成lmdict时候会报错
        :return:
        """
        # 1 读取
        impath = impath or self.impath
        if impath is None:
            return
        impath = XlPath(impath)
        jsonpath = impath.with_suffix('.json')
        os.makedirs(os.path.dirname(jsonpath), exist_ok=True)

        # 2 生成格式
        if save_image:
            xlcv.write(self['img'], impath)  # 图片有概率也发生的变动
        lmdict = LabelmeDict.gen_data(impath)
        for sp in self.shapes:
            DictTool.isub(sp, ['img'])
            shape = LabelmeDict.gen_shape(json.dumps(sp.data, ensure_ascii=False, default=str),
                                          sp['points'], sp['shape_type'],
                                          group_id=sp.get('group_id'), flags=sp.get('flags', {}))
            lmdict['shapes'].append(shape)

        # 3 保存
        jsonpath.write_json(lmdict, indent=2)

    def create_sub_view(self, sub_view_loc, anchor_shape_loc=None, det_shape=None):
        """ 在当前view生成一个子view对象

        注意这个函数跟convert_to_view非常像，
        主要区别在多了锚点shape，这个sub_view位置可以根据锚点进行偏移，从而实现滚动窗口中元素的动态定位

        :param sub_view_loc: 目标shape
        :param anchor_shape_loc: 锚点shape
        :param det_shape: 检测到的动态位置sp，用来和anchor_shape对齐后，生成实际动态的sub_view整个区域
        """
        # 1 获取原本静态位置的sub_view，这步基本等价于 convert_to_view
        sub_view = self[sub_view_loc].convert_to_view()
        if anchor_shape_loc is None:
            return sub_view

        # 2 计算偏移量：检测到的动态位置与静态锚点位置的差值，这里仅以左上角为锚点，后续可以扩展center等形式的锚点
        anchor_shape = sub_view[anchor_shape_loc]
        dx = det_shape['xywh'][0] - anchor_shape['xywh'][0] - sub_view['xywh'][0]
        dy = det_shape['xywh'][1] - anchor_shape['xywh'][1] - sub_view['xywh'][1]
        sub_view.move(dx, dy)

        return sub_view


class AnRegion(AnShape):

    def __init__(self,
                 folder=None,
                 *,
                 read_views=False,
                 initialdata=None,
                 parent=None,
                 ):
        """
        :param folder: region标注数据所在文件夹
            有原图jpg、png和对应的json标注数据
            允许不输入，即没有目录的情况
        :param read_views: 是否直接读取目录下的views标注数据
        """
        super().__init__(initialdata, parent)
        self.folder = None if folder is None else XlPath(folder)
        self.views = {}  # views由于是文件存储，是不会重名的，所以直接用字典存储

        if read_views:  # todo 支持惰性加载？使用到对应view的时候才读取？不然以后标注文件特别多，初始化内存不爆炸？
            self.read_views()

    def _read_view(self, json_file):
        view_name = json_file.relative_to(self.folder).as_posix()[:-5]  # 注意 要去掉后缀.json
        view = AnView.init_from_labelme_json_file(json_file)
        view['text'] = view_name
        view.parent = self
        self.views[view_name] = view
        return view

    def read_views(self):
        if not (self.folder and self.folder.exists()):
            return

        self.views = {}
        for json_file in self.folder.rglob_files('*.json'):
            self._read_view(json_file)

    def loc_view(self, item):
        self._read_view(self.folder / f'{item}.json')
        return self.views.get(item)

    def loc(self, item):
        # views的具体值，采用惰性加载机制，只有使用到时，才会读取文件
        if '/' in item:
            # 注意功能效果：os.path.split('a/b/c') -> ('a/b', 'c')
            # 注意还有种很特殊的：'a' -> ('', 'a')
            view_name, shape_name = os.path.split(item)
            self._read_view(self.folder / f'{view_name}.json')
            return self.views[view_name][shape_name]
        else:
            return self.loc_view(item)

    def __getitem__(self, item):
        if item in self.data:
            return self.data[item]
        else:
            return self.loc(item)


class AnWindow(AnRegion):
    """ 某个软件的窗口区域 """

    def __0_构建(self):
        pass

    def __init__(self, folder=None, **kwargs):
        super().__init__(folder, **kwargs)

        w, h = pyautogui.size()
        self['text'] = 'screen'
        self['xywh'] = [0, 0, w, h]  # 默认获取全屏幕

        self.ctrl = None

        self._xlapi = None  # ocr工具
        self._speaker = None  # 语音播报工具
        # todo 添加数据库db？

    def set_ctrl(self,
                 ctrl=None,  # 直接输入ctrl初始化
                 hwnd=None,  # 输入窗口句柄
                 class_name=None, name=None,  # 通过名称检索
                 **kwargs):

        """ 设置窗口控件 """
        # 1 定位控件
        if ctrl is not None:
            ctrl = ctrl
        elif hwnd is not None:
            ctrl = uia.ControlFromHandle(hwnd)
        else:
            ctrl = find_ctrl(class_name=class_name, name=name, **kwargs)

        # 2 配置
        self.ctrl = UiCtrlNode(ctrl)
        self.ctrl.activate()
        self['text'] = self.ctrl.text
        self['xywh'] = self.ctrl.xywh

    @property
    def xlapi(self):
        if self._xlapi is None:
            self._xlapi = get_xlapi()
        return self._xlapi

    @property
    def speaker(self):
        if self._speaker is None:
            self._speaker = win32com.client.Dispatch('SAPI.SpVoice')
        return self.speaker

    def speak_text(self, text):
        self.speaker.Speak(text)


if __name__ == '__main__':
    pass
