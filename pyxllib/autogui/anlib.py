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
from typing import Literal

from pyxllib.prog.lazyimport import lazy_import

try:
    from loguru import logger
except ModuleNotFoundError:
    logger = lazy_import('from loguru import logger')

try:
    from pydantic import BaseModel
except ModuleNotFoundError:
    BaseModel = lazy_import('from pydantic import BaseModel')

try:
    import cv2
except ModuleNotFoundError:
    cv2 = lazy_import('cv2')

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
from pyxllib.prog.specialist import XlBaseModel, resolve_params
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

详细文档：https://www.yuque.com/xlpr/pyxllib/anlib
"""


@run_once()
def get_xlapi():
    xlapi = XlAiClient(auto_login=False, check=False)
    xlapi.login_priu(os.getenv('XL_API_PRIU_TOKEN'), os.getenv('MAIN_WEBSITE'))
    return xlapi


class LocateParams(XlBaseModel):
    """ 图像定位与匹配相关的参数配置 """

    # 是否使用灰度模式进行匹配，默认为 False (彩色匹配)
    grayscale: bool = False

    # 匹配置信度阈值 (0.0 - 1.0)，值越高匹配要求越严格
    confidence: float = 0.9

    # 颜色容忍度 (0-255)，用于像素级对比时允许的 RGB 通道差值
    # 默认 10，允许轻微的渲染色差
    color_tolerance: int = 10

    # 在返回结果前，是否根据置信度分数进行降序排序
    # 若为 True，将强制使用 OpenCV 引擎进行匹配
    sort_by_confidence: bool = False


class ImageTools:
    """ 图像处理与匹配工具类 """

    @classmethod
    def pixel_distance(cls, pixel1, pixel2):
        """ 计算两个像素点的最大通道差值
        :return: max(abs(r1-r2), abs(g1-g2), abs(b1-b2))
        """
        return max([abs(x - y) for x, y in zip(pixel1, pixel2)])

    @classmethod
    @resolve_params(LocateParams, mode='pass')
    def img_distance(cls, img1, img2, **resolved_params):
        """ 计算两张图片的差异程度

        :param img1: 图片对象1 (PIL Image, numpy array, 或文件路径)
        :param img2: 图片对象2
        :param resolved_params: LocateParams 参数
            grayscale: 是否转灰度对比
            color_tolerance: 判定像素是否不同的阈值
        :return: 差异度 (0.0 - 1.0)，0 表示完全相同，1 表示完全不同
        """
        params: LocateParams = resolved_params['LocateParams']

        img1 = np.array(img1, dtype=int)
        img2 = np.array(img2, dtype=int)

        if params.grayscale:
            # 如果是灰度图比较，将图像转换为单通道
            if len(img1.shape) == 3:
                img1 = np.mean(img1, axis=2, dtype=int)
            if len(img2.shape) == 3:
                img2 = np.mean(img2, axis=2, dtype=int)

            # 灰度图直接比较数值差异
            cmp = np.array(abs(img1 - img2) > params.color_tolerance)
        else:
            # 彩色图情况，计算每个通道的差值
            cmp = np.array(abs(img1 - img2) > params.color_tolerance)
            # 只要有一个通道的差值超过阈值，就认为该像素不同
            cmp = np.any(cmp, axis=-1)

        return cmp.sum() / cmp.size

    @classmethod
    @resolve_params(LocateParams, mode='pass')
    def base_find_img(cls, img, haystack=None, **resolved_params):
        """ 在大图中查找目标小图的所有出现位置，并进行非极大值抑制(NMS)去重

        :param img: 目标子图 (Needle)
        :param haystack: 背景大图 (Haystack)，为 None 则默认为当前屏幕截图
        :param resolved_params: LocateParams 参数
            grayscale: 是否灰度匹配
            confidence: 置信度阈值
            sort_by_confidence: 是否按匹配度排序
        :return: 匹配结果列表，格式为 list[[x, y, w, h]]，坐标为整数
        """
        params: LocateParams = resolved_params['LocateParams']

        # 根据是否需要排序选择底层实现引擎
        if not params.sort_by_confidence:
            return cls._find_with_pyautogui(img, haystack, params)
        else:
            return cls._find_with_opencv(img, haystack, params)

    @classmethod
    def _find_with_pyautogui(cls, img, haystack, params: LocateParams):
        """ 使用 PyAutoGUI 引擎进行图像查找 """
        try:
            boxes = pyautogui.locateAll(img, haystack,
                                        grayscale=params.grayscale,
                                        confidence=params.confidence)

            try:
                boxes = list(boxes)
            except pyscreeze.ImageNotFoundException:
                return []

            # 过滤掉重叠超过一半面积的框 (NMS)
            rects = ComputeIou.nms_xywh(boxes)

            # 统一转换为 Python int 类型 (rects 可能包含 numpy.int64)
            return [[int(x) for x in rect] for rect in rects]
        except pyautogui.ImageNotFoundException:
            return []

    @classmethod
    def _find_with_opencv(cls, img, haystack, params: LocateParams):
        """ 使用 OpenCV 引擎进行图像查找，支持按置信度排序 """
        try:
            # 准备 OpenCV 格式的图像数据
            template, search_img = cls._prepare_images(img, haystack, params.grayscale)

            # 获取匹配结果及置信度
            matches_with_conf = cls._match_template_with_confidence(template, search_img, params.confidence)

            # 按置信度降序排序
            matches_with_conf.sort(key=lambda x: x[4], reverse=True)

            # 提取位置信息 (x, y, w, h)
            boxes = [(x, y, w, h) for x, y, w, h, _ in matches_with_conf]

            if boxes:
                # NMS 去重
                rects = ComputeIou.nms_xywh(boxes)
                return [[int(x) for x in rect] for rect in rects]
            return []
        except Exception as e:
            logger.error(f"Error in OpenCV template matching: {e}")
            return []

    @classmethod
    def _prepare_images(cls, img, haystack, grayscale):
        """ 将各种输入的图像源转换为 OpenCV 可处理的 numpy 数组格式 """
        # 处理 Template (子图)
        if isinstance(img, str):
            template = cv2.imread(img, cv2.IMREAD_COLOR)
        elif isinstance(img, np.ndarray):
            template = img
        else:
            # PIL Image 转 OpenCV (RGB -> BGR)
            template = np.array(img)
            template = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)

        # 处理 Haystack (大图)
        if isinstance(haystack, str):
            search_img = cv2.imread(haystack, cv2.IMREAD_COLOR)
        elif isinstance(haystack, np.ndarray):
            search_img = haystack
        else:
            search_img = np.array(haystack)
            search_img = cv2.cvtColor(search_img, cv2.COLOR_RGB2BGR)

        # 灰度转换处理
        if grayscale:
            if len(template.shape) > 2:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            if len(search_img.shape) > 2:
                search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)

        return template, search_img

    @classmethod
    def _match_template_with_confidence(cls, template, search_img, confidence):
        """ 执行 OpenCV 模板匹配并筛选结果 """
        # 使用相关系数归一化匹配 (TM_CCOEFF_NORMED)
        result = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
        w, h = template.shape[1], template.shape[0]

        # 筛选大于置信度阈值的位置
        locations = np.where(result >= confidence)
        matches = []

        # 收集结果 (OpenCV 返回坐标是 y, x)
        for pt in zip(*locations[::-1]):
            x, y = pt
            conf = float(result[y, x])
            matches.append((x, y, w, h, conf))

        return matches


class DragParams(XlBaseModel):
    """ 拖拽与滚动搜索参数

    用于在当前视图未找到目标时，尝试通过拖拽/滚动屏幕来寻找目标。
    """
    # 单次拖拽的幅度百分比 (0-100)
    # 基于当前 Shape 的宽高计算偏移量
    drag_percent: float = 50.0

    # 拖拽方向
    # 可选: 'up', 'down', 'left', 'right'
    # 注意：方向是指鼠标拖拽的方向，通常意味着内容会向反方向滚动
    drag_direction: Literal['up', 'down', 'left', 'right'] = 'up'

    # 拖拽/滚动的尝试次数
    # 默认为 0，即仅在当前画面查找，不进行任何拖拽尝试
    # 原参数名: drag
    drag_count: int = 0

    # 拖拽动作的持续时间 (秒)
    # 设置合理的持续时间可以避免操作过快导致滚动惯性过大或未被识别
    drag_duration: float = 1.0


class MouseParams(XlBaseModel):
    """ 鼠标交互行为参数

    用于控制 move_to, click 等操作的精准度、拟人化偏移以及操作后的反馈等待。
    """

    # 点击的目标名称（用于链式调用或代理点击）
    target: str | None = None

    # 目标位置 X 轴固定偏移量 (像素)
    # 正值向右，负值向左
    x_bias: int = 0

    # 目标位置 Y 轴固定偏移量 (像素)
    # 正值向下，负值向上
    y_bias: int = 0

    # 随机偏移半径 (像素)
    # 在目标点周围的正方形区域内随机偏移，用于模拟人工操作，规避反作弊检测
    random_bias: int = 0

    # 操作完成后，是否将鼠标移回操作前的原始位置
    # 适用于 click 点击后复位，或者 move_to 探测后复位
    # 原参数名: back
    move_back: bool = False

    # 操作后是否等待画面产生变化
    # 若为 True，会对比操作前后的截图，直到差异度超过 LocateParams.confidence 设定
    wait_change: bool = False

    # 操作后的硬性等待时间 (秒)
    # 即使 wait_change 满足了，也会继续 sleep 这么多秒
    wait_seconds: float = 0


class WaitParams(XlBaseModel):
    """ 等待与超时控制参数

    用于 wait_img, wait_text 等轮询查找函数。
    """

    # 总超时时间 (秒)
    # None 表示无限等待，直到目标出现
    # 原参数名: limit
    timeout: float | None = None

    # 轮询检测的间隔时间 (秒)
    # 决定了查找频率，值越小响应越快但 CPU 占用越高
    interval: float = 1.0


class _AnShapeBasic(UserDict):
    """ 形状基础类

    负责底层数据结构管理、图像/像素获取、坐标转换以及核心的参数解析逻辑。
    """

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
        :param parent: 父节点对象
            用于构建层级关系，实现参数继承 (get_parent_argv) 和 相对坐标计算。
        """
        super().__init__(initialdata)
        self.parent = parent or None

    def update_data(self, *, shot=False):
        """ 更新衍生字段数据

        根据核心的 xywh 或 points 数据，自动计算并更新 center, ltrb 等衍生数据。

        :param shot: 是否重新截取当前区域的屏幕快照
        """
        # 1. 补充 points (如果只有 xywh)
        if 'xywh' in self.data and 'points' not in self.data:
            self.data['shape_type'] = 'rectangle'
            l, t, r, b = xywh2ltrb(self.data['xywh'])
            self.data['points'] = [[l, t], [r, b]]

        # 2. 根据 points 反推外接矩形 xywh
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

            # 将 ltrb 转换为 xywh
            self.data['xywh'] = ltrb2xywh(ltrb)

        # 3. 更新中心点 center
        if 'xywh' in self.data:
            xywh = self['xywh']
            self.data['center'] = [xywh[0] + xywh[2] // 2, xywh[1] + xywh[3] // 2]

        # 4. 更新图片数据 (img)
        if shot:
            self.data['img'] = self.shot()
        elif 'xywh' in self.data and self.parent and self.parent.get('img') is not None:
            # 从父节点图片中裁剪
            ltrb = xywh2ltrb(self.data['xywh'])
            self.data['img'] = xlcv.get_sub(self.parent['img'], ltrb)

        # 5. 更新像素数据 (pixel)
        if 'center' in self.data and self.parent and self.parent.get('img') is not None:
            w, h = self.data['xywh'][2:]
            if w > 0 and h > 0:
                # 获取中心点的颜色值 (BGR 转 RGB 或保持原样，视 xlcv 实现而定，这里保持原逻辑倒序)
                self.data['pixel'] = tuple(self.data['img'][h // 2, w // 2].tolist()[::-1])

    def move(self, dx, dy, *, shot=True):
        """ 移动当前形状的位置

        :param dx: X轴偏移量
        :param dy: Y轴偏移量
        :param shot: 移动后是否更新快照
        """
        if 'points' in self:
            # 调整 points 中所有点的坐标
            for point in self['points']:
                point[0] += dx
                point[1] += dy
        elif 'xywh' in self:
            # 调整 xywh 坐标
            self['xywh'][0] += dx
            self['xywh'][1] += dy

        # 移动后更新图像及衍生数据
        self.update_data(shot=shot)

    def read_lmshape(self, lmshape):
        """ 解析 labelme 格式的 shape 字典 """
        # 1. 解析 label 字段 (可能是 JSON 字符串)
        # 如果是来自普通的label，会把原文本自动转换为text字段。如果是来自xllabelme，则一般默认就是字典值，带text字段。
        if isinstance(lmshape['label'], str):
            anshape = DictTool.json_loads(lmshape['label'], 'text')
        else:
            anshape = lmshape['label']

        # 2. 合并数据，移除原 label 字段
        anshape.update(DictTool.sub(lmshape, ['label']))

        # 3. 更新自身数据
        self.update(anshape)
        self.update_data(shot=False)

    def __2_转换与保存(self):
        pass

    def convert_to_view(self, mode='inner', *, shot=False):
        """ 将当前 Shape 升级为 View 对象 (包含子元素)

        :param mode: 初始化模式
            'inner': 包含几何上位于自身内部的子元素
            'ocr': 使用 OCR 识别内部文本生成子元素
            'empty': 空 View
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
        """ 保存当前形状的快照 """
        self.raw_save_image(self.shot(), region_folder, view_name, timetag)

    @classmethod
    def _save_view(cls, view, region_folder, view_name=None, timetag=None):
        timetag_ = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        if not view_name:
            view_name = timetag_
        elif timetag:
            view_name = timetag_ + view_name

        impath = XlPath(region_folder) / f'{view_name}.jpg'
        view.save_labelme_file(impath)

    def save_view(self, region_folder, view_name=None, timetag=None):
        """ 保存当前视图 (包含图片和标注数据) """
        # 如果自身不是 View，先转换为 OCR 模式的 View
        view = self if isinstance(self, AnView) else self.convert_to_view('ocr', shot=True)
        self._save_view(view, region_folder, view_name, timetag)

    def __3_基础功能(self):
        pass

    def get_parent_argv(self, arg_name, cur_value=None):
        """ 级联查找参数配置

        从当前节点开始，向上级 Parent 查找第一个非 None 的配置值。
        这是实现 "Parent 配置继承" 的基础。
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

    def resolve_model(self, model_instance: BaseModel) -> BaseModel:
        """ 解析 Pydantic 模型，融合继承逻辑

        策略优先级：
        1. 函数调用时显式传入的参数 (检查 model_fields_set)
        2. 父节点 (Parent) 的配置 (检查 get_parent_argv)
        3. 模型的默认值 (Model Default)

        :param model_instance: 由 @resolve_params 解析出的初步模型实例
        :return: 融合了继承逻辑后的新模型实例
        """
        # 复制模型以避免污染原始对象
        new_model = model_instance.model_copy()

        for field_name in model_instance.model_fields.keys():
            # 1. 如果用户显式传入了参数，直接使用用户的设定，跳过继承查找
            if field_name in model_instance.model_fields_set:
                continue

            # 2. 尝试从父节点继承配置
            # 注意：父节点的 key 必须与 model 的 field_name 一致才能被继承
            parent_val = self.get_parent_argv(field_name)

            # 3. 如果父节点有配置，则覆盖模型的默认值
            if parent_val is not None:
                setattr(new_model, field_name, parent_val)

        return new_model

    def get_abs_point(self, point=None):
        """ 将相对坐标转换为屏幕绝对坐标

        :param point: [x, y] 坐标点
            默认为 None，使用当前形状的中心点 (self['center'])。
            注意：这里的坐标系是相对于**父节点 (Parent) 左上角**的。
            例如：self['xywh'] = [10, 10, 50, 50]，意味着当前形状位于父节点 (10, 10) 的位置。
        :return: [abs_x, abs_y] 屏幕绝对坐标
        """
        if point is None:
            # 默认为自身中心点 (已经是在 Parent 坐标系下)
            point = self['center'].copy() if 'center' in self else [0, 0]
        else:
            # 确保不修改原列表
            point = list(point)

        # 向上遍历所有父节点，累加父节点的左上角偏移量
        current = self
        while current.parent and current.parent.get('xywh'):
            # 父节点的 xywh 格式为 [x, y, width, height]
            x, y = current.parent['xywh'][:2]
            # 将父节点的偏移量添加到绝对坐标
            point[0] += x
            point[1] += y
            # 移动到上一级父节点
            current = current.parent

        return point

    def get_abs_xywh(self):
        """ 获取当前形状在屏幕上的绝对坐标和尺寸 [abs_x, abs_y, w, h] """
        # get_abs_point 接收的是相对于 Parent 的坐标
        # 传入自身的 xy (即相对于 Parent 的偏移)，计算出的就是自身的绝对 Top-Left
        abs_top_left = self.get_abs_point(self['xywh'][:2])
        return [abs_top_left[0], abs_top_left[1], self['xywh'][2], self['xywh'][3]]

    def get_xywh_in(self, scope_shape=None):
        """ 获取当前形状相对于 scope_shape 坐标系的位置
        即：将 self 映射到 scope_shape 的局部坐标系中

        :param scope_shape: 参考系对象。
            - 如果为 None，则返回屏幕绝对坐标 (scope 为屏幕)
            - 如果为 AnShape 对象，则返回相对于该对象左上角的坐标
        :return: [x, y, w, h]
        """
        # 1. 获取自身的绝对坐标
        self_abs = self.get_abs_xywh()

        if scope_shape is None:
            return self_abs

        # 2. 获取参考系的绝对坐标 (只关心左上角)
        scope_abs = scope_shape.get_abs_xywh()

        # 3. 计算相对坐标
        # 向量减法：Self_Abs - Scope_Abs = Self_Rel
        rel_x = self_abs[0] - scope_abs[0]
        rel_y = self_abs[1] - scope_abs[1]

        return [rel_x, rel_y, self_abs[2], self_abs[3]]

    def total_name(self):
        """ 获取形状的全路径名称 (e.g., "Screen/Window/Button") """
        names = []
        cur = self
        while cur:
            names.append(cur.get('text', ''))
            cur = cur.parent
        return '/'.join(names[::-1])

    def __3_获取图像(self):
        pass

    def shot(self):
        """ 截取当前形状区域的屏幕快照 """
        x, y = self.get_abs_point(self['xywh'][:2])  # 左上角相对坐标转绝对坐标
        w, h = self['xywh'][2:]  # 宽度和高度
        # logger.info(f'{x} {y} {w} {h}')
        pil_img = pyautogui.screenshot(region=[x, y, w, h])
        return xlpil.to_cv2_image(pil_img)

    def get_pixel(self):
        """ 获取当前形状中心点的像素值 (RGB)

		 官方pyautogui.pixel版本，在windows有时候会出bug
         OSError: windll.user32.ReleaseDC failed : return 0
		"""
        w, h = self['xywh'][2:]
        shot = self.shot()
        # 获取中心点颜色，并处理通道顺序 (BGR -> RGB)
        # 注意：shot 是 cv2 image (BGR)，tolist()[::-1] 转换为 RGB
        return tuple(shot[h // 2, w // 2].tolist()[::-1])


class _AnShapePupil(_AnShapeBasic):
    """ 基础交互操作类

    包含 OCR 识别、鼠标移动、点击、拖拽等原子操作。
    所有操作均通过 @resolve_params 接收统一的参数模型。
    """

    # --- OCR 相关功能 ---

    def ocr_text(self):
        """ 识别当前区域的单行文本 """
        text = get_xlapi().rec_singleline(self.shot())
        return text

    def ocr_value(self):
        """ 识别并返回第一个数值 """
        vals = self.ocr_values()
        return vals[0] if vals else 0

    def ocr_values(self):
        """ 识别当前区域内的所有数值 """

        def parse_val(v):
            return float(v) if '.' in v else int(v)

        text = self.ocr_text()
        vals = re.findall(r'\d+(?:\.\d+)?', text) or []
        return [parse_val(v) for v in vals]

    def __2_基础操作(self):
        pass

    @resolve_params(MouseParams, mode='pass')
    def move_to(self, x_bias=None, y_bias=None, /, **resolved_params):
        """ 移动鼠标到该形状的中心位置

        :param resolved_params: 包含 MouseParams
            - x_bias, y_bias: 目标位置的固定修正
            - random_bias: 随机扰动范围
            - move_back: 移动过去后是否立刻移回原位 (类似"探一下"或悬停动作)
            - wait_seconds: 移动后的停留时间
        :return: 移动到的目标绝对坐标 [x, y]
        """
        # 1. 解析参数
        params: MouseParams = self.resolve_model(resolved_params['MouseParams'])
        params.update_valid(x_bias=x_bias, y_bias=y_bias)

        # 2. 计算目标位置 (含固定偏移)
        point = self.get_abs_point(self['center'])
        point[0] += params.x_bias
        point[1] += params.y_bias

        # 3. 应用随机偏移 (拟人化)
        if params.random_bias:
            point[0] += random.randint(-params.random_bias, params.random_bias)
            point[1] += random.randint(-params.random_bias, params.random_bias)

        # 4. 执行移动
        with get_autogui_lock():
            origin_point = pyautogui.position()
            pyautogui.moveTo(*point)

            # 如果配置了 move_back，移动后立即归位
            if params.move_back:
                pyautogui.moveTo(*origin_point)

        # 5. 后置硬等待
        if params.wait_seconds:
            time.sleep(params.wait_seconds)

        return point

    @resolve_params(MouseParams, LocateParams, WaitParams, mode='pass')
    def click(self, target=None, x_bias=None, y_bias=None, /, **resolved_params):
        """ 点击当前形状

        支持两种调用模式的智能点击：
        1. 代理模式: click('子控件名', -10, 20)
        2. 自身模式: click(-10, 20)  <-- 兼容旧代码（target是可选字段，可以不输入，支持'参数漂移'）

        集成了多种参数模型，提供精细化的点击控制：
        1. MouseParams: 控制点击位置偏移、随机抖动、点击后复位、是否等待变化。
        2. LocateParams: 控制 wait_change 时的图像对比算法 (灰度、容差、置信度)。
        3. WaitParams: 控制 wait_change 时的超时时间 (timeout) 和检测频率 (interval)。

        :param resolved_params: 包含 MouseParams, LocateParams, WaitParams
        """
        # 1. 解析核心点击参数
        if isinstance(target, (int, float)):
            # 发生错位，手动归位
            # 说明用户没传 target 字符串，直接传了坐标
            y_bias = x_bias  # 把原来的第2个参数给 y
            x_bias = target  # 把原来的第1个参数给 x
            target = None    # 清空 target

        params: MouseParams = self.resolve_model(resolved_params['MouseParams'])
        params.update_valid(target=target, x_bias=x_bias, y_bias=y_bias)

        # 2. 代理模式分支
        if params.target is not None:
            # 如果指定了 target，转交控制权给子元素
            # 注意：这里我们要把修正后的 bias 传下去
            # 并且我们要透传 resolved_params (里面包含了 WaitParams 等其他配置)
            return self[params.target].click(
                x_bias=params.x_bias,
                y_bias=params.y_bias,
                **resolved_params
            )

        # 3. 计算点击坐标
        point = self.get_abs_point(self['center'])

        # 应用固定偏移
        point[0] += params.x_bias
        point[1] += params.y_bias

        # 应用随机偏移 (拟人化)
        if params.random_bias:
            point[0] += random.randint(-params.random_bias, params.random_bias)
            point[1] += random.randint(-params.random_bias, params.random_bias)

        # 4. 执行点击操作 (加锁保护)
        with get_autogui_lock():
            origin_point = pyautogui.position()

            # 如果配置了等待画面变化，需在点击前截图
            before_click_shot = None
            if params.wait_change:
                before_click_shot = self.shot()

            pyautogui.click(*point)

            # 点击后复位 (悬停/探测模式)
            if params.move_back:
                pyautogui.moveTo(*origin_point)

        # 5. 后置处理：等待画面变化
        if params.wait_change and before_click_shot is not None:
            # 解析辅助参数
            locate_params: LocateParams = self.resolve_model(resolved_params['LocateParams'])
            wait_params: WaitParams = self.resolve_model(resolved_params['WaitParams'])

            def check_changed():
                # 计算点击前后的差异度
                # ImageTools.img_distance 会从 resolved_params 中提取 LocateParams
                diff = ImageTools.img_distance(
                    before_click_shot,
                    self.shot(),
                    **resolved_params
                )

                # 判定逻辑：差异度 > (1 - 相似度阈值)
                # 例如 confidence=0.95，则 diff > 0.05 视为已变化
                return diff > (1.0 - locate_params.confidence)

            # 使用 WaitParams 控制等待的超时和频率
            # 这样用户可以通过 timeout=10 来控制最大等待时间
            xlwait(
                check_changed,
                timeout=wait_params.timeout,
                interval=wait_params.interval
            )

        # 6. 后置处理：硬性等待
        # 即使 wait_change 结束了，可能还需要额外 sleep 一会儿
        if params.wait_seconds:
            time.sleep(params.wait_seconds)

        return point

    @resolve_params(DragParams, mode='pass')
    def drag_to(self, drag_percent=None, drag_direction=None, /, **resolved_params):
        """ 在当前形状区域内执行拖拽操作

        通常用于滚动屏幕查找目标。
        """
        params: DragParams = self.resolve_model(resolved_params['DragParams'])
        params.update_valid(drag_percent=drag_percent, drag_direction=drag_direction)

        # 1. 计算起止点
        x, y = self.get_abs_point(self['center'])
        width, height = self['xywh'][2:]

        # 限制百分比范围
        percent = max(0.0, min(100.0, params.drag_percent))
        offset = percent / 2.0

        # 根据方向计算起点和终点
        # drag 'up' -> 鼠标向上拖 -> 内容向下滚动
        if params.drag_direction == 'up':
            start_point = [x, y + int(height * offset / 100)]
            end_point = [x, y - int(height * offset / 100)]
        elif params.drag_direction == 'down':
            start_point = [x, y - int(height * offset / 100)]
            end_point = [x, y + int(height * offset / 100)]
        elif params.drag_direction == 'left':
            start_point = [x + int(width * offset / 100), y]
            end_point = [x - int(width * offset / 100), y]
        elif params.drag_direction == 'right':
            start_point = [x - int(width * offset / 100), y]
            end_point = [x + int(width * offset / 100), y]
        else:
            raise ValueError(f"Unknown drag direction: {params.drag_direction}")

        # 2. 执行拖拽
        with get_autogui_lock():
            pyautogui.moveTo(*start_point)
            # dragTo(x, y, duration, ...)
            pyautogui.dragTo(*end_point, duration=params.drag_duration)

        # 本来有想过返回拖拽前后图片是否有变化，但是这个会较影响性能，还是另外设计更合理


class AnShape(_AnShapePupil):
    """ 高级交互功能类

    集成了图像查找、文本查找以及基于查找结果的组合交互操作。
    """

    def __1_图像类(self):
        pass

    def _resolve_img_target(self, dst):
        """ [内部辅助] 归一化匹配目标，将 dst 解析为标准的三元组

        :return: (target_img, target_rect, default_text)
            - target_img: 用于匹配的标准图片数据 (Needle)
            - target_rect: 目标相对于 self 的坐标 [x,y,w,h] (用于 Anchor 裁剪)，如果无法确定位置则为 None
            - default_text: 生成结果时的默认文本
        """
        target_img = None
        target_rect = None
        default_text = ''

        # Case A: 自身校验 (Verify Self)
        if dst is None:
            target_img = self['img']
            target_rect = None  # None 表示不需要裁剪子区域，直接对比全图
            default_text = self.get('text', '')

        # Case B: 传入了具体的对象 (Child Resolution)
        else:
            target_obj = None

            # 尝试解析为 Shape 对象
            if isinstance(dst, str):
                # 假设是 key，尝试从自身子元素获取
                # 注意：如果 key 不存在可能会报错，视原有逻辑是否容忍 KeyError
                # 这里假设用户传 str 要么是 key，要么是文件路径，需要 try-catch 或者判断
                if hasattr(self, 'find_shape_by_text') and self.find_shape_by_text(dst):
                    target_obj = self[dst]
                # 否则视为文件路径，将在后面处理
            elif isinstance(dst, _AnShapeBasic):
                target_obj = dst

            if target_obj:
                # -> 目标是 Shape 对象
                target_img = target_obj['img']
                default_text = target_obj.get('text', '')
                # [关键更新] 使用通用方法获取相对坐标，不再假设是直接子节点
                target_rect = target_obj.get_xywh_in(self)
            else:
                # -> 目标是纯图片资源/路径
                # 这种情况下没有“预存坐标”，target_rect 为 None
                target_img = xlcv.read(dst)
                default_text = str(dst)

        # 兜底：如果 target_img 加载失败（比如 dst 是路径但文件不存在），回退到自身
        if target_img is None:
            # 这里保持旧逻辑的容错性，也可以选择抛出异常
            target_img = self['img']

        return target_img, target_rect, default_text

    def _verify_anchor_mode(self, target_img, target_rect, default_text, locate_params: LocateParams):
        """ [内部辅助] Anchor 模式：原位校验

        :param target_rect: 必须提供。如果为 None，表示对比 self 的全图。
        :return:
            - 校验子区域: 返回代表该区域的新 AnShape 对象
            - 校验自身: 返回 self
            - 失败: 返回 None
        """
        current_shot = self.shot()  # 获取当前 Self 区域截图
        img_to_compare = current_shot

        # 如果有特定相对坐标，从当前截图中“抠图”出来对比
        if target_rect:
            x, y, w, h = target_rect
            # 边界保护
            sh, sw = current_shot.shape[:2]
            x, y = max(0, x), max(0, y)
            w, h = min(w, sw - x), min(h, sh - y)

            if w > 0 and h > 0:
                img_to_compare = current_shot[y:y + h, x:x + w]
            else:
                return None  # 坐标越界

        # 执行对比
        diff = ImageTools.img_distance(img_to_compare, target_img, LocateParams=locate_params)
        conf = 1.0 - diff

        logger.info(f'{self.total_name()} {conf:.0%}')

        if conf > locate_params.confidence:
            if target_rect:
                # [关键修改] 返回子对象，而不是 self
                # 构造一个新的 Shape 代表这个被校验通过的子区域
                sp = AnShape({'text': default_text, 'xywh': target_rect}, parent=self)

                # 直接复用刚才抠出来的图，性能更好
                sp['img'] = img_to_compare

                # 更新一下 center 等衍生数据，但不需要重新 shot
                sp.update_data(shot=False)
                return sp
            else:
                # 校验的是自身
                return self

        return None

    def _search_full_mode(self, target_img, default_text, locate_params: LocateParams, drag_params: DragParams):
        """ [内部辅助] Search 模式：全图搜索 + 拖拽重试

        :return: List[AnShape] 找到的新形状列表
        """

        def find_subimg():
            """ 闭包：执行单次全区域搜索 """
            current_shot = self.shot()
            # 基础图像查找
            rects = ImageTools.base_find_img(target_img, current_shot, LocateParams=locate_params)

            found_shapes = []
            for rect in rects:
                # 构造新的 Shape，坐标是相对于 self 的
                sp = AnShape({'text': default_text, 'xywh': rect}, parent=self)
                sp.update_data(shot=True)
                found_shapes.append(sp)
            return found_shapes

        # 1. 首次查找
        shapes = find_subimg()

        # 2. 拖拽重试机制
        # 如果没找到，且配置了拖拽次数，则尝试滚动屏幕寻找
        k = 0
        while not shapes and k < drag_params.drag_count:
            self.drag_to(DragParams=drag_params)  # 透传 DragParams 模型
            shapes = find_subimg()
            k += 1

        return shapes

    @resolve_params(LocateParams, DragParams, mode='pass')
    def find_img(self, dst=None, *, scan=None, **resolved_params):
        """ 查找图像匹配 (基于 Verify/Search 二元架构)

        :param dst: 匹配目标 (Key / AnShape / Image Path / None)
        :param scan: 搜索策略 (是否扫描全图)
            - False: Anchor Mode (原位校验)。假定目标位置固定，仅在预期的坐标区域进行像素级比对。速度快，适用于位置固定的静态元素。
            - True: Search Mode (在self范围内扫描)。假定目标位置浮动，在当前视图的全范围内进行模板匹配搜索。适用于动态移动或位置未知的元素。
            - None: Auto Mode (智能推断)。根据目标是否包含坐标信息自动决策：有坐标则默认为 False (校验)，无坐标则默认为 True (搜索)。
        """
        # 1. 从 resolved_params 提取强类型的配置模型
        #    得益于 @resolve_params，我们不需要手动处理 **kwargs
        locate_params: LocateParams = resolved_params['LocateParams']
        drag_params: DragParams = resolved_params['DragParams']

        # 2. 归一化目标数据
        target_img, target_rect, default_text = self._resolve_img_target(dst)

        # 3. 智能决策模式 (Auto Mode Logic)
        if scan is None:
            # 如果有具体的相对坐标 (target_rect)，倾向于相信坐标 -> Anchor Mode
            if target_rect is not None:
                scan = False
            else:
                # 如果是纯图片资源，没有坐标，必须搜索 -> Search Mode
                scan = True

        # 4. 分发执行
        if not scan:
            # --- Anchor Mode ---
            if dst is None and scan is True:
                # 逻辑互斥检查：dst=None (校验自身) 只能是 Anchor 模式
                raise ValueError("检测自身(dst=None)不能使用 scan=True (Search模式)")

            return self._verify_anchor_mode(target_img, target_rect, default_text, locate_params)

        else:
            # --- Search Mode ---
            # 如果强制要求 Search (scan=True) 但又依赖坐标 (Verify Logic)，这里会按 Search 处理
            # (即忽略 target_rect，直接在全图中找 target_img)
            return self._search_full_mode(target_img, default_text, locate_params, drag_params)

    @resolve_params(WaitParams, LocateParams, DragParams, mode='pass')
    def wait_img(self, dst=None, **resolved_params):
        """ 等待目标图片出现

        :param dst: 匹配目标
        :param resolved_params:
            - WaitParams: 控制超时(timeout)和检测间隔(interval)
            - LocateParams & DragParams: 透传给 find_img
        :return: 成功返回 result (self 或 shapes列表)，超时抛出异常
        """
        wait_params: WaitParams = self.resolve_model(resolved_params['WaitParams'])

        # 使用 xlwait 轮询
        return xlwait(
            lambda: self.find_img(dst, **resolved_params),
            timeout=wait_params.timeout,
            interval=wait_params.interval
        )

    @resolve_params(WaitParams, LocateParams, DragParams, mode='pass')
    def waitleave_img(self, dst=None, **resolved_params):
        """ 等待目标图片消失 (不再能匹配到)

        参数同 wait_img
        """
        wait_params: WaitParams = self.resolve_model(resolved_params['WaitParams'])

        # 逻辑取反：直到 find_img 返回 None 或 空列表
        return xlwait(
            lambda: not self.find_img(dst, **resolved_params),
            timeout=wait_params.timeout,
            interval=wait_params.interval
        )

    @resolve_params(WaitParams, LocateParams, DragParams, mode='pass')
    def ensure_img(self, dst=None, **resolved_params):
        """ 链式调用专用：等待图片出现。

        与 wait_img 的区别：
        1. wait_img 返回找到的 shape 对象（焦点转移）。
        2. ensure_img 返回 self 自身（焦点保持），允许继续链式操作。
        """
        # 利用 resolve_params 的透传特性，直接将解析好的模型传给 wait_img
        self.wait_img(dst, **resolved_params)
        return self

    @resolve_params(WaitParams, LocateParams, DragParams, mode='pass')
    def ensure_leave_img(self, dst=None, **resolved_params):
        """ 链式调用专用：等待图片消失，返回 self """
        self.waitleave_img(dst, **resolved_params)
        return self

    def __2_文本类(self):
        """ find_系列统一返回shapes匹配列表 """
        pass

    @resolve_params(DragParams, mode='pass')
    def find_text(self, dst=None, *, scan=True, **resolved_params):
        """ 查找文本匹配

        :param dst: 匹配目标 (正则 Pattern)
            - None: 默认使用 self['text'] 作为匹配规则
            - AnShape: 使用 dst['text']
            - str: 直接作为正则 pattern
        :param scan: 匹配模式
            - False: 全匹配模式。识别当前 Shape 区域内的整体文本，匹配 pattern。返回 self 或 None。
            - True: (默认) 局部匹配模式。调用 OCR 解析当前区域内的文本布局，返回匹配 pattern 的子 Shape 列表。
        :param resolved_params: 包含 DragParams (用于局部匹配时的滚动查找)
        :return:
            - 全匹配模式: 成功返回 self，失败返回 None
            - 局部匹配模式: 返回 AnShape 对象列表 (空列表表示未找到)
        """
        # 1. 解析参数
        drag_params: DragParams = self.resolve_model(resolved_params['DragParams'])

        # 2. 确定匹配目标 (pattern)
        if dst is None:
            pattern = self.get('text', '')
        elif isinstance(dst, _AnShapeBasic):
            pattern = dst.get('text', '')
        else:
            pattern = str(dst)

        # 3. 全匹配模式 (通常用于断言当前状态，不涉及拖拽)
        if not scan:
            # ocr_text 会截取当前 Shape 区域并调用 OCR
            return self if re.search(pattern, self.ocr_text()) else None

        # 4. 局部匹配模式 (支持拖拽重试)
        def find_subtext():
            shapes = []
            # convert_to_view('ocr') 会触发 OCR 识别并将文本行转换为子 Shape
            sub_view = self.convert_to_view('ocr', shot=True)
            for sp in sub_view.shapes:
                if re.search(pattern, sp.get('text', '')):
                    shapes.append(sp)
            return shapes

        # 首次查找
        k, shapes = 0, find_subtext()

        # 拖拽重试循环
        while not shapes and k < drag_params.drag_count:
            self.drag_to(drag_params)
            shapes = find_subtext()
            k += 1

        return shapes

    @resolve_params(WaitParams, DragParams, mode='pass')
    def wait_text(self, dst=None, **resolved_params):
        """ 等待目标文本出现

        :param dst: 匹配目标
        :param resolved_params:
            - WaitParams: 控制超时和检测间隔
            - DragParams: 透传给 find_text 用于滚动查找
        """
        wait_params: WaitParams = self.resolve_model(resolved_params['WaitParams'])

        return xlwait(
            lambda: self.find_text(dst, **resolved_params),
            timeout=wait_params.timeout,
            interval=wait_params.interval
        )

    @resolve_params(WaitParams, DragParams, mode='pass')
    def waitleave_text(self, dst=None, **resolved_params):
        """ 等待目标文本消失 """
        wait_params: WaitParams = self.resolve_model(resolved_params['WaitParams'])

        return xlwait(
            lambda: not self.find_text(dst, **resolved_params),
            timeout=wait_params.timeout,
            interval=wait_params.interval
        )

    @resolve_params(WaitParams, DragParams, mode='pass')
    def ensure_text(self, dst=None, **resolved_params):
        """ 链式调用专用：等待文本出现，返回 self """
        self.wait_text(dst, **resolved_params)
        return self

    @resolve_params(WaitParams, DragParams, mode='pass')
    def ensure_leave_text(self, dst=None, **resolved_params):
        """ 链式调用专用：等待文本消失，返回 self """
        self.waitleave_text(dst, **resolved_params)
        return self

    def __3_查找点击功能(self):
        """ 一些常用情况的简化名称，使用起来更快捷 """
        pass

    @classmethod
    def _try_click(cls, shapes, **resolved_params):
        """ 辅助方法：尝试点击找到的形状

        :param shapes: find/wait 函数返回的结果 (单个 Shape 或 Shape 列表)
        :param resolved_params: 包含 MouseParams, LocateParams, WaitParams 等
            直接透传给 sp.click()，后者会自动提取所需参数
        """
        if shapes:
            # 兼容 find_img 返回单个对象(全匹配)或列表(局部匹配)的情况
            sp = shapes[0] if isinstance(shapes, list) else shapes

            # sp.click 是被 @resolve_params 装饰的
            # 这里透传包含所有 Model 的字典，click 内部会自动提取 MouseParams 等
            sp.click(**resolved_params)
            return sp

    @resolve_params(LocateParams, DragParams, MouseParams, WaitParams, mode='pass')
    def find_img_click(self, dst=None, **resolved_params):
        """ 查找图片并点击

        组合了 find_img 和 click 的功能。

        :param resolved_params:
            - LocateParams, DragParams: 用于 find_img
            - MouseParams: 用于 click (控制偏移、复位等)
            - LocateParams, WaitParams: 用于 click 的 wait_change (可选)
        """
        shapes = self.find_img(dst, **resolved_params)
        return self._try_click(shapes, **resolved_params)

    @resolve_params(WaitParams, LocateParams, DragParams, MouseParams, mode='pass')
    def wait_img_click(self, dst=None, **resolved_params):
        """ 等待图片出现并点击 """
        shapes = self.wait_img(dst, **resolved_params)
        return self._try_click(shapes, **resolved_params)

    @resolve_params(DragParams, MouseParams, LocateParams, WaitParams, mode='pass')
    def find_text_click(self, dst=None, **resolved_params):
        """ 查找文本并点击

        注意：虽然 find_text 本身只需要 DragParams，
        但为了支持 click 中的 wait_change 功能，这里也引入了 LocateParams 和 WaitParams。
        """
        shapes = self.find_text(dst, **resolved_params)
        return self._try_click(shapes, **resolved_params)

    @resolve_params(WaitParams, DragParams, MouseParams, LocateParams, mode='pass')
    def wait_text_click(self, dst=None, **resolved_params):
        """ 等待文本出现并点击 """
        shapes = self.wait_text(dst, **resolved_params)
        return self._try_click(shapes, **resolved_params)

    def __4_其他高级功能(self):
        pass

    @resolve_params(WaitParams, LocateParams, mode='pass')
    def wait_img_notchange(self, **resolved_params):
        """ 等待直到图片画面不再变化 (画面静止/加载完成)

        以当前截图为基准，不断对比后续截图，直到差异度小于阈值。
        常用于等待页面加载完成、动画结束等场景。

        :param resolved_params:
            - WaitParams: 控制超时(timeout)和检测间隔(interval)
            - LocateParams: 控制判定"未变化"的相似度阈值 (confidence)
        """
        wait_params: WaitParams = self.resolve_model(resolved_params['WaitParams'])
        locate_params: LocateParams = self.resolve_model(resolved_params['LocateParams'])

        # 特殊逻辑：检测画面静止通常不需要太高频，默认将间隔翻倍
        # 例如 WaitParams 默认 interval=1.0，这里使用 2.0
        interval = wait_params.interval * 2

        # 初始截图
        prev_shot = self.shot()

        def check_not_change():
            nonlocal prev_shot
            current_shot = self.shot()

            # 计算前后两帧的差异度
            # ImageTools.img_distance 支持接收 **resolved_params
            diff = ImageTools.img_distance(prev_shot, current_shot, **resolved_params)

            # 更新基准图 (滑动窗口对比)
            prev_shot = current_shot

            # 判定逻辑：
            # 如果 差异度 < (1 - 相似度阈值)，说明变化很小，认为静止
            # 例如 confidence=0.95，则 diff < 0.05 视为静止
            return diff < (1.0 - locate_params.confidence)

        return xlwait(
            check_not_change,
            timeout=wait_params.timeout,
            interval=interval
        )

    def __6_其他一些常见组合使用的封装(self):
        pass

    @resolve_params(LocateParams, DragParams, mode='pass')
    def is_on(self, **resolved_params):
        """ 判断当前开关控件是否为开启状态

        逻辑：
        1. 如果 text 属性以 'off' 开头 (如 'off_switch')：
           - 能找到图片 -> 实际是关闭状态 -> 返回 False
           - 找不到图片 -> 实际是开启状态 -> 返回 True
        2. 其他情况 (默认 text 代表开启状态图片)：
           - 能找到图片 -> 开启状态 -> 返回 True
           - 找不到图片 -> 关闭状态 -> 返回 False

        :param resolved_params: 透传给 find_img 用于图像匹配
        """
        # 获取自身的文本标签
        text = self.get('text', '')

        # 调用 find_img 检测当前状态
        # 显式传递参数模型以支持 locate 和 drag 配置
        found = self.find_img(**resolved_params)

        if text.startswith('off'):
            return not found
        else:
            return bool(found)

    @resolve_params(MouseParams, LocateParams, DragParams, WaitParams, mode='pass')
    def turn_on(self, **resolved_params):
        """ 确保开关处于开启状态

        如果当前检测为关闭，则执行点击操作。

        :param resolved_params:
            - MouseParams: 用于 click
            - LocateParams, DragParams: 用于 is_on 中的检测
            - WaitParams: 用于 click 的潜在等待
        """
        # 1. 检查状态 (复用 resolved_params 中的 Locate/Drag)
        if not self.is_on(**resolved_params):
            # 2. 执行点击 (复用 resolved_params 中的 Mouse/Wait)
            self.click(**resolved_params)

    @resolve_params(MouseParams, LocateParams, DragParams, WaitParams, mode='pass')
    def turn_off(self, **resolved_params):
        """ 确保开关处于关闭状态 """
        if self.is_on(**resolved_params):
            self.click(**resolved_params)

    @resolve_params(WaitParams, LocateParams, DragParams, MouseParams, mode='pass')
    def wait_img_click_leave(self, dst=None, **resolved_params):
        """ 经典组合操作：等待出现 -> 点击 -> 直到消失

        常用于处理弹窗广告、确认按钮等需要反复点击直到生效的场景。
        """
        # 1. 等待目标出现
        self.wait_img(dst, **resolved_params)

        # 2. 循环点击直到目标不再被检测到
        # 注意：这里隐式依赖 find_img，如果 find_img 能找到，就点击
        while self.find_img(dst, **resolved_params):
            self.click(**resolved_params)

            # 为了防止死循环过快，可以加一个微小的间隔，或者依赖 click 内部的 wait_seconds
            # 如果 MouseParams 没有配置 wait_seconds，这里通常不需要额外 sleep，
            # 因为 find_img 本身有一定耗时

    @resolve_params(LocateParams, DragParams, MouseParams, WaitParams, mode='pass')
    def if_img_click_leave(self, dst=None, **resolved_params):
        """ 如果出现 -> 点击 -> 直到消失 (非阻塞版)

        与 wait_img_click_leave 的区别在于：如果起初没找到，直接跳过，不报错不等待。
        """
        while self.find_img(dst, **resolved_params):
            self.click(**resolved_params)


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

        # 2 计算偏移量：检测到的动态位置与静态锚点位置的差值
        # 使用绝对坐标来计算相对位移，兼容 det_shape 来自深层子节点的情况
        anchor_shape = sub_view[anchor_shape_loc]

        diff_rect = det_shape.get_xywh_in(anchor_shape)
        sub_view.move(diff_rect[0], diff_rect[1])

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
            f1 = self.folder / f'{item}.json'
            if f1.is_file():
                return self.loc_view(item)
            else:
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
