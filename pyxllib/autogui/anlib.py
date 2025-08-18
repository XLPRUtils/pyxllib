#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2025/06/05

""" 这是一套基于labelme标注来进行 """

import sys
from collections import defaultdict
import json
import os
import time
import random
import re
import datetime
from collections import UserDict

import numpy as np
import win32com
from loguru import logger

from pyxlpr.ai.clientlib import XlAiClient

if sys.platform == 'win32':
    import pyautogui
    import win32gui

from pyxllib.prog.newbie import first_nonnone, round_int
from pyxllib.prog.pupil import xlwait, DictTool, run_once
from pyxllib.algo.geo import ComputeIou, ltrb2xywh, xywh2ltrb
from pyxllib.file.specialist import XlPath
from pyxllib.cv.expert import xlcv, xlpil
from pyxlpr.data.labelme import LabelmeDict

from pyxllib.autogui.uiautolib import find_ctrl, UiCtrlNode

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
    xlapi = XlAiClient()
    xlapi.login_priu(os.getenv('XL_API_PRIU_TOKEN'), 'http://xmutpriu.com')
    return xlapi


class ImageTools:

    @classmethod
    def pixel_distance(cls, pixel1, pixel2):
        """ 返回最大的像素值差，如果相同返回0 """
        return max([abs(x - y) for x, y in zip(pixel1, pixel2)])

    @classmethod
    def img_distance(cls, img1, img2, *, color_tolerance=10):
        """ 返回距离：不相同像素占的百分比，相同返回0

        :param color_tolerance: 每个像素允许的颜色差值。RGB情况计算的是3个值分别的差的和。
        """
        cmp = np.array(abs(np.array(img1, dtype=int) - img2) > color_tolerance)
        return cmp.sum() / cmp.size


class AnShape(UserDict):

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

    def set_shape(self, text, xywh):
        """ 配置形状 """
        self['text'] = text
        self['xywh'] = xywh
        # 通过xywh计算出center
        self['center'] = [xywh[0] + xywh[2] // 2, xywh[1] + xywh[3] // 2]

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

        # 2 中心点 center（无论任何原始形状，都转成矩形处理。这跟下游功能有关，目前只能做矩形。）
        shape_type = lmshape['shape_type']
        pts = lmshape['points']
        if shape_type in ('rectangle', 'polygon', 'line'):
            anshape['center'] = np.array(np.array(pts).mean(axis=0), dtype=int).tolist()
        elif shape_type == 'circle':
            anshape['center'] = pts[0]

        # 3 外接矩形 rect
        if shape_type in ('rectangle', 'polygon'):
            ltrb = np.array(pts, dtype=int).reshape(-1).tolist()
        elif shape_type == 'circle':
            x, y = pts[0]
            r = ((x - pts[1][0]) ** 2 + (y - pts[1][1]) ** 2) ** 0.5
            ltrb = [round_int(v) for v in [x - r, y - r, x + r, y + r]]
        # 把ltrb转换为xywh
        anshape['xywh'] = ltrb2xywh(ltrb)

        # 4 图片数据 img, etag
        if self.parent.get('img') is not None and anshape['xywh']:
            anshape['img'] = xlcv.get_sub(self.parent['img'], ltrb)
            # anshape['etag'] = get_etag(anshape['img'])
            # TODO 这里目的其实就是让相同图片对比尽量相似，所以可以用dhash而不是etag

        # 5 中心点像素值 pixel
        p = anshape['center']
        if self.parent.get('img') is not None and p:
            anshape['pixel'] = tuple(self.parent['img'][p[1], p[0]].tolist()[::-1])

        self.update(anshape)

    def __2_转换与保存(self):
        pass

    def convert_to_view(self, view_name=None):
        """ 将这个shape升级为一个view对象 """
        img = self.shot()
        view = AnView.init_from_image(img)
        view.parent = self
        view['text'] = view_name
        return view

    def save_image(self, region_folder, view_name=None):
        """ 保存当前快照图片 """
        if not view_name:  # 未输入则用时间戳代替
            view_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        impath = XlPath(region_folder) / f'{view_name}.jpg'
        os.makedirs(os.path.dirname(impath), exist_ok=True)
        xlcv.write(self.shot(), impath)

    def save_view(self, region_folder, view_name=None):
        """ 保存当前视图（即保存图片和对应的标注数据） """
        view = self.convert_to_view(view_name)
        if not view_name:  # 未输入则用时间戳代替
            view_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        impath = XlPath(region_folder) / f'{view_name}.jpg'
        view.save_labelme_file(impath)

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

    def __4_分析图像(self):
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

    def check_pixel(self, *, color_tolerance=None):
        """ 判断对应位置上的像素值是否相同
        """
        color_tolerance = self.get_parent_argv('color_tolerance', color_tolerance) or 20
        p1 = self['pixel']
        p2 = self.get_pixel()
        return ImageTools.pixel_distance(p1, p2) < color_tolerance

    def check_img(self, *, color_tolerance=None, grayscale=None, confidence=None):
        """
        :param color_tolerance:
        :param grayscale: 灰度图的功能效果暂时还没做
            因为这套参数是从原本pyautogui.locateAll迁移过来的，原本是能直接支持这个参数的
        :param confidence:
        :return:
        """
        color_tolerance = self.get_parent_argv('color_tolerance', color_tolerance) or 20
        confidence = self.get_parent_argv('confidence', confidence) or 0.95

        shot = self.shot()
        if confidence >= 1:
            return np.array_equal(self['img'], shot)
        else:
            dist = ImageTools.img_distance(self['img'], shot, color_tolerance=color_tolerance)
            return dist < 1 - confidence

    def __5_ocr识别(self):
        pass

    def ocr_text(self):
        text = get_xlapi().rec_singleline(self.shot())
        return text

    def ocr_value(self):
        text = self.ocr_text()
        m = re.search(r'\d+', text) or 0
        if m:
            m = int(m.group())
        return m

    def __6_操作(self):
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
        pyautogui.moveTo(*point)
        return point

    def click(self, *, random_bias=None, back=None, wait_change=None, wait_seconds=None):
        """ 点击这个shape

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

        # 2 添加随机偏移
        if random_bias:
            point[0] += random.randint(-random_bias, random_bias)
            point[1] += random.randint(-random_bias, random_bias)

        # 3 鼠标移动
        origin_point = pyautogui.position()
        if wait_change:
            before_click_shot = self.shot()
        pyautogui.click(*point)
        if back:
            pyautogui.moveTo(*origin_point)  # 恢复鼠标原位置

        # 4 等待
        if wait_change:
            color_tolerance = self.get_parent_argv('color_tolerance', None) or 20
            confidence = self.get_parent_argv('confidence', None) or 0.95
            func = lambda: ImageTools.img_distance(before_click_shot, self.shot(),
                                                   color_tolerance=color_tolerance) > confidence
            xlwait(func)
        if wait_seconds:
            time.sleep(wait_seconds)

        return point

    def drag_to(self, percentage=50, direction='up', duration=None, **kwargs):
        """ 在当前shape里拖拽

        :param int percentage: 拖拽比例，范围0-100，表示滚动区域的百分比
        :param str direction: 滚动方向，可选 'up', 'down', 'left', 'right'
        :param float duration: 拖拽动作持续时间(秒)
        """
        # 0 检索配置参数
        drag_duration = self.get_parent_argv('drag_duration', duration) or 1

        # 1 计算起点和终点位置
        x, y = self.get_abs_point(self['center'])
        width, height = self['xywh'][2:]

        # 限制百分比在0-100之间
        percentage = max(0, min(100, percentage))
        offset = percentage / 2  # 从中心点偏移的百分比

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
        pyautogui.moveTo(*start_point)
        pyautogui.dragTo(*end_point, drag_duration, **kwargs)

        # 本来有想过返回拖拽前后图片是否有变化，但是这个会较影响性能，还是另外设计更合理

    def wait_pixel(self, *, limit=None, interval=1):
        """ 等到目标匹配像素出现 """
        xlwait(lambda: self.check_pixel(), limit=limit, interval=interval)

    def wait_img(self, *, limit=None, interval=1):
        """ 等到目标匹配图片出现 """
        xlwait(lambda: self.check_img(), limit=limit, interval=interval)

    def wait_pixel_leave(self, *, limit=None, interval=1):
        """ 等到目标像素消失 """
        xlwait(lambda: not self.check_pixel(), limit=limit, interval=interval)

    def wait_img_leave(self, *, limit=None, interval=1):
        xlwait(lambda: not self.check_img(), limit=limit, interval=interval)

    def check_pixel_click(self, **kwargs):
        """ 检查目标对应才点击
        注意子类view、region可以设计更高级的check_click，实现位置可变的目标检索
        """
        if self.check_pixel():
            self.click(**kwargs)

    def check_img_click(self, **kwargs):
        if self.check_img():
            self.click(**kwargs)

    def wait_pixel_click(self, **kwargs):
        self.wait_pixel()
        self.click(**kwargs)

    def wait_img_click(self, **kwargs):
        self.wait_img()
        self.click(**kwargs)


class AnView(AnShape):

    def __1_构建(self):
        pass

    def __init__(self, initialdata=None, parent=None):
        """ 这个类基本都要用预设的几个特殊的init接口去初始化，实际使用中一般不直接用这个raw init
        """
        super().__init__(initialdata, parent)
        self['text'] = 'view'

        self.impath = None  # 如果是从文件读取来的数据，存储原始图片路径，在save重写回去的时候就可以缺省保存参数
        self.shapes = []  # 以列表的形式顺序存储的anshapes

    def init_from_xywh(self, xywh=None):
        if xywh is None and 'img' in self:
            xywh = [0, 0, self['img'].shape[1], self['img'].shape[0]]
        if xywh is None:
            return

        self['xywh'] = xywh
        self['center'] = [xywh[0] + xywh[2] // 2, xywh[1] + xywh[3] // 2]

        if self.get('img') is not None:
            self['pixel'] = tuple(self['img'][xywh[3] // 2, xywh[2] // 2].tolist()[::-1])

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
        view['img'] = xlcv.read(view.impath)
        view.read_lmshapes(lmdict['shapes'])
        view.init_from_xywh()

        return view

    @classmethod
    def init_from_image(cls, image):
        """ 从一张图片初始化
        这里默认会调用xlapi来生成labelme标注文件
        """
        # 1 读取图片
        view = AnView()
        view['img'] = xlcv.read(image)
        if isinstance(image, (str, XlPath)):
            view.impath = image

        # 2 调用ocr识别
        xlapi = get_xlapi()
        lmdict = xlapi.common_ocr(view['img'])
        view.read_lmshapes(lmdict['shapes'])
        view.init_from_xywh()

        return view

    def init_from_shot(self):
        """ 已有xywh等形状参数的情况下，可以用这个从shot初始化 """
        self['img'] = self.shot()
        xlapi = get_xlapi()
        lmdict = xlapi.common_ocr(self['img'])
        self.read_lmshapes(lmdict['shapes'])

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
                                          group_id=sp['group_id'], flags=sp['flags'])
            lmdict['shapes'].append(shape)

        # 3 保存
        jsonpath.write_json(lmdict, indent=2)


class AnRegion(AnShape):

    def __init__(self,
                 folder=None,
                 *,
                 read_views=True,
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

        if read_views:
            self.read_views()

    def read_views(self):
        if not (self.folder and self.folder.exists()):
            return

        self.views = {}
        for json_file in self.folder.rglob_files('*.json'):
            view_name = json_file.relative_to(self.folder).as_posix()[:-5]  # 注意 要去掉后缀.json
            view = AnView.init_from_labelme_json_file(json_file)
            view['text'] = view_name
            view.parent = self
            self.views[view_name] = view

    def loc(self, item):
        if '/' in item:
            # 注意功能效果：os.path.split('a/b/c') -> ('a/b', 'c')
            # 注意还有种很特殊的：'a' -> ('', 'a')
            view_name, shape_name = os.path.split(item)
            return self.views[view_name][shape_name]
        else:
            return self.views.get(item)

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

        self.ctrl = None

        self._xlapi = None  # ocr工具
        self._speaker = None  # 语音播报工具
        # todo 添加数据库db？

    def set_ctrl(self, class_name=None, name=None, **kwargs):
        """ 设置窗口控件 """

        ctrl = find_ctrl(class_name=class_name, name=name, **kwargs)
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


def __x_旧代码备份():
    pass


# 这个是"数据库"用法，存储了一些特定shape类型的图片数据，用于后续检索使用
# 有点像图片分类任务，用这个框架可以完成简单的图片分类而不需要专门训练分类模型
def update_loclabel_img(self, loclabel, img, *, if_exists='update'):
    """ 添加一个标注框
    注：这个功能未被任何地方使用过

    :param if_exists:
        update，更新
        skip，跳过，不更新
    """
    loc, label = os.path.split(loclabel)
    h, w = img.shape[:2]

    # 1 如果不存在这组loc，则新建一个jpg图片
    update = True
    if loc not in self.locs or not self.locs[loc]:
        imfile = xlcv.write(img, self.folder / f'{loc}.jpg')
        self.imfiles[loc] = imfile
        shape = LabelmeDict.gen_shape(label, [[0, 0], [w, h]])
        self.locs[loc][label] = self.parse_shape(shape)
    # 2 不存在的标签，则在最后一行新建一张图
    elif label not in self.locs[loc]:
        image = xlcv.read(self.imfiles[loc])
        height, width = image.shape[:2]
        assert width == w  # 先按行拼接，以后有时间可以扩展更灵活的拼接操作
        # 拼接，并重新存储为图片
        image = np.concatenate([image, img])
        xlcv.write(image, self.imfiles[loc])
        shape = LabelmeDict.gen_shape(label, [[0, height], [width, height + h]])
        self.locs[loc][label] = self.parse_shape(shape)
    # 3 已有的图，则进行替换
    elif if_exists == 'update':
        image = xlcv.read(self.imfiles[loc])
        [x1, y1, x2, y2] = self.locs[loc][label]['ltrb']
        image[y1:y2, x1:x2] = img
        xlcv.write(image, self.imfiles[loc])
    else:
        update = False

    if update:  # 需要实时保存到文件中
        self.write(loc)


class NamedLocate:
    """ 对有命名的标注数据进行定位

    注意labelme在每张图片上写的label值最好不要有重复，否则按字典存储后可能会被覆盖

    特殊取值：
    '@IMAGE_ID' 该状态图的标志区域，即游戏中在该位置出现此子图，则认为进入了对应的图像状态

    截图：screenshot，update_shot
    检查固定位像素：point2pixel，pixel_distance，check_pixel
    检查固定位图片：rect2img，img_distance，check_img
    检索图片：img2rects, img2rect，img2point，img2img
    操作：click，move_to
    高级：wait，check_click，wait_click
    """

    def img2rects(self, img, haystack=None, *, grayscale=None, confidence=None):
        """ 根据预存的img数据，匹配出多个内容对应的所在的rect位置 """
        # 1 配置参数
        if isinstance(img, str):
            img = self[img]['img']
        grayscale = first_nonnone([grayscale, self.grayscale, False])
        confidence = first_nonnone([confidence, self.confidence, 0.95])
        # 2 查找子图
        if haystack is None:
            self.update_shot()
            haystack = self.last_shot
        boxes = pyautogui.locateAll(img, haystack,
                                    grayscale=grayscale,
                                    confidence=confidence)
        # 3 过滤掉重叠超过一半面积的框
        rects = ComputeIou.nms_ltrb([xywh2ltrb(box) for box in list(boxes)])
        return rects

    def img2rect(self, img, haystack=None, *, grayscale=None, confidence=None):
        """ img2rects的简化，只返回一个匹配框
        """
        rects = self.img2rects(img, haystack, grayscale=grayscale, confidence=confidence)
        return rects[0] if rects else None

    def img2point(self, img, haystack=None, *, grayscale=None, confidence=None):
        """ 将 img2rect 的结果统一转成点 """
        res = self.img2rect(img, haystack, grayscale=grayscale, confidence=confidence)
        if res:
            return np.array(np.array(res).reshape(2, 2).mean(axis=0), dtype=int).tolist()

    def img2img(self, img, haystack=None, *, grayscale=None, confidence=None):
        """ 找到rect后，返回匹配的目标img图片内容 """
        ltrb = self.img2rect(img, haystack, grayscale=grayscale, confidence=confidence)
        if ltrb:
            l, t, r, b = ltrb
            if haystack is not None:
                haystack = self.last_shot
            return haystack[t:b, l:r]


if __name__ == '__main__':
    pass
