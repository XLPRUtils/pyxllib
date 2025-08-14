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

import numpy as np
import win32com
from pandas.api.types import is_list_like

from pyxlpr.ai.clientlib import XlAiClient

if sys.platform == 'win32':
    import pyautogui
    import win32gui

from pyxllib.prog.newbie import first_nonnone, round_int
from pyxllib.prog.pupil import xlwait, DictTool
from pyxllib.algo.geo import ComputeIou, ltrb2xywh, xywh2ltrb, ltrb2polygon
from pyxllib.algo.shapelylib import ShapelyPolygon
from pyxllib.file.specialist import XlPath
from pyxllib.cv.expert import xlcv, xlpil
from pyxlpr.data.labelme import LabelmeDict


class AutoGuiLabelData:

    def __init__(self, root=None):
        """
        :param root:
            注意None不是处理当前目录，而是表示不分析读取目录下的文件，由后续其他机制进一步初始化

        data：双层字典
            第一层key: 目录相对路径/图片stem （这一节称为loc），这里对应一组标注文件（配对的图片和json）
            第二层key：是每份json读取出来的更具体的标注，这一层称为label。
                两层组合，就是loclabel

            示例1：目录下有图片 "界面.jpg"，对应的"界面.json"里有个label是"姓名"，那loclabel就是"界面/姓名"
            示例2：如果有子目录，比如"战斗/界面.jpg"，那loclabel就是"战斗/界面/姓名"
        """
        self.root = XlPath() if root is None else XlPath(root)
        self.data = defaultdict(dict)  # [loc][label] -> attrs
        self.imfiles = {}  # loc -> img

        if root is not None:
            self._read_files()

    def _read_files(self):
        for file in self.root.rglob_files('*.json'):
            lmdict = file.read_json()
            imfile = file.with_name(lmdict['imagePath'])
            img = xlcv.read(imfile)
            loc = str(file.relpath(self.root)).replace('\\', '/')[:-5]  # 这个是支持子目录分类结构的
            self.imfiles[loc] = imfile

            for shape in lmdict['shapes']:
                attrs = self.parse_shape(shape, img)
                self.data[loc][attrs['text']] = attrs

    def add_loc_data(self, loc, img, lmdict):
        """ 从内存数据来初始化该类，这种情况配置的数据，loc直接标记为空字符串
                这种情况是能兼容__getitem__、__setitem__等操作的

        :param loc: loc标记为空字符串或数值都可以
        """
        imfile = self.root / f'{loc}.jpg'
        self.imfiles[loc] = imfile

        for shape in lmdict['shapes']:
            attrs = self.parse_shape(shape, img)
            self.data[loc][attrs['text']] = attrs

    @classmethod
    def parse_shape(cls, shape, full_image=None):
        """ 解析一个shape的数据为dict字典，会把一些主要的几何特征也加进label字典字段中 """
        # 1 解析原label为字典
        # 如果是来自普通的label，会把原文本自动转换为text字段。如果是来自xllabelme，则一般默认就是字典值，带text字段。
        attrs = DictTool.json_loads(shape['label'], 'text')
        attrs.update(DictTool.sub(shape, ['label']))  # 因为原本的label被展开成字典了，这里要删掉label

        # 2 中心点 center（无论任何原始形状，都转成矩形处理。这跟下游功能有关，目前只能做矩形。）
        shape_type = shape['shape_type']
        pts = shape['points']
        if shape_type in ('rectangle', 'polygon', 'line'):
            attrs['center'] = np.array(np.array(pts).mean(axis=0), dtype=int).tolist()
        elif shape_type == 'circle':
            attrs['center'] = pts[0]

        # 3 外接矩形 rect
        if shape_type in ('rectangle', 'polygon'):
            attrs['ltrb'] = np.array(pts, dtype=int).reshape(-1).tolist()
        elif shape_type == 'circle':
            x, y = pts[0]
            r = ((x - pts[1][0]) ** 2 + (y - pts[1][1]) ** 2) ** 0.5
            attrs['ltrb'] = [round_int(v) for v in [x - r, y - r, x + r, y + r]]

        # 4 图片数据 img, etag
        if full_image is not None and attrs['ltrb']:
            attrs['img'] = xlcv.get_sub(full_image, attrs['ltrb'])
            # attrs['etag'] = get_etag(attrs['img'])
            # TODO 这里目的其实就是让相同图片对比尽量相似，所以可以用dhash而不是etag

        # 5 中心点像素值 pixel
        p = attrs['center']
        if full_image is not None and p:
            attrs['pixel'] = tuple(full_image[p[1], p[0]].tolist()[::-1])

        # if 'rect' in attrs:
        #     del attrs['rect']  # 旧版的格式数据，删除

        return attrs

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
        if loc not in self.data or not self.data[loc]:
            imfile = xlcv.write(img, self.root / f'{loc}.jpg')
            self.imfiles[loc] = imfile
            shape = LabelmeDict.gen_shape(label, [[0, 0], [w, h]])
            self.data[loc][label] = self.parse_shape(shape)
        # 2 不存在的标签，则在最后一行新建一张图
        elif label not in self.data[loc]:
            image = xlcv.read(self.imfiles[loc])
            height, width = image.shape[:2]
            assert width == w  # 先按行拼接，以后有时间可以扩展更灵活的拼接操作
            # 拼接，并重新存储为图片
            image = np.concatenate([image, img])
            xlcv.write(image, self.imfiles[loc])
            shape = LabelmeDict.gen_shape(label, [[0, height], [width, height + h]])
            self.data[loc][label] = self.parse_shape(shape)
        # 3 已有的图，则进行替换
        elif if_exists == 'update':
            image = xlcv.read(self.imfiles[loc])
            [x1, y1, x2, y2] = self.data[loc][label]['ltrb']
            image[y1:y2, x1:x2] = img
            xlcv.write(image, self.imfiles[loc])
        else:
            update = False

        if update:  # 需要实时保存到文件中
            self.write(loc)

    def write(self, loc):
        """ 数据在内存中可能有变动，这里保存某一组loc的数据 """
        f = self.root / f'{loc}.json'
        imfile = self.imfiles[loc]
        lmdict = LabelmeDict.gen_data(imfile)
        for label, ann in self.data[loc].items():
            a = ann.copy()
            DictTool.isub(a, ['img'])
            shape = LabelmeDict.gen_shape(json.dumps(a, ensure_ascii=False),
                                          a['points'], a['shape_type'],
                                          group_id=a['group_id'], flags=a['flags'])
            lmdict['shapes'].append(shape)
        f.write(lmdict, indent=2)

    def writes(self):
        """ 保存所有loc """
        for loc in self.data.keys():
            self.write(loc)

    def __getitem__(self, loclabel):
        """ 支持 self['主菜单/道友'] 的格式获取 self.data['主菜单']['道友'] 数据
        """
        # 注意功能效果：os.path.split('a/b/c') -> ('a/b', 'c')
        # 注意还有种很特殊的：'a' -> ('', 'a')，这种就是纯label没有loc，或者说loc是空字符串''，也是可行的。这种一般数据直接来自内存。
        loc, label = os.path.split(loclabel)
        try:
            return self.data[loc][label]
        except KeyError:
            return None

    def __setitem__(self, loclabel, value):
        loc, label = os.path.split(loclabel)
        self.data[loc][label] = value


class NamedLocate(AutoGuiLabelData):
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

    def __init__(self, root, *, region=None, grayscale=None, confidence=0.95, tolerance=20):
        """

        :param root: 标注数据所在文件夹
            有原图jpg、png和对应的json标注数据
        :param region: 可以限定窗口位置 (xywh格式)
        :param grayscale: 转成灰度图比较
        :param confidence: 用于图片比较，相似度阈值
        :param tolerance: 像素比较时能容忍的误差变化范围

        TODO random_click: 增加该可选参数，返回point时，进行上下左右随机扰动
        """
        if root is not None:
            super().__init__(root)
        else:
            pass

        self.grayscale = grayscale
        self.confidence = confidence
        self.tolerance = tolerance
        self.region = region

        self.last_shot = self.update_shot()

    def __1_几何坐标换算(self):
        pass

    def point_local2global(self, p):
        """ 一个self.region中相对坐标点p，转换到全局坐标位置 """
        if self.region:
            x, y = p
            x += self.region[0]
            y += self.region[1]
            return [x, y]
        else:
            return p

    def point_global2local(self, p):
        """ 一个self.region中相对坐标点p，转换到全局坐标位置 """
        if self.region:
            x, y = p
            x -= self.region[0]
            y -= self.region[1]
            return [x, y]
        else:
            return p

    def loclabel2global_xywh(self, loclabel):
        """ 换算到全局的xywh坐标值 """
        # 先找到loclabel的ltrb
        ltrb = self[loclabel]['ltrb']
        l, t, r, b = ltrb
        # 转换为全局坐标
        l += self.region[0]
        t += self.region[1]
        r += self.region[0]
        b += self.region[1]
        # 转换为xywh格式
        x, y, w, h = l, t, r - l, b - t
        return [x, y, w, h]

    def point2loclabels(self, point, loclabels):
        """ 判断loc这张图里有哪些标签覆盖到了point这个点

        :param loclabels: 可以输入一张图的定位
            也可以输入多个loclabel清单，会返回所有满足的loclabel名称
        """
        from shapely.geometry import Point

        # 1 loclabels
        if not is_list_like(loclabels):
            loclabels = [loclabels]
        point = Point(*point)

        # 2 result
        res = []
        for loclabel in loclabels:
            if loclabel in self.data:
                for k, v in self.data.items():
                    if 'ltrb' in v:
                        if point.within(ShapelyPolygon.gen(v['ltrb'])):
                            res.append(f'{loclabel}/{k}')
            else:
                v = self[loclabel]
                if 'ltrb' in v:
                    if point.within(ShapelyPolygon.gen(v['ltrb'])):
                        res.append(f'{loclabel}')

        return res

    def point2loclabel(self, point, loclabels):
        """ 只返回第一个匹配结果
        """
        from shapely.geometry import Point

        # 1 loclabels
        if not is_list_like(loclabels):
            loclabels = [loclabels]
        point = Point(*point)

        # 2 result
        for loclabel in loclabels:
            if loclabel in self.data:
                for k, v in self.data.items():
                    if 'ltrb' in v:
                        if point.within(ShapelyPolygon.gen(v['ltrb'])):
                            return f'{loclabel}/{k}'
            else:
                v = self[loclabel]
                if 'ltrb' in v:
                    if point.within(ShapelyPolygon.gen(v['ltrb'])):
                        return f'{loclabel}'

    def __2_截图相关(self):
        pass

    def screenshot(self, region=None):
        """
        :param region: ltrb
        """
        if isinstance(region, str):
            region = ltrb2xywh(self[region]['ltrb'])
        im = pyautogui.screenshot(region=region)
        return xlpil.to_cv2_image(im)

    def update_shot(self, region=None):
        region = region or self.region
        self.last_shot = self.screenshot(region)
        return self.last_shot

    def point2pixel(self, point, haystack=None):
        """

        :param point: 支持输入 (x, y) 坐标，或者 loclabel
            后面很多参数都是类似，除了标准数据类型，同时支持 loclabel 定位模式

         官方的pixel版本，在windows有时候会出bug
         OSError: windll.user32.ReleaseDC failed : return 0
        """
        if isinstance(point, str):
            point = self[point]['center']
        if haystack is None:
            haystack = self.update_shot()
        return tuple(haystack[point[1], point[0]].tolist()[::-1])

    def rect2img(self, ltrb, haystack=None):
        """
        :param ltrb: 可以输入坐标定位，也可以输入 loclabel 定位
        """
        if isinstance(ltrb, str):
            ltrb = self[ltrb]['ltrb']
        l, t, r, b = ltrb
        if haystack is None:
            haystack = self.update_shot()
        return haystack[t:b, l:r]

    def __3_图像算法(self):
        pass

    @classmethod
    def pixel_distance(cls, pixel1, pixel2):
        """ 返回最大的像素值差，如果相同返回0 """
        return max([abs(x - y) for x, y in zip(pixel1, pixel2)])

    @classmethod
    def img_distance(cls, img1, img2, *, tolerance=10):
        """ 返回距离：不相同像素占的百分比，相同返回0

        :param tolerance: 每个像素允许10的差距
        """
        cmp = np.array(abs(np.array(img1, dtype=int) - img2) > tolerance)
        return cmp.sum() / cmp.size

    def check_pixel(self, loclabel, haystack=None, *, tolerance=None):
        """ 判断对应位置上的像素值是否相同
        """
        tolerance = first_nonnone([tolerance, self.tolerance, 20])
        p1 = self[loclabel]['pixel']
        p2 = self.point2pixel(loclabel, haystack)
        return self.pixel_distance(p1, p2) < tolerance

    def check_img(self, loclabel, needle=None, *, grayscale=None, confidence=None):
        grayscale = first_nonnone([grayscale, self.grayscale, False])
        confidence = first_nonnone([confidence, self.confidence, 0.95])
        if needle is None:
            needle = self[loclabel]['img']
        elif isinstance(needle, str):
            needle = self[needle]['img']
        haystack = self.screenshot(loclabel)

        if confidence >= 1:
            return np.array_equal(needle, haystack)
        else:
            boxes = pyautogui.locateAll(needle, self.screenshot(loclabel),
                                        grayscale=grayscale, confidence=confidence)
            boxes = [xywh2ltrb(box) for box in list(boxes)]
            return len(boxes)

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


class AutoPc(NamedLocate):
    """ 我原创的一个游戏自动化脚本框架 """

    def __init__(self, ctrl_name='', label_dir=None, **kwargs):
        super().__init__(label_dir, **kwargs)

        self.ctrl_name = ctrl_name  # 例如'AirDroid Cast v1.2.3.1'
        self._ctrl = None  # 窗口控件
        self._xlapi = None  # ocr等工具
        self._speaker = None  # 语音播报工具

        # todo 添加数据库db？

    @property
    def ctrl(self):
        if self._ctrl is None:
            from pyxllib.autogui.uiautolib import find_ctrl, UiCtrlNode
            ctrl = find_ctrl(self.ctrl_name)
            self._ctrl = UiCtrlNode(ctrl)
            self._ctrl.activate()

        return self._ctrl

    @property
    def xlapi(self):
        if self._xlapi is None:
            from pyxlpr.ai.clientlib import XlAiClient

            self._xlapi = XlAiClient()
            self._xlapi.login_priu(os.getenv('XL_API_PRIU_TOKEN'), 'http://xmutpriu.com')
        return self._xlapi

    @property
    def speaker(self):
        if self._speaker is None:
            self._speaker = win32com.client.Dispatch('SAPI.SpVoice')
        return self.speaker

    def __1_交互操作(self):
        pass

    def click(self, point, *, random_bias=0, back=False, wait_change=False, wait_seconds=0):
        """
        :param back: 点击后鼠标移回原坐标位置
        :param wait_change: 点击后，等待画面产生变化，才结束函数
            point为 loclabel 才能用这个功能
            注意这个只是比较loclabel的局部区域图片变化，不是全图的变化
        :param int random_bias: 允许在一定范围内随机点击点
        """
        # 1 判断类型
        if isinstance(point, str):
            loclabel = point
            point = self[loclabel]['center']
        else:
            loclabel = ''

        # 随机偏移点
        if random_bias:
            x, y = point
            x += random.randint(-random_bias, +random_bias)
            y += random.randint(-random_bias, +random_bias)
            point = (x, y)

        # 2 鼠标移动
        pos0 = pyautogui.position()
        pyautogui.click(*self.point_local2global(point))
        if back:
            pyautogui.moveTo(*pos0)  # 恢复鼠标原位置

        # 3 等待画面变化
        if loclabel and wait_change:
            func = lambda: self.img_distance(self[loclabel]['img'],
                                             self.rect2img(loclabel)) > self.confidence
            xlwait(func)
        if wait_seconds:
            time.sleep(wait_seconds)

        return point

    def move_to(self, point):
        if isinstance(point, str):
            point = self[point]['center']
        pyautogui.moveTo(*self.point_local2global(point))

    def drag_to(self, loclabel, direction=0, duration=1, *, to_tail=False, **kwargs):
        """ 在loclabel区域内进行拖拽操作

        :param direction: 方向，0123顺时针分别表示向"上/右/下/左"拖拽。
        :param duration: 拖拽用时
        :param to_tail:
            False，默认只拖拽一次
            True，一直拖拽到末尾，即内容不再变化为止
        """

        def core():
            x = self[loclabel]
            l, t, r, b = x['ltrb']
            x2, y2 = x['center']

            self.move_to(loclabel)
            if direction == 0:
                y2 = (y2 + t) // 2
            elif direction == 1:
                x2 = (x2 + r) // 2
            elif direction == 2:
                y2 = (y2 + b) // 2
            elif direction == 3:
                x2 = (x2 + l) // 2
            else:
                raise ValueError

            x2, y2 = self.point_local2global([x2, y2])
            pyautogui.dragTo(x2, y2, duration=duration, **kwargs)

        if to_tail:
            last_im = self.rect2img(loclabel)
            while True:
                core()
                time.sleep(1)
                new_im = self.rect2img(loclabel)
                if self.img_distance(last_im, new_im) > 0.95:  # 95%的像素相似
                    break
                last_im = new_im
        else:
            core()

    def wait(self, loclabel, *, fixpos=False, limit=None, interval=1):
        """ 封装的比较高层的功能 """
        # dprint(loclabel, fixpos)
        if fixpos:
            xlwait(lambda: self.check_img(loclabel))
            point = self[loclabel]['center']
        else:
            point = xlwait(lambda: self.img2point(loclabel), limit=limit, interval=interval)
        return point

    def wait_leave(self, loclabel, *, fixpos=False, limit=None, interval=1):
        """ 封装的比较高层的功能 """
        if fixpos:
            xlwait(lambda: self.check_img(loclabel))
            point = self[loclabel]['center']
        else:
            point = xlwait(lambda: not self.img2point(loclabel), limit=limit, interval=interval)
        return point

    def check_click(self, loclabel, *, back=False, wait_change=False):
        point = self.img2point(loclabel)
        if point:
            self.click(point, back=back, wait_change=wait_change)

    def wait_click(self, loclabel, *, fixpos=False, back=False, wait_change=False):
        """ 封装的比较高层的功能 """
        point = self.wait(loclabel, fixpos=fixpos)
        self.click(point, back=back, wait_change=wait_change)


if __name__ == '__main__':
    # agld = AutoGuiLabelData(r'D:\slns\py4101\py4101\touhou\label')
    agld = AutoGuiLabelData(r'D:\home\chenkunze\data\m2508凡修')
    agld.writes()
