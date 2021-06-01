#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/06

from collections import defaultdict

from pyxllib.basic.most import *
from pyxllib.cv.most import *
from pyxllib.data.label import *

import pyautogui
import keyboard
import pyscreeze  # NOQA pyautogui安装的时候会自动安装依赖的pyscreeze


class AutoGuiLabelData:
    """ AutoGuiLabelData """

    def __init__(self, root):
        """
        data：dict
            key: 相对路径/图片stem （这一节称为loc）
                label名称：对应属性dict1 （跟loc拼在一起，称为loclabel）
                label2：dict2
        """
        self.root = Dir(root)
        self.data = defaultdict(dict)
        self.imfiles = {}

        for file in self.root.select('**/*.json').subfiles():
            lmdata = file.read()
            imfile = file.with_name(lmdata['imagePath'])
            img = imread(imfile)
            loc = file.relpath(self.root).replace('\\', '/')[:-5]
            self.imfiles[loc] = imfile

            for shape in lmdata['shapes']:
                attrs = self.parse_shape(shape, img)
                self.data[loc][attrs['label']] = attrs

    @classmethod
    def parse_shape(cls, shape, image=None):
        """ 解析一个shape的数据为dict字典 """
        # 1 解析原label为字典
        label = shape['label']
        attrs = shape.copy()
        try:
            data = json.loads(label)
            if isinstance(data, dict):
                attrs.update(data)
        except json.decoder.JSONDecodeError:
            pass
        # 如果label是普通字符串，则labelattr强升为字典
        if not attrs:
            attrs['label'] = label

        # 2 中心点 center
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
            attrs['ltrb'] = [int(v) for v in [x - r, y - r, x + r, y + r]]

        # 4 图片数据 img, etag
        if image is not None and attrs['ltrb']:
            attrs['img'] = get_sub_image(image, attrs['ltrb'])
            # attrs['etag'] = get_etag(attrs['img'])
            # TODO 这里目的其实就是让相同图片对比尽量相似，所以可以用dhash而不是etag

        # 5 中心点像素值 pixel
        p = attrs['center']
        if image is not None and p:
            attrs['pixel'] = tuple(image[p[1], p[0]].tolist()[::-1])

        # if 'rect' in attrs:
        #     del attrs['rect']  # 旧版的格式数据，删除

        return attrs

    def update_loclabel_img(self, loclabel, img):
        """ 添加一张图片数据 """
        loc, label = os.path.split(loclabel)
        h, w = img.shape[:2]

        # 1 如果loc图片不存在，则新建一个jpg图片
        if loc not in self.data:
            imfile = imwrite(img, File(loc, self.root, suffix='.jpg'))
            self.imfiles[loc] = imfile
            shape = LabelmeData.gen_shape(label, [[0, 0], [w, h]])
            self.data[loc][label] = self.parse_shape(shape)
        # 2 不存在的标签，则在最后一行新建一张图
        elif label not in self.data[loc]:
            image = imread(self.imfiles[loc])
            height, width = image.shape[:2]
            assert width == w  # 先按行拼接，以后有时间可以扩展更灵活的拼接操作
            # 拼接，并重新存储为图片
            image = np.concatenate([image, img])
            imwrite(image, self.imfiles[loc])
            shape = LabelmeData.gen_shape(label, [[0, height], [width, height + h]])
            self.data[loc][label] = self.parse_shape(shape)
        # 3 已有的图，则进行替换
        else:
            image = imread(self.imfiles[loc])
            [x1, y1, x2, y2] = self.data[loc][label]['ltrb']
            image[y1:y2, x1:x2] = img
            imwrite(image, self.imfiles[loc])
        # 该更新，需要实时保存到文件中
        self.write(loc)

    def write(self, loc):
        f = File(loc, self.root, suffix='.json')
        imfile = self.imfiles[loc]
        lmdata = LabelmeData.gen_data(imfile)
        for label, ann in self.data[loc].items():
            a = ann.copy()
            if 'img' in a:
                del a['img']
            shape = LabelmeData.gen_shape(json.dumps(a, ensure_ascii=False),
                                          a['points'], a['shape_type'],
                                          group_id=a['group_id'], flags=a['flags'])
            lmdata['shapes'].append(shape)
        f.write(lmdata, indent=2)

    def writes(self):
        for loc in self.data.keys():
            self.write(loc)

    def __getitem__(self, loclabel):
        loc, label = os.path.split(loclabel)
        return self.data[loc][label]

    def __setitem__(self, loclabel, value):
        loc, label = os.path.split(loclabel)
        self.data[loc][label] = value


def xlwait(func, condition=bool, *, limit=None, interval=1):
    """ 不断重复执行func，直到得到满足condition条件的期望值

    :param condition: 退出等待的条件，默认为bool真值
    :param limit: 重复执行的上限时间（单位 秒），默认一直等待
    :param interval: 重复执行间隔 （单位 秒）

    """
    t = TicToc()
    while True:
        res = func()
        if condition(res):
            return res
        elif limit and t.tocvalue() > limit:
            return res  # 超时也返回目前得到的结果
        time.sleep(interval)


class NamedLocate(AutoGuiLabelData):
    """ 对有命名的标注数据进行定位

    注意labelme在每张图片上写的label值最好不要有重复，否则按字典存储后可能会被覆盖

    特殊取值：
    '@IMAGE_ID' 该状态图的标志区域，即游戏中在该位置出现此子图，则认为进入了对应的图像状态

    TODO 生成一个简略的概括txt，方便快速查阅有哪些接口；或者生成某种特定格式的python
    TODO 吃土乡人物清单要用特殊技巧去获得，不能手动暴力搞；可以自动循环拼接所有图
    """

    def __init__(self, root, *, grayscale=None, confidence=0.95, tolerance=20):
        """

        :param root: 标注数据所在文件夹
            有原图jpg、png和对应的json标注数据
        :param grayscale: 转成灰度图比较
        :param confidence: 用于图片比较，相似度阈值
        :param tolerance: 像素比较时能容忍的误差变化范围

        TODO random_click: 增加该可选参数，返回point时，进行上下左右随机扰动
        """
        super().__init__(root)
        self.grayscale = grayscale
        self.confidence = confidence
        self.tolerance = tolerance
        self.last_shot = self.update_shot()

    def screenshot(self, region=None):
        """
        :param region: ltrb
        """
        if isinstance(region, str):
            region = ltrb2xywh(self[region])
        im = pyautogui.screenshot(region=region)
        return pil2cv(im)

    def update_shot(self, region=None):
        self.last_shot = self.screenshot(region)
        return self.last_shot

    def point2pixel(self, point):
        """

        :param point: 支持输入 (x, y) 坐标，或者 loclabel
            后面很多参数都是类似，除了标准数据类型，同时支持 loclabel 定位模式

         官方的pixel版本，在windows有时候会出bug
         OSError: windll.user32.ReleaseDC failed : return 0
        """
        if isinstance(point, str):
            point = self[point]['center']
        return tuple(self.last_shot[point[1], point[0]].tolist()[::-1])

    def rect2img(self, ltrb):
        """
        :param ltrb: 可以输入坐标定位，也可以输入 loclabel 定位
        """
        if isinstance(ltrb, str):
            ltrb = self[ltrb]['ltrb']
        l, t, r, b = ltrb
        return self.last_shot[t:b, l:r]

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

    def check_pixel(self, loclabel, *, tolerance=None):
        """ 判断对应位置上的像素值是否相同
        """
        tolerance = tolerance if tolerance is not None else self.tolerance
        p1 = self[loclabel]['pixel']
        p2 = self.point2pixel(loclabel)
        return self.pixel_distance(p1, p2) < tolerance

    def check_img(self, loclabel, *, grayscale=None, confidence=None):
        grayscale = grayscale if grayscale is not None else self.grayscale
        confidence = confidence if confidence is not None else self.confidence
        boxes = pyautogui.locateAll(self[loclabel]['img'], self.screenshot(loclabel),
                                    grayscale=grayscale, confidence=confidence)
        boxes = [xywh2ltrb(box) for box in list(boxes)]
        return len(boxes)

    def img2rects(self, img, *, grayscale=None, confidence=None):
        """ 根据预存的img数据，匹配出多个内容对应的所在的rect位置 """
        # 1 配置参数
        if isinstance(img, str):
            img = self[img]['img']
        grayscale = grayscale if grayscale is not None else self.grayscale
        confidence = confidence if confidence is not None else self.confidence
        # 2 查找子图
        self.update_shot()
        boxes = pyautogui.locateAll(img, self.last_shot,
                                    grayscale=grayscale,
                                    confidence=confidence)
        # 3 过滤掉重叠超过一半面积的框
        return ComputeIou.nms_ltrb([xywh2ltrb(box) for box in list(boxes)])

    def img2rect(self, img, *, grayscale=None, confidence=None):
        """ img2rects的简化，只返回一个匹配框
        """
        rects = self.img2rects(img, grayscale=grayscale, confidence=confidence)
        return rects[0] if rects else None

    def img2point(self, img, *, grayscale=None, confidence=None):
        """ 将 img2rect 的结果统一转成点 """
        res = self.img2rect(img, grayscale=grayscale, confidence=confidence)
        if res:
            return np.array(np.array(res).reshape(2, 2).mean(axis=0), dtype=int).tolist()

    def img2img(self, img, *, grayscale=None, confidence=None):
        """ 找到rect后，返回匹配的目标img图片内容 """
        ltrb = self.img2rect(img, grayscale=grayscale, confidence=confidence)
        if ltrb:
            l, t, r, b = ltrb
            return self.last_shot[t:b, l:r]

    def click(self, point, *, back=False, wait_change=False):
        """
        :param back: 点击后鼠标移回原坐标位置
        :param wait_change: 点击后，等待画面产生变化，才结束函数
            point为 loclabel 才能用这个功能
        """
        # 1 判断类型
        if isinstance(point, str):
            loclabel = point
            point = self[loclabel]['center']
        else:
            loclabel = ''

        # 2 鼠标移动
        pos0 = pyautogui.position()
        pyautogui.click(*point)
        if back:
            pyautogui.moveTo(*pos0)  # 恢复鼠标原位置

        # 3 等待画面变化
        if loclabel and wait_change:
            func = lambda: self.img_distance(self[loclabel]['img'],
                                             self.rect2img(loclabel)) > self.confidence
            xlwait(func)

        return point

    def move_to(self, point):
        if isinstance(point, str):
            point = self[point]['center']
        pyautogui.moveTo(*point)

    def wait(self, loclabel, *, fixpos=False):
        """ 封装的比较高层的功能 """
        if fixpos:
            xlwait(lambda: self.check_img(loclabel))
            point = self[loclabel]['center']
        else:
            point = xlwait(lambda: self.img2point(loclabel))
        return point

    def check_click(self, loclabel, *, back=False, wait_change=False):
        point = self.img2point(loclabel)
        if point:
            self.click(point, back=back, wait_change=wait_change)

    def wait_click(self, loclabel, *, fixpos=False, back=False, wait_change=False):
        """ 封装的比较高层的功能 """
        point = self.wait(loclabel, fixpos=fixpos)
        self.click(point, back=back, wait_change=wait_change)


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
    """ 查看鼠标位置的工具
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
            print('区域(x y w h)：', left_top_point.x, left_top_point.y, p.x - left_top_point.x, p.y - left_top_point.y)
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
            r1 = [left_top_point.x, left_top_point.y, p.x - left_top_point.x, p.y - left_top_point.y]
            r2 = postran.w2region(r1)
            if reverse: r1, r2 = r2, r1
            print('区域(x y w h)：', *r1, '-->', *r2)
        elif k == 'esc':
            break
        time.sleep(0.4)


____press_key = """
TODO 有待进一步封装整理

简化按键映射、简化PressKey和ReleaseKey过程
"""

if sys.platform == 'win32':
    import ctypes

    SendInput = ctypes.windll.user32.SendInput

    # C struct redefinitions
    PUL = ctypes.POINTER(ctypes.c_ulong)


    class KeyBdInput(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort),
                    ("wScan", ctypes.c_ushort),
                    ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong),
                    ("dwExtraInfo", PUL)]


    class HardwareInput(ctypes.Structure):
        _fields_ = [("uMsg", ctypes.c_ulong),
                    ("wParamL", ctypes.c_short),
                    ("wParamH", ctypes.c_ushort)]


    class MouseInput(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long),
                    ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong),
                    ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong),
                    ("dwExtraInfo", PUL)]


    class Input_I(ctypes.Union):
        _fields_ = [("ki", KeyBdInput),
                    ("mi", MouseInput),
                    ("hi", HardwareInput)]


    class Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong),
                    ("ii", Input_I)]


    # Actuals Functions

    def PressKey(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


    def ReleaseKey(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    # while True:
    #     PressKey(0x20)
    #     ReleaseKey(0x20)
    #     time.sleep(0.1)

if __name__ == '__main__':
    with TicToc(__name__):
        agld = AutoGuiLabelData(r'D:\slns\py4101\py4101\touhou\label')
        agld.writes()
