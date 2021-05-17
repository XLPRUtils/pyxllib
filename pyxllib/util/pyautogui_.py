#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/06


from pyxllib.cv import *

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

import pyscreeze  # NOQA pyautogui安装的时候会自动安装依赖的pyscreeze


class NamedLocate:
    """ 对有命名的标注数据进行定位

    注意labelme在每张图片上写的label值最好不要有重复，否则按字典存储后可能会被覆盖

    特殊取值：
    '@IMAGE_ID' 该状态图的标志区域，即游戏中在该位置出现此子图，则认为进入了对应的图像状态

    TODO 生成一个简略的概括txt，方便快速查阅有哪些接口；或者生成某种特定格式的python
    TODO 吃土乡人物清单要用特殊技巧去获得，不能手动暴力搞；可以自动循环拼接所有图
    """

    def __init__(self, label_dir, *, grayscale=None, confidence=0.95, tolerance=20, fix_pos=True):
        """

        :param label_dir: 标注数据所在文件夹
            有原图jpg、png和对应的json标注数据
        :param grayscale: 转成灰度图比较
        :param confidence: 用于图片比较，相似度阈值
        :param tolerance: 像素比较时能容忍的误差变化范围
        :param fix_pos: 使用固定位置可以提高运行速度，如果窗口位置是移动的，则可以关闭该功能

        TODO random_click: 增加该可选参数，返回point时，进行上下左右随机扰动
        """
        self.root = Dir(label_dir)
        self.grayscale = grayscale
        self.confidence = confidence
        self.tolerance = tolerance
        self.fix_pos = fix_pos
        self.points, self.rects, self.images = dict(), dict(), dict()
        self._init()
        self.last_shot = self.update_shot()

    def update_from_shape(self, stem, img, shape):
        # 0 key
        if shape['label'] == '@IMAGE_ID':
            key = f'{stem}'
        else:
            key = f'{stem}/{shape["label"]}'

        # 1 读取point
        # 无论任何形状，都取其points的均值作为中心点
        # 这在point、rectangle、polygon、line等都是正确的
        # circle类计算出的虽然不是中心点，但也在圆内（circle第1个点圆心，第2个点是圆周上的1个点）
        # 目前情报shape都是有points成员的，还没见过没有points的类型
        point = list(np.array(np.array(shape['points']).mean(axis=0), dtype=int))
        self.points[key] = point

        # 2 读取rect
        rect = None
        if shape['shape_type'] == 'rectangle':
            rect = np.array(shape['points'], dtype=int).reshape(-1).tolist()
            self.rects[key] = rect

        # 3 读取image
        if rect:
            subimg = get_sub_image(img, rect)
            # temp = File(..., Dir.TEMP, suffix=img_file.suffix)
            # imwrite(subimg, str(temp))
            # temp2 = temp.rename(get_etag(str(temp)) + temp.suffix, if_exists='delete')
            # images[key] = temp2
            self.images[key] = subimg
        elif point:  # 用list存储这个位置的像素值，顺序改为RGB
            self.images[key] = tuple(img[point[1], point[0]].tolist()[::-1])

    def _init(self):
        """
        points 目标的中心点坐标
        rects 目标的矩形框位置
        images 从rects截取出来的子图，先存在临时目录，字典值记录了图片所在路径（后还是改成了存储图片）
            对point对象，则存储了对应位置的像素值
        """
        for f in self.root.select(['**/*.jpg', '**/*.png']).subfiles():
            stem, suffix = f.stem, f.suffix
            img_file = File(f, self.root)
            json_file = img_file.with_suffix('.json')
            if not json_file:
                continue
            img = imread(img_file)

            shapes = json_file.read()['shapes']
            for shape in shapes:
                self.update_from_shape(stem, img, shape)

    def update_shot(self, region=None):
        self.last_shot = pil2cv(pyautogui.screenshot(region=region))
        return self.last_shot

    def screenshot(self, region=None):
        if isinstance(region, str):
            im = pyautogui.screenshot(region=ltrb2xywh(self.rects[region]))
        else:
            im = pyautogui.screenshot(region=region)
        return pil2cv(im)

    def get_pixel(self, point):
        """
         官方的pixel版本，在windows有时候会出bug
         OSError: windll.user32.ReleaseDC failed : return 0
        """
        if isinstance(point, str):
            point = self.points[point]
        return tuple(self.last_shot[point[1], point[0]].tolist()[::-1])

    def get_image(self, ltrb):
        if isinstance(ltrb, str):
            ltrb = self.rects[ltrb]
        l, t, r, b = ltrb
        return self.last_shot[t:b, l:r]

    @classmethod
    def _cmp_pixel(cls, pixel1, pixel2):
        """ 返回最大的像素值差，如果相同返回0 """
        return max([abs(x - y) for x, y in zip(pixel1, pixel2)])

    @classmethod
    def _cmp_image(cls, image1, image2):
        """ 返回不相同像素占的百分比，相同返回0 """
        cmp = np.array(abs(np.array(image1, dtype=int) - image2) > 10)  # 每个像素允许10的差距
        return cmp.sum() / cmp.size

    def _locate_point(self, name):
        """ 这里shot是指不那么考虑内容是否匹配，返回能找到的一个最佳结果
                后续会有find来进行最后的阈值过滤
        """
        if self.fix_pos:
            return self.points[name]
        else:
            raise NotImplementedError

    def locate_rects(self, name):
        self.update_shot()
        boxes = pyautogui.locateAll(self.images[name],
                                    self.last_shot,
                                    grayscale=self.grayscale,
                                    confidence=self.confidence)
        # 过滤掉重叠超过一半面积的框
        return non_maximun_suppression([xywh2ltrb(box) for box in list(boxes)], 0.5)

    def locate_rect(self, name, *, fix_pos=None):
        fix_pos = fix_pos if fix_pos is not None else self.fix_pos
        if fix_pos:
            return self.rects[name]
        else:
            res = self.locate_rects(name)
            return res[0] if res else None

    def find(self, name):
        self.update_shot()
        if name in self.rects:  # 优先按照区域规则进行匹配分析

            rect = self.locate_rect(name)
            # print(name, '图片相似度', 1 - self._cmp_image(self.images[name], self.get_image(rect)))
            if (1 - self._cmp_image(self.images[name], self.get_image(rect))) >= self.confidence:
                return rect
            else:
                return False
        elif name in self.points:  # 否则进行像素匹配
            point = self._locate_point(name)
            if self._cmp_pixel(self.images[name], self.get_pixel(point)) <= self.tolerance:
                return point
            else:
                return False
        else:
            raise ValueError(f'{name}')

    def find_point(self, name):
        """ 将find的结果统一转成点 """
        res = self.find(name)
        if not res:
            return False
        else:
            n = len(res)
            if n == 4:
                res = list(np.array(np.array(res).reshape(2, 2).mean(axis=0), dtype=int))
            elif n != 2:
                raise ValueError
            return res

    def find_image(self, name):
        """ 将find的结果统一转成图片数据 """
        res = self.find(name)
        if not res:
            return False
        else:
            n = len(res)
            if n == 4:
                res = self.get_image(res)
            elif n != 2:
                res = self.get_pixel(res)
            return res

    def try_click(self, name, *, back=False):
        """ 检查是否出现了目标，有则点击，否则不进行任何操作

        :param back: 点击后将鼠标move回原来的位置
        """
        pos0 = pyautogui.position()
        if isinstance(name, (tuple, list)) and len(name) == 2:  # 直接输入坐标
            pos = name
        elif isinstance(name, str):
            pos = self.find_point(name)
        else:
            raise TypeError

        if pos:
            # print(pos)
            pyautogui.click(*pos)
        if back:
            pyautogui.moveTo(*pos0)  # 恢复鼠标原位置
        return pos

    def click_point(self, name, *, back=False):
        """ 不进行内容匹配，直接点击对应位置的点 """
        if isinstance(name, str):
            name = self.points[name]
        return self.try_click(name, back=back)

    def wait(self, name, *, limit_seconds=None, interval_seconds=1, back=False):
        pos = None
        t = TicToc()
        while not pos:
            pos = self.find_point(name)
            time.sleep(interval_seconds)
            if limit_seconds and t.tocvalue() > limit_seconds:
                break
        return pos

    def wait_leave(self, name, *, limit_seconds=None, interval_seconds=1, back=False):
        """ 和wait逻辑相反，确保当前界面没有name元素的时候结束循环，没有返回值 """
        pos = True
        t = TicToc()
        while pos:
            pos = self.find_point(name)
            time.sleep(interval_seconds)
            if limit_seconds and t.tocvalue() > limit_seconds:
                break

    def wait_click(self, name, *, limit_seconds=None, interval_seconds=1, back=False):
        """ 等待图标出现运行成功后再点击

        :poram limit_seconds: 等待秒数上限
        :param interval_seconds: 每隔几秒检查一次
        :param back: 点击后将鼠标move回原来的位置
        """
        pos = self.wait(name, limit_seconds=limit_seconds, interval_seconds=interval_seconds)
        self.try_click(pos, back=back)

    def move_to(self, name):
        pyautogui.moveTo(*self.points[name])


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
