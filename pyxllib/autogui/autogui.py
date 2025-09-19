#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/06

import sys
import time

from pyxllib.prog.lazyimport import lazy_import

try:
    import mss
except ModuleNotFoundError:
    mss = lazy_import('mss')

try:
    import pyautogui
except ModuleNotFoundError:
    pyautogui = lazy_import('pyautogui')

try:
    import win32gui
except ModuleNotFoundError:
    win32gui = lazy_import('win32gui', 'pywin32')

from pyxllib.algo.geo import ltrb2polygon


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


def type_text(text):
    """ 打印出文本内容

    相比pyautogui.write，这里支持中文等unicode格式

    这种需求一般也可以用剪切板实现，但是剪切板不够独立轻量，可能会有意想不到的副作用
    """
    from pynput.keyboard import Controller

    keyboard = Controller()
    keyboard.type(text)


def clipboard_decorator(rtype='text', *, copy=True, paste=False, typing=False):
    """ 装饰器，能方便地将函数的返回值转移到剪切板文本、富文本，甚至typing打印出来。

    Args:
        rtype: 函数返回值类型
        paste: 是否执行剪切板的粘贴功能
        typing: 是否键入文本内容

    Returns:

    """
    from bs4 import BeautifulSoup
    import pyautogui
    import klembord

    def decorator(func):
        def wrapper(*args, **kwargs):
            # 1 运行函数获得结果
            s = func(*args, **kwargs)

            # 2 复制到剪切板
            if copy:
                if rtype == 'text' and isinstance(s, str):
                    klembord.set_text(s)
                elif rtype == 'html' and isinstance(s, str):
                    s0 = BeautifulSoup(s, 'lxml').text
                    klembord.set_with_rich_text(s0, s)
                elif rtype == 'dict' and isinstance(s, dict):
                    klembord.set(s)

            # 3 输出
            if paste:
                pyautogui.hotkey('ctrl', 'v')  # 目前来看就这个方法最靠谱
            if typing:
                type_text(s)

            return s

        return wrapper

    return decorator


def get_clipboard_content(rich=False, *, head=False):
    """ klembord的get_with_rich_text获取富文本有点小问题，所以自己重写了一个版本
    好在有底层接口，所以也不难改

    :param rich: 是否返回html格式的富文本，默认True
    :param head: 富文本时，是否要返回<html><body>的标记
    """
    import re
    import html

    import klembord
    from klembord import W_HTML, W_UNICODE

    if not rich:
        return klembord.get_text()

    content = klembord.get((W_HTML, W_UNICODE))
    html_content = content[W_HTML]
    if html_content:
        html_content = html_content.decode('utf8')
        html_content = re.search(r'<body>\s*(?:<!--StartFragment-->)?(.+?)(?:<!--EndFragment-->)?\s*</body>',
                                 html_content).group(1)
    else:
        # 没有html的时候，可以封装成html格式返回
        text = content[W_UNICODE].decode('utf16')
        html_content = html.escape(text)

    if head:
        html_content = '<html><body>' + html_content + '</body></html>'

    return html_content


def set_clipboard_content(content, *, rich=False):
    """ 对klembord剪切板的功能做略微的简化 """
    from bs4 import BeautifulSoup
    import klembord

    if rich:
        klembord.set_with_rich_text(BeautifulSoup(content, 'lxml').text, content)
    else:
        klembord.set_text(content)


def grab_pixel_color(radius=0):
    """
    :param radius: 以中心像素为例，计算矩阵范围内的颜色分布情况（未实装）
    """
    from pyxllib.cv.rgbfmt import RgbFormatter
    last_pos = None
    print('精确全体/精确中文')

    while True:
        pos = pyautogui.position()
        if pos == last_pos:
            continue
        else:
            last_pos = pos

        color = RgbFormatter(*pyautogui.pixel(*pos))
        desc = color.relative_color_desc(color_range=2, precise_mode=True)
        print('\r' + f'{color.to_hex()} {desc}', end='', flush=True)
        time.sleep(0.1)


def grab_monitor(order=0, to_cv2=True):
    """ 截屏

    :param int order:
        默认0，截取全部屏幕
        1，截取第1个屏幕
        2，截取第2个屏幕
        ...
    :param to_cv2:
        True, 转成cv2格式的图片
        False, 使用pil格式的图片
    """
    from PIL import Image
    from pyxllib.xlcv import PilImg

    with mss.mss() as sct:
        assert order < len(sct.monitors), '屏幕下标越界'
        monitor = sct.monitors[order]
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        if to_cv2:
            img = PilImg.read(img).to_cv2_image()  # 返回是np矩阵
        return img


def list_windows(mode=1):
    """ 列出所有窗口

    :param mode:
        0, 列出所有窗口
        1, 只列出有名称，有宽、高的窗口
        str, 指定名称的窗口
            注意，当结果有多个的时候，一般后面那个才是紧贴边缘的窗口位置

    :return: [[窗口名, xywh], [窗口2, xywh]...]
    """
    import win32gui

    ls = []

    def callback(hwnd, extra):
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        w, h = r - l, b - t
        name = win32gui.GetWindowText(hwnd)
        if mode == 0:
            ls.append([name, [l, t, w, h]])
        elif mode == 1 and name and w and h:
            ls.append([name, [l, t, w, h]])
        elif isinstance(mode, str) and name == mode:
            ls.append([name, [l, t, w, h]])

    win32gui.EnumWindows(callback, None)
    return ls


class UiTracePath:
    """ 在window窗口上用鼠标勾画路径 """

    @classmethod
    def from_polygon(cls, points, duration_per_circle=3, num_circles=1):
        """
        让鼠标顺时针围绕任意多边形移动。

        :param points: 多边形顶点的坐标列表，按顺时针或逆时针顺序排列
        :param duration_per_circle: 每圈耗时
        :param num_circles: 总圈数
        """
        num_points = len(points)
        segment_duration = duration_per_circle / num_points  # 每条边的耗时

        for _ in range(num_circles):
            # 顺时针移动鼠标
            for i in range(num_points):
                start = points[i]
                pyautogui.moveTo(start[0], start[1])
                end = points[(i + 1) % num_points]
                # 使用 moveTo 的 duration 参数控制每条边的移动时间
                pyautogui.moveTo(end[0], end[1], duration=segment_duration)

    @classmethod
    def from_ltrb(cls, ltrb, **kwargs):
        points = ltrb2polygon(ltrb)
        return cls.from_polygon(points, **kwargs)
