#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/05

"""
以uiautomation为核心的相关工具库
"""

import os
import sys
import textwrap
import time
from typing import Iterable, Callable, List
import subprocess
import tempfile

import psutil
import pandas as pd
from fastcore.basics import GetAttr

from loguru import logger
# ui组件大多是树形组织结构，auto库自带树形操作太弱。没有专业的树形结构库，能搞个毛线。
from anytree import NodeMixin

import ctypes
from ctypes import wintypes

if sys.platform == 'win32':
    import win32con
    import win32gui
    import win32process
    import win32clipboard

    import uiautomation as uia
    from uiautomation import WindowControl


def __1_clipboard_utils():
    pass


def retry_on_failure(max_retries: int = 5):
    """
    一个装饰器，用于在失败时重试执行被装饰的函数。

    Args:
        max_retries (int): 最大重试次数。

    Returns:
        Callable: 包装后的函数。
    """

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    if func(*args, **kwargs):
                        return True
                except Exception as e:
                    time.sleep(.05)
                    print(f"Attempt {attempt + 1} failed: {e}")
            return False

        return wrapper

    return decorator


def set_clipboard_data(fmt: int, buf: ctypes.Array) -> bool:
    """
    将数据设置到Windows剪切板中。

    Args:
        fmt (int): 数据格式，例如 win32clipboard.CF_HDROP。
        buf (ctypes.Array): 要设置到剪切板的数据。

    Returns:
        bool: 操作成功返回 True，否则返回 False。
    """
    try:
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(fmt, buf)
        return True
    except Exception as e:
        print(f"Error setting clipboard data: {e}")
        return False
    finally:
        win32clipboard.CloseClipboard()


def get_clipboard_files() -> List[str]:
    """
    获取剪切板中的文件路径列表。

    Returns:
        List[str]: 包含剪切板中文件路径的列表，如果没有文件路径或操作失败，返回空列表。
    """
    try:
        win32clipboard.OpenClipboard()
        if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_HDROP):
            return list(win32clipboard.GetClipboardData(win32clipboard.CF_HDROP))
        else:
            return list()
    finally:
        win32clipboard.CloseClipboard()


@retry_on_failure(max_retries=5)
def validate_clipboard_files(file_paths: Iterable[str], fmt: int, buf: ctypes.Array) -> bool:
    """
    验证剪切板中的文件路径是否与给定的文件路径一致。

    Args:
        file_paths (Iterable): 一个包含文件路径的可迭代对象，每个路径都是一个字符串。
        fmt (int): 数据格式，例如 win32clipboard.CF_HDROP。
        buf (ctypes.Array): 要验证的剪切板数据。

    Returns:
        bool: 如果剪切板中的文件路径与给定的文件路径一致，则返回 True

    Raises:
        ValueError: 如果剪切板文件路径与给定文件路径不一致。
    """
    # 设置文件到剪切板
    set_clipboard_data(fmt, buf)
    # 验证剪切板中的文件路径是否与给定的文件路径一致
    if set(get_clipboard_files()) == set(file_paths):
        return True
    raise ValueError("剪切板文件路径不对哇！")


def copy_files_to_clipboard(file_paths: Iterable[str]) -> bool:
    """
    将一系列文件路径复制到Windows剪切板。这允许用户在其他应用程序中，如文件资源管理器中粘贴这些文件。

    Args:
        file_paths (Iterable): 一个包含文件路径的可迭代对象，每个路径都是一个字符串。

    Returns:
        bool: 如果成功将文件路径复制到剪切板，则返回 True，否则返回 False
    """
    # 定义所需的 Windows 结构和函数
    CF_HDROP = 15

    class DROPFILES(ctypes.Structure):
        _fields_ = [("pFiles", wintypes.DWORD),
                    ("pt", wintypes.POINT),
                    ("fNC", wintypes.BOOL),
                    ("fWide", wintypes.BOOL)]

    offset = ctypes.sizeof(DROPFILES)
    length = sum(len(p) + 1 for p in file_paths) + 1
    size = offset + length * ctypes.sizeof(wintypes.WCHAR)
    buf = (ctypes.c_char * size)()
    df = DROPFILES.from_buffer(buf)
    df.pFiles, df.fWide = offset, True
    for path in file_paths:
        path = os.path.normpath(path)
        array_t = ctypes.c_wchar * (len(path) + 1)
        path_buf = array_t.from_buffer(buf, offset)
        path_buf.value = path
        offset += ctypes.sizeof(path_buf)
    buf[offset:offset + ctypes.sizeof(wintypes.WCHAR)] = b'\0\0'

    # 验证文件是否成功复制到剪切板
    return validate_clipboard_files([os.path.normpath(file) for file in file_paths], CF_HDROP, buf=buf)


def __2_窗口功能():
    pass


def get_windows_info():
    """ 得到当前机器的全部窗口信息清单 """
    window_list = []

    def get_all_hwnd(hwnd, mouse):
        thread_id, process_id = win32process.GetWindowThreadProcessId(hwnd)
        try:
            proc = psutil.Process(process_id)
        except psutil.NoSuchProcess:
            return

        is_window = win32gui.IsWindow(hwnd)
        is_enabled = win32gui.IsWindowEnabled(hwnd)
        is_visible = win32gui.IsWindowVisible(hwnd)
        text = win32gui.GetWindowText(hwnd)

        data = {
            'proc_name': proc.name(),
            'process_id': process_id,
            'thread_id': thread_id,
            'hwnd': hwnd,
            'ClassName': win32gui.GetClassName(hwnd),
            'ControlTypeName': '',
            'WindowText': text,
            'IsWindow': is_window,
            'IsWindowEnabled': is_enabled,
            'IsWindowVisible': is_visible
        }

        if not data['proc_name'].endswith('.tmp') and is_visible:
            ctrl = uia.ControlFromHandle(hwnd)
            data['ControlTypeName'] = ctrl.ControlTypeName

        window_list.append(data)

    win32gui.EnumWindows(get_all_hwnd, 0)
    return pd.DataFrame(window_list)


def find_ctrl(class_name=None, name=None, **kwargs):
    if class_name is not None:
        kwargs['ClassName'] = class_name
    if name is not None:
        kwargs['Name'] = name
    ctrl = uia.WindowControl(**kwargs)
    return ctrl


class UiCtrlNode(NodeMixin, GetAttr, WindowControl):
    _default = 'ctrl'

    def __0_构建(self):
        pass

    def __init__(self, ctrl, parent=None, *, build_depth=-1):
        """
        :param ctrl: 当前节点
        :param parent: 父结点
        :param build_depth: 自动构建多少层树节点，默认-1表示构建全部节点
        """
        # 初始化节点信息
        self.ctrl = ctrl
        # 试过了，没用，因为新找出来的都是新构建的类，找不到proxy的
        # self.ctrl.proxy: 'UiCtrlNode' = self  # 再给其扩展一个proxy属性，指向其升级过的对象
        self.ctrl_type = ctrl.ControlTypeName
        self.text = ctrl.Name
        self.parent = parent  # 指定父节点，用于形成树结构

        # 自动递归创建子节点
        self.build_children(build_depth)

    @property
    def ltrb(self):
        rect = self.ctrl.BoundingRectangle
        return [rect.left, rect.top, rect.right, rect.bottom]

    @property
    def ltwh(self):
        rect = self.ctrl.BoundingRectangle
        l, t, r, b = [rect.left, rect.top, rect.right, rect.bottom]
        w, h = r - l, b - t
        return [l, t, w, h]

    @classmethod
    def init_from_name(cls, class_name=None, name=None, *, build_depth=-1, **kwargs):
        ctrl = find_ctrl(class_name=class_name, name=name, **kwargs)
        return cls(ctrl, build_depth=build_depth)

    def activate(self, check_seconds=2):
        """ 激活当前窗口
        """
        while True:
            # todo 这种限定情况并不严谨，有概率出现重复的~
            hwnd = win32gui.FindWindow(self.ctrl.ClassName, self.text)
            # logger.info(hwnd)
            if not hwnd:
                return

            if win32gui.GetForegroundWindow() != hwnd:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

                try:
                    # 在这里执行SetForegroundWindow，只有程序的第1次运行有效，之后就会被很多全屏类的应用占用最前置，覆盖不了了
                    # 为了解决这问题，就只能暴力每次都新开一个程序来执行这个SetForegroundWindow操作
                    subprocess.run(
                        [sys.executable, "-c", f"import win32gui; win32gui.SetForegroundWindow({hwnd})"],
                        stdout=subprocess.PIPE,
                    )
                except Exception as e:
                    pass
                # 理论上并不需要等待，但加个等待，有助于稳定性检测，如果当前窗口在check_seconds秒内频繁切换，
                #   使用activate虽然激活了，但并不安全，只有check_seconds秒内维持稳定在这个窗口，再进行下游任务会更好
                time.sleep(check_seconds)
            else:
                break

    def build_children(self, build_depth=-1, child_node_class=None):
        """ 创建并添加子节点到树中 """
        if build_depth == 0:
            return
        self.children = []  # 删除现有的所有子结点
        child_node_class = child_node_class or self.__class__
        for child_ctrl in self.ctrl.GetChildren():
            child_node_class(child_ctrl, parent=self, build_depth=build_depth - 1)

    def __1_调试(self):
        pass

    def trace_rect(self, duration_per_circle=2, num_circles=1):
        """ 用鼠标勾画出当前组件的矩形位置区域 """
        from pyxllib.autogui.autogui import UiTracePath

        rect = self.ctrl.BoundingRectangle
        ltrb = [rect.left, rect.top, rect.right, rect.bottom]
        UiTracePath.from_ltrb(ltrb,
                              duration_per_circle=duration_per_circle,
                              num_circles=num_circles)

    def __2_功能(self):
        pass

    def __getattr__(self, item):
        # 尝试从self.ctrl中获取属性
        return getattr(self.ctrl, item)

    def __getitem__(self, index):
        """ 通过索引直接访问子节点

        ui操作经常要各种结构化的访问，加个这个简化引用方式
        """
        try:
            return self.children[index]
        except IndexError:  # 如果出现下标错误，需要自动重新刷新所有控件
            # self.parent重建是不够的，但我也不知道为什么self.root重建后就可以了
            # 我的理解是重建后self自己不是都不存在了？
            self.root.build_children()
            # 应该在有些情况下self.root重建还能继续使用，但有些特殊情况应该会炸
            return self.children[index]

    def get_ctrl_hash_tag(self, level=1):
        """ 生成节点的哈希字符串，以反映子树结构，一般用来对节点做分类及映射到对应处理函数 """
        # 当前节点的类型标识符
        hash_strs = [f"{level}{self.ctrl_type[0].lower()}"]
        # 遍历所有子节点，递归生成子节点的哈希值
        for child in self.children:
            hash_strs.append(f"{child.get_ctrl_hash_tag(level + 1)}")
        return ''.join(hash_strs)

    def __3_展示(self):
        pass

    def _format_text(self, text):
        """ 将换行替换为空格的小工具方法 """
        return text.replace('\n', ' ')

    def __repr__(self):
        """ 用于在打印节点时显示关键信息 """
        return f"UiNode(ctrl_type={self.ctrl_type}, text={self._format_text(self.text)})"

    def render_tree(self):
        """ 展示以self为根节点的整体内容结构 """
        # 1 渲染自身
        line = [self.ctrl_type]
        line.append(self._format_text(self.text))

        # 加上控件的坐标信息
        rect = self.ctrl.BoundingRectangle
        line.append(f"[{rect.left}, {rect.top}, {rect.right}, {rect.bottom}]")

        # 我的hash值
        tag = self.get_ctrl_hash_tag()
        if len(tag) <= 64:
            line.append(tag)

        lines = [' '.join(line)]

        # 2 子结点情况
        for child in self.children:
            line = child.render_tree()
            line = textwrap.indent(line, '    ')
            lines.append(line)
        return '\n'.join(lines)
