#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/05

"""
以uiautomation为核心的相关工具库
"""

import textwrap

import psutil
import pandas as pd

# ui组件大多是树形组织结构，auto库自带树形操作太弱。没有专业的树形结构库，能搞个毛线。
from anytree import NodeMixin

import win32gui
import win32process
import uiautomation as auto


def get_windows_info():
    """ 得到当前机器的全部窗口信息清单 """
    window_list = []

    def get_all_hwnd(hwnd, mouse):
        thread_id, process_id = win32process.GetWindowThreadProcessId(hwnd)
        proc = psutil.Process(process_id)

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
            ctrl = auto.ControlFromHandle(hwnd)
            data['ControlTypeName'] = ctrl.ControlTypeName

        window_list.append(data)

    win32gui.EnumWindows(get_all_hwnd, 0)
    return pd.DataFrame(window_list)


class UiCtrlNode(NodeMixin):
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
        self.ctrl_type = ctrl.ControlTypeName
        self.text = ctrl.Name
        self.parent = parent  # 指定父节点，用于形成树结构

        # 自动递归创建子节点
        self.build_children(build_depth)

    @classmethod
    def get_window(cls, class_name=None, name=None, *, build_depth=-1, **kwargs):
        # wake_up_window(class_name, name=name)
        if class_name is not None:
            kwargs['ClassName'] = class_name
        if name is not None:
            kwargs['Name'] = name
        ctrl = auto.WindowControl(**kwargs)
        return cls(ctrl, build_depth=build_depth)

    def build_children(self, build_depth, child_node_class=None):
        """ 创建并添加子节点到树中 """
        if build_depth == 0:
            return
        child_node_class = child_node_class or self.__class__
        for child_ctrl in self.ctrl.GetChildren():
            child_node_class(child_ctrl, parent=self, build_depth=build_depth - 1)

    def __1_调试(self):
        pass

    def trace_rect(self, duration_per_circle=3, num_circles=1):
        """ 用鼠标勾画出当前组件的矩形位置区域 """
        from pyxllib.autogui.autogui import UiTracePath

        rect = self.ctrl.BoundingRectangle
        ltrb = [rect.left, rect.top, rect.right, rect.bottom]
        UiTracePath.from_ltrb(ltrb,
                              duration_per_circle=duration_per_circle,
                              num_circles=num_circles)

    def __2_功能(self):
        pass

    def __getitem__(self, index):
        """ 通过索引直接访问子节点

        ui操作经常要各种结构化的访问，加个这个简化引用方式
        """
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
