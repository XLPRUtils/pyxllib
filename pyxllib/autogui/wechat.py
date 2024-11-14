#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/10/31

"""
代码参考(抄自)：https://github.com/Frica01/WeChatMassTool
"""

import time
from copy import deepcopy
from typing import Iterable
import os
from typing import (Iterable, Callable, List)
import json
import re
import textwrap

import uiautomation as auto
from anytree import NodeMixin, RenderTree

import win32con
import win32gui

from pyxllib.prog.pupil import print2string
from pyxllib.autogui.wechat_msg import msg_parsers
from pyxllib.autogui.uiautolib import find_ctrl, UiCtrlNode, copy_files_to_clipboard


def __1_config():
    pass


class WeChatConfig:
    WeChat_PROCESS_NAME = 'WeChat.exe'
    APP_NAME = 'WeChatMassTool'
    APP_PROCESS_NAME = 'WeChatMassTool.exe'
    APP_LOCK_NAME = 'WeChatMassTool.lock'
    WINDOW_NAME = '微信'
    WINDOW_CLASSNAME = 'WeChatMainWndForPC'


# WeChat = WeChatConfig


class IntervalConfig:
    BASE_INTERVAL = 0.1  # 基础间隔（秒）
    SEND_TEXT_INTERVAL = 0.05  # 发送文本间隔（秒）
    SEND_FILE_INTERVAL = 0.25  # 发送文件间隔（秒）
    MAX_SEARCH_SECOND = 0.1
    MAX_SEARCH_INTERVAL = 0.05


Interval = IntervalConfig


def __2_window_utils():
    pass


def minimize_wechat(class_name, name):
    """
    关闭Windows窗口

    Args:
        name(str):  进程名
        class_name(str):  进程class_name

    Returns:

    """
    hwnd = win32gui.FindWindow(class_name, name)
    if win32gui.IsWindowVisible(hwnd):
        win32gui.SendMessage(hwnd, win32con.WM_CLOSE, 0, 0)


def wake_up_window(class_name, name):
    """
    唤醒Windows窗口

    Args:
        name(str):  进程名
        class_name(str):  进程class_name

    Returns:

    """
    if hwnd := win32gui.FindWindow(class_name, name):
        # 恢复窗口
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        # 检查窗口是否已在前
        if win32gui.GetForegroundWindow() != hwnd:
            # 尝试将窗口置前
            try:
                win32gui.SetForegroundWindow(hwnd)
            except Exception as e:
                pass
                # print(f"尝试将窗口置前时出错: {e}")


def is_window_visible(class_name, name):
    """
    唤醒Windows 窗口的可见性

    Args:
        name(str):  进程名
        class_name(str):  进程class_name

    Returns:

    """
    if hwnd := win32gui.FindWindow(class_name, name):
        # 判断窗口可见性
        if win32gui.IsWindowVisible(hwnd):
            return True
    return False


def __3_wechat():
    pass


class MsgNode(UiCtrlNode):
    """ 一条消息的节点 """

    def __init__(self, msg_ctrl, parent=None, *, build_depth=-1):
        super().__init__(msg_ctrl, parent, build_depth=build_depth)
        self.time = None  # 每条消息都建立一个最近出现的时间作为时间戳？
        self.msg_type = None  # 消息类型：system, receive, send
        self.content_type = None  # 内容类型：text、image、file、time

        self.user = None  # 用户名
        self.user2 = None  # 群聊时用户会多一个群聊昵称
        self.cite_text = None  # 引用的内容，引用的不一定是文本，但是统一转成文本显示的

        self.render_text = None  # 供人预览，和简化给ai查看的格式

    def build_children(self, build_depth, child_node_class=None):
        if build_depth == 0:
            return
        self.children = []
        child_node_class = child_node_class or UiCtrlNode
        for child_ctrl in self.ctrl.GetChildren():
            child_node_class(child_ctrl, parent=self, build_depth=build_depth - 1)

    def init(self):
        """ 解析出更精细的结构化信息 """
        from pyxllib.autogui.wechat_msg import msg_parsers

        tag = self.get_ctrl_hash_tag()
        if tag in msg_parsers:
            msg_parsers[tag](self)

    def render_tree(self):
        """ 展示以self为根节点的整体内容结构 """
        # 1 未解析过的节点，原样输出树形结构供添加解析
        if not self.msg_type:
            return super().render_tree()

        # 2 已设置好预览文本的直接输出
        if self.render_text:
            return self.render_text

        # 3 否则用一套通用的机制渲染输出
        fmt = self._format_text
        # 定义类型到符号的映射
        type_to_symbol = {
            'send': '↑',
            'receive': '↓',
            'system': '⚙️'  # 假设 'system' 类型对应的符号是 ⚙️
        }
        # 根据 msg_type 获取相应的符号
        msg = [f"{type_to_symbol.get(self.msg_type, self.msg_type)}"]

        if self.user:
            msg.append(f"{self.user}: ")
        msg.append(fmt(self.text))

        if self.cite_text:
            msg.append(f"【引用】{fmt(self.cite_text)}")
        return ' '.join(msg)

    def is_match(self, msg_node):
        """ 两条msg_node是否对应的上 """
        from datetime import datetime

        # 1 比如内容一致
        if self.render_tree() == msg_node.render_tree():
            return 2  # 强一致

        # 2 或者"撤回"格式对应的上
        is_recall = (self.content_type == 'recall' or msg_node.content_type == 'recall')
        user_matches = is_recall and (self.user == msg_node.user or self.user2 == msg_node.user2)
        if is_recall and user_matches:
            return 1  # 弱一致

        # 3 判断是否为 'system'、'time' 类型
        is_system_time_type = (self.msg_type == msg_node.msg_type == 'time')
        if is_system_time_type:
            # 检查时间匹配，确保都是 datetime 类型，并只比较年月日时分部分
            same_time = (
                    isinstance(self.time, datetime) and
                    isinstance(msg_node.time, datetime) and
                    self.time.strftime('%Y-%m-%d %H:%M') == msg_node.time.strftime('%Y-%m-%d %H:%M')
            )
            if same_time:
                return 2

            if self.text == '以下是新消息' or msg_node.text == '以下是新消息':
                return 1

        # 4 如果有一条.content_type=='button_more'，也直接弱匹配
        if self.content_type == 'button_more' or msg_node.content_type == 'button_more':
            return 1


class MsgBoxNode(UiCtrlNode):
    """ 当前会话消息窗的节点 """

    def __init__(self, chat_box_ctrl, parent=None, *, build_depth=-1):
        super().__init__(chat_box_ctrl, parent, build_depth=build_depth)

    def build_children(self, build_depth=-1, child_node_class=None):
        # 1 遍历消息初始化
        if build_depth == 0:
            return
        self.children = []
        for child_ctrl in self.ctrl.GetChildren():
            node = MsgNode(child_ctrl, parent=self, build_depth=build_depth - 1)
            node.init()  # 节点扩展的初始化操作

        # 2 设置每条消息的时间
        last_time = None
        for c in self.children:
            if c.time:
                last_time = c.time
            else:
                c.time = last_time

    def findidx_system_time(self):
        """ 最后条系统时间标记 """
        for idx in range(len(self.children) - 1, -1, -1):
            if self.children[idx].content_type == 'time':
                return idx
        return -1

    def findidx_last_send(self):
        """ 最后条发出去的消息 """
        for idx in range(len(self.children) - 1, -1, -1):
            if self.children[idx].content_type == 'receive':
                return idx
        return -1

    def findidx_last_match(self, old_childrens):
        """
        :param old_chat_box: 旧的消息队列
        :return: 在当前 self.children 中首次匹配到的 old_chat_box 最后几条消息的起始下标
        """
        x, y = old_childrens, self.children
        n, m = len(x), len(y)

        def is_system(a):
            """ 非"撤回"的系统消息 """
            if a.msg_type == 'system' and a.content_type != 'recall':
                return True

        def check_bais(k):
            """ 检查偏移量k是否能匹配 """
            # i, j分别指向"最后一条"数据，然后开始匹配
            i, j = n - 1, m - k - 1
            while min(i, j) > 0:  # 不匹配第0条，第0条太特别
                if is_system(x[i]):
                    i -= 1
                    continue
                if is_system(y[j]):
                    j -= 1
                    continue
                if y[j].is_match(x[i]):
                    i, j = i - 1, j - 1
                    continue
                return False
            return True

        for k in range(m):
            if check_bais(k):
                return m - k - 1

        return -1  # 如果没有匹配，返回 -1

    def check_update(self):
        """ 更新当前消息记录，并返回新收到的消息节点 """
        import copy
        old_childrens = copy.copy(self.children)
        self.build_children()
        idx = self.findidx_last_match(old_childrens)
        return self.children[idx + 1:]


class EditorNode(UiCtrlNode):
    """ 编辑器节点 """


class WeChatMainWnd(UiCtrlNode):

    def __1_结构建构(self):
        pass

    def __init__(self, ctrl=None, parent=None, *, build_depth=12):
        """
        :param ctrl: 当前节点
        :param parent: 父结点
        :param build_depth: 自动构建多少层树节点，默认-1表示构建全部节点
            目前微信发现12层一般够大部分情况使用了
        """
        if ctrl is None:
            ctrl = find_ctrl('WeChatMainWndForPC', '微信')
        super().__init__(ctrl, parent=parent, build_depth=build_depth)

    def build_children(self, build_depth=12, child_node_class=None):
        # 1 构建树结构
        if build_depth == 0:
            return

        # 跳过两级
        self.children = []  # 删除现有的所有子结点
        ctrl = self.ctrl.GetChildren()[-1].GetChildren()[0]
        for child_ctrl in ctrl.GetChildren():
            UiCtrlNode(child_ctrl, parent=self, build_depth=build_depth - 1)

        # 2 预先定义好一些常用的控件
        self.nav = self[0]  # 导航
        self.nav_avatar = self.nav[0]  # 导航_头像
        self.nav_chat = self.nav[1]  # 导航_聊天
        self.nav_contacts = self.nav[2]  # 导航_通讯录
        self.nav_favorites = self.nav[3]  # 导航_收藏
        self.nav_files = self.nav[4]  # 导航_聊天文件
        self.nav_moments = self.nav[5]  # 导航_朋友圈

    def __2_常用控件(self):
        pass

    @property
    def column3(self):
        # 订阅号、聊天都共通
        return self[2][0]

    @property
    def chat_window(self):
        return self.column3[0][0][0]

    @property
    def msg_box(self):
        parent = self.chat_window[1][0][0]
        if not isinstance(parent[0], MsgBoxNode):  # 只初始化一次
            parent.children = (MsgBoxNode(parent[0]), *parent.children[1:])
        return parent[0]

    @property
    def edit_box(self):
        parent = self.chat_window[1][1][1]
        if not isinstance(parent, EditorNode):
            parent.children = (EditorNode(parent[0]), *parent.children[1:])
        return parent[0]

    @property
    def editor(self):
        return self.edit_box[1][0]

    @property
    def subscription_window(self):
        return self.column3[0][0]  # 订阅号窗口

    @property
    def subscription_window_title(self):
        return self.subscription_window[0][0]  # 订阅号窗口标题

    def __3_窗口切换功能(self):
        pass

    def get_chat_with(self):
        """ 当前在跟谁聊天 """
        try:
            return self.editor.Name
        except IndexError:
            return

    def set_chat_with(self, name):
        """ 要跟谁聊天，确保聊天框切换过去 """
        self.nav_chat.Click()

        if self.get_chat_with() == name:
            return

        column2 = self[1]  # 订阅号、聊天都共通
        search_box = column2[0][0][1][0]

        search_box.Click()
        search_box.SendKeys(name)
        search_box.SendKeys('{Enter}')

    def __4_编辑器(self):
        pass

    def get_editor_content(self):
        """ 获得编辑器正在编辑的内容（涉及到换行数据的时候好像不太准确） """
        return self.editor.GetValuePattern().Value

    def clear_editor_content(self):
        """ 删除编辑区中的所有内容 """
        edit_box = self.edit_box
        edit_box.SendKeys('{Ctrl}a')
        edit_box.SendKeys('{Delete}')

    def write_text(self, text, clear=False, send=False):
        # todo 加at人的功能
        # 1 是否清空旧数据
        if clear:
            self.clear_editor_content()

        # 2 写入新数据
        def should_use_clipboard(text):
            # 简单的策略：如果文本过长或包含特殊字符，则使用剪贴板
            return len(text) > 30 or not text.isprintable() or '{' in text

        edit_box = self.edit_box
        edit_box.Click()
        if should_use_clipboard(text):
            auto.SetClipboardText(text)
            time.sleep(1)
            edit_box.SendKeys('{Ctrl}v')
        else:
            edit_box.SendKeys(text)

        # 3 发送内容
        if send:
            time.sleep(1)
            edit_box.SendKeys('{Enter}')

    def send_text(self, text, clear=False):
        self.write_text(text, clear=clear, send=True)

    def send_files(self, file_paths):
        """
        发送多个文件

        :param list[str] file_paths: 必选参数，为文件的路径
        """
        if copy_files_to_clipboard(file_paths=file_paths):
            edit_box = self.edit_box
            edit_box.Click()
            time.sleep(1)
            edit_box.SendKeys('{Ctrl}v')
            time.sleep(1)
            edit_box.SendKeys('{Enter}')
            time.sleep(1)  # 等待发送动作完成


def __4_wx():
    """ 别人的原版实现 """


class WxOperation:
    """
    微信群发消息的类，提供了与微信应用交互的方法集，用于发送消息，管理联系人列表等功能。

    Attributes:
    ----------
    wx_window: auto.WindowControl
        微信控制窗口
    input_edit: wx_window.EditControl
        聊天界面输入框编辑控制窗口

    Methods:
    -------
    goto_chat_box(name):
        跳转到 指定好友窗口
    __send_text(*msgs):
        发送文本。
    __send_file(*filepath):
        发送文件
    get_friend_list(tag, num):
        可指定tag，获取好友num页的好友数量
    send_msg(name, msgs, file_paths=None, add_remark_name=False, at_everyone=False,
            text_interval=0.05, file_interval=0.5) -> None:
        向指定的好友或群聊发送消息和文件。支持同时发送文本和文件。
    """

    def __init__(self):
        self.wx_window = None
        self.input_edit = None
        self.wx_window: auto.WindowControl
        self.input_edit: auto.EditControl
        auto.SetGlobalSearchTimeout(Interval.BASE_INTERVAL)
        self.visible_flag: bool = False

    def locate_wechat_window(self):
        if not self.visible_flag:
            wake_up_window(class_name=WeChatConfig.WINDOW_CLASSNAME, name=WeChatConfig.WINDOW_NAME)
            self.wx_window = auto.WindowControl(Name=WeChatConfig.WINDOW_NAME, ClassName=WeChatConfig.WINDOW_CLASSNAME)
            if not self.wx_window.Exists(Interval.MAX_SEARCH_SECOND,
                                         searchIntervalSeconds=Interval.MAX_SEARCH_INTERVAL):
                raise Exception('微信似乎并没有登录!')
            self.input_edit = self.wx_window.EditControl()
            self.visible_flag = bool(self.visible_flag)
        # 微信窗口置顶
        self.wx_window.SetTopmost(isTopmost=True)

    def match_nickname(self, name):
        """获取当前面板的好友昵称"""
        self.input_edit = self.wx_window.EditControl(Name=name)
        if self.input_edit.Exists(Interval.MAX_SEARCH_SECOND, searchIntervalSeconds=Interval.MAX_SEARCH_INTERVAL):
            return self.input_edit
        return False

    def goto_chat_box(self, name: str) -> bool:
        """
        跳转到指定 name好友的聊天窗口。

        Args:
            name(str): 必选参数，好友名称

        Returns:
            None
        """
        if ctrl := self.match_nickname(name):
            ctrl.SetFocus()
            return ctrl

        assert name, "无法跳转到名字为空的聊天窗口"
        self.wx_window.SendKeys(text='{Ctrl}F', waitTime=Interval.BASE_INTERVAL)
        self.wx_window.SendKeys(text='{Ctrl}A', waitTime=Interval.BASE_INTERVAL)
        self.wx_window.SendKey(key=auto.SpecialKeyNames['DELETE'])
        auto.SetClipboardText(text=name)
        time.sleep(Interval.BASE_INTERVAL)
        self.wx_window.SendKeys(text='{Ctrl}V', waitTime=Interval.BASE_INTERVAL)
        # 若有匹配结果，第一个元素的类型为PaneControl
        search_nodes = self.wx_window.ListControl(foundIndex=2).GetChildren()
        if not isinstance(search_nodes.pop(0), auto.PaneControl):
            self.wx_window.SendKeys(text='{Esc}', waitTime=Interval.BASE_INTERVAL)
            raise ValueError("昵称不匹配")
        # 只考虑全匹配, 不考虑好友昵称重名, 不考虑好友昵称与群聊重名
        if search_nodes[0].Name == name:
            self.wx_window.SendKey(key=auto.SpecialKeyNames['ENTER'], waitTime=Interval.BASE_INTERVAL)
            time.sleep(Interval.BASE_INTERVAL)
            return True
        # 无匹配用户, 取消搜索框
        self.wx_window.SendKeys(text='{Esc}', waitTime=Interval.BASE_INTERVAL)
        return False

    def at_at_everyone(self, group_chat_name: str):
        """
        @全部人的操作
        Args:
            group_chat_name(str): 群聊名称

        """
        # 个人 定位 聊天框，取 foundIndex=2，因为左侧聊天List 也可以匹配到foundIndex=1
        # 群聊 定位 聊天框 需要带上群人数，故会匹配失败，所以匹配失败的就是群聊
        result = self.wx_window.TextControl(Name=group_chat_name, foundIndex=2)
        # 只要匹配不上，说明这是个群聊窗口
        if not result.Exists(Interval.MAX_SEARCH_SECOND, searchIntervalSeconds=Interval.MAX_SEARCH_INTERVAL):
            # 寻找是否有 @所有人 的选项
            self.input_edit.SendKeys(text='{Shift}2', waitTime=Interval.BASE_INTERVAL)
            everyone = self.wx_window.ListItemControl(Name='所有人')
            if not everyone.Exists(Interval.MAX_SEARCH_SECOND, searchIntervalSeconds=Interval.MAX_SEARCH_INTERVAL):
                self.input_edit.SendKeys(text='{Ctrl}A', waitTime=Interval.BASE_INTERVAL)
                self.input_edit.SendKeys(text='{Delete}', waitTime=Interval.BASE_INTERVAL)
                return
            self.input_edit.SendKeys(text='{Up}', waitTime=Interval.BASE_INTERVAL)
            self.input_edit.SendKeys(text='{Enter}', waitTime=Interval.BASE_INTERVAL)
            self.input_edit.SendKeys(text='{Enter}', waitTime=Interval.BASE_INTERVAL)

    def __send_text(self, *msgs, wait_time, send_shortcut) -> None:
        """
        发送文本.

        Args:
            input_name(str): 必选参数, 为输入框
            *msgs(str): 必选参数，为发送的文本
            wait_time(float): 必选参数，为动态等待时间
            send_shortcut(str): 必选参数，为发送快捷键

        Returns:
            None
        """

        def should_use_clipboard(text: str):
            # 简单的策略：如果文本过长或包含特殊字符，则使用剪贴板
            return len(text) > 30 or not text.isprintable()

        for msg in msgs:
            assert msg, "发送的文本内容为空"
            self.input_edit.SendKeys(text='{Ctrl}a', waitTime=wait_time)
            self.input_edit.SendKey(key=auto.SpecialKeyNames['DELETE'], waitTime=wait_time)
            self.input_edit.SendKeys(text='{Ctrl}a', waitTime=wait_time)
            self.input_edit.SendKey(key=auto.SpecialKeyNames['DELETE'], waitTime=wait_time)

            if should_use_clipboard(msg):
                auto.SetClipboardText(text=msg)
                time.sleep(wait_time * 2.5)
                self.input_edit.SendKeys(text='{Ctrl}v', waitTime=wait_time * 2)
            else:
                self.input_edit.SendKeys(text=msg, waitTime=wait_time * 2)

            # 设置到剪切板再黏贴到输入框
            self.wx_window.SendKeys(text=f'{send_shortcut}', waitTime=wait_time * 2)

    def __send_file(self, *file_paths, wait_time, send_shortcut) -> None:
        """
        发送文件.

        Args:
            *file_paths(str): 必选参数，为文件的路径
            wait_time(float): 必选参数，为动态等待时间
            send_shortcut(str): 必选参数，为发送快捷键

        Returns:
            None
        """
        # 复制文件到剪切板
        if copy_files_to_clipboard(file_paths=file_paths):
            # 粘贴到输入框
            self.input_edit.SendKeys(text='{Ctrl}V', waitTime=wait_time)
            # 按下回车键
            self.wx_window.SendKeys(text=f'{send_shortcut}', waitTime=wait_time / 2)

            time.sleep(wait_time)  # 等待发送动作完成

    def get_friend_list(self, tag: str = None) -> list:
        """
        获取微信好友名称.

        Args:
            tag(str): 可选参数，如不指定，则获取所有好友

        Returns:
            list
        """
        # 定位到微信窗口
        self.locate_wechat_window()
        # 取消微信窗口置顶
        self.wx_window.SetTopmost(isTopmost=False)
        # 点击 通讯录管理
        self.wx_window.ButtonControl(Name="通讯录").Click(simulateMove=False)
        self.wx_window.ListControl(Name="联系人").ButtonControl(Name="通讯录管理").Click(simulateMove=False)
        # 切换到通讯录管理，相当于切换到弹出来的页面
        contacts_window = auto.GetForegroundControl()
        contacts_window.ButtonControl(Name='最大化').Click(simulateMove=False)

        if tag:
            try:
                contacts_window.ButtonControl(Name="标签").Click(simulateMove=False)
                contacts_window.PaneControl(Name=tag).Click(simulateMove=False)
                time.sleep(Interval.BASE_INTERVAL * 2)
            except LookupError:
                contacts_window.SendKey(auto.SpecialKeyNames['ESC'])
                raise LookupError(f'找不到 {tag} 标签')

        name_list = list()
        last_names = None
        while True:
            # TODO 修改成使用 foundIndex 的方式
            try:
                nodes = contacts_window.ListControl(foundIndex=2).GetChildren()
            except LookupError:
                nodes = contacts_window.ListControl().GetChildren()
            cur_names = [node.TextControl().Name for node in nodes]

            # 如果滚动前后名单未变，认为到达底部
            if cur_names == last_names:
                break
            last_names = cur_names
            # 处理当前页的名单
            for node in nodes:
                # TODO 如果有需要, 可以处理成导出为两列的csv格式
                nick_name = node.TextControl().Name  # 用户名
                remark_name = node.ButtonControl(foundIndex=2).Name  # 用户备注名，索引1会错位，索引2是备注名，索引3是标签名
                name_list.append(remark_name if remark_name else nick_name)
            # 向下滚动页面
            contacts_window.WheelDown(wheelTimes=8, waitTime=Interval.BASE_INTERVAL / 2)
        # 结束时候关闭 "通讯录管理" 窗口
        contacts_window.SendKey(auto.SpecialKeyNames['ESC'])
        # 简单去重，但是存在误判（如果存在同名的好友), 保持获取时候的顺序
        return list(dict.fromkeys(name_list))

    def get_group_chat_list(self) -> list:
        """获取群聊通讯录中的用户名称"""
        name_list = list()
        auto.ButtonControl(Name='聊天信息').Click()
        time.sleep(0.5)
        chat_members_win = self.wx_window.ListControl(Name='聊天成员')
        if not chat_members_win.Exists():
            return list()
        self.wx_window.ButtonControl(Name='查看更多').Click()
        for item in chat_members_win.GetChildren():
            name_list.append(item.ButtonControl().Name)
        return name_list

    def send_msg(self, name, msgs=None, file_paths=None, add_remark_name=False, at_everyone=False,
                 text_interval=Interval.SEND_TEXT_INTERVAL, file_interval=Interval.SEND_FILE_INTERVAL,
                 send_shortcut='{Enter}') -> None:
        """
        发送消息，可同时发送文本和文件（至少选一项

        Args:
            name(str):必选参数，接收消息的好友名称, 可以单发
            msgs(Iterable[str], Optional): 可选参数，发送的文本消息
            file_paths(Iterable[str], Optional):可选参数，发送的文件路径
            add_remark_name(bool): 可选参数，是否添加备注名称发送
            at_everyone(bool): 可选参数，是否@全部人
            text_interval(float): 可选参数，默认为0.05
            file_interval(float): 可选参数，默认为0.5
            send_shortcut(str): 可选参数，默认为 Enter

        Raises:
            ValueError: 如果用户名为空或发送的消息和文件同时为空时抛出异常
            TypeError: 如果发送的文本消息或文件路径类型不是列表或元组时抛出异常
        """
        # 定位到微信窗口
        self.locate_wechat_window()

        if not name:
            raise ValueError("用户名不能为空")

        if not any([msgs, file_paths]):
            raise ValueError("发送的消息和文件不可同时为空")

        if msgs and not isinstance(msgs, Iterable):
            raise TypeError("发送的文本消息必须是可迭代的")

        if file_paths and not isinstance(file_paths, Iterable):
            raise TypeError("发送的文件路径必须是可迭代的")

        # 如果当前面板已经是需发送好友, 则无需再次搜索跳转
        if not self.match_nickname(name=name):
            if not self.goto_chat_box(name=name):
                raise NameError('昵称不匹配')

        # 设置输入框为当前焦点
        self.input_edit = self.wx_window.EditControl(Name=name)
        self.input_edit.SetFocus()

        # @所有人
        if at_everyone:
            auto.SetGlobalSearchTimeout(Interval.BASE_INTERVAL)
            self.at_at_everyone(group_chat_name=name)
            auto.SetGlobalSearchTimeout(Interval.BASE_INTERVAL * 25)

        # TODO 添加备注可以多做一个选项，添加到每条消息的前面，如xxx，早上好
        if msgs and add_remark_name:
            new_msgs = deepcopy(list(msgs))
            new_msgs.insert(0, name)
            self.__send_text(*new_msgs, wait_time=text_interval, send_shortcut=send_shortcut)
        elif msgs:
            self.__send_text(*msgs, wait_time=text_interval, send_shortcut=send_shortcut)
        if file_paths:
            self.__send_file(*file_paths, wait_time=file_interval, send_shortcut=send_shortcut)

        # 取消微信窗口置顶
        self.wx_window.SetTopmost(isTopmost=False)


if __name__ == '__main__':
    # wx = WxOperation()
    # data = wx.get_friend_list('无标签')
    # print(data)
    # print(len(data))
    pass
