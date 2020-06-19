#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tkinter相关工具
"""

__author__ = '陈坤泽'
__email__ = '877362867@qq.com'
__date__ = '2018/07/24 19:42'


import tkinter as tk

from pyxllib.debug.all import *


def askquestion(title='code4101py', message='void', **options):
    """对tk.messagebox.askquestion的封装，用于消息提示

    默认会弹出一个主窗口，这里关掉了主窗口
    """

    from tkinter import messagebox

    win = tk.Tk()
    win.withdraw()  # 不显示主窗口

    # 弹出信息提示窗口，根据选择会返回yes、no
    return tk.messagebox.askquestion(title, message, **options)


def askstring(title='code4101py', prompt='void', **kwargs):
    """类似askquestion
    """
    from tkinter import simpledialog

    # 不显示主窗口
    win = tk.Tk()
    win.withdraw()

    # 如果提示内容太短，要增加空格，使得标题的显示长度足够
    #   tk的askstring默认最短有11个汉字多的长度
    #   标题非文本内容已经占掉大概13个汉字的宽度
    #   根据以上规律可推导出下述宽度控制算法
    len1 = strwidth(title)*2 + 26
    len2 = strwidth(prompt)
    if len2 < len1:
        prompt += ' '*(len1-len2)
    # 显示窗口
    s = tk.simpledialog.askstring(title, prompt, **kwargs)
    return s


def tk_init(title='tk', geometry='300x200'):
    """主窗口初始化"""
    from tkinter import Tk
    tk = Tk()
    tk.title(title)
    tk.geometry(geometry)

    # 可以设置字体
    # tk.option_add('*font', ('verdana', 12, 'bold'))
    return tk


class AskStringByList:
    """使用举例：
    ob = AskStringByList(['选项1', '选项2'], title='AskStringByList')
    print(ob.value.get())
    """
    def __init__(self, ls=tuple(range(3)), **kwargs):
        """
        :param ls:
        :param kwargs:
            title： 设置主窗口名
            geometry： 设置主窗口大小，例如'300x200'
        """
        from tkinter import Frame
        from tkinter.constants import YES, BOTH

        self.ls = ls
        self.kwargs = kwargs

        self.tk = self.tk_init()
        self.frame = Frame()
        self.value = None
        self.entry = None
        self.listbox = None
        self.frame_init()

        self.frame.pack(fill=BOTH, expand=YES)
        self.tk.mainloop()

    def tk_init(self):
        # 1 设置基本属性
        from tkinter import Tk
        tk = Tk()

        # 设置标题
        if 'title' not in self.kwargs:
            self.kwargs['title'] = 'AskStringByList'
        tk.title(self.kwargs['title'])

        # 设置窗口大小
        if 'geometry' not in self.kwargs:
            self.kwargs['geometry'] = '300x400'
        tk.geometry(self.kwargs['geometry'])

        # 禁用窗口调整大小
        tk.resizable(False, False)

        # 2 绑定快捷键
        tk.bind("<Return>", self.enter)  # 回车键跟 enter() 绑定
        tk.bind("<Escape>", self.esc)  # Esc键跟 esc() 绑定

        return tk

    def enter(self, ev=None):
        """按下确认键后的功能"""
        self.tk.quit()
        # self.result = xxx

    def esc(self, ev=None):
        """按下esc键后的反应"""
        self.tk.quit()
        self.value.set('')

    def onselect(self, ev):
        d = self.listbox.curselection()
        d = d[0] if d else 0  # 如果没有选择，默认选第0个
        self.value.set(self.ls[d])

    def listbox_init(self):
        from tkinter import Listbox, Scrollbar
        from tkinter.constants import END

        # 1 创建listbox
        listbox = Listbox(self.frame, width=35)
        for t in self.ls:
            listbox.insert(END, t)
        listbox.place(relx=0.1, rely=0.3)
        listbox.bind('<<ListboxSelect>>', self.onselect)

        # 2 为列表添加滚动条
        s = Scrollbar(listbox)
        s.place(relx=0.94, relheight=1)
        s.config(command=listbox.yview)
        listbox.config(yscrollcommand=s.set)

        return listbox

    def frame_init(self):
        """主要是设置布局"""
        from tkinter import Button, Label, Entry, StringVar

        fm = self.frame
        Label(fm, text='请选择一个值或输入自定义新值：').place(relx=0.1, rely=0.1)

        self.value = StringVar()
        self.entry = Entry(fm, textvariable=self.value, width=35).place(relx=0.1, rely=0.2)
        self.listbox = self.listbox_init()

        # 默认第1个为初始值
        self.listbox.select_set(0)
        self.value.set(self.ls[0])

        Button(fm, text='确认（Enter）', command=self.enter, width=12).place(relx=0.15, rely=0.85)
        Button(fm, text='取消（Esc）', command=self.esc, width=12).place(relx=0.5, rely=0.85)


if __name__ == '__main__':
    ob = AskStringByList()
    print(ob.value.get())
