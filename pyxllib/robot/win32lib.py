#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/09/07 10:21

import win32com.client as win32
from win32com.client import constants
import pythoncom

from pyxllib.debug.specialist import TicToc
from pyxllib.prog.newbie import RunOnlyOnce
from pyxllib.prog.pupil import EnchantBase


def get_win32_app(name, visible=False):
    """ 启动可支持pywin32自动化处理的应用

    Args:
        str name: 应用名称，不区分大小写，比如word, excel, powerpoint, onenote
            不带'.'的情况下，会自动添加'.Application'的后缀
        visible: 应用是否可见

    Returns: app

    """
    # 1 name
    name = name.lower()
    if '.' not in name:
        name += '.application'

    # 2 app
    # 这里可能还有些问题，不同的应用，机制不太一样，后面再细化完善吧
    try:
        app = win32.GetActiveObject(f'{name}')  # 不能关联到普通方式打开的应用。但代码打开的应用都能找得到。
    except pythoncom.com_error:
        app = win32.gencache.EnsureDispatch(f'{name}')

    if visible is not None:
        app.Visible = visible

    return app


def __word():
    pass


class EnchantWin32WordApplication(EnchantBase):
    @classmethod
    @RunOnlyOnce
    def enchant(cls, app, recursion_enchant=False):
        """
        :param app: win32的类是临时生成的，需要给一个参考对象，才方便type(word)算出类型
        :param recursion_enchant: 是否递归，对目前有的各种子类扩展功能都绑定上
            默认关闭，如果影响到性能，可以关闭，后面运行中需要时手动设定enchant
            开启，能方便业务层开发
        """
        # app
        _cls = type(app)
        names = cls.check_enchant_names([_cls])
        exclude_names = {'get_app'}
        cls._enchant(_cls, names - exclude_names, mode='staticmethod2objectmethod')

        if recursion_enchant:
            with TicToc('win32com的WordApplication初始化', disable=True):
                # doc，win32的Word.Application比较特别，要提前创建对象
                doc = app.Documents.Add()
                EnchantWin32WordDocument.enchant(doc)
                doc.Close(False)

    @classmethod
    def get_app(cls, mode='default', *, visible=None, display_alerts=0, recursion_enchant=False):
        """
        Args:
            mode: 目前除了默认default，只有new，强制新建一个app
            visible: 是否可见
            display_alerts: 是否关闭警告
            recursion_enchant: 是否递归执行enchant

        """
        app = None
        if mode == 'default':
            try:
                app = win32.GetActiveObject('Word.Application')
            except pythoncom.com_error:
                app = None
        if app is None:
            app = win32.gencache.EnsureDispatch('Word.Application')
        cls.enchant(app, recursion_enchant)

        if visible is not None:
            app.Visible = visible
        if display_alerts is not None:
            app.DisplayAlerts = display_alerts  # 不警告

        return app

    @staticmethod
    def wd(_cls, name, part=None):
        """ 输入字符串名称，获得对应的常量值

        :param name: 必须省略前缀wd。这个函数叫wd，就是帮忙省略掉前缀wd的意思
        :param part: 特定组别的枚举值，可以输入特殊的name模式来取值
        """
        if part is None:
            return getattr(constants, 'wd' + name)
        elif part == 'SaveFormat':
            """ word保存类型的枚举

            >> _('.html')
            8
            >> _('Pdf')
            17
            >> _('wdFormatFilteredHTML')
            10
            """
            common = {'doc': 'FormatDocument97',
                      'html': 'FormatHTML',
                      'txt': 'FormatText',
                      'docx': 'FormatDocumentDefault',
                      'pdf': 'FormatPDF'}
            name = common.get(name.lower().lstrip('.'), name)
            return getattr(constants, 'wd' + name)


class EnchantWin32WordDocument(EnchantBase):
    @classmethod
    @RunOnlyOnce
    def enchant(cls, doc):
        _cls = type(doc)
        names = cls.check_enchant_names([_cls])
        propertys = {'content'}
        cls._enchant(_cls, propertys, mode='staticmethod2property')
        cls._enchant(_cls, names - propertys, mode='staticmethod2objectmethod')

    @staticmethod
    def content(_self):
        # 有特殊换行，ch.Text可能会得到 '\r\x07'，为了位置对应，只记录一个字符
        return ''.join([ch.Text[0] for ch in _self.Characters])
