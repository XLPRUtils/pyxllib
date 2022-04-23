#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/05/26 17:24

from pyxllib.prog.pupil import check_install_package

check_install_package('qtpy', 'QtPy')

import json
import os.path as osp
import sys
import time

from PyQt5.QtCore import pyqtSignal
from qtpy import QtWidgets, QtGui
from qtpy.QtWidgets import QFrame, QInputDialog, QApplication, QMainWindow, QMessageBox

from pyxllib.prog.newbie import CvtType

here = osp.dirname(osp.abspath(__file__))


class QHLine(QFrame):
    """ https://stackoverflow.com/questions/5671354/how-to-programmatically-make-a-horizontal-line-in-qt """

    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class QVLine(QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)


class XlLineEdit(QtWidgets.QLineEdit):
    correctChanged = pyqtSignal(object)  # 符合数值类型的修改
    wrongChanged = pyqtSignal(str)  # 不符合数值类型的修改

    def __init__(self, text=None, parent=None, *, valcvt=None):
        """
        :param valcvt: 数值类型转换器
        """
        super().__init__(str(text), parent)

        def check():
            # TODO 目前是强制重置样式，可以考虑怎么保留原样式基础上修改属性值
            s = self.text()
            try:
                if valcvt:
                    s = valcvt(s)
                self.setStyleSheet('')
                self.setToolTip('')
                self.correctChanged.emit(s)
            except ValueError:
                self.setStyleSheet('background-color: lightpink;')
                self.setToolTip(f'输入数据不是{valcvt}类型')
                self.wrongChanged.emit(s)

        self.textChanged.connect(check)
        if text:
            check()

        # self.setStyleSheet(self.styleSheet() + 'qproperty-cursorPosition: 0;')
        self.setStyleSheet(self.styleSheet())


class XlComboBox(QtWidgets.QComboBox):
    # 这个控件一般没有类型检查，但在支持填入自定义值时，是存在类型错误问题的
    #   但在工程上还是依然写了 correctChanged，方便下游任务统一接口
    correctChanged = pyqtSignal(object)  # 符合数值类型的修改
    wrongChanged = pyqtSignal(str)  # 不符合数值类型的修改

    def __init__(self, parent=None, *, text=None, items=None, valcvt=None, editable=False):
        """
        """
        # 1 基础设置
        super().__init__(parent)
        self.reset_items(items)
        self.editable = editable
        if self.editable:
            self.setEditable(True)

        # 2 检查功能
        def check(s):
            try:
                if valcvt:
                    s = valcvt(s)

                if self.editable:  # 支持自定义值
                    self.setStyleSheet('')
                    self.setToolTip('')
                    self.correctChanged.emit(s)
                elif s not in self.items_set:  # 不支持自定义值，但出现自定义值
                    self.setStyleSheet('background-color: yellow;')
                    self.setToolTip(f'不在清单里的非法值')
                    self.wrongChanged.emit(s)
                else:  # 不支持自定义值，且目前值在清单中
                    self.setEditable(False)
                    self.setStyleSheet('')
                    self.setToolTip('')
                    self.correctChanged.emit(s)
            except ValueError:
                self.setStyleSheet('background-color: lightpink;')
                self.setToolTip(f'输入数据不是{valcvt}类型')
                self.wrongChanged.emit(s)

        self.currentTextChanged.connect(check)
        # self.wrongChanged.connect(lambda s: print('非法值：', s))  # 可以监控非法值

        # 3 是否有预设值
        if text:
            self.setText(text)

        # 4 补充格式
        # self.setStyleSheet(self.styleSheet() + 'qproperty-cursorPosition: 0;')
        self.setStyleSheet(self.styleSheet())

    def setText(self, text):
        text = str(text)
        if text not in self.items and not self.editable:
            # 虽然不支持editable，但是出现了意外值，需要强制升级为可编辑
            self.setEditable(True)
        self.setCurrentText(text)

    def reset_items(self, items):
        # 1 存储配置清单
        self.clear()
        self.raw_items = items  # noqa
        self.items = [str(x) for x in items if x is not None]  # noqa
        self.items_set = set(self.items)  # noqa 便于判断是否存在的集合类型
        self.addItems(self.items)

        # 2 画出 元素值、分隔符
        cnt = 0
        for i in range(len(items)):
            if items[i] is None:
                self.insertSeparator(i - cnt)  # 不过这个分割符没那么显眼
                cnt += 1


def get_input_widget(items=None, cur_value=None, *, valcvt=None,
                     n_widget=1, enabled=True,
                     correct_changed=None):
    """ 根据items参数情况，智能判断生成对应的widget

    :param items:
        None, 普通的文本编辑框
        普通数组，下拉框  （可以用None元素表示分隔符）
            list，除了列表中枚举值，也支持自定义输入其他值
            tuple，只能用列表中的枚举值
        多级嵌套数组，多级下拉框  （未实装）  （一般都是不可改的，并且同类型的数据）
            [('福建', [('龙岩', ['连城', '长汀', ...], ...)]), ('北京', ...)]
            这种情况会返回多个widget
    :param cur_value: 当前显示的文本值
    :param valcvt: 数值类型转换函数，非法时返回ValueError  （未实装）
        很多输入框是传入文本，有时需要转为int、float、list等类型
        支持输入常见类型转换的字符串名称，比如int、float
    :param correct_changed: 文本改变时的回调函数
    :param n_widget: 配合items为嵌套数组使用，需要指定嵌套层数
        此时cur_value、cvt、enabled、text_changed等系列值可以传入n_widget长度的list
    :param enabled: 是否可编辑
    """
    if n_widget > 1:
        raise NotImplementedError

    # 1 封装类型检查功能
    if isinstance(valcvt, str):
        cvtfunc = CvtType.factory(valcvt)
    else:
        cvtfunc = valcvt

    # 2 正式生成控件
    if isinstance(items, (list, tuple)):
        # 带有 items 的字段支持候选下拉菜单
        w = XlComboBox(text=cur_value, items=items, valcvt=cvtfunc, editable=isinstance(items, list))
    elif items is None:
        # 普通填充框
        w = XlLineEdit(cur_value, valcvt=cvtfunc)
    else:
        raise ValueError(f'{type(items)}')

    # 3 通用配置
    if callable(correct_changed):
        w.correctChanged.connect(correct_changed)
    if not enabled:
        w.setEnabled(enabled)

    return w


def __other():
    pass


def main_qapp(window):
    """ 执行Qt应用 """
    app = QApplication(sys.argv)
    window.show()  # 展示窗口
    sys.exit(app.exec_())


def qt_clipboard_monitor(func=None, verbose=1, *, cooldown=0.5):
    """ qt实现的剪切板监控器

    :param cooldown: cd，冷切时间，防止短时间内因为重复操作响应剪切板，重复执行功能

    感觉这个组件还有很多可以扩展的，比如设置可以退出的快捷键
    """
    import pyperclip

    last_response = time.time()

    if func is None:
        func = lambda s: s

    def on_clipboard_change():
        # 1 数据内容一样则跳过不处理，表示很可能是该函数调用pyperclip.copy(s)产生的重复响应
        nonlocal last_response
        s0 = pyperclip.paste()
        s0 = s0.replace('\r\n', '\n')

        cur_time = time.time()

        if cur_time - last_response < cooldown:
            return
        last_response = cur_time

        # 2 处理函数
        s1 = func(s0)
        if s1 != s0:
            if verbose:
                print('【处理前】', time.strftime('%H:%M:%S'))
                print(s0)
                print('【处理后】')
                print(s1)
                print()
            pyperclip.copy(s1)

    app = QApplication([])
    clipboard = app.clipboard()
    clipboard.dataChanged.connect(on_clipboard_change)
    app.exec_()


class XlMainWindow(QMainWindow):
    """ 根据自己开发app经验，封装一些常用的功能接口，简化代码量 """
    pass


class WaitMessageBox(QMessageBox):
    """ 在执行需要一定时长的程序时，弹出提示窗，并在执行完功能后自动退出提示窗

    【示例】
    with WaitMessageBox(self.mainwin, 'PaddleOCR模型初始化中，请稍等一会...'):
        from pyxlpr.paddleocr import PaddleOCR
        self.ppocr = PaddleOCR.build_ppocr()
    """
    finished = pyqtSignal()

    def __init__(self, parent=None, text=None):
        super().__init__(parent)
        if text:
            self.setText(text)
        self.setStyleSheet("QLabel{min-width: 350px;}")
        self.setWindowTitle('WaitMessageBox（任务完成后会自动退出该窗口）')
        self.setStandardButtons(QMessageBox.NoButton)
        self.finished.connect(self.accept)

    def __enter__(self):
        self.show()
        QApplication.processEvents()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finished.emit()
