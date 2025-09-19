#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/05/26 17:24

import os.path as osp
import sys
import time
from datetime import datetime, timedelta

from pyxllib.prog.lazyimport import lazy_import

try:
    from PyQt5 import QtWidgets
    from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QEventLoop
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QFrame, QInputDialog, QMessageBox,
        QVBoxLayout, QTextEdit, QSizePolicy, QLabel, QProgressBar, QDialog
    )
    from PyQt5.QtGui import QTextOption
except ModuleNotFoundError:
    QtWidgets = lazy_import('from PyQt5 import QtWidgets', 'PyQt5')
    QTimer = lazy_import('from PyQt5.QtCore import QTimer', 'PyQt5')
    Qt = lazy_import('from PyQt5.QtCore import Qt', 'PyQt5')
    QThread = lazy_import('from PyQt5.QtCore import QThread', 'PyQt5')
    pyqtSignal = lazy_import('from PyQt5.QtCore import pyqtSignal', 'PyQt5')
    QEventLoop = lazy_import('from PyQt5.QtCore import QEventLoop', 'PyQt5')

    QApplication = lazy_import('from PyQt5.QtWidgets import QApplication', 'PyQt5')
    QMainWindow = lazy_import('from PyQt5.QtWidgets import QMainWindow', 'PyQt5')
    QFrame = lazy_import('from PyQt5.QtWidgets import QFrame', 'PyQt5')
    QInputDialog = lazy_import('from PyQt5.QtWidgets import QInputDialog', 'PyQt5')
    QMessageBox = lazy_import('from PyQt5.QtWidgets import QMessageBox', 'PyQt5')
    QVBoxLayout = lazy_import('from PyQt5.QtWidgets import QVBoxLayout', 'PyQt5')
    QTextEdit = lazy_import('from PyQt5.QtWidgets import QTextEdit', 'PyQt5')
    QSizePolicy = lazy_import('from PyQt5.QtWidgets import QSizePolicy', 'PyQt5')
    QLabel = lazy_import('from PyQt5.QtWidgets import QLabel', 'PyQt5')
    QProgressBar = lazy_import('from PyQt5.QtWidgets import QProgressBar', 'PyQt5')
    QDialog = lazy_import('from PyQt5.QtWidgets import QDialog', 'PyQt5')

    QTextOption = lazy_import('from PyQt5.QtGui import QTextOption', 'PyQt5')

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


def get_input_widget(items=None, cur_value=None, *, parent=None, valcvt=None,
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
    :param valcvt: 数值类型转换函数，非法时抛出ValueError
        很多输入框是传入文本，有时需要转为int、float、list等类型
        支持输入常见类型转换的字符串名称，比如int、float
    :param correct_changed: 文本改变时的回调函数，一般用于数值有效性检查
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
        w = XlComboBox(parent, text=cur_value, items=items, valcvt=cvtfunc, editable=isinstance(items, list))
    elif items is None:
        # 普通填充框
        w = XlLineEdit(cur_value, parent=parent, valcvt=cvtfunc)
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


class XlThreadWorker(QThread):
    result = pyqtSignal(object)  # 运行结果信号
    error = pyqtSignal(Exception)  # 错误信号
    progress = pyqtSignal(int)  # 进度信号

    def __init__(self, func, *args, use_progress=False, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        if use_progress:
            self.kwargs['progress_callback'] = lambda v: self.progress.emit(v)

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.result.emit(result)  # Emit the result when done
        except Exception as e:
            self.error.emit(e)


class WaitDialog(QDialog):

    def __init__(self, parent=None, text='', title='正在执行任务...', delay_seconds=5):
        super().__init__(parent)
        self.base_text = text
        self.setWindowTitle(title)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_text)
        self.start_time = None
        self.result = None
        self.worker = None
        self.error = None

        self.delay_milliseconds = delay_seconds * 1000  # 延迟弹窗
        self.is_running = False

        self.layout = QVBoxLayout()  # 布局
        self.label = QLabel(self.base_text)  # 标签
        self.layout.addWidget(self.label)
        self.pbar = QProgressBar()  # 进度条
        self.layout.addWidget(self.pbar)
        self.setLayout(self.layout)

    def handle_result(self, result):
        self.result = result
        self.is_running = False

    def handle_error(self, error):
        self.error = error
        self.label.setText(f"{self.base_text}\n运行出现错误: {error}")
        self.is_running = False

    def handle_progress(self, progress):
        self.pbar.setValue(progress)

    def update_text(self):
        elapsed_time = int((datetime.now() - self.start_time).total_seconds()) + self.delay_milliseconds // 1000
        self.label.setText(f"{self.base_text}\n已运行 {elapsed_time} 秒")

    def run(self, func, *args, **kwargs):
        """
        def func():
            ...  # 程序功能

        msg = WaitDialog().run(func)  # 开一个等待窗口等程序运行
        """
        self.worker = XlThreadWorker(func, *args, **kwargs)
        self.worker.result.connect(self.handle_result)
        self.worker.error.connect(self.handle_error)
        self.is_running = True
        self.worker.start()

        QTimer.singleShot(self.delay_milliseconds, self.check_and_show)

        # 阻塞主线程，直到子线程完成
        while self.is_running:
            QApplication.processEvents()  # 刷新UI，保持其响应性
            time.sleep(0.1)  # 等待一段时间，以减少CPU使用率

        self.timer.stop()
        self.accept()

        return self.result

    def run_with_progress(self, func, *args, **kwargs):
        """
        def func(progress_callback):
            progress_callback(50)  # 可以在运行中设置进度，进度值为0~100
            ...  # 其他功能

        msg = WaitDialog().run_with_progress(func)  # 运行完获得返回值
        """
        self.worker = XlThreadWorker(func, *args, use_progress=True, **kwargs)
        self.worker.result.connect(self.handle_result)
        self.worker.error.connect(self.handle_error)
        self.worker.progress.connect(self.handle_progress)
        self.is_running = True
        self.worker.start()

        QTimer.singleShot(self.delay_milliseconds, self.check_and_show)

        # 阻塞主线程，直到子线程完成
        while self.is_running:
            QApplication.processEvents()  # 刷新UI，保持其响应性
            time.sleep(0.1)  # 等待一段时间，以减少CPU使用率

        self.timer.stop()
        self.accept()

        return self.result

    def start_timer(self):
        self.start_time = datetime.now()
        self.timer.start(1000)

    def check_and_show(self):
        if self.is_running:
            self.show()
            self.start_timer()

    def __enter__(self):
        """ with写法比较简洁，但不太推荐这种使用方法，这样并不工程化
        这样会把要运行的功能变成主线程，这个提示窗口会被挂起

        这里功能设计上也比较简单些，不考虑写的很完善强大了。
        """
        self.show()
        QApplication.processEvents()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.accept()


class CustomMessageBox(QMessageBox):
    def __init__(self, icon, title, text, copyable):
        super().__init__(icon, title, "")
        self.init_ui(title, text, copyable)

    def init_ui(self, title, text, copyable=False):
        layout = QVBoxLayout()

        if copyable:
            widget = QTextEdit()
            widget.setText(text)
            widget.setReadOnly(True)
            widget.setWordWrapMode(QTextOption.WrapAnywhere)
            widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            widget.document().documentLayout().documentSizeChanged.connect(
                lambda: widget.setMinimumHeight(min(widget.document().size().height(), 700))
            )
            widget.setMinimumHeight(100)
            widget.setMaximumHeight(700)
        else:
            widget = QLabel()
            widget.setText(text)
            widget.setWordWrap(True)

        min_width = max(len(title) * 15, 600)
        widget.setMinimumWidth(min_width)

        layout.addWidget(widget)
        self.layout().addLayout(layout, 1, 1)


def show_message_box(text, title=None, icon=None, detail=None,
                     buttons=QMessageBox.Ok | QMessageBox.Cancel,
                     default_button=QMessageBox.Ok, copyable=False):
    """ 显示一个提示框

    :param text: 提示框的文本内容
    :param title: 提示框的标题，默认值为 "提示"
    :param icon: 提示框的图标，默认值为 QMessageBox.NoIcon
        注意Information、Warning、Critical等都会附带一个提示音
        而Question是不带提示音的，默认的NoIcon也是不带提示音的
    :param detail: 提示框的详细信息，默认值为 None
    :param buttons: 提示框的按钮，默认值为 QMessageBox.Ok
    :param copyable: 消息窗中的文本是否可复制

    :return: 选择的按钮

    实现上，本来应该依据setMinimumWidth可以搞定的事，但不知道为什么就是会有bug问题，总之最后问gpt靠实现一个类来解决了

    """
    if title is None:
        title = "提示"

    if icon is None:
        icon = QMessageBox.NoIcon

    msg_box = CustomMessageBox(icon, title, text, copyable)

    if detail is not None:
        msg_box.setDetailedText(detail)
    msg_box.setStandardButtons(buttons)
    msg_box.setDefaultButton(default_button)

    return msg_box.exec_()
