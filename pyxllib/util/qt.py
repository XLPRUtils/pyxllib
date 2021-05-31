#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/05/26 17:24

from pyxllib.basic import *
import os.path as osp

try:
    from qtpy import QtWidgets
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'QtPy'])
    from qtpy import QtWidgets

from qtpy import QtGui
from qtpy.QtWidgets import QFrame, QInputDialog, QApplication
from PyQt5.QtCore import pyqtSignal

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

        self.setStyleSheet(self.styleSheet() + 'qproperty-cursorPosition: 0;')


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
        self.setStyleSheet(self.styleSheet() + 'qproperty-cursorPosition: 0;')

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
                     text_changed=None):
    """ 根据items参数情况，智能判断生成对应的windet

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
    :param text_changed: 文本改变时的回调函数
    :param n_widget: 配合items为嵌套数组使用，需要指定嵌套层数
        此时cur_value、cvt、enabled、text_changed等系列值可以传入n_widget长度的list
    :param enabled: 是否可编辑
    """
    if n_widget > 1:
        raise NotImplementedError

    # 1 封装类型检查功能
    if isinstance(valcvt, str):
        cvtfunc = {'int': int, 'float': float, 'str': str}.get(valcvt, None)
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
    if callable(text_changed):
        w.correctChanged.connect(text_changed)
    if not enabled:
        w.setEnabled(enabled)

    return w


def newIcon(icon):
    icons_dir = osp.join(here, "../icons")
    return QtGui.QIcon(osp.join(":/", icons_dir, "%s.png" % icon))


def newAction(
        parent,
        text,
        slot=None,
        shortcut=None,
        icon=None,
        tip=None,
        checkable=False,
        enabled=True,
        checked=False,
):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QtWidgets.QAction(text, parent)
    if icon is not None:
        a.setIconText(text.replace(" ", "\n"))
        a.setIcon(newIcon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    a.setChecked(checked)
    return a


class XlActionFunc:
    """ 跟action有关的函数调用封装 （该类也可以作为基础的check版本使用）

    一般逻辑结构是这样：
        有个可运行功能的函数func，运行时会配置一些需要存储起来的变量值value
        并且功能需要关联action，绑定到menu等菜单中时，可以使用该装饰器
    """

    def __init__(self, parent, title, value=None, checked=None, **kwargs):
        self.parent = parent
        self.title = title
        self.value = value
        self.checked = bool(checked)
        self.checkable = checked is not None

        self.action = newAction(self.parent, self.parent.tr(self.title), self.__call__,
                                checkable=self.checkable, checked=self.checked, **kwargs)

    def __call__(self, checked):
        self.checked = checked


class GetMultiLineTextAction(XlActionFunc):
    """ 该类value是直接存储原始的完整文本内容 """

    def __call__(self, checked):
        super().__call__(checked)
        self.value = self.value or ''
        inputs = QInputDialog.getMultiLineText(self.parent, self.title, '编辑文本：', self.value)
        if inputs[1]:  # “确定操作” 才更新属性
            self.value = inputs[0]


class GetItemsAction(XlActionFunc):
    """ 该类value目前是存储为list类型 """

    def __call__(self, checked):
        super().__call__(checked)
        self.value = self.value or []
        inputs = QInputDialog.getMultiLineText(self.parent, self.title, '编辑多行文本：',
                                               '\n'.join(self.value))
        if inputs[1]:  # “确定操作” 才更新属性
            self.value = inputs[0].splitlines()


class GetJsonAction(XlActionFunc):
    """ 该类value是直接存储原始的完整文本内容 """

    def __call__(self, checked):
        super().__call__(checked)
        self.value = self.value or ''
        inputs = QInputDialog.getMultiLineText(self.parent, self.title, '编辑json数据：',
                                               json.dumps(self.value, indent=2))
        if inputs[1]:  # “确定操作” 才更新属性
            self.value = json.loads(inputs[0])


def main_qapp(window):
    """ 执行Qt应用 """
    app = QApplication(sys.argv)
    window.show()  # 展示窗口
    sys.exit(app.exec_())
