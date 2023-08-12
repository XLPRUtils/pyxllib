#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2023/08/02 14:05

import os
import re
import sys
import json
import time
from types import SimpleNamespace

from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QListWidget, QLineEdit, QVBoxLayout, \
    QSplitter, QTreeView, QPlainTextEdit, QPushButton, QLabel, QHBoxLayout, QSizePolicy, QWidget, QStatusBar, \
    QAbstractItemView, QHeaderView, QMessageBox
from PyQt5.QtWidgets import QItemDelegate, QTextEdit
from PyQt5.QtWidgets import QItemDelegate, QDialog, QVBoxLayout, QTextEdit, QPushButton

from PyQt5.QtGui import QTextOption, QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QModelIndex, QSettings, QFileInfo

from pyxllib.file.specialist import XlPath

# 一个专门存储大字符串的命名空间
LargeStrings = SimpleNamespace()

# 命名并存储QTreeView的样式（旧的格式配置，在新版中已经不起作用）
LargeStrings.treeViewStyles = """
QTreeView::item {  /* 设置网格线 */
    border: 1px solid black;
}

QTreeView::item:selected {
    background: black;
    color: white;
}

QTreeView::item:selected:active {
    background: black;
    color: white;
}

QTreeView::item:selected:!active {
    background: black;
    color: white;
}
""".strip()


class MyTreeView(QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)

    def edit(self, index, trigger, event):
        if trigger == QAbstractItemView.DoubleClicked:
            return False
        return super().edit(index, trigger, event)


class KeyStandardItem(QStandardItem):
    def data(self, role=None):
        if role == Qt.TextAlignmentRole:
            return Qt.AlignLeft | Qt.AlignVCenter
        return super().data(role)


class TextEditDelegate(QItemDelegate):
    def createEditor(self, parent, option, index):
        return QTextEdit(parent)

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.EditRole)
        editor.setPlainText(value)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(20)  # 限制最大高度为20像素
        return size

    def setModelData(self, editor, model, index):
        value = editor.toPlainText()
        model.setData(index, value, Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def paint(self, painter, option, index):
        text = index.model().data(index)

        # 只显示前100个字符
        elided_text = text[:100] + '...' if len(text) > 100 else text

        painter.drawText(option.rect, Qt.AlignLeft, elided_text)


class JLineViewer(QMainWindow):
    def __init__(self):
        super(JLineViewer, self).__init__()
        self.load_settings()

        # 初始化 allItemsLoaded 变量
        self.allItemsLoaded = False

        self.initUI()
        # 开启部件接受拖放的能力（在windows中测试该功能失败）
        self.setAcceptDrops(True)

    def load_settings(self):
        self.settings = QSettings('pyxllib', 'JLineViewer')
        self.lastOpenDir = self.settings.value('lastOpenDir', '')

    def save_settings(self):
        self.settings.setValue('lastOpenDir', self.lastOpenDir)

    def initUI(self):
        self.listWidget = QListWidget()
        self.treeView = MyTreeView()
        self.plainTextEdit = QPlainTextEdit()
        self.searchLineEdit = QLineEdit()
        self.searchButton = QPushButton("普通搜索")

        self.listWidget.itemClicked.connect(self.loadJson)
        self.searchButton.clicked.connect(self.searchItems)
        self.searchLineEdit.returnPressed.connect(self.searchItems)
        self.treeView.clicked.connect(self.editItem)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.addPane(self.listWidget, 'JSONL Items'))
        splitter.addWidget(self.addPane(self.treeView, 'JSON Tree View'))
        splitter.addWidget(self.addPane(self.plainTextEdit, 'Selected Content'))
        splitter.setSizes([100, 300, 200])

        layout = QVBoxLayout()
        searchLayout = QHBoxLayout()
        searchLayout.addWidget(QLabel("搜索条目："))
        searchLayout.addWidget(self.searchLineEdit)
        searchLayout.addWidget(self.searchButton)
        layout.addLayout(searchLayout)
        layout.addWidget(splitter)

        self.regexSearchButton = QPushButton("正则搜索")
        self.regexSearchButton.clicked.connect(self.regexSearchItems)  # 连接新的槽函数
        searchLayout.addWidget(self.regexSearchButton)  # 添加按钮到布局

        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        centralWidget.setLayout(layout)

        openFile = QAction('打开文件', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('打开新文件，可以打开jsonl或json格式的文件')
        openFile.triggered.connect(self.showDialog)

        self.loadAllButton = QPushButton("加载全部")
        self.loadAllButton.setStatusTip('对jsonl最多只会预加载1000行，点击该按钮可以加载剩余全部条目')
        self.loadAllButton.clicked.connect(self.loadAllItems)

        saveFile = QAction('保存文件', self)
        saveFile.setShortcut('Ctrl+S')
        saveFile.setStatusTip('保存文件')
        saveFile.triggered.connect(self.saveFile)

        toolbar = self.addToolBar('文件')
        toolbar.addAction(openFile)
        toolbar.addWidget(self.loadAllButton)  # 将按钮添加到布局中
        # toolbar.addAction(saveFile)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('JLineEditor')
        self.treeView.setAlternatingRowColors(True)
        self.treeView.setIndentation(20)
        self.treeView.setSortingEnabled(True)
        self.treeView.setStyleSheet(LargeStrings.treeViewStyles)
        self.plainTextEdit.setWordWrapMode(QTextOption.WordWrap)
        self.plainTextEdit.setReadOnly(True)
        self.showMaximized()

        self.treeView.setSortingEnabled(False)  # 禁止排序
        self.plainTextEdit.textChanged.connect(self.updateJson)  # 连接 textChanged 信号到新的槽函数

    def addPane(self, widget, title):
        layout = QVBoxLayout()
        layout.addWidget(QLabel(title))
        layout.addWidget(widget)
        pane = QWidget()
        pane.setLayout(layout)
        return pane

    def saveFile(self):
        fname = QFileDialog.getSaveFileName(self, '保存文件',
                                            self.lastOpenDir if hasattr(self, 'lastOpenPath') else '/home')

        if fname[0]:
            self.lastOpenDir = fname[0]
            with open(fname[0], 'w', encoding='utf8') as f:
                for line in self.lines:
                    f.write(line)

    def updateJson(self):
        newText = self.plainTextEdit.toPlainText()
        self.currentlyEditingItem.setText(newText)  # 更新模型项的内容

        # 更新 JSON 数据
        # self.lines[self.listWidget.currentRow()] = self.modelToJson(self.model)

    def showDialog(self, *, fname=None):
        if fname is None:
            fname = QFileDialog.getOpenFileName(self, '打开文件',
                                                self.lastOpenDir,
                                                "JSON files (*.json *.jsonl)")
            fname = fname[0]

        if fname:
            self.lastOpenDir = os.path.dirname(QFileInfo(fname).absolutePath())
            self.save_settings()

            # 打开新文件时，重置 allItemsLoaded 变量
            self.allItemsLoaded = False

            # 清空旧数据
            self.lines = []
            self.listWidget.clear()

            # 1 打开文件
            start_time = time.time()  # 开始计时
            self.lastOpenDir = fname
            if fname.endswith('.json'):
                with open(fname, 'r', encoding='utf8') as f:
                    jsonData = json.load(f)
                    self.lines = [json.dumps(jsonData, ensure_ascii=False)]
            else:
                with open(fname, 'r', encoding='utf8') as f:
                    self.lines = f.readlines()
            self.statusBar.showMessage(f"文件打开耗时: {time.time() - start_time:.2f} 秒")
            QApplication.processEvents()

            # 2 加载条目数据
            start_time = time.time()
            self.setWindowTitle(f'JLineViewer - {fname}')
            self.listWidget.addItems([f'{i + 1}. {line.strip()}' for i, line in enumerate(self.lines[:1000])])
            QApplication.processEvents()

            # TODO 只有总条目数大于1000时才显示"仅预加载1000条"
            if len(self.lines) > 1000:
                self.statusBar.showMessage(f"总条目数: {len(self.lines)}, 仅预加载1000条，"
                                           f"加载条目耗时: {time.time() - start_time:.2f} 秒")
            else:
                self.statusBar.showMessage(f"总条目数: {len(self.lines)}, 加载条目耗时: {time.time() - start_time:.2f} 秒")

    def loadJson(self, item):
        index = self.listWidget.row(item)
        jsonData = json.loads(self.lines[index])
        self.model = self.dictToModel(jsonData)
        self.treeView.setModel(self.model)
        self.treeView.expandAll()

        # 使用自定义的delegate
        delegate = TextEditDelegate(self.treeView)
        self.treeView.setItemDelegate(delegate)

        self.treeView.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.treeView.header().setSectionResizeMode(1, QHeaderView.Stretch)

    def loadAllItems(self):
        if not self.allItemsLoaded:
            start_time = time.time()
            self.listWidget.addItems([f'{i + 1}. {line.strip()}' for i, line in enumerate(self.lines[1000:])])
            QApplication.processEvents()
            self.statusBar.showMessage(f"全部加载完毕, 总条目数: {len(self.lines)}, 加载耗时: {time.time() - start_time:.2f} 秒")

            # 加载完所有项目后，设置 allItemsLoaded 变量为 True
            self.allItemsLoaded = True
        else:
            self.statusBar.showMessage(f"所有条目已经加载完毕。")

    def searchItems(self):
        if hasattr(self, 'lines'):
            start_time = time.time()
            searchText = self.searchLineEdit.text()
            foundCount = 0
            for i in range(self.listWidget.count()):
                item = self.listWidget.item(i)
                if searchText in item.text():
                    item.setHidden(False)
                    foundCount += 1
                else:
                    item.setHidden(True)
            QApplication.processEvents()
            self.statusBar.showMessage(
                f"总条目数: {len(self.lines)}, 找到: {foundCount}, 搜索耗时: {time.time() - start_time:.2f} 秒")

    def regexSearchItems(self):
        if hasattr(self, 'lines'):
            start_time = time.time()
            searchText = self.searchLineEdit.text()
            regexPattern = re.compile(searchText)  # 使用输入的文本创建正则表达式
            foundCount = 0
            for i in range(self.listWidget.count()):
                item = self.listWidget.item(i)
                if regexPattern.search(item.text()):  # 使用正则表达式搜索
                    item.setHidden(False)
                    foundCount += 1
                else:
                    item.setHidden(True)
            self.statusBar.showMessage(
                f"总条目数: {len(self.lines)}, 找到: {foundCount}, 搜索耗时: {time.time() - start_time:.2f} 秒")

    def editItem(self, index):
        self.currentlyEditingItem = self.model.itemFromIndex(index)  # 保存当前正在编辑的项
        target_data = self.currentlyEditingItem.data(Qt.UserRole + 1)  # 获取存储的 JSON 数据
        if target_data is not None:
            target_text = json.dumps(target_data, indent=2)  # 把 JSON 对象格式化为字符串
        else:
            target_text = self.currentlyEditingItem.text()
        self.plainTextEdit.setPlainText(target_text)

    def dictToModel(self, data, parent=None):
        if parent is None:
            parent = QStandardItemModel()
            parent.setHorizontalHeaderLabels(['Key', 'Value'])
        if isinstance(data, dict):
            for key, value in data.items():
                self.dataToModel(key, value, parent)
        return parent

    def dataToModel(self, key, value, parent):
        if isinstance(value, dict):
            item = KeyStandardItem(key)
            parent.appendRow([item, QStandardItem('')])
            for k, v in value.items():
                self.dataToModel(k, v, item)
        elif isinstance(value, list):
            # 方案 1
            # for i, v in enumerate(value):
            #     self.dataToModel(f"{key}[{i}]", v, parent)

            # 方案2 添加一个父节点
            list_parent = KeyStandardItem(key)
            parent.appendRow([list_parent, QStandardItem('')])
            # 将list的元素添加到父节点下
            for i, v in enumerate(value):
                self.dataToModel(f"{key}[{i}]", v, list_parent)
        else:
            parent.appendRow([KeyStandardItem(key), QStandardItem(str(value))])

    # def itemChanged(self, item):
    #     # 当一个模型项改变时，重新生成 JSON 数据
    #     self.lines[self.listWidget.currentRow()] = self.modelToJson(self.model)
    #
    # def modelToJson(self, model, parent=QModelIndex(), key=None):
    #     """ 这段功能有问题，暂不能开启 """
    #     rows = model.rowCount(parent)
    #     if rows == 0:
    #         # leaf node
    #         sibling = model.sibling(parent.row(), 1, parent)
    #         return model.data(sibling)
    #     else:
    #         # branch node
    #         json_data = {}
    #         for i in range(rows):
    #             index = model.index(i, 0, parent)
    #             child_key = model.data(index)
    #             child_value = self.modelToJson(model, index, child_key)
    #             json_data[child_key] = child_value
    #         return json.dumps(json_data, ensure_ascii=False)

    def dragEnterEvent(self, event):
        """
        当用户开始拖动文件到部件上时，这个方法会被调用。
        我们需要检查拖动的数据是不是文件类型（mime类型是'text/uri-list'）。
        """
        print("dragEnterEvent called")
        # if event.mimeData().hasFormat('text/uri-list'):
        event.acceptProposedAction()

    def dropEvent(self, event):
        """
        当用户在部件上释放（drop）文件时，这个方法会被调用。
        我们需要获取文件路径，然后判断文件类型是不是我们支持的类型。
        如果文件是我们支持的类型，我们就可以处理这个文件。
        """
        print("dropEvent called")
        # 获取文件路径
        file_paths = event.mimeData().urls()
        if file_paths:
            file_path = file_paths[0].toLocalFile()  # 取第一个文件
            # 检查文件扩展名是不是我们支持的类型
            if file_path.endswith(('.json', '.jsonl')):
                # 调用处理文件的方法
                self.showDialog(fname=file_path)
            else:
                QMessageBox.warning(self, "File Type Error",
                                    "Only .json or .jsonl files are supported.")

    # 当应用程序关闭时，保存设置
    def closeEvent(self, event):
        self.save_settings()
        event.accept()


def start_jlineviewer(fname=None):
    app = QApplication(sys.argv)
    ex = JLineViewer()
    if isinstance(fname, list):  # 可以输入一个list字典数据，会转存到临时目录里查看
        tempfile = XlPath.tempfile(suffix='.jsonl')
        tempfile.write_jsonl(fname)
        fname = tempfile
    if fname:
        ex.showDialog(fname=fname)
    sys.exit(app.exec_())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = JLineViewer()
    sys.exit(app.exec_())
