#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 厦门理工计算机学院 王大寒/(PRIU)福建省模式识别与图像理解重点实验室
# @Document : https://www.yuque.com/xlpr/doai/hesuan
# @Email  : (陈)877362867@qq.com
# @Address: 厦门理工综合楼1905
# @Date   : 2022/08/31

import datetime
import os
import re
import sys
import time

import fire
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import xlrd2
import cv2

from PyQt5.QtCore import QSize, Qt, QUrl
from PyQt5.QtGui import QFont, QIcon, QDesktopServices
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QGridLayout, QWidget,
                             QLineEdit, QToolButton, QFileDialog, QProgressBar)

from pyxllib.file.xlsxlib import openpyxl
from pyxllib.gui.qt import get_input_widget
from pyxllib.xl import TicToc, XlPath, matchpairs, get_etag

from pyxlpr.ai.clientlib import XlAiClient


class PPOCR:
    def __init__(self):
        self.xlapi = XlAiClient()
        self.xlapi.login_priu('hs14TkixP7fD#')
        # 本地缓存，已经识别过的，不重复提交服务器。
        self.cache = self.read_cache()  # etag: result_attrs

    def read_cache(self):
        """ 每个月的结果会在本地做缓存 """
        f = XlPath.tempdir() / datetime.date.today().strftime('hesuan%Y%m.pkl')
        if f.is_file():
            return f.read_pkl()
        else:
            return {}

    def write_cache(self):
        f = XlPath.tempdir() / datetime.date.today().strftime('hesuan%Y%m.pkl')
        f.write_pkl(self.cache)

    def get_image_buffer(self, f):
        # 核酸识别的图片大概率不需要很清晰的图片，每张图文件大小可以控制在300kb以内的jpg文件
        # ratio用不到，api接口只会返回关键类别字典值
        buffer, ratio = self.xlapi.adjust_image(f, limit_b64buffer_size=500 * 1024, b64encode=True)
        return buffer.decode()

    def parse_light(self, f):
        xlapi = self.xlapi
        etag = 'parse_light,' + get_etag(XlPath(f).read_bytes())
        if etag not in self.cache:
            data = {'image': self.get_image_buffer(f)}
            attrs = requests.post(f'{xlapi._priu_host}/api/hesuan/parse_light',
                                  json=data, headers=xlapi._priu_header).json()
            if 'xlapi' in attrs:
                del attrs['xlapi']
            self.cache[etag] = attrs
        return self.cache[etag]

    def parse_multi_layout_light(self, files):
        xlapi = self.xlapi
        etag = 'parse_multi_layout_light,' + ','.join([get_etag(XlPath(f).read_bytes()) for f in files])
        if etag not in self.cache:
            data = {'images': [self.get_image_buffer(x) for x in files]}
            attrs = requests.post(f'{xlapi._priu_host}/api/hesuan/parse_multi_layout_light',
                                  json=data, headers=xlapi._priu_header).json()
            # print(attrs)
            if 'xlapi' in attrs:
                del attrs['xlapi']
            self.cache[etag] = attrs
        return self.cache[etag]


class Hesuan:
    """ 核酸检测的各种功能接口 """

    def __init__(self, task, *, use_online_api=True):
        self.task = task
        # 这里类似配置，把所有需要得变化点封装在这里
        # 双码跟三码用一样的版面分析，但是只保存想要的值，不会保存核酸检测结果之类（无核酸检测报告）
        if self.task == '日常核酸检测报告分析':
            self.columns = ['文件', '姓名', '采样时间', '检测时间', '核酸结果', '联系电话', '身份证号']  # 没有匹配名单，直接生成得列属性
            self.stu_columns = ['班级', '文件', '学号', '姓名', '采样时间', '检测时间', '核酸结果', '身份证号', '联系电话']  # 有匹配名单，直接生成得列属性
        elif self.task == '双码（健康码、行程码）分析':
            self.columns = ['文件', '姓名', '联系电话', '14天经过或途经', '健康码颜色', '身份证号']
            self.stu_columns = ['班级', '文件', '学号', '姓名', '联系电话', '14天经过或途经', '健康码颜色', '身份证号']
            self.file_num = 2
        elif self.task == '双码（健康码、行程码）+ 24小时核酸检测报告分析':
            self.columns = ['文件', '姓名', '采样时间', '检测时间', '核酸结果', '联系电话', '14天经过或途经', '健康码颜色', '身份证号']
            self.stu_columns = ['班级', '文件', '学号', '姓名', '采样时间', '检测时间', '核酸结果',
                                '14天经过或途经', '健康码颜色', '身份证号', '联系电话']
            self.file_num = 3

        self.ppocr = PPOCR() if use_online_api else None

    def ensure_images(self, imdir):
        """ 一些非图片个数数据自动转图片

        其中压缩包只支持zip格式
        """
        # 1 先解压所有的压缩包
        for f in imdir.rglob_files('*.zip'):
            from pyxllib.file.packlib import unpack_archive
            # 没有对应同名目录则解压
            if not f.with_name(f.stem).is_dir():
                unpack_archive(f, wrap=1)

        # 2 再检索所有的pdf
        for f in imdir.rglob_files('*'):
            # 可能存在pdf格式的文件，将其转成jpg图片
            suffix = f.suffix.lower()
            if suffix == '.pdf':
                f2 = f.with_suffix('.jpg')
                if not f2.exists():
                    from pyxllib.file.pdflib import FitzDoc
                    # im = FitzDoc(f).load_page(0).get_pil_image()
                    f = FitzDoc(f)
                    im = f.load_page(0).get_pil_image()
                    im.save(f2)

    def similar(self, x, y):
        """ 计算两条数据间的相似度

        这里不适合用编辑距离，因为levenshtein涉及到c++编译，会让配置变麻烦。
        这里就用正则等一些手段计算相似度。

        相关字段：姓名(+文件)、联系电话、身份证号
        """
        t = 0
        if y['姓名'] in x['文件']:
            t += 100
        if y['姓名'] == x['姓名']:
            t += 200
        elif re.match(re.sub(r'\*+', r'.*', x['姓名']) + '$', y['姓名']):
            t += 50

        def check_key_by_asterisk(k):
            # 带星号*的相似度匹配
            nonlocal t
            if isinstance(y[k], str):
                if y[k] == x[k]:
                    t += 100
                else:
                    if y[k][-2:] == x[k][-2:]:
                        t += 20
                    if y[k][:2] == x[k][:2]:
                        t += 10

        check_key_by_asterisk('联系电话')
        check_key_by_asterisk('身份证号')
        return t

    def link_table(self, df1, df2):
        # 1 找出df1中每一张图片，匹配的是df2中的谁
        idxs = []
        for idx1, row1 in df1.iterrows():
            max_idx2, max_sim = -1, 0
            for idx2, row2 in df2.iterrows():
                sim = self.similar(row1, row2)
                if sim > max_sim:
                    max_idx2 = idx2
                    max_sim = sim
            if max_idx2 == -1:
                # 如果没有匹配到，到时候要直接展示原来的数据
                idxs.append(-idx1)
            else:
                idxs.append(max_idx2)

        for k in ['姓名', '联系电话', '身份证号']:
            df1[k] = [(df2[k][i] if i > 0 else df1[k][-i]) for i in idxs]
        for k in ['班级', '学号']:
            df1[k] = [(df2[k][i] if i > 0 else '') for i in idxs]

        # 2 列出涉及到的班级的成员名单
        classes = set(df1['班级'])  # 0.7版是只显示图片涉及到的班级
        # classes = set(df2['班级'])  # 0.9版是直接以清单中的情况显示
        idxs_ = set(idxs)
        for idx2, row2 in df2.iterrows():
            if idx2 not in idxs_ and row2['班级'] in classes:
                line = {}
                for k in ['文件', '采样时间', '检测时间', '核酸结果']:
                    line[k] = ''
                for k in ['姓名', '联系电话', '身份证号', '班级', '学号']:
                    line[k] = row2[k].strip()
                df1 = df1.append(line, ignore_index=True)

        # 3 重新调整列
        # 保护隐私，不显示电话、身份证
        df = df1[['班级', '文件', '学号', '姓名', '采样时间', '检测时间', '核酸结果']]
        df.sort_values(['班级', '学号'], inplace=True)
        df.index = np.arange(1, len(df) + 1)

        return df

    def link_table2(self, df1, df2):
        # 1 匹配情况
        xs = [row for idx, row in df1.iterrows()]
        ys = [row for idx, row in df2.iterrows()]
        ms = matchpairs(xs, ys, self.similar, least_score=40, index=True)
        idxs = {m[1]: m[0] for m in ms}

        # + 辅助函数
        def quote(s):
            # 带上括号注释
            return f'({s})' if s else ''

        # df2如果有扩展列，也显示出来
        custom_cols = []
        for c in df2.columns:  # 按照原表格的列顺序展示
            c = str(c)
            if c not in {'班级', '学号', '姓名', '身份证号', '联系电话'}:
                custom_cols.append(c)

        def extend_cols(y=None):
            if y is None:
                return [''] * len(custom_cols)
            else:
                return [('' if y[c] != y[c] else y[c]) for c in custom_cols]

        # 2 模板表清单
        ls = []

        # self.stu_colunm, 放在这里方便调试, 分别是日常检测任务的属性列, 双码的任务的属性列, 三码任务的属性列
        # columns = ['班级', '文件', '学号', '姓名', '采样时间', '检测时间', '核酸结果', '身份证号', '联系电话'] + custom_cols
        # colunms = ['班级', '文件', '学号', '姓名', '联系电话', '14天经过或途经', '健康码颜色', '身份证号'] + custom_cols
        # columns = ['班级', '文件', '学号', '姓名', '采样时间', '检测时间', '核酸结果', '14天经过或途经', '健康码颜色', '身份证号', '联系电话'] + custom_cols

        columns = self.stu_columns + custom_cols
        for idx, y in df2.iterrows():
            i = idxs.get(int(idx), -1)
            if self.task == '日常核酸检测报告分析':
                record = [y['班级'], '', y['学号'], y['姓名'], '', '', '', y['身份证号'], y['联系电话']] + extend_cols(y)
            elif self.task == '双码（健康码、行程码）分析':
                record = [y['班级'], '', y['学号'], y['姓名'], y['联系电话'], '', '', y['身份证号']] + extend_cols(y)
            elif self.task == '双码（健康码、行程码）+ 24小时核酸检测报告分析':
                record = [y['班级'], '', y['学号'], y['姓名'], '', '', '', '', '', y['身份证号'], y['联系电话']] + extend_cols(y)
            else:
                raise NotImplementedError

            if i != -1:  # 找得到匹配项
                x = xs[i]
                if self.task == '日常核酸检测报告分析':
                    record[1] = x['文件']
                    record[3] += quote(x['姓名'])
                    record[4] = x['采样时间']
                    record[5] = x['检测时间']
                    record[6] = x['核酸结果']
                    record[7] += quote(x['身份证号'])
                    record[8] += quote(x['联系电话'])
                elif self.task == '双码（健康码、行程码）分析':
                    record[1] = x['文件']
                    record[3] += quote(x['姓名'])
                    record[4] += quote(x['联系电话'])
                    record[5] = x['14天经过或途经']
                    record[6] = x['健康码颜色']
                    record[7] = quote(x['身份证号'])
                elif self.task == '双码（健康码、行程码）+ 24小时核酸检测报告分析':
                    record[1] = x['文件']
                    record[3] += quote(x['姓名'])
                    record[4] = x['采样时间']
                    record[5] = x['检测时间']
                    record[6] = x['核酸结果']
                    record[7] = x['14天经过或途经']
                    record[8] = x['健康码颜色']
                    record[9] += quote(x['身份证号'])
                    record[10] += quote(x['联系电话'])

            ls.append(record)

        # 3 df1中剩余未匹配图片加在最后面
        idxs = {m[0] for m in ms}
        for i, x in enumerate(xs):
            if i not in idxs:
                if self.task == '日常核酸检测报告分析':
                    record = ['', x['文件'], '', quote(x['姓名']),
                              x['采样时间'], x['检测时间'], x['核酸结果'],
                              quote(x['身份证号']), quote(x['联系电话'])]
                elif self.task == '双码（健康码、行程码）分析':
                    record = ['', x['文件'], '', quote(x['姓名']),
                              x['14天经过或途经'], x['健康码颜色'], quote(x['身份证号'])]
                elif self.task == '双码（健康码、行程码）+ 24小时核酸检测报告分析':
                    record = ['', x['文件'], '', quote(x['姓名']),
                              x['采样时间'], x['检测时间'], x['核酸结果'], x['14天经过或途经'], x['健康码颜色'],
                              quote(x['身份证号']), quote(x['联系电话']), ]
                else:
                    raise NotImplementedError

                ls.append(record + extend_cols())

        df = pd.DataFrame.from_records(ls, columns=columns)
        df.index = np.arange(1, len(df) + 1)
        return df

    def parse(self, imdir, students=None, *, pb=None):
        """
        :param imdir: 图片所在目录
        :param students: 【可选】参考学生清单
        :param pb: qt的进度条控件
        """
        # 1 基本的信息解析
        ls = []

        imdir = XlPath(imdir)
        self.ensure_images(imdir)

        # 这里将单码、双码和三码的遍历方式简单区分，双码、三码要求同一个人的两张或者三张放在一个目录下面，单码不要求
        # 单码跟其他两个任务简单区分成两个逻辑
        if self.task == '日常核酸检测报告分析':
            files = list(imdir.glob_images('**/*'))
            total_number = len(files)
            if pb:
                pb.setMaximum(total_number)

            tt = time.time()
            for i, f in tqdm(enumerate(files), '识别中'):
                if pb:
                    pb.setFormat(f'%v/%m，已用时{int(time.time() - tt)}秒')
                    pb.setValue(i)
                    QApplication.processEvents()  # 好像用这个刷新就能避免使用QThread来解决刷新问题
                if f.name == 'xmut.jpg':
                    # 有的人没设目录，直接在根目录全量查找了，此时需要过滤掉我的一个资源图片
                    continue
                try:
                    attrs = self.ppocr.parse_light(f)
                except Exception as e:
                    print(f'{f.as_posix()} 图片有问题，跳过未处理。')
                    continue

                rf, ff = f.relpath(imdir).as_posix(), f.resolve()
                attrs['文件'] = f'<a href="{ff}" target="_blank">{rf}</a><br/>'
                # 如果要展示图片，可以加：<img src="{ff}" width=100/>，但实测效果不好
                row = []
                for col in self.columns:
                    row.append(attrs.get(col, ''))
                ls.append(row)
        else:
            img_lists = self.get_img_fileslist(imdir, self.file_num)
            total_number = len(img_lists)
            if pb:
                pb.setMaximum(total_number)

            tt = time.time()
            for i, img_list in tqdm(enumerate(img_lists), '识别中'):
                if pb:
                    pb.setFormat(f'%v/%m，已用时{int(time.time() - tt)}秒')
                    pb.setValue(i)
                    QApplication.processEvents()  # 好像用这个刷新就能避免使用QThread来解决刷新问题

                if self.file_num == 1 and img_list[0].name == 'xmut.jpg':
                    # 有的人没设目录，直接在根目录全量查找了，此时需要过滤掉我的一个资源图 片
                    continue
                attrs = self.ppocr.parse_multi_layout_light(img_list)

                for f in img_list:
                    rf, ff = f.relpath(imdir).as_posix(), f.resolve()
                    # if attrs['文件'] != '':
                    if '文件' in attrs.keys():
                        attrs['文件'] = attrs['文件'] + ' ' + f'<a href="{ff}" target="_blank">{rf}</a><br/>'
                    else:
                        attrs['文件'] = f'<a href="{ff}" target="_blank">{rf}</a><br/>'
                # 如果要展示图片，可以加：<img src="{ff}" width=100/>，但实测效果不好
                row = []
                for col in self.columns:
                    row.append(attrs.get(col, ''))
                ls.append(row)
        if pb:
            pb.setFormat(f'识别图片%v/%m，已用时{int(time.time() - tt)}秒')
            pb.setValue(total_number)
            QApplication.processEvents()
        df = pd.DataFrame.from_records(ls, columns=self.columns)

        # 2 如果有表格清单，进一步优化数据展示形式
        if students is not None:
            df = self.link_table2(df, students)

        # 3 不同的日期用不同的颜色标记
        # https://www.color-hex.com/
        # 绿，蓝，黄，青，紫，红; 然后深色版本再一套
        colors = ['70ea2a', '187dd8', 'f3cf83', '99ffcc', 'ccaacc', 'ff749e',
                  '138808', '9999ff', 'd19a3f', '99ccaa', 'aa33aa', 'cc0000']

        def set_color(m):
            s1, s2 = m.groups()

            parts = s1.split('-')
            if len(parts) > 1:
                parts[1] = f'{min(12, int(parts[1])):02}'
            s1 = '-'.join(parts)

            # 每个日期颜色是固定的，而不是按照相距今天的天数来选的
            try:
                i = (datetime.date.fromisoformat(s1) - datetime.date(2022, 1, 1)).days
                c = colors[i % len(colors)]
                return f'<td bgcolor="#{c}">{s1}{s2}</td>'
            except ValueError:
                return f'<td>{s1}{s2}</td>'

        res = df.to_html(escape=False)
        res = re.sub(r'<td>(\d{4}-\d{2}-\d{2})(.*?)</td>', set_color, res)
        res = re.sub(r'<td>阳性</td>', r'<td bgcolor="#cc0000">阳性</td>', res)

        return res

    def browser(self):
        """ 打开浏览器查看 """
        pass

    def to_excel(self):
        """ 导出xlsx文件

        我前面功能格式是基于html设计的，这里要导出excel高亮格式的话，还得额外开发工作量。
        而且又会导致一大堆依赖包安装，暂时不考虑扩展。
        """
        raise NotImplementedError

    def get_img_fileslist(self, dir_path, filenum):
        """
        返回目录下大于filenum的文件列表

        :param dirpath:
        :param filenum:
        :return:
        """
        root = XlPath(dir_path)
        files_lists = []
        for _dir in root.rglob_dirs('*'):
            files_list = list(_dir.glob_images('*'))
            if len(files_list) >= filenum:
                files_lists.append(files_list)
        return files_lists


class MainWindow(QMainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()

        # 设置中文尝试
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        # 1 窗口核心属性
        self.setMinimumSize(QSize(900, 300))
        self.setWindowIcon(QIcon('models/xmut.jpg'))
        self.setWindowTitle("核酸检测分析v4.0（在线版） @(厦门理工学院)福建省模式识别与图像理解重点实验室")

        # 2 使用网格布局
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        gridLayout = QGridLayout(self)
        centralWidget.setLayout(gridLayout)

        # 3 每列标签
        for i, text in enumerate(["核酸截图目录：", "人员名单【可选】：", "sheet：", "导出报表：", "检测任务", "进度条："]):
            q = QLabel(text, self)
            q.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            gridLayout.addWidget(q, i, 0)

        # 4 每行具体功能
        def add_line1():
            """ 第一行的控件，选择截图目录 """
            # 1 输入框
            le = self.srcdirEdit = QLineEdit(self)
            le.setText(os.path.abspath(os.curdir))  # 默认当前目录
            gridLayout.addWidget(le, 0, 1)

            # 2 浏览选择
            def select_srcdir():
                directory = str(QFileDialog.getExistingDirectory()).replace('/', '\\')
                if directory:
                    self.srcdirEdit.setText(directory)

            btn = QToolButton(self)
            btn.setText(u"浏览选择...")
            btn.clicked.connect(select_srcdir)
            gridLayout.addWidget(btn, 0, 2)

        def add_line2():
            """ 第二行的控件，选择excel文件位置 """

            # 1 输入框
            def auto_search_xlsx(root=None):
                """
                :param root: 参考目录，可以不输入，默认以当前工作环境为准
                """
                if root is None:
                    root = XlPath('.')
                root = XlPath(root)

                res = ''
                for f in root.glob('*.xlsx'):
                    res = f  # 取最后一个文件
                return str(res)

            le = self.refxlsxEdit = QLineEdit(self)
            # 自动找当前目录下是否有excel文件，没有则置空
            le.setText(auto_search_xlsx(self.srcdir))
            gridLayout.addWidget(le, 1, 1)

            # 2 浏览选择
            def select_refxlsx():
                file = str(QFileDialog.getOpenFileName(filter='*.xlsx')[0]).replace('/', '\\')
                if file:
                    self.refxlsxEdit.setText(file)

            btn = QToolButton(self)
            btn.setText(u"浏览选择...")
            btn.clicked.connect(select_refxlsx)
            gridLayout.addWidget(btn, 1, 2)

        def add_line3():
            def update_combo_box():
                xlsxfile = self.refxlsx
                if xlsxfile and xlsxfile.is_file():
                    wb = openpyxl.open(str(xlsxfile), read_only=True, data_only=True)
                    self.sheets_cb.reset_items(tuple([ws.title for ws in wb.worksheets]))
                else:
                    self.sheets_cb.reset_items(tuple())

            self.sheets_cb = get_input_widget(tuple(), parent=self)
            gridLayout.addWidget(self.sheets_cb, 2, 1)
            update_combo_box()
            self.refxlsxEdit.textChanged.connect(update_combo_box)

        def add_line4():
            self.dstfileEdit = QLineEdit(self)
            self.dstfileEdit.setText(str(XlPath('./报表.html').absolute()))
            gridLayout.addWidget(self.dstfileEdit, 3, 1)

        def add_line5():
            # 任务选择、单码、双码、三码、
            def get_value():
                self.task = self.task_cb.currentText()

            self.task_cb = get_input_widget(('日常核酸检测报告分析',
                                             '双码（健康码、行程码）分析',
                                             '双码（健康码、行程码）+ 24小时核酸检测报告分析'),
                                            cur_value='日常核酸检测报告分析', parent=self)
            self.task = '日常核酸检测报告分析'
            gridLayout.addWidget(self.task_cb, 4, 1)
            self.task_cb.currentTextChanged.connect(get_value)

        def add_line6():
            # 进度条
            pb = self.pb = QProgressBar(self)
            pb.setAlignment(Qt.AlignHCenter)
            gridLayout.addWidget(pb, 5, 1)

        def add_line7():
            # 1 生成报表
            btn = QToolButton(self)
            btn.setText("生成报表")
            btn.setFont(QFont('Times', 20))
            btn.clicked.connect(self.stat)
            gridLayout.addWidget(btn, 6, 1)

            # 2 帮助文档
            def help():
                url = QUrl("https://www.yuque.com/xlpr/doai/hesuan")
                QDesktopServices.openUrl(url)

            btn = QToolButton(self)
            btn.setText("使用手册")
            btn.setFont(QFont('Times', 20))
            btn.clicked.connect(help)
            gridLayout.addWidget(btn, 6, 1, alignment=Qt.AlignHCenter)

        add_line1()
        add_line2()
        add_line3()
        add_line4()
        add_line5()
        add_line6()
        add_line7()

    @property
    def srcdir(self):
        return XlPath(self.srcdirEdit.text())

    @property
    def refxlsx(self):
        p = XlPath(self.refxlsxEdit.text())
        if p.is_file():
            return p

    def get_sheet_name(self):
        return self.sheets_cb.currentText()

    def get_students(self):
        if self.refxlsx:
            df = pd.read_excel(self.refxlsx, self.get_sheet_name(),
                               dtype={'学号': str, '联系电话': str, '身份证号': str, '姓名': str})
            df = df[(~df['姓名'].isna()) & (~df['班级'].isna()) & (~df['学号'].isna())]
            return df
        return None

    @property
    def dstfile(self):
        return XlPath(self.dstfileEdit.text())

    def stat(self):
        hs = Hesuan(task=self.task)
        res = hs.parse(self.srcdir, self.get_students(), pb=self.pb)
        dstfile = self.dstfile
        dstfile.write_text(res, encoding='utf8')
        hs.ppocr.write_cache()
        os.startfile(dstfile)


def gui():
    """ 打开可视化窗口使用 """
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())


def 性能测试():
    model = PPOCR()

    files = list(XlPath('/home/chenkunze/data/hesuan/data/A1 核酸报告-按模板分类').rglob_images())
    tt = time.time()
    for f in tqdm(files):
        model.parse_light(f)
    print((time.time() - tt) / len(files), '秒/张')

    model.write_cache()


if __name__ == '__main__':
    if len(sys.argv) == 1:  # 默认执行main函数
        sys.argv += ['gui']
    with TicToc(__name__):
        fire.Fire()
