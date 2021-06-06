#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/02 16:06

import concurrent.futures
import math
import os
import pprint
import re
import tempfile

import cv2
import numpy as np

import subprocess

try:
    import fitz
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'PyMuPdf'])
    import fitz

from pyxllib.file.specialist import File, Dir, writefile, filescopy, filesdel
from pyxllib.debug.pupil import dprint
from pyxllib.debug.specialist import browser
from pyxllib.cv.specialist import imwrite, zoomsvg


class FitzPdf:
    def __init__(self, file):
        self.src_file = File(file)
        self.doc = fitz.open(file)

    def to_images(self, dst_dir=None, file_fmt='{filestem}_{number}.png', num_width=None, *,
                  scale=1, start=1, fmt_onepage=False):
        """ 将pdf转为若干页图片

        :param dst_dir: 目标目录
            默认情况下，只有一页pdf则存储到对应的pdf目录，多页则存储到同名子目录下
            如果不想这样被智能控制，只要指定明确的dst即可
        :param file_fmt: 后缀格式，包括修改导出的图片类型，注意要用 {} 占位符表示页码编号
        :param num_width: 生成的每一页文件编号，使用的数字前导0域宽
            默认根据pdf总页数来设置对应所用域宽
            0表示不设域宽
        :param scale: 对每页图片进行缩放
        :param start: 起始页码
        :param fmt_onepage: 当pdf就只有一页的时候，是否还对导出的图片编号
            默认只有一页的时候，进行优化，不增设后缀格式

        注：如果要导出单张图，可以用 FitzPdfPage.get_cv_image
        """
        # 1 基本参数计算
        srcfile, doc = self.src_file, self.doc
        filestem, n_page = srcfile.stem, doc.pageCount

        # 自动推导目标目录
        if dst_dir is None:
            dst_dir = Dir(srcfile.stem, srcfile.parent) if n_page > 1 else Dir(srcfile.parent)
        Dir(dst_dir).ensure_dir()

        # 域宽
        if num_width is None:
            num_width = math.ceil(math.log10(n_page + 1))  # 根据总页数计算需要的对齐域宽

        # 2 导出图片
        if fmt_onepage or n_page != 1:  # 多页的处理规则
            for i in range(n_page):
                im = FitzPdfPage(self.doc.loadPage(i)).get_cv_image(scale)
                number = ('{:0' + str(num_width) + 'd}').format(i + start)  # 前面的括号不要删，这样才是完整的一个字符串来使用format
                imwrite(im, File(file_fmt.format(filestem=filestem, number=number), dst_dir))
        else:
            im = FitzPdfPage(self.doc.loadPage[0]).get_cv_image(scale)
            imwrite(im, File(srcfile.stem + os.path.splitext(file_fmt)[1], dst_dir))

    def get_page(self, number):
        return self.doc.loadPage(number)


class FitzPdfPage:
    def __init__(self, page):
        self.page = page

    def get_svg_image(self, scale=1):
        # svg 是一段表述性文本
        txt = self.page.getSVGimage()
        if scale != 1:
            txt = zoomsvg(txt, scale)
        return txt

    def _get_png_data(self, scale=1):
        # TODO 增加透明通道？
        if scale != 1:
            pix = self.page.getPixmap(fitz.Matrix(scale, scale))  # 长宽放大到scale倍
        else:
            pix = self.page.getPixmap()
        return pix.getPNGData()

    def get_cv_image(self, scale=1):
        arr = np.fromstring(self._get_png_data(scale), dtype=np.uint8)
        return cv2.imdecode(arr, flags=1)

    def get_pil_image(self, scale=1):
        raise NotImplementedError

    def get_text(self, fmt='text'):
        """
        :param fmt: 存储格式，可以获得整页的纯文本，也可以获得dict结构存储的内容
        """
        return self.page.getText(fmt)


class DemoFitz:
    """
    安装： pip install PyMuPdf
    使用： import fitz
    官方文档： https://pymupdf.readthedocs.io/en/latest/intro/
        demo： https://github.com/rk700/PyMuPDF/tree/master/demo
        examples： https://github.com/rk700/PyMuPDF/tree/master/examples
    """

    def __init__(self, file):
        self.doc = fitz.open(file)

    def message(self):
        """查看pdf文档一些基础信息"""
        dprint(fitz.version)  # fitz模块的版本
        dprint(self.doc.pageCount)  # pdf页数
        dprint(self.doc._getXrefLength())  # 文档的对象总数

    def getToC(self):
        """获得书签目录"""
        toc = self.doc.getToC()
        browser(toc)

    def setToC(self):
        """设置书签目录
        可以调层级、改名称、修改指向页码
        """
        toc = self.doc.getToC()
        toc[1][1] = '改标题名称'
        self.doc.setToC(toc)
        file = File('a.pdf', Dir.TEMP).to_str()
        self.doc.save(file, garbage=4)
        browser(file)

    def setToC2(self):
        """修改人教版教材的标签名"""
        toc = self.doc.getToC()
        newtoc = []
        for i in range(len(toc)):
            name = toc[i][1]
            if '.' in name: continue
            # m = re.search(r'\d+', name)
            # if m: name = name.replace(m.group(), digits2chinese(int(m.group())))
            m = re.search(r'([一二三四五六]年级).*?([上下])', name)
            if i < len(toc) - 1:
                pages = toc[i + 1][2] - toc[i][2] + 1
            else:
                pages = self.doc.pageCount - toc[i][2] + 1
            toc[i][1] = m.group(1) + m.group(2) + '，' + str(pages)
            newtoc.append(toc[i])
        self.doc.setToC(newtoc)
        file = writefile(b'', 'a.pdf', if_exists='delete')
        self.doc.save(file, garbage=4)

    def rearrange_pages(self):
        """重新布局页面"""
        self.doc.select([0, 0, 1])  # 第1页展示两次后，再跟第2页
        file = writefile(b'', 'a.pdf', root=Dir.TEMP, if_exists='delete')
        self.doc.save(file, garbage=4)  # 注意要设置garbage，否则文档并没有实际删除内容压缩文件大小
        browser(file)

    def page2png(self, page=0):
        """ 查看单页渲染图片 """
        page = self.doc.loadPage(page)  # 索引第i页，下标规律同py，支持-1索引最后页
        # dprint(page.bound())  # 页面边界，x,y轴同图像处理中的常识定义，返回Rect(x0, y0, x1, y1)

        pix = page.getPixmap(fitz.Matrix(2, 2))  # 获得页面的RGBA图像，Pixmap类型；还可以用page.getSVGimage()获得矢量图
        # pix.writePNG('page-0.png')  # 将Pixmal
        pngdata = pix.getPNGData()  # 获png文件的bytes字节码
        # print(len(pngdata))
        # browser(pngdata, 'a.png')  # 用我的工具函数打开图片

        return pngdata

    def pagetext(self):
        """单页上的文本"""
        page = self.doc[0]

        # 获得页面上的所有文本，还支持参数： html，dict，xml，xhtml，json
        text = page.getText('text')
        dprint(text)

        # 获得页面上的所有文本（返回字典对象）
        textdict = page.getText('dict')
        textdict['blocks'] = textdict['blocks'][:-1]
        browser(pprint.pformat(textdict))

    def text(self):
        """获得整份pdf的所有文本"""
        return '\n'.join([page.getText('text') for page in self.doc])

    def xrefstr(self):
        """查看pdf文档的所有对象"""
        xrefstr = []
        n = self.doc._getXrefLength()
        for i in range(1, n):  # 注意下标实际要从1卡开始
            # 可以边遍历边删除，不影响下标位置，因为其本质只是去除关联引用而已
            xrefstr.append(self.doc._getXrefString(i))
        browser('\n'.join(xrefstr))

    def page_add_ele(self):
        """往页面添加元素
        添加元素前后xrefstr的区别： https://paste.ubuntu.com/p/Dxhnzp4XJ2/
        """
        self.doc.select([0])
        page = self.doc.loadPage(0)
        # page.insertText(fitz.Point(100, 200), 'test\ntest')
        file = File('a.pdf', Dir.TEMP).to_str()
        dprint(file)
        self.doc.save(file, garbage=4)
        browser(file)


def pdf2svg_oldversion(pdffile, target=None, *, trim=False):
    """新版的，pymupdf生成的svg无法配合inkscape进行trim，所以这个旧版暂时还是要保留

    :param pdffile: 一份pdf文件
    :param target: 目标目录
        None：
            如果只有一页，转换到对应目录下同名svg文件
            如果有多页，转换到同名目录下多份svg文件
    :param trim:
        True: 去除边缘空白
    :return:

    需要第三方工具：pdf2svg（用于文件格式转换），inkscape（用于svg编辑优化）
        注意pdf2svg的参数不支持中文名，因为这个功能限制，搞得我这个函数实现好麻烦！
        还要在临时文件夹建立文件，因为重名文件+多线程问题，还曾引发一个bug搞了一下午。
        （这些软件都以绿色版形式整理在win3/imgtools里）

    注意！！！ 这个版本的代码先不要删！先不要删！先不要删！包括pdf2svg.exe那个蠢货软件也先别删！
        后续研究inkscape这个蠢货的-D参数在处理pymupdf生成的svg为什么没用的时候可以进行对比
    """
    import fitz
    pages = fitz.open(pdffile).pageCount

    basename = tempfile.mktemp()
    f1 = basename + '.pdf'
    filescopy(pdffile, f1)  # 复制到临时文件，防止中文名pdf2svg工具处理不了

    if pages == 1:
        if target is None: target = pdffile[:-3] + 'svg'
        f2 = basename + '.svg'
        # print(['pdf2svg.exe', f1, f2])
        subprocess.run(['pdf2svg.exe', f1, f2])

        if trim: subprocess.run(['inkscape.exe', '-f', f2, '-D', '-l', f2])
        filescopy(f2, target)
    else:
        if target is None: target = pdffile[:-4] + '_svg\\'
        executor = concurrent.futures.ThreadPoolExecutor()
        File(basename + '/').ensure_parent()

        def func(f1, f2, i):
            subprocess.run(['pdf2svg.exe', f1, f2, str(i)])
            if trim: subprocess.run(['inkscape.exe', '-f', f2, '-D', '-l', f2])
            filescopy(f2, target + f'{i}.svg')

        for i in range(1, pages + 1):
            f2 = basename + f'\\{i}.svg'
            executor.submit(func, f1, f2, i)
        executor.shutdown()
        filescopy(basename, target[:-1])
        filesdel(basename + '/')

    filesdel(f1)


def pdf2imagebase(pdffile, target=None, scale=None, ext='.png'):
    """
    使用python的PyMuPdf模块，不需要额外插件
    导出的图片从1开始编号
    TODO 要加多线程？效率影响大吗？

    :param pdffile: pdf原文件
    :type target: 相对于原文件所在目录的目标目录名，也可以写文件名，表示重命名
        None：
            当该pdf只有1页时，才默认把图片转换到当前目录。
            否则默认新建一个文件夹来存储图片。（目录名默认为文件名）
    :param scale: 缩放尺寸
        1：原尺寸
        1.5：放大为原来的1.5倍
    :param ext: 导出的图片格式
    :return: 返回生成的图片列表
    """

    import fitz
    # 1 基本参数计算
    pdf = fitz.open(pdffile)
    num_pages = pdf.pageCount

    # 大于1页的时候，默认新建一个文件夹来存储图片
    if target is None:
        if num_pages > 1:
            target = File(pdffile).stem + '/'
        else:
            target = File(pdffile).dirname + '/'

    newfile = File(target, pdffile.parent).to_str()
    if newfile.endswith('.pdf'): newfile = os.path.splitext(newfile)[0] + ext
    File(newfile).ensure_parent()

    # 2 图像数据的获取
    def get_svg_image(n):
        page = pdf.loadPage(n)
        txt = page.getSVGimage()
        if scale: txt = zoomsvg(txt, scale)
        return txt

    def get_png_image(n):
        """获得第n页的图片数据"""
        page = pdf.loadPage(n)
        if scale:
            pix = page.getPixmap(fitz.Matrix(scale, scale))  # 长宽放大到scale倍
        else:
            pix = page.getPixmap()
        return pix.getPNGData()

    # 3 分析导出的图片文件名
    files = []
    if num_pages == 1:
        image = get_svg_image(0) if ext == '.svg' else get_png_image(0)
        files.append(newfile)
        File(newfile).write(image, if_exists='delete')
    else:  # 有多页
        number_width = math.ceil(math.log10(num_pages + 1))  # 根据总页数计算需要的对齐域宽
        stem, ext = os.path.splitext(newfile)
        for i in range(num_pages):
            image = get_svg_image(i) if ext == '.svg' else get_png_image(i)
            name = ('-{:0' + str(number_width) + 'd}').format(i + 1)  # 前面的括号不要删，这样才是完整的一个字符串来使用format
            files.append(stem + name + ext)
            File(stem + name + ext).write(image, if_exists='delete')
    return files


def pdf2png(pdffile, target=None, scale=None):
    """
    :param pdffile: pdf路径
    :param target: 目标位置
    :param scale: 缩放比例
    :return: list，生成的png图片清单

    # 可以不写target，默认处理：如果单张png则在同目录，多张则会建个同名目录存储
    >> pdf2png(r'D:\slns+\immovables\immovables_data\test\X\A0001.pdf')

    # 指定存放位置：
    >> pdf2png(r'D:\slns+\immovables\immovables_data\test\X\A0001.pdf', r'D:\slns+\immovables\immovables_data\test\X')

    """
    return pdf2imagebase(pdffile, target=target, scale=scale, ext='.png')


def pdf2svg(pdffile, target=None, scale=None, trim=False):
    """
    :param pdffile: 见pdf2imagebase
    :param target: 见pdf2imagebase
    :param scale: 见pdf2imagebase
    :param trim: 如果使用裁剪功能，会调用pdf-crop-margins第三方工具
        https://pypi.org/project/pdfCropMargins/
    :return:
    """
    if trim:  # 先对pdf文件进行裁剪再转换
        pdf = File(pdffile)
        newfile = File('origin.pdf', pdf.parent).to_str()
        pdf.copy(newfile)
        # subprocess.run(['pdf-crop-margins.exe', '-p', '0', newfile, '-o', pdffile], stderr=subprocess.PIPE) # 本少： 会裁过头！
        # 本少： 对于上下边处的 [] 分数等，会裁过头，先按百分比 -p 0 不留边，再按绝对点数收缩/扩张 -a -1  负数为扩张，单位为bp
        # 本少被自己坑了，RamDisk 与 pdf-crop-margins.exe 配合，只能取 SCSI 硬盘，如果 Direct-IO 就不行，还不报错，还以为是泽少写的代码连报错都不会
        subprocess.run(['pdf-crop-margins.exe', '-p', '0', '-a', '-1', newfile, '-o', pdffile],
                       stderr=subprocess.PIPE)
    # TODO 有时丢图
    pdf2imagebase(pdffile, target=target, scale=scale, ext='.svg')


def pdfs2pngs(src, scale=None, pinterval=None):
    """ 将目录下所有pdf转png

    :param src: 原pdf数据路径
    :param scale: 转图片时缩放比例，例如2表示长宽放大至2被
    :param pinterval: 每隔多少个pdf输出处理进度
        默认None，不输出
    """
    from functools import partial
    func = partial(pdf2png, scale=scale)
    Dir(src).select('**/*.pdf').procpaths(func, pinterval=pinterval)
