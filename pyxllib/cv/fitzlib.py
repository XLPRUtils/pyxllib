#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/02 16:06

import itertools
import json
import os
import pprint
import re
import subprocess

try:
    import fitz
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'PyMuPdf>=1.18.17'])
    import fitz

from pyxllib.prog.newbie import round_int, RunOnlyOnce, decode_bitflags
from pyxllib.prog.pupil import DictTool
from pyxllib.algo.newbie import round_unit
from pyxllib.algo.pupil import get_number_width
from pyxllib.file.specialist import File, Dir, writefile
from pyxllib.debug.pupil import dprint
from pyxllib.debug.specialist import browser
from pyxllib.cv.expert import xlcv, xlpil
from pyxllib.data.labelme import LabelmeDict


class FitzDoc:
    """ 原名叫FitzPdf，但不一定是处理pdf，也可能是其他文档，所以改名 FitzDoc
    """

    def __init__(self, file):
        self.src_file = File(file)
        self.doc = fitz.open(str(file))

    def write_images(self, dst_dir=None, file_fmt='{filestem}_{number}.png', num_width=None, *,
                     scale=1, start=1, fmt_onepage=False):
        """ 将pdf转为若干页图片

        :param dst_dir: 目标目录
            默认情况下，只有一页pdf则存储到对应的pdf目录，多页则存储到同名子目录下
            如果不想这样被智能控制，只要指定明确的dst即可
        :param file_fmt: 后缀格式，包括修改导出的图片类型，注意要用 {} 占位符表示页码编号
        :param num_width: 生成的每一页文件编号，使用的数字前导0域宽
            默认根据pdf总页数来设置对应所用域宽
            0表示不设域宽
        :param scale: 对每页图片进行缩放，一般推荐都要设成2，导出的图片才清晰
        :param start: 起始页码，一般建议从1开始比较符合常识直觉
        :param fmt_onepage: 当pdf就只有一页的时候，是否还对导出的图片编号
            默认只有一页的时候，进行优化，不增设后缀格式
        :return: 返回转换完的图片名称清单

        注：如果要导出单张图，可以用 FitzPdfPage.get_cv_image
        """
        # 1 基本参数计算
        srcfile, doc = self.src_file, self.doc
        filestem, n_page = srcfile.stem, doc.page_count

        # 自动推导目标目录
        if dst_dir is None:
            dst_dir = Dir(srcfile.stem, srcfile.parent) if n_page > 1 else Dir(srcfile.parent)
        Dir(dst_dir).ensure_dir()

        # 域宽
        num_width = num_width or get_number_width(n_page)  # 根据总页数计算需要的对齐域宽

        # 2 导出图片
        if fmt_onepage or n_page != 1:  # 多页的处理规则
            res = []
            for i in range(n_page):
                im = self.load_page(i).get_cv_image(scale)
                number = ('{:0' + str(num_width) + 'd}').format(i + start)  # 前面的括号不要删，这样才是完整的一个字符串来使用format
                f = xlcv.write(im, File(file_fmt.format(filestem=filestem, number=number), dst_dir))
                res.append(f)
            return res
        else:
            im = self.load_page(0).get_cv_image(scale)
            return [xlcv.write(im, File(srcfile.stem + os.path.splitext(file_fmt)[1], dst_dir))]

    def write_labelmes(self, imfiles, opt='dict', *, views=(0, 0, 1, 0), scale=1, indent=None):
        """ 生成图片对应的标注，常跟to_images配合使用 """
        for i, imfile in enumerate(imfiles):
            page = self.load_page(i)
            lmdict = LabelmeDict.gen_data(imfile)
            lmdict['shapes'] = page.get_labelme_shapes(opt, views=views, scale=scale)
            imfile.with_suffix('.json').write(lmdict, indent=indent)

    def __getattr__(self, item):
        return getattr(self.doc, item)


class FitzPageExtend:
    """ 对fitz.fitz.Page的扩展成员方法 """

    @staticmethod
    def get_svg_image2(self, scale=1):
        # svg 是一段表述性文本
        if scale != 1:
            txt = self.get_svg_image(matrix=fitz.Matrix(scale, scale))
        else:
            txt = self.get_svg_image()
        return txt

    @staticmethod
    def _get_png_data(self, scale=1):
        # TODO 增加透明通道？
        if scale != 1:
            pix = self.get_pixmap(matrix=fitz.Matrix(scale, scale))  # 长宽放大到scale倍
        else:
            pix = self.get_pixmap()
        return pix.getPNGData()

    @staticmethod
    def get_cv_image(self, scale=1):
        return xlcv.read_from_buffer(self._get_png_data(scale), flags=1)

    @staticmethod
    def get_pil_image(self, scale=1):
        # TODO 可以优化，直接从内存数据转pil，不用这样先转cv再转pil
        return xlpil.read_from_buffer(self._get_png_data(scale), flags=1)

    @staticmethod
    def write_image(self, outfile, *, scale=1, if_exists=None):
        """ 转成为文件 """
        f = File(outfile)
        suffix = f.suffix.lower()

        if suffix == '.svg':
            content = self.get_svg_image()
            f.write(content, if_exists=if_exists)
        else:
            im = self.get_cv_image(scale)
            xlcv.write(im, if_exists=if_exists)

    @staticmethod
    def get_labelme_shapes(self, opt='dict', *, views=1, scale=1):
        """ 得到labelme版本的shapes标注信息

        :param opt: get_text的参数，默认使用无字符集标注的精简的dict
            也可以使用rawdict，带有字符集标注的数据
        :param views: 若非list或者长度不足4，会补足
            各位标记依次代表是否显示对应细粒度的标注：blocks、lines、spans、chars
            默认只显示blocks
            例如 (0, 0, 1, 0)，表示只显示spans的标注
        :param scale: 是否需要对坐标按比例放大 （pdf经常放大两倍提取图片，则这里标注也要对应放大两倍）

        【字典属性解释】
        blocks:
            number: int, 区块编号
            type: 0表示文本行，1表示图片
        lines:
            wmode: 好像都是0，不知道啥东西
            dir: [1, 0]，可能是文本方向吧
        spans:
            size: 字号
            flags: 格式标记
                1，superscript，上标
                2，italic，斜体
                4，serifed，有衬线。如果没开，对立面就是"sans"，无衬线。
                8，monospaced，等距。对立面proportional，均衡。
                16，bold，加粗
            font：字体名称（直接用字符串赋值）
            color：颜色
            ascender：？
            descender：？
            origin：所在方格右上角坐标
            text/chars: dict模式有text内容，rawdict有chars详细信息。我扩展的版本，rawdict也会有text属性。
        char:
            origin: 差不多是其所在方格的右上角坐标，同一行文本，其top位置是会对齐的
            c: 字符内容
        """
        from pyxllib.data.labelme import LabelmeDict

        # 1 参数配置
        if isinstance(views, int):
            views = [views]
        if len(views) < 4:
            views += [0] * (4 - len(views))

        shapes = []
        page_dict = self.get_text(opt)

        # 2 辅助函数
        def add_shape(name, refdict, add_keys, drop_keys=('bbox',)):
            """ 生成一个标注框 """
            msgdict = {'category_name': name}
            msgdict.update(add_keys)
            DictTool.ior(msgdict, refdict)
            DictTool.isub(msgdict, drop_keys)
            bbox = [round_int(v * scale) for v in refdict['bbox']]

            if 'size' in msgdict:
                x = round_unit(msgdict['size'], 0.5)
                msgdict['size'] = round_int(x) if (x * 10) % 10 < 1 else x  # 没有小数的时候，优先展示为11，而不是11.0
            if 'color' in msgdict:
                # 把color映射为直观的(r, g, b)
                # 这个pdf解析器获取的color，不一定精确等于原值，可能会有偏差，小一个像素
                v = msgdict['color']
                msgdict['color'] = (v // 256 // 256, (v // 256) % 256, v % 256)
            if 'origin' in msgdict:
                msgdict['origin'] = [round_int(v) for v in msgdict['origin']]

            sp = LabelmeDict.gen_shape(json.dumps(msgdict), bbox)
            shapes.append(sp)

        # 3 遍历获取标注数据
        for block in page_dict['blocks']:
            if block['type'] == 0:  # 普通的文本行
                if views[0]:
                    add_shape('text_block', block, {'n_lines': len(block['lines'])}, ['bbox', 'lines'])
                for line in block['lines']:
                    if views[1]:
                        add_shape('line', line, {'n_spans': len(line['spans'])}, ['bbox', 'spans'])
                    for span in line['spans']:
                        if 'text' not in span and 'chars' in span:
                            span['text'] = ''.join([x['c'] for x in span['chars']])
                        if views[2]:
                            add_shape('span', span, {'n_chars': len(span.get('text', ''))}, ['bbox', 'chars'])
                        if views[3] and 'chars' in span:  # 最后层算法不太一样，这样写可以加速
                            for char in span['chars']:
                                add_shape('char', char, {}, ['bbox'])
            elif block['type'] == 1:  # 应该是图片
                add_shape('image', block, {'image_filesize': len(block['image'])}, ['bbox', 'image'])
            else:
                raise ValueError

        return shapes

    @staticmethod
    def parse_flags(cls, n):
        """ 解析spans的flags参数明文含义 """
        flags = decode_bitflags(n, ('superscript', 'italic', 'serifed', 'monospaced', 'bold'))
        flags['sans'] = not flags['serifed']
        flags['proportional'] = not flags['monospaced']
        return flags


def check_names_fitzpageextend():
    exist_names = {'fitz.fitz.Page': set(dir(fitz.fitz.Page))}
    names = {x for x in dir(FitzPageExtend) if x[:2] != '__'}

    for name, k in itertools.product(names, exist_names):
        if name in exist_names[k]:
            print(f'警告！同名冲突！ {k}.{name}')

    return names


@RunOnlyOnce
def binding_fitzpage_extend():
    names = check_names_fitzpageextend()
    cls_names = {'parse_flags'}

    for name in cls_names:
        setattr(fitz.fitz.Page, name, classmethod(getattr(FitzPageExtend, name)))

    for name in (names - cls_names):
        setattr(fitz.fitz.Page, name, getattr(FitzPageExtend, name))


binding_fitzpage_extend()


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
        file = writefile(b'', 'a.pdf', if_exists='replace')
        self.doc.save(file, garbage=4)

    def rearrange_pages(self):
        """重新布局页面"""
        self.doc.select([0, 0, 1])  # 第1页展示两次后，再跟第2页
        file = writefile(b'', 'a.pdf', root=Dir.TEMP, if_exists='replace')
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


__backup = """
下面这些代码先留着不要删，以后做旧代码兼容分析，升级的时候，需要参考

这里有些功能是编校还是不断使用的，但接口等需要重新优化
"""

# def pdf2svg_oldversion(pdffile, target=None, *, trim=False):
#     """新版的，pymupdf生成的svg无法配合inkscape进行trim，所以这个旧版暂时还是要保留
#
#     :param pdffile: 一份pdf文件
#     :param target: 目标目录
#         None：
#             如果只有一页，转换到对应目录下同名svg文件
#             如果有多页，转换到同名目录下多份svg文件
#     :param trim:
#         True: 去除边缘空白
#     :return:
#
#     需要第三方工具：pdf2svg（用于文件格式转换），inkscape（用于svg编辑优化）
#         注意pdf2svg的参数不支持中文名，因为这个功能限制，搞得我这个函数实现好麻烦！
#         还要在临时文件夹建立文件，因为重名文件+多线程问题，还曾引发一个bug搞了一下午。
#         （这些软件都以绿色版形式整理在win3/imgtools里）
#
#     注意！！！ 这个版本的代码先不要删！先不要删！先不要删！包括pdf2svg.exe那个蠢货软件也先别删！
#         后续研究inkscape这个蠢货的-D参数在处理pymupdf生成的svg为什么没用的时候可以进行对比
#     """
#     import fitz
#     pages = fitz.open(pdffile).pageCount
#
#     basename = tempfile.mktemp()
#     f1 = basename + '.pdf'
#     filescopy(pdffile, f1)  # 复制到临时文件，防止中文名pdf2svg工具处理不了
#
#     if pages == 1:
#         if target is None: target = pdffile[:-3] + 'svg'
#         f2 = basename + '.svg'
#         # print(['pdf2svg.exe', f1, f2])
#         subprocess.run(['pdf2svg.exe', f1, f2])
#
#         if trim: subprocess.run(['inkscape.exe', '-f', f2, '-D', '-l', f2])
#         filescopy(f2, target)
#     else:
#         if target is None: target = pdffile[:-4] + '_svg\\'
#         executor = concurrent.futures.ThreadPoolExecutor()
#         File(basename + '/').ensure_parent()
#
#         def func(f1, f2, i):
#             subprocess.run(['pdf2svg.exe', f1, f2, str(i)])
#             if trim: subprocess.run(['inkscape.exe', '-f', f2, '-D', '-l', f2])
#             filescopy(f2, target + f'{i}.svg')
#
#         for i in range(1, pages + 1):
#             f2 = basename + f'\\{i}.svg'
#             executor.submit(func, f1, f2, i)
#         executor.shutdown()
#         filescopy(basename, target[:-1])
#         filesdel(basename + '/')
#
#     filesdel(f1)


# def pdf2imagebase(pdffile, target=None, *, ext, scale=None, if_exists=None):
#     """
#     使用python的PyMuPdf模块，不需要额外插件
#     导出的图片从1开始编号
#     TODO 要加多线程？效率影响大吗？
#
#     :param pdffile: pdf原文件
#     :type target: 相对于原文件所在目录的目标目录名，也可以写文件名，表示重命名
#         None：
#             当该pdf只有1页时，才默认把图片转换到当前目录。
#             否则默认新建一个文件夹来存储图片。（目录名默认为文件名）
#     :param scale: 缩放尺寸
#         1：原尺寸
#         1.5：放大为原来的1.5倍
#     :param ext: 导出的图片格式
#     :return: 返回生成的图片列表
#     """
#
#     import fitz
#     # 1 基本参数计算
#     pdffile = File(pdffile)
#     pdf = fitz.open(pdffile)
#     num_pages = pdf.pageCount
#
#     # 大于1页的时候，默认新建一个文件夹来存储图片
#     if target is None:
#         if num_pages > 1:
#             target = File(pdffile).stem + '/'
#         else:
#             target = File(pdffile).dirname + '/'
#
#     newfile = str(pathlib.Path(pdffile.parent / target))
#     if newfile.endswith('.pdf'):
#         newfile = os.path.splitext(newfile)[0] + ext
#     Dir(os.path.dirname(newfile)).ensure_parent()
#
#     # 2 分析导出的图片文件名
#     files = []
#     if num_pages == 1:
#         FitzPdfPage(pdf.loadPage(0)).write_image(newfile, if_exists=if_exists)
#         files.append(newfile)
#     else:  # 有多页
#         number_width = get_number_width(num_pages)  # 根据总页数计算需要的对齐域宽
#         stem, ext = os.path.splitext(newfile)
#         for i in range(num_pages):
#             name = ('-{:0' + str(number_width) + 'd}').format(i + 1)  # 前面的括号不要删，这样才是完整的一个字符串来使用format
#             file = stem + name + ext
#             files.append(file)
#             FitzPdfPage(pdf.loadPage(i)).write_image(file, if_exists=if_exists)
#     return files


# def pdf2png(pdffile, target=None, scale=None):
#     """
#     :param pdffile: pdf路径
#     :param target: 目标位置
#     :param scale: 缩放比例
#     :return: list，生成的png图片清单
#
#     # 可以不写target，默认处理：如果单张png则在同目录，多张则会建个同名目录存储
#     >> pdf2png(r'D:\slns+\immovables\immovables_data\test\X\A0001.pdf')
#
#     # 指定存放位置：
#     >> pdf2png(r'D:\slns+\immovables\immovables_data\test\X\A0001.pdf', r'D:\slns+\immovables\immovables_data\test\X')
#
#     """
#     return pdf2imagebase(pdffile, target=target, scale=scale, ext='.png')


# def pdf2jpg(pdffile, target=None, scale=None):
#     return pdf2imagebase(pdffile, target=target, scale=scale, ext='.jpg')
#


# def pdf2svg(pdffile, target=None, scale=None, trim=False):
#     """
#     :param pdffile: 见pdf2imagebase
#     :param target: 见pdf2imagebase
#     :param scale: 见pdf2imagebase
#     :param trim: 如果使用裁剪功能，会调用pdf-crop-margins第三方工具
#         https://pypi.org/project/pdfCropMargins/
#     :return:
#     """
#     if trim:  # 先对pdf文件进行裁剪再转换
#         pdf = File(pdffile)
#         newfile = File('origin.pdf', pdf.parent).to_str()
#         pdf.copy(newfile)
#         # subprocess.run(['pdf-crop-margins.exe', '-p', '0', newfile, '-o', pdffile], stderr=subprocess.PIPE) # 本少： 会裁过头！
#         # 本少： 对于上下边处的 [] 分数等，会裁过头，先按百分比 -p 0 不留边，再按绝对点数收缩/扩张 -a -1  负数为扩张，单位为bp
#         # 本少被自己坑了，RamDisk 与 pdf-crop-margins.exe 配合，只能取 SCSI 硬盘，如果 Direct-IO 就不行，还不报错，还以为是泽少写的代码连报错都不会
#         subprocess.run(['pdf-crop-margins.exe', '-p', '0', '-a', '-1', newfile, '-o', pdffile],
#                        stderr=subprocess.PIPE)
#     # TODO 有时丢图
#     pdf2imagebase(pdffile, target=target, scale=scale, ext='.svg')


# def pdfs2pngs(src, scale=None, pinterval=None):
#     """ 将目录下所有pdf转png
#
#     :param src: 原pdf数据路径
#     :param scale: 转图片时缩放比例，例如2表示长宽放大至2被
#     :param pinterval: 每隔多少个pdf输出处理进度
#         默认None，不输出
#     """
#     from functools import partial
#     func = partial(pdf2png, scale=scale)
#     Dir(src).select('**/*.pdf').procpaths(func, pinterval=pinterval)
