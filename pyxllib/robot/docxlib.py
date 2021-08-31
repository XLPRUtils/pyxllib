#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/31 09:56

import itertools

import docx

from pyxllib.prog.newbie import RunOnlyOnce
from pyxllib.file.specialist import File, Dir
from pyxllib.debug.specialist import browser
from pyxllib.cv.expert import binding_pilimage_xlpil

binding_pilimage_xlpil()


class DocumentExtend:

    @staticmethod
    def browser(self, outfile_pdf=None):
        """ 转pdf，使用浏览器的查看效果

        :param outfile_pdf: 转pdf的路径，自己指定可以避免每次重新生成一个不同名的文件
        """
        import docx2pdf

        if outfile_pdf is None:
            file_pdf = File(..., Dir.TEMP, suffix='.pdf')
        else:
            file_pdf = File(outfile_pdf)

        file_docx = file_pdf.with_suffix('.docx')
        self.save(str(file_docx))

        docx2pdf.convert(str(file_docx), str(file_pdf))

        browser(file_pdf)

    @staticmethod
    def display(self):
        """ 转图片，使用jupyter环境的查看效果
        """
        import docx2pdf
        from pyxllib.cv.fitzlib import FitzDoc

        # 1 生成docx
        file_docx = File(..., Dir.TEMP, suffix='.docx')
        self.save(str(file_docx))

        # 2 转pdf
        file_pdf = file_docx.with_suffix('.pdf')
        docx2pdf.convert(str(file_docx), str(file_pdf))

        # 3 转图片，并且裁剪，加边框输出
        doc = FitzDoc(file_pdf)
        for i in range(doc.page_count):
            page = doc.load_page(i)
            print('= ' * 10 + f' {page} ' + '= ' * 10)
            img = page.get_pil_image()
            img.trim(border=5).plot_border().display()
            del page

        # 4 清除中间文件
        del doc
        file_docx.delete()
        file_pdf.delete()

    @staticmethod
    def to_labelme(self, outfile_docx, *, views=(0, 0, 1, 0), detele_temp=False, scale=1):
        """ 转labelme格式查看
        本质是把docx转成pdf，利用pdf的解析生成labelme格式的标准框查看

        :param outfile_docx: 注意写的是docx文件的路径，然后pdf、png、json都会放在同目录下
            这个不适合设成可选参数，需要显式指定一个输出目录比较好
        :param views: 详见write_labelmes的描述
        :param detele_temp: 是否删除中间文件 docx、pdf，默认不删除
        """
        import docx2pdf  # pip install docx2pdf，并且本地要安装word软件
        from pyxllib.cv.fitzlib import FitzDoc

        # 1 转 docx
        file_docx = File(outfile_docx)
        self.save(str(file_docx))

        # 2 转 pdf
        file_pdf = file_docx.with_suffix('.pdf')
        docx2pdf.convert(str(file_docx), str(file_pdf))

        # 3 转图片，及json标注
        doc = FitzDoc(file_pdf)
        files = doc.write_images(file_pdf.parent, '{filestem}_{number}.png', scale=scale)
        doc.write_labelmes(files, views=views)

        # 4 删除中间文件
        if detele_temp:
            file_docx.delete()
            file_pdf.delete()


def check_names_documentextend():
    exist_names = {'docx.document.Document': set(dir(docx.document.Document))}
    names = {x for x in dir(DocumentExtend) if x[:2] != '__'}

    for name, k in itertools.product(names, exist_names):
        if name in exist_names[k]:
            print(f'警告！同名冲突！ {k}.{name}')

    return names


@RunOnlyOnce
def binding_document_extend():
    names = check_names_documentextend()
    for name in names:
        setattr(docx.document.Document, name, getattr(DocumentExtend, name))


binding_document_extend()
