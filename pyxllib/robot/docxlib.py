#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/31 09:56
import copy
import json

import itertools

import docx
from docx.shared import RGBColor

from pyxllib.data.labelme import LabelmeDict
from pyxllib.prog.newbie import RunOnlyOnce
from pyxllib.file.specialist import File, Dir, get_etag
from pyxllib.debug.specialist import browser
from pyxllib.cv.expert import binding_pilimage_xlpil
from pyxllib.prog.pupil import DictTool

binding_pilimage_xlpil()


class DocumentExtend:

    @staticmethod
    def save_with_pdf(self, file_docx=None):
        import docx2pdf  # pip install docx2pdf，并且本地要安装word软件

        origin_file_docx = file_docx
        if file_docx is None:
            file_docx = File(..., Dir.TEMP, suffix='.docx')
        else:
            file_docx = File(file_docx)
        self.save(str(file_docx))

        file_pdf = file_docx.with_suffix('.pdf')
        docx2pdf.convert(str(file_docx), str(file_pdf))

        # 默认还是不删文件，如果不想产生太多冗余临时文件，可以明确指定文件名
        # if origin_file_docx is None:
        #     file_docx.delete()

        return file_pdf

    @staticmethod
    def get_fitzdoc(self, file_docx=None):
        """ 获得 fitz的pdf文档对象

        :param file_docx: 需要传入中间文件docx的存储路径
            传入docx路径，会在同目录下生成同名的pdf文件
            默认会在临时文件夹生成，并且这种情况的临时文件都会删除，最后只会得到内存解析的对象
        :return: FitzDoc对象
        """
        from pyxllib.cv.fitzlib import FitzDoc

        file_pdf = self.save_with_pdf(file_docx)
        doc = FitzDoc(file_pdf)

        return doc

    @staticmethod
    def write_images(self, file_docx=None, file_fmt='{filestem}_{number}.png', *, scale=1):
        doc = self.get_fitzdoc(file_docx)
        files = doc.write_images(doc.src_file.parent, file_fmt, scale=scale)
        return files

    @staticmethod
    def browser(self, file_docx=None):
        """ 转pdf，使用浏览器的查看效果

        :param outfile_pdf: 转pdf的路径，自己指定可以避免每次重新生成一个不同名的文件
        """
        file_pdf = self.save_with_pdf(file_docx)
        browser(file_pdf)

    @staticmethod
    def display(self, file_docx=None):
        """ 转图片，使用jupyter环境的查看效果
        """
        # 转图片，并且裁剪，加边框输出
        doc = self.get_fitzdoc(file_docx)
        for i in range(doc.page_count):
            page = doc.load_page(i)
            print('= ' * 10 + f' {page} ' + '= ' * 10)
            img = page.get_pil_image()
            img.trim(border=5).plot_border().display()
            del page

    @staticmethod
    def write_labelmes(self, file_docx, file_fmt='{filestem}_{number}.png', *, views=(0, 0, 1, 0), scale=1,
                       advance=False, indent=None):
        """ 转labelme格式查看
        本质是把docx转成pdf，利用pdf的解析生成labelme格式的标准框查看

        :param file_docx: 注意写的是docx文件的路径，然后pdf、png、json都会放在同目录下
            这个不适合设成可选参数，需要显式指定一个输出目录比较好
        :param views: 详见write_labelmes的描述
        :param bool|dict advance: 是否开启“高级”功能，开启后能获得下划线等属性，但速度会慢很多
            源生的fitz pdf解析是处理不了下划线的，开启高级功能后，有办法通过特殊手段实现下划线的解析
            默认会修正目前已知的下划线、颜色偏差问题
            dict类型：以后功能多了，可能要支持自定义搭配，比如只复原unberline，但不管颜色偏差
        """
        # 1 转成图片，及json标注
        doc = self.get_fitzdoc(file_docx)
        imfiles = doc.write_images(doc.src_file.parent, file_fmt, scale=scale)

        # 2 高级功能
        def write_labelmes_advance():
            # 1 将带有下划线的run对象，使用特殊的hash规则存储起来
            content = []  # 使用已遍历到的文本内容作为hash值
            elements = {}
            for p in self.paragraphs:
                for r in p.runs:
                    # 要去掉空格，不然可能对不上。试过strip不行。分段会不太一样，还是把所有空格删了靠谱。
                    content.append(r.text.replace(' ', ''))
                    if r.underline or r.font.color.rgb:  # 对有下划线、颜色的对象进行存储
                        etag = get_etag(''.join(content))
                        elements[etag] = r

            # 2 检查json标注中为span的，其哈希规则是否有对应，则要单独设置扩展属性
            content = ''
            for i, file in enumerate(imfiles):
                page = doc.load_page(i)
                lmdict = LabelmeDict.gen_data(file)
                lmdict['shapes'] = page.get_labelme_shapes('dict', views=views, scale=scale)

                for sp in lmdict['shapes']:
                    attrs = DictTool.json_loads(sp['label'], 'label')
                    if attrs['category_name'] == 'span':
                        content += attrs['text'].replace(' ', '')
                        etag = get_etag(content)
                        if etag in elements:
                            r = elements[etag]  # 对应的原run对象
                            attrs = DictTool.json_loads(sp['label'])
                            attrs['underline'] = int(r.underline)
                            attrs['color'] = list(r.font.color.rgb)
                            sp['label'] = json.dumps(attrs)
                file.with_suffix('.json').write(lmdict, indent=indent)

        # 3 获得json
        if advance:
            write_labelmes_advance()
        else:
            doc.write_labelmes(imfiles, views=views)


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
