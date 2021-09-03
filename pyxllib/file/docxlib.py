#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/31 09:56

import json
import re
import subprocess

import docx

from pyxllib.prog.pupil import DictTool
from pyxllib.file.specialist import File, Dir, get_etag
from pyxllib.debug.specialist import browser
from pyxllib.cv.expert import binding_pilimage_xlpil
from pyxllib.data.labelme import LabelmeDict

binding_pilimage_xlpil()


class Document:
    """ 这个库写英文文档还不错。但不能做中文，字体会错乱。
    """

    def __init__(self, file_docx=None):
        """
        Args:
            file_docx:
                已有的word文件路径：打开
                还没创建的word文件路径：在个别功能需要的时候，会自动创建
                None：在临时文件夹生成一个默认的word文件
        """
        if file_docx is None:
            file_docx = File(..., Dir.TEMP, suffix='.docx')
        else:
            self.file_docx = File(file_docx)
        if self.file_docx:
            self.doc = docx.Document(str(file_docx))
        else:
            self.doc = docx.Document()

    def write(self):
        Dir(self.file_docx.parent).ensure_dir()
        self.doc.save(str(self.file_docx))

    def write_pdf(self):
        try:
            import docx2pdf  # pip install docx2pdf，并且本地要安装word软件
        except ModuleNotFoundError:
            subprocess.run(['pip3', 'install', 'docx2pdf'])
            import docx2pdf

        self.write()
        file_pdf = self.file_docx.with_suffix('.pdf')
        docx2pdf.convert(str(self.file_docx), str(file_pdf))

        return file_pdf

    def to_fitzdoc(self):
        """ 获得 fitz的pdf文档对象
        :return: FitzDoc对象
        """
        from pyxllib.file.fitzlib import FitzDoc
        file_pdf = self.write_pdf()
        doc = FitzDoc(file_pdf)
        return doc

    def write_images(self, file_fmt='{filestem}_{number}.png', *args, scale=1, **kwargs):
        doc = self.to_fitzdoc()
        files = doc.write_images(doc.src_file.parent, file_fmt, *args, scale=scale, **kwargs)
        return files

    def browser(self):
        """ 转pdf，使用浏览器的查看效果
        """
        file_pdf = self.write_pdf()
        browser(file_pdf)

    def display(self):
        """ 转图片，使用jupyter环境的查看效果
        """
        # 转图片，并且裁剪，加边框输出
        doc = self.to_fitzdoc()
        for i in range(doc.page_count):
            page = doc.load_page(i)
            print('= ' * 10 + f' {page} ' + '= ' * 10)
            img = page.get_pil_image()
            img.trim(border=5).plot_border().display()
            del page

    def write_labelmes(self, dst_dir=None, file_fmt='{filestem}_{number}.png', *, views=(0, 0, 1, 0), scale=1,
                       advance=False, indent=None):
        """ 转labelme格式查看
        本质是把docx转成pdf，利用pdf的解析生成labelme格式的标准框查看

        :param file_docx: 注意写的是docx文件的路径，然后pdf、png、json都会放在同目录下
            这个不适合设成可选参数，需要显式指定一个输出目录比较好
        :param views: 详见write_labelmes的描述
            各位依次代表是否显示对应细粒度的标注：blocks、lines、spans、chars
        :param bool|dict advance: 是否开启“高级”功能，开启后能获得下划线等属性，但速度会慢很多
            源生的fitz pdf解析是处理不了下划线的，开启高级功能后，有办法通过特殊手段实现下划线的解析
            默认会修正目前已知的下划线、颜色偏差问题
            dict类型：以后功能多了，可能要支持自定义搭配，比如只复原unberline，但不管颜色偏差
        """
        # 1 转成图片，及json标注
        doc = self.to_fitzdoc()
        imfiles = doc.write_images(dst_dir, file_fmt, scale=scale)

        # 2 高级功能
        def is_color(x):
            return x and sum(x)

        def write_labelmes_advance():
            m = 50  # 匹配run时，上文关联的文字长度，越长越严格

            # 1 将带有下划线的run对象，使用特殊的hash规则存储起来
            content = []  # 使用已遍历到的文本内容作为hash值
            elements = {}
            for p in self.paragraphs:
                for r in p.runs:
                    # 要去掉空格，不然可能对不上。试过strip不行。分段会不太一样，还是把所有空格删了靠谱。
                    content.append(re.sub(r'\s+', '', r.text))
                    if r.underline or is_color(r.font.color.rgb):  # 对有下划线、颜色的对象进行存储
                        # print(r.text + ',', r.underline, r.font.color.rgb, ''.join(content))
                        etag = get_etag(''.join(content)[-m:])  # 全部字符都判断的话太严格了，所以减小区间~
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
                        content += re.sub(r'\s+', '', attrs['text'])
                        etag = get_etag(content[-m:])
                        # print(content)
                        if etag in elements:
                            # print(content)
                            r = elements[etag]  # 对应的原run对象
                            attrs = DictTool.json_loads(sp['label'])
                            x = r.underline
                            if x:
                                attrs['underline'] = int(x)
                            x = r.font.color.rgb
                            if is_color(x):
                                attrs['color'] = list(x)
                            sp['label'] = json.dumps(attrs)
                file.with_suffix('.json').write(lmdict, indent=indent)

        # 3 获得json
        if advance:
            write_labelmes_advance()
        else:
            doc.write_labelmes(imfiles, views=views, scale=scale)

    def __getattr__(self, item):
        # 属性：
        # core_properties
        # element
        # inline_shapes
        # paragraphs
        # part
        # sections
        # settings
        # styles
        # tables
        # 方法：
        # add_heading
        # add_page_break
        # add_paragraph
        # add_picture
        # add_section
        # add_table
        # save
        return getattr(self.doc, item)
