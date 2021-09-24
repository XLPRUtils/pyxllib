#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/31 09:56

import json
import os
import re
import subprocess

try:
    import win32com
except ModuleNotFoundError:
    subprocess.run('pip install pypiwin32')

import pythoncom
from win32com.client import constants
import win32com.client as win32

from pyxllib.prog.newbie import RunOnlyOnce
from pyxllib.prog.pupil import DictTool, EnchantBase
from pyxllib.text.pupil import strwidth
from pyxllib.debug.specialist import File, Dir, get_etag, browser


def __docx():
    """ python-docx 相关封装
    """
    pass


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
        import docx
        # pip install python-docx

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
        from pyxllib.file.pdflib import FitzDoc
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
        from pyxllib.cv.expert import xlpil
        xlpil.enchant()

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
        from pyxllib.data.labelme import LabelmeDict

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


def __win32_word():
    """ 使用win32com调用word

    vba的文档：示例代码更多，vba语法也更熟悉，但显示的功能更不全
        https://docs.microsoft.com/en-us/office/vba/api/word.saveas2
    .net的文档：功能显示更全，应该是所有COM接口都有但示例代码更少、更不熟系
        https://docs.microsoft.com/en-us/dotnet/api/microsoft.office.interop.word.documentclass.saveas2?view=word-pia
    """
    pass


class EnchantWin32WordApplication(EnchantBase):
    @classmethod
    @RunOnlyOnce.decorator(distinct_args=False)
    def enchant(cls, app, recursion_enchant=False):
        """
        :param app: win32的类是临时生成的，需要给一个参考对象，才方便type(word)算出类型
        :param recursion_enchant: 是否递归，对目前有的各种子类扩展功能都绑定上
            默认关闭，如果影响到性能，可以关闭，后面运行中需要时手动设定enchant
            开启，能方便业务层开发

            之前有想过可以生成doc里的时候再enchant这些对象，但如果是批量处理脚本，每次建立doc都判断我觉得也麻烦
            长痛不如短痛，建立app的时候就把所有对象enchant更方便
        """
        # app
        _cls = type(app)
        names = cls.check_enchant_names([_cls])
        exclude_names = {'get_app'}
        cls._enchant(_cls, names - exclude_names, mode='staticmethod2objectmethod')

        if recursion_enchant:
            # 建一个临时文件，把各种需要绑定的对象都生成绑定一遍
            # 确保初始化稍微慢点，但后面就方便了
            doc = app.Documents.Add()
            EnchantWin32WordDocument.enchant(doc)

            rng = doc.Range()  # 全空的文档，有区间[0,1)
            EnchantWin32WordRange.enchant(rng)

            doc.Hyperlinks.Add(rng, 'url')  # 因为全空，这里会自动生成对应的明文url
            EnchantWin32WordHyperlink.enchant(doc.Hyperlinks(1))

            # 处理完关闭文档，不用保存
            doc.Close(False)

    @classmethod
    def get_app(cls, mode='default', *, visible=None, display_alerts=0, recursion_enchant=True):
        """
        Args:
            mode: 目前除了默认default，只有new，强制新建一个app
                210912周日10:07，可以先不管这个参数，使用默认模式就好了，现在所有app能统一为一个了
                    之前是好像app会有多个版本，导致Documents管理不集中，但同名文件又会夸app冲突很麻烦
                    所以才搞的这个复杂机制。如果该问题暂未出现，那么mode其实没用，多此一举。
            visible: 是否可见
            display_alerts: 是否关闭警告
            recursion_enchant: 是否递归执行enchant

        """
        app = None
        if mode == 'default':
            try:
                app = win32.GetActiveObject('Word.Application')
            except pythoncom.com_error:
                app = None
        if app is None:
            # 必须用gencache方法，才能获得 from win32com.client import constants 的常量
            app = win32.gencache.EnsureDispatch('Word.Application')
            # print('gencache')
        cls.enchant(app, recursion_enchant=recursion_enchant)

        if visible is not None:
            app.Visible = visible
        if display_alerts is not None:
            app.DisplayAlerts = display_alerts  # 不警告

        return app

    @staticmethod
    def check_close(app, outfile):
        """ 检查是否有指定名称的文件被打开，将其关闭，避免new_doc等操作出现问题
        """
        outfile = File(outfile)
        for x in app.Documents:
            # 有可能文件来自onedrive，这里用safe_init更合理
            if File.safe_init(x.Name, x.Path) == outfile:
                x.Close()

    @staticmethod
    def open_doc(app, file_name):
        """ 打开已有的文件
        """
        doc = app.Documents.Open(str(file_name))
        return doc

    @staticmethod
    def new_doc(app, file=None):
        """ 创建一个新的文件
        Args:
            file: 文件路径
                空：新建一个doc，到时候保存会默认到临时文件夹
                不存在的文件名：新建对应的空文件
                已存在的文件名：重置、覆盖一个新的空文件

        使用该函数，会自动执行EnchantWin32WordDocument扩展。
        """
        if file is None:
            file = File(..., Dir.TEMP, suffix='.docx')
        else:
            file = File(file)

        doc = app.Documents.Add()  # 创建新的word文档
        doc.save(file)
        return doc

    @staticmethod
    def wd(app, name, part=None):
        """ 输入字符串名称，获得对应的常量值

        :param name: 必须省略前缀wd。这个函数叫wd，就是帮忙省略掉前缀wd的意思
        :param part: 特定组别的枚举值，可以输入特殊的name模式来取值
        """
        if part is None:
            return getattr(constants, 'wd' + name)
        else:
            raise ValueError


class EnchantWin32WordDocument(EnchantBase):
    @classmethod
    @RunOnlyOnce.decorator(distinct_args=False)
    def enchant(cls, doc):
        _cls = type(doc)
        names = cls.check_enchant_names([_cls])
        propertys = {'n_page', 'content'}
        cls._enchant(_cls, propertys, mode='staticmethod2property')
        cls._enchant(_cls, names - propertys, mode='staticmethod2objectmethod')

    @staticmethod
    def save(doc, file_name=None, fmt=None, retain=False, **kwargs):
        """ 我自己简化的保存接口

        :param file_name: 保存到指定路径，如果带有后缀
        :param fmt: 毕竟是底层的com接口，不能做的太智能吧。连通过文件名后缀自动选择格式的功能都没有，要手动指定。
            为了方便，对这些进行智能自动处理，得到一个合理的save接口。
        :param retain: SaveAs2的机制：如果目标格式仍是word支持的，则doc会切换到目标文件。否则doc保留原文件对象。
            这里retain若打开，则会自动做切换，保留原文件对象
        :return: 跟retain有关，可能会"重置", outfile
            默认返回 outfile
            开启retain时，返回 outfile, doc
        """

        # 1 辅助函数
        def save_format(fmt):
            """ 枚举值映射，word保存类型的枚举

            >> _('.html')
            8
            >> _('Pdf')
            17
            >> _('wdFormatFilteredHTML')
            10
            """
            # 复杂格式可能无法完美支持所有功能。比如复杂的pdf无法使用SaveAs2实现，要用ExportAsFixedFormat。
            common = {'doc': 'FormatDocument97',
                      'html': 'FormatHTML',
                      'txt': 'FormatText',
                      'docx': 'FormatDocumentDefault',
                      'pdf': 'FormatPDF'}
            name = common.get(fmt.lower().lstrip('.'), fmt)
            return getattr(constants, 'wd' + name)

        # 2 确认要存储的文件格式
        if isinstance(fmt, str):
            fmt = fmt.lower().lstrip('.')
        elif file_name is not None:
            fmt = File(file_name).suffix[1:].lower()
        elif doc.Path:
            fmt = os.path.splitext(doc.Name)[1][1:].lower()
        else:
            fmt = 'docx'

        # 3 保存一份原始的文件路径
        origin_file = File(doc.Name, doc.Path) if doc.Path else None

        # 4 如果有指定保存文件路径
        if file_name is not None:
            outfile = File(file_name)
            if outfile.suffix[1:].lower() != fmt:
                # 已有文件名，但这里指定的fmt不同于原文件，则认为是要另存为一个同名的不同格式文件
                outfile = File(outfile.stem, outfile.parent, suffix=fmt)
            doc.SaveAs2(str(outfile), save_format(fmt), **kwargs)
        # 5 如果没指定保存文件路径
        else:
            if doc.Path:
                outfile = File(doc.Name, doc.Path, suffix='.' + fmt)
                doc.SaveAs2(str(outfile), save_format(outfile.suffix), **kwargs)
            else:
                etag = get_etag(doc.content)
                outfile = File(etag, Dir.TEMP, suffix=fmt)
                doc.SaveAs2(str(outfile), save_format(fmt), **kwargs)

        # 6 是否恢复原doc
        cur_file = File(doc.Name, doc.Path)  # 当前文件不一定是目标文件f，如果是pdf等格式也不会切换过去
        if retain and origin_file and origin_file != cur_file:
            app = doc.Application
            doc.Close()
            doc = app.open_doc(origin_file)

        # 7 返回值
        if retain:
            return outfile, doc
        else:
            return outfile

    @staticmethod
    def content(doc):
        return doc.Range().content

    @staticmethod
    def n_page(doc):
        return doc.ActiveWindow.Panes(1).Pages.Count

    @staticmethod
    def browser(doc, file_name=None, fmt='html', retain=False):
        """ 这个函数可能会导致原doc指向对象被销毁，建议要不追返回值doc继续使用
        """
        res = doc.save(file_name, fmt, retain=retain)

        if retain:
            outfile, doc = res
        else:
            outfile = res

        browser(outfile)
        return doc

    @staticmethod
    def add_section_size(doc, factor=1):
        """ 增加每节长度的标记
        一般在这里计算比在html计算方便
        """
        from humanfriendly import format_size

        n = doc.Paragraphs.Count
        style_names, text_lens = [], []
        for p in doc.Paragraphs:
            style_names.append(str(p.Style))
            text_lens.append(strwidth(p.Range.Text))

        for i, p in enumerate(doc.Paragraphs):
            name = style_names[i]
            if name.startswith('标题'):
                cumulate_size = 0
                for j in range(i + 1, n):
                    if style_names[j] != name:
                        cumulate_size += text_lens[j - 1]
                    else:
                        break
                if cumulate_size:
                    size = format_size(cumulate_size * factor).replace(' ', '').replace('bytes', 'B')
                    r = p.Range
                    doc.Range(r.Start, r.End - 1).InsertAfter(f'，{size}')


class EnchantWin32WordRange(EnchantBase):
    """ range是以下标0开始，左闭右开的区间

    当一个区间出现混合属性，比如有的有加粗，有的没加粗时，标记值为 app.wd('Undefined') 9999999
    vba的True是值-1，False是值0
    """

    @classmethod
    @RunOnlyOnce.decorator(distinct_args=False)
    def enchant(cls, rng):
        _cls = type(rng)
        names = cls.check_enchant_names([_cls])
        propertys = {'content'}
        cls._enchant(_cls, propertys, mode='staticmethod2property')
        cls._enchant(_cls, names - propertys, mode='staticmethod2objectmethod')

    @staticmethod
    def set_hyperlink(rng, url):
        """ 给当前rng添加超链接
        """
        doc = rng.Parent
        doc.Hyperlinks.Add(rng, url)

    @staticmethod
    def content(rng):
        # 有特殊换行，ch.Text可能会得到 '\r\x07'，为了位置对应，只记录一个字符
        return ''.join([ch.Text[0] for ch in rng.Characters])

    @staticmethod
    def char_range(rng, start=0, end=None):
        """ 定位rng中的子range对象，这里是以可见字符Characters计数的

        :param start: 下标类似切片的规则
        :param end: 见start描述，允许越界，允许负数
            默认不输入表示匹配到末尾
        """
        n = rng.Characters.Count
        if end is None or end > n:
            end = n
        elif end < 0:
            end = n + end
        start_idx, end_idx = rng.Characters(start + 1).Start, rng.Characters(end).End
        return rng.Document.Range(start_idx, end_idx)


class EnchantWin32WordHyperlink(EnchantBase):
    @classmethod
    @RunOnlyOnce.decorator(distinct_args=False)
    def enchant(cls, link):
        _cls = type(link)
        names = cls.check_enchant_names([_cls])
        propertys = {'netloc', 'name'}
        cls._enchant(_cls, propertys, mode='staticmethod2property')
        cls._enchant(_cls, names - propertys, mode='staticmethod2objectmethod')

    @staticmethod
    def netloc(link):
        from urllib.parse import urlparse
        linkp = urlparse(link.Name)  # 链接格式解析
        # netloc = linkp.netloc or Path(linkp.path).name
        netloc = linkp.netloc or linkp.scheme  # 可能是本地文件，此时记录其所在磁盘
        return netloc

    @staticmethod
    def name(link):
        """ 这个是转成明文的完整链接，如果要编码过的，可以取link.Name """
        from urllib.parse import unquote
        return unquote(link.Name)


def rebuild_document_by_word(fmt='html', translate=False, navigation=False, visible=False, quit=None):
    """ 将剪切板的内容粘贴到word重新排版，再转成fmt格式的文档，用浏览器打开

    这个功能只能在windows平台使用，并且必须要安装有Word软件。

    一般用于英文网站，生成双语阅读的模板，再调用谷歌翻译。
    生成的文档如果有需要，一般是手动复制整理到另一个docx文件中。

    Args:
        fmt: 输出文件类型
            常见的可以简写：html、pdf、txt
            其他特殊需求可以用word原变量名：wdFormatDocument
        visible: 是否展示运行过程，如果不展示，默认最后会close文档
        quit: 运行完是否退出应用
        translate: html专用业务功能，表示是否对p拷贝一份notranslate的对象，用于谷歌翻译双语对照
        navigation: 是否增加导航栏
            注意，使用导航栏后，页面就无法直接使用谷歌翻译了
            但可以自己进入_content文件，使用谷歌翻译处理，自覆盖保存
            然后再回到_index文件，刷新即可
    """
    import pyperclip
    from pyxllib.text.xmllib import BeautifulSoup, html_bitran_template, MakeHtmlNavigation

    # 1 保存的临时文件名采用etag
    f = File(get_etag(pyperclip.paste()), Dir.TEMP, suffix=fmt)
    app = EnchantWin32WordApplication.get_app(visible=visible)
    app.check_close(f)
    doc = app.new_doc(f)
    app.Selection.Paste()

    # 2 如果需要，也可以在这个阶段，插入word自动化的操作，而不是后续在html层面操作
    # 统计每节内容长度，每个字母1B，每个汉字2B
    doc.add_section_size()
    file = doc.save(f, fmt)
    doc.Close()

    # 3 html格式扩展功能
    if fmt == 'html':
        # 3.1 默认扩展功能
        bs = BeautifulSoup(file.read(encoding='gbk'), 'lxml')
        bs.head_add_number()  # 给标题加上编号
        # bs.head_add_size()  # 显示每节内容长短
        content = bs.xltext()

        # TODO 识别微信、pydoc，然后做一些自动化清理？
        # TODO 过度缩进问题？

        # 3.2 双语对照阅读
        if translate:
            # word生成的html固定是gbk编码
            content = html_bitran_template(content)

        # 原文是gbk，但到谷歌默认是utf8，所以改一改
        # 这样改后问题是word可能又反而有问题了，不过word本来只是跳板，并不是要用word编辑html
        file.write(content, encoding='utf8')

        # 3.3 导航栏功能
        # 作为临时使用可以开，如果要复制到word，并没有必要
        if navigation:
            file = MakeHtmlNavigation.from_file(file, encoding='utf8', number=False, text_catalogue=False)

    if quit:
        app.Quit()

    return file