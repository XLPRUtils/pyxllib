#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/09/07 10:21

import os

import pywintypes
import win32com.client as win32
from win32com.client import constants
import pythoncom

from pyxllib.debug.specialist import TicToc, File, Dir, get_etag, browser
from pyxllib.prog.newbie import RunOnlyOnce
from pyxllib.prog.pupil import EnchantBase


def get_win32_app(name, visible=False):
    """ 启动可支持pywin32自动化处理的应用

    Args:
        str name: 应用名称，不区分大小写，比如word, excel, powerpoint, onenote
            不带'.'的情况下，会自动添加'.Application'的后缀
        visible: 应用是否可见

    Returns: app

    """
    # 1 name
    name = name.lower()
    if '.' not in name:
        name += '.application'

    # 2 app
    # 这里可能还有些问题，不同的应用，机制不太一样，后面再细化完善吧
    try:
        app = win32.GetActiveObject(f'{name}')  # 不能关联到普通方式打开的应用。但代码打开的应用都能找得到。
    except pythoncom.com_error:
        app = win32.gencache.EnsureDispatch(f'{name}')
        # 还有种常见的初始化方法，是 win32com.client.Dispatch和win32com.client.dynamic.Dispatch
        # from win32com.client.dynamic import Disypatch

    if visible is not None:
        app.Visible = visible

    return app


def __word():
    """
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
    def get_app(cls, mode='default', *, visible=None, display_alerts=0, recursion_enchant=False):
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
        cls.enchant(app, recursion_enchant=True)

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
            if File(x.Name, x.Path) == outfile:
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
