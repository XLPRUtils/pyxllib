#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/20 11:46

""" 一些特殊的专用业务功能

因为是业务方面的，所以函数名可能多用中文~~
"""

import re

from pyxllib.debug.specialist import browser
from pyxllib.file.specialist import File, Dir, get_etag, BeautifulSoup


def __check():
    """
    检查异常系列的功能
    """


def __refine():
    """
    文本优化的功能
    """


def py_remove_interaction_chars(s):
    """ 去掉复制的一段代码中，前导的“>>>”标记 """
    # 这个算法可能还不够严谨，实际应用中再逐步写鲁棒
    # ">>> "、"... "
    lines = [line[4:] for line in s.splitlines()]
    return '\n'.join(lines)


def pycode_sort__import(s):
    from pyxllib.text.nestenv import PyNestEnv

    def cmp(line):
        """ 将任意一句import映射为一个可比较的list对象

        :return: 2个数值
            1、模块优先级
            2、import在前，from在后
        """
        name = re.search(r'(?:import|from)\s+(\S+)', line).group(1)
        for i, x in enumerate('stdlib prog algo text file debug cv data database gui ai robot tool ex'.split()):
            name = name.replace('pyxllib.' + x, f'{i:02}')
        for i, x in enumerate('pyxllib pyxlpr xlproject'.split()):
            name = name.replace(x, f'~{i:02}')
        for i, x in enumerate('newbie pupil specialist expert'.split()):
            name = name.replace('.' + x, f'{i:02}')

        # 忽略大小写
        return [name.lower(), line.startswith('import')]

    def sort_part(m):
        parts = PyNestEnv(m.group()).imports().strings()
        parts = [p.rstrip() + '\n' for p in parts]
        parts.sort(key=cmp)
        return ''.join(parts)

    res = PyNestEnv(s).imports().sub(sort_part, adjacent=True)  # 需要邻接，分块处理
    return res


def translate_html(htmlcontent):
    """ 将word导出的html文件，转成方便谷歌翻译操作，进行双语对照的格式 """
    from bs4 import BeautifulSoup, NavigableString
    from pyxllib.text.nestenv import NestEnv

    # 1 区间定位分组
    ne = NestEnv(htmlcontent)
    ne2 = ne.xmltag('p')
    for name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre'):
        ne2 += ne.xmltag(name)

    # 以下是针对python document复制到word的情况，不一定具有广泛泛用性
    # 目的是让代码块按块复制，而不是按行复制
    ne2 += ne.search2("<div style=['\"]mso-element:para-border-div;border:solid #AACC99", '</div>')

    # 2 每个区间的处理规则
    def func(s):
        """ 找出p、h后，具体要执行的操作 """
        s1, s2 = s, s  # 分前后两波文本s1，s2

        # 1 s2 只要加 notranslate
        bs = BeautifulSoup(s2, 'lxml')
        x = next(bs.body.children)
        cls_ = x.get('class', None)
        x['class'] = (cls_ + ['notranslate']) if cls_ else 'notranslate'
        if re.match(r'h\d+$', x.name):
            x.name = 'p'  # 去掉标题格式，统一为段落格式

        s2 = x.xltext()

        # 2 s1 可能要做些骚操作
        # 比如自定义翻译，这个无伤大雅的，如果搞不定，可以先注释掉，后面再说
        # bs = BeautifulSoup(s1, 'lxml')
        # x = list(bs.body.children)[0]
        # if re.match(r'h\d+$', x.name):
        #     for y in x.descendants:
        #         if isinstance(y, NavigableString):
        #             y.replace_with(re.sub(r'Conclusion', '总结', str(y)))
        #         else:
        #             for z in y.strings:
        #                 z.replace_with(re.sub(r'Conclusion', '总结', str(z)))
        #     y.replace_with(re.sub(r'^Abstract$', '摘要', str(y)))
        # s1 = x.xltext()
        if x.name == 'div':
            # 实际使用体验，想了下，代码块还是不如保留原样最方便，不用拷贝翻译
            s1 = ''

        return s1 + '\n' + s2

    res = ne2.replace(func)

    return res


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
    from pyxllib.robot.win32lib import EnchantWin32WordApplication

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
            content = translate_html(content)

        # 原文是gbk，但到谷歌默认是utf8，所以改一改
        # 这样改后问题是word可能又反而有问题了，不过word本来只是跳板，并不是要用word编辑html
        file.write(content, encoding='utf8')

        # 3.3 导航栏功能
        # 作为临时使用可以开，如果要复制到word，并没有必要
        if navigation:
            from pyxllib.text.specialist import MakeHtmlNavigation
            file = MakeHtmlNavigation.from_file(file, encoding='gbk')

    if quit:
        app.Quit()

    return file


def __extract():
    """
    信息摘要提取功能
    """
