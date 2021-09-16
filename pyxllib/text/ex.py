#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/20 11:46

""" 一些特殊的专用业务功能

因为是业务方面的，所以函数名可能多用中文~~
"""

import re


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


def translate_html(htmlfile):
    """ 将word导出的html文件，转成方便谷歌翻译操作，进行双语对照的格式 """
    from bs4 import BeautifulSoup
    from pyxllib.text.nestenv import NestEnv

    def func(s):
        """ 找出p、h后，具体要执行的操作 """
        sp = BeautifulSoup(s, 'lxml')
        x = list(sp.body.children)[0]
        cls_ = x.get('class', None)
        x['class'] = (cls_ + ['notranslate']) if cls_ else 'notranslate'
        x.name = 'p'  # 去掉标题格式，统一为段落格式
        # &nbsp;的处理比较特别，要替换回来
        return s + '\n' + x.prettify(formatter=lambda s: s.replace(u'\xa0', '&nbsp;'))

    ne = NestEnv(htmlfile.read())
    ne2 = ne.xmltag('p')
    for name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
        ne2 += ne.xmltag(name)

    htmlfile.write(ne2.replace(func))


def __extract():
    """
    信息摘要提取功能
    """
