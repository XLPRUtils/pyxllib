#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 17:22

""" 百分注格式处理 """

import re

import bs4
from bs4 import BeautifulSoup

from pyxllib.text.pupil import grp_bracket
from pyxllib.debug.pupil import dprint


def gettag_name(tagstr):
    """
    >>> gettag_name('%<topic type=danxuan description=单选题>')
    'topic'
    >>> gettag_name('</topic>')
    'topic'
    """
    m = re.search(r'</?([a-zA-Z_]+)', tagstr)
    if m:
        return m.group(1)
    else:
        return None


def settag_name(tagstr, *, new_name=None, switch=None):
    """设置标签名称，或者将标签类型设为close类型

    >>> settag_name('%<topic type=danxuan description=单选题>', new_name='mdzz')
    '%<mdzz type=danxuan description=单选题>'
    >>> settag_name('<topic type=danxuan description=单选题>', switch=False)
    '</topic>'
    """
    if new_name:  # 是否设置新名称
        tagstr = re.sub(r'(</?)([a-zA-Z_]+)', lambda m: m.group(1) + new_name, tagstr)

    if switch is not None:  # 是否设置标签开关
        if switch:  # 将标签改为开
            tagstr = tagstr.replace('</', '<')
        else:  # 将标签改为关
            name = gettag_name(tagstr)
            res = f'</{name}>'  # 会删除所有attr属性
            tagstr = '%' + res if '%<' in tagstr else res

    return tagstr


def gettag_attr(tagstr, attrname):
    r"""tagstr是一个标签字符串，attrname是要索引的名字
    返回属性值，如果不存在该属性则返回None

    >>> gettag_attr('%<topic type=danxuan description=单选题> 123\n<a b=c></a>', 'type')
    'danxuan'
    >>> gettag_attr('%<topic type="dan xu an" description=单选题>', 'type')
    'dan xu an'
    >>> gettag_attr("%<topic type='dan xu an' description=单选题>", 'type')
    'dan xu an'
    >>> gettag_attr('%<topic type=dan xu an description=单选题>', 'description')
    '单选题'
    >>> gettag_attr('%<topic type=dan xu an description=单选题>', 'type')
    'dan'
    >>> gettag_attr('%<topic type=danxuan description=单选题 >', 'description')
    '单选题'
    >>> gettag_attr('%<topic type=danxuan description=单选题 >', 'description123') is None
    True
    """
    soup = BeautifulSoup(tagstr, 'lxml')
    try:
        for tag in soup.p.contents:
            if isinstance(tag, bs4.Tag):
                return tag.get(attrname, None)
    except AttributeError:
        dprint(tagstr)
    return None


def settag_attr(tagstr, attrname, target_value):
    r"""tagstr是一个标签字符串，attrname是要索引的名字
    重设该属性的值，设置成功则返回新的tagstr；否则返回原始值

    close类型不能用这个命令，用了的话不进行任何处理，直接返回

    >>> settag_attr('%<topic type=danxuan> 123\n<a></a>', 'type', 'tiankong')
    '%<topic type="tiankong"> 123\n<a></a>'
    >>> settag_attr('%<topic>', 'type', 'tiankong')
    '%<topic type="tiankong">'
    >>> settag_attr('</topic>', 'type', 'tiankong')
    '</topic>'
    >>> settag_attr('<seq value="1">', 'value', '练习1.2')
    '<seq value="练习1.2">'
    >>> settag_attr('<seq type=123 value=1>', 'type', '')  # 删除attr操作
    '<seq value=1>'
    >>> settag_attr('<seq type=123 value=1>', 'value', '')  # 删除attr操作
    '<seq type=123>'
    >>> settag_attr('<seq type=123 value=1>', 'haha', '')  # 删除attr操作
    '<seq type=123 value=1>'
    """
    # 如果是close类型是不处理的
    if tagstr.startswith('</'): return tagstr

    # 预处理targetValue的值，删除空白
    target_value = re.sub(r'\s', '', target_value)
    r = re.compile(r'(<|\s)(' + attrname + r'=)(.+?)(\s+\w+=|\s*>)')
    gs = r.search(tagstr)
    if target_value:
        if not gs:  # 如果未找到则添加attr与value
            n = tagstr.find('>')
            return tagstr[:n] + ' ' + attrname + '="' + target_value + '"' + tagstr[n:]
        else:  # 如果找到则更改value
            # TODO: 目前的替换值是直接放到正则式里了，这样会有很大的风险，后续看看能不能优化这个处理算法
            return r.sub(r'\1\g<2>"' + target_value + r'"\4', tagstr)
    else:
        if gs:
            return r.sub(r'\4', tagstr)
        else:
            return tagstr


def brieftexstr(s):
    """对比两段tex文本
    """
    # 1 删除百分注
    s = re.sub(r'%' + grp_bracket(2, '<', '>'), r'', s)
    # 2 删除所有空白字符
    # debuglib.dprint(debuglib.typename(s))
    s = re.sub(r'\s+', '', s)
    # 3 转小写字符
    s = s.casefold()
    return s


# 默认不建议开，编校如果用的多，可以在那边定义
# 定义常用的几种格式，并且只匹配抓取花括号里面的值，不要花括号本身
# SQUARE3 = r'\\[(' + grp_bracket(3, '[')[3:-3] + r')\\]'
# BRACE1 = '{(' + grp_bracket(1)[1:-1] + ')}'
# BRACE2 = '{(' + grp_bracket(2)[1:-1] + ')}'
# BRACE3 = '{(' + grp_bracket(3)[1:-1] + ')}'
# BRACE4 = '{(' + grp_bracket(4)[1:-1] + ')}'
# BRACE5 = '{(' + grp_bracket(5)[1:-1] + ')}'
"""使用示例
>> m = re.search(r'\\multicolumn' + BRACE3*3, r'\multicolumn{2}{|c|}{$2^{12}$个数}')
>> m.groups()
('2', '|c|', '$2^{12}$个数')
"""


def grp_topic(*, type_value=None):
    """定位topic

    :param type_value: 设置题目类型（TODO: 功能尚未开发）
    """
    s = r'%<topic.*?%</topic>'  # 注意外部使用的re要开flags=re.DOTALL
    return s


def grp_figure(cnt_groups=0, parpic=False):
    """生成跟图片匹配相关的表达式

    D:\2017LaTeX\D招培试卷\高中地理，用过  \captionfig{3-3.eps}{图~3}
    奕本从2018秋季教材开始使用多种图片格式

    191224周二18:20 更新：匹配到的图片名不带花括号
    """
    ibrace3 = grp_bracket(3, inner=True)

    if cnt_groups == 0:  # 不分组
        s = r'\\(?:includegraphics|figt|figc|figr|fig).*?' + grp_bracket(3)  # 注意第1组fig要放最后面
    elif cnt_groups == 1:  # 只分1组，那么只对图片括号内的内容分组
        s = r'\\(?:includegraphics|figt|figc|figr|fig).*?' + ibrace3
    elif cnt_groups == 2:  # 只分2组，那么只对插图命令和图片分组
        s = r'\\(includegraphics|figt|figc|figr|fig).*?' + ibrace3
    elif cnt_groups == 3:
        s = r'\\(includegraphics|figt|figc|figr|fig)(.*?)' + ibrace3
    else:
        s = None

    if s and parpic:
        s = r'{?\\parpic(?:\[.\])?{' + s + r'}*'

    return s
