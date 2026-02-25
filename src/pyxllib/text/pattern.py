#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

import re


def grp_bracket(n=1):
    r"""获得n层括号嵌套的正则匹配式

    :param n: n层嵌套，默认为1
    :return: 匹配的正则字符串

    >>> grp_bracket(1)
    '\\([^()]*\\)'
    >>> grp_bracket(2)
    '\\((?:[^()]|\\([^()]*\\))*\\)'
    >>> grp_bracket(3)
    '\\((?:[^()]|\\((?:[^()]|\\([^()]*\\))*\\))*\\)'
    """
    if n <= 0: return ''
    s = r'\([^()]*\)'  # 1层括号匹配
    for i in range(n - 1):
        s = f'\\((?:[^()]|{s})*\\)'
    return s


# 常用汉字正则
# 这个范围是根据网上资料整合的，可能不是特别准确，但大部分情况够用了
# \u4e00-\u9fa5: 基本汉字
# \u9fa6-\u9fcb: 基本汉字补充
# \u3400-\u4db5: 扩展A
# \u20000-\u2a6d6: 扩展B
# \u2a700-\u2b734: 扩展C
# \u2b740-\u2b81d: 扩展D
# \u2b820-\u2ceaf: 扩展E
# \u2f00-\u2fd5: 康熙部首
# \u2e80-\u2ef3: 部首补充
# \uf900-\ufad9: 兼容汉字
# \u2f800-\u2fa1d: 兼容扩展
# \u3000-\u303f: CJK标点符号
# \uff00-\uffef: 全角ASCII、全角标点
grp_chinese_char = (r'[\u4e00-\u9fcb\u3400-\u4db5\u20000-\u2a6d6\u2a700-\u2b734\u2b740-\u2b81d\u2b820-\u2ceaf'
                    r'\u2f00-\u2fd5\u2e80-\u2ef3\uf900-\ufad9\u2f800-\u2fa1d\u3000-\u303f\uff00-\uffef]')


def grp_topic(name=None):
    r""" 匹配题目编号的正则

    :param name:
        None，默认，匹配 '1、', '(1)', '1.' 这种编号
        str，指定前缀，例如 name='思考题'，则匹配 '思考题1、', '思考题(1)', '思考题1.'
        True，匹配任意前缀，即不限制前缀内容，但要求有数字编号

    >>> re.match(grp_topic(), '1、')
    <re.Match object; span=(0, 2), match='1、'>
    >>> re.match(grp_topic(), '1.')
    <re.Match object; span=(0, 2), match='1.'>
    >>> re.match(grp_topic(), '(1)')
    <re.Match object; span=(0, 3), match='(1)'>
    >>> re.match(grp_topic('思考题'), '思考题1、')
    <re.Match object; span=(0, 5), match='思考题1、'>
    """
    if name is None:
        p = r'(?:^|\s)\(?(\d+)(?:[、.)]|\s)'
    elif name is True:
        p = r'(?:^|\s)(?:.*?)\(?(\d+)(?:[、.)]|\s)'
    else:
        p = r'(?:^|\s)' + re.escape(name) + r'\(?(\d+)(?:[、.)]|\s)'
    return p


def grp_figure(name=None):
    r""" 匹配图片编号的正则

    :param name: 图片名称，默认为 '图'
        也可以输入 'Figure', 'Fig.' 等

    >>> re.match(grp_figure(), '图1')
    <re.Match object; span=(0, 2), match='图1'>
    >>> re.match(grp_figure(), '图 1')
    <re.Match object; span=(0, 3), match='图 1'>
    >>> re.match(grp_figure('Figure'), 'Figure 1')
    <re.Match object; span=(0, 8), match='Figure 1'>
    """
    if name is None:
        name = '图'
    return r'(?:^|\s)' + re.escape(name) + r'\s*(\d+)'


def calc_chinese_ratio(s):
    """ 计算字符串中汉字的比例 """
    if not s:
        return 0
    cnt = len(re.findall(grp_chinese_char, s))
    return cnt / len(s)
