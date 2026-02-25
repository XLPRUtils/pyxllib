#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

import os
import re


def linux_path_fmt(p):
    p = str(p)
    p = p.replace('\\', '/')
    return p


def refinepath(s, reserve=''):
    """ 
    :param reserve: 保留的字符，例如输入'*?'，会保留这两个字符作为通配符
    """
    if not s: return s
    s = s.replace(chr(8234), '')
    chars = set(r'\/:*?"<>|') - set(reserve)
    for ch in chars:
        s = s.replace(ch, '')
    s = re.sub(r'\s+([/\\])', r'\1', s)
    s = re.sub(r'([/\\])\s+', r'\1', s)
    return s


def filename_tail(fn, tail):
    """在文件名末尾和扩展名前面加上一个tail"""
    names = os.path.splitext(fn)
    return names[0] + tail + names[1]


def change_ext(filename, ext):
    """更改文件名后缀
    返回第1个参数是新的文件名，第2个参数是这个文件是否存在

    输入的fileName可以没有扩展名，如'A/B/C/a'，仍然可以找对应的扩展名为ext的文件
    输入的ext不要含有'.'，例如正确格式是输入'tex'、'txt'
    """
    name = os.path.splitext(filename)[0]  # 'A/B/C/a.txt' --> 'A/B/C/a'
    newname = name + '.' + ext
    return newname, os.path.exists(newname)
