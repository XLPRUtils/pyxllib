#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 17:34

import re
import sys
import textwrap
from pathlib import Path

from pyxllib.prog.lazyimport import lazy_import

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    BeautifulSoup = lazy_import('from bs4 import BeautifulSoup', 'beautifulsoup4')

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import('pandas')

try:
    import requests
except ModuleNotFoundError:
    requests = lazy_import('requests')

from pyxllib.prog.newbie import len_in_dim2
from pyxllib.prog.pupil import check_install_package
from pyxllib.prog.specialist import dataframe_str
from pyxllib.text.pupil import ContentLine
from pyxllib.file.specialist import get_encoding


def regularcheck(pattern, string, flags=0):
    arr = []
    cl = ContentLine(string)
    for i, m in enumerate(re.finditer(pattern, string, flags)):
        ss = map(lambda x: textwrap.shorten(x, 200), m.groups())
        arr.append([i + 1, cl.in_line(m.start(0)), *ss])
    tablehead = ['行号'] + list(map(lambda x: f'第{x}组', range(len_in_dim2(arr) - 2)))
    df = pd.DataFrame.from_records(arr, columns=tablehead)
    res = f'正则模式：{pattern}，匹配结果：\n' + dataframe_str(df)
    return res


def readtext(filename, encoding=None):
    """读取普通的文本文件
    会根据tex、py文件情况指定默认编码
    """
    try:
        with open(filename, 'rb') as f:  # 以二进制读取文件，注意二进制没有\r\n参数
            bstr = f.read()
    except FileNotFoundError:
        return None

    if not encoding:
        encoding = get_encoding(bstr)
    s = bstr.decode(encoding=encoding, errors='ignore')
    if '\r' in s:  # 注意这个问题跟gb2312和gbk是独立的，用gbk编码也要做这个处理
        s = s.replace('\r\n', '\n')  # 如果用\r\n作为换行符会有一些意外不好处理
    return s


def ensure_content(ob=None, encoding=None):
    """
    :param ob:
        未输入：从控制台获取文本
        存在的文件名：读取文件的内容返回
            tex、py、
            docx、doc
            pdf
        有read可调用成员方法：返回f.read()
        其他字符串：返回原值
    :param encoding: 强制指定编码
    """
    # TODO: 如果输入的是一个文件指针，也能调用f.read()返回所有内容
    # TODO: 增加鲁棒性判断，如果输入的不是字符串类型也要有出错判断
    if ob is None:
        return sys.stdin.read()  # 注意输入是按 Ctrl + D 结束
    if hasattr(ob, 'read'):
        return ob.read()

    try:
        p = Path(ob)
    except TypeError:
        return ob

    if p.is_file():
        suffix = p.suffix.lower()
        if suffix == '.docx':
            check_install_package('textract')
            import textract
            text = textract.process(str(p))
            return text.decode('utf8', errors='ignore')
        elif suffix == '.doc':
            raise NotImplementedError
        elif suffix == '.pdf':
            raise NotImplementedError
        else:
            return readtext(str(p), encoding)
    return ob


def file_lastlines(fn, n):
    """获得一个文件最后的几行内容
    参考资料: https://stackoverflow.com/questions/136168/get-last-n-lines-of-a-file-with-python-similar-to-tail

    >> s = FileLastLine('book.log', 1)
    'Output written on book.dvi (2 pages, 7812 bytes).'
    """
    assert n >= 0
    p = Path(fn)
    if not p.is_file():
        return ''

    with p.open('rb') as f:
        pos, lines = n + 1, []
        while len(lines) <= n:
            try:
                f.seek(-pos, 2)
            except OSError:
                f.seek(0)
                break
            finally:
                lines = f.readlines()
            pos *= 2

    b = b''.join(lines[-n:])
    enc = get_encoding(b)
    return b.decode(enc, errors='ignore')


def readurl(url):
    """从url读取文本"""
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    s = soup.get_text()
    return s
