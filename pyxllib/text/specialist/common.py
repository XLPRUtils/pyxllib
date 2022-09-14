#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 17:34

import re
import sys
import textwrap

from bs4 import BeautifulSoup
import pandas as pd
import requests

from pyxllib.prog.newbie import len_in_dim2
from pyxllib.prog.pupil import check_install_package
from pyxllib.prog.specialist import dataframe_str
from pyxllib.text.pupil import ContentLine
from pyxllib.file.specialist import get_encoding, File


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
    elif File(ob):  # 如果存在这样的文件，那就读取文件内容（bug点：如果输入是目录名会PermissionError）
        if ob.endswith('.docx'):  # 这里还要再扩展pdf、doc文件的读取
            # 安装详见： https://blog.csdn.net/code4101/article/details/79328636
            check_install_package('textract')
            text = textract.process(ob)
            return text.decode('utf8', errors='ignore')
        elif ob.endswith('.doc'):
            raise NotImplementedError
        elif ob.endswith('.pdf'):
            raise NotImplementedError
        else:  # 按照普通的文本文件读取内容
            return readtext(ob, encoding)
    else:  # 判断不了的情况，也认为是字符串
        return ob


def file_lastlines(fn, n):
    """获得一个文件最后的几行内容
    参考资料: https://stackoverflow.com/questions/136168/get-last-n-lines-of-a-file-with-python-similar-to-tail

    >> s = FileLastLine('book.log', 1)
    'Output written on book.dvi (2 pages, 7812 bytes).'
    """
    f = ensure_content(fn)
    assert n >= 0
    pos, lines = n + 1, []
    while len(lines) <= n:
        try:
            f.seek(-pos, 2)
        except IOError:
            f.seek(0)
            break
        finally:
            lines = list(f)
        pos *= 2
    f.close()
    return ''.join(lines[-n:])


def readurl(url):
    """从url读取文本"""
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    s = soup.get_text()
    return s
