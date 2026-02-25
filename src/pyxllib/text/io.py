#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

import sys
import os
from pathlib import Path

from pyxllib.file.xlpath import XlPath
from pyxllib.prog.lazyimport import lazy_import

try:
    import requests
except ModuleNotFoundError:
    requests = lazy_import('requests')


def readtext(f, encoding=None):
    """ 读取文本文件内容

    :param f: 文件路径
    :param encoding: 编码格式，默认None会尝试自动识别
    """
    return XlPath(f).read_text(encoding=encoding)


def ensure_content(arg, encoding=None):
    """ 确保输入的是文本内容

    :param arg:
        str: 视为文件路径，尝试读取
        Path: 读取文件内容
        其他: 视为文本内容，直接返回str(arg)
    """
    if isinstance(arg, (str, Path)):
        if os.path.exists(arg):
            try:
                return readtext(arg, encoding)
            except OSError:
                pass  # 可能是路径过长等原因，此时把arg当做文本内容返回

    return str(arg)


def file_lastlines(f, n=10, encoding=None):
    """ 读取文件最后n行 """
    lines = readtext(f, encoding).splitlines()
    return lines[-n:] if n > 0 else []


def readurl(url, encoding=None, **kwargs):
    """ 读取url内容 """
    try:
        res = requests.get(url, **kwargs)
        if encoding:
            res.encoding = encoding
        return res.text
    except Exception as e:
        print(f'readurl error: {e}', file=sys.stderr)
        return None


class Stdout:
    """重定向输出"""

    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = open(self.file, 'w', encoding='utf8')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.stdout
