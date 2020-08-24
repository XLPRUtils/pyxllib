#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/30 21:32


import chardet

from .dprint import dprint
from .judge import is_file


def get_encoding(bstr):
    """ 输入二进制字符串或文本文件，返回字符编码

    https://www.yuque.com/xlpr/pyxllib/get_encoding

    :param bstr: 二进制字符串、文本文件
    :return: utf8, utf-8-sig, gbk, utf16
    """
    # 1 读取编码
    if isinstance(bstr, bytes):  # 如果输入是一个二进制字符串流则直接识别
        encoding = chardet.detect(bstr[:1024])['encoding']  # 截断一下，不然太长了，太影响速度
    elif is_file(bstr):  # 如果是文件，则按二进制打开
        # 如果输入是一个文件名则进行读取
        if bstr.endswith('.pdf'):
            dprint(bstr)  # 二进制文件，不应该进行编码分析，暂且默认返回utf8
            return 'utf8'
        with open(bstr, 'rb') as f:  # 以二进制读取文件，注意二进制没有\r\n的值
            bstr = f.read()
        encoding = chardet.detect(bstr[:1024])['encoding']
    else:  # 其他类型不支持
        return 'utf8'
    # 检测结果存储在encoding

    # 2 智能适应优化，最终应该只能是gbk、utf8两种结果中的一种
    if encoding in ('ascii', 'utf-8', 'ISO-8859-1'):
        # 对ascii类编码，理解成是utf-8编码；ISO-8859-1跟ASCII差不多
        encoding = 'utf8'
    elif encoding in ('GBK', 'GB2312'):
        encoding = 'gbk'
    elif encoding == 'UTF-16':
        encoding = 'utf16'
    elif encoding == 'UTF-8-SIG':
        # 保留原值的一些正常识别结果
        encoding = 'utf-8-sig'
    elif bstr.strip():  # 如果不在预期结果内，且bstr非空，则用常见的几种编码尝试
        # dprint(encoding)
        type_ = ('utf8', 'gbk', 'utf16')

        def try_encoding(bstr, encoding):
            try:
                bstr.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                return False

        for t in type_:
            encoding = try_encoding(bstr, t)
            if encoding: break
    else:
        encoding = 'utf8'

    return encoding
