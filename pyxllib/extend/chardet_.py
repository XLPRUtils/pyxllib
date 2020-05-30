#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/30 21:32


import subprocess


try:
    import chardet
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'chardet'])
    import chardet


from pyxllib.util.ddprint import dprint
from pyxllib.util.judge import is_file


def get_encoding(bstr):
    """输入二进制字符串，返回字符编码，并会把GB2312改为GBK
    :return: 'utf-8' 或 'GBK'

    备注：又想正常情况用chardet快速识别，又想异常情况暴力编码试试，代码就写的这么又臭又长了~~

    200530周六21:31 附： 这个函数太别扭了，无特殊情况还是不要用吧，写的并不好
    """
    # 1、读取编码
    detect = None
    if isinstance(bstr, bytes):  # 如果输入是一个二进制字符串流则直接识别
        detect = chardet.detect(bstr[:1024])  # 截断一下，不然太长了，太影响速度
        encoding = detect['encoding']
    elif is_file(bstr):  # 如果是文件，则按二进制打开
        # 如果输入是一个文件名则进行读取
        if bstr.endswith('.pdf'):
            dprint(bstr)  # 二进制文件，不应该进行编码分析，暂且默认返回utf8
            return 'utf-8'
        with open(bstr, 'rb') as f:  # 以二进制读取文件，注意二进制没有\r\n的值
            bstr = f.read()
        encoding = get_encoding(bstr)
    else:  # 其他类型不支持
        return 'utf-8'
    # 检测结果存储在encoding

    # 2、智能适应优化，最终应该只能是gbk、utf8两种结果中的一种
    if encoding in ('ascii', 'utf-8', 'ISO-8859-1'):
        # 对ascii类编码，理解成是utf-8编码；ISO-8859-1跟ASCII差不多
        encoding = 'utf-8'
    elif encoding in ('GBK', 'GB2312'):
        encoding = 'GBK'
    elif bstr.strip():  # 如果bstr非空
        # 进入这个if分支算是比较异常的情况，会输出原识别结果detect
        try:  # 先尝试utf8编码，如果编码成功则认为是utf8
            bstr.decode('utf8')
            encoding = 'utf-8'
            dprint(detect)  # chardet编码识别异常，根据文件内容已优化为utf8编码
        except UnicodeDecodeError:
            try:  # 否则尝试gbk编码
                bstr.decode('gbk')
                encoding = 'GBK'
                dprint(detect)  # chardet编码识别异常，根据文件内容已优化为gbk编码
            except UnicodeDecodeError:  # 如果两种都有问题
                encoding = 'utf-8'
                dprint(detect)  # 警告：chardet编码识别异常，已强制使用utf8处理
    else:
        encoding = 'utf-8'

    return encoding
