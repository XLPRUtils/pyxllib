#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 21:17

import os
import re
import shutil
import struct


def struct_unpack(f, fmt):
    r""" 类似np.fromfile的功能，读取并解析二进制数据

    :param f:
        如果带有read方法，则用read方法读取指定字节数
        如果bytes对象则直接处理
    :param fmt: 格式
        默认按小端解析(2, 1, 0, 0) -> 258，如果需要大端，可以加前缀'>'
        字节：c=char, b=signed char, B=unsigned char, ?=bool
        2字节整数：h=short, H=unsigned short（后文同理，大写只是变成unsigned模式，不在累述）
        4字节整数：i, I, l, L
        8字节整数：q, Q
        浮点数：e=2字节，f=4字节，d=8字节

    >>> b = struct.pack('B', 127)
    >>> b
    b'\x7f'
    >>> struct_unpack(b, 'c')
    b'\x7f'
    >>> struct_unpack(b, 'B')
    127

    >>> b = struct.pack('I', 258)
    >>> b
    b'\x02\x01\x00\x00'
    >>> struct_unpack(b, 'I')  # 默认都是按小端打包、解析
    258
    >>> struct_unpack(b, '>I') # 错误示范，按大端解析的值
    33619968
    >>> struct_unpack(b, 'H'*2)  # 解析两个值，fmt*2即可
    (258, 0)

    >>> f = io.BytesIO(b'\x02\x01\x03\x04')
    >>> struct_unpack(f, 'B'*3)  # 取前3个值，等价于np.fromfile(f, dtype='uint8', count=3)
    (2, 1, 3)
    >>> struct_unpack(f, 'B')  # 取出第4个值
    4
    """
    # 1 取数据
    size_ = struct.calcsize(fmt)
    if hasattr(f, 'read'):
        data = f.read(size_)
        if len(data) < size_:
            raise ValueError(f'剩余数据长度 {len(data)} 小于 fmt 需要的长度 {size_}')
    else:  # 对于bytes等矩阵，可以多输入，但是只解析前面一部分
        data = f[:size_]

    # 2 解析
    res = struct.unpack(fmt, data)
    if len(res) == 1:  # 解析结果恰好只有一个的时候，返回值本身
        return res[0]
    else:
        return res


def recreate_folders(*dsts):
    """重建一个空目录"""
    for dst in dsts:
        try:
            # 删除一个目录（含内容），设置ignore_errors可以忽略目录不存在时的错误
            shutil.rmtree(dst, ignore_errors=True)
            os.makedirs(dst)  # 重新新建一个目录，注意可能存在层级关系，所以要用makedirs
        except TypeError:
            pass


def checkpathfile(name):
    r"""判断环境变量path下是否有name这个文件，有则返回绝对路径，无则返回None
    常用的有：BCompare.exe、Chrome.exe、mogrify.exe、xelatex.exe

    >> checkpathfile('xelatex.exe')
    'C:\\CTEX\\MiKTeX\\miktex\\bin\\xelatex.exe'
    >> checkpathfile('abcd.exe')
    """
    for path in os.getenv('path').split(';'):
        fn = os.path.join(path, name)
        if os.path.exists(fn):
            return fn
    return None


def filename_tail(fn, tail):
    """在文件名末尾和扩展名前面加上一个tail"""
    names = os.path.splitext(fn)
    return names[0] + tail + names[1]


def hasext(f, *exts):
    """判断文件f是否是exts扩展名中的一种，如果不是返回False，否则返回对应的值

    所有文件名统一按照小写处理
    """
    ext = os.path.splitext(f)[1].lower()
    exts = tuple(map(lambda x: x.lower(), exts))
    if ext in exts:
        return ext
    else:
        return False


def isdir(fn):
    """判断输入的是不是合法的路径格式，且存在确实是一个文件夹"""
    try:
        return os.path.isdir(fn)
    except ValueError:  # 出现文件名过长的问题
        return False
    except TypeError:  # 输入不是字符串类型
        return False


__mygetfiles = """
py有os.walk可以递归遍历得到一个目录下的所有文件
但是“我们”常常要过滤掉备份文件（171020-153959），Old、temp目、.git等目录
特别是windows还有一个很坑爹的$RECYCLE.BIN目录。
所以在os.walk的基础上，再做了封装得到myoswalk。

然后在myoswalk基础上，实现mygetfiles。
"""


def gen_file_filter(s):
    """生成一个文件名过滤函数"""
    if s[0] == '.':
        return lambda x: x.endswith(s)
    else:
        s = s.replace('？', r'[\u4e00-\u9fa5]')  # 中文问号可以匹配任意中文字符
        return lambda x: re.search(s, x)


def getfiles(root, filter_rule=None):
    r""" 对os.walk进一步封装，返回所有匹配的文件

    可以这样遍历一个目录下的所有文件：
    for f in getfiles(r'C:\pycode\code4101py', r'.py'):
        print(f)
    筛选规则除了“.+后缀”，还可以写正则匹配
    """
    if isinstance(filter_rule, str):
        filter_rule = gen_file_filter(filter_rule)

    for root, _, files in os.walk(root, filter_rule):
        for f in files:
            if filter_rule and not filter_rule(f):
                continue
            yield root + '\\' + f


def tex_content_filefilter(f):
    """只获取正文类tex文件"""
    if f.endswith('.tex') and 'Conf' not in f and 'settings' not in f:
        return True
    else:
        return False


def tex_conf_filefilter(f):
    """只获取配置类tex文件"""
    if f.endswith('.tex') and ('Conf' in f or 'settings' in f):
        return True
    else:
        return False


def change_ext(filename, ext):
    """更改文件名后缀
    返回第1个参数是新的文件名，第2个参数是这个文件是否存在

    输入的fileName可以没有扩展名，如'A/B/C/a'，仍然可以找对应的扩展名为ext的文件
    输入的ext不要含有'.'，例如正确格式是输入'tex'、'txt'
    """
    name = os.path.splitext(filename)[0]  # 'A/B/C/a.txt' --> 'A/B/C/a'
    newname = name + '.' + ext
    return newname, os.path.exists(newname)
