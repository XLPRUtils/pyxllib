#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 21:17


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

def linux_path_fmt(p):
    p = str(p)
    p = p.replace('\\', '/')
    return p
