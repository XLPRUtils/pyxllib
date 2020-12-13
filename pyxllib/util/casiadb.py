#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/11/08 09:30

"""

Database Home: http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html

CASIA 在线和离线中文手写数据库的一些数据读取功能

"""

from pyxllib.basic import *
import numpy as np


def read_from_dgrl(dgrl):
    """ 解析中科院的DGRL格式数据

    参考代码：https://blog.csdn.net/DaGongJiGuoMaLu09/article/details/107050519
        有做了大量简化、工程封装

    TODO 可以考虑做一个返回类似labelme格式的接口，会更通用
        因为有时候会需要取整张原图
        而且如果有整个原图，那么每个文本行用shape形状标记即可，不需要取出子图

    :param dgrl: dgrl 格式的文件，或者对应的二进制数据流
    :return: [(img0, label0), (img1, label1), ...]
    """
    # 输入参数可以是bytes，也可以是文件
    f = XlBytesIO(dgrl)
    # 表头尺寸
    header_size = f.unpack('I')
    # 表头剩下内容，提取 code_length
    header = f.read(header_size - 4)
    code_length = struct_unpack(header[-4:-2], 'H')  # 每个字符存储的字节数，一般都是用gbk编码，2个字节
    # 读取图像尺寸信息，文本行数量
    height, width, line_num = f.unpack('I' * 3)

    # 读取每一行的信息
    res = []
    for k in range(line_num):
        # 读取该行的字符数量
        char_num = f.unpack('I')
        label = f.readtext(char_num, code_length=code_length)
        label = label.replace('\x00', '')  # 去掉不可见字符 \x00，这一步不加的话后面保存的内容会出现看不见的问题

        # 读取该行的位置和尺寸
        y, x, h, w = f.unpack('I' * 4)

        # 读取该行的图片
        bitmap = f.unpack('B' * (h * w))
        bitmap = np.array(bitmap).reshape(h, w)

        res.append((bitmap, label))

    return res
