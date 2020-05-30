#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2018/12/11 15:30

"""
用户私人配置信息
"""

import os
import socket

________config________ = """
一些用户配置数据

每个使用者可以修改_current_username、_get_texpath来配置两个全局变量：
USERNAME： 用户名
TEXPATH： latex文档数据的根目录

从而在一些函数中可以使用USERNAME进行分支个性化定制功能
或者使用TEXPATH，让涉及文档路径的操作对所有人的电脑都通用
"""

HOSTHAME = socket.getfqdn()


def _current_username():
    """返回当前程序调用者"""
    if HOSTHAME in ('INSPIRON15', 'CODE4101PC', 'codepc-mi'):
        return '坤泽'
    elif HOSTHAME in ('DESKTOP-CLLBF1T', 'HJJ'):
        return '韩某'
    elif HOSTHAME in ('DESKTOP-6RELMTT', 'LyeebnAcerLaTeX', 'Lyeebn-AMD'):
        return '奕本'
    elif HOSTHAME in ('histudy0001', 'olong', 'OLONG', r'OL'):
        return '欧龙'
    elif HOSTHAME in ('surface20180412.www.tendawifi.com', 'TEC-PC'):
        return '英杰'
    else:
        return HOSTHAME  # 原来是写成“游客”，但还是设成HOSTNAME吧~~


def _get_texpath():
    if HOSTHAME == 'INSPIRON15':  # 陈坤泽笔记本
        return 'D:\\'
    elif HOSTHAME == 'CODE4101PC':  # 陈坤泽公司的电脑
        return 'D:\\'
    elif HOSTHAME == 'DESKTOP-CLLBF1T':  # 锦锦公司电脑DESKTOP-ATA31JH
        return 'C:\坚果云'
    elif HOSTHAME == 'HJJ':  # 锦锦个人电脑DESKTOP-A5C5G04
        return 'H:\坚果云'
    elif HOSTHAME == 'DESKTOP-6M75D8U':  # 奕本公司电脑
        return 'E:/ChernKZ'
    elif HOSTHAME == 'surface20180412':  # 陈英杰笔记本
        return 'C:\\'
    elif HOSTHAME == 'TEC-PC':  #陈英杰公司电脑TEC-PC
        return 'C:\\'
    elif HOSTHAME == 'olong':  # 欧龙公司电脑
        return 'E:\\'
    else:  # 其他统一通过局域网访问
        return '\\\\code4101pc\d'


USERNAME = _current_username()
TEXPATH = _get_texpath()

CODE4101PY_PATH = os.path.dirname(os.path.dirname(__file__))
