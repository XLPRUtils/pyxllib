#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/30 20:49


import os


def is_url(arg):
    """输入是一个字符串，且值是一个合法的url"""
    from urllib.parse import urlparse
    if not isinstance(arg, str): return False
    try:
        result = urlparse(arg)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def is_file(arg, exists=True):
    """相较于标准库的os.path.isfile，对各种其他错误类型也会判False

    :param exists: arg不仅需要是一个合法的文件名，还要求其实际存在
        设为False，则只判断文件名合法性，不要求其一定要存在
    """
    if not isinstance(arg, str): return False
    if not exists:
        raise NotImplementedError
    return os.path.isfile(arg)
