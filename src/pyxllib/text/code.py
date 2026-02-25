#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

import base64
import zlib


class Base85Coder:
    """ Base85编码器，用于压缩字符串

    >>> s = 'hello world'
    >>> e = Base85Coder.encode(s)
    >>> d = Base85Coder.decode(e)
    >>> d == s
    True
    """

    @classmethod
    def encode(cls, s):
        """ 压缩编码 """
        if isinstance(s, str):
            s = s.encode('utf-8')
        c = zlib.compress(s)
        return base64.b85encode(c).decode('utf-8')

    @classmethod
    def decode(cls, s):
        """ 解压解码 """
        b = base64.b85decode(s)
        return zlib.decompress(b).decode('utf-8')
