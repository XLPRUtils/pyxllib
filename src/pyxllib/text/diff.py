#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

import re

from pyxllib.prog.basic import typename


class StrDiffType:
    """ 比较两个字符串的差异 """

    @classmethod
    def _split(cls, s):
        """ 将字符串拆分成行列表 """
        if isinstance(s, str):
            return s.splitlines()
        elif isinstance(s, (list, tuple)):
            return [str(x) for x in s]
        else:
            raise TypeError(f'Unsupported type: {typename(s)}')

    @classmethod
    def diff(cls, s1, s2):
        """ 比较两个字符串差异，返回差异行

        TODO 使用 difflib 库实现更详细的差异比较
        """
        lines1 = cls._split(s1)
        lines2 = cls._split(s2)

        res = []
        import difflib
        for line in difflib.unified_diff(lines1, lines2):
            res.append(line)

        return '\n'.join(res)


def briefstr(s, length=50):
    """ 简略显示字符串

    :param length: 限制长度
    """
    s = str(s)
    if len(s) <= length:
        return s
    return s[:length // 2] + '...' + s[-length // 2:]
