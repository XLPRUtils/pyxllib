#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/20 11:46

""" 一些特殊的专用业务功能

因为是业务方面的，所以函数名可能多用中文~~
"""

import re

____check = """
检查异常系列的功能
"""

____refine = """
文本优化的功能
"""


def 去除py交互标记(s):
    """ 去掉复制的一段代码中，前导的“>>>”标记 """
    # 这个算法可能还不够严谨，实际应用中再逐步写鲁棒
    # ">>> "、"... "
    lines = [line[4:] for line in s.splitlines()]
    return '\n'.join(lines)


def import重排序(s):
    from pyxllib.text.nestenv import PycodeNestEnv

    def cmp(line):
        """ 将任意一句import映射为一个可比较的list对象

        :return: 2个数值
            1、模块优先级
            2、import在前，from在后
        """
        name = re.search(r'(?:import|from)\s+(\S+)', line).group(1)
        for i, x in enumerate('stdlib prog algo text file debug excel cv data database gui ai robot tool ex'.split()):
            name = name.replace('pyxllib.' + x, f'{i:02}')
        for i, x in enumerate('pyxllib pyxlpr xlproject'.split()):
            name = name.replace(x, f'~{i:02}')
        for i, x in enumerate('newbie pupil specialist expert'.split()):
            name = name.replace('.' + x, f'{i:02}')

        # 忽略大小写
        return [name.lower(), line.startswith('import')]

    def sort_part(m):
        parts = PycodeNestEnv(m.group()).imports().strings()
        parts = [p.rstrip() + '\n' for p in parts]
        parts.sort(key=cmp)
        return ''.join(parts)

    res = PycodeNestEnv(s).imports().sub(sort_part, adjacent=True)  # 需要邻接，分块处理
    return res


____extract = """
信息摘要提取功能
"""
