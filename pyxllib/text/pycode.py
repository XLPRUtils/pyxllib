#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/20 11:46


import re

from pyxllib.text.nestenv import PyNestEnv


def remove_interaction_chars(text):
    """ 去掉复制的一段代码中，前导的“>>>”标记 """
    # 这个算法可能还不够严谨，实际应用中再逐步写鲁棒
    # ">>> "、"... "
    lines = [line[4:] for line in text.splitlines()]
    return '\n'.join(lines)


def sort_import(text):
    def cmp(line):
        """ 将任意一句import映射为一个可比较的list对象

        :return: 2个数值
            1、模块优先级
            2、import在前，from在后
        """
        name = re.search(r'(?:import|from)\s+(\S+)', line).group(1)
        for i, x in enumerate('stdlib prog algo text file cv data extend'.split()):
            name = name.replace('pyxllib.' + x, f'{i:02}')
        for i, x in enumerate('pyxllib pyxlpr xlproject'.split()):
            name = name.replace(x, f'~{i:02}')
        for i, x in enumerate('newbie pupil specialist expert'.split()):
            name = name.replace('.' + x, f'{i:02}')

        # 忽略大小写
        return [name.lower(), line.startswith('import')]

    def sort_part(m):
        parts = PyNestEnv(m.group()).imports().strings()
        parts = [p.rstrip() + '\n' for p in parts]
        parts.sort(key=cmp)
        return ''.join(parts)

    res = PyNestEnv(text).imports().sub(sort_part, adjacent=True)  # 需要邻接，分块处理
    return res


def rename_identifier(text, old_name, new_name):
    """ 标识符重命名
    """
    ne = PyNestEnv(text).identifier(old_name)
    new_text = ne.replace(new_name)
    return new_text
