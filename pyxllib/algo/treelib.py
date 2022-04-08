#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/04/07 15:45

"""
树形结构相关的处理
"""

from pyxllib.prog.pupil import check_install_package

check_install_package('anytree')

import re

from anytree import Node

from pyxllib.prog.pupil import EnchantBase, run_once


class EnchantNode(EnchantBase):
    @classmethod
    @run_once
    def enchant(cls):
        names = cls.check_enchant_names([Node])
        cls._enchant(Node, names)

    @staticmethod
    def render(self, *, childiter=list, maxlevel=None):
        """

        也可以这样可视化：
        >> from anytree import RenderTree
        >> print(RenderTree(root))
        """
        from anytree import RenderTree

        ls = []
        for pre, fill, node in RenderTree(self, childiter=childiter, maxlevel=maxlevel):
            msg = re.search(r',\s*(.+?)\)', str(node))  # 显示node的属性值
            msg = ('\t' + msg.group(1)) if msg else ''
            ls.append(f'{pre}{node.name}{msg}')

        return '\n'.join(ls)


EnchantNode.enchant()
