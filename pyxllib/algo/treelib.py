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
    def sign_node(self, check_func, *, flag_name='_flag', child_depth=-1, reset_flag=False):
        """ 遍历所有结点

        :param check_func: 检查node是否符合要求
            True，自身打上标记2，其余父节点、子结点全部打上1 （这里处理方式后续可以再扩展更灵活）
        """
        from anytree import PreOrderIter

        if child_depth == -1:
            child_depth = None
        else:
            child_depth += 1

        if reset_flag:
            for x in PreOrderIter(self):
                setattr(x, flag_name, 0)

        def set_flag(node, flag):
            # 结点没有值或者flag小余当前值时，才标记更新为当前flag
            setattr(node, flag_name, max(flag, getattr(node, flag_name, 0)))

        cnt = 0
        for x in PreOrderIter(self, filter_=check_func):
            # 对找到的结点标记2
            cnt += 1
            set_flag(x, 2)
            # 其父结点、子结点标记1
            for y in x.ancestors:
                set_flag(y, 1)
            for y in PreOrderIter(x, maxlevel=child_depth):
                set_flag(y, 1)
        return cnt

    @staticmethod
    def render(self, *, childiter=list, maxlevel=None, filter_=None, dedent=0):
        """

        :param dedent: 统一减少缩进层级

        也可以这样可视化：
        >> from anytree import RenderTree
        >> print(RenderTree(root))
        """
        from anytree import RenderTree

        def fold_text(s):
            return s.replace('\n', ' ')[:150]

        ls = []
        for pre, fill, node in RenderTree(self, childiter=childiter, maxlevel=maxlevel):
            if filter_:
                if not filter_(node):
                    continue
                # 使用filter_参数时，前缀统一成空格
                pre_len = (len(pre) // 4 - dedent)
            else:
                pre_len = len(pre)

            if pre_len < 0:
                ls.append('')  # 加入一个空行
            else:
                pre = pre_len * '\t'
                msg = re.search(r',\s*(.+?)\)', str(node))  # 显示node的属性值
                msg = ('\t' + msg.group(1)) if msg else ''
                ls.append(fold_text(f'{pre}{node.name}{msg}'))

        return '\n'.join(ls)

    @staticmethod
    def render_html(self, html_attr_name='name', *,
                    childiter=list, maxlevel=None, filter_=None, padding_mode=0,
                    dedent=0):
        """ 渲染成html页面内容

        :param html_attr_name: 最好原node有一个字段存储需要渲染的html内容
            默认使用name的值作为渲染对象
        :param padding_mode:
            0，默认配置，使用html自带的悬挂缩进功能。这种展示效果最好，但复制内容的时候不带缩进。
            1，使用空格缩进。长文本折行显示效果不好。但是方便复制。
        :param dedent: 统一减少缩进层级
            根据这里的实现原理，这个其实也可以是负数，表示增加缩进层级~

        也可以这样可视化：
        >> from anytree import RenderTree
        >> print(RenderTree(root))
        """
        from anytree import RenderTree

        ls = []
        for pre, fill, node in RenderTree(self, childiter=childiter, maxlevel=maxlevel):
            if filter_ and not filter_(node):
                continue
            pre_len = len(pre) - dedent * 4

            if pre_len < 0:
                ls.append('<br/>')
            else:
                msg = re.search(r',\s*(.+?)\)', str(node))  # 显示node的属性值
                msg = ('\t' + msg.group(1)) if msg else ''
                content = f'{getattr(node, html_attr_name, node.name)}{msg}'
                if not content.strip():
                    content = '<br/>'
                if padding_mode == 1:
                    ls.append(f'<div>{"&nbsp;" * pre_len}{content}</div>')
                else:
                    ls.append(f'<div style="padding-left:{pre_len // 2 + 1}em;text-indent:-1em">{content}</div>')

        return '\n'.join(ls)


EnchantNode.enchant()
