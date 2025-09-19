#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/04/07 15:45

"""
树形结构相关的处理
"""

import re

from pyxllib.prog.lazyimport import lazy_import

try:
    from anytree import Node
except ModuleNotFoundError:
    Node = lazy_import('from anytree import Node')
    # 由于代码中使用了 class XlNode(Node)，所以缺失这个包是会立马报错的


class XlNode(Node):

    def sign_node(self, check_func, *, flag_name='_flag', child_depth=-1, reset_flag=False):
        """ 遍历所有结点

        :param check_func: 检查node是否符合要求
            True，自身打上标记2，其余父节点、子结点全部打上1 （这里处理方式后续可以再扩展更灵活）

        不是那么通用，但也可能存在复用的功能，做了个封装
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

    def render(self, attrname='name', *,
               childiter=list, maxlevel=None, filter_=None,  # 结点的迭代、过滤等规则
               prefix='\t',
               dedent=0):
        """ 渲染展示结构树

        :pram attrname:
            str, 要展示的成员变量值
            def func(node, depth)，自定义展示样式
        :param prefix: 缩进样式
            str，每个层级增加的样式，会直接用 prefix*depth作为前缀
            tuple(3), 长度3的tuple，对应AbstractStyle(vertical, cont, end)
            def func(depth), 输入深度，自定义前缀样式 （未实装，想到这个理念，但觉得用处不大）
        :param dedent: 统一减少缩进层级

        """
        from anytree import RenderTree

        # 1 文本格式化的规则
        if isinstance(attrname, str):
            def to_str(node, depth):
                if depth < 0:
                    return ''
                return getattr(node, attrname)
        elif callable(attrname):
            to_str = attrname
        else:
            raise ValueError

        # 2 遍历结点
        ls = []
        for pre, fill, node in RenderTree(self, childiter=childiter, maxlevel=maxlevel):
            if filter_ and not filter_(node):
                continue

            depth = len(pre) // 4 - dedent
            ls.append(prefix * depth + to_str(node, depth))

        return '\n'.join(ls)

    def render_html(self, attrname='name', **kwargs):
        """ 渲染成html页面内容
        """
        if isinstance(attrname, str):
            def attrname(node, depth):
                if depth < 0:
                    return '<br/>'
                content = getattr(node, attrname)
                div = f'<div style="padding-left:{depth * 2 + 1}em;text-indent:-1em">{content}</div>'
                return div

        return XlNode.render(self, attrname, **kwargs)

    def find_parent(self, check_func):
        """ 查找包含自身的所有父结点中，name符合check_func的结点 """
        if isinstance(check_func, str):
            def _check_func(x):
                return x.name == check_func
        elif isinstance(check_func, re.Pattern):
            def _check_func(x):
                return check_func.search(x.name)
        elif callable(check_func):
            _check_func = check_func
        else:
            raise ValueError

        p = self
        while p and not _check_func(p):
            p = p.parent
        return p

    def next_preorder_node(self, iter_child=True):
        """ 自己写的先序遍历

        主要应用在xml、bs4相关遍历检索时，有时候遇到特殊结点
            可能子结点不需要解析
            或者整个cur_node和子结点已经被解析完了，不需要再按照通常的先序遍历继续进入子结点
        此时可以 iter_child=False，进入下一个兄弟结点
        """
        from anytree.util import rightsibling

        if iter_child and self.children:
            return self.children[0]
        else:
            cur_node = self
            while True:
                parent = cur_node.parent
                if parent is None:
                    return None
                sibing = rightsibling(cur_node)
                if sibing:
                    return sibing
                cur_node = parent


if __name__ == '__main__':
    # XlNode(1)
    pass
