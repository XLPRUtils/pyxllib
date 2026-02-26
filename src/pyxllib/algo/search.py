#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 14:22

import textwrap

from pyxllib.prog.basic import typename
from pyxllib.text.format import listalign
from pyxllib.text.convert import int2myalphaenum


class SearchBase:
    """ 一个dfs、bfs模板类 """

    def __init__(self, root):
        """
        Args:
            root: 根节点
        """
        self.root = root

    def get_neighbors(self, node):
        """ 获得邻接节点，必须要用yield实现，方便同时支持dfs、bfs的使用

        对于树结构而言，相当于获取直接子结点

        这里默认是bs4中Tag规则；不同业务需求，可以重定义该函数
        例如对图结构、board类型，可以在self存储图访问状态，在这里实现遍历四周的功能
        """
        try:
            for node in node.children:
                yield node
        except AttributeError:
            pass

    def dfs_nodes(self, node=None, depth=0):
        """ 返回深度优先搜索得到的结点清单

        :param node: 起始结点，默认是root根节点
        :param depth: 当前node深度
        :return: list，[(node1, depth1), (node2, depth2), ...]
        """
        if not node:
            node = self.root

        ls = [(node, depth)]
        for t in self.get_neighbors(node):
            ls += self.dfs_nodes(t, depth + 1)
        return ls

    def bfs_nodes(self, node=None, depth=0):
        if not node:
            node = self.root

        ls = [(node, depth)]
        i = 0

        while i < len(ls):
            x, d = ls[i]
            nodes = self.get_neighbors(x)
            ls += [(t, d + 1) for t in nodes]
            i += 1

        return ls

    def fmt_node(self, node, depth, *, prefix='    ', show_node_type=False):
        """ node格式化显示 """
        s1 = prefix * depth
        s2 = typename(node) + '，' if show_node_type else ''
        s3 = textwrap.shorten(str(node), 200)
        return s1 + s2 + s3

    def fmt_nodes(self, *, nodes=None, select_depth=None, linenum=False,
                  msghead=True, show_node_type=False, prefix='    '):
        """ 结点清单格式化输出

        :param nodes: 默认用dfs获得结点，也可以手动指定结点
        :param prefix: 缩进格式，默认用4个空格
        :param select_depth: 要显示的深度
            单个数字：获得指定层
            Sequences： 两个整数，取出这个闭区间内的层级内容
        :param linenum：节点从1开始编号
            行号后面，默认会跟一个类似Excel列名的字母，表示层级深度
        :param msghead: 第1行输出一些统计信息
        :param show_node_type:

        Requires
            textwrap：用到shorten
            align.listalign：生成列编号时对齐
        """
        # 1 生成结点清单
        ls = nodes if nodes else self.dfs_nodes()
        total_node = len(ls)
        total_depth = max(map(lambda x: x[1], ls))
        head = f'总节点数：1~{total_node}，总深度：0~{total_depth}'

        # 2 过滤与重新整理ls（select_depth）
        logo = True
        cnt = 0
        tree_num = 0
        if isinstance(select_depth, int):

            for i in range(total_node):
                if ls[i][1] == select_depth:
                    ls[i][1] = 0
                    cnt += 1
                    logo = True
                elif ls[i][1] < select_depth and logo:  # 遇到第1个父节点添加一个空行
                    ls[i] = ''
                    tree_num += 1
                    logo = False
                else:  # 删除该节点，不做任何显示
                    ls[i] = None
            head += f'；挑选出的节点数：{cnt}，所选深度：{select_depth}，树数量：{tree_num}'

        elif hasattr(select_depth, '__getitem__'):
            for i in range(total_node):
                if select_depth[0] <= ls[i][1] <= select_depth[1]:
                    ls[i][1] -= select_depth[0]
                    cnt += 1
                    logo = True
                elif ls[i][1] < select_depth[0] and logo:  # 遇到第1个父节点添加一个空行
                    ls[i] = ''
                    tree_num += 1
                    logo = False
                else:  # 删除该节点，不做任何显示
                    ls[i] = None
            head += f'；挑选出的节点数：{cnt}，所选深度：{select_depth[0]}~{select_depth[1]}，树数量：{tree_num}'
        """注意此时ls[i]的状态，有3种类型
            (node, depth)：tuple类型，第0个元素是node对象，第1个元素是该元素所处层级
            None：已删除元素，但为了后续编号方便，没有真正的移出，而是用None作为标记
            ''：已删除元素，但这里涉及父节点的删除，建议此处留一个空行
        """

        # 3 格式处理
        def mystr(item):
            return self.fmt_node(item[0], item[1], prefix=prefix, show_node_type=show_node_type)

        line_num = listalign(range(1, total_node + 1))
        res = []
        for i in range(total_node):
            if ls[i] is not None:
                if isinstance(ls[i], str):  # 已经指定该行要显示什么
                    res.append(ls[i])
                else:
                    if linenum:  # 增加了一个能显示层级的int2excel_col_name
                        res.append(line_num[i] + int2myalphaenum(ls[i][1]) + ' ' + mystr(ls[i]))
                    else:
                        res.append(mystr(ls[i]))

        s = '\n'.join(res)

        # 是否要添加信息头
        if msghead:
            s = head + '\n' + s

        return s
