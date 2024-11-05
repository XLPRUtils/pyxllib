#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/05

"""
以uiautomation为核心的相关工具库
"""

import uiautomation as auto

# ui组件大多是树形组织结构，没有专业的树形结构库，能搞个毛线。什么？auto库有自带树形操作？那点基础功能能干毛线。
from anytree import NodeMixin, RenderTree


class UiCtrlNode(NodeMixin):
    def __init__(self, ctrl, parent=None):
        # 初始化节点信息
        self.ctrl = ctrl
        self.ctrl_type = ctrl.ControlTypeName
        self.text = ctrl.Name
        self.parent = parent  # 指定父节点，用于形成树结构

        # 自动递归创建子节点
        self.build_children()

    def build_children(self):
        """ 创建并添加子节点到树中 """
        for child_ctrl in self.ctrl.GetChildren():
            self.__class__(child_ctrl, parent=self)  # 递归构建子节点

    def __repr__(self):
        """ 用于在打印节点时显示关键信息 """
        return f"UiNode(ctrl_type={self.ctrl_type}, text={self.text.replace('\n', ' ')})"

    def __getitem__(self, index):
        """ 通过索引直接访问子节点

        ui操作经常要各种结构化的访问，加个这个简化引用方式
        """
        return self.children[index]

    def render_tree(self):
        """ 可视化输出树结构信息进行数据结构分析 """
        for pre, fill, node in RenderTree(self):  # 使用 self.root 确保从根节点渲染
            print(f"{pre}{node.ctrl_type} {node.get_hash_tag()} {node.text.replace('\n', ' ')}")

    def get_hash_tag(self, level=1):
        """ 生成节点的哈希字符串，以反映子树结构，一般用来对节点做分类及映射到对应处理函数 """
        # 当前节点的类型标识符
        hash_strs = [f"{level}{self.ctrl_type[0].lower()}"]
        # 遍历所有子节点，递归生成子节点的哈希值
        for child in self.children:
            hash_strs.append(f"{child.get_hash_tag(level + 1)}")
        return ''.join(hash_strs)
