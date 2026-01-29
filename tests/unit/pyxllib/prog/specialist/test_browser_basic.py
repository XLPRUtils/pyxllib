#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pyxllib.prog.specialist.browser import inspect_object

class GrandParent:
    """祖父类"""
    def __init__(self):
        self.gp_attr = "grandparent"
    
    def gp_method(self):
        return "gp"

class Parent(GrandParent):
    """父类"""
    def __init__(self):
        super().__init__()
        self.p_attr = 123
    
    def p_method(self):
        return "p"

class Child(Parent):
    """子类"""
    def __init__(self):
        super().__init__()
        self.c_attr = [1, 2, 3]
    
    def c_method(self):
        return "c"

def test_rendering():
    # 1. 创建一个具有复杂继承关系的对象
    obj = Child()
    
    # 2. 调用 inspect_object，使用 browser 模式展示
    # 这将生成 HTML 并自动在默认浏览器中打开
    print("正在生成报告并打开浏览器...")
    inspect_object(obj, mode="browser", width=100)

if __name__ == "__main__":
    test_rendering()
