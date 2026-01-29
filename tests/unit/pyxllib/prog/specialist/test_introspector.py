#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import pandas as pd
from pyxllib.prog.specialist.browser import Introspector, ObjectFormatter, inspect_object

class Base:
    """基类"""
    def __init__(self):
        self.base_attr = "base"
    
    def base_method(self):
        return "base"

class Sub(Base):
    """子类"""
    def __init__(self):
        super().__init__()
        self.sub_attr = 123
    
    def sub_method(self):
        return "sub"

@pytest.fixture
def test_obj():
    return Sub()

def test_introspector_meta(test_obj):
    """测试 Introspector 的元数据获取功能"""
    intro = Introspector(test_obj)
    
    # 1. 内存信息
    memory_info = intro.get_memory_info()
    assert "内存消耗" in memory_info
    assert "递归子类总大小" in memory_info
    
    # 2. MRO
    df_mro = intro.get_mro_dataframe()
    assert isinstance(df_mro, pd.DataFrame)
    assert len(df_mro) >= 3  # Sub, Base, object
    assert "Sub" in str(df_mro.iloc[0, 0])
    
    # 3. 元数据字符串
    meta_info = intro.get_meta_info()
    assert "类继承关系" in meta_info
    assert "Sub" in meta_info

def test_introspector_members(test_obj):
    """测试 Introspector 的成员提取功能"""
    intro = Introspector(test_obj)
    df_fields, df_methods = intro.get_members()
    
    # 1. 成员变量
    fields = df_fields["成员变量"].tolist()
    assert "base_attr" in fields
    assert "sub_attr" in fields
    
    # 2. 成员函数
    methods = df_methods["成员函数"].tolist()
    assert "base_method" in methods
    assert "sub_method" in methods

def test_formatter_text(test_obj):
    """测试 ObjectFormatter 的文本渲染"""
    intro = Introspector(test_obj)
    formatter = ObjectFormatter(intro)
    text = formatter.to_text()
    
    assert "[类继承关系]" in text
    assert "[成员变量]" in text
    assert "[成员函数]" in text
    assert "base_attr" in text
    assert "sub_method" in text

def test_formatter_html(test_obj):
    """测试 ObjectFormatter 的 HTML 渲染"""
    intro = Introspector(test_obj)
    formatter = ObjectFormatter(intro)
    html_content = formatter.to_html(title_name="TestReport")
    
    assert "<h1>TestReport 查看报告</h1>" in html_content
    assert "<table" in html_content
    assert "base_attr" in html_content
    assert "sub_method" in html_content

def test_inspect_object_modes(test_obj):
    """测试 inspect_object 的不同模式"""
    # 1. str 模式
    res_str = inspect_object(test_obj, mode="str")
    assert isinstance(res_str, str)
    assert "[成员变量]" in res_str
    
    # 2. html_str 模式
    res_html = inspect_object(test_obj, mode="html_str")
    assert "<h1>Sub 查看报告</h1>" in res_html
    
    # 3. text 模式 (由于会调用 logger.info，我们主要检查返回值)
    res_text = inspect_object(test_obj, mode="text")
    assert res_text == res_str

if __name__ == "__main__":
    pytest.main([__file__])
