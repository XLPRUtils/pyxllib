#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2026/02/01

"""
view_obj.py - Object Inspection and Visualization Tool

这是一个用于深度分析 Python 对象结构的工具，类似于内置的 dir() 但功能更强大。
它可以提取对象的类继承链、成员变量、成员函数，并计算递归内存占用。
支持多种输出模式：控制台、纯文本字符串、HTML 报告以及直接在浏览器中打开。

核心用法示例：
>>> class MyClass:
...     def __init__(self):
...         self.a = 1
...         self.b = 'hello'
...     def my_method(self):
...         pass
>>> obj = MyClass()
>>> # 在控制台查看对象信息
>>> view_obj(obj, mode='console')  # doctest: +SKIP
>>> # 获取 HTML 格式的报告
>>> html_report = view_obj(obj, mode='html')
"""

import builtins
import enum
import html
import inspect
import sys
import types

from loguru import logger

from pyxllib.prog.lazyimport import lazy_import
from pyxllib.prog.newbie import typename
from pyxllib.text.pupil import shorten

try:
    from humanfriendly import format_size
except ModuleNotFoundError:
    format_size = lazy_import('from humanfriendly import format_size')

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import('pandas')

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    BeautifulSoup = lazy_import('from bs4 import BeautifulSoup', 'beautifulsoup4')

from pyxllib.text.renderer.pandas import to_text, to_html


def __1_基础工具():
    """ 包含一些通用的辅助函数 """
    pass


def getasizeof(*objs, **opts):
    """获得所有类的大小，底层用 pympler.asizeof 实现

    :param objs: 要计算大小的对象
    :param opts: 传递给 asizeof 的其他参数
    :return int: 对象总大小，如果计算失败返回 -1
    """
    from pympler import asizeof

    try:
        res = asizeof.asizeof(*objs, **opts)
    except Exception:
        res = -1
    return res


def __2_内省引擎():
    """ 负责分析对象结构，提取成员变量和方法 """
    pass


class Introspector:
    """负责分析对象结构，提取成员变量和方法，不处理任何展示逻辑"""

    def __init__(self, obj):
        """初始化内省器

        :param obj: 要分析的对象
        """
        self.obj = obj

    def get_memory_info(self):
        """获取内存消耗信息

        :return str: 内存信息描述字符串
        """
        size = sys.getsizeof(self.obj)
        recursive_size = getasizeof(self.obj)
        lines = [f'内存消耗：{format_size(size, binary=True)}']
        lines.append(f"递归子类总大小：{format_size(recursive_size, binary=True) if recursive_size != -1 else 'Unknown'}")
        return '\n'.join(lines)

    def get_mro_dataframe(self):
        """获取 MRO 继承链的 DataFrame

        :return pd.DataFrame: 包含继承链信息的表格
        """
        mro = inspect.getmro(type(self.obj))
        data = [[str(cls)] for cls in mro]
        df = pd.DataFrame(data, columns=['类继承层级'])
        return df

    def get_meta_info(self):
        """获取对象的元数据（继承关系、内存大小等）

        :return str: 元数据描述字符串
        """
        memory_info = self.get_memory_info()
        mro = inspect.getmro(type(self.obj))
        return f'==== 类继承关系：{mro}，{memory_info} ===='

    def get_html_meta_info(self):
        """获取对象的 HTML 格式元数据

        :return str: HTML 格式的元数据
        """
        return '<p>' + html.escape(self.get_meta_info()) + '</p>'

    def get_members(self):
        """获取成员列表，返回 (Fields_DataFrame, Methods_DataFrame)

        :return tuple: (df_fields, df_methods)
        """
        members = self._get_all_members()

        # 分离变量与方法
        fields_data = []
        methods_data = []

        for name, value in members:
            # 过滤掉内部特殊标记
            if name.endswith('________'):
                continue

            # 简单的判断：可调用的是方法，不可调用的是变量
            if callable(value):
                methods_data.append([name, str(value)])
            else:
                fields_data.append([name, self._format_field_value(value)])

        df_fields = pd.DataFrame(fields_data, columns=['成员变量', '描述'])
        df_methods = pd.DataFrame(methods_data, columns=['成员函数', '描述'])

        return df_fields, df_methods

    def _get_all_members(self):
        """提取所有成员

        :return list: 包含 (name, value) 元组的列表
        """
        results = []
        processed = set()

        # 1. 尝试 dir()
        names = dir(self.obj)

        # 2. 补漏 (DynamicClassAttribute 等)
        if hasattr(self.obj, '__bases__'):
            for base in self.obj.__bases__:
                for k, v in base.__dict__.items():
                    if isinstance(v, types.DynamicClassAttribute):
                        names.append(k)

        for key in names:
            try:
                value = getattr(self.obj, key)
            except Exception:
                # 某些属性可能在 getattr 时报错，尝试从类字典获取
                found = False
                for base in inspect.getmro(type(self.obj)):
                    if key in base.__dict__:
                        value = base.__dict__[key]
                        found = True
                        break
                if not found:
                    continue

            if key not in processed:
                results.append((key, value))
                processed.add(key)

        results.sort(key=lambda pair: pair[0])
        return results

    def _format_field_value(self, value):
        """处理单个变量值的格式化

        :param value: 变量值
        :return str: 格式化后的字符串
        """
        if isinstance(value, enum.IntFlag):
            return f'{typename(value)}，{int(value)}，{value}'
        try:
            return f'{typename(value)}，{value}'
        except Exception:
            return '无法转换为str'


def __3_渲染引擎():
    """ 负责将内省数据渲染成不同格式 """
    pass


class ObjectFormatter:
    """负责将 Introspector 提供的数据渲染成不同格式"""

    def __init__(self, introspector, width=200):
        """初始化格式化器

        :param Introspector introspector: 内省器对象
        :param int width: 字符串显示的最大宽度
        """
        self.introspector = introspector
        self.width = width

    def to_text(self):
        """生成适合控制台输出的纯文本

        :return str: 纯文本报告
        """
        memory_info = self.introspector.get_memory_info()
        df_mro = self.introspector.get_mro_dataframe()
        df_fields, df_methods = self.introspector.get_members()

        # 截断过长的字符串
        for df in [df_fields, df_methods]:
            if not df.empty:
                df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: shorten(x, width=self.width))

        res = ['==== 内存信息 ====']
        res.append(memory_info)
        res.append('[对象值]')
        res.append(to_text(self.introspector.obj))
        res.append('[类继承关系]')
        res.append(to_text(df_mro))
        res.append('[成员变量]')
        res.append(to_text(df_fields))
        res.append('[成员函数]')
        res.append(to_text(df_methods))
        return '\n'.join(res)

    def to_html(self, title_name='Object'):
        """生成适合浏览器查看的 HTML

        :param str title_name: 报告标题
        :return str: HTML 格式的报告
        """
        memory_info = self.introspector.get_memory_info()
        df_mro = self.introspector.get_mro_dataframe()
        df_fields, df_methods = self.introspector.get_members()

        # HTML 内容构建
        html_parts = []

        # 1. Header
        html_parts.append(f'<h1>{title_name} 查看报告</h1>')
        html_parts.append(f'<p>{html.escape(memory_info).replace(chr(10), "<br/>")}</p>')

        # 1.5 Object Value
        html_parts.append('<h2>对象值</h2>')
        html_parts.append(to_html(self.introspector.obj))

        # 2. Helper to style tables
        def _style_df(df, type_char, header_color='LightSkyBlue'):
            if df.empty:
                return '<p>无成员信息</p>'

            # 转字符串防止 HTML 注入或编码错误
            df_str = df.map(str)
            # 截断（如果只有一列则不截断第二列，因为 MRO 只有一列）
            if df_str.shape[1] > 1:
                df_str.iloc[:, 1] = df_str.iloc[:, 1].apply(lambda x: shorten(x, width=self.width))

            df_str.index += 1
            html_content = df_str.to_html()

            # 使用 BS4 美化
            soup = BeautifulSoup(html_content, 'lxml')
            if soup.thead and soup.thead.tr:
                soup.thead.tr['bgcolor'] = header_color
                # 设置表头
                th_label = f'编号{type_char}{len(df)}'
                if soup.thead.tr.th:
                    soup.thead.tr.th.string = th_label
            return soup.prettify()

        # 3. Append Tables
        html_parts.append('<h2>类继承关系</h2>')
        html_parts.append(_style_df(df_mro, 'C', header_color='Khaki'))

        html_parts.append('<h2>成员变量</h2>')
        html_parts.append(_style_df(df_fields, 'F', header_color='LightGreen'))

        html_parts.append('<h2>成员函数</h2>')
        html_parts.append(_style_df(df_methods, 'M', header_color='LightSkyBlue'))

        return '\n'.join(html_parts)


def __4_便捷接口():
    """ 提供用户直接调用的入口函数 """
    pass


def view_obj(obj, mode='str', width=200):
    """查看对象信息的通用入口函数

    :param obj: 要查看的对象
    :param str mode: 查看模式
        - auto: 自动选择，Windows 默认 browser，其他环境默认 console
        - console|text: 打印到控制台
        - str: 返回纯文本字符串
        - html: 返回 HTML 字符串
        - browser: 在浏览器中打开报告
    :param int width: 字符串截断宽度
    :return str: 报告内容

    >>> res = view_obj(123, mode='str')
    >>> 'int' in res
    True
    """
    introspector = Introspector(obj)
    formatter = ObjectFormatter(introspector, width=width)

    if mode == 'auto':
        mode = 'browser' if sys.platform == 'win32' else 'console'

    if mode in ('console', 'text'):
        content = formatter.to_text()
        logger.info(content)
        return content
    elif mode == 'str':
        return formatter.to_text()
    elif mode == 'html':
        obj_name = type(obj).__name__
        return formatter.to_html(title_name=obj_name)
    elif mode == 'browser':
        from pyxllib.prog.browser import browser, get_hash

        obj_name = type(obj).__name__
        content = formatter.to_html(title_name=obj_name)
        # 使用类名作为前缀，4位内容哈希作为后缀，防止多次调研结果互相覆盖
        h = get_hash(content)[:4]
        browser(content, name=f'{obj_name}_{h}')
        return content
    else:
        raise ValueError(f'不支持的查看模式：{mode}')


setattr(builtins, 'view_obj', view_obj)
