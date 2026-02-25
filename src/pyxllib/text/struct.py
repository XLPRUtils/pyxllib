#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

import copy
import re
from urllib.parse import urlencode, parse_qs, urlsplit, urlunsplit


class ContentPartSpliter:
    """ 按照正则pattern将内容拆分为Part的功能类
    """

    def __init__(self, pattern):
        self.pattern = re.compile(pattern)

    def split(self, content):
        """ 拆分内容

        :return: [part1, part2, ...]
            其中part是str
        """
        parts = []
        last_end = 0
        for m in self.pattern.finditer(content):
            parts.append(content[last_end:m.start()])
            parts.append(m.group())
            last_end = m.end()
        parts.append(content[last_end:])
        return [p for p in parts if p]


class ContentLine:
    """ 内容行分析工具 """

    @classmethod
    def count_indent(cls, line):
        """ 计算缩进空格数 """
        m = re.match(r'^ *', line)
        return len(m.group()) if m else 0


def check_text_row_column(text):
    """ 检查文本的行数和列数 """
    lines = text.splitlines()
    n_rows = len(lines)
    n_cols = max(len(line) for line in lines) if lines else 0
    return n_rows, n_cols


class JsonEditConverter:
    """ json数据编辑转换器 """

    def __init__(self, data):
        self.data = data

    def to_lines(self):
        """ 转为可编辑的行列表 """
        pass  # 具体实现略

    def parse_lines(self, lines):
        """ 解析行列表回json """
        pass  # 具体实现略


class UrlQueryBuilder:
    """ URL查询参数构建器

    >>> u = UrlQueryBuilder('http://www.baidu.com/s?wd=python')
    >>> u.set('wd', 'c++')
    'http://www.baidu.com/s?wd=c%2B%2B'
    >>> u.add('ie', 'utf-8')
    'http://www.baidu.com/s?wd=c%2B%2B&ie=utf-8'
    """

    def __init__(self, url):
        self.url = url
        self.scheme, self.netloc, self.path, self.query_string, self.fragment = urlsplit(url)
        self.query_params = parse_qs(self.query_string)

    def set(self, key, value):
        """ 设置参数（覆盖） """
        self.query_params[key] = [str(value)]
        return self.build()

    def add(self, key, value):
        """ 添加参数（追加） """
        if key not in self.query_params:
            self.query_params[key] = []
        self.query_params[key].append(str(value))
        return self.build()

    def build(self):
        """ 构建新URL """
        # parse_qs返回的是dict[str, list[str]]，urlencode需要处理一下
        # doseq=True表示处理列表值为多个参数
        new_query = urlencode(self.query_params, doseq=True)
        return urlunsplit((self.scheme, self.netloc, self.path, new_query, self.fragment))
