#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2025/06/05

r"""
pstr.py - Pattern String (增强型模式字符串类)
=========================================

设计理念 (Design Philosophy)
---------------------------
PStr 旨在解决'字符串匹配'与'字符串本身'分离的问题。
它继承自 Python 原生 `str`，因此具有完全的兼容性：
1. 可以像普通字符串一样进行比较、哈希、作为字典 Key。
2. 携带了模式匹配的元数据 (Regex/Glob)，能够自解释其匹配逻辑。
3. 提供了统一的 `match`, `search`, `findall`, `sub` 等接口，消除了外部的 `isinstance` 判断。

核心特性 (Features)
------------------
- **完全兼容**: `isinstance(s, str)` 为 True，`s == 'text'` 行为不变。
- **多态匹配**: 自动根据模式类型 (Literal/Regex/Glob) 选择匹配算法。
- **混合处理**: 可以在一个列表中混合放普通字符串和模式字符串，统一处理。

使用示例 (Usage Examples)
------------------------
>>> p1 = PStr.re(r'user_\d+')   # 正则模式
>>> p2 = PStr.glob('data_*.csv') # 通配符模式
>>> s1 = PStr('admin')          # 普通字符串 (等价于 'admin')
>>> p1.match('user_123')
True
>>> p1.search('id: user_123').group(0)
'user_123'
>>> p1.findall('user_1, user_2')
['user_1', 'user_2']
>>> p1.sub('user_XXX', 'user_123')
'user_XXX'
"""

import re
import fnmatch


def __1_pstr_base():
    """PStr 核心基类定义"""
    pass


class PStr(str):
    """Pattern String (PStr) - 增强型字符串类

    完全兼容 str 的所有功能，同时携带模式匹配的元数据（Regex/Glob）。
    采用了多态设计，根据模式类型自动分发到不同的子类实现。
    """

    def __new__(cls, content, *, ignore_case=False):
        """创建一个新的 PStr 对象

        :param str|PStr content: 字符串内容或已有的 PStr 对象
        :param bool ignore_case: 是否忽略大小写
        :return PStr: 返回具体的子类实例 (PStrLiteral, PStrRegex, 或 PStrGlob)
        """
        # 1. 如果 content 本身就是 PStr，则继承其 ignore_case，且 PStr(content) 会保留子类类型
        if isinstance(content, PStr):
            if not ignore_case:
                ignore_case = getattr(content, "_ignore_case", False)
            if cls is PStr:
                cls = content.__class__
            content = str(content)

        if cls is PStr:
            cls = PStrLiteral

        # 2. 初始化对象
        obj = super().__new__(cls, content)
        obj._ignore_case = bool(ignore_case)
        obj._init_pattern()
        return obj

    def _init_pattern(self):
        """子类重写此方法进行初始化工作"""
        pass

    # --- 工厂方法 (Factory Methods) ---

    @classmethod
    def re(cls, pattern, *, ignore_case=False):
        """创建一个正则模式字符串

        :param str pattern: 正则表达式
        :param bool ignore_case: 是否忽略大小写
        :return PStrRegex: 正则模式对象
        """
        return PStrRegex(pattern, ignore_case=ignore_case)

    @classmethod
    def glob(cls, pattern, *, ignore_case=False):
        """创建一个通配符模式字符串

        :param str pattern: glob 风格的模式字符串
        :param bool ignore_case: 是否忽略大小写
        :return PStrGlob: 通配符模式对象
        """
        return PStrGlob(pattern, ignore_case=ignore_case)

    @classmethod
    def auto(cls, pattern, *, ignore_case=False):
        """智能推断模式类型并创建对象

        推断逻辑：
        1. 强正则特征 -> PStrRegex
        2. 通配符特征 -> PStrGlob
        3. 其他 -> PStrLiteral
        """
        # 如果已经是 PStr 对象，直接返回（或转换类型）
        if isinstance(pattern, PStr):
            return cls(pattern, ignore_case=ignore_case)

        text = str(pattern)

        # 1. 判定 Regex
        # 特征：转义符, 锚点, 分组, 量词(+, {}), 管道符
        # 注意：不包含 '.', '*', '?', '[]'，因为这些在 Glob 或文件名中很常见
        regex_indicators = {"\\", "^", "$", "(", ")", "{", "}", "+", "|"}
        if any(char in regex_indicators for char in text):
            return cls.re(pattern, ignore_case=ignore_case)

        # 2. 判定 Glob
        # 特征：通配符 *, ?, 字符集 []
        # 前提：已经排除了强正则特征
        glob_indicators = {"*", "?", "["}
        if any(char in glob_indicators for char in text):
            return cls.glob(pattern, ignore_case=ignore_case)

        # 3. 默认为字面量
        # 包含 '.' 的文件名（如 'data.csv'）会落入这里，这是符合直觉的
        return cls(pattern, ignore_case=ignore_case)

    # --- 属性判断 (Properties) ---

    @property
    def pattern_type(self):
        """每个子类返回自己的类型标识"""
        return "literal"

    @property
    def is_re(self):
        """是否为正则模式"""
        return isinstance(self, PStrRegex)

    @property
    def is_glob(self):
        """是否为通配符模式"""
        return isinstance(self, PStrGlob)

    @property
    def is_literal(self):
        """是否为普通字面量模式"""
        return isinstance(self, PStrLiteral)

    @property
    def ignore_case(self):
        """是否忽略大小写"""
        return bool(getattr(self, "_ignore_case", False))

    # --- 核心能力接口 ---

    def match(self, text):
        """验证 text 是否满足模式匹配条件

        :param str text: 待匹配的文本
        :return bool: 匹配成功返回 True，否则返回 False

        不同模式下的行为：
        - Literal: text 与模式内容完全一致（或忽略大小写后一致）。
        - Regex: text 中包含满足正则表达式的部分（search 行为）。
        - Glob: text 整体满足 fnmatch 规则。
        """
        raise NotImplementedError

    def search(self, text):
        """在 text 中搜索满足模式的第一个匹配项

        :param str text: 待搜索的文本
        :return re.Match|None: 返回 re.Match 对象或 None

        注意：对于 Literal 和 Glob 模式，其实现也会尝试返回类似的 Match 对象或 None。
        """
        raise NotImplementedError

    def findall(self, text):
        """在 text 中找出所有满足模式的匹配项

        :param str text: 待处理的文本
        :return list[str]: 匹配到的字符串列表
        """
        raise NotImplementedError

    def sub(self, repl, text, count=0):
        """将 text 中满足模式的部分替换为 repl

        :param str|callable repl: 替换后的内容或处理函数
        :param str text: 原始文本
        :param int count: 替换次数，0 表示全部替换
        :return str: 替换后的文本
        """
        raise NotImplementedError

    def find_matches(self, candidates):
        """在候选列表中筛选出匹配的项

        :param list[str] candidates: 候选字符串列表
        :return list[str]: 匹配成功的子集
        """
        return [c for c in candidates if self.match(c)]

    def __repr__(self):
        name = self.__class__.__name__.replace("PStr", "")
        if not name:
            name = "Literal"
        return f"PStr.{name.lower()}('{self}')"


def __2_pstr_implementations():
    """各种模式的具体实现子类"""
    pass


class PStrLiteral(PStr):
    """普通字符串匹配类"""

    def match(self, text):
        if self.ignore_case:
            return self.lower() == str(text).lower()
        return self == str(text)

    def search(self, text):
        pat = re.escape(self)
        return re.search(pat, str(text), flags=re.IGNORECASE if self.ignore_case else 0)

    def findall(self, text):
        pat = re.escape(self)
        return re.findall(pat, str(text), flags=re.IGNORECASE if self.ignore_case else 0)

    def sub(self, repl, text, count=0):
        if self.ignore_case:
            pat = re.escape(self)
            return re.sub(pat, repl, str(text), count=count, flags=re.IGNORECASE)
        else:
            if count == 0:
                return str(text).replace(self, repl)
            return str(text).replace(self, repl, count)


class PStrRegex(PStr):
    """正则表达式匹配类"""

    @property
    def pattern_type(self):
        return "regex"

    def _init_pattern(self):
        flags = re.IGNORECASE if self.ignore_case else 0
        try:
            self._compiled = re.compile(str(self), flags=flags)
        except re.error:
            self._compiled = None

    def match(self, text):
        if self._compiled:
            return bool(self._compiled.search(str(text)))
        return False

    def search(self, text):
        if self._compiled:
            return self._compiled.search(str(text))
        return None

    def findall(self, text):
        if self._compiled:
            return self._compiled.findall(str(text))
        return []

    def sub(self, repl, text, count=0):
        if self._compiled:
            return self._compiled.sub(repl, str(text), count=count)
        return str(text)


class PStrGlob(PStr):
    """通配符匹配类"""

    @property
    def pattern_type(self):
        return "glob"

    def _init_pattern(self):
        pat = fnmatch.translate(self)
        flags = re.IGNORECASE if self.ignore_case else 0
        self._compiled = re.compile(pat, flags=flags)

    def match(self, text):
        return bool(self._compiled.match(str(text)))

    def search(self, text):
        return self._compiled.match(str(text))

    def findall(self, text):
        return [str(text)] if self.match(text) else []

    def sub(self, repl, text, count=0):
        return repl if self.match(text) else str(text)
