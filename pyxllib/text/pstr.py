#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2025/06/05

r"""
PStr (Pattern String) - 增强型模式字符串类
=========================================

设计理念 (Design Philosophy)
---------------------------
PStr 旨在解决"字符串匹配"与"字符串本身"分离的问题。
它继承自 Python 原生 `str`，因此具有完全的兼容性：
1. 可以像普通字符串一样进行比较、哈希、作为字典 Key。
2. 携带了模式匹配的元数据 (Regex/Glob)，能够自解释其匹配逻辑。
3. 提供了统一的 `match`, `search`, `findall`, `sub` 等接口，消除了外部的 `isinstance` 判断。

核心特性 (Features)
------------------
- **完全兼容**: `isinstance(s, str)` 为 True，`s == "text"` 行为不变。
- **多态匹配**: 自动根据模式类型 (Literal/Regex/Glob) 选择匹配算法。
- **混合处理**: 可以在一个列表中混合放普通字符串和模式字符串，统一处理。

使用示例 (Usage Examples)
------------------------
1. 创建模式对象
    >>> p1 = PStr.re(r"user_\d+")   # 正则模式
    >>> p2 = PStr.glob("data_*.csv") # 通配符模式
    >>> s1 = PStr("admin")          # 普通字符串 (等价于 "admin")

2. 统一匹配与搜索接口
    >>> p1.match("user_123")
    True
    >>> p1.search("id: user_123").group(0)
    'user_123'
    >>> p1.findall("user_1, user_2")
    ['user_1', 'user_2']
    >>> p1.sub("user_XXX", "user_123")
    'user_XXX'

3. 批量筛选 (这是最典型的应用场景)
    >>> columns = ["id", "user_id", "user_name", "data_01", "meta_info"]
    >>> selectors = ["id", PStr.re(r"user_\w+"), PStr.glob("data_*")]
    >>>
    >>> selected = []
    >>> for sel in selectors:
    ...     if isinstance(sel, PStr) and not sel.is_literal:
    ...         selected.extend(sel.find_matches(columns))
    ...     elif sel in columns:
    ...         selected.append(sel)
    >>> print(selected)
    ['user_id', 'user_name', 'data_01', 'id']
"""

import re
import fnmatch


class PStr(str):
    """
    Pattern String (PStr) - 增强型字符串类

    完全兼容 str 的所有功能，同时携带模式匹配的元数据（Regex/Glob）。
    旨在替代专门的 FieldSelector 类，使代码更加通用和简洁。
    """

    # 模式类型常量
    _LITERAL = 0
    _REGEX = 1
    _GLOB = 2

    def __new__(cls, content, mode=_LITERAL) -> 'PStr':
        # str 是不可变类型，必须在 __new__ 中初始化
        obj = super().__new__(cls, content)
        # 使用私有属性存储模式信息
        obj._mode = mode
        obj._compiled = None

        # 预编译正则，提高后续匹配效率
        if mode == cls._REGEX:
            try:
                obj._compiled = re.compile(content)
            except re.error:
                # 即使正则编译失败，仍作为字符串存在，但在匹配时会失败或退化
                pass

        return obj

    # --- 工厂方法 (Factory Methods) ---

    @classmethod
    def re(cls, pattern):
        """创建一个正则模式字符串"""
        return cls(pattern, mode=cls._REGEX)

    @classmethod
    def glob(cls, pattern):
        """创建一个通配符模式字符串"""
        return cls(pattern, mode=cls._GLOB)

    # --- 属性判断 (Properties) ---

    @property
    def is_re(self):
        """是否为正则模式"""
        return self._mode == self._REGEX

    @property
    def is_glob(self):
        """是否为通配符模式"""
        return self._mode == self._GLOB

    @property
    def is_literal(self):
        """是否为普通字面量模式"""
        return self._mode == self._LITERAL

    # --- 核心能力 (Core Methods) ---

    def match(self, text):
        """
        判断 text 是否匹配当前模式。

        - Regex: 使用 re.search (包含匹配)
        - Glob: 使用 fnmatch.fnmatch (通常是全匹配)
        - Literal: 字符串相等比较
        """
        if self._mode == self._REGEX:
            if self._compiled:
                return bool(self._compiled.search(str(text)))
            return False
        elif self._mode == self._GLOB:
            return fnmatch.fnmatch(str(text), self)
        else:
            return self == str(text)

    def search(self, text):
        """
        执行正则风格的搜索，返回匹配对象 (Match Object) 或 None。

        - Regex: 使用 re.search (包含匹配)
        - Glob: 将通配符转为正则后匹配 (全匹配，因为 Glob 语义通常是全匹配)
        - Literal: 查找子串 (包含匹配，类似 str.find)
        """
        text = str(text)
        if self._mode == self._REGEX:
            if self._compiled:
                return self._compiled.search(text)
            return None
        elif self._mode == self._GLOB:
            # Glob 转正则后，通常带有 \Z 结尾锚点，但不带开头锚点。
            # 使用 re.match 可以确保从头匹配，符合 Glob 语义。
            import os
            flags = 0
            # fnmatch 在 Windows 下是不区分大小写的，这里做简单适配
            if os.path.normcase("A") == os.path.normcase("a"):
                flags = re.IGNORECASE
            
            pat = fnmatch.translate(self)
            return re.match(pat, text, flags=flags)
        else:
            # Literal 模式下，search 行为定义为"查找子串"
            pat = re.escape(self)
            return re.search(pat, text)

    def findall(self, text):
        """
        查找所有匹配项，返回列表。

        - Regex: 使用 re.findall
        - Glob: 如果全匹配，返回 [text]；否则返回 []
        - Literal: 查找所有非重叠出现的子串
        """
        text = str(text)
        if self._mode == self._REGEX:
            if self._compiled:
                return self._compiled.findall(text)
            return []
        elif self._mode == self._GLOB:
            # Glob 是全匹配
            if self.match(text):
                return [text]
            return []
        else:
            # Literal: 查找所有非重叠出现的子串
            pat = re.escape(self)
            return re.findall(pat, text)

    def sub(self, repl, text, count=0):
        """
        替换匹配项。

        - Regex: 使用 re.sub
        - Glob: 如果全匹配，返回 repl；否则返回原 text
        - Literal: 使用 str.replace
        """
        text = str(text)
        if self._mode == self._REGEX:
            if self._compiled:
                return self._compiled.sub(repl, text, count=count)
            return text
        elif self._mode == self._GLOB:
            if self.match(text):
                return repl
            return text
        else:
            # Literal
            if count == 0:
                return text.replace(self, repl)
            else:
                return text.replace(self, repl, count)

    def find_matches(self, candidates):
        """
        在候选列表中筛选出匹配的项。
        相当于 [c for c in candidates if self.match(c)]
        """
        return [c for c in candidates if self.match(c)]

    # --- 魔法方法 (Magic Methods) ---

    def __repr__(self):
        # 增强 repr 显示，方便调试时区分普通字符串和 PStr
        if self.is_re:
            return f"PStr.re('{self}')"
        elif self.is_glob:
            return f"PStr.glob('{self}')"
        else:
            return f"PStr('{self}')"

    # 注意：PStr 保持了 str 的 __eq__ 和 __hash__ 行为
    # 这意味着 PStr.re("abc") == "abc" 为 True
    # 这符合"完全兼容普通 str"的设计目标


# ==========================================
# 单元测试与使用示例
# ==========================================
if __name__ == '__main__':
    from loguru import logger

    logger.info("开始测试 PStr 类...")

    # 1. 测试普通字符串兼容性
    s1 = PStr("hello")
    assert s1 == "hello"
    assert isinstance(s1, str)
    assert s1.upper() == "HELLO"
    assert s1.is_literal
    logger.success("普通字符串兼容性测试通过")

    # 2. 测试 Regex 模式
    r1 = PStr.re(r"user_\d+")
    assert r1 == r"user_\d+"  # 字符串内容本身不变
    assert r1.is_re
    assert not r1.is_glob

    # 匹配测试
    candidates = ["user_1", "user_123", "admin", "superuser_1"]
    matches = r1.find_matches(candidates)
    # search 是包含匹配，所以 superuser_1 也会匹配上 (如果正则没写 ^$)
    assert "user_1" in matches
    assert "admin" not in matches
    logger.success(f"Regex 模式测试通过。匹配结果: {matches}")

    # 3. 测试 Glob 模式
    g1 = PStr.glob("*.txt")
    assert g1.is_glob

    files = ["a.txt", "b.py", "c.txt", "d.log"]
    matches = g1.find_matches(files)
    assert matches == ["a.txt", "c.txt"]
    logger.success(f"Glob 模式测试通过。匹配结果: {matches}")

    # 4. 混合列表处理测试 (模拟实际业务场景)
    # 假设我们有一个字段选择器列表，里面混杂了普通字符串和模式
    selectors = [
        "id",
        PStr.glob("analysis.*.score"),
        PStr.re(r"meta_\w+")
    ]

    all_columns = [
        "id", "name",
        "analysis.eye.score", "analysis.nose.score", "analysis.mouth.width",
        "meta_created", "meta_updated", "other"
    ]

    selected = []
    for sel in selectors:
        # PStr 的优势：既可以当 key 用 (如果完全匹配)，也可以当 pattern 用
        if isinstance(sel, PStr) and not sel.is_literal:
            # 如果是模式，则在候选池里筛选
            selected.extend(sel.find_matches(all_columns))
        else:
            # 如果是普通字符串(或 Literal PStr)，直接添加 (或者也可以检查是否存在)
            if sel in all_columns:
                selected.append(sel)

    # 去重保持顺序
    selected = list(dict.fromkeys(selected))

    expected = ["id", "analysis.eye.score", "analysis.nose.score", "meta_created", "meta_updated"]
    assert selected == expected
    logger.success(f"混合场景测试通过。最终选择字段: {selected}")

    # 5. 测试相等性 (Design Choice Check)
    # PStr.re("a") 应该等于 "a" 吗？ 是的，为了兼容性。
    assert PStr.re("abc") == "abc"
    assert PStr.glob("abc") == "abc"
    # 但是它们的功能不同
    assert PStr.re("a.").match("ab") is True
    assert PStr.glob("a.").match("ab") is False  # glob 的 . 只是普通字符
    logger.success("相等性与功能区分测试通过")

    # 6. 测试 search 方法
    # Regex search
    m = PStr.re(r"(\d+)").search("user_123_data")
    assert m is not None
    assert m.group(1) == "123"
    
    # Glob search (全匹配)
    m = PStr.glob("*.txt").search("data.txt")
    assert m is not None
    assert m.group() == "data.txt"
    
    # Literal search (子串匹配)
    m = PStr("def").search("abc def ghi")
    assert m is not None
    assert m.start() == 4
    
    logger.success("Search 方法测试通过")

    # 7. 测试 findall 方法
    # Regex
    assert PStr.re(r"\d+").findall("a123b456") == ["123", "456"]
    # Literal
    assert PStr("ab").findall("cababc") == ["ab", "ab"]
    # Glob
    assert PStr.glob("*.txt").findall("a.txt") == ["a.txt"]
    assert PStr.glob("*.txt").findall("a.py") == []

    # 8. 测试 sub 方法
    # Regex
    assert PStr.re(r"user_\d+").sub("user_new", "id: user_123") == "id: user_new"
    # Literal
    assert PStr("abc").sub("XYZ", "abc def abc") == "XYZ def XYZ"
    assert PStr("abc").sub("XYZ", "abc def abc", count=1) == "XYZ def abc"
    # Glob
    assert PStr.glob("*.txt").sub("new", "a.txt") == "new"
    assert PStr.glob("*.txt").sub("new", "a.py") == "a.py"

    logger.success("所有 PStr 方法测试通过")
