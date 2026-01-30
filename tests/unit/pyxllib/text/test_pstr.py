#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pytest
from pyxllib.text.pstr import PStr, PStrLiteral, PStrRegex, PStrGlob

def test_pstr_literal():
    """测试普通字符串兼容性"""
    s1 = PStr("hello")
    assert s1 == "hello"
    assert isinstance(s1, str)
    assert isinstance(s1, PStrLiteral)
    assert s1.upper() == "HELLO"
    assert s1.is_literal

def test_pstr_regex():
    """测试 Regex 模式"""
    r1 = PStr.re(r"user_\d+")
    assert r1 == r"user_\d+"
    assert isinstance(r1, PStrRegex)
    assert r1.is_re

    candidates = ["user_1", "user_123", "admin", "superuser_1"]
    matches = r1.find_matches(candidates)
    assert "user_1" in matches
    assert "user_123" in matches
    assert "admin" not in matches

def test_pstr_glob():
    """测试 Glob 模式"""
    g1 = PStr.glob("*.txt")
    assert isinstance(g1, PStrGlob)
    assert g1.is_glob

    files = ["a.txt", "b.py", "c.txt", "d.log"]
    matches = g1.find_matches(files)
    assert matches == ["a.txt", "c.txt"]

def test_mixed_list_processing():
    """混合列表处理测试"""
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
        if isinstance(sel, PStr) and not sel.is_literal:
            selected.extend(sel.find_matches(all_columns))
        else:
            if sel in all_columns:
                selected.append(sel)

    selected = list(dict.fromkeys(selected))
    expected = ["id", "analysis.eye.score", "analysis.nose.score", "meta_created", "meta_updated"]
    assert selected == expected

def test_equality_and_differentiation():
    """测试相等性与功能区分"""
    assert PStr.re("abc") == "abc"
    assert PStr.glob("abc") == "abc"
    assert PStr.re("a.").match("ab") is True
    assert PStr.glob("a.").match("ab") is False

def test_search():
    """测试 search 方法"""
    # Regex search
    m = PStr.re(r"(\d+)").search("user_123_data")
    assert m is not None and m.group(1) == "123"
    
    # Glob search
    m = PStr.glob("*.txt").search("data.txt")
    assert m is not None and m.group() == "data.txt"
    
    # Literal search
    m = PStr("def").search("abc def ghi")
    assert m is not None and m.start() == 4

def test_findall():
    """测试 findall 方法"""
    assert PStr.re(r"\d+").findall("a123b456") == ["123", "456"]
    assert PStr("ab").findall("cababc") == ["ab", "ab"]
    assert PStr.glob("*.txt").findall("a.txt") == ["a.txt"]

def test_sub():
    """测试 sub 方法"""
    assert PStr.re(r"user_\d+").sub("user_new", "id: user_123") == "id: user_123".replace("user_123", "user_new")
    assert PStr("abc").sub("XYZ", "abc def abc") == "XYZ def XYZ"
    assert PStr.glob("*.txt").sub("new", "a.txt") == "new"

def test_ignore_case():
    """测试忽略大小写"""
    p = PStr("abc", ignore_case=True)
    assert p.match("ABC")
    assert p.match("abc")
    
    p2 = PStr.re("user", ignore_case=True)
    assert p2.match("USER_1")

def test_auto():
    """测试 auto 智能推断模式类型"""
    p1 = PStr.auto(r"user_\d+")
    assert isinstance(p1, PStrRegex)
    assert p1.match("USER_1") is False

    p2 = PStr.auto("data_*.csv")
    assert isinstance(p2, PStrGlob)
    assert p2.match("data_01.csv") is True

    p3 = PStr.auto("data.csv")
    assert isinstance(p3, PStrLiteral)
    assert p3.match("data.csv") is True

    p4 = PStr.auto("*.TXT", ignore_case=True)
    assert isinstance(p4, PStrGlob)
    assert p4.match("a.txt") is True

    p5 = PStr.auto(r"user_\d+", ignore_case=True)
    assert isinstance(p5, PStrRegex)
    assert p5.match("USER_1") is True

    p6 = PStr.auto("AbC", ignore_case=True)
    assert isinstance(p6, PStrLiteral)
    assert p6.match("abc") is True

    p7 = PStr.auto(PStr.glob("*.txt"))
    assert isinstance(p7, PStrGlob)
    assert p7.match("a.txt") is True

if __name__ == '__main__':
    pytest.main([__file__])
