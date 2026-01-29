# -*- coding: utf-8 -*-

from pyxllib.text.pycode import refine_quotes

def test_basic_conversion():
    assert refine_quotes('s = "hello"') == "s = 'hello'"
    assert refine_quotes('s = "hello world"') == "s = 'hello world'"
    assert refine_quotes('print("foo")') == "print('foo')"

def test_escaped_quotes():
    # "foo\"bar" -> 'foo"bar'
    assert refine_quotes(r's = "foo\"bar"') == "s = 'foo\"bar'"
    
    # Input: 's = "foo\'bar"' (raw string: s = "foo'bar")
    # Body contains single quote, so should keep double quotes
    assert refine_quotes(r's = "foo\'bar"') == r's = "foo\'bar"'

def test_backslashes():
    # Input: "foo\\" -> 'foo\\'
    assert refine_quotes(r'"foo\\"') == r"'foo\\'"
    
    # Input: "foo\\\"" -> 'foo\\"'
    # Expected: 'foo\\"' (foo + \ + \ + ")
    # Using explicit construction to avoid confusion
    expected = "'" + "foo" + "\\" + "\\" + '"' + "'"
    assert refine_quotes(r'"foo\\\""') == expected

def test_prefixes():
    # Raw strings - Should change if no internal single quotes
    assert refine_quotes(r'r"foo\"bar"') == r"r'foo\"bar'"
    assert refine_quotes(r'R"foo"') == r"R'foo'"
    
    # F-strings - Should change if no internal single quotes
    assert refine_quotes(r'f"foo"') == r"f'foo'"
    assert refine_quotes(r'F"foo{bar}"') == r"F'foo{bar}'"
    assert refine_quotes(r'fr"foo"') == r"fr'foo'"
    
    # Bytes - Should change
    assert refine_quotes(r'b"foo"') == r"b'foo'"
    
    # Unicode (legacy/valid) - Should change
    assert refine_quotes(r'u"foo"') == r"u'foo'"

def test_nested_quotes_rule():
    # Rule: If content contains single quotes, keep double quotes (outer)
    
    # Normal string
    assert refine_quotes(r'"I\'m here"') == r'"I\'m here"'
    # Note: If input is "I'm here" (no escape needed in double quotes)
    assert refine_quotes(r'"I\'m here"') == r'"I\'m here"' # Input was already escaped?
    # Let's test unescaped single quote in double quote input
    assert refine_quotes('"I\'m here"') == '"I\'m here"'
    
    # Raw string
    # r"I'm" -> r"I'm" (Conversion to r'I\'m' would change content)
    assert refine_quotes(r'r"I\'m"') == r'r"I\'m"'

    # F-string
    assert refine_quotes(r'f"I\'m {x}"') == r'f"I\'m {x}"'

def test_triple_quotes():
    # Should NOT change
    text = r'"""hello "world" """'
    assert refine_quotes(text) == text
    
    text = r"'''hello 'world' '''"
    assert refine_quotes(text) == text
    
    # Triple quotes containing double quotes
    text = r's = """ "foo" """'
    assert refine_quotes(text) == text

def test_comments():
    # Should NOT change
    assert refine_quotes('# "comment"') == '# "comment"'
    assert refine_quotes('s = "foo" # "comment"') == "s = 'foo' # \"comment\""

def test_nested_structures():
    input_code = r'''
def foo(x):
    return {"key": "value", 'k2': "v2"}
'''
    expected_code = r'''
def foo(x):
    return {'key': 'value', 'k2': 'v2'}
'''
    assert refine_quotes(input_code) == expected_code

def test_mixed_content():
    # Complex case with all types
    input_code = r'''
import os
s1 = "double"
s2 = 'single'
s3 = """triple
double"""
s4 = r"raw double"
s5 = f"format double {x}"
# comment with "double"
print("done")
'''
    expected_code = r'''
import os
s1 = 'double'
s2 = 'single'
s3 = """triple
double"""
s4 = r'raw double'
s5 = f'format double {x}'
# comment with "double"
print('done')
'''
    assert refine_quotes(input_code) == expected_code

def test_nested_quotes_in_fstring_expression():
    # Ensure we don't break f-strings with nested quotes
    # Complex f-strings with braces should be skipped to avoid parsing errors
    # UPDATE: With robust tokenize implementation, we CAN optimize nested strings safely!
    code = r'f"foo{"bar"}"' 
    expected = r"f'foo{'bar'}'" # Inner quotes refined, outer quotes refined if safe
    assert refine_quotes(code) == expected

def test_multiline_implicit_concatenation():
    input_code = r'''
s = ("line1"
     "line2")
'''
    expected_code = r'''
s = ('line1'
     'line2')
'''
    assert refine_quotes(input_code) == expected_code

if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
