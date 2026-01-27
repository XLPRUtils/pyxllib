
import unittest
from pyxllib.text.pycode import refine_quotes

class TestRefineQuotes(unittest.TestCase):
    def test_basic_conversion(self):
        self.assertEqual(refine_quotes('s = "hello"'), "s = 'hello'")
        self.assertEqual(refine_quotes('s = "hello world"'), "s = 'hello world'")
        self.assertEqual(refine_quotes('print("foo")'), "print('foo')")

    def test_escaped_quotes(self):
        # "foo\"bar" -> 'foo"bar'
        self.assertEqual(refine_quotes(r's = "foo\"bar"'), "s = 'foo\"bar'")
        
        # Input: 's = "foo\'bar"' (raw string: s = "foo'bar")
        # Body contains single quote, so should keep double quotes
        self.assertEqual(refine_quotes(r's = "foo\'bar"'), r's = "foo\'bar"')

    def test_backslashes(self):
        # Input: "foo\\" -> 'foo\\'
        self.assertEqual(refine_quotes(r'"foo\\"'), r"'foo\\'")
        
        # Input: "foo\\\"" -> 'foo\\"'
        # Expected: 'foo\\"' (foo + \ + \ + ")
        # Using explicit construction to avoid confusion
        expected = "'" + "foo" + "\\" + "\\" + '"' + "'"
        self.assertEqual(refine_quotes(r'"foo\\\""'), expected)

    def test_prefixes(self):
        # Raw strings - Should change if no internal single quotes
        self.assertEqual(refine_quotes(r'r"foo\"bar"'), r"r'foo\"bar'")
        self.assertEqual(refine_quotes(r'R"foo"'), r"R'foo'")
        
        # F-strings - Should change if no internal single quotes
        self.assertEqual(refine_quotes(r'f"foo"'), r"f'foo'")
        self.assertEqual(refine_quotes(r'F"foo{bar}"'), r"F'foo{bar}'")
        self.assertEqual(refine_quotes(r'fr"foo"'), r"fr'foo'")
        
        # Bytes - Should change
        self.assertEqual(refine_quotes(r'b"foo"'), r"b'foo'")
        
        # Unicode (legacy/valid) - Should change
        self.assertEqual(refine_quotes(r'u"foo"'), r"u'foo'")

    def test_nested_quotes_rule(self):
        # Rule: If content contains single quotes, keep double quotes (outer)
        
        # Normal string
        self.assertEqual(refine_quotes(r'"I\'m here"'), r'"I\'m here"')
        # Note: If input is "I'm here" (no escape needed in double quotes)
        self.assertEqual(refine_quotes(r'"I\'m here"'), r'"I\'m here"') # Input was already escaped?
        # Let's test unescaped single quote in double quote input
        self.assertEqual(refine_quotes('"I\'m here"'), '"I\'m here"')
        
        # Raw string
        # r"I'm" -> r"I'm" (Conversion to r'I\'m' would change content)
        self.assertEqual(refine_quotes(r'r"I\'m"'), r'r"I\'m"')

        # F-string
        self.assertEqual(refine_quotes(r'f"I\'m {x}"'), r'f"I\'m {x}"')


    def test_triple_quotes(self):
        # Should NOT change
        text = r'"""hello "world" """'
        self.assertEqual(refine_quotes(text), text)
        
        text = r"'''hello 'world' '''"
        self.assertEqual(refine_quotes(text), text)
        
        # Triple quotes containing double quotes
        text = r's = """ "foo" """'
        self.assertEqual(refine_quotes(text), text)

    def test_comments(self):
        # Should NOT change
        self.assertEqual(refine_quotes('# "comment"'), '# "comment"')
        self.assertEqual(refine_quotes('s = "foo" # "comment"'), "s = 'foo' # \"comment\"")

    def test_nested_structures(self):
        input_code = r'''
def foo(x):
    return {"key": "value", 'k2': "v2"}
'''
        expected_code = r'''
def foo(x):
    return {'key': 'value', 'k2': 'v2'}
'''
        self.assertEqual(refine_quotes(input_code), expected_code)

    def test_mixed_content(self):
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
        self.assertEqual(refine_quotes(input_code), expected_code)

    def test_nested_quotes_in_fstring_expression(self):
        # Ensure we don't break f-strings with nested quotes
        # Complex f-strings with braces should be skipped to avoid parsing errors
        # UPDATE: With robust tokenize implementation, we CAN optimize nested strings safely!
        # code = r'f"foo{\"bar\"}"' 
        # But f"foo{'bar'}" is valid.
        code = r'f"foo{"bar"}"' 
        expected = r"f'foo{'bar'}'" # Inner quotes refined, outer quotes refined if safe
        # Wait, if inner quotes refined to 'bar', then outer quotes (f") must stay f" because inner contains '
        # Or inner becomes 'bar'. f"...{'bar'}..."
        # If outer converted to f', then f'...{'bar'}...' -> nested ' inside f-string.
        # This is valid in python 3.12, but older python?
        # Python 3.12 allows arbitrary nesting.
        # But our rule: "If body contains single quote, keep double quotes".
        # f-string body parts: "foo{", "}"
        # Inner string "bar" is NOT part of f-string body in tokenize!
        # It is a separate STRING token.
        # So f-string body does NOT contain single quote.
        # So f-string CAN be converted to f'.
        # And inner "bar" CAN be converted to 'bar'.
        # Result: f'foo{'bar'}'
        # This is valid Python 3.12+.
        # If test environment is Python 3.12+, this is expected.
        self.assertEqual(refine_quotes(code), expected)

    def test_multiline_implicit_concatenation(self):
        input_code = r'''
s = ("line1"
     "line2")
'''
        expected_code = r'''
s = ('line1'
     'line2')
'''
        self.assertEqual(refine_quotes(input_code), expected_code)

if __name__ == '__main__':
    unittest.main()
