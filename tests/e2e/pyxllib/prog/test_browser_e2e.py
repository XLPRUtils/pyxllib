# -*- coding: utf-8 -*-
"""
test_browser_e2e.py - 浏览器功能的端到端（E2E）测试。
这些测试会真实打开应用程序，具有副作用，通常不作为自动化 CI 的一部分。
手动运行方式：
    python d:\home\chenkunze\slns\pyxllib\tests\e2e\pyxllib\prog\test_browser_e2e.py
或者使用 pytest 运行特定函数：
    pytest d:\home\chenkunze\slns\pyxllib\tests\e2e\pyxllib\prog\test_browser_e2e.py -k test_open_url
"""

import os
import time
from pyxllib.prog.browser import browser, view_files

def test_open_url():
    """测试打开一个真实的 URL"""
    print("正在打开网页：https://www.baidu.com")
    browser("https://www.baidu.com")

def test_view_html_content():
    """测试直接展示 HTML 字符串内容"""
    content = """
    <html>
    <head><title>E2E Test</title></head>
    <body>
        <h1 style="color: blue;">Hello, this is an E2E test for Browser!</h1>
        <p>Generated at: {}</p>
    </body>
    </html>
    """.format(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("正在展示 HTML 字符串内容")
    browser(content)

def test_view_text_content():
    """测试展示纯文本内容（会自动转存为 .txt 文件并用浏览器打开）"""
    content = "这是一段纯文本内容，用于测试 Browser 类的兜底展示功能。\n" * 5
    print("正在展示纯文本内容")
    browser(content)

def test_view_dataframe_mock():
    """模拟展示 DataFrame（带有 to_html 方法的对象）"""
    class MockDF:
        def to_html(self):
            return '<table border="1"><tr><th>Col1</th><th>Col2</th></tr><tr><td>Data1</td><td>Data2</td></tr></table>'
    
    print("正在展示模拟的 DataFrame 表格")
    browser(MockDF())

def test_view_files_with_notepad():
    """测试使用 notepad 打开临时文件（仅限 Windows）"""
    if os.name == 'nt':
        print("正在使用 notepad 打开临时文件")
        view_files('notepad', "这是测试 notepad 打开的内容", name="e2e_test_notepad")
    else:
        print("非 Windows 系统，跳过 notepad 测试")

if __name__ == "__main__":
    # 手动运行时依次执行这些测试
    print("开始执行 Browser E2E 测试...")
    
    test_open_url()
    time.sleep(1)
    
    test_view_html_content()
    time.sleep(1)
    
    test_view_text_content()
    time.sleep(1)
    
    test_view_dataframe_mock()
    time.sleep(1)
    
    test_view_files_with_notepad()
    
    print("\nE2E 测试触发完毕，请检查浏览器和相关程序是否已正确打开目标内容。")
