# -*- coding: utf-8 -*-
import os
import pathlib
import pytest
from unittest.mock import patch, MagicMock
from pyxllib.prog.browser import Browser, get_hash, Explorer

def test_get_hash():
    assert get_hash("hello") == get_hash("hello")
    assert get_hash("hello") != get_hash("world")
    assert isinstance(get_hash(123), str)

def test_explorer_init():
    e = Explorer(app='notepad')
    assert e.app == 'notepad'
    assert e.shell is False

@patch('subprocess.Popen')
def test_explorer_call(mock_popen):
    e = Explorer(app='notepad')
    e('test.txt')
    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    assert args[0] == ['notepad', 'test.txt']

def test_browser_write_temp():
    b = Browser()
    content = "<html><body>Test</body></html>"
    path = b._write_temp(content, suffix='.html')
    
    assert os.path.exists(path)
    assert path.suffix == '.html'
    assert open(path, encoding='utf-8').read() == content
    
    # Cleanup
    os.remove(path)

def test_browser_to_file_basic():
    b = Browser()
    
    # 1. Test URL
    url = "https://www.google.com"
    assert b.to_file(url) == url
    
    # 2. Test existing file
    cur_file = __file__
    assert b.to_file(cur_file) == cur_file
    
    # 3. Test string to temp file
    content = "Hello World"
    path = b.to_file(content)
    assert os.path.exists(path)
    assert path.endswith('.txt')
    assert open(path, encoding='utf-8').read() == content
    os.remove(path)

def test_browser_to_file_html_detection():
    b = Browser()
    content = "  <html><body>Test</body></html>  "
    path = b.to_file(content)
    assert path.endswith('.html')
    os.remove(path)

def test_browser_to_file_mock_objects():
    b = Browser()
    
    # Mock object with to_html (like pandas)
    mock_df = MagicMock()
    mock_df.to_html.return_value = "<table></table>"
    path = b.to_file(mock_df)
    assert path.endswith('.html')
    assert open(path, encoding='utf-8').read() == "<table></table>"
    os.remove(path)
    
    # Mock object with render (like pyecharts)
    mock_chart = MagicMock()
    del mock_chart.to_html  # 确保不触发 to_html 逻辑
    def fake_render(path):
        with open(path, 'w') as f: f.write("chart")
    mock_chart.render.side_effect = fake_render
    path = b.to_file(mock_chart)
    assert path.endswith('.html')
    assert open(path, encoding='utf-8').read() == "chart"
    os.remove(path)

@patch('webbrowser.open')
def test_browser_call_default(mock_web_open):
    b = Browser(app='webbrowser')
    b("test content")
    mock_web_open.assert_called_once()
    # Check if the argument is a path
    args, _ = mock_web_open.call_args
    assert os.path.exists(args[0])
    os.remove(args[0])

@patch('subprocess.Popen')
def test_browser_call_custom_app(mock_popen):
    b = Browser(app='chrome')
    b("test content")
    mock_popen.assert_called_once()
    args, _ = mock_popen.call_args
    assert args[0][0] == 'chrome'
    assert os.path.exists(args[0][1])
    os.remove(args[0][1])

@patch('subprocess.Popen')
def test_view_files(mock_popen):
    from pyxllib.prog.browser import view_files
    view_files('notepad', "content 1", "content 2", name="multi_test")
    
    mock_popen.assert_called_once()
    args, _ = mock_popen.call_args
    # args[0] is ['notepad', 'path1', 'path2']
    assert args[0][0] == 'notepad'
    assert len(args[0]) == 3
    for path in args[0][1:]:
        assert os.path.exists(path)
        os.remove(path)

def test_browser_detect_logic():
    b = Browser()
    # On different platforms it might return different things, but it shouldn't be None
    assert b.app is not None

if __name__ == "__main__":
    pytest.main([__file__])
