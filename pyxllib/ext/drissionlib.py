#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/04/12

from pyxllib.prog.pupil import check_install_package

check_install_package('DrissionPage')

import re
from urllib.parse import unquote

from DrissionPage import ChromiumPage


def get_latest_not_dev_tab(page=None):
    """ 开发工具本身也会算一个tab，这个函数返回最新的一个不是开发工具的tab """
    if page is None:
        page = ChromiumPage()
    tabs = page.get_tabs()
    for tab in tabs:
        if tab.url.startswith('devtools://'):
            continue
        return tab


def set_input_text(input_ele, text):
    """ 因为input输入框可能原本就自带了内容，为了不重复输入，先清空再输入 """
    input_ele.clear()
    input_ele.input(text)


def get_download_files(page=None):
    """ 获取下载列表

    :param search_name: 搜索文件名，输入该参数时，只会从上往下找到第一个匹配的文件
        否则返回一个list结构，存储下载清单里的文件
    :return:

    todo 默认应该显示是不全的，有时间可以考虑往下滑动继续检索的功能
    """
    if page is None:
        page = ChromiumPage()

    files = []
    page2 = page.new_tab('chrome://downloads/')
    items = page2('tag:downloads-manager', timeout=1).sr('#mainContainer')('#downloadsList').eles('tag:downloads-item')
    for item in items:
        loc = unquote(item.sr('tag:img').attr('src'))
        file = re.search(r'path=(.+?)(&scale=\d+x)?$', loc).group(1)

        files.append({
            'file': file,
            'url': unquote(item.sr('#url').attr('href'))
        })

    page2.close()

    return files


def search_download_file(file_name):
    files = get_download_files()
    for file in files:
        if file_name in file['file']:
            return file
