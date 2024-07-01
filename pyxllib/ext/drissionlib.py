#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/04/12

from pyxllib.prog.pupil import check_install_package

check_install_package('DrissionPage')

import re
from urllib.parse import unquote

import DrissionPage
from DrissionPage import ChromiumPage
from DrissionPage._base.base import BasePage, BaseElement

from pyxllib.prog.pupil import inject_members
from pyxllib.text.pupil import strfind

def get_latest_not_dev_tab(page=None):
    """ 开发工具本身也会算一个tab，这个函数返回最新的一个不是开发工具的tab """
    if page is None:
        page = ChromiumPage()
    tabs = page.get_tabs()
    for tab in tabs:
        if strfind(tab.url, ['devtools://', 'chrome-extension://']) != -1:
            continue
        return tab


def set_input_text(input_ele, text):
    """ 因为input输入框可能原本就自带了内容，为了不重复输入，先清空再输入 """
    input_ele.clear()
    input_ele.input(text)


def search_download_file(file_name):
    file_name = file_name.replace(':', '_')
    files = ChromiumPage().get_download_files()
    for file in files:
        if file_name in file['file']:  # 正常情况下的匹配
            return file
        if file_name in file['file'].replace('+', ' '):  # 但有时候'+'好像有点特别
            return file


class XlBasePage(BasePage):
    def get2(self, url, show_errmsg=False, retry=None, interval=None):
        """
        240418周四21:57，DrissionPage-4.0.4.21 官方自带page.get，有时候会有bug，不会实际刷新url，这里加个代码进行fix
        """
        old_url = self.url
        if old_url == url:
            # page.refresh()
            return

        ele = self.active_ele
        self.get(url, show_errmsg=show_errmsg, retry=retry, interval=interval)
        try:  # 如果新页面获取成功，理论上旧的ele会失效
            ele
            self.refresh()  # 如果不报错，这里网站要强制更新
            return self
        except DrissionPage.errors.ElementLostError:
            return self

    def get_download_files(self):
        """ 获取下载列表

        :param search_name: 搜索文件名，输入该参数时，只会从上往下找到第一个匹配的文件
            否则返回一个list结构，存储下载清单里的文件
        :return:

        todo 默认应该显示是不全的，有时间可以考虑往下滑动继续检索的功能
        """
        files = []
        page2 = self.new_tab('chrome://downloads/')
        items = page2('tag:downloads-manager', timeout=1).sr('#mainContainer')('#downloadsList').eles(
            'tag:downloads-item')
        for item in items:
            loc = unquote(item.sr('tag:img').attr('src').replace('+', ' '))
            file = re.search(r'path=(.+?)(&scale=(\d+(\.\d+)?)x)?$', loc).group(1)

            files.append({
                'file': file,
                'url': unquote(item.sr('#url').attr('href'))
            })

        page2.close()

        return files


inject_members(XlBasePage, BasePage)


class XlBaseElement(BaseElement):
    def input2(self, vals, clear=False, by_js=False):
        self.clear()
        self.input(vals)


inject_members(XlBaseElement, BaseElement)
