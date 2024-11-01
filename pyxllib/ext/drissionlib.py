#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/04/12
import time

from pyxllib.prog.pupil import check_install_package

check_install_package('DrissionPage')

import re
from urllib.parse import unquote

from deprecated import deprecated
import DrissionPage
from DrissionPage import ChromiumPage, Chromium
from DrissionPage._pages.chromium_base import ChromiumBase
from DrissionPage._pages.chromium_tab import ChromiumTab
from DrissionPage._base.base import BasePage, BaseElement

from pyxllib.prog.pupil import inject_members
from pyxllib.text.pupil import strfind
from pyxllib.file.specialist import GetEtag


@deprecated(reason='get_dp_page逻辑不太对，请换用get_dp_tab')
def get_dp_page(dp_page=None) -> 'XlPage':
    """

    :param dp_page:
        默认None, 返回默认的page，一般就是当前页面
        True, 新建一个page
        str, 新建一个对应url的page
        func(tab), 通过规则筛选tab，返回符合条件的第1个tab，否则新建一个tab
    """

    if isinstance(dp_page, Chromium):
        return dp_page
    elif isinstance(dp_page, ChromiumTab):
        return dp_page.page
    elif callable(dp_page):
        page0 = Chromium()
        for tab in page0.get_tabs():
            if dp_page(tab):
                return tab.page
        return page0.new_tab().page
    elif dp_page is True:
        return Chromium().new_tab().page
    elif isinstance(dp_page, str):
        return Chromium().new_tab(dp_page).page
    else:
        return Chromium()


def get_dp_tab(dp_tab=None) -> 'XlTab':
    """ 智能获取一个标签页tab

    :param dp_tab:
        默认None, 返回默认的tab，一般就是当前页面
        True, 新建一个tab
        str, 新建一个对应url的tab，如果发现已存在的tab中已经有该url，则直接复用
        func(tab), 通过规则筛选tab，返回符合条件的第1个tab，否则新建一个tab
    """

    if isinstance(dp_tab, Chromium):
        return dp_tab.latest_tab
    elif isinstance(dp_tab, ChromiumTab):
        return dp_tab
    elif callable(dp_tab):
        page0 = Chromium()
        for tab in page0.get_tabs():
            if dp_tab(tab):
                return tab
        return page0.new_tab()
    elif dp_tab is True:
        return Chromium().new_tab()
    elif isinstance(dp_tab, str):
        def dp_page2(tab):  # 默认开启页面复用
            if tab.url == dp_tab:
                return tab
        tab = get_dp_tab(dp_page2)
        if tab.url == 'about:blank':
            tab.get(dp_tab)
        return tab
    else:
        return Chromium().latest_tab


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
    for file in files:
        file2 = file['file'].replace('+', ' ')
        if file_name in file2:  # 但有时候'+'好像有点特别
            return file
        if file_name in re.sub(r'\s+', ' ', file2):
            return file


class XlChromiumBase(ChromiumBase):
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

    def get_download_files(self: Chromium):
        """ 获取下载列表

        :param search_name: 搜索文件名，输入该参数时，只会从上往下找到第一个匹配的文件
            否则返回一个list结构，存储下载清单里的文件
        :return:

        todo 默认应该显示是不全的，有时间可以考虑往下滑动继续检索的功能
        """
        files = []
        page2 = Chromium().new_tab('chrome://downloads/')
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

    def wait_page_not_change(self, interval=3):
        """ 等待直到页面内容不再变化

        :param interval: 时间间隔，需要判断当前内容和interval秒后的内容，看内容是否欧发生改变
        """
        last_html, last_etag = None, None
        while True:
            html = self.html
            etag = GetEtag.from_text(html)
            if etag == last_etag:
                break

            last_html, last_etag = html, etag
            time.sleep(interval)
        return last_html

    def action_type(self, ele, text, clear=True):
        """ 基于action实现的重写入，常用于日期相关操作
        因为很多日期类组件，直接使用ele.input是不生效的，哪怕看似显示了文本，但其实并没有触发js改动，需要用动作链来实现
        """
        from DrissionPage.common import Keys
        if clear:
            self.actions.click(ele).key_down(Keys.CTRL).type('a').key_up(Keys.CTRL).type(text)
        else:
            self.actions.click(ele).type(text)


inject_members(XlChromiumBase, ChromiumBase)


class XlPage(XlChromiumBase, ChromiumTab, Chromium):
    """ 只作为一个类型标记，无实质功能。在猴子补丁背景下，让ide能正确跳转函数定义。 """
    pass


class XlTab(XlChromiumBase, ChromiumTab, Chromium):
    pass


def wait_page_not_change(page, interval=3):
    page.wait_page_not_change(interval)
