#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/19

""" 问卷星 相关工具 """

import os
import io
import time

from pyxllib.prog.lazyimport import lazy_import

try:
    from loguru import logger
except ModuleNotFoundError:
    logger = lazy_import('from loguru import logger')

try:
    from DrissionPage import Chromium
except ModuleNotFoundError:
    Chromium = lazy_import('from DrissionPage import Chromium')

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import('pandas')

from pyxllib.ext.drissionlib import DpWebBase


class WjxWeb(DpWebBase):
    """ 问卷星网页的爬虫 """

    HOME_URL = 'https://www.wjx.cn/'
    LOGIN_URL = 'https://www.wjx.cn/login.aspx'
    RESULT_LIMIT_URL = 'https://www.wjx.cn/wjx/activitystat/resultlimit.aspx'

    def __init__(self, url=HOME_URL):
        super().__init__(url)
        self.login()
        if url and url != self.LOGIN_URL and self.tab.url != url:
            self.tab.get(url)

    def _ensure_credentials(self):
        username = os.getenv('WJX_USERNAME')
        password = os.getenv('WJX_PASSWORD')
        if not username or not password:
            raise RuntimeError('未设置环境变量 WJX_USERNAME / WJX_PASSWORD，无法自动登录问卷星')
        return username, password

    def _wait_login_result(self, timeout=60, interval=1):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.tab.url.startswith(self.LOGIN_URL):
                return True
            time.sleep(interval)
        return False

    def login(self, wait_timeout=60):
        tab = self.tab

        if tab.url.startswith(self.RESULT_LIMIT_URL):
            tab('t:a@@text():登录').click()
            time.sleep(2)

        if tab.url.startswith(self.LOGIN_URL):
            username, password = self._ensure_credentials()
            time.sleep(2)
            tab('t:input@@name=UserName').input(username, clear=True)
            time.sleep(2)
            tab('t:input@@name=Password').input(password, clear=True)
            time.sleep(2)
            tab('t:input@@type=submit').click()
            if not self._wait_login_result(timeout=wait_timeout):
                raise RuntimeError('问卷星登录后仍停留在登录页，可能需要人工完成验证码或额外验证')

    def get_page_num(self):
        """
        返回当前页编号和总页数 (idx, num)。
        """
        idx, num = map(int, self.tab('tag:span@@class=paging-num').text.split('/'))
        return idx, num

    def prev_page(self):
        self.tab('tag:a@@class=go-pre').click()

    def next_page(self):
        self.tab('tag:a@@class=go-next').click()

    def _parse_table(self):
        """处理并解析网页中的表格数据"""
        self.tab.find_ele_with_refresh(f't:table')
        table_html = self.tab.eles('t:table')[-1].html
        df = pd.read_html(io.StringIO(table_html))[0]  # 读取表格
        df.columns = [col.replace('\ue645', '') for col in df.columns]
        # "星标"的内容特殊字符
        df.replace('\ue66b', '', regex=True, inplace=True)
        # "操作"的内容特殊字符
        df.replace('\ue6a3\ue6d4', '', regex=True, inplace=True)
        return df

    def set_num_of_page(self, num_of_page):
        """ 查看数据页面，设置每页显示多少条记录 """
        select = self.tab('tag:span@@text():每页显示').next('tag:select')
        select.click()
        opt = select(f'tag:option@@text()={num_of_page}')
        if opt.attr('selected') != 'selected':
            opt.click()
        else:
            select.click()

    def get_df(self, all_pages=False):
        """ 获得当前页面的表格数据，如果 all_pages 为 True，则下载所有页面的数据 """
        # 初始化DataFrame列表，用于存储每页的数据
        dfs = [self._parse_table()]  # 获取当前页面的数据

        # 如果需要下载所有页面数据
        if all_pages:
            current_idx, total_pages = self.get_page_num()
            while current_idx < total_pages:
                self.next_page()  # 翻到下一页
                time.sleep(2)
                dfs.append(self._parse_table())  # 获取并处理新一页的数据
                current_idx, total_pages = self.get_page_num()  # 更新页码信息

        # 将所有数据合并为一个DataFrame
        final_df = pd.concat(dfs, ignore_index=True) if all_pages else dfs[0]
        return final_df
