#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/19

""" 问卷星 相关工具 """

import os
import io
import time

from loguru import logger
from DrissionPage import Chromium
import pandas as pd


class WjxWeb:
    """ 问卷星网页的爬虫 """

    def __init__(self, url):
        self.tab = Chromium().new_tab(url)
        self.login()

    def login(self):
        tab = self.tab

        if tab.url.startswith('https://www.wjx.cn/wjx/activitystat/resultlimit.aspx'):
            logger.info(1)
            tab('tag:a@@text():登录').click()

        if tab.url.lower().startswith('https://www.wjx.cn/login.aspx'):
            logger.info(2)
            tab('tag:input@@name=UserName').input(os.getenv('WJX_USERNAME'), clear=True)
            tab('tag:input@@name=Password').input(os.getenv('WJX_PASSWORD'), clear=True)
            tab('tag:input@@type=submit').click()

    def close_if_not_unique(self):
        """ 检查同网站的tab数量，如果不唯一则关闭当前页面 """
        if len(Chromium().get_tabs(url='www.wjx.cn')) > 1:
            self.tab.close()

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
        table_html = self.tab('tag:table').html
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
        """获得当前页面的表格数据，如果 all_pages 为 True，则下载所有页面的数据"""
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
