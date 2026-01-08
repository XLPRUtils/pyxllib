#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Date   : 2024/07/31

import os
import re

from pyxllib.prog.lazyimport import lazy_import

try:
    import requests
except ModuleNotFoundError:
    requests = lazy_import('requests')

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import('pandas')

try:
    from DrissionPage import Chromium
except ModuleNotFoundError:
    Chromium = lazy_import('from DrissionPage import Chromium')


class WpsOnlineBook:
    """ wps的"脚本令牌"调用模式

    官方文档：https://airsheet.wps.cn/docs/apitoken/api.html
    """

    def __1_基础功能(self):
        pass

    def __init__(self, book_id, script_id=None, *, token=None):
        self.headers = {
            'Content-Type': "application/json",
            'AirScript-Token': token or os.getenv('WPS_SCRIPT_TOKEN', ''),
        }
        self.book_id = book_id
        self.default_script_id = script_id

    def post_request(self, url, payload):
        """
        发送 POST 请求到指定的 URL 并返回响应结果
        """
        try:
            resp = requests.post(url, json=payload, headers=self.headers)
            resp.raise_for_status()  # 如果请求失败会抛出异常
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return None

    def run_script(self, script_id=None, context_argv=None, sync=True):
        """ 原本执行 WPS 脚本并返回执行结果

        :param script_id: 脚本 ID
        :param context_argv: 脚本参数 (可选)
            context本来能支持这些参数的：
                dict argv: 传入的上下文参数对象，比如传入{name: 'xiaomeng', age: 18}， 在 AS 代码中可通过Context.argv.name获取到传入的值
                str sheet_name: et,ksheet 运行时所在表名
                str range: et,ksheet 运行时所在区域，例如$B$156
                str link_from: et,ksheet 点击超链接所在单元格
                str db_active_view: db 运行时所在 view 名
                str db_selection: db 运行时所在选区
            但是，sheet_name, range等，并看不到对运行代码有什么实质影响，不影响active，而且as里也引用不了sheet_name等值
                所以退化，简化为只要传入content_argv参数就行
        :param sync:
            True, 同步运行
            False, 异步运行

        这个接口跟普通as一样，运行有30秒时限
        """
        url = f"https://www.kdocs.cn/api/v3/ide/file/{self.book_id}/script/{script_id}/{'sync_task' if sync else 'task'}"
        payload = {
            "Context": {'argv': context_argv or {}}
        }
        res = self.post_request(url, payload)
        if sync:
            return res['data']['result']
        else:
            return res

    def __2_封装的更高级的接口(self):
        """ 这系列的功能需要配套这个框架范式使用：
        https://github.com/XLPRUtils/pyxllib/blob/master/pyxllib/text/airscript.js
        """
        pass

    def run_func(self, func_name, *args):
        """ 我自己常用的jsa框架，jsa那边已经简化了对接模式，所以一般都只用这个高级的接口即可
        （旧函数名run_script2不再使用）
        """
        return self.run_script(self.default_script_id, context_argv={'funcName': func_name, 'args': args})

    def write_arr(self, rows, start_cell, batch_size=None):
        """ 把一个二维数组数据写入表格

        :param rows: 一个n*m的数据
        :param start_cell: 写入的起始位置，例如'A1'，也可以使用Sheet1!A1的格式表示具体的表格位置
        :param batch_size: 为了避免一次写入内容过多，超时写入失败，可以分成多批运行
            这里写每批的数据行数
            默认表示一次性全部提交
        :return:
        """
        if batch_size is None:
            batch_size = len(rows)  # 如果未指定批次大小，一次性写入所有行

        def func(m):
            return str(int(m.group()) + batch_size)

        current_cell = start_cell
        for start in range(0, len(rows), batch_size):
            end = start + batch_size
            batch_rows = rows[start:end]
            self.run_func('writeArrToSheet', batch_rows, current_cell)
            current_cell = re.sub(r'\d+$', func, current_cell)

    def __3_增删改查(self):
        """ 表格数据很多概念跟sql数据库是类似的，也有增删改查系列的功能需求 """
        pass

    def sql_select(self, sheet_name, fields,
                   data_row=0,
                   filter_empty_rows=True, *,
                   return_mode='pd') -> pd.DataFrame:
        """ 获取某张sheet表格数据

        :param sheet_name: sheet表名
        :param list[str] fields: 字段名列表
        :param int data_row: 数据起始行，详细用法见sqlSelect
        :param return_mode: 'pd' or 'json'
        """
        data = self.run_func('sqlSelect', sheet_name, fields, data_row, filter_empty_rows)
        if return_mode == 'json':
            return data
        elif return_mode == 'pd':
            return pd.DataFrame(data)

    def __4_其他(self):
        pass

    def browser_refresh(self, duration=10):
        """ 使用 dp 爬虫打开在线表格文件，等待指定时间后关闭，相当于通过浏览器刷新下表格
        :param duration: 打开后等待秒数，默认 10 秒
        """
        browser = Chromium()
        tab = browser.new_tab(f'https://www.kdocs.cn/l/{self.book_id}')
        tab.wait.doc_loaded()
        tab.wait(duration)
        tab.close()


if __name__ == '__main__':
    wb = WpsOnlineBook('chQzbASABLcN')
    wb.browser_refresh()
