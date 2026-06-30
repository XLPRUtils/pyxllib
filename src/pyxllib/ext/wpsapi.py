#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Date   : 2024/07/31

import os
import re
import time

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


class WpsScriptSoftTimeoutError(TimeoutError):
    """WPS 脚本调用超过业务软阈值。

    WPS sync_task 偶尔会 HTTP 200 返回，但表格侧效果不完整。对表格批量读写，
    超过阈值后由上层按更小批次重试，比继续当成功更安全。
    """

    def __init__(self, *, func_name=None, elapsed_seconds, threshold_seconds):
        self.func_name = func_name
        self.elapsed_seconds = elapsed_seconds
        self.threshold_seconds = threshold_seconds
        super().__init__(
            f"WPS脚本调用超过软超时阈值：func={func_name or '<script>'}, "
            f"elapsed={elapsed_seconds:.2f}s, threshold={threshold_seconds:.2f}s"
        )


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

    def post_request(self, url, payload, *, timeout=None, attempts=None):
        """
        发送 POST 请求到指定的 URL 并返回响应结果
        """
        attempts = max(1, int(attempts or os.getenv('WPS_SCRIPT_API_RETRY_COUNT') or '5'))
        timeout = float(timeout or os.getenv('WPS_SCRIPT_API_TIMEOUT_SECONDS') or '120')
        delay = float(os.getenv('WPS_SCRIPT_API_RETRY_DELAY_SECONDS') or '5')
        last_error = None

        for attempt in range(1, attempts + 1):
            try:
                resp = requests.post(url, json=payload, headers=self.headers, timeout=timeout)
                resp.raise_for_status()  # 如果请求失败会抛出异常
                return resp.json()
            except requests.exceptions.RequestException as e:
                status_code = getattr(getattr(e, 'response', None), 'status_code', None)
                if status_code is not None and 400 <= int(status_code) < 500:
                    raise
                last_error = e
                if attempt >= attempts:
                    break
                print(f"WPS脚本请求失败，准备重试 {attempt}/{attempts}: {e}")
                time.sleep(delay * attempt)

        raise RuntimeError(f"WPS脚本请求失败，已重试{attempts}次: {last_error}") from last_error

    @staticmethod
    def _unwrap_script_result(result):
        """还原 AirScript API 包装结果，并把脚本内异常转成 Python 异常。"""
        if isinstance(result, dict) and result.get('__wps_call_ok') is False:
            func_name = result.get('funcName') or '<unknown>'
            message = result.get('message') or result.get('error') or '未知 AirScript 错误'
            stack = result.get('stack') or ''
            detail = f"WPS脚本函数执行失败：{func_name}: {message}"
            if stack:
                detail += f"\n{stack}"
            raise RuntimeError(detail)
        if isinstance(result, dict) and result.get('__wps_call_ok') is True:
            return result.get('value')
        return result

    @staticmethod
    def _check_sync_script_response(res):
        """检查 WPS sync_task 的返回体，避免 HTTP 200 但脚本实际失败被当成成功。"""
        if not isinstance(res, dict):
            raise RuntimeError(f"WPS脚本同步执行返回结构异常: {res}")
        status = res.get('status')
        error = res.get('error') or res.get('errmsg') or res.get('message')
        if error or (status and status != 'finished'):
            detail = repr(res)
            if len(detail) > 2000:
                detail = detail[:2000] + '...'
            raise RuntimeError(f"WPS脚本同步执行失败: status={status!r}, error={error!r}, response={detail}")

    @staticmethod
    def _resolve_soft_timeout_seconds(soft_timeout_seconds, *, default=None):
        if soft_timeout_seconds is None:
            soft_timeout_seconds = default
        if soft_timeout_seconds is None:
            return None
        seconds = float(soft_timeout_seconds)
        return seconds if seconds > 0 else None

    @staticmethod
    def _table_soft_timeout_seconds():
        return float(os.getenv('WPS_TABLE_SOFT_TIMEOUT_SECONDS') or '25')

    @staticmethod
    def _offset_cell_rows(start_cell, row_offset):
        def repl(m):
            return str(int(m.group()) + row_offset)

        return re.sub(r'\d+$', repl, start_cell)

    def run_script(self, script_id=None, context_argv=None, sync=True, *, soft_timeout_seconds=None):
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
        threshold = self._resolve_soft_timeout_seconds(soft_timeout_seconds)
        start_time = time.perf_counter()
        res = self.post_request(
            url,
            payload,
            timeout=threshold,
            attempts=1 if threshold else None,
        )
        elapsed = time.perf_counter() - start_time
        if sync:
            self._check_sync_script_response(res)
            if not isinstance(res, dict) or 'data' not in res or 'result' not in (res.get('data') or {}):
                raise RuntimeError(f"WPS脚本同步执行返回结构异常: {res}")
            if threshold and elapsed > threshold:
                func_name = (context_argv or {}).get('funcName')
                raise WpsScriptSoftTimeoutError(
                    func_name=func_name,
                    elapsed_seconds=elapsed,
                    threshold_seconds=threshold,
                )
            return self._unwrap_script_result(res['data']['result'])
        else:
            return res

    def __2_封装的更高级的接口(self):
        """ 这系列的功能需要配套这个框架范式使用：
        https://github.com/XLPRUtils/pyxllib/blob/master/pyxllib/text/airscript.js
        """
        pass

    def run_func(self, func_name, *args, soft_timeout_seconds=None):
        """ 我自己常用的jsa框架，jsa那边已经简化了对接模式，所以一般都只用这个高级的接口即可
        （旧函数名run_script2不再使用）
        """
        return self.run_script(
            self.default_script_id,
            context_argv={'funcName': func_name, 'args': args},
            soft_timeout_seconds=soft_timeout_seconds,
        )

    def write_arr(self, rows, start_cell, batch_size=None, *, soft_timeout_seconds=None, adaptive=True):
        """ 把一个二维数组数据写入表格

        :param rows: 一个n*m的数据
        :param start_cell: 写入的起始位置，例如'A1'，也可以使用Sheet1!A1的格式表示具体的表格位置
        :param batch_size: 为了避免一次写入内容过多，超时写入失败，可以分成多批运行
            这里写每批的数据行数
            默认表示一次性全部提交
        :return:
        """
        if not rows:
            return
        if batch_size is None:
            batch_size = len(rows)  # 如果未指定批次大小，一次性写入所有行
        threshold = self._resolve_soft_timeout_seconds(
            soft_timeout_seconds,
            default=self._table_soft_timeout_seconds(),
        )

        def write_segment(row_offset, segment):
            current_cell = self._offset_cell_rows(start_cell, row_offset)
            try:
                self.run_func('writeArrToSheet', segment, current_cell, soft_timeout_seconds=threshold)
            except Exception:
                if adaptive and len(segment) > 1:
                    mid = len(segment) // 2
                    write_segment(row_offset, segment[:mid])
                    write_segment(row_offset + mid, segment[mid:])
                    return
                raise

        for start in range(0, len(rows), batch_size):
            end = start + batch_size
            write_segment(start, rows[start:end])

    def __3_增删改查(self):
        """ 表格数据很多概念跟sql数据库是类似的，也有增删改查系列的功能需求 """
        pass

    @staticmethod
    def _normalize_sql_select_data(data, fields=None):
        """ WPS 的 sqlSelect 在整列为空时，偶尔会返回长度更短的数组。

        这里把各列统一补齐到相同长度，避免 pandas DataFrame 构造时报
        ``All arrays must be of the same length``。
        """
        if not isinstance(data, dict):
            return data

        normalized = {}
        keys = list(fields or data.keys())
        for key in data.keys():
            if key not in keys:
                keys.append(key)

        list_lengths = [len(v) for v in data.values() if isinstance(v, list)]
        if not list_lengths:
            return data
        max_len = max(list_lengths)

        for key in keys:
            value = data.get(key, [])
            if isinstance(value, list):
                if len(value) < max_len:
                    value = value + [None] * (max_len - len(value))
            elif max_len:
                value = [value] + [None] * (max_len - 1)
            normalized[key] = value

        return normalized

    def sql_select(self, sheet_name, fields,
                   data_row=0,
                   filter_empty_rows=True, *,
                   return_mode='pd',
                   field_batch_size=None,
                   soft_timeout_seconds=None,
                   adaptive=True) -> pd.DataFrame:
        """ 获取某张sheet表格数据

        :param sheet_name: sheet表名
        :param list[str] fields: 字段名列表
        :param int data_row: 数据起始行，详细用法见sqlSelect
        :param return_mode: 'pd' or 'json'
        """
        threshold = self._resolve_soft_timeout_seconds(
            soft_timeout_seconds,
            default=self._table_soft_timeout_seconds(),
        )
        if field_batch_size is None and not filter_empty_rows:
            field_batch_size = int(os.getenv('WPS_SQL_SELECT_FIELD_BATCH_SIZE') or '20')

        def select_fields(sub_fields):
            try:
                return self.run_func(
                    'sqlSelect',
                    sheet_name,
                    sub_fields,
                    data_row,
                    filter_empty_rows,
                    soft_timeout_seconds=threshold,
                )
            except Exception:
                if adaptive and not filter_empty_rows and len(sub_fields) > 1:
                    mid = len(sub_fields) // 2
                    data1 = select_fields(sub_fields[:mid])
                    data2 = select_fields(sub_fields[mid:])
                    merged = {}
                    if isinstance(data1, dict):
                        merged.update(data1)
                    if isinstance(data2, dict):
                        merged.update(data2)
                    return merged
                raise

        if field_batch_size and len(fields) > field_batch_size and not filter_empty_rows:
            data = {}
            for start in range(0, len(fields), field_batch_size):
                part = select_fields(fields[start:start + field_batch_size])
                if isinstance(part, dict):
                    data.update(part)
                else:
                    raise RuntimeError(f"WPS sqlSelect 分批返回结构异常: {part}")
        else:
            data = select_fields(fields)
        data = self._normalize_sql_select_data(data, fields)
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
