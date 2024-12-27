#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Date   : 2024/07/31

import os
import requests


class WpsOnlineBook:
    """ wps的"脚本令牌"调用模式

    官方文档：https://airsheet.wps.cn/docs/apitoken/api.html
    """

    def __init__(self, file_id=None, script_id=None, *, token=None):
        self.headers = {
            'Content-Type': "application/json",
            'AirScript-Token': token or os.getenv('WPS_SCRIPT_TOKEN', ''),
        }
        self.default_file_id = file_id
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

    def run_script(self, script_id=None, file_id=None, context_argv=None, sync=True):
        """
        执行 WPS 脚本并返回执行结果

        :param file_id: 文件 ID
            虽然提供了file_id，但并不支持跨文件调用as脚本
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
        file_id = file_id or self.default_file_id
        url = f"https://www.kdocs.cn/api/v3/ide/file/{file_id}/script/{script_id}/{'sync_task' if sync else 'task'}"
        payload = {
            "Context": {'argv': context_argv or {}}
        }
        res = self.post_request(url, payload)
        if sync:
            return res['data']['result']
        else:
            return res

    def run_script2(self, func_name, *args):
        """ 我自己常用的jsa框架，jsa那边已经简化了对接模式，所以一般都只用这个高级的接口即可

        配合这个框架范式使用：https://github.com/XLPRUtils/pyxllib/blob/master/pyxllib/text/airscript.js
        """
        return self.run_script(self.default_script_id, context_argv={'funcName': func_name, 'args': args})


WpsOnlineScriptApi = WpsOnlineBook

if __name__ == '__main__':
    pass
