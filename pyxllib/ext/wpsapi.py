#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Date   : 2024/07/31
import os
import time
import json
from pathlib import Path
from unittest.mock import MagicMock
import traceback
import datetime
import tempfile
import textwrap
import re

from loguru import logger
import requests
import pandas as pd

from pyxllib.prog.pupil import format_exception
from pyxllib.ext.drissionlib import get_dp_tab


class WpsApi:
    """ 基于wps给的一些基础api，进行了一定的增强封装

    官方很多功能都没有提供，所以有些功能都是做了一些"骚操作"的
    以及对高并发稳定性的考虑等
    """

    def __init__(self, token=None, *, name=None, base_url=None, logger=None,
                 folder_id=None):
        """
        :param str token: 登录的token
        :param str name: 记一个账号昵称，方便区分，默认会有token后6位
        :param str base_url: 基础url，一般不用改
        :param logger: 日志记录器，可以不输入，默认置空不记录
        :param folder_id: 默认的文件夹id，有些文件操作相关的功能，需要放到指定目录下
        """
        # 这里登录获得的是自己账号的token
        # Get the token from https://solution.wps.cn/weboffice-go-sdk/api/script/token
        self.token = token or os.getenv('WPS_API_TOKEN')
        self.name = name or self.token[-6:]
        self.base_url = base_url or 'https://solution.wps.cn/weboffice-go-sdk/api'
        self.default_template = 'https://kdocs.cn/l/cjVHm9Zy9jU1'  # 空白表格，默认的一个表格模板文件，用于创建新文件、测试等

        self.logger = logger or MagicMock()
        self._folder_id = folder_id or None

    def __1_基本请求(self):
        pass

    def request(self, method, endpoint, data=None, **kwargs):
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json',
        }
        resp = requests.request(method, f"{self.base_url}/{endpoint}", headers=headers,
                                data=json.dumps(data), **kwargs)
        return resp.json()

    def post(self, endpoint, data=None, **kwargs):
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json',
        }
        resp = requests.post(f"{self.base_url}/{endpoint}", headers=headers,
                             data=json.dumps(data), **kwargs)
        return resp.json()

    def put(self, endpoint, data=None, **kwargs):
        """ 主要是上传文件用的 """
        headers = {  # 注意这个相比post删掉了'Content-Type'
            'Authorization': self.token,
        }
        resp = requests.put(f"{self.base_url}/{endpoint}", headers=headers,
                            data=data, **kwargs)
        return resp.json()

    def __2_目录操作(self):
        pass

    def create_folder(self, name):
        """ 创建目录

        wps未提供
            检索已创建成功目录id
            删除指定目录
        """
        resp_json = self.post('files/folder', {'name': name})
        # 创建成功返回：{'folder_id': 316928828842}
        # 如果文件已存在，返回的是：{'msg': '文件名冲突', 'result': 'fileNameConflict'}
        return resp_json['folder_id']

    def get_folder_id(self, check_folder=True):
        """
        :param bool check_folder: 是否严格校验缓存的目录是否有效，如果无效会自动重置
            一般情况下为了效率的话，可以不开
            为了保险稳定性的话，最好打开
        """
        if self._folder_id is None:
            # 1 看本地是否有缓存folder_id
            config_dir = Path(tempfile.gettempdir()) / 'WpsApiPresetFolders'
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / f'{self.token}.txt'
            if config_file.is_file():
                self._folder_id = config_file.read_text()

                if check_folder:
                    try:
                        file_id = self.save_as_file()
                        self.delete_file(file_id)
                        return self._folder_id
                    except ValueError as e:  # 目录可能被删掉了，还是要重新创建的
                        self._folder_id = None
                else:
                    return self._folder_id

            # 2 获得folder_id，并保存到本地
            for i in range(5):
                try:
                    now_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')  # 以当前时间点新建一个目录
                    self._folder_id = str(self.create_folder(now_tag))  # 虽然是一串数字，但大部分场合都是要当做str形式使用
                    config_file.write_text(self._folder_id)
                    break
                except KeyError as e:
                    self.logger.warning(f'wps api {self.name} 请求失败，重试第{i + 1}次，{format_exception(e)}')
                    time.sleep(2)
            else:
                raise ValueError('无法获取folder_id，命名全部冲突的概率极低，请检查是否联网')

        return self._folder_id

    @property
    def folder_id(self):
        self.get_folder_id()
        return self._folder_id

    def __3_基础功能(self):
        pass

    def save_as_file(self, src_file_id=None, dst_folder_id=None):
        """ 另存文件 """
        # 1
        if src_file_id is None:
            src_file_id = self.default_template.split('/')[-1]
        if dst_folder_id is None:
            dst_folder_id = self.folder_id

        # 2
        resp_json = self.post('files/save_as', {
            'source_fileid': str(src_file_id),
            'target_parentid': int(dst_folder_id),
        })

        # 3
        if 'new_fileid' in resp_json:
            return resp_json['new_fileid']
        else:
            traceback.print_exc()
            raise ValueError

    def delete_file(self, file_id):
        """ 这里要输入的是表格的url

        删除成功：{'result': 'ok'}
        如果不存在，返回：{'result': 'fileNotExists', 'msg': '文件(夹)不存在'}
        注意这个接口其实不支持删除目录，输入目录类id会得到：{'result': 'InvalidArgument', 'msg': '请求参数错误'}
        """
        return self.post('files/delete', {'file_id': str(file_id)})

    def download(self, source_file_id, target_file_path):
        """ 下载文件到本地

        注意使用团队版api上传的文件，如果不是.ksheet而是.xlsx等类型，这里下载会报错：{'result': 'InvalidArgument'}
        """
        resp_json = self.post('files/export_xlsx', {'file_id': str(source_file_id)})
        # 使用requests下载文件
        with requests.get(resp_json['url'], stream=True) as r:
            r.raise_for_status()
            with open(target_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    def run_airscript(self, file_id, code, return_mode=None):
        """ 执行as/jsa脚本

        :param return_mode: 返回格式
            None，提取到的结果
            json: 返回json格式

        注意这个接口使用的都是wpsapi 1.0版本
        """
        reply = self.post('script/evaluate', {
            'file_id': str(file_id),
            'code': code,
        })

        if return_mode == 'json':
            return reply

        if 'error' in reply:
            raise Exception('evaluate script failed: {}\nstack: {}'.format(reply['error'], reply['stack']))

        if 'return' not in reply:
            logger.error(reply)

        # 有运行30秒的限制，超时可能返回：{'result': 'Unavailable'}

        return reply['return']


class WpsApiTimeoutVersion(WpsApi):
    """ 特供版 """

    def request(self, method, endpoint, data=None, **kwargs):
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json',
        }
        # 重试机制
        for i in range(5):
            try:
                # 根据 method 动态选择请求方式
                resp = requests.request(method, f"{self.base_url}/{endpoint}", headers=headers,
                                        data=json.dumps(data), timeout=5, **kwargs)
                return resp.json()
            except Exception as e:
                self.logger.warning(f'wps api {self.name} 请求失败，重试第{i + 1}次，{endpoint}，{format_exception(e)}')
                time.sleep(5)  # 等待5秒后再重试
        return {'error': 'requests.exceptions.Timeout', 'stack': ''}

    def post(self, endpoint, data=None, **kwargs):
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json',
        }
        # 重试机制
        for i in range(5):
            try:
                resp = requests.post(f"{self.base_url}/{endpoint}", headers=headers,
                                     data=json.dumps(data), timeout=5, **kwargs)
                return resp.json()
            except Exception as e:
                self.logger.warning(f'wps api {self.name} 请求失败，重试第{i + 1}次，{endpoint}，{format_exception(e)}')
                time.sleep(5)  # 等待5秒后再重试
        return {'error': 'requests.exceptions.Timeout', 'stack': ''}

    def put(self, endpoint, data=None, **kwargs):
        """ 主要是上传文件用的 """
        headers = {  # 注意这个相比post删掉了'Content-Type'
            'Authorization': self.token,
        }
        # 重试机制
        for i in range(5):
            try:
                # 注意put版本相比post，不仅改了requests.put，还改了data的传输形式
                resp = requests.put(f"{self.base_url}/{endpoint}", headers=headers,
                                    data=data, timeout=5, **kwargs)
                return resp.json()
            except Exception as e:
                self.logger.warning(f'wps api {self.name} 请求失败，重试第{i + 1}次，{endpoint}，{format_exception(e)}')
                time.sleep(5)  # 等待5秒后再重试
        return {'error': 'requests.exceptions.Timeout', 'stack': ''}


class WpsOnlineWorkbook:
    def __init__(self, file_id=None, token=None):
        """
        初始化WpsOnlineWorkbook实例。

        :param file_id: 可选参数，指定要操作的文件ID。如果未提供，将创建一个新文件。
        :param token: 可选参数，用于与WPS API交互的令牌。如果未提供，将从环境变量中获取。
        """
        self.wpsapi = WpsApi(token=token)
        self.file_id = file_id or self.create()

    def create(self, src_file_id=None):
        """
        创建一个新的在线工作簿，并返回文件ID。

        :return: 新创建的文件ID
        """
        new_file_id = self.wpsapi.save_as_file(src_file_id)  # 使用默认模板创建新文件
        self.file_id = new_file_id
        return new_file_id

    def save_as_file(self, dst_folder_id=None):
        """
        将当前工作簿另存为新文件。

        :param dst_folder_id: 可选参数，指定保存文件的目标文件夹ID
        :return: 新文件的ID
        """
        new_file_id = self.wpsapi.save_as_file(self.file_id, dst_folder_id)
        return new_file_id

    def delete(self):
        """
        删除当前工作簿。
        """
        self.wpsapi.delete_file(self.file_id)

    def download(self, target_file_path):
        """
        下载当前工作簿到本地。

        :param target_file_path: 本地保存文件的路径
        """
        self.wpsapi.download(self.file_id, target_file_path)

    def run_airscript(self, code, return_mode=None):
        """
        在当前工作簿上运行AirScript代码。

        :param code: 要执行的AirScript代码
        :param return_mode: 返回格式，None表示提取结果，'json'表示返回json格式
        :return: 执行结果
        """
        from pyxllib.text.jscode import assemble_dependencies_from_jstools, remove_js_comments

        # 这里的版本默认支持扩展的js工具, 并且这套api只支持旧版的jsa1.0
        _code = code
        code = assemble_dependencies_from_jstools(code, old_jsa=True)
        code = remove_js_comments(code)
        return self.wpsapi.run_airscript(self.file_id, code, return_mode)

    def write_arr(self, rows, sheet_name, start_cell, batch_size=None):
        """
        把一个矩阵写入表格

        :param rows: 一个n*m的数据
        :param sheet_name: 目标表格名
        :param start_cell: 写入的起始位置
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
            json_data = json.dumps(batch_rows, ensure_ascii=False)
            jscode = f"""
const jsonData = {json_data}
writeArrToSheet(jsonData, Sheets('{sheet_name}').Range('{current_cell}'))
""".strip()
            current_cell = re.sub(r'\d+', func, current_cell)
            self.run_airscript(jscode)

    def get_df(self, sheet_name, fields, data_row=None):
        """ 获得表格数据

        :param sheet_name: 表格名
        :param fields: 要提取的字段名
        :param data_row: 行标记，一般是两个整数，具体用法见jsa里的接口解释
        :return: DataFrame
        """
        data_row = data_row or [0, 0]
        jscode = f"return packTableDataFields('{sheet_name}', {fields}, {data_row})"
        data = self.run_airscript(jscode)
        return pd.DataFrame(data)


class WpsGroupApi(WpsApi):
    """ 团队版的api
    """

    def __init__(self, token=None, *, group_id=None, group_folder_id=None, **kwargs):
        super().__init__(token, **kwargs)
        self.group_id = group_id
        self.group_folder_id = group_folder_id  # 团队版的默认存储目录

    def upload_file(self, local_file, remote_file_name=None, group_folder_id=None, group_id=None):
        """ 只有团队版才有上传文件的功能，但上传后的文件，和其他文件功能，爬虫的处理，都是兼容的

        :param local_file: 要上传的本地文件
        :param remote_file_name: 存储在远程的参考文件名
            注意这个是要写后缀的，否则会成为无法识别类型的文件
            写xlsx是可以的，但是一般建议是写成 .ksheet，这是官方标准的智能表格格式，目前测试也只有这种格式才支持api再重复下载
        :param group_folder_id: 团队目录id
        :param group_id: 团队id
        """
        # 1 配置参数
        local_file = Path(local_file)
        remote_file_name = remote_file_name or local_file.name
        group_folder_id = group_folder_id or self.group_folder_id
        group_id = group_id or self.group_id

        # 2 上传文件
        files = {
            "file": (remote_file_name, open(local_file, "rb"))
        }
        data = {
            "group_id": str(group_id),
            "parent_id": str(group_folder_id)
        }
        resp = self.put('files/upload', data, files=files)
        print(resp)
        # {'result': 'InvalidArguments', 'msg': "request Content-Type isn't multipart/form-data"}
        return resp.get('file_id', '')


class WpsWeb:
    """ wps网页版的爬虫 """

    def __init__(self, dp_page=None):
        """ 初始化及登录 """
        # 1 找是否已经存在页面，确定页面的复用规则，并激活窗口
        if dp_page is None:
            def dp_page(tab):
                if tab.url.startswith('https://www.kdocs.cn'):
                    return tab

        tab = get_dp_tab(dp_page)
        tab.set.activate()

        # # 2 看是否需要登录
        while not tab.url == 'https://www.kdocs.cn/latest':
            tab.get('https://www.kdocs.cn')

        self.tab = tab
        self.page = self.tab.page

    def get_token(self):
        """ 获取token的值
        需要确保账号提前有登录好
        """
        tab = self.page.new_tab('https://solution.wps.cn/weboffice-go-sdk/api/script/token')
        tab.wait.load_start()
        token = json.loads(tab('tag:pre').inner_html)['token']
        return token

    def share_file(self, file_id, link_permission=1):
        """ 分享文件

        :param link_permission:
            0，关闭分享
            1，查看权限
            2，评论权限
            3，编辑权限
        """
        # 1 访问文档
        if not str(file_id).startswith('https://www.kdocs.cn/l/'):
            file_id = f'https://www.kdocs.cn/l/{file_id}'
        tab = self.tab
        tab.get(file_id)

        # 2 获得当前文档的权限设置情况
        tab('tag:span@@text()=分享').click()
        for i in range(5):
            eles = tab.eles('tag:div@@class=kso-group-item__after')
            if len(eles) > 2:
                break
            tab.wait(1)
        if '开启后分享' in eles[0].text:
            cur_permission = 0
        else:
            eles[1].click()  # 点开按钮触发后，才能获得当前文件权限
            premission_text = eles[1]('tag:span@@class=permission-text').text
            if '可查看' in premission_text:
                cur_permission = 1
            elif '可评论' in premission_text:
                cur_permission = 2
            elif '可编辑' in premission_text:
                cur_permission = 3
            else:
                raise ValueError

        # 3 是否需要修改权限
        if cur_permission != link_permission:
            if link_permission == 0:  # 目标是关闭分享
                eles[0].click()
            else:
                # 其他都是要开启分享，如果当前没启用分享，要先启用
                if cur_permission == 0:
                    eles[0].click()
                    tab.wait(2)
                    eles[1].click()  # trick, 这句是写在if里面的，只有这种情况需要重新点链接权限，否则前面已经点开确认过
                dst_text = ['查看', '查看和评论', '编辑'][link_permission - 1]
                eles2 = tab.eles('tag:div@@class=kso-shareV3-menu-item')  # 这一组标签比较特别，需要自己遍历处理
                for ele in eles2:
                    if ele.text == dst_text:
                        # DrissionPage.errors.NoRectError: 该元素没有位置及大小。
                        ele.wait.clickable(timeout=10)
                        ele.click()
                        break


class WpsOnlineScriptApi:
    """ wps的"脚本令牌"调用模式 """

    def __init__(self, token=None):
        self.headers = {
            'Content-Type': "application/json",
            'AirScript-Token': token or os.getenv('WPS_SCRIPT_TOKEN', ''),
        }
        self.default_file_id = None

    def post_request(self, url, payload):
        """
        发送 POST 请求到指定的 URL 并返回响应结果
        """
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()  # 如果请求失败会抛出异常
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return None

    def run_script(self, script_id, file_id=None, args=None):
        """
        执行 WPS 脚本并返回执行结果

        :param file_id: 文件 ID
        :param script_id: 脚本 ID
        :param args: 脚本参数 (可选)

        todo 官方除了同步接口，还有异步接口。https://airsheet.wps.cn/docs/apitoken/api.html
        """
        file_id = file_id or self.default_file_id
        url = f"https://www.kdocs.cn/api/v3/ide/file/{file_id}/script/{script_id}/sync_task"
        payload = {
            "Context": {
                "argv": args if args else {}
            }
        }
        res = self.post_request(url, payload)
        return res['data']['result']


if __name__ == '__main__':
    pass
