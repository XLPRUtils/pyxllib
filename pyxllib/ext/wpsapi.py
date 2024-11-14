#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Date   : 2024/07/31

import time
import json
from pathlib import Path
from unittest.mock import MagicMock
import traceback
import datetime
import tempfile

import requests

from DrissionPage import ChromiumPage
from DrissionPage._pages.chromium_tab import ChromiumTab


def format_exception(e, mode=3):
    if mode == 1:
        # 仅获取异常类型的名称
        text = ''.join(traceback.format_exception_only(type(e), e)).strip()
    elif mode == 2:
        # 获取异常类型的名称和附加的错误信息
        text = f"{type(e).__name__}: {e}"
    elif mode == 3:
        text = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    else:
        raise ValueError
    return text


def get_dp_tab(dp_page=None):
    """ 智能获取一个标签页tab

    :param dp_page:
        默认None, 返回默认的page，一般就是当前页面
        True, 新建一个page
        str, 新建一个对应url的page
        func(tab), 通过规则筛选tab，返回符合条件的第1个tab，否则新建一个tab
    """

    if isinstance(dp_page, ChromiumPage):
        return dp_page.latest_tab
    elif isinstance(dp_page, ChromiumTab):
        return dp_page
    elif callable(dp_page):
        page0 = ChromiumPage()
        for tab in page0.get_tabs():
            if dp_page(tab):
                return tab
        return page0.new_tab()
    elif dp_page is True:
        return ChromiumPage().new_tab()
    elif isinstance(dp_page, str):
        return ChromiumPage().new_tab(dp_page)
    else:
        return ChromiumPage().latest_tab


class WpsApi:
    """ 基于wps给的一些基础api，进行了一定的增强封装

    官方很多功能都没有提供，所以有些功能都是做了一些"骚操作"的
    以及对高并发稳定性的考虑等
    """

    def __init__(self, token, *, name=None, base_url=None, logger=None,
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
        self.token = token
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
        """
        reply = self.post('script/evaluate', {
            'file_id': str(file_id),
            'code': code,
        })

        if return_mode == 'json':
            return reply

        if 'error' in reply:
            raise Exception('evaluate script failed: {}\nstack: {}'.format(reply['error'], reply['stack']))

        return reply['return']


class WpsGroupApi(WpsApi):
    """ 团队版的api
    """

    def __init__(self, token, *, group_id=None, group_folder_id=None, **kwargs):
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


class WpsScriptApi:
    """ wps的"脚本令牌"调用模式 """


if __name__ == '__main__':
    pass
