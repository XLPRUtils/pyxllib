#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/10/12

"""
我们自己的一套利用后端服务进行文件同步的工具
"""

import os

from tqdm import tqdm
import requests

from pyxllib.file.specialist import XlPath, GetEtag


class SyncFileClient:
    """

    注意：该类填写的路径，一律是相对local_root、remote_root的路径，不是当前实际工作目录下的路径！
        可以通过把local_host设为空的来实现一些特殊精细化的操作
        但一般推荐还是都把local_root、remote_root设置上
    """

    def __1_basic(self):
        pass

    def __init__(self, host, token=None):
        """
        :param host: 可以只写主机名
        """
        token = token or os.getenv("XL_COMMON_SERVER_TOKEN")
        self.headers = {
            'Authorization': f'Bearer {token}',
        }
        self.local_root = None
        self.remote_root = None
        self.host = self.link_host(host)

    def link_host(self, hostname):
        from requests.exceptions import Timeout

        host = f'https://xmutpriu.com/{hostname}'
        resp = requests.get(f'{host}/common/get_local_server', headers=self.headers)
        ip = resp.json()['ip']

        if ip:
            try:
                resp = requests.get(f'{ip}/{hostname}/healthy', timeout=5)
                if resp.status_code == 200 and resp.json() == 'OK':
                    host = f'{ip}/{hostname}'
            except Timeout:
                pass
            except Exception as e:
                raise e

        remote_root = requests.get(f'{host}/common/get_wkdir', headers=self.headers).json()['wkdir']
        self.remote_root = XlPath(remote_root)

        return host

    def get_abs_local_path(self, local_path, remote_path=None):
        # 1 如果需要，从远程路径推测本地路径
        if local_path is None and remote_path is not None:
            local_path = remote_path

        # 2 本地存储的文件路径
        if self.local_root:
            return self.local_root / local_path
        else:
            return XlPath(local_path)

    def get_abs_remote_path(self, remote_path, local_path=None):
        # 1 如果需要，从本地路径推测远程路径
        if remote_path is None and local_path is not None:
            remote_path = local_path

        # 2 远程存储的文件路径
        if self.remote_root:
            return self.remote_root / remote_path
        else:
            return XlPath(remote_path)

    def __2_sync(self):
        pass

    def upload_file(self, local_file=None, remote_file=None, relpath=True):
        """
        :param str remote_file: 要存储的远程文件路径，默认可以不指定，自动通过local_file存储到对称位置

        使用示例：
        sfc = SyncFileClient('codepc_mi15')
        sfc.upload_file('abc/README.md')
        sfc.upload_file('abc/README.md', 'bcd/1.md')
        """
        file = [local_file]
        if remote_file:
            file.append(remote_file)
        return self.upload_files([file], relpath=relpath)

    def upload_files(self, files, relpath=True):
        """
        :param list[local_file<, remote_path>] files:
            remote_path可选，未写的时候会自动填充
        :return:

        todo 加进度条？不过问了下gpt，这个实现好像有点复杂，先不折腾了。本来其实就推荐单个单个文件上传，在外部进行进度展示管控。

        使用示例：
        sfc = SyncFileClient('codepc_mi15')
        print(sfc.upload_files(['data/d231120禅宗1期4阶.xlsx', 'data/d231120禅宗1期4阶2.xlsx']))
        """
        # 1 映射出目标路径
        files2 = []
        for file in files:
            if not isinstance(file, (list, tuple)):
                local_file, remote_file = file, None
            else:
                local_file, remote_file = file

            if relpath:
                local_file = self.get_abs_local_path(local_file, remote_file)
                remote_file = self.get_abs_remote_path(remote_file, local_file)

            files2.append([local_file, remote_file.as_posix()])

        files3 = [('files', (remote_file, open(local_file, 'rb'))) for local_file, remote_file in files2]

        # 4 发送请求
        resp = requests.post(f'{self.host}/common/upload_files',
                             headers=self.headers,
                             files=files3)
        return resp.json()

    def download_file(self, remote_file=None, local_file=None, relpath=True):
        """
        :param str remote_file: 要下载的远程文件路径，相对get_wkdir下的路径
        :param str local_file: 下载到本地的文件路径，默认可以不指定，自动通过remote_file存储到对称位置

        使用示例：
        sfc = SyncFileClient('http://yourserver.com', headers={"Authorization": "Bearer your_token"})
        sfc.download_file('.vscode/launch.json', 'launch.json')
        """
        if relpath:
            local_file = self.get_abs_local_path(local_file, remote_file)
            remote_file = self.get_abs_remote_path(remote_file, local_file)

        data = {'file': remote_file.as_posix()}
        # 使用 stream=True 开启流式处理
        with requests.post(f'{self.host}/common/download_file',
                           headers=self.headers, json=data, stream=True) as resp:
            if resp.status_code == 200:
                # 以二进制方式写入文件
                with open(local_file, 'wb') as f:
                    # 分块写入文件，每次 1MB (1024 * 1024 bytes)
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:  # 忽略keep-alive的空块
                            f.write(chunk)
                return local_file
            else:
                # 处理其他状态码，抛出异常或进行错误处理
                print(f"Error: Unable to download file, status code {resp.status_code}")
                return None

    def verify_etag(self, local_file=None, remote_file=None, relpath=True):
        """
        :param relpath: 是否使用的是相对路径
            当不使用relpath的时候，注意local_file、remote_file都是手动必填的XlPath类型字段
        :return:
        """
        if relpath:
            local_file = self.get_abs_local_path(local_file, remote_file)
            remote_file = self.get_abs_remote_path(remote_file, local_file)
        local_etag = GetEtag.from_file(local_file)
        json_data = {'file': remote_file.as_posix(), 'etag': local_etag}
        resp = requests.post(f'{self.host}/common/check_file_etag',
                             headers=self.headers,
                             json=json_data)
        return resp.json()

    def upload_dir(self,
                   local_dir=None,
                   remote_dir=None,
                   *,
                   delete_local_file=False,
                   verify_etag=False,
                   relpath=True):
        """
        :param local_dir: 本地目录
        :param remote_dir: 远程目录
        :param delete_local_file: 每个文件上传成功后，是否删除本地文件
            默认不删除，一般不删除是配合etag校验使用的
        :param verify_etag: 上传前是否校验etag，如果服务器上对应位置已经有对应etag的文件，其实就不用上传了

        该方法已修改以支持上传子目录和空目录
        """
        if relpath:
            # D:\home\chenkunze\data\temp2，举个具体的例子案例值，供参考理解
            local_dir = self.get_abs_local_path(local_dir, remote_dir)
            # /home/chenkunze/data/temp2
            remote_dir = self.get_abs_remote_path(remote_dir, local_dir)

        for root, dirs, files in os.walk(local_dir):
            # root拿到的是这样的绝对路径：D:\home\chenkunze\data\temp2\数值问题
            # '数值问题'
            subdir = XlPath(root).relpath(local_dir)
            if not files and not dirs:
                # 如果该目录为空，则在远程服务器上创建空目录
                self.create_remote_dir(remote_dir / subdir, relpath=False)

            for file in tqdm(files, desc=f'上传目录：{root}'):
                local_path = local_dir / subdir / file
                remote_path = remote_dir / subdir / file
                if verify_etag and self.verify_etag(local_path, remote_path, relpath=False):
                    pass  # 如果etag校验通过，跳过上传
                else:
                    self.upload_file(local_path, remote_path, relpath=False)

                if delete_local_file:
                    os.remove(local_path)

    def create_remote_dir(self, remote_dir, relpath=True):
        """
        创建远程目录的辅助方法
        """
        if relpath:
            remote_dir = self.get_abs_remote_path(remote_dir)
        json_data = {'dir': remote_dir.as_posix()}
        resp = requests.post(f'{self.host}/common/create_dir',
                             headers=self.headers,
                             json=json_data)
        return resp.json()

    def download_dir(self, remote_dir=None, local_dir=None, *, verify_etag=False, relpath=True):
        """
        下载远程目录到本地
        :param remote_dir: 远程目录路径，相对路径或绝对路径都支持
        :param local_dir: 本地保存目录，默认会自动映射到对应的远程位置
        :param verify_etag: 是否进行etag校验，跳过已经在本地存在并且未修改的文件
        :return:

        示例：
        sfc = SyncFileClient('codepc_mi15')
        sfc.download_dir('data/temp2')
        """
        # 1. 获取绝对路径
        if relpath:
            local_dir = self.get_abs_local_path(local_dir, remote_dir)
            remote_dir = self.get_abs_remote_path(remote_dir, local_dir)

        # 2. 从服务器获取远程目录的文件和子目录结构
        json_data = {'dir': remote_dir.as_posix()}
        resp = requests.post(f'{self.host}/common/list_dir', headers=self.headers, json=json_data)
        if resp.status_code != 200:
            raise Exception(f"Failed to list remote directory: {resp.text}")

        dir_structure = resp.json()  # 期望返回格式为 {'dirs': [...], 'files': [...]}

        # 3. 创建本地目录
        if not local_dir.exists():
            local_dir.mkdir(parents=True)

        # 4. 先处理子目录
        for subdir in dir_structure.get('dirs', []):
            local_subdir = local_dir / subdir
            if not local_subdir.exists():
                local_subdir.mkdir(parents=True)

            # 递归下载子目录中的文件
            self.download_dir(remote_dir / subdir, local_dir / subdir, verify_etag=verify_etag)

        # 5. 下载文件
        for file in tqdm(dir_structure.get('files', []), desc=f'下载目录：{remote_dir}'):
            remote_file = remote_dir / file
            local_file = local_dir / file

            # 校验etag，如果verify_etag为True并且校验成功，则跳过下载
            if verify_etag and self.verify_etag(local_file, remote_file, relpath=False):
                continue

            # 下载文件
            self.download_file(remote_file, local_file, relpath=False)

    def download_path(self, remote_path=None, local_path=None, *, verify_etag=False, relpath=True):
        """
        下载远程文件或目录到本地
        :param remote_path: 远程路径，可以是文件或者目录
        :param local_path: 本地保存路径，默认会自动映射到对应的远程位置
        :param verify_etag: 是否进行etag校验，跳过已经在本地存在并且未修改的文件
        :return:
        """
        if relpath:
            # 获取远程路径的绝对路径
            remote_path = self.get_abs_remote_path(remote_path, local_path)
            local_path = self.get_abs_local_path(local_path, remote_path)

        # 判断是文件还是目录
        json_data = {'path': remote_path.as_posix()}
        resp = requests.post(f'{self.host}/common/check_path_type', headers=self.headers, json=json_data)

        if resp.status_code != 200:
            raise Exception(f"Failed to check remote path type: {resp.text}")

        path_info = resp.json()

        if path_info['type'] == 'file':
            # 如果是文件，调用 download_file
            self.download_file(remote_file=remote_path, local_file=local_path, relpath=False)
        elif path_info['type'] == 'dir':
            # 如果是目录，调用 download_dir
            self.download_dir(remote_dir=remote_path, local_dir=local_path,
                              verify_etag=verify_etag, relpath=False)
        else:
            raise ValueError(f"Unknown path type: {path_info['type']}")
