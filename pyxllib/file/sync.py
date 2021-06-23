#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 20:41

import os
import subprocess

try:
    import paramiko
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'paramiko'])
    import paramiko

# 对 paramiko 进一步封装的库
# try:
#     import fabric
# except ModuleNotFoundError:
#     subprocess.run(['pip3', 'install', 'fabric'])
#     import fabric

try:
    import scp
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'scp'])
    import scp

import humanfriendly

from pyxllib.file.newbie import linux_path_fmt
from pyxllib.debug.specialist.tictoc import TicToc
from pyxllib.file.specialist import Dir, file_or_dir_size


def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


class DataSync:
    """ 在windows和linux之间同步文件数据

    DataSync只有在win平台才能用

    TODO 如果本地目录a已存在，从服务器a再下载，会跑到a/a的位置
    TODO 选择性从服务器拷贝？ 怎么获得服务器有哪些文件内容，使用ssh来获得？

    TODO 好像不需要使用scp，也能自动传文件？
    """

    def __init__(self, server, port, user, passwd):
        self.client = createSSHClient(server, port, user, passwd)
        self.scp = scp.SCPClient(self.client.get_transport())

    def put(self, local_path, remote_path, *, prt=True):
        """ 将本地的local_path上传到服务器的remote_path

        差不多是标准的scp接口，可以强制上传一个文件，或者一个目录
        """

        tt = TicToc()
        tt.tic()
        local_path, remote_path = str(local_path), linux_path_fmt(remote_path)
        # 其实scp也支持同时同步多文件，但是目标位置没法灵活控制，所以我这里还是一个一个同步
        self.client_exec(f'mkdir -p {os.path.dirname(remote_path)}')  # remote如果不存在父目录则建立
        self.scp.put(local_path, remote_path, recursive=True)
        t = tt.tocvalue()
        speed = humanfriendly.format_size(file_or_dir_size(local_path) / t, binary=True)
        if prt:
            print(f'upload to {remote_path}, ↑{speed}/s, {t:.2f}s')

    def get(self, local_path, remote_path, *, prt=True):
        """ 将服务器的remote_path下载到本地的local_path """
        tt = TicToc()
        tt.tic()
        local_path, remote_path = str(local_path), linux_path_fmt(remote_path)
        # 目录的同步必须要开recursive，其他没什么区别
        Dir(os.path.dirname(local_path)).ensure_dir()
        self.scp.get(remote_path, local_path, recursive=True, preserve_times=True)
        t = tt.tocvalue()
        speed = humanfriendly.format_size(file_or_dir_size(local_path) / t, binary=True)
        if prt: print(f'download {local_path}, ↓{speed}/s, {t:.2f}s')

    def client_exec(self, command, bufsize=-1, timeout=None, get_pty=False, environment=None):
        """ 在服务器执行命令

        如果stderr出错，则抛出异常
        否则返回运行结果的文本数据
        """
        stdin, stdout, stderr = self.client.exec_command(command, bufsize, timeout, get_pty, environment)
        stderr = list(stderr)
        if stderr:
            print(''.join(stderr))
            raise ValueError(f'服务器执行命令报错: {command}')
        return '\n'.join([f.strip() for f in list(stdout)])

    def remote_files(self, remote_dir):
        """ 远程某个目录下，递归获取所有文件清单

        这里要调用到事先在pyxllib.tool写好的listfiles工具
        """
        remote_dir = linux_path_fmt(remote_dir)
        stdout = self.client_exec(f'python3 -m pyxllib.tool.listfiles "{remote_dir}"')
        return stdout.splitlines()

    def rput(self, local_dir, remote_dir):
        """ rput相比put多了个前缀r，表示递归处理，是扩展的的一个高级功能

        允许输入一个目录，递归处理其子目录、文件同步
        当服务器上已有同名文件时不操作，没有的则将本地文件上传至服务器

        TODO 未来可以通过文件名时间、大小、哈希值等判断是否相同，然后更智能地选择是否需要同步
        """
        from tqdm import tqdm
        from pyxllib.tool.listfiles import listfiles

        local_dir, remote_dir = str(local_dir), str(remote_dir)

        with TicToc('初始化计算'):
            local_files = listfiles(local_dir)
            remote_files = self.remote_files(remote_dir)
            # 其实windows是可以把大小写不同的文件上传到linux的，但为了避免混乱，暂时不这样搞；先统一按照不区分大小写处理
            remote_files = {f.lower() for f in remote_files}

        for f in tqdm(local_files, '上传文件到服务器', smoothing=0, mininterval=1):  # 这里开多线程好像会有点问题，先串行使用
            if f.lower() not in remote_files:
                self.put(os.path.join(local_dir, f), os.path.join(remote_dir, f), prt=False)

    def rget(self, local_dir, remote_dir, *, maxn=None):
        """ 控制同步的文件上限数 """
        from tqdm import tqdm
        from pyxllib.tool.listfiles import listfiles

        local_dir, remote_dir = str(local_dir), str(remote_dir)

        with TicToc('初始化计算'):
            local_files = listfiles(local_dir)
            remote_files = self.remote_files(remote_dir)
            if maxn: remote_files = remote_files[:maxn]
            # windows文件不区分大小写，所以把linux文件拉到windows要格外注意，必须按小写情况统一判断下
            local_files = {f.lower() for f in local_files}

        for f in tqdm(remote_files, '从服务器下载文件到本地', smoothing=0, mininterval=1):
            if f.lower() not in local_files:
                self.get(os.path.join(local_dir, f), os.path.join(remote_dir, f), prt=False)
