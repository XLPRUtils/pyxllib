#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 20:41
import time

from pyxllib.prog.pupil import check_install_package

check_install_package('paramiko')
check_install_package('scp')
# 对 paramiko 进一步封装的库
# check_install_package('fabric')

from collections import defaultdict
import os
import re
import pathlib
import shutil

import paramiko
from tqdm import tqdm
import scp as scplib
import humanfriendly

from pyxllib.algo.pupil import natural_sort
from pyxllib.file.specialist import XlPath
from pyxllib.debug.specialist import get_xllog
from pyxllib.prog.specialist import XlOsEnv

logger = get_xllog('location')


class SshCommandError(Exception):
    pass


class ScpLimitError(Exception):
    pass


class ScpProgress:
    # 这里的scp类比较特别，要用特殊的方式来实现一个tqdm式的进度条
    def __init__(self, info, **kwargs):
        self.info = info
        self.kwargs = kwargs

        self.trans_size = 0  # 总共传输了多少
        self.this_file_sent = 0  # 当前文件传输了多少

        if info == 0:
            self.tqdm = None
        elif info == 1:
            self.kwargs['frequency'] = 0  # 传输频数
            self.tqdm = tqdm(desc=self.kwargs.get('desc'), disable=False,
                             total=self.kwargs['total'], unit_scale=True)
        else:
            raise NotImplementedError

    def __call__(self, filename, size, sent):
        # 1 记录更新信息
        if isinstance(filename, bytes):
            filename = filename.decode('utf8')
        if 'rf' in self.kwargs:  # 仅显示相对路径
            filename = self.kwargs['rf']
        elif 'desc' in self.kwargs and filename.startswith(self.kwargs["desc"][1:]):
            filename = filename[len(self.kwargs["desc"]):]

        finish_file = (size == sent)  # 传完一个文件
        increment = sent - self.this_file_sent  # 默认每次是上传16kb，可以在SCP初始化修改buff_size
        self.trans_size += increment
        self.this_file_sent = 0 if finish_file else sent

        # 2 不同info的处理
        if self.info == 0:
            pass
        elif self.info == 1:
            self.kwargs['frequency'] += 1
            self.tqdm.desc = f'{self.kwargs["desc"]} > {sent / size:4.0%} ' + filename
            self.tqdm.n += increment
            if finish_file or self.kwargs['frequency'] % 100 == 0:  # 减小展示间隔
                self.tqdm.update(0)

        # 3 检查limit_bytes
        if sent == size and self.kwargs.get('limit_bytes'):
            if self.trans_size >= self.kwargs.get('limit_bytes'):
                if self.info:
                    logger.warning('达到限定上传大小，早停。')
                raise ScpLimitError


class XlSSHClient(paramiko.SSHClient):
    def __1_初始化和执行(self):
        pass

    def __init__(self, server, port, user, passwd, *, map_path=None,
                 relogin=0, relogin_interval=1):
        """

        :param map_path: 主要在上传、下载文件的时候，可以用来自动定位路径
            参考写法：{'D:/': '/'}  # 将D盘映射到服务器位置
        :param int relogin: 当连接失败的时候，可能只是某种特殊原因，可以尝试重新连接
            该参数设置重连的次数
        :param int|float relogin_interval: 每次重连的间隔秒数

        """

        super().__init__()
        self.load_system_host_keys()
        self.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # 重试登录
        for k in range(relogin + 1):
            try:
                self.connect(server, port, user, passwd)
                break
            except paramiko.ssh_exception.SSHException:
                if k < relogin:
                    time.sleep(relogin_interval)

        self.map_path = map_path

        class Path(pathlib.PurePosixPath):
            """ 生成一个可以管理远程服务器上的路径类

            该类只是在确实有单文件判断需求下方便使用
            实际处理问题中，推荐优先使用多文件一次性统一处理的命令，减小和服务器间的交互，提高运行速度
            """
            client = self

            def exists_type(self):
                # TODO 还没有考虑符号链接等情况，在特殊情况下可能会出意外
                t = self.client.exec(f'if test -d "{self}"; then echo -1;'
                                     f'elif test -f "{self}"; then echo 1;'
                                     'else echo 0; fi')
                return int(t)

            def sub_rel_paths(self, mtime=False, *, extcmd=''):
                """ 目录对象的时候，可以获取远程目录下的文件清单（相对路径）

                :param extcmd: 一般是增加 -maxdepth 1 的参数，限制只遍历第1层
                :param mtime: 增加返回mtime信息
                """
                printf = '%P %Ts\n' if mtime else r'%P\n'
                cmd = f'find "{self}" -mindepth 1'
                if extcmd: cmd += ' ' + extcmd
                cmd += f' -printf "{printf}"'
                stdout = self.client.exec(cmd)
                lines = stdout.splitlines()
                if mtime:
                    lines = [list(re.match(r'(.+?)\s(\d+)$', line).groups()) for line in lines]
                    return {line[0]: int(line[1]) for line in lines}
                else:
                    return set(lines)

            def mtime(self):
                res = self.client.exec(f'find "{self}" -maxdepth 0 -printf "%Ts"')
                return int(res)

            def set_mtime(self, timestamp):
                """ 手动修改文件的mtime

                我也不想搞这么复杂，但是scp底层传输时间戳有误差，没办法啊~~~
                """
                pass

            def size(self, *, human_readable=False):
                # 其实这里用-h也能美化，但怕计量方式跟XlPath.size不同，所以还是用humanfriendly再算
                try:
                    res = self.client.exec(f'du "{self}" -s -b')
                except SshCommandError:
                    res = 0

                sz = int(re.match(r'\d+', res).group())
                if human_readable:
                    return humanfriendly.format_size(self.size, binary=True)
                else:
                    return sz

        self.Path = Path

    @classmethod
    def log_in(cls, host_name, user_name, **kwargs):
        r""" 使用XlprDb里存储的服务器、账号信息，进行登录
        """
        from pyxllib.data.pglib import connect_xlprdb

        with connect_xlprdb() as con:
            if host_name.startswith('g_'):
                host_ip = con.execute("SELECT host_ip FROM hosts WHERE host_name='xlpr0'").fetchone()[0]
                pw, port = con.execute('SELECT (pgp_sym_decrypt(accounts, %s)::jsonb)[%s]::text, frpc_port'
                                       ' FROM hosts WHERE host_name=%s',
                                       (con.seckey, user_name, host_name[2:])).fetchone()
            else:
                port = 22
                host_ip, pw = con.execute('SELECT host_ip, (pgp_sym_decrypt(accounts, %s)::jsonb)[%s]::text'
                                          ' FROM hosts WHERE host_name=%s',
                                          (con.seckey, user_name, host_name)).fetchone()

        return cls(host_ip, port, user_name, pw[1:-1], **kwargs)

    def exec(self, command, *args, **kwargs):
        """ exec_command的简化版

        如果stderr出错，则抛出异常
        否则返回运行结果的文本数据
        """
        stdin, stdout, stderr = self.exec_command(command, *args, **kwargs)
        stderr = list(stderr)
        if stderr:  # TODO 目前警告也会报错，其实警告没关系
            print(''.join(stderr))
            raise SshCommandError(f'服务器执行命令报错: {command}')
        return '\n'.join([f.strip() for f in list(stdout)])

    def exec_script(self, main_cmd, script='', *, file=None):
        r""" 执行批量、脚本命令

        :paramn main_cmd: 主命令
        :param script: 用字符串表达的一套命令
        :param file: 也可以直接传入一个文件

        【使用示例】
        ssh = XlSSHClient.log_in('xlpr10', 'chenkunze')
        text = textwrap.dedent('''\
        import os
        print(len(os.listdir()))
        ''')
        print(ssh.exec_script('python3', text))
        """
        # 1 将脚本转成文件，上传到服务器
        scp = scplib.SCPClient(self.get_transport())

        # 虽然是基于本地的情况生成随机名称脚本，但一般在服务器也不会冲突，概率特别小
        local_file = XlPath.tempfile()
        host_file = '/tmp/' + local_file.name

        if file is not None:
            shutil.copy2(XlPath(file), local_file)
        elif script:
            local_file.write_text(script)
        else:
            raise ValueError(f'没有待执行的脚本')

        scp.put(local_file, host_file)
        local_file.delete()

        # 2 执行程序
        res = self.exec(f'{main_cmd} {host_file}')
        self.exec(f'rm {host_file}')
        return res

    def __2_scp(self):
        """ 以下是为scp准备的功能 """
        pass

    def __local_dir(self, remote_path, local_dir):
        if local_dir is not None:
            return XlPath(local_dir)
        else:
            for k, v in self.map_path.items():
                try:
                    relpath = remote_path.relative_to(v)
                    return (XlPath(k) / relpath).parent
                except ValueError:
                    pass
            raise ValueError('找不到对应的map_path路径映射规则')

    def __remote_dir(self, local_path, remote_dir):
        if remote_dir is not None:
            return self.Path(remote_dir)
        else:
            for k, v in self.map_path.items():
                try:
                    relpath = local_path.relative_to(k)
                    return (self.Path(v) / relpath.as_posix()).parent
                except ValueError:
                    pass
            raise ValueError('找不到对应的map_path路径映射规则')

    def __check_filetype(self, local_path, remote_path):
        """ 同类型或有一边不存在都可以

        不存在记为0，文件记为1，目录记为-1，那么只要乘积不为负数即可。

        尽量少用这个，毕竟这个要连接服务器，是有资源开销的。
        """
        a = local_path.exists_type()
        b = remote_path.exists_type()
        if a * b < 0:
            raise TypeError(f'本地和服务器的文件类型不匹配 {local_path} {remote_path}')
        return a, b

    def __scp_base(self, func, progress, from_path, to_dir, to_path, if_exists):
        """

        Args:
            func:
            progress:
            from_path: 来源路径，可能是scp.put，来源在本地，也肯是scp.get，来源在服务器
            to_dir: to_同理，跟from_相反
            to_path:
            if_exists:

        Returns:

        """

        def scp_core(mtime=False):
            a, b = self.__check_filetype(from_path, to_path)
            if b == 0:  # 目标位置不存在对应文件或目录，可以直接传输
                func(str(from_path), str(to_dir), recursive=True, preserve_times=True)
            elif a == b == 1:
                if mtime and from_path.mtime() > to_path.mtime():
                    func(str(from_path), str(to_dir), preserve_times=True)
            elif a == b == -1:
                to_paths = to_path.sub_rel_paths(mtime)
                if mtime:
                    # 因为不知道是from还是to是远程服务器，都统一一次性获得mtime更好
                    from_paths = from_path.sub_rel_paths(mtime)

                def put_dir(dir0):
                    # 1 获得当前目录文件情况
                    sub_dirs, sub_files = [], []
                    if isinstance(dir0, self.Path):
                        sub_files = [(dir0 / f) for f in dir0.sub_rel_paths(extcmd='-maxdepth 1 -type f')]
                        sub_dirs = [(dir0 / d) for d in dir0.sub_rel_paths(extcmd='-maxdepth 1 -type d')]
                    else:
                        for p in dir0.glob('*'):  # 本机文件
                            if p.is_file():
                                sub_files.append(p)
                            else:
                                sub_dirs.append(p)

                    # 2 上传文件
                    sub_files = natural_sort(sub_files)
                    for f in sub_files:
                        rf = f.relative_to(from_path).as_posix()
                        progress.kwargs['rf'] = rf
                        # 检查时间戳
                        # print(from_path, to_path, rf, from_paths[rf], to_paths.get(rf, 0),
                        #       from_paths[rf] - to_paths.get(rf, 0))
                        if rf in to_paths and ((not mtime) or (from_paths[rf] <= to_paths[rf])):
                            if progress.tqdm:
                                if isinstance(f, self.Path):
                                    # 如果不需要从服务器下载到本地，直接用本地文件尺寸代替去服务器找文件尺寸
                                    progress.tqdm.total -= (to_path / rf).size()
                                else:
                                    progress.tqdm.total -= f.size()
                        else:
                            func(str(f), str(to_path / rf), preserve_times=True)

                    # 3 上传目录
                    sub_dirs = natural_sort(sub_dirs)
                    for d in sub_dirs:
                        rd = d.relative_to(from_path).as_posix()
                        if rd in to_paths:
                            put_dir(d)
                        else:
                            func(str(d), str(to_path / rd), recursive=True, preserve_times=True)

                put_dir(from_path)

        if if_exists is None or if_exists == 'replace':
            func(str(from_path), str(to_dir), recursive=True, preserve_times=True)
        elif if_exists == 'skip':
            scp_core(False)
        elif if_exists == 'mtime':
            scp_core(True)
        else:
            raise ValueError

    def scp_get(self, remote_path=None, local_dir=None, *, local_path=None, info=True, limit_bytes=None,
                if_exists=None):
        """ 文档参考 self.scp_put

        :param local_path: 可以不输入远程remote_path，仅输入local_path来映射、运行
        get应该会比put更慢一点，因为get需要使用命令从服务器获得更多参考信息，而put很多文件信息可以在本地直接获得

        >> self.scp_get('/home/datasets/doc3D', '/home/dataset')
        """
        if local_path:
            local_path = XlPath(local_path)
            remote_path = self.__remote_dir(local_path, None) / local_path.name
            local_dir = local_path.parent
        else:
            remote_path = self.Path(remote_path)
            local_dir = self.__local_dir(remote_path, local_dir)
            local_path = local_dir / remote_path.name

        # 远程文件不存在，不用运行
        if not remote_path.exists_type():
            return

        if not local_dir.exists():
            local_dir.mkdir(parents=True, exist_ok=True)

        if info == 0:
            progress = ScpProgress(info, limit_bytes=limit_bytes)
        elif info == 1:
            progress = ScpProgress(info, desc=f'↓{local_path}',
                                   total=remote_path.size(),
                                   limit_bytes=limit_bytes)
        else:
            raise NotImplementedError

        scp = scplib.SCPClient(self.get_transport(), progress=progress)

        try:
            self.__scp_base(scp.get, progress, remote_path, local_dir, local_path, if_exists)
        except ScpLimitError:
            pass

    def scp_put(self, local_path, remote_dir=None, *, mkdir=True, print_mode=True, limit_bytes=None, if_exists=None):
        """ 将本地的local_path上传到服务器的remote_path

        差不多是标准的scp接口，可以强制上传一个文件，或者一个目录

        :param local_path: 可以是文件，也可以是目录
            上传目录时
                同名子文件会被替换
                不存在的文件会增量
                本地没有，但服务器有的文件不做操作
        :param remote_dir: 远程位置的父级目录，注意是父级
            因为DataSync侧重数据同步，所以上传的文件或目录默认同名
                如果要改名字，建议使用底层的scp接口实现
        :param mkdir: remote_dir可能不存在，为了避免出错，是否执行下mkdir
        :param bool|int print_mode:
            0: 不显示进度
            1: 显示整体上传进度（字节）
            2: 显示每个文件上传进度  （这个功能好像不常用没那么重要，暂未实现）
        :param limit_bytes: 限制传输的字节上限，可以用来筛选部分样例数据，不用下载整个庞大的目录
            注意该限制
        :param if_exists:
            None, 'replace' 不处理，也就是直接替换掉
            'skip'，跳过不处理，保留远程原文件
            'mtime'，对比时间戳，如果本地文件时间更新，则会上传到远程，否则跳过不处理

        >> self.scp_put('D:/home/chenkunze/test')
        >> self.scp_put('D:/home/chenkunze/test', '/home/chenkunze')
        """
        local_path = XlPath(local_path)
        remote_dir = self.__remote_dir(local_path, remote_dir)
        remote_path = remote_dir / local_path.name

        if mkdir:  # 判断服务器remote_dir存不存在，也要用命令，还不如直接用mkdir了
            self.exec(f'mkdir -p {remote_dir}')  # remote如果不存在父目录则建立

        if print_mode == 0:
            # 虽然不显示运行信息，但也要记录已上传了多少流量
            progress = ScpProgress(print_mode, limit_bytes=limit_bytes)
        elif print_mode == 1:
            progress = ScpProgress(print_mode, desc=f'↑{remote_path}',
                                   total=local_path.size(),
                                   limit_bytes=limit_bytes)
        else:
            raise NotImplementedError

        # 这里可以设置 buff_size
        scp = scplib.SCPClient(self.get_transport(), progress=progress)

        try:
            self.__scp_base(scp.put, progress, local_path, remote_dir, remote_path, if_exists)
        except ScpLimitError:
            pass

    def scp_sync(self, local_path, *, mkdir=True, info=True, limit_bytes=None):
        """ 服务器和本地目录的数据同步

        该功能目前还不稳定，还有时间戳可能无法准确复制的bug，这个bug影响
        1、无法准确对比时间戳，会冗余传送文件
        2、误判时间，导致新文件时间戳不是最新的，漏同步

        sync其实就是get、put都跑一次就行了
        注意此时limit_bytes表示的不是总流量，而是上传、下载分别的最大流量
        及没有if_exists参数，默认都通过mtime时间戳来更新
        """
        self.scp_get(local_path=local_path, info=info, limit_bytes=limit_bytes, if_exists='mtime')
        self.scp_put(local_path, mkdir=mkdir, print_mode=info, limit_bytes=limit_bytes, if_exists='mtime')

    def __3_其他封装的功能(self):
        pass

    def set_user_passwd(self, name, passwd):
        """ 修改账号密码

        How to set user passwords using passwd without a prompt? - Ask Ubuntu:
        https://askubuntu.com/questions/80444/how-to-set-user-passwords-using-passwd-without-a-prompt
        """
        self.exec(f'usermod --password $(echo {passwd} | openssl passwd -1 -stdin) {name}')

    def add_user(self, name, passwd, sudo=False):
        """ 添加新用户

        :param name: 用户名
        :param passwd: 密码
        :param sudo: 是否开sudo权限
        """
        exists_user = self.exec("awk -F: '{ print $1}' /etc/passwd").splitlines()
        if name in exists_user:
            raise ValueError(f'{name} already exists')

        self.exec(f'useradd -d /home/{name} -s /bin/bash -m {name}')
        self.set_user_passwd(name, passwd)
        if sudo:
            self.exec(f'usermod -aG sudo {name}')

    def check_gpu_usage(self, *, print_mode=False):
        """ 检查(每个用户)显存使用量

        使用这个命令，必须要求远程服务器安装了pip install gpustat
        """

        def printf(*args, **kwargs):
            if print_mode:
                print(*args, **kwargs)

        ls = self.exec('gpustat')
        printf(ls)

        total_memory = 0
        user_usage = defaultdict(float)
        for line in ls.splitlines()[1:]:
            name, temperature, capacity, uses = line.split('|')
            used, total = map(int, re.findall(r'\d+', capacity))
            user_usage['other'] += used / 1024
            total_memory += total / 1024
            for x in uses.split():
                user, used = re.search(r'(.+?)\((\d+)M\)', x).groups()
                user_usage[user] += int(used) / 1024
                user_usage['other'] -= int(used) / 1024

        # 使用量从多到少排序。但注意如果转存到PG，jsonb会重新排序。
        user_usage = {k: round(user_usage[k], 2) for k in sorted(user_usage, key=lambda k: -user_usage[k])}
        if user_usage['other'] < 0.01:
            del user_usage['other']

        return total_memory, user_usage

    def check_disk_usage(self):
        """ 检查(每个用户)磁盘空间使用量

        :return: total 总字节MBytes数, msg 所有用户、包括其他非/home目录的使用MBytes数
        """
        GB = 1024 ** 2  # df、du本身默认单位已经是KB，所以2次方后，就是GB了
        # 1 整体情况
        used, total_memory = 0, 0
        for line in self.exec('df').splitlines()[1:]:
            if not line.startswith('/dev/'):
                continue
            # 为了避免遇到路径中有空格的问题，用正则做了较复杂的判断、切割
            _total, _used = map(int, re.search(r'\s+(\d+)\s+(\d+)\s+\d+\s+\d+', line).groups())
            used += _used
            total_memory += _total

        # 2 /home目录下每个占用情况
        user_usage = defaultdict(int)
        for line in self.exec('du -d 1 /home').splitlines():
            _bytes, _dir = line.split(maxsplit=1)
            _dir = _dir[6:]
            if _dir and int(_bytes) > GB:  # 达到1GB的才记录
                user_usage[_dir] += int(_bytes)

        user_usage['other'] = used - sum(user_usage.values())
        user_usage = {k: (v // GB) for k, v in user_usage.items()}

        return total_memory // GB, user_usage
