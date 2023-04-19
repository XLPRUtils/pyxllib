#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 18:40

from pyxllib.prog.specialist.common import *
from pyxllib.prog.specialist.xllog import *
from pyxllib.prog.specialist.browser import *
from pyxllib.prog.specialist.bc import *
from pyxllib.prog.specialist.tictoc import *
from pyxllib.prog.specialist.datetime import *

import base64
import concurrent.futures
import json
import os
import re
import subprocess
import time

from tqdm import tqdm

from pyxllib.prog.xlosenv import XlOsEnv


def mtqdm(func, iterable, *args, max_workers=1, check_per_seconds=0.01, **kwargs):
    """ 对tqdm的封装，增加了多线程的支持

    这里名称前缀多出的m有multi的意思

    :param max_workers: 默认是单线程，改成None会自动变为多线程
        或者可以自己指定线程数
        注意，使用负数，可以用对等绝对值数据的“多进程”
    :param smoothing: tqdm官方默认值是0.3
        这里关掉指数移动平均，直接计算整体平均速度
        因为对我个人来说，大部分时候需要严谨地分析性能，得到整体平均速度，而不是预估当前速度
    :param mininterval: 官方默认值是0.1，表示显示更新间隔秒数
        这里不用那么频繁，每秒更新就行了~~
    :param check_per_seconds: 每隔多少秒检查队列
        有些任务，这个值故意设大一点，可以减少频繁的队列检查时间，提高运行速度
    整体功能类似Iterate
    """

    # 0 个人习惯参数
    kwargs['smoothing'] = kwargs.get('smoothing', 0)
    kwargs['mininterval'] = kwargs.get('mininterval', 1)

    if max_workers == 1:
        # 1 如果只用一个线程，则不使用concurrent.futures.ThreadPoolExecutor，能加速
        for x in tqdm(iterable, *args, **kwargs):
            func(x)
    else:
        # 2 默认的多线程运行机制，出错是不会暂停的；这里对原函数功能进行封装，增加报错功能
        error = False

        def wrap_func(x):
            nonlocal error
            try:
                func(x)
            except Exception as e:
                error = e

        # 3 多线程/多进程 和 进度条 功能的结合
        if max_workers > 1:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers)
            for x in tqdm(iterable, *args, **kwargs):
                while executor._work_queue.qsize():
                    if check_per_seconds:
                        time.sleep(check_per_seconds)
                executor.submit(wrap_func, x)
                if error:
                    raise error
        else:
            executor = concurrent.futures.ProcessPoolExecutor(-max_workers)
            for x in tqdm(iterable, *args, **kwargs):
                # while executor._call_queue.pending_work_items:
                #     if check_per_seconds:
                #         time.sleep(check_per_seconds)
                executor.submit(wrap_func, x)
                if error:
                    raise error

        executor.shutdown()


def distribute_package(root, version=None, repository=None, *,
                       upload=True,
                       version_file='setup.py',
                       delete_dist=True):
    """ 发布包的工具函数

    :param root: 项目的根目录，例如 'D:/slns/pyxllib'
        根目录下有对应的 setup.py 等文件
    :param repository: 比如我配置了 [xlpr]，就可以传入 'xlpr'
    :param version_file: 保存版本号的文件，注意看正则规则，需要满足特定的范式，才会自动更新版本号
    :param delete_dist: 上传完是否自动删除dist目录，要检查上传包是否有遗漏时，要关闭
    """
    from pyxllib.file.specialist import XlPath

    # 1 切换工作目录
    os.chdir(str(root))

    # 2 改版本号
    if version:
        f = XlPath(root) / version_file
        s = re.sub(r"(version\s*=\s*)(['\"])(.+?)(\2)", fr'\1\g<2>{version}\4', f.read_text())
        f.write_text(s)

    # 3 打包
    subprocess.run('python setup.py sdist')

    # 4 上传
    if upload:
        # 上传
        cmd = 'twine upload dist/*'
        if repository:
            cmd += f' -r {repository}'
        subprocess.run(cmd)
        # 删除打包生成的中间文件
        if delete_dist:
            XlPath('dist').delete()
        XlPath('build').delete()

        # 这个不能删，不然importlib会读取不到模块的版本号
        # [d.delete() for d in XlPath('.').select_dirs(r'*.egg-info')]
