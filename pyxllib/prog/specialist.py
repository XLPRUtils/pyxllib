#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 11:16

import base64
import concurrent.futures
import json
import os
import re
import subprocess
import time

from tqdm import tqdm

from pyxllib.text.newbie import add_quote


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


class XlOsEnv:
    """ pyxllib库自带的一套环境变量数据解析类

    会将json的字符串值，或者普通str，存储到环境变量中

    环境变量也可以用来实现全局变量的信息传递，虽然不太建议这样做

    >> XlOsEnv.persist_set('TP10_ACCOUNT',
                           {'server': '172.16.250.250', 'port': 22, 'user': 'ckz', 'passwd': '123456'},
                           True)
    >> print(XlOsEnv.get('TP10_ACCOUNT'), True)  # 展示存储的账号信息
    eyJzZXJ2ZXIiOiAiMTcyLjE2LjE3MC4xMzQiLCAicG9ydCI6IDIyLCAidXNlciI6ICJjaGVua3VuemUiLCAicGFzc3dkIjogImNvZGV4bHByIn0=
    >> XlOsEnv.unset('TP10_ACCOUNT')
    """

    @classmethod
    def get(cls, name, *, decoding=False):
        """ 获取环境变量值

        :param name: 环境变量名
        :param decoding: 是否需要先进行base64解码
        :return:
            返回json解析后的数据
            或者普通的字符串值
        """
        value = os.getenv(name, None)
        if value is None:
            return value

        if decoding:
            value = base64.b64decode(value.encode())

        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            return value

    @classmethod
    def set(cls, name, value, encoding=False):
        """ 临时改变环境变量

        :param name: 环境变量名
        :param value: 要存储的值
        :param encoding: 是否将内容转成base64后，再存储环境变量
            防止一些密码信息，明文写出来太容易泄露
            不过这个策略也很容易被破解；只防君子，难防小人

            当然，谁看到这有闲情功夫的话，可以考虑做一套更复杂的加密系统
            并且encoding支持多种不同的解加密策略，这样单看环境变量值就很难破译了
        :return: str, 最终存储的字符串内容
        """
        # 1 打包
        if isinstance(value, str):
            value = add_quote(value)
        else:
            value = json.dumps(value)

        # 2 编码
        if encoding:
            value = base64.b64encode(value.encode()).decode()

        # 3 存储到环境变量
        os.environ[name] = value

        return value

    @classmethod
    def persist_set(cls, name, value, encoding=False):
        """ python里默认是改不了系统变量的，需要使用一些特殊手段

        https://stackoverflow.com/questions/17657686/is-it-possible-to-set-an-environment-variable-from-python-permanently/17657905
        """
        # 写入环境变量这里是有点小麻烦的，要考虑unix和windows不同平台，以及怎么持久化存储的问题，这里直接调用一个三方库来解决
        from envariable import setenv

        value = cls.set(name, value, encoding)
        if value[0] == value[-1] == '"':
            value = '\\' + value + '\\'
        setenv(name, value)

    @classmethod
    def unset(cls, name):
        """ 删除 """
        from envariable import unsetenv
        unsetenv(name)
