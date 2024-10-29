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

import concurrent.futures
import os
import re
import subprocess
import time
from statistics import mean
from threading import Thread

from tqdm import tqdm
import requests
from humanfriendly import parse_size

from pyxllib.prog.newbie import human_readable_size
from pyxllib.prog.pupil import get_installed_packages, aligned_range, percentage_and_value
from pyxllib.prog.xlosenv import XlOsEnv
from pyxllib.file.specialist import cache_file


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
    import sys
    from pyxllib.file.specialist import XlPath

    # 1 切换工作目录
    os.chdir(str(root))

    # 2 改版本号
    if version:
        f = XlPath(root) / version_file
        s = re.sub(r"(version\s*=\s*)(['\"])(.+?)(\2)", fr'\1\g<2>{version}\4', f.read_text())
        f.write_text(s)

    # 3 打包
    subprocess.run(f'{sys.executable} setup.py sdist')

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


def estimate_package_size(package):
    """ 估计一个库占用的存储大小 """

    # 将cache文件存储到临时目录中，避免重复获取网页
    def get_size(package):
        r = requests.get(f'https://pypi.org/project/{package}/#files')
        if r.status_code == 404:
            return '(0 MB'  # 找不到的包默认按0MB计算
        else:
            return r.text

    s = cache_file(package + '.pypi', lambda: get_size(package))
    # 找出所有包大小，计算平均值作为这个包大小的预估
    # 注意，这里进位是x1000，不是x1024
    v = mean(list(map(parse_size, re.findall(r'\((\d+(?:\.\d+)?\s*\wB(?:ytes)?)', s))) or [0])
    return v


def estimate_pip_packages(*, print_mode=False):
    """ 检查pip list中包的大小，从大到小排序

    :param print_mode:
        0，不输出，只返回运算结果，[(package_name, package_size), ...]
        1，输出最后的美化过的运算表格
        2，输出中间计算过程
    """

    def printf(*args, **kwargs):
        # dm表示mode增量
        if print_mode > 1:
            print(*args, **kwargs)

    packages = get_installed_packages()
    package_sizes = []
    for package_name in packages:
        package_size = estimate_package_size(package_name)
        package_sizes.append((package_name, package_size))
        printf(f"{package_name}: {human_readable_size(package_size)}")

    package_sizes.sort(key=lambda x: (-x[1], x[0]))
    if print_mode > 0:
        if print_mode > 1: print('- ' * 20)
        for package_name, package_size in package_sizes:
            print(f"{package_name}: {human_readable_size(package_size)}")
    return package_sizes


class ProgressBar:
    """ 对运行可能需要较长时间的任务，添加进度条显示

    # 示例用法
    with ProgressBar(100) as pb:
        for i in range(100):
            time.sleep(0.1)  # 模拟耗时工作
            pb.progress = i + 1  # 更新进度
    """

    def __init__(self, total):
        self.total = total  # 总进度
        self.progress = 0  # 当前进度
        self.stop_flag = False  # 停止标志

    def __enter__(self):
        # 启动进度显示线程
        self.progress_thread = Thread(target=self.display_progress)
        self.progress_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 强制将进度设置为100%
        self.progress = self.total
        # 停止进度显示线程
        self.stop_flag = True
        self.progress_thread.join()

    def display_progress(self):
        with tqdm(total=self.total) as pbar:
            while not self.stop_flag:
                pbar.n = self.progress
                pbar.refresh()
                time.sleep(1)
            pbar.n = self.progress
            pbar.refresh()


class BitMaskTool:
    """ 二进制位掩码工具

    概念术语
        bitval，每一个位上的具体取值，0或1
        bitsum，所有位上的取值的和，即二进制数的十进制表示
    """

    def __init__(self, bit_names=None, bitsum_counter=None):
        """ 初始化 BitMaskTool 对象

        :param list bit_names: 每一位功能开启时，显示的标签名。未输入时，填充0,1,2,3...注意总数，要按域宽对齐
        :param dict/list bitsum_counter: 各种值出现的次数，可以是一个字典或者一个包含出现次数的列表
        """
        # 1 每一位功能开启时，显示的标签名。未输入时，填充0,1,2,3...注意总数，要按域宽对齐
        if bit_names is None:
            bit_names = list(aligned_range(len(bit_names)))
        self.bit_names = bit_names
        # 2 各种值出现的次数
        if isinstance(bitsum_counter, (list, tuple)):
            bitsum_counter = Counter(bitsum_counter)
        self.bitsum_counter = bitsum_counter or {}

    def get_bitsum_names(self, bitsum):
        """ 从bitsum得到names的拼接

        >> get_bitsum_names(3)
        '语义定位,图表'
        """
        if not isinstance(bitsum, int):
            try:
                bitsum = int(bitsum)
            except ValueError:
                bitsum = 0

        tags = []
        for i, k in enumerate(self.bit_names):
            if (1 << i) & bitsum:
                tags.append(k)
        return ','.join(tags)

    def count_bitsum_relations(self, target_bitsum, relation='='):
        """ 计算特定关系的 bitsum 数量

        :param int target_bitsum: 目标 bitsum
        :param str relation: 关系类型，可以是 '=', '⊂', '⊃'
            假设bitval对应的bit_names为n1,n2,n3,n4。
            那么bitsum相当于是bitval的一个集合
            比如a={n1,n3,n4}，b={n1,n3}，因为a完全包含b，所以认为a⊃b，或者a⊋b、b⊂a、b⊊a
        :return int: 符合条件的 bitsum 数量
        """
        count = 0
        if relation == '=':
            # 直接计算等于 target_bitsum 的数量
            count = self.bitsum_counter.get(target_bitsum, 0)
        elif relation == '⊂':
            # 计算所有被 target_bitsum 包含的 bitsum 的数量
            for bitsum, num in self.bitsum_counter.items():
                if bitsum and bitsum & target_bitsum == bitsum:
                    count += num
        elif relation == '⊃':
            # 计算所有包含 target_bitsum 的 bitsum 的数量
            for bitsum, num in self.bitsum_counter.items():
                if bitsum & target_bitsum == target_bitsum:
                    count += num
        return count

    def check_bitflag(self, max_bitsum_len=None, reletion='=',
                      filter_zero=False, sort_by=None, *,
                      min_bitsum_len=0):
        """ 检查并返回 bitsum 关系的 DataFrame

        :param int max_bitsum_len: 最大 bitsum 长度
        :param str reletion: 关系类型，可以是 '=', '⊂', '⊃'
            支持输入多个字符，表示要同时计算多种关系
        :param bool filter_zero: 是否过滤掉零值
        :param None|str sort_by: 排序字段
            None, 默认排序
            count, 按照数量从大到小排序
            bitsum, 按照 bitsum 从小到大排序
        :param int min_bitsum_len: 最小 bitsum 长度
        :return: 包含 bitsum 关系的 DataFrame
        """
        from itertools import combinations

        total = sum(self.bitsum_counter.values())
        rows, columns = [], ['类型', '名称', '百分比.次数']
        rows.append([-1, '总计', total])

        if max_bitsum_len is None:
            max_bitsum_len = len(self.bit_names)

        bitvals = [(1 << i) for i in range(len(self.bit_names))]
        for m in range(min_bitsum_len, max_bitsum_len + 1):
            for comb in combinations(bitvals, m):
                bitsum = sum(comb)
                count = self.count_bitsum_relations(bitsum, relation=reletion)
                if filter_zero and count == 0:
                    continue
                rows.append([f'{reletion}{bitsum}',
                             self.get_bitsum_names(bitsum),
                             count])

        if sort_by == 'count':
            rows.sort(key=lambda x: x[2], reverse=True)
        elif sort_by == 'bitsum':
            rows.sort(key=lambda x: int(x[0][1:]) if isinstance(x[0], str) else x[0])

        df = pd.DataFrame.from_records(rows, columns=columns)
        df['百分比.次数'] = percentage_and_value(df['百分比.次数'], 2, total=total)
        return df

    def report(self):
        """ 生成统计报告 """
        html_content = []

        html_content.append('<h1>1 包含每一位bitval特征的数量</h1>')
        df1 = self.check_bitflag(1, '⊃')
        html_content.append(df1.to_html())

        html_content.append('<h1>2 每一种具体bitsum组合的数量</h1>')
        df2 = self.check_bitflag(reletion='=', filter_zero=True, sort_by='bitsum')
        html_content.append(df2.to_html())

        return '\n'.join(html_content)


def loguru_setup_jsonl_logfile(logger, log_dir, rotation_size="10 MB"):
    """
    给loguru的日志器添加导出文件的功能，使用jsonl格式

    :param logger: 日志记录器，一般是from loguru import logger的logger
    :param log_dir: 存储日志的目录，因为有多个文件，这里要输入的是所在的目录
    :param rotation_size: 文件多大后分割
    :return:
    """
    from datetime import datetime

    os.makedirs(log_dir, exist_ok=True)  # 自动创建日志目录

    # 日志文件名匹配的正则表达式，格式为 年月日_时分秒.log
    log_filename_pattern = re.compile(r"(\d{8}_\d{6})\.jsonl")

    # 找到最新的日志文件
    def find_latest_log_file(log_dir):
        log_files = []
        for file in os.listdir(log_dir):
            if log_filename_pattern.match(file):
                log_files.append(file)

        if log_files:
            # 根据时间排序，选择最新的日志文件
            log_files.sort(reverse=True)
            return os.path.join(log_dir, log_files[0])
        return None

    # 检查是否有未写满的日志文件
    latest_log_file = find_latest_log_file(log_dir)

    if latest_log_file:
        log_path = latest_log_file
    else:
        # 生成新的日志文件名
        log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jsonl"
        log_path = os.path.join(log_dir, log_filename)

    # 配置 logger，写入日志文件，设置旋转条件，使用 JSON 序列化
    logger.add(log_path, rotation=rotation_size, serialize=True)

    # 输出初始化成功信息
    logger.info(f"日志系统已初始化，日志文件路径：{log_path}")
