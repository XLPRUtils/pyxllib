#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 11:00

import concurrent.futures
import math
import sys
import time
from threading import Thread

from loguru import logger

from pyxllib.prog.lazyimport import lazy_import

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lazy_import('from tqdm import tqdm')

from pyxllib.prog.debug import format_exception
from pyxllib.prog.run import EmptyPoolExecutor
from pyxllib.text.base import shorten


class GenFunction:
    """ 一般用来生成高阶函数的函数对象

    这个名字可能还不是很精确，后面有想法再改
    """

    @classmethod
    def ensure_func(cls, x, default):
        """ 确保x是callable对象，否则用default初始化 """
        if callable(x):
            return x
        else:
            return default


def first_nonnone(args, judge=None):
    """ 返回第1个满足条件的值

    :param args: 参数清单
    :param judge: 判断器，默认返回第一个非None值，也可以自定义判定函数
    """
    judge = GenFunction.ensure_func(judge, lambda x: x is not None)
    for x in args:
        if judge(x):
            return x
    return args[-1]  # 全部都不满足，返回最后一个值


class Iterate:
    """ 迭代器类，用来封装一些特定模式的for循环操作

    TODO 双循环，需要内部两两对比的迭代功能

    200920周日18:20，最初设计的时候，是提供run_pair、run_pair2的功能的
        不过后来想想，这个其实就是排列组合，在itertools里有combinations, permutations可以代替
        甚至有放回的组合也有combinations_with_replacement，我实在是不需要再这里写这些冗余的功能
        所以就移除了
    """

    def __init__(self, items):
        # 没有总长度倒也能接受，关键是可能要用start、end切片，所以还是先转成tuple更方便操作
        self.items = tuple(items)
        self.n_items = len(self.items)
        self.format_width = math.ceil(math.log10(self.n_items + 1))

    def run(self, func, start=0, end=None, pinterval=None, max_workers=1, interrupt=True):
        """
        :param func: 对每个item执行的功能
        :param start: 跳过<start的数据，只处理>=start编号以上
        :param end: 只处理 < end 的数据
        :param pinterval: 每隔多少条目输出进度日志，默认不输出进度日志（但是错误日志依然会输出）
            支持按百分比进度显示，例如每20%，pinterval='20%'，不过一些底层实现机制原因，会有些许误差
            TODO 支持按指定时间间隔显示？ 例如每15秒，pinterval='15s' 感觉这种功能太花哨了，没必要搞
        :param max_workers: 默认线程数，默认1，即串行
        :type max_workers: int, None
        :param interrupt: 出现错误时是否中断，默认True会终止程序，否则只会输出错误日志
        :return:
        """
        # 1. 参数处理
        # 1.1 范围切片
        if start is None: start = 0
        if end is None: end = len(self.items)
        n_tasks = end - start
        if n_tasks <= 0: return

        # 1.2 进度间隔
        if isinstance(pinterval, str) and pinterval.endswith('%'):
            pinterval = int(round(self.n_items * float(pinterval[:-1]) / 100))

        if pinterval:
            sys.stdout.flush()
            logger.info(
                f"使用 {func.__name__} 处理 {self.n_items} 个数据 (idx {start} to {end}) {shorten(str(self.items), 30)}")
            if max_workers != 1:
                logger.info(f'多线程执行，线程数：{max_workers}')

        # 2. 执行任务
        from humanfriendly import format_timespan
        start_time = time.time()
        error = None

        def safe_func(idx):
            item = self.items[idx]
            try:
                func(item)
            except Exception as e:
                nonlocal error
                error = e
                logger.error(f'💔idx={idx}运行出错：{item}\n{format_exception(e)}')
                if interrupt:
                    raise e

        try:
            if max_workers == 1:
                # 串行执行
                for i in range(start, end):
                    safe_func(i)
                    self._log_progress(i, pinterval)
            else:
                # 并行执行
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(safe_func, i): i for i in range(start, end)}
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        future.result()  # 触发异常
                        # 注意：多线程下进度打印可能乱序，这里简单按完成数量打印
                        self._log_progress(start + i, pinterval)
        except Exception:
            if interrupt:
                raise

        # 3. 结束统计
        end_time = time.time()
        span = end_time - start_time
        msg = f'总用时：{format_timespan(span)}'
        if span > 0:
            speed = n_tasks / span
            msg += f'，速度：{speed:.2f}it/s'

        if pinterval:
            logger.info(f'100% 完成迭代，{msg}')
            sys.stderr.flush()

    def _log_progress(self, idx, pinterval):
        if pinterval and (idx or pinterval == 1) and idx % pinterval == 0:
            message = f' {self.items[idx]}' if pinterval == 1 else ''
            logger.info(f'{idx:{self.format_width}d}/{self.n_items}={idx / self.n_items:6.2%}{message}')


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
