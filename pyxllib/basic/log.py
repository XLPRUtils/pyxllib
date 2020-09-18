#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/09/18 22:16

import traceback
import concurrent.futures

from .timer import *
from pyxllib.basic.strlib import *
from pyxllib.basic.pathlib_ import *

XLLOG_CONF_FILE = 'xllog.yaml'


def get_xllog():
    """ 获得pyxllib库的日志类

    由于日志类可能要读取yaml配置文件，需要使用Path类，所以实现代码先放在pathlib_.py

    TODO 类似企业微信机器人的机制怎么设？或者如何配置出问题发邮件？
    """
    import logging

    if 'pyxllib.xllog' in logging.root.manager.loggerDict:
        # 1 判断xllog是否已存在，直接返回
        pass
    elif os.path.isfile(XLLOG_CONF_FILE):
        # 2 若不存在，尝试在默认位置是否有自定义配置文件，读取配置文件来创建
        import logging.config
        data = Path(XLLOG_CONF_FILE).read()
        if isinstance(data, dict):
            # 推荐使用yaml的字典结构，格式更简洁清晰
            logging.config.dictConfig(data)
        else:
            # 但是普通的conf配置文件也支持
            logging.config.fileConfig(XLLOG_CONF_FILE)
    else:
        # 3 否则生成一个非常简易版的xllog
        # TODO 不同级别能设不同的格式（颜色）？
        xllog = logging.getLogger('pyxllib.xllog')
        xllog.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S'))
        xllog.addHandler(ch)
    return logging.getLogger('pyxllib.xllog')


def format_exception(e):
    return ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))


class Iterate:
    """ 迭代器类，用来封装一些特定模式的for循环操作

    TODO 双循环，需要内部两两对比的迭代功能
    """

    def __init__(self, items):
        # 没有总长度倒也能接受，关键是可能要用start、end切片，所以还是先转成tuple更方便操作
        self.items = tuple(items)
        self.n_items = len(self.items)
        self.format_width = math.ceil(math.log10(self.n_items + 1))
        self.xllog = get_xllog()

    def _step1_check_number(self, pinterval, func):
        if pinterval:
            sys.stdout.flush()  # 让逻辑在前的标准输出先print出来，但其实这句也不一定能让print及时输出的~~可能会被日志提前抢输出了
            self.xllog.info(f"使用 {func} 处理 {self.n_items} 个数据 {shorten(str(self.items), 30)}")

    def _step2_check_range(self, start, end):
        if start:
            self.xllog.info(f"使用start参数，只处理≥{start}的条目")
        else:
            start = 0
        if end:
            # 这里空格是为了对齐，别删
            self.xllog.info(f"使用 end 参数，只处理<{end}的条目")
        else:
            end = len(self.items)
        return start, end

    def _step3_executor(self, pinterval, max_workers):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers)
        if executor._max_workers != 1:
            if pinterval:
                self.xllog.info(f'多线程执行，当前迭代所用线程数：{executor._max_workers}')
        return executor

    def _step4_iter(self, i, pinterval, executor):
        if pinterval and (i or pinterval == 1) and i % pinterval == 0:
            message = f' {self.items[i]}' if pinterval == 1 else ''
            self.xllog.info(f'{i:{self.format_width}d}/{self.n_items}={i / self.n_items:6.2%}{message}')
        # 队列中没有新任务时，才放入新任务，这样能确保pinterval的输出能反应实时情况，而不是一下全部进入队列，把for循环跑完了
        while executor._work_queue.qsize(): pass

    def _step5_finish(self, pinterval, interrupt, executor):
        executor.shutdown()
        if not interrupt and pinterval:
            self.xllog.info(f'{self.n_items:{self.format_width}d}/{self.n_items}='
                            f'{self.n_items / self.n_items:6.2%} 完成迭代')
            sys.stderr.flush()

    def run(self, func, start=0, end=None, pinterval=None, max_workers=1, interrupt=True):
        """
        :param func: 对每个item执行的功能
        :param start: 跳过<start的数据，只处理>=start编号以上
        :param end: 只处理 < end 的数据
        :param pinterval: 每隔多少条目输出进度日志，默认不输出进度日志（但是错误日志依然会输出）
            TODO 支持按百分比进度显示？  例如每20%，pinterval='20%'
            TODO 支持按指定时间间隔显示？ 例如每15秒，pinterval='15s'
        :param max_workers: 默认线程数，默认1，即串行
        :type max_workers: int, None
        :param interrupt: 出现错误时是否中断，默认True会终止程序，否则只会输出错误日志
        :return:
        """

        # 1 统一的参数处理部分
        self._step1_check_number(pinterval, func)
        start, end = self._step2_check_range(start, end)
        error = False
        executor = self._step3_executor(pinterval, max_workers)

        # 2 封装的子处理部分
        def wrap_func(func, i):
            nonlocal error
            item = self.items[i]
            try:
                func(item)
            except Exception as e:
                error = True
                self.xllog.error(f'💔idx={i}运行出错：{item}\n{format_exception(e)}')

        # 3 执行迭代
        for i in range(start, end):
            self._step4_iter(i, pinterval, executor)
            executor.submit(wrap_func, func, i)
            if interrupt and error: break
        self._step5_finish(pinterval, interrupt and error, executor)

    def run_pair(self, func, start=0, end=None, pinterval=None, max_workers=1, interrupt=True):
        """ 对items两两运算
            func(x, y) 等同于 func(y, x)，不重复运算

        :param start: 这里的start、end特指第一层迭代器i的取值范围

        TODO starti, endi, startj, endj，i和j支持单独设置遍历区间？
        """
        # 1 统一的参数处理部分
        self._step1_check_number(pinterval, func)
        start, end = self._step2_check_range(start, end)
        error = False
        executor = self._step3_executor(pinterval, max_workers)

        # 2 封装的子处理部分
        def wrap_func(func, i, j):
            nonlocal error
            item1, item2 = self.items[i], self.items[j]
            try:
                func(item1, item2)
            except Exception as e:
                error = True
                self.xllog.error(f'💔idxs=({i},{j})运行出错：{item1},{item2}\n{format_exception(e)}')

        # 3 执行迭代
        for i in range(start, end):
            self._step4_iter(i, pinterval, executor)
            for j in range(i + 1, self.n_items):
                executor.submit(wrap_func, func, i, j)
                if interrupt and error: break
        self._step5_finish(pinterval, interrupt and error, executor)

    def run_pair2(self, func, start=0, end=None, pinterval=None, max_workers=1, interrupt=True):
        """ 对items两两运算
            func(x, y) 不同于 func(y, x)，需要全量运算

        :param start: 这里的start、end特指第一层迭代器i的取值范围
        """
        # 1 统一的参数处理部分
        self._step1_check_number(pinterval, func)
        start, end = self._step2_check_range(start, end)
        error = False
        executor = self._step3_executor(pinterval, max_workers)

        # 2 封装的子处理部分
        def wrap_func(func, i, j):
            nonlocal error
            item1, item2 = self.items[i], self.items[j]
            try:
                func(item1, item2)
            except Exception as e:
                error = True
                self.xllog.error(f'💔idxs=({i},{j})运行出错：{item1},{item2}\n{format_exception(e)}')

        # 3 执行迭代
        for i in range(start, end):
            self._step4_iter(i, pinterval, executor)
            for j in range(self.n_items):
                if j == i: continue
                executor.submit(wrap_func, func, i, j)
                if interrupt and error: break
        self._step5_finish(pinterval, interrupt and error, executor)
