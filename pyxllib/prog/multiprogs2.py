#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/12

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock
import ctypes
import datetime
import os
import socketserver
import subprocess
import sys
import time
import textwrap
import threading
import re
import itertools

from pyxllib.prog.lazyimport import lazy_import

try:
    from deprecated import deprecated
except ModuleNotFoundError:
    deprecated = lazy_import('deprecated', 'Deprecated')

try:
    from fastcore.basics import GetAttr
except ModuleNotFoundError:
    GetAttr = lazy_import('from fastcore.basics import GetAttr', 'fastcore')

try:
    from loguru import logger
except ModuleNotFoundError:
    logger = lazy_import('from loguru import logger')

try:
    from croniter import croniter
except ModuleNotFoundError:
    croniter = lazy_import('from croniter import croniter')

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import('pandas')

try:
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse, HTMLResponse
except ModuleNotFoundError:
    FastAPI = lazy_import('from fastapi import FastAPI')
    PlainTextResponse = lazy_import('from fastapi.responses import PlainTextResponse')

try:
    from apscheduler import Scheduler
except ModuleNotFoundError:
    Scheduler = lazy_import('from apscheduler import Scheduler',
                            'apscheduler[psycopg,sqlalchemy]==4.0.0a6')

from pyxllib.prog.specialist import parse_datetime
from pyxllib.algo.stat import print_full_dataframe
from pyxllib.file.specialist import XlPath


def __1_定时工具():
    pass


class XlTrigger:
    """ 对现有触发器的规则组合 """

    def __1_类方法(self):
        pass

    @classmethod
    def calculate_future_time(cls, start_time, wait_seconds):
        """ 计算延迟时间

        :param datetime start_time: 开始时间
        :param int wait_seconds: 等待秒数
            todo 先只支持秒数这种标准秒数，后续可以考虑支持更多智能的"1小时"等这种解析
        """
        return start_time + datetime.timedelta(seconds=wait_seconds)

    @classmethod
    def calculate_next_cron_time(cls, cron_tag, base_time=None):
        """ 使用crontab标记的运行周期，然后计算相对当前时间，下一次要启动运行的时间

        :param str cron_tag: 自定义的cron标记，跟asp的差不多，但星期几部分做了调整
            30 2 * * 1: 这部分是时间和日期的设定，具体含义如下：
                30: 表示分钟，即每小时的第 30 分钟。
                2: 表示小时，即凌晨 2 点。
                第三个星号 *: 表示日，这里的星号意味着每天。
                第四个星号 *: 表示月份，星号同样表示每个月。
                1: 表示星期中的日子，这里的 1 代表星期一，7表示星期日。不能写1~7以外的值。
        :param datetime base_time: 基于哪个时间点计算下次时间
        """

        # 如果没有提供基准时间，则使用当前时间
        if base_time is None:
            base_time = datetime.datetime.now()
        # 初始化 croniter 对象
        cron = croniter(cron_tag, base_time)
        # 计算下一次运行时间
        next_time = cron.get_next(datetime.datetime)
        return next_time

    @classmethod
    def wait_until_time(cls, dst_time):
        """
        :param datetime dst_time: 一直等待到目标时间
            期间可以用time.sleep进行等待
        """
        # 一般来说，只要计算一轮待等待秒数就行。但是time.sleep机制好像不一定准确的，所以使用无限循环重试会更好。
        while True:
            # 先计算当前时间和目标时间的相差秒数
            wait_seconds = (dst_time - datetime.datetime.now()).total_seconds()
            if wait_seconds <= 0:
                break
            time.sleep(min(1, wait_seconds))  # 分段休眠，避免长时间阻塞

    @classmethod
    def smart_wait(cls, start_time, end_time, wait_tag, print_mode=0):
        """ 智能等待，一般用在对进程的管理重启上

        :param datetime start_time: 程序启动的时间
        :param datetime end_time: 程序结束的时间
        :param str|float|int wait_tag: 等待标记
            str，按crontab解析
                在end_time后满足条件的下次时间重启
            int|float，表示等待的秒数
                正值是end_time往后等待，负值是start_time开始计算下次时间。
                比如1点开始的程序，等待半小时，但是首次运行到2点才结束
                    那么正值就是2:30再下次运行
                    但是负值表示1:30就要运行，已经错过了，马上2点结束就立即启动复跑
        """
        # 1 尝试把wait_tag转成数值
        try:
            wait_tag = float(wait_tag)
        except ValueError:  # 转不成也没关系
            pass

        if start_time is None:
            start_time = datetime.datetime.now()
        if end_time is None:
            end_time = datetime.datetime.now()

        # 2 计算下一次启动时间
        if isinstance(wait_tag, str):
            # 按照crontab解析
            next_time = cls.calculate_next_cron_time(wait_tag, end_time)
        elif wait_tag >= 0:
            # 正值则是从end_time开始往后等待
            next_time = cls.calculate_future_time(end_time, wait_tag)
        elif wait_tag < 0:
            # 负值则是从start_time开始往前等待
            next_time = cls.calculate_future_time(start_time, wait_tag)
        else:
            raise ValueError

        if print_mode:
            print(f'等待到时间{next_time}...')

        cls.wait_until_time(next_time)

    def __2_对象方法(self):
        pass

    def __init__(self, desc=None):
        """
        :param desc: 通过一个参数，就可以描述多种触发规则
            int/float, 每间隔多少秒运行一次，以启动时间为基准
                可以输入负值，此时表示是在程序实际结束后等待的秒数时间
            str，cron格式的定时器
            None，默认值，指当前时间
            datetime，手动指定某个运行时间点
            list[datetime], 手动指定的多个时间段，一般用的不多
        """
        self.desc = desc

    def get_next_fire_time(self):
        """ 返回下一次应该运行的时间datetime，如果已经不需要运行，返回None """
        pass


def __2_进程作业():
    pass


def set_pdeathsig(sig=None):
    """ 在主服务退出时，这些程序也会全部自动关闭 """

    def callable():
        import signal
        sig2 = signal.SIGTERM if sig is None else sig
        libc = ctypes.CDLL("libc.so.6")
        return libc.prctl(1, sig2)

    if sys.platform == 'win32':
        # windows系统暂设为空
        return None
    else:
        return callable


class SubprocessTask:
    """ subprocess类型的job作业
    即subprocess+apscheduler后需要的基础作业类型
    """

    def __init__(self,
                 args,
                 *,
                 name=None,
                 popen_kwargs=None,
                 timeout_seconds=None,
                 trigger=None,
                 ):
        """
        :param args: 要运行的指令程序
        :param name: 程序昵称
        :param popen_kwargs: popen初始化用的其他参数
        :param timeout_seconds: 超时设置，单位秒
        """
        self.scheduler = None

        # 执行程序需要使用的参数
        self.args = args

        # 关键参数
        self.name = name
        self.popen_kwargs = popen_kwargs or {}
        self.proc = None
        self.timeout_seconds = timeout_seconds
        self.trigger = XlTrigger(trigger) if trigger else None

        # 运行记录 list[dict]结构
        # dict: time 时间点, event 事件(启动, 完成, 中断)
        self.records = []

    def is_running(self):
        """ 检查程序是否在运行 """
        status = self.proc is not None and self.proc.poll() is None
        return status

    def add_record(self, event, **kwargs):
        self.records.append({'time': datetime.datetime.now().isoformat(timespec='seconds'),
                             'event': event,  # 启动|完成|中断
                             **kwargs,
                             })

    def status_tag(self):
        """ 当前程序的状态标签

        时间戳 + "待启动|已启动|已完成|被中断"
        """

    @classmethod
    def from_dict(cls, dict_data):
        """ 从字典读取配置数据来创建SubprocessTask实例

        :param dict_data: 配置字典
        :return: SubprocessTask实例
        """
        # 创建实例
        instance = cls(
            args=dict_data.get('args'),
            name=dict_data.get('name'),
            popen_kwargs=dict_data.get('popen_kwargs'),
            timeout_seconds=dict_data.get('timeout_seconds')
        )
        instance.records = dict_data.get('records', [])
        return instance

    def to_dict(self):
        """导出配置参数为字典，与from_dict保持对称"""
        config = {
            'args': self.args,
            'name': self.name,
            'popen_kwargs': self.popen_kwargs,
            'timeout_seconds': self.timeout_seconds,
            'records': self.records
        }
        return config

    def __call__(self):
        """ 启动程序 """
        # 1 目前不支持重复执行，只允许有一个实例，但要扩展修改这个功能是相对容易的
        if self.is_running():
            return self.proc

        # 2 记录并启动
        self.proc = subprocess.Popen(self.args, preexec_fn=set_pdeathsig(), **self.popen_kwargs)
        self.add_record('启动',
                        args=self.args,
                        pid=self.proc.pid,
                        popen_kwargs=self.popen_kwargs,
                        timeout_seconds=self.timeout_seconds,
                        )

        # 3 等待其运行结束后标记
        try:
            # 但proc启动后，使用阻塞模式监控程序什么时候结束
            stdout_data, stderr_data = self.proc.communicate(input=None, timeout=self.timeout_seconds)
            # stdout_data只有在Popen时配置stdout=subprocess.PIPE时才能获取，否则输出是直接到主进程的命令行窗口里，这里只能得到None
            self.add_record(
                '中断' if self.proc.returncode else '完成',
                pid=self.proc.pid,
                returncode=self.proc.returncode,
                stdout=stdout_data,
                stderr=stderr_data,
            )

        except subprocess.TimeoutExpired:
            self.kill()
            stdout_data, stderr_data = self.proc.communicate()
            self.add_record(
                '超时',
                pid=self.proc.pid,
                returncode=self.proc.returncode,
                stdout=stdout_data,
                stderr=stderr_data,
            )

        return self.proc

    def terminate(self):
        """ 比较优雅结束进程的方法 """
        if self.is_running():
            self.proc.terminate()
            self.proc.communicate()  # 等待程序确实退出了

    def kill(self):
        """ 有时候需要强硬的kill方法来结束进程 """
        if self.is_running():
            self.proc.kill()
            self.proc.communicate()  # 等待程序确实退出了

    def restart(self):
        self.kill()
        self.__call__()


def __3_后端服务作业():
    pass


def find_free_ports(count=1):
    """ 随机获得可用端口

    :param count: 需要的端口数量（会保证给出的端口号不重复）
    :return: list
    """
    ports = set()
    while len(ports) < count:
        with socketserver.TCPServer(("localhost", 0), None) as s:
            ports.add(s.server_address[1])

    return list(ports)


class XlServerSubprocessTask(SubprocessTask):
    """ xlserver后端服务所用进程作业 """
    pass


def __4_调度器():
    pass


class XlScheduler(GetAttr):
    """ 对apscheduler的Scheduler扩展版
	"""
    _default = 'scheduler'

    def __init__(self):
        self._scheduler = None
        self.jobs = []

    @property
    def scheduler(self):
        """ 调度器 """
        if self._scheduler is None:
            self._scheduler = Scheduler()

        return self._scheduler

    def add_job_cmd(self, args):
        pass


def some_job():
    time.sleep(10)


if __name__ == '__main__':
    from apscheduler.triggers.date import DateTrigger

    scheduler = XlScheduler()
    # scheduler._scheduler = BlockingScheduler(pause=True)

    job = SubprocessTask([sys.executable, '-c', 'import time; time.sleep(10); print(123)'], timeout_seconds=5)
    # scheduler.add_job(some_job, trigger=DateTrigger(run_date='2025-10-02 20:00:00'))
    scheduler.add_job(job)
    scheduler.start()

    # scheduler.print_jobs()

    # time.sleep(5)
    # job.restart()
    time.sleep(12)

    print(job.records)
