#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/12

import inspect
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
from functools import partial
import copy

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
    # 我其实大部分使用场景都是后台 BackgroundScheduler
    # from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.schedulers.background import BackgroundScheduler

    # 3.11版本无apscheduler.abc.Trigger，直接从triggers.base导入Trigger基类
    # from apscheduler.triggers.base import Trigger

    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.date import DateTrigger
    from apscheduler.triggers.cron import CronTrigger
except ModuleNotFoundError:
    BaseScheduler = lazy_import('from apscheduler.schedulers.base import BaseScheduler',
                                'apscheduler==3.11.1')

# 为了兼容funboost，还不能把aps升级到4.x版本
# try:
#     from apscheduler import Scheduler, TaskDefaults
#     from apscheduler.abc import Trigger
#     from apscheduler.triggers.interval import IntervalTrigger
#     from apscheduler.triggers.date import DateTrigger
#     from apscheduler.triggers.cron import CronTrigger
# except ModuleNotFoundError:
#     Scheduler = lazy_import('from apscheduler import Scheduler',
#                             'apscheduler[psycopg,sqlalchemy]==4.0.0a6')

try:
    import uvicorn
except ModuleNotFoundError:
    uvicorn = lazy_import('import uvicorn')

from pyxllib.prog.specialist import parse_datetime
from pyxllib.algo.stat import print_full_dataframe
from pyxllib.file.specialist import XlPath


def __1_定时():
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

    def __2_扩展触发器(self):
        """ 用create_trigger可以标准触发器，这里还提供一些其他特殊的非标准触发器 """
        pass

    @classmethod
    def create_cron_trigger(cls, desc):
        """ 自定义扩展过的cron表达式
        主要是第6位的星期标记，直接用1~7来表示星期一到星期日，比较直观
        """
        cron_parts = desc.split()

        # 如果是 5 个字段，则补齐 "秒" 字段为 '0'
        if len(cron_parts) == 5:
            cron_parts.insert(0, '0')

        # 检查是否为有效的 6 字段 cron 表达式
        if len(cron_parts) == 6:
            # 统一处理为 6 字段格式
            # 把我自定义的cron的星期标记转换为aps的星期标记。前者用1234567，后者用0123456表示星期一到星期日
            x = cron_parts[5]
            if x != '*':  # 写0或7都表示周日
                # 改成正则获取x每个数值减去1（遇到0则改为6）：相当于把原本1~7的周标记，改为这里0~6的标记
                def f(m):
                    x = int(m.group(0))
                    return '6' if x == 0 else str(x - 1)

                x = re.sub(r'\d+', f, x)

            return CronTrigger(
                second=cron_parts[0],
                minute=cron_parts[1],
                hour=cron_parts[2],
                day=cron_parts[3],
                month=cron_parts[4],
                day_of_week=x,
            )
        else:
            raise ValueError(f"无效的 cron 表达式：{desc}")

        # 使用apscheduler4.0.0a6，可以改用下述更简洁版本
        # # 检查是否为有效的 6 字段 cron 表达式
        # if len(cron_parts) == 6:
        #     return CronTrigger(
        #         second=cron_parts[0],
        #         minute=cron_parts[1],
        #         hour=cron_parts[2],
        #         day=cron_parts[3],
        #         month=cron_parts[4],
        #         day_of_week=cron_parts[5],  # 取值[0,7]，1~6表示周一~周六，0,7都表示周日
        #     )
        # else:
        #     raise ValueError(f"无效的 cron 表达式：{desc}")

    @classmethod
    def post_execution_schedule(cls, scheduler, task, interval_seconds):
        task()
        next = datetime.datetime.now() + datetime.timedelta(seconds=interval_seconds)
        scheduler.add_schedule(cls.post_execution_schedule, DateTrigger(next), args=[scheduler, task, interval_seconds])


def __2_任务():
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


class SubprocessTask:
    """ subprocess类型的job作业
    即subprocess+apscheduler后需要的基础作业类型
    """

    def __init__(self,
                 args,
                 *,
                 name=None,
                 timeout_seconds=None,
                 **popen_kwargs,
                 ):
        """
        :param str|list args: 要运行的指令程序
        :param name: 程序昵称
        :param timeout_seconds: 超时设置，单位秒
        :param popen_kwargs: popen初始化用的其他参数，常用的比如
            shell，默认False不使用shell直接运行，一般用在一些不需要管道等的场景
            stdin, stdout, stderr，标准输入、输出、错误流
        """
        self.scheduler = None
        # aps4.0.0a6使用仿函数机制的话还有些瑕疵不兼容，必须要配置这个值，不然会运行不了
        # 而且这个也算是唯一标识，必须不同实例值不同，才能确保aps4不会判别为同一task
        self.__qualname__ = str(id(self))

        # 执行程序需要使用的参数
        self.args = args

        # 关键参数
        self.name = name
        self.popen_kwargs = popen_kwargs or {}
        self.popen_kwargs['stdin'] = subprocess.PIPE
        self.proc = None
        self.timeout_seconds = timeout_seconds

    def is_running(self):
        """ 检查程序是否在运行 """
        status = self.proc is not None and self.proc.poll() is None
        return status

    def set_arg(self, arg_name, arg_value):
        """ 修订arg的值

        注意self.args可能是str或list类型，不要修改args类型的情况设置arg
        如果不存在arg，则添加，否则修改arg的值

        使用示例：
        >> self.set_arg('--port', 5034)
        """
        if isinstance(self.args, str):
            # 按空格拆分，支持简单 key=value 或 --key value 形式
            parts = self.args.split()
            updated = False
            for i, part in enumerate(parts):
                if part == arg_name:
                    # --key value 形式，替换下一个值
                    if i + 1 < len(parts):
                        parts[i + 1] = str(arg_value)
                        updated = True
                        break
                elif part.startswith(f'{arg_name}='):
                    # key=value 形式，直接替换
                    parts[i] = f'{arg_name}={arg_value}'
                    updated = True
                    break
            if not updated:
                # 不存在则追加，优先尝试 --key value 风格
                parts.extend([arg_name, str(arg_value)])
            self.args = ' '.join(parts)

        elif isinstance(self.args, list):
            updated = False
            # 扫描 list，尝试找到并替换
            for i, item in enumerate(self.args):
                if item == arg_name:
                    # --key value 形式
                    if i + 1 < len(self.args):
                        self.args[i + 1] = str(arg_value)
                        updated = True
                        break
                elif item.startswith(f'{arg_name}='):
                    # key=value 形式
                    self.args[i] = f'{arg_name}={arg_value}'
                    updated = True
                    break
            if not updated:
                # 不存在则追加
                self.args.extend([arg_name, str(arg_value)])

    def run(self):
        """ 启动程序，这个在下游子类中可以重载重新定制 """
        # 1 目前不支持重复执行，只允许有一个实例，但要扩展修改这个功能是相对容易的
        logger.info(self.args)
        if self.is_running():
            return self.proc

        # 2 启动
        self.proc = subprocess.Popen(self.args, preexec_fn=set_pdeathsig(), **self.popen_kwargs)

        # 3 等待其运行结束后标记
        # try:
        #     # proc启动后，使用阻塞模式监控程序什么时候结束
        #     self.proc.communicate(timeout=self.timeout_seconds)
        # except subprocess.TimeoutExpired:
        #     self.kill()
        #     self.proc.communicate()

        return self.proc

    def __call__(self):
        return self.run()

    def terminate(self):
        """ 比较优雅结束进程的方法 """
        if self.is_running():
            self.proc.terminate()
            self.proc.communicate()  # 等待程序确实退出了

    def kill(self):
        """ 有时候需要强硬的kill方法来结束进程 """
        if self.is_running():
            if sys.platform == 'win32':
                # windows里用kill不一定关的掉的，还得用专门的命令暴力点
                subprocess.run(f"taskkill /F /T /PID {self.proc.pid}", shell=True)
            else:
                self.proc.kill()
            self.proc.communicate()  # 等待程序确实退出了

    def restart(self):
        self.kill()
        self.__call__()


class XlServerSubprocessTask(SubprocessTask):
    def __init__(self,
                 args,
                 *,
                 name=None,
                 timeout_seconds=None,
                 **popen_kwargs,
                 ):
        super().__init__(args, name=name, timeout_seconds=timeout_seconds, **popen_kwargs)

        # 我用nginx需要额外补充的两个属性值
        self.port = None
        # list[dict]结构，dict长度只有1，类似 [{'/原url路径/a', '/目标url路径/b'}, ...]
        # 如果k,v相同，可以简写为一个值 ['/a', '/b', ...]
        self.locations = None

    def set_port(self, port):
        self.port = port
        self.set_arg('--port', port)


def __3_调度器():
    pass


class XlScheduler(GetAttr, BackgroundScheduler):
    """ 对apscheduler的Scheduler扩展版
	"""
    _default = 'scheduler'
    _proc_class = SubprocessTask

    def __init__(self):
        self._scheduler = None
        # aps也可以用get_tasks获得全部tasks，但那个task更底层，不太是我需要的更高层业务的tasks管理单元，所以这里我自己扩展一个成员
        self.tasks = []
        self.task_defaults = {}

    @property
    def scheduler(self):
        """ 调度器 """
        if self._scheduler is None:
            self._scheduler = BackgroundScheduler()
        return self._scheduler

    def __1_添加schedule(self):
        pass

    def add_schedule(self, task, trigger=None, **kwargs):
        """

        :param task:
        :param int|float|str|datetime|callable trigger: 自定义的trigger格式
            None，马上开始执行一次，这一般分两种应用场景
                （1）需要马上启动，且只需要启动一次的后端，前端服务
                （2）这里只是一个初始触发器，传入的task是一个二次开发的函数，
                    在原始task外额外补充了一个特殊的trigger，借助scheduler对象会在运行后动态添加任务
            Trigger, 标准触发器
            int/float
                >0, 每间隔多少秒执行一次，是对齐启动时间，也是IntervalTrigger的用法
                <0, 每次程序结束后，等待多久再重复执行，这个是需要特殊机制来实现支持的
            str, cron模式
            datetime, 等待到目标时间启动
            trigger(scheduler, task), 自定义函数触发器。前2个参数是强制配置要传递的，才能实现作业的动态安排，
                但是可以添加额外的默认参数，在函数内部再去灵活扩展功能，参考post_execution_schedule的实现。
            注：还可以用标准触发器的AndTrigger, OrTrigger来设计组合触发器
        :param kwargs: 其他aps原本add_schedule所用参数
            args, kwargs: 这里的参数是给task用的，注意不是给trigger用的，trigger这里的callable模式如果有需要可以使用偏函数等实现。
        :return:
        """
        # 在self.task_defaults数据基础上，叠加kwargs的值作为新的kwargs
        kwargs = {**self.task_defaults, **kwargs}

        if trigger is None:
            if 'id' in kwargs:
                self.scheduler.add_job(task, DateTrigger(datetime.datetime.now()), **kwargs)
            else:
                self.scheduler.add_job(task, **kwargs)
        # elif isinstance(trigger, Trigger):
        #     self.scheduler.add_job(task, trigger, **kwargs)
        elif isinstance(trigger, (int, float)):
            if trigger > 0:
                self.scheduler.add_job(task, trigger, **kwargs)
            elif trigger < 0:
                if 'args' in kwargs or 'kwargs' in kwargs:
                    task = partial(task, *kwargs.pop('args', ()), **kwargs.pop('kwargs', {}))
                self.add_job(XlTrigger.post_execution_schedule, args=[self, task, -trigger], **kwargs)
            else:
                raise ValueError('IntervalTrigger模式下值必须不为0')
        elif isinstance(trigger, str):
            self.scheduler.add_job(task, XlTrigger.create_cron_trigger(trigger), **kwargs)
        elif callable(trigger):
            if 'args' in kwargs or 'kwargs' in kwargs:
                task = partial(task, *kwargs.pop('args', ()), **kwargs.pop('kwargs', {}))
            self.add_job(trigger, args=[self, task], **kwargs)
        else:
            raise ValueError(f'不支持的触发器类型 {type(trigger)}')

    def add_cmd_basic(self, cmd, trigger=None, **kwargs):
        """ 用命令行启动一个程序，或仅存储任务，并添加进管理列表

        :param SubprocessTask|str|list cmd: 启动程序的命令args
            简单场景可以直接传递args命令初始化一个subprocess任务
            复杂场景可以上游先初始化好一个SubprocessTask对象后，再传递进来
        :param trigger: 触发器
        """
        # 1 对kwargs参数分组
        task_kwargs = ['name', 'shell', 'stdin', 'stdout', 'stderr']
        # 把kwargs中的task_kwargs分离出来
        task_kwargs = {k: kwargs.pop(k, None) for k in task_kwargs if k in kwargs}

        # 2 添加任务调度
        task = self._proc_class(cmd, **task_kwargs) if isinstance(cmd, (str, list)) else copy.deepcopy(cmd)
        self.tasks.append(task)
        self.add_schedule(task, trigger, **kwargs)
        return task

    def add_cmd(self, cmd, trigger=None, name=None, **kwargs):
        return self.add_cmd_basic(cmd, trigger, name=name, **kwargs)

    def add_py(self, args, trigger=None, *, executer=None, **kwargs):
        """ 执行一个py任务
        """
        executer = executer or sys.executable
        if isinstance(args, str): args = [args]
        cmd = [executer] + args
        return self.add_cmd(cmd, trigger, **kwargs)

    def add_script_task(self,
                        script_content,
                        extension=None,
                        trigger=None,
                        **kwargs):
        """ 添加一个操作系统命令或脚本（如 .bat、.sh、.ps1 等）并启动。

        :param script_content: 脚本的内容（字符串）
        :param extension: 脚本文件的扩展名（如 .bat, .sh, .ps1）；如果为 None 则根据操作系统自动选择
        """
        # 1 自动选择脚本文件扩展名
        if extension is None:
            if sys.platform == 'win32':
                extension = '.ps1'  # Windows 默认使用 PowerShell
            else:
                extension = '.sh'  # Linux 和 macOS 默认使用 Bash

        # 2 添加编码配置到脚本内容
        if extension == '.bat':
            # 为 .bat 文件添加 UTF-8 支持
            script_content = f"chcp 65001 >nul\n{script_content}"
        elif extension == '.ps1':
            # 为 .ps1 文件添加 UTF-8 支持
            script_content = f"[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n" \
                             f"[Console]::InputEncoding = [System.Text.Encoding]::UTF8\n{script_content}"

        # 3 创建临时脚本文件
        script_file = XlPath.create_tempfile_path(extension)
        script_file.write_text(script_content)

        # 4 根据文件扩展名和操作系统选择执行命令
        if extension == '.sh':
            cmd = ['bash', str(script_file)]
        elif extension == '.ps1':
            if sys.platform == 'win32':
                cmd = ['powershell', '-ExecutionPolicy', 'Bypass', '-File', str(script_file)]
            else:
                raise ValueError("PowerShell 脚本仅在 Windows 系统上受支持")
        elif extension == '.bat':
            cmd = [str(script_file)]  # 直接运行 .bat 文件
        else:
            raise ValueError(f"不支持的脚本类型: {extension}")

        # 5 启动脚本
        return self.add_cmd(cmd, trigger, **kwargs)

    def __2_任务管理(self):
        pass

    def count_running(self):
        return sum(1 for task in self.tasks if task.is_running())

    def list_tasks(self):
        """ 返回所有任务的状态 DataFrame """
        ls = []
        for i, task in enumerate(self.tasks, start=1):
            ls.append({
                'order': i,
                'name': task.name or '',
                'args': task.args,
                'pid': task.proc.pid if task.proc else None,
                'running': True if task.is_running() else '',
                'port': task.port,
                'locations': task.locations or '',
            })
        df = pd.DataFrame(ls)

        # 将可能包含空值的整数字段转换为可空整数类型
        int_columns = ['pid', 'port']  # 以及你认为需要是整数的其他列
        for col in int_columns:
            df[col] = df[col].astype('Int64')  # 注意 'I' 大写

        return df

    def stop_all(self):
        """ 停止所有后台程序 """
        # 关闭所有调度器
        if self.scheduler:
            self.scheduler.stop()

        # 停止所有单启动任务
        for task in self.tasks:
            task.kill()
        self.tasks = []

    def run_with_cli_interaction(self):
        """ 在后台启动程序，并开启命令行交互功能 """
        self.scheduler.start()
        # self.start_in_background()  # 4.x写法
        while True:
            cmd = input('>')
            if cmd in ['stop', 'quit']:
                logger.info("检测到stop，正在终止所有程序...")
                self.scheduler.shutdown()
                # self.stop_all()  # 关闭所有正在运行中的task
                # self.stop()  # 关闭apscheduler
                print("所有程序已终止，退出。")
                break
            elif cmd:
                print(eval(cmd))


class XlServerScheduler(XlScheduler):
    _proc_class = XlServerSubprocessTask

    @property
    def scheduler(self):
        """ 调度器 """
        if self._scheduler is None:
            # misfire_grace_time默认是未配置，设置None表示无论错过多久，都要补跑任务
            # max_running_jobs默认是1，一般不用配置，表示相同任务不允许重复并发跑
            #   但会有特殊情景，不同任务被错判到相同任务，导致这一组只能串行运行的时候，可以开大这个参数
            # task_defaults = TaskDefaults(misfire_grace_time=None)
            # self._scheduler = Scheduler(task_defaults=task_defaults)
            self._scheduler = BackgroundScheduler()
            self.task_defaults = {'misfire_grace_time': None}
        return self._scheduler

    def __1_添加命令行工具(self):
        pass

    def _add_cmd_with_ports(self, cmd, trigger=None, *, name=None, ports=None, **kwargs):
        """ 支持 ports 的处理

        :param ports:
            int 表示要开启的进程数，端口号随机生成
            list 表示指定的端口号
            None 不做特殊配置
        """
        # 1 处理 ports 参数，找到空闲端口或使用指定端口
        if isinstance(ports, int):
            ports = find_free_ports(ports)

        # 2 遍历端口，依次启动进程
        tasks = []
        if ports:
            name = name or ''
            for port in ports:
                task = self.add_cmd_basic(cmd, name=f'{name}:{port}', **kwargs)
                task.set_port(port)
                tasks.append(task)
        else:
            task = self.add_cmd_basic(cmd, trigger, name=name, **kwargs)
            tasks = [task]

        return tasks

    def add_cmd(self, cmd, trigger=None, *, locations=None, **kwargs):
        """
        增强版 add_schedule_cmd_with_ports，支持nginx映射配置
            以前还支持devices，现在删除了，这类参数可以.env文件中配置

        :param locations: URL 映射规则
        :return: 返回一个包含所有启动程序的 task 列表。
        """
        # 1 处理 ports 参数
        tasks = self._add_cmd_with_ports(cmd, trigger, **kwargs)

        # 2 处理locations
        if locations:
            if isinstance(locations, str):
                locations = [locations]
            locations2 = []
            for x in locations:
                if not isinstance(x, dict):
                    locations2.append({x: x})
            for task in tasks:
                task.locations = locations2  # 不需要copy，因为这一组确实是同一个url映射逻辑

        return tasks

    def __2_导出nginx配置(self):
        pass

    def get_all_locations(self):
        locations = defaultdict(list)
        for task in self.tasks:
            if not task.locations:
                continue
            for x in task.locations:
                for dst, src in x.items():
                    if task.port:  # 250126周日21:02，有端口才添加
                        locations[dst].append(f'localhost:{task.port}{src}')
        return locations

    def configure_nginx(self, nginx_template, locations=None):
        if locations is None:
            locations = self.get_all_locations()

        upstreams = []  # 外部的配置
        servers = [nginx_template.rstrip()]  # 内部的配置

        for dst, srcs in locations.items():
            if len(srcs) == 1:  # 只有1个不开负载
                server = f'location {dst} {{\n\tproxy_pass http://{srcs[0]};\n}}\n'
            else:  # 有多个端口功能则开负载
                hosts = '\n'.join([f'\tserver {src.split("/")[0]};' for src in srcs])
                upstream_name = 'upstream' + str(len(upstreams) + 1)
                upstreams.append(f'upstream {upstream_name} {{\n{hosts}\n}}\n')
                sub_urls = [src.split('/', maxsplit=1)[1] for src in srcs]
                assert len(set(sub_urls)) == 1, f'负载均衡的子url必须一致 {sub_urls}'
                server = f'location {dst} {{\n\tproxy_pass http://{upstream_name}/{sub_urls[0]};\n}}\n'
            servers.append(server)

        content = '\n'.join(upstreams) + '\nserver {\n' + textwrap.indent('\n'.join(servers), '\t') + '\n}'
        return content


def __4_看板与交互():
    pass


class SchedulerDashboard:
    def __init__(self, scheduler, *, app=None):
        self.scheduler: XlScheduler = scheduler
        self.app = app or FastAPI()

    def setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def home():
            """ 主页，以美观的HTML表格展示所有任务状态 """
            df = self.scheduler.list_tasks()

            # 这是通用的df表格渲染，不受df结构改变而需要额外修改

            # 使用to_html生成基础表格，并隐藏索引
            basic_html_table = df.to_html(index=False, classes='my-table', border=0, escape=False)

            # 嵌入到完整的HTML页面中，并添加CSS样式
            html_content = f"""
            <html>
            <head>
                <title>任务调度面板</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h2 {{ color: #333; }}
                    table.my-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 10px 0;
                    }}
                    table.my-table th, table.my-table td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                    table.my-table th {{
                        background-color: #f2f2f2;
                        font-weight: bold;
                    }}
                    table.my-table tr:nth-child(even) {{
                        background-color: #f9f9f9;
                    }}
                    table.my-table tr:hover {{
                        background-color: #f1f1f1;
                    }}
                </style>
            </head>
            <body>
                <h2>任务状态监控</h2>
                {basic_html_table}
            </body>
            </html>
            """
            return html_content

        return self

    def run(self, port=8080):
        """启动 FastAPI 服务"""
        uvicorn.run(self.app, host="0.0.0.0", port=port, log_level="warning")

    def run_background(self, *args, **kwargs):
        threading.Thread(target=lambda: self.run(*args, **kwargs), daemon=True).start()


def __进程管理():
    """ 这里的功能后续做通用后可以进入pyxllib """


def run_python_module(*args,
                      repeat_num=None,
                      wait_mode=0,
                      success_continue=False,
                      **kwargs):
    """
    用于重复启动 Python 模块的函数。

    :param repeat_num: 重复次数。如果为None，则无限重试。
    :param wait_mode: 每次重试之间的等待时间（秒）
        正值是程序运行结束(报错)后，等待的秒数
        负值则是另一种特殊的等待机制，是以程序启动时间作为相对计算的

    :param success_continue: 运行成功的情况下，是否也要重试，默认成功后就不重试了

    python -m xlproject.code4101 run_python_module
    """
    start_time = end_time = None
    round_num = itertools.count(1) if repeat_num is None else range(1, 1 + int(repeat_num))
    for round_id in round_num:
        # 0 上一轮次的等待
        XlTrigger.smart_wait(start_time, end_time, 0 if start_time is None else wait_mode, print_mode=1)

        # 1 标记当前轮次
        logger.info(f'进程运行轮次：{round_id}')
        start_time = datetime.datetime.now()

        # 2 配置参数
        cmds = [f'{sys.executable}', '-m']
        cmds.extend(map(str, args))  # 添加位置参数
        for k, v in kwargs.items():
            cmds.append(f'--{k}')  # 添加关键字参数的键
            cmds.append(str(v))  # 添加关键字参数的值

        # 3 custom 自定义配置
        # 第2个参数是特殊参数，一般是模块名、启动位置。支持一定的缩略写法
        # 但是这个不太好些，就暂时不写
        cmds[2] = cmds[2].replace('/', '.')  # 支持输入路径形式

        # 4 执行程序
        res = subprocess.run(cmds)
        end_time = datetime.datetime.now()

        # 5 执行完成
        if res.returncode == 0:
            logger.info('进程成功完成')
            if not success_continue:
                break
        else:
            logger.info(f'遇到错误，返回码：{res.returncode}。尝试重启进程')


def support_retry_process(repeat_num=None, wait_mode=0, success_continue=False):
    """ 对函数进行扩展，支持重启运行

    被装饰的函数，会扩展支持重启需要的几个参数：
    repeat_num, wait_mode, success_continue

    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal repeat_num, wait_mode, success_continue

            retry = int(kwargs.pop('retry', False))

            # 1 非重试，正常运行
            if not retry:
                return func(*args, **kwargs)

            # 2 需要重试
            wait_mode = int(kwargs.pop('wait_mode', wait_mode))
            repeat_num = kwargs.pop('repeat_num', repeat_num)
            success_continue = int(kwargs.pop('success_continue', success_continue))

            start_time = end_time = None
            round_num = itertools.count(1) if repeat_num is None else range(1, 1 + int(repeat_num))
            for round_id in round_num:
                # 0 上一轮次的等待
                XlTrigger.smart_wait(start_time, end_time, 0 if start_time is None else wait_mode, print_mode=1)

                # 1 标记当前轮次
                logger.info(f'进程运行轮次：{round_id}')
                start_time = datetime.datetime.now()

                # 2 配置参数
                cmds = [sys.executable, sys.argv[0], func.__name__]
                cmds.extend(map(str, args))  # 添加位置参数
                for k, v in kwargs.items():
                    cmds.append(f'--{k}')  # 添加关键字参数的键
                    cmds.append(str(v))  # 添加关键字参数的值

                # 3 custom 自定义配置
                # 第2个参数是特殊参数，一般是模块名、启动位置。支持一定的缩略写法
                # 但是这个不太好些，就暂时不写
                # cmds[2] = cmds[2].replace('/', '.')  # 支持输入路径形式

                # 4 执行程序
                print(cmds)
                res = subprocess.run(cmds)

                # todo 可以改成更高效的execv？估计有很多问题，那原本的实现如何修改避免无线嵌套执行子程序？
                # os.execv(sys.executable, cmds)

                end_time = datetime.datetime.now()

                # 5 执行完成
                if res.returncode == 0:
                    logger.info('进程成功完成')
                    if not success_continue:
                        break
                else:
                    logger.info(f'遇到错误，返回码：{res.returncode}。尝试重启进程')

        return wrapper

    return decorator


@support_retry_process(repeat_num=None, wait_mode=0)
def healthy():
    print('Hello')
    exit(1)


if __name__ == '__main__':
    pass
