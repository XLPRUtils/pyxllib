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

from pyxllib.prog.specialist import parse_datetime
from pyxllib.algo.stat import print_full_dataframe
from pyxllib.file.specialist import XlPath


def __1_定时工具():
    pass


class SchedulerUtils:
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


def __2_程序管理():
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


class ProgramWorker(SimpleNamespace):
    """
    代表一个单独的程序（进程），基于 SimpleNamespace 实现。
    """

    def __init__(self, name,
                 cmd,
                 shell=False,
                 run=True,
                 port=None,
                 locations=None,
                 raw_cmd=None,
                 **attrs):
        """
        :param name: 程序昵称
        :param program: 程序对象
        :param port: 是否有执行所在端口
        :param locations: 是否有url地址映射，一般用于nginx等配置
        """
        super().__init__(**attrs)

        # 执行程序需要使用的参数
        self.cmd = cmd
        self.shell = shell
        self.run = run

        # 关键参数
        self.name = name
        self.program = None
        self.raw_cmd = raw_cmd

        # 这两个是比较特别的属性，在我的工程框架中常用
        self.port = port
        self.locations = locations

        # 特殊调度模式，需要用到程序最近一次启动、结束时间
        self.last_start_time = None
        self.last_end_time = None

    def _set_pdeathsig(self, sig=None):
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

    def launch(self):
        """ 启动程序 """
        self.last_start_time = datetime.datetime.now()
        self.last_end_time = None

        if not self.run:
            # 如果不需要立即启动，则返回一个 Mock 对象
            proc = Mock()
            proc.pid = None
            proc.poll.return_value = 'tag'
            self.program = proc
            return self.program

        kwargs = {}
        for name in ['stdin', 'stdout', 'stderr']:
            if hasattr(self, name):
                kwargs[name] = getattr(self, name)

        if sys.platform == 'win32':
            self.program = subprocess.Popen(self.cmd, shell=self.shell, **kwargs)
        else:
            self.program = subprocess.Popen(self.cmd, shell=self.shell, **kwargs,
                                            preexec_fn=self._set_pdeathsig())

        return self.program

    def terminate(self):
        """
        比较优雅结束进程的方法
        """
        if self.program is not None:
            self.program.terminate()
        if self.last_end_time is None:
            self.last_end_time = datetime.datetime.now()

    def kill(self):
        """
        有时候需要强硬的kill方法来结束进程
        """
        if self.program is not None:
            self.program.kill()
        if self.last_end_time is None:
            self.last_end_time = datetime.datetime.now()

    def is_running(self):
        """ 检查程序是否在运行 """
        status = self.program is not None and self.program.poll() is None
        if not status and self.last_end_time is None:  # 检测到程序运行结束，则标记下
            self.last_end_time = datetime.datetime.now()
        return status


class MultiProgramLauncher:
    """
	管理多个程序的启动与终止
	"""

    def __init__(self):
        self.workers = []
        self.scheduler = None

    def __1_进场管理的核心底层函数(self):
        pass

    def init_scheduler(self):
        from apscheduler.schedulers.background import BackgroundScheduler
        # from apscheduler.schedulers.asyncio import AsyncIOScheduler

        if self.scheduler is None:
            self.scheduler = BackgroundScheduler()

        return self.scheduler

    def worker_add_schedule(self, worker, schedule=None, misfire_grace_time=None):
        """ 将程序添加为定时任务

        :param int|float|str|list schedule: 定时任务配置，如果提供则添加到 APScheduler 调度器
            int/float: 每隔多少秒执行一次，可能同时有多个实例存在
            tuple(value1, value2): 只允许单实例运行，这里记录的是上次实例结束后，等待多久再开启下次实例
                此时第1个数值，正数表示等待上一次运行结束后，下次开始运行前的等待时间
                    负数表示是在上次启动后，等待多久就可以运行下一次
                    但下一次一定会在上一次运行结束后再续上
                第2个参数，为了实现这套功能，其实需要一个监视器不断监控程序的运行状态
                    第2个参数是监控器检测的频率秒数
            str: cron 表达式，支持 5 或 6 个字段
            list[datetime]: 在指定时间节点列表运行
                值只要是可以解析为datetime类型的都可以，底层使用特殊的解析器
        """
        from datetime import datetime, timedelta

        def task():
            """ 启动任务并更新状态 """
            # todo schedule不一定都是单实例阻塞情景的需求，以后有需要可以扩展支持多实例同时存在的非阻塞模式
            if worker.is_running():
                logger.warning(f'由于程序"{name}"在上一次周期还没运行完，新周期不重复启动')
            else:
                worker.launch()

        self.init_scheduler()

        name = worker.name

        # 处理 schedule 类型
        if isinstance(schedule, (int, float)):
            # 如果是单一数值，按固定间隔执行，不考虑任务是否完成
            self.scheduler.add_job(task, 'interval', seconds=abs(schedule))
            logger.info(f"已添加定时任务：{name}，监测频率：{schedule} 秒")

        elif isinstance(schedule, tuple) and len(schedule) == 2:
            wait_time, monitor_frequency = schedule

            def interval_task():
                # 如果还没有启动过任务，直接启动
                if worker.last_start_time is None:
                    worker.launch()
                    return

                # 如果任务正在运行，则不启动新任务
                if worker.is_running():
                    return

                # 计算下次启动时间
                if wait_time < 0:  # 基于上次启动时间
                    next_run = worker.last_start_time + timedelta(seconds=abs(wait_time))
                else:  # 基于上次结束时间
                    next_run = worker.last_end_time + timedelta(seconds=wait_time)

                # 看现在是否需要重启
                if datetime.now() >= next_run:
                    worker.launch()

            # 以 monitor_frequency 作为定时检查的间隔
            self.scheduler.add_job(interval_task, 'interval', seconds=monitor_frequency)
            logger.info(f"已添加定时任务：{name}，等待时间: {wait_time} 秒，监测频率: {monitor_frequency} 秒")

        elif isinstance(schedule, str):
            from apscheduler.triggers.cron import CronTrigger
            cron_parts = schedule.split()
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

                self.scheduler.add_job(
                    task,
                    CronTrigger(
                        second=cron_parts[0],
                        minute=cron_parts[1],
                        hour=cron_parts[2],
                        day=cron_parts[3],
                        month=cron_parts[4],
                        day_of_week=x,
                    ),

                    # 检测的时候有概率错过了精确时间点。但一般不论延迟了多久，都要补运行上。
                    misfire_grace_time=misfire_grace_time,
                )
                logger.info(f"已添加定时任务：{name}，触发器: cron，表达式: {schedule}")
            else:
                logger.warning(f"无效的 cron 表达式：{schedule}")

        elif isinstance(schedule, list):
            # 支持 list[datetime] 格式
            for run_time in schedule:
                run_time = parse_datetime(run_time)
                self.scheduler.add_job(task, 'date', run_date=run_time)
            logger.info(f"已添加定时任务：{name}，触发器: dates，运行时间: {schedule}")

        else:
            logger.warning(f"无效的调度格式，跳过任务：{name}")

    def add_program_cmd(self,
                        cmd,
                        name=None,
                        shell=False,
                        run=True,
                        schedule=None,
                        **attrs):
        """
        启动一个程序，或仅存储任务，并添加进管理列表

        :param cmd: 启动程序的命令
        :param name: 程序名称，如果未提供则从cmd中自动获取
        :param shell:
            False，(优先推荐)直接跟系统交互，此时cmd应该输入数组格式
            True，启用shell进行交互操作，这种情况会更适合管道等模式的处理，此时cmd应该输入字符串格式
        :param int|float|str|list schedule: 定时任务配置，如果提供则添加到 APScheduler 调度器
        :param run: 如果为True，则立即启动程序；否则仅存储任务信息
        :param attrs: 其他需要传递给ProgramWorker的参数
        :return: 返回一个ProgramWorker实例，表示启动的程序或存储的任务
        """
        # 1 如果未显式传入name，自动从cmd中获取程序名称
        # logger.info(cmd)
        if name is None:
            _cmd = cmd.split() if isinstance(cmd, str) else cmd
            name = _cmd[0]

        # 2 创建 ProgramWorker 实例
        worker = ProgramWorker(name, cmd, shell, run, raw_cmd=cmd, **attrs)
        self.workers.append(worker)

        # 3 如果有 schedule 配置，添加调度任务；否则就是正常的启动任务
        if schedule:
            self.worker_add_schedule(worker, schedule=schedule, misfire_grace_time=attrs.get('misfire_grace_time'))
        else:
            worker.launch()

        return worker

    def add_program_cmd2(self, cmd, ports=1, name=None, **kwargs):
        """
        增强版 add_program_cmd，支持 ports 的处理

        :param ports:
            int 表示要开启的进程数，端口号随机生成
            list 表示指定的端口号
            None 不做特殊配置
        """
        # 1 处理 ports 参数，找到空闲端口或使用指定端口
        if isinstance(ports, int):
            ports = find_free_ports(ports)

        # 2 处理 locations 参数，自动设置默认 URL 映射
        if name is None:
            _cmd = cmd.split() if isinstance(cmd, str) else cmd
            name = _cmd[0]

        # 3 遍历端口，依次启动进程
        workers = []
        if ports:
            for port in ports:
                cmd_with_port = cmd + [f'--port', str(port)]
                kwargs['port'] = port
                worker = self.add_program_cmd(cmd_with_port, name=f'{name}:{port}', **kwargs)
                workers.append(worker)
        else:
            workers = [self.add_program_cmd(cmd, name=name, **kwargs)]

        return workers

    def add_program_cmd3(self, cmd, ports=None, locations=None, *, devices=None, **kwargs):
        """
        增强版 add_program_cmd2，支持 devices 等其他更多特殊的扩展参数的处理

        :param locations: URL 映射规则
        :param int|str|None devices: 使用的设备编号（显卡编号或 CPU）
            未设置的时候，使用cpu运行
        :return: 返回一个包含所有启动程序的 worker 列表。
        """
        # 1 处理locations
        if locations:
            if isinstance(locations, str):
                locations = [locations]
            for i, x in enumerate(locations):
                if not isinstance(x, dict):
                    locations[i] = {x: x}
            kwargs['locations'] = locations

        # 2 处理 devices 参数，设置显卡编号或使用 CPU
        if devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(devices)
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']  # 如果没设置 devices，就清除环境变量，使用 CPU

        # 3 处理 ports 参数
        workers = self.add_program_cmd2(cmd, ports=ports, **kwargs)

        return workers

    def __2_各种添加进程的机制(self):
        pass

    def add_program_python(self, py_file, args='',
                           ports=None, locations=None,
                           name=None, shell=False, executer=None,
                           **kwargs):
        """ 添加并启动一个Python文件作为后台程序

        :param str|list args:
        """
        if executer is None:
            executer = sys.executable
        cmd = [str(executer), str(py_file)]
        if isinstance(args, str):
            cmd.append(args)
        else:
            cmd += list(args)
        return self.add_program_cmd3(cmd, name, ports=ports, locations=locations, shell=shell, **kwargs)

    def add_program_python_module(self, module, args='',
                                  ports=None, locations=None,
                                  name=None, shell=False, executer=None,
                                  **kwargs):
        """
        添加并启动一个Python模块作为后台程序

		:param module: 要执行的Python模块名（python -m 后面的部分）
        :param str|list args: 模块的参数
        :param name: 进程的名称，默认为模块名
		"""
        if executer is None:
            executer = sys.executable
        cmd = [f'{executer}', '-m', f'{module}']
        if isinstance(args, str):
            cmd.append(args)
        else:
            cmd += list(args)
        if name is None:
            name = module
        return self.add_program_cmd3(cmd, ports=ports, name=name, locations=locations, shell=shell, **kwargs)

    def add_prog(self, prog, extcmds='',
                 ports=None, locations=None, *,
                 name=None, devices=None,
                 executer=None,
                 run=True,
                 schedule=None,
                 ):
        """
        :param int|list ports:
        :param str|list extcmds:
        """
        if locations is None:
            locations = f'/api/{prog.split(".")[-1]}'
        self.add_program_python_module(prog,
                                       extcmds,
                                       ports=ports, locations=locations,
                                       name=name, devices=devices,
                                       executer=executer,
                                       run=run,
                                       schedule=schedule,
                                       )

    def add_server(self, prog, ports=None, locations=None, *args, **kwargs):
        """ 我自己部署的服务，基本都有特定的start_server启动函数 """
        self.add_prog(prog, extcmds='start_server', ports=ports, locations=locations, *args, **kwargs)

    def add_os_command_task(self,
                            script_content,
                            extension=None,
                            shell=False,
                            run=True,
                            name=None,
                            schedule=None,
                            **kwargs):
        """
        添加一个操作系统命令或脚本（如 .bat、.sh、.ps1 等）并启动。

        :param script_content: 脚本的内容（字符串）
        :param extension: 脚本文件的扩展名（如 .bat, .sh, .ps1）；如果为 None 则根据操作系统自动选择
        :param shell: 是否使用 shell 执行
        :param run: 是否立即启动
        :param name: 程序名称
        :param schedule: 定时任务配置
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

        # 5 使用 add_program_cmd3 启动脚本
        return self.add_program_cmd3(
            cmd=cmd,
            name=name or f"command_{script_file.stem}",
            shell=shell,
            run=run,
            schedule=schedule,
            **kwargs
        )

    def __3_nginx相关(self):
        pass

    def get_all_locations(self):
        locations = defaultdict(list)
        for worker in self.workers:
            if not worker.locations:
                continue
            for x in worker.locations:
                for dst, src in x.items():
                    if worker.port:  # 250126周日21:02，有端口才添加
                        locations[dst].append(f'localhost:{worker.port}{src}')
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

        content = '\n'.join(upstreams) + '\nserver {\n' + textwrap.indent('\n'.join(servers), '\t') + '}'
        return content

    def __4_多程序管理(self):
        pass

    def count_running(self):
        return sum(1 for worker in self.workers if worker.is_running())

    def list_workers(self):
        """返回所有任务的状态 DataFrame"""
        ls = []
        for worker in self.workers:
            ls.append({
                'name': worker.name,
                'pid': worker.program.pid if worker.program else None,
                'poll': worker.program.poll() if worker.program else None,
                'args': worker.raw_cmd,
                'port': worker.port,
                'locations': worker.locations,
            })
        return pd.DataFrame(ls)

    def cleanup_finished(self, exit_code=None):
        """
        清除已运行完的程序

        :param exit_code:
            None - 清除所有已结束的程序（默认行为）。
            0 - 只清除正常结束的程序。
            非0 - 只清除异常结束的程序。
        """
        new_workers = []
        for worker in self.workers:
            code = worker.program.poll()
            if code is None:  # 进程仍在运行
                new_workers.append(worker)
            elif exit_code is None or code == exit_code:
                print(f'清理已结束的程序: {worker.name} (pid: {worker.program.pid}, exit code: {code})')

        self.workers = new_workers

    def stop_all(self):
        """ 停止所有后台程序 """
        # 关闭所有调度器
        if self.scheduler:
            self.scheduler.shutdown()

        # 停止所有单启动任务
        for worker in self.workers:
            worker.terminate()
        self.workers = []

    def proc_cmd(self, cmd):
        if cmd == 'kill':
            self.stop_all()
            return False
        elif cmd == 'count':
            print(f'有{self.count_running()}个程序正在运行')
        elif cmd.startswith('cleanup'):
            args = cmd.split()
            exit_code = int(args[1]) if len(args) > 1 else None
            self.cleanup_finished(exit_code)
            print("清理完成")
        elif cmd == 'list':  # 列出所有程序（转df查看）
            df = self.list_workers()
            print_full_dataframe(df)
        return True

    def run_endless(self, cmd=True, wait_seconds=1, *, debug_port=None):
        """
        一直运行，直到用户输入 kill 命令或 Ctrl+C

        :param bool cmd: 是否支持命令行input输入指令监控状态的模式
            默认支持，但在有scheduler调度的情况不建议开启，有input阻塞其他子程的风险
        :param int|float wait_seconds: 每次循环之间停顿秒数，用来给其他子程等运行时间，避免阻塞
        :param int debug_port: 是否要开一个后端服务，支持查询程序运行状态

    	poll：
            如果进程仍在运行，poll()方法返回None。
            如果进程已经结束，poll()方法返回进程的退出码（exit code）。
                如果进程正常结束，退出码通常为0。
                如果进程异常结束，退出码通常是一个非零值，表示异常的类型或错误码。
    	"""
        if self.scheduler:  # 如果有定时任务，启动调度器
            self.scheduler.start()

        port = debug_port
        if port:
            dashboard = LauncherDashboard(self)
            threading.Thread(target=lambda: dashboard.run(port), daemon=True).start()

        try:
            while True:
                if cmd:
                    _cmd = input(">>> ")
                    if not self.proc_cmd(_cmd):
                        break
                time.sleep(wait_seconds)
        except KeyboardInterrupt:
            print("\n检测到 Ctrl+C，正在终止所有程序...")
            self.stop_all()
            print("所有程序已终止，退出。")


class LauncherDashboard:
    def __init__(self, launcher):
        self.launcher = launcher
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def home():
            """主页，展示所有任务状态"""
            df = self.launcher.list_workers()
            # return self.render_text(df)
            return self.render_html(df)

        @self.app.get("/count", response_class=PlainTextResponse)
        async def get_count():
            """返回正在运行的任务数量"""
            count = self.launcher.count_running()
            return f"正在运行的任务数量: {count}\n"

        @self.app.post("/stop", response_class=PlainTextResponse)
        async def stop_all():
            """停止所有任务"""
            self.launcher.stop_all()
            return "所有任务已停止。\n"

        @self.app.post("/cleanup", response_class=PlainTextResponse)
        async def cleanup(exit_code: int = None):
            """清理已完成的任务"""
            self.launcher.cleanup_finished(exit_code)
            return "清理完成。\n"

    def render_text(self, df):
        """生成纯文本格式的任务状态"""
        if df.empty:
            return "没有任务正在运行。\n"

        # 调整表头以显示新的属性：port 和 locations
        output = ["当前任务状态:\n"]
        output.append(f"{'编号':<5} {'名称':<15} {'PID':<10} {'状态':<10} {'端口':<10} {'位置':<20} {'命令'}")
        output.append("-" * 120)

        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            name = row['name']

            # 更鲁棒地处理 pid，考虑 NaN 情况
            pid = "N/A" if pd.isna(row['pid']) else int(row['pid'])

            # 获取任务状态
            status = "运行中" if row['poll'] is None else "已结束"

            # 处理 args，确保路径展示更清晰
            args = [a for a in row['args'] if a] if isinstance(row['args'], list) else []
            args_str = "[" + ", ".join(
                repr(arg).replace("\\\\", "\\") if '\\' in arg else repr(arg) for arg in args
            ) + "]"

            # 处理 port 和 locations
            port = row['port'] if not pd.isna(row['port']) else "N/A"
            locations = str(row['locations'])

            # 格式化输出
            output.append(f"{idx:<5} {name:<15} {pid:<10} {status:<10} {port:<10} {locations:<20} {args_str}")

        output.append("\n")
        return "\n".join(output)

    def render_html(self, df):
        """生成HTML格式的任务状态报表"""
        if df.empty:
            return """
            <div style="padding: 20px; text-align: center; color: #666;">
                <h3>没有任务正在运行</h3>
            </div>
            """

        # 准备显示数据
        display_df = pd.DataFrame()

        # 添加编号列
        display_df['编号'] = range(1, len(df) + 1)

        # 添加基本信息列
        display_df['任务名称'] = df['name']
        display_df['进程ID'] = df['pid'].apply(
            lambda x: f'<span style="font-family: monospace;">{int(x)}</span>' if pd.notna(
                x) else '<span style="color: #999;">N/A</span>')

        # 状态列，使用颜色标识
        display_df['状态'] = df['poll'].apply(
            lambda x: '<span style="color: #28a745; font-weight: bold;">● 运行中</span>' if x is None
            else '<span style="color: #dc3545; font-weight: bold;">● 已结束</span>'
        )

        # 端口列
        display_df['端口'] = df['port'].apply(
            lambda x: f'<span style="font-family: monospace; color: #007bff;">{int(x)}</span>' if pd.notna(x) and x != 0
            else '<span style="color: #999;">-</span>'
        )

        # 位置列，处理长路径
        display_df['位置'] = df['locations'].apply(
            lambda
                x: f'<span title="{x}" style="max-width: 300px; display: inline-block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{x}</span>'
        )

        # 命令列，格式化显示
        def format_command(args):
            if not isinstance(args, list):
                return '<span style="color: #999;">N/A</span>'

            args = [str(a) for a in args if a]
            if not args:
                return '<span style="color: #999;">N/A</span>'

            # 第一个参数（通常是命令）用粗体
            if len(args) > 0:
                cmd = f'<strong>{args[0]}</strong>'
                params = ' '.join(args[1:]) if len(args) > 1 else ''
                full_cmd = f'{cmd} {params}'.strip()

                # 如果命令太长，添加悬停提示
                if len(' '.join(args)) > 50:
                    return f'<span title="{" ".join(args)}" style="cursor: help;">{full_cmd[:50]}...</span>'
                return full_cmd
            return '<span style="color: #999;">N/A</span>'

        display_df['命令'] = df['args'].apply(format_command)

        # 生成HTML表格
        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 20px;">
            <h2 style="color: #333; margin-bottom: 20px;">任务状态报表</h2>
            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <span style="color: #666;">生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                <span style="float: right; color: #666;">总任务数: {len(df)}</span>
            </div>
            <table style="width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <thead>
                    <tr style="background: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                        <th style="padding: 12px; text-align: left; font-weight: 600; color: #495057;">编号</th>
                        <th style="padding: 12px; text-align: left; font-weight: 600; color: #495057;">任务名称</th>
                        <th style="padding: 12px; text-align: left; font-weight: 600; color: #495057;">进程ID</th>
                        <th style="padding: 12px; text-align: left; font-weight: 600; color: #495057;">状态</th>
                        <th style="padding: 12px; text-align: left; font-weight: 600; color: #495057;">端口</th>
                        <th style="padding: 12px; text-align: left; font-weight: 600; color: #495057;">位置</th>
                        <th style="padding: 12px; text-align: left; font-weight: 600; color: #495057;">命令</th>
                    </tr>
                </thead>
                <tbody>
        """

        # 添加表格行
        for idx, row in display_df.iterrows():
            row_style = "background: #fff;" if idx % 2 == 0 else "background: #f8f9fa;"
            html += f"""
                    <tr style="{row_style} border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 12px; color: #666;">{row['编号']}</td>
                        <td style="padding: 12px; font-weight: 500;">{row['任务名称']}</td>
                        <td style="padding: 12px;">{row['进程ID']}</td>
                        <td style="padding: 12px;">{row['状态']}</td>
                        <td style="padding: 12px;">{row['端口']}</td>
                        <td style="padding: 12px; font-size: 0.9em; color: #666;">{row['位置']}</td>
                        <td style="padding: 12px; font-family: 'Consolas', 'Monaco', monospace; font-size: 0.9em;">{row['命令']}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
            <div style="margin-top: 15px; padding: 10px; background: #e9ecef; border-radius: 5px; font-size: 0.9em; color: #666;">
                <strong>说明:</strong> 
                <span style="color: #28a745;">●</span> 运行中 
                <span style="margin-left: 20px;"><span style="color: #dc3545;">●</span> 已结束</span>
            </div>
        </div>
        """

        return html

    def run(self, port=8080):
        """启动 FastAPI 服务"""
        import uvicorn
        uvicorn.run(self.app, host="0.0.0.0", port=port, log_level="warning")


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
    from pyxllib.prog.multiprogs import SchedulerUtils

    start_time = end_time = None
    round_num = itertools.count(1) if repeat_num is None else range(1, 1 + int(repeat_num))
    for round_id in round_num:
        # 0 上一轮次的等待
        SchedulerUtils.smart_wait(start_time, end_time, 0 if start_time is None else wait_mode, print_mode=1)

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
    from pyxllib.prog.multiprogs import SchedulerUtils

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
                SchedulerUtils.smart_wait(start_time, end_time, 0 if start_time is None else wait_mode, print_mode=1)

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


def support_multi_processes_hyx(default_processes=1):
    """ 对函数进行扩展，支持并发多进程运行
    增加重跑
    注意被装饰的函数，需要支持 process_count、process_id 两个参数，来获得总进程数，当前进程id的信息
    """

    def decorator(func):

        def wrapper(*args, **kwargs):
            process_count = int(kwargs.pop('process_count', default_processes))
            process_id = kwargs.pop('process_id', None)
            shell = kwargs.pop('shell', False)

            if process_count == 1 or process_id is not None:
                if process_id is None:
                    return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs, process_count=process_count, process_id=int(process_id))
            else:
                mpl = MultiProgramLauncher()
                for i in range(int(process_count)):
                    if isinstance(process_id, int) and i != process_id:
                        continue

                    '''
                    sys.argv[0] 为 /Users/youx/NutstoreCloudBridge/slns/xlproject/xlproject/m2404ragdata/b清洗/hyx240806统计图表数.py
                    将其转化为 xlproject.m2404ragdata.b清洗.hyx240806统计图表数 
                    '''
                    header = 'xlproject.code4101'

                    root_directory_name = 'xlproject'
                    occurrence = 2
                    path = sys.argv[0]

                    # 查找第 occurrence 次出现的 root_directory_name 位置
                    positions = [i for i in range(len(path)) if path.startswith(root_directory_name, i)]
                    if len(positions) < occurrence:
                        raise ValueError(
                            f"Path does not contain {occurrence} occurrences of the root directory name: {root_directory_name}")

                    # 提取并转换为模块名称
                    index = positions[occurrence - 1]
                    module_name = os.path.splitext(path[index:])[0].replace(os.path.sep, '.')

                    # todo 这样使用有个坑，process_count、process_id都是以str类型传入的，开发者下游使用容易出问题
                    cmds = [
                        'run_python_module',
                        '--wait_mode',
                        '60',
                        module_name,
                        func.__name__,
                        '--process_count', str(process_count),
                        '--process_id', str(i)
                    ]
                    cmds.extend(map(str, args))  # 添加位置参数
                    for k, v in kwargs.items():
                        cmds.append(f'--{k}')  # 添加关键字参数的键
                        cmds.append(str(v))  # 添加关键字参数的值

                    mpl.add_program_python_module(header, cmds, shell=shell)
                mpl.run_endless()

        return wrapper

    return decorator


def support_multi_processes(default_processes=1):
    """ 对函数进行扩展，支持并发多进程运行

    注意被装饰的函数，需要支持 process_count、process_id 两个参数，来获得总进程数，当前进程id的信息
    """

    def decorator(func):

        def wrapper(*args, **kwargs):
            process_count = int(kwargs.pop('process_count', default_processes))
            process_id = kwargs.pop('process_id', None)
            shell = kwargs.pop('shell', False)

            if process_count == 1 or process_id is not None:
                if process_id is None:
                    return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs, process_count=process_count, process_id=int(process_id))
            else:
                mpl = MultiProgramLauncher()
                for i in range(int(process_count)):
                    if isinstance(process_id, int) and i != process_id:
                        continue

                    # todo 这样使用有个坑，process_count、process_id都是以str类型传入的，开发者下游使用容易出问题
                    cmds = [func.__name__,
                            '--process_count', str(process_count),
                            '--process_id', str(i)
                            ]
                    cmds.extend(map(str, args))  # 添加位置参数
                    for k, v in kwargs.items():
                        cmds.append(f'--{k}')  # 添加关键字参数的键
                        cmds.append(str(v))  # 添加关键字参数的值

                    mpl.add_program_python(sys.argv[1], cmds, shell=shell)
                mpl.run_endless()

        return wrapper

    return decorator


if __name__ == '__main__':
    pass
