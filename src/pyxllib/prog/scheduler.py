#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/12

from collections import defaultdict
import ctypes
import datetime
from functools import partial
import itertools
import os
import re
import shlex
import socketserver
import subprocess
import sys
import textwrap
import threading
import time
import typing
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Type, Literal, Sequence

# from apscheduler import triggers

from pyxllib.prog.lazyimport import lazy_import

try:
    from deprecated import deprecated
except ModuleNotFoundError:
    deprecated = lazy_import('deprecated', 'Deprecated')

try:
    from pydantic import BaseModel, ConfigDict
except ModuleNotFoundError:
    BaseModel, ConfigDict = lazy_import('from pydantic import BaseModel, ConfigDict')

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

from pyxllib.prog.xltime import XlTime
from pyxllib.prog.specialist import XlBaseModel, resolve_params
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

    def __2_扩展触发器工厂函数(self):
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

    @staticmethod
    def parse_time_interval(interval_str: str) -> float:
        """ 解析 10s, 5m, 1h 格式的时间为秒数 """
        unit_map = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
        interval_str = interval_str.lower().strip()

        # 匹配数字+单位 (例如 1.5h)
        m = re.match(r'^(\d+(?:\.\d+)?)\s*([smhd])$', interval_str)
        if m:
            value = float(m.group(1))
            unit = m.group(2)
            return value * unit_map[unit]

        # 纯数字默认为秒
        try:
            return float(interval_str)
        except ValueError:
            raise ValueError(f"无法解析时间间隔: {interval_str}")

    @classmethod
    def _parse_smart_datetime(cls, time_str: str) -> datetime.datetime:
        """ 智能解析时间
        1. 如果是纯时间格式 (HH:MM 或 HH:MM:SS)，自动定位到最近的未来时间点（今天或明天）。
        2. 否则使用 XlTime 通用解析（包含日期的完整格式）。
        """
        # 正则匹配 HH:MM 或 HH:MM:SS，强制要求开头结尾匹配，避免匹配到 "2024-11-12 12:00"
        # ^(\d{1,2})  : 1-2位数字的小时
        # :(\d{1,2})  : 冒号+1-2位数字的分钟
        # (?::(\d{1,2}))?$ : 可选的冒号+1-2位数字的秒
        m = re.match(r'^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$', time_str)

        if m:
            hour, minute = int(m.group(1)), int(m.group(2))
            second = int(m.group(3)) if m.group(3) else 0

            now = datetime.datetime.now()
            # 替换当前时间的时分秒
            target_time = now.replace(hour=hour, minute=minute, second=second, microsecond=0)

            # 核心逻辑：如果时间已过，则加一天
            if target_time <= now:
                target_time += datetime.timedelta(days=1)

            return target_time

        # 否则交给通用库处理 (例如 '2024-11-12 12:00' 或 'tomorrow')
        return XlTime(time_str).datetime

    @classmethod
    def parse_str(cls, trigger_str: str):
        """ 智能解析字符串类型的触发器
        对字符串的解析扩展，一方面是方便配置trigger，另一方面也是方便funboost发布任务，因为funboost要求参数可json化

        支持格式：
        1. Cron模式 (自动识别或显式前缀):
           - "*/5 * * * *"
           - "0 12 * * * *"
           - "cron: 0 12 * * *"

        2. 间隔模式 (Interval):
           - "30s" / "10m" / "1h" (默认为间隔循环，等同于 int 参数)
           - "interval: 30s"
           - "every: 1h"

        3. 延时/日期模式 (Date):
           - "2024-11-12 12:00:00" (具体时间)
           - "date: 2024-11-12"
           - "in: 30s" (表示从现在起延时30秒执行一次，区别于 interval)
        """
        trigger_str = trigger_str.strip()
        lower_str = trigger_str.lower()

        # --- A. 显式前缀模式 (明确意图) ---

        # A1. Cron 前缀
        if lower_str.startswith(('cron:', 'cron ')):
            return cls.create_cron_trigger(trigger_str.split(':', 1)[1].strip())

        # A2. Date/At 前缀 (绝对时间)
        if lower_str.startswith(('date:', 'at:')):
            val = trigger_str.split(':', 1)[1].strip()
            return DateTrigger(cls._parse_smart_datetime(val))

        # A3. Interval/Every 前缀 (循环间隔)
        if lower_str.startswith(('interval:', 'every:')):
            val = trigger_str.split(':', 1)[1].strip()
            return IntervalTrigger(seconds=cls.parse_time_interval(val))

        # A4. In 前缀 (相对延时，执行一次)
        # 例如 "in: 30s" -> 30秒后执行一次
        if lower_str.startswith(('in:', 'after:')):
            val = trigger_str.split(':', 1)[1].strip()
            seconds = cls.parse_time_interval(val)
            run_time = datetime.datetime.now() + datetime.timedelta(seconds=seconds)
            return DateTrigger(run_time)

        # --- B. 特征推断模式 (智能猜测) ---

        # B1. Cron 特征检测
        # 包含 '*' 或 '/' 或者是 5-6 段空格分隔的字符串
        if '*' in trigger_str or '/' in trigger_str:
            return cls.create_cron_trigger(trigger_str)

        parts = trigger_str.split()
        if len(parts) >= 5:
            # 简单的启发式：如果空格分隔段数>=5，且不像时间日期格式，大概率是 cron
            # (XlTime通常能处理带空格的日期，这里如果 create_cron_trigger 失败，
            # 其实可以捕获异常再 fallback 到 Date，但为了效率，这里优先判定 Cron)
            try:
                return cls.create_cron_trigger(trigger_str)
            except ValueError:
                pass  # 解析 Cron 失败，可能是长格式的日期文本，继续往下走

        # B2. 简写时间间隔检测 (例如 "30s", "1.5h")
        # 这种格式默认视为 IntervalTrigger (为了跟 add_schedule(..., 30) 保持一致)
        if re.match(r'^\d+(\.\d+)?\s*[smhd]$', lower_str):
            return IntervalTrigger(seconds=cls.parse_time_interval(trigger_str))

        # B3. 默认兜底：尝试解析为具体日期时间
        try:
            return DateTrigger(XlTime(trigger_str).datetime)
        except Exception as e:
            raise ValueError(f"无法识别的触发器格式: '{trigger_str}'") from e


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


class PopenParams(XlBaseModel):
    """
    subprocess.Popen 常用到的参数的结构化封装。

    这个 dataclass 旨在提供一个类型安全、有智能提示的配置对象，
    以取代不透明的 **kwargs，让代码更清晰、更易于维护。
    """
    # args是必传参数，可以用位置方式的模式传递
    args: Union[str, List[str], None] = ''

    # 是否用shell（如bash/cmd）来执行命令。默认为False，直接执行程序。
    # 警告：当命令来自外部输入时，使用 shell=True 可能会带来安全风险（命令注入）。
    shell: bool = False

    # 输入，输出，错误流重定向。可以是 subprocess.PIPE, DEVNULL, a file descriptor, or a file object.
    stdin: Optional[int] = None
    stdout: Optional[int] = None
    stderr: Optional[int] = None

    # 指定要执行的程序。默认情况下，执行的程序是 args 列表的第一个元素。
    # 如果设置了 executable，它会替换 args[0] 作为要执行的程序。
    executable: Optional[str] = None

    # 设置子进程的当前工作目录。
    # 如果为 None，则子进程的工作目录将是父进程的当前工作目录。
    cwd: Optional[str] = None

    # 定义子进程的环境变量。如果为 None，子进程将继承父进程的环境变量。
    # 如果提供一个字典，它将作为子进程的完整环境变量。
    env: Optional[Dict[str, str]] = None

    # --- 文本模式与编码 ---

    # 如果为 True，stdin、stdout 和 stderr 将作为文本流处理，并使用 `encoding` 指定的编码。
    # 这个参数在 Python 3.7+ 中是 `text` 的别名。
    # 默认为 False，流将作为二进制字节流处理。
    text: bool = False
    universal_newlines: bool = False  # text 的旧别名

    # 当 text=True 时，用于解码/编码 stdin, stdout, stderr 的编码格式。
    encoding: Optional[str] = None

    # 当 text=True 时，指定如何处理编码和解码错误。
    errors: Optional[str] = None

    # --- 进程创建与管理 (POSIX) ---

    # （仅限 POSIX）在子进程执行前调用的可调用对象（函数）。
    # 警告：如果在多线程程序中使用，可能会导致死锁。
    # preexec_fn: Optional[Callable[[], Any]] = None  # 这个参数我会额外配置，所以不支持用户输入

    # （仅限 POSIX）如果为 True，则在子进程中创建一个新的进程会话。
    start_new_session: bool = False

    # --- 进程创建与管理 (Windows) ---

    # （仅限 Windows）用于设置子进程的 STARTUPINFO 结构。
    startupinfo: Any = None  # 类型依赖于 `subprocess` 内部的 `STARTUPINFO` 类

    # （仅限 Windows）一个整数，指定传递给 Windows CreateProcess 函数的标志。
    # 例如 subprocess.CREATE_NEW_CONSOLE。
    creationflags: int = 0

    # --- 其他高级参数 ---

    # 如果为 True，除了 0, 1, 2 之外的所有文件描述符将在子进程执行前被关闭。
    # 在 POSIX 上默认为 True，在 Windows 上默认为 False。
    close_fds: bool = True

    # 设置I/O管道的缓冲区大小。
    # 0 表示不缓冲（读写是单个系统调用，可能返回不完整的数据）。
    # 1 表示行缓冲（仅在 text=True 时可用）。
    # 任何其他正值表示使用大约该大小的缓冲区。
    # 负值表示使用系统默认值。
    bufsize: int = -1

    # （仅限 POSIX）设置子进程的用户 ID。
    user: Optional[Union[int, str]] = None

    # （仅限 POSIX）设置子进程的组 ID。
    group: Optional[Union[int, str]] = None

    # （仅限 POSIX）设置子进程的附加组 ID。
    extra_groups: Optional[Sequence[Union[int, str]]] = None

    # （仅限 POSIX）设置子进程的 umask。
    umask: int = -1

    # （仅限 POSIX）一个文件描述符序列，在子进程中保持打开状态。
    pass_fds: Sequence[int] = ()

    # 如果为 False，则不恢复 SIGPIPE 信号的处理。默认为 True。
    restore_signals: bool = True

    # 在 Python 3.10+ 中可用，用于设置管道的大小（字节数）。仅在部分平台（如 Linux）上生效。
    pipesize: int = -1


class SubprocessTaskParams(PopenParams):
    # 设置一个名称
    name: Union[str, None] = None

    # 设置超时秒数
    timeout: Optional[int | float] = None

    @property
    def popen_params(self) -> PopenParams:
        all_data = self.model_dump(exclude_unset=True)
        child_only_fields = set(self.__class__.model_fields) - set(PopenParams.model_fields)
        for field in child_only_fields: all_data.pop(field, None)
        return PopenParams(**all_data)


class SubprocessTask:
    """ subprocess类型的task任务，后续可能跟apscheduler结合使用 """

    @resolve_params(SubprocessTaskParams, mode='pass')
    def __init__(self, args='', /, **resolved_params):
        # aps4.0.0a6使用仿函数机制的话还有些瑕疵不兼容，必须要配置这个值，不然会运行不了
        # 而且这个也算是唯一标识，必须不同实例值不同，才能确保aps4不会判别为同一task
        self.__qualname__ = str(id(self))

        params: SubprocessTaskParams = resolved_params['SubprocessTaskParams']

        # 执行程序需要使用的参数
        self.args = args or params.args
        self.name = params.name
        self.timeout = params.timeout
        self.popen_params = params.popen_params

        self.proc = None

    def __1_args相关处理(self):
        pass

    def _normalize_args_to_list(self, args_input: Union[str, List[str], None]) -> List[str]:
        """ 将任何格式的参数输入规范化为列表，并兼容 Windows 和 POSIX """
        if args_input is None:
            return []
        if isinstance(args_input, str):
            # 关键修正：
            # os.name == 'nt' 用于判断是否为 Windows 系统。
            # 在 Windows 上，我们使用 posix=False 模式。
            # 在其他系统（Linux, macOS等）上，使用 posix=True 模式。
            is_windows = (os.name == 'nt')
            return shlex.split(args_input, posix=not is_windows)

        return list(args_input)

    def _add_args(self, new_args: Union[str, List[str]], *, position: Literal['append', 'prepend']):
        """【核心重构】私有辅助方法，处理所有参数添加逻辑。"""
        original_type = type(self.args)

        # 步骤1: 规范化
        current_args_list = self._normalize_args_to_list(self.args)
        new_args_list = self._normalize_args_to_list(new_args)

        # 步骤2: 操作 (根据 position 参数决定合并顺序)
        if position == 'prepend':
            result_list = new_args_list + current_args_list
        else:  # 默认为 append
            result_list = current_args_list + new_args_list

        # 步骤3: 还原
        if original_type is str:
            self.args = shlex.join(result_list)
        else:
            self.args = result_list

    def append_args(self, new_args: Union[str, List[str]]):
        """
        在原有args上，在末尾新增args参数。
        """
        self._add_args(new_args, position='append')

    def prepend_args(self, new_args: Union[str, List[str]]):
        """
        在原有args上，在开头插入args参数。
        """
        self._add_args(new_args, position='prepend')

    def set_arg(self, arg_name: str, arg_value):
        """ 修订或添加一个参数的值 """
        original_type = type(self.args)
        arg_value_str = str(arg_value)

        args_list = self._normalize_args_to_list(self.args)

        updated = False
        for i, item in enumerate(args_list):
            if item == arg_name:
                if i + 1 < len(args_list):
                    args_list[i + 1] = arg_value_str
                    updated = True
                    break
            elif item.startswith(f'{arg_name}='):
                args_list[i] = f'{arg_name}={arg_value_str}'
                updated = True
                break

        if not updated:
            args_list.extend([arg_name, arg_value_str])

        if original_type is str:
            self.args = shlex.join(args_list)
        else:
            self.args = args_list

    def __2_进程处理(self):
        pass

    def is_running(self):
        """ 检查程序是否在运行 """
        status = self.proc is not None and self.proc.poll() is None
        return status

    def run(self, check=False):
        """ 启动程序，这个在下游子类中可以重载重新定制 """
        # 目前不支持重复执行，只允许有一个实例，但要扩展修改这个功能是相对容易的
        if self.is_running():
            return self.proc

        try:
            # subprocess.run 会等待命令完成，check=True 会在返回码非0时自动引发 CalledProcessError
            subprocess.run(
                self.args,
                preexec_fn=set_pdeathsig(),
                timeout=self.timeout,
                check=check,  # check=True，把subprocess的报错抛出给主进程
                **self.popen_params.model_dump(exclude_unset=True, exclude={'args'}),
            )
            # 如果代码能执行到这里，说明子进程成功了（退出码为0）
            return self.proc
        except subprocess.CalledProcessError as e:
            # 如果子进程返回非0退出码，check=True 就会引发这个异常
            # 将子进程的异常重新在父进程中抛出，以便 funboost 捕获
            raise RuntimeError(f"子进程执行失败: {e.stderr}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("子进程执行超时") from e
        finally:
            self.kill()

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

    @resolve_params(SubprocessTaskParams, mode='pass')
    def __init__(self, args='', /, port=None, locations=None, **resolve_params):
        super().__init__(args, resolve_params['SubprocessTaskParams'])

        # 我用nginx需要额外补充的两个属性值
        self.port = None
        self.set_port(port)

        # list[dict]结构，dict长度只有1，类似 [{'/原url路径/a', '/目标url路径/b'}, ...]
        # 如果k,v相同，可以简写为一个值 ['/a', '/b', ...]
        self.locations = []
        if locations:
            # 1. 如果传入的是单字符串（如 locations='/api'），转为列表
            if isinstance(locations, str):
                locations = [locations]

            # 2. 遍历列表，将字符串简写转换为字典
            for loc in locations:
                if isinstance(loc, str):
                    self.locations.append({loc: loc})
                elif isinstance(loc, dict):
                    self.locations.append(loc)
                else:
                    # 还可以加个报错或者日志提醒
                    pass

    def set_port(self, port):
        if port:
            self.port = port
            self.set_arg('--port', port)

    @classmethod
    @resolve_params(SubprocessTaskParams, mode='pass')
    def create_from_ports(cls, args='', /, ports=None, locations=None, **resolve_params):
        """
        :param ports:
            list[int]，指定配置若干端口的程序
            int, 配置几个端口
        """
        # 1 处理 ports 参数，找到空闲端口或使用指定端口
        if isinstance(ports, int):
            ports = find_free_ports(ports)
        if ports is None:
            ports = [None]

        # 2 遍历端口，依次启动进程
        tasks = []
        for port in ports:
            task = cls(args, resolve_params['SubprocessTaskParams'], port=port, locations=locations)
            if port:
                task.name = task.name or ''
                task.name = f'{task.name}:{port}'
            tasks.append(task)

        return tasks


def __3_调度器():
    pass


class XlScheduler(GetAttr):
    """ 对apscheduler的Scheduler扩展版
	"""
    _default = 'scheduler'

    def __init__(self):
        self._scheduler = None
        # aps也可以用get_tasks获得全部tasks，但那个task更底层，不太是我需要的更高层业务的tasks管理单元，所以这里我自己扩展一个成员
        self.tasks = []
        self.task_defaults = {}

    def fix_hints(self):
        class Hint(XlScheduler, BackgroundScheduler): pass

        return typing.cast(Hint, self)

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

        :param SubprocessTask task:
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
        :param kwargs: 其他aps原本add_schedule、add_job所用参数
            args, kwargs: 这里的参数是给task用的，注意不是给trigger用的，trigger这里的callable模式如果有需要可以使用偏函数等实现。
        :return:
        """
        self.tasks.append(task)
        # 在self.task_defaults数据基础上，叠加kwargs的值作为新的kwargs
        kwargs = {**self.task_defaults, **kwargs}

        if trigger is None:
            if 'id' in kwargs:
                self.scheduler.add_job(task, DateTrigger(datetime.datetime.now()), **kwargs)
            else:
                self.scheduler.add_job(task, **kwargs)
        elif isinstance(trigger, (IntervalTrigger, DateTrigger, CronTrigger)):
            self.scheduler.add_job(task, trigger, **kwargs)
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
            # 使用新的智能解析方法
            real_trigger = XlTrigger.parse_str(trigger)
            self.scheduler.add_job(task, real_trigger, **kwargs)
        elif callable(trigger):
            if 'args' in kwargs or 'kwargs' in kwargs:
                task = partial(task, *kwargs.pop('args', ()), **kwargs.pop('kwargs', {}))
            self.add_job(trigger, args=[self, task], **kwargs)
        else:
            raise ValueError(f'不支持的触发器类型 {type(trigger)}')

    @resolve_params(SubprocessTaskParams, mode='pass')
    def add_cmd(self, args: str | list = '', trigger=None, /, ports=None, locations=None, **resolve_params):
        """ 添加subprocess.Popen类型的任务
        :param ports: 数量取决于ports的配置
        """
        # 虽然从通用角度来说，如果没使用ports参数，这里使用SubprocessTask任务类型也是可以的
        #   但XlServerSubprocessTask其实也兼容SubprocessTask的所有功能，问题不大
        tasks = XlServerSubprocessTask.create_from_ports(args, resolve_params['SubprocessTaskParams'],
                                                         ports=ports, locations=locations)
        del resolve_params['SubprocessTaskParams']
        for task in tasks:
            self.add_schedule(task, trigger, **resolve_params)
        return tasks

    @resolve_params(SubprocessTaskParams, mode='pass')
    def add_py(self, args: str | list = '', trigger=None, /, ports=None, locations=None, **resolve_params):
        if isinstance(args, str):
            args = f'{sys.executable} {args}'
        else:
            args = [sys.executable, *args]
        params = resolve_params['SubprocessTaskParams']
        del resolve_params['SubprocessTaskParams']
        return self.add_cmd(args, trigger, params, ports=ports, locations=locations, **resolve_params)

    @resolve_params(SubprocessTaskParams, mode='pass')
    def add_module(self, args: str | list = '', trigger=None, /, ports=None, locations=None, **resolve_params):
        if isinstance(args, str):
            args = f'{sys.executable} -m {args}'
        else:
            args = [sys.executable, '-m', *args]
        params = resolve_params['SubprocessTaskParams']
        del resolve_params['SubprocessTaskParams']
        return self.add_cmd(args, trigger, params, ports=ports, locations=locations, **resolve_params)

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
            # todo 要判断task类型分类处理
            if not isinstance(task, XlServerSubprocessTask):
                continue

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
        if len(df):
            int_columns = ['pid', 'port']  # 以及你认为需要是整数的其他列
            for col in int_columns:
                df[col] = df[col].astype('Int64')  # 注意 'I' 大写

        return df

    def stop_all(self):
        """ 停止所有后台程序 """
        # 关闭所有调度器
        # if self.scheduler:
        # self.scheduler.shutdown()

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
                self.stop_all()  # 关闭所有正在运行中的task
                # self.stop()  # 关闭apscheduler
                print("所有程序已终止，退出。")
                break
            elif cmd:
                print(eval(cmd))


class XlServerScheduler(XlScheduler):
    def fix_hints(self):
        class Hint(XlServerScheduler, BackgroundScheduler): pass

        return typing.cast(Hint, self)

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

    def __x_导出nginx配置(self):
        pass

    def get_all_locations(self):
        locations = defaultdict(list)
        for task in self.tasks:
            if not (isinstance(task, XlServerSubprocessTask) and task.locations):
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


if __name__ == '__main__':
    XlServerSubprocessTask(args=['2', '3'], shell=True)
