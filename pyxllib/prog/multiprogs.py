#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/12

from types import SimpleNamespace
from unittest.mock import Mock
import ctypes
import os
import socketserver
import subprocess
import sys

from deprecated import deprecated


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

    def __init__(self, name, program, port=None, locations=None,
                 raw_cmd=None, **attrs):
        """
        :param name: 程序昵称
        :param program: 程序对象
        :param port: 是否有执行所在端口
        :param locations: 是否有url地址映射，一般用于nginx等配置
        """
        super().__init__(**attrs)
        self.name = name
        self.program = program
        self.raw_cmd = raw_cmd

        # 这两个是比较特别的属性，在我的工程框架中常用
        self.port = port
        self.locations = locations

    def terminate(self):
        """
        比较优雅结束进程的方法
        """
        if self.program is not None:
            self.program.terminate()

    def kill(self):
        """
        有时候需要强硬的kill方法来结束进程
        """
        if self.program is not None:
            self.program.kill()


class MultiProgramLauncher:
    """
	管理多个程序的启动与终止
	"""

    def __init__(self):
        self.workers = []

    def __1_各种添加进程的机制(self):
        pass

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

    def add_program_cmd(self,
                        cmd,
                        name=None,
                        shell=False,
                        run=True,
                        **attrs):
        """
        启动一个程序，或仅存储任务，并添加进管理列表

        :param cmd: 启动程序的命令
        :param name: 程序名称，如果未提供则从cmd中自动获取
        :param shell:
            False，(优先推荐)直接跟系统交互，此时cmd应该输入数组格式
            True，启用shell进行交互操作，这种情况会更适合管道等模式的处理，此时cmd应该输入字符串格式
        :param attrs: 其他需要传递给ProgramWorker的参数
        :return: 返回一个ProgramWorker实例，表示启动的程序或存储的任务
        """
        # 1 如果未显式传入name，自动从cmd中获取程序名称
        # logger.info(cmd)
        if name is None:
            _cmd = cmd.split() if isinstance(cmd, str) else cmd
            name = _cmd[0]

        # 2 启动或模拟进程
        if run:
            # 2.1 运行进程
            if sys.platform == 'win32':
                proc = subprocess.Popen(cmd, shell=shell)
            else:
                proc = subprocess.Popen(cmd, shell=shell, preexec_fn=self._set_pdeathsig())
        else:
            # 2.2 如果run=False，使用Mock对象占位
            proc = Mock()  # 模拟一个proc对象
            proc.pid = None
            proc.poll.return_value = 'tag'  # 仅标签，此处未实际启动

        # 3 创建ProgramWorker实例并添加到workers列表中
        worker = ProgramWorker(name, proc, raw_cmd=cmd, **attrs)
        self.workers.append(worker)

        return worker

    def add_program_cmd2(self, cmd, ports=1, name=None, **kwargs):
        """
        增强版 add_program_cmd，支持 ports 的处理

        :param ports:
            int 表示要开启的进程数，端口号随机生成；
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

        # 2 遍历端口，依次启动进程
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

    def add_program_python(self, py_file, args='',
                           ports=None, locations=None,
                           name=None, shell=False, executer=None,
                           **kwargs):
        """ 添加并启动一个Python文件作为后台程序 """
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
        return self.add_program_cmd3(cmd, ports=ports, name=name, locations=locations, shell=shell, **kwargs)

    def __2_多进程管理(self):
        pass

    def stop_all(self):
        """ 停止所有后台程序 """
        for worker in self.workers:
            worker.terminate()
        self.workers = []

    def run_endless(self):
        """
        一直运行，直到用户输入 kill 命令或 Ctrl+C

		poll：
            如果进程仍在运行，poll()方法返回None。
            如果进程已经结束，poll()方法返回进程的退出码（exit code）。
                如果进程正常结束，退出码通常为0。
                如果进程异常结束，退出码通常是一个非零值，表示异常的类型或错误码。
		"""
        import pandas as pd
        from pyxllib.algo.stat import print_full_dataframe

        try:
            while True:
                cmd = input('>>>')
                if cmd == 'kill':
                    self.stop_all()
                    break
                elif cmd == 'count':
                    # 需要实际检查现有程序
                    m = sum([1 for worker in self.workers if worker.program.poll() is None])
                    print(f'有{m}个程序正在运行')
                elif cmd == 'list':  # 列出所有程序（转df查看）
                    ls = []
                    for worker in self.workers:
                        ls.append([worker.name, worker.program.pid, worker.program.poll(), worker.raw_cmd])
                    df = pd.DataFrame(ls, columns=['name', 'pid', 'poll', 'args'])
                    print_full_dataframe(df)
        except KeyboardInterrupt:
            print("\n检测到 Ctrl+C，正在终止所有程序...")
            self.stop_all()
            print("所有程序已终止，退出。")


@deprecated(reason='已改名ProgramWorker')
class ProcessWorker(ProgramWorker):
    pass


@deprecated(reason='已改名MultiProgramLauncher')
class MultiProcessLauncher(MultiProgramLauncher):
    pass


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
                mpl = MultiProcessLauncher()
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

                    mpl.add_process_python_module(header, cmds, shell=shell)
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
                mpl = MultiProcessLauncher()
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

                    mpl.add_process_python(sys.argv[0], cmds, shell=shell)
                mpl.run_endless()

        return wrapper

    return decorator
