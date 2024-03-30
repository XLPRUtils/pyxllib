#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/03/28

""" 使用subprocess来实现"多进程"并发 """

import os
import subprocess
import sys
import time
import atexit

from pyxllib.prog.pupil import tprint


class MultiProcessLauncher:
    def __init__(self):
        self.workers = []
        atexit.register(self.cleanup)  # 在程序退出时注册清理方法

    def _add_posix_process(self, cmd):
        import ctypes
        import signal

        def _set_pdeathsig(sig=signal.SIGTERM):
            """ Help function to ensure once parent process exits,
            its children processes will automatically die on Linux.
            """

            def callable():
                libc = ctypes.CDLL("libc.so.6")
                return libc.prctl(1, sig)

            return callable

        preexec_fn = _set_pdeathsig(signal.SIGTERM)
        p = subprocess.Popen(cmd.split(), preexec_fn=preexec_fn)

        return p

    def add_process(self, cmd, name=None, **kwargs):
        # 240329周五16:57，最新测试，好像不捕捉，主进程结束后子进程也是会自动结束的。跟我以前了解的机制似乎有点不一样。
        #   但加着也算双重保险，不会出错吧。
        if os.name == 'posix':
            p = self._add_posix_process(cmd)
        else:
            p = subprocess.Popen(cmd)

        worker = {
            'name': cmd.split()[0] if name is None else name,
            'process': p,
            'cmd': cmd,
            'status': 'running'  # 初始状态设为running
        }
        worker.update(kwargs)

        self.workers.append(worker)

    def add_cur_py_process(self, cmd, name=None, **kwargs):
        """ 添加当前python环境下的进程 """
        if isinstance(cmd, str):
            cmd = f'"{sys.executable}" {cmd}'
        elif isinstance(cmd, list):
            cmd = [sys.executable] + cmd

        self.add_process(cmd, name, **kwargs)

    def manage(self):
        """ 进入交互管理界面 """

        while True:
            running_count = 0
            for worker in self.workers:
                if worker['process'].poll() is None:  # 如果进程仍然在运行
                    running_count += 1
                else:
                    worker['status'] = 'completed'

            # 检查并输出状态
            if running_count == 0:
                print("所有进程都已完成。")
                break

            cmd = input(f'>>>  ')
            if cmd == "status":
                print(f"正在运行的进程数：{running_count}/{len(self.workers)}")
            # 你可以在这里根据需要扩展其他指令，比如停止特定进程等

    def wait_all(self):
        """ 进入交互管理界面 """

        seconds = 1  # 逐步增加等待间隔
        while True:
            running_count = 0
            for worker in self.workers:
                if worker['process'].poll() is None:  # 如果进程仍然在运行
                    running_count += 1
                else:
                    worker['status'] = 'completed'
                    tprint(f"{worker['name']} 已完成。")

            # 检查并输出状态
            if running_count == 0:
                print("所有进程都已完成。")
                break

            time.sleep(seconds)
            seconds = min(seconds + 1, 10)

    def cleanup(self):
        """ 在退出时终止所有子进程 """
        for worker in self.workers:
            process = worker['process']
            if process.poll() is None:  # 如果进程还在运行
                try:
                    process.terminate()  # 尝试终止进程
                    process.wait()  # 等待进程结束
                except Exception as e:
                    print(f"Error terminating process {worker['name']}: {e}")
