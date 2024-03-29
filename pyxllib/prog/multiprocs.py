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

import fire

from pyxllib.prog.pupil import tprint


class MultiProcessLauncher:
    def __init__(self):
        self.workers = []

    def add_process(self, cmd, name=None, **kwargs):
        # if isinstance(cmd, str):
        #     cmd = cmd.split()  # todo 这种硬切分是不严谨的，但应急先这样处理

        if name is None:
            name = cmd.split()[0]

        p = subprocess.Popen(cmd)
        # preexec_fn=_set_pdeathsig(signal.SIGTERM))

        worker = {
            'name': name,
            'process': p,
            'cmd': cmd,
            'status': 'running'  # 初始状态设为running
        }
        worker.update(kwargs)

        self.workers.append(worker)

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


def trial_func(n):
    for i in range(n):
        print(i)
        time.sleep(1)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        fire.Fire()
    else:
        launcher = MultiProcessLauncher()
        launcher.add_process(f"{sys.executable} multiprocs.py trial_func 5", name="a")
        launcher.add_process(f"{sys.executable} multiprocs.py trial_func 10", name="b")
        launcher.add_process(f"{sys.executable} multiprocs.py trial_func 15", name="c")
        launcher.wait_all()
