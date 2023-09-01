#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2023/08/26

# from pyxllib.prog.pupil import check_install_package

# check_install_package('IPython', 'ipython')

import os
import sys
from multiprocessing import Process, Pipe
from io import StringIO
import threading

# from IPython.core.interactiveshell import InteractiveShell

from timeout_decorator import timeout

from pyxllib.prog.pupil import format_exception
from pyxllib.file.specialist import XlPath

if sys.platform == 'win32':
    def timeout(seconds):
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = None
                exception = None

                # 定义一个内部函数来运行目标函数
                def worker():
                    nonlocal result, exception
                    try:
                        result = func(*args, **kwargs)
                    except Exception as e:
                        exception = e

                thread = threading.Thread(target=worker)
                thread.start()
                thread.join(timeout=seconds)

                if thread.is_alive():
                    thread.join()  # 确保线程结束
                    raise TimeoutError(f"Function '{func.__name__}' exceeded {seconds} seconds of execution time.")
                if exception:
                    raise exception
                return result

            return wrapper

        return decorator


class InProcessExecutor:
    def __init__(self, base_dir=None):
        """ 注意为了应急，这里暂时是很不工程化的写法，一旦初始化这个类，工作目录、stdout、stderr 就被改变了 """
        self.base_dir = XlPath(base_dir) if base_dir else None
        self.user_ns = {}

        # 保存原始的工作目录、stdout 和 stderr
        self._original_cwd = os.getcwd()
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        if self.base_dir:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(self.base_dir)

    def _execute_code(self, code):
        response = {}
        try:
            # 捕获输出
            old_stdout, old_stderr = sys.stdout, sys.stderr
            redirected_output = StringIO()
            sys.stdout, sys.stderr = redirected_output, redirected_output

            # 使用 exec 执行代码
            exec(code, self.user_ns)

            # 恢复原始的 stdout 和 stderr
            sys.stdout, sys.stderr = old_stdout, old_stderr

            response['output'] = redirected_output.getvalue().strip()

        except Exception as e:
            response['error'] = format_exception(e)

        return response

    # @timeout(10)  # 先把超时写死了，以后有空再研究怎么加到函数的timeout参数里
    def execute(self, code: str, silent=False):
        """ 在主进程中执行代码 """
        response = self._execute_code(code)

        if not silent and response.get('output'):
            print(response['output'])

        if response.get('error'):
            print(f"执行错误: {response['error']}")

        return response

    def get_var(self, var_name: str):
        """ 获取指定的变量值 """
        return self.user_ns.get(var_name, None)

    def set_var(self, var_name: str, value):
        """ 设置变量的值 """
        self.user_ns[var_name] = value

    def close(self):
        """ 恢复原始的工作目录、stdout 和 stderr """
        os.chdir(self._original_cwd)
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class SubprocessExecutor:
    """ 子进程版本 """

    def __init__(self, base_dir=None):
        """ 初始化执行器并启动子进程 """
        self.parent_conn, self.child_conn = Pipe()
        self.base_dir = XlPath(base_dir)
        self.process = Process(target=self._child_process, args=(self.child_conn, base_dir))
        self.process.start()

    def _child_process(self, conn, base_dir):
        """ 子进程主循环 """
        if base_dir:
            os.chdir(base_dir)

        user_ns = {}  # 初始化一个空的用户命名空间来替代 InteractiveShell 的 user_ns

        while True:
            message = conn.recv()  # 从父进程接收消息
            if message.get('action') == 'exit':
                break
            elif message.get('action') == 'execute':
                code = message.get('code')
                response = self._execute_code(code, user_ns)  # 传递user_ns为参数
                conn.send(response)
            elif message.get('action') == 'get_var':
                var_name = message.get('name')
                response = {'output': str(user_ns.get(var_name, None))}
                conn.send(response)
            elif message.get('action') == 'set_var':
                var_name = message.get('name')
                value = message.get('value')
                user_ns[var_name] = value

    def _execute_code(self, code, user_ns):
        response = {}
        try:
            # Capture output
            old_stdout, old_stderr = sys.stdout, sys.stderr
            redirected_output = StringIO()
            sys.stdout, sys.stderr = redirected_output, redirected_output

            # 使用 exec 来执行代码
            exec(code, user_ns)  # 使用传入的用户命名空间来执行代码

            # Restore original stdout and stderr
            sys.stdout, sys.stderr = old_stdout, old_stderr

            response['output'] = redirected_output.getvalue().strip()

        except Exception as e:
            response['error'] = format_exception(e)

        return response

    # @timeout(10)
    def execute(self, code: str, silent=False):
        """ 在子进程中执行代码 """
        self.parent_conn.send({'action': 'execute', 'code': code})
        response = self.parent_conn.recv()

        # 如果不处于静默模式，则处理响应
        if not silent:
            if response.get('output'):
                print(response['output'], end='')
            if response.get('error'):
                print(f"Error: {response['error']}")

        return response

    def get_var(self, var_name: str):
        """ 获取指定的环境变量值 """
        self.parent_conn.send({'action': 'get_var', 'name': var_name})
        response = self.parent_conn.recv()
        return response['output']

    def set_var(self, var_name: str, value):
        """ 设置环境变量的值 """
        self.parent_conn.send({'action': 'set_var', 'name': var_name, 'value': value})

    def close(self):
        """ 关闭子进程 """
        self.parent_conn.send({'action': 'exit'})
        self.process.join()


if os.getppid() == os.getpid():  # 主进程的时候，另外建立子进程运行更好，也便于调试
    AutoProcessExecutor = SubprocessExecutor
else:  # 并发跑的时候，只能在每个进程里，不能另外再套进程
    AutoProcessExecutor = InProcessExecutor


class InteractiveExecutor(SubprocessExecutor):
    """ 这个类是比较定制化的，还不算通用工具类

    但对我来说，暂时会经常用到，所以就写到这里来了
    """

    def execute(self, code: str, silent=False):
        """ 在子进程中执行代码

        相比父类，运行前会先记录当前文件清单，运行后会记录新的文件清单，然后对比两个文件清单，找出新的文件
        1. 为了简化问题，暂时只考虑工作目录下的直接文件，不考虑子目录，也不考虑对同名文件可能进行的修改
        2. 假设操作都只会在工作目录里，不会对工作目录外的文件使用绝对路径等手动操作
        """
        # 1 记录运行前的文件清单
        files1 = {f.name for f in self.base_dir.glob_files()}

        # 2 执行代码
        self.parent_conn.send({'action': 'execute', 'code': code})
        response = self.parent_conn.recv()

        # 3 记录运行后的文件清单
        files2 = {f.name for f in self.base_dir.glob_files()}
        out_files = list(files2 - files1)  # 相减后视为新增的文件
        if out_files:
            response['out_files'] = out_files

        # 如果不处于静默模式，则处理响应
        if not silent:
            if response['output']:
                print(response['output'])
            if 'error' in response:
                print(f"Error: {response['error']}")

        return response


if __name__ == '__main__':
    # 读取指定工作目录里的文件；只在指定工作目录里创建文件
    ie = SubprocessExecutor(XlPath.desktop())
    ie.execute('with open("a.txt", "r", encoding="utf8") as f: print(f.read())')
    # Hello
    r = ie.execute('with open("b.txt", "w", encoding="utf8") as f: f.write ("hello123")')
    print(r)  # {'output': '', 'files': {'b.txt'}}

    print(ie.get_var('test'))  # 索引不存在时返回None
