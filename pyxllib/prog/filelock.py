#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/13

import os
import tempfile
import time

from pyxllib.prog.lazyimport import lazy_import

try:
    from filelock import FileLock, Timeout
except ModuleNotFoundError:
    FileLock = lazy_import('from filelock import FileLock')
    Timeout = lazy_import('from filelock import Timeout')


class XlFileLock(FileLock):
    def __init__(self, lock_file: str, timeout: float = -1, **kwargs):
        """
        初始化 XlFileLock 实例，锁文件默认存储在系统临时目录中。

        :param str filename: 锁文件的文件名（不包含路径）。
        :param float timeout: 获取锁的超时时间（以秒为单位），默认值为 -1 表示无限等待。
        """
        lock_file_path = os.path.join(tempfile.gettempdir(), lock_file)
        super().__init__(lock_file=lock_file_path, timeout=timeout, **kwargs)


def get_autogui_lock(lock_file='autogui.lock', timeout=-1):
    """
    注意：XlFileLock中的lock_file无法设置默认值，这跟FileLock的底层有关
        所以无法继承一个AutoUiLock类，但是可以通过这种函数的方式，绕开它特殊的初始化限制
    """
    lock = XlFileLock(lock_file=lock_file, timeout=timeout)
    return lock


if __name__ == "__main__":
    try:
        with get_autogui_lock(timeout=5):
            print("Lock acquired, doing some work...")
            time.sleep(3)
            print("Work done.")
    except Timeout:
        print("Failed to acquire lock within the timeout.")
