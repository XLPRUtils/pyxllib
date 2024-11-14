#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/13

import os
import time
import platform
import tempfile

if platform.system() == 'Windows':
    import msvcrt
else:
    import fcntl


class FileLock:
    """
    文件锁类，用于在不同进程之间实现互斥锁，确保同一时间只有一个程序实例可以访问共享资源。

    适用于以下场景：
    - 防止多个独立程序同时操作同一资源（如文件、窗口、数据库等）。
    - 解决多个自动化脚本并发访问相同应用程序时的冲突问题。
    - 跨平台支持：在 Windows 上使用 `msvcrt`，在 Unix 系统上使用 `fcntl`。

    使用方式：
    1. 作为上下文管理器：
        >> with FileLock('wechat.lock'):
        >>     # 执行独占的操作
        >>     print("操作微信...")

    2. 手动调用 acquire() 和 release()：
        >> lock = FileLock('wechat.lock')
        >> try:
        >>     lock.acquire()
        >>     print("正在执行任务...")
        >> finally:
        >>     lock.release()

    属性：
    :param bool locked: 指示当前是否已获得文件锁。

    异常：
    - TimeoutError: 如果在指定的超时时间内无法获取锁，则抛出此异常。
    """

    def __init__(self, lock_file='lockfile.lock', timeout=10, check_interval=0.1):
        """

        :param str lock_file: 锁文件的路径，默认值为 'lockfile.lock'。
        :param int timeout: 获取锁的超时时间，单位为秒，默认值为 10 秒。
        :param float check_interval: 重试获取锁的时间间隔，单位为秒，默认值为 0.1 秒。
        """
        # 如果 lock_file 没有提供绝对路径，则将其存储在临时目录下
        if not os.path.isabs(lock_file):
            lock_file = os.path.join(tempfile.gettempdir(), lock_file)

        self.lock_file = lock_file
        self.timeout = timeout
        self.check_interval = check_interval
        self.locked = False
        self.file = None

    def acquire(self):
        """
        手动获取文件锁。

        如果文件已被其他进程锁定，则会阻塞，直到成功获取锁或者超时。
        如果在指定的超时时间内无法获取锁，则抛出 TimeoutError。

        Raises:
            TimeoutError: 如果在指定的超时时间内无法获取锁。
        """
        start_time = time.time()
        while True:
            try:
                self.file = open(self.lock_file, 'w')
                if platform.system() == 'Windows':
                    # Windows 系统使用 msvcrt.locking() 实现文件锁
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    # Unix 系统使用 fcntl.flock() 实现文件锁
                    fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.locked = True
                break
            except (BlockingIOError, OSError):
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"无法在 {self.timeout} 秒内获取文件锁：{self.lock_file}")
                time.sleep(self.check_interval)

    def release(self):
        """
        手动释放文件锁。

        如果锁已被获取，则解锁并关闭文件句柄。
        """
        if self.locked and self.file:
            if platform.system() == 'Windows':
                # Windows 系统释放锁
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                # Unix 系统释放锁
                fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()
            self.locked = False

    def __enter__(self):
        """
        进入上下文管理器时自动获取文件锁。

        Returns:
            self: 返回当前 FileLock 实例。
        """
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文管理器时自动释放文件锁。
        """
        self.release()


# 示例：如何使用 FileLock 类
if __name__ == '__main__':
    # 使用上下文管理器
    with FileLock('wechat.lock') as lock:
        print("通过上下文管理器获取锁，正在操作微信...")
        time.sleep(3)
        print("操作完成")

    # 手动调用 acquire 和 release
    lock = FileLock('wechat.lock')
    try:
        lock.acquire()
        print("手动获取锁，正在执行任务...")
        time.sleep(3)
    finally:
        lock.release()
        print("任务完成并已释放锁")
