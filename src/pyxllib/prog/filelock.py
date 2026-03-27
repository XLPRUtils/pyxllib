#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/13

import json
import logging
import os
import socket
import tempfile
import time
import uuid

from pyxllib.prog.lazyimport import lazy_import

try:
    from filelock import Timeout
except ModuleNotFoundError:
    Timeout = lazy_import('from filelock import Timeout')

logger = logging.getLogger(__name__)


class XlFileLock:
    """可强制夺取的文件锁。

    这里不再直接使用 filelock 的 Windows 硬锁实现。
    原实现一旦持锁进程卡死但不退出，其他进程无法“强制释放”。
    现在改为基于锁文件存在性的软锁：
    1. 正常情况下，依赖 O_CREAT | O_EXCL 原子创建保证互斥。
    2. 等待超过 force_break_timeout 后，后继进程会删除旧锁文件并重新抢锁。

    代价是：若旧持锁方还在运行，强制夺锁后可能出现并发操作。
    这是为了满足 Windows UI 自动化场景“宁可抢锁，也不要永久卡死”的需求。
    """

    def __init__(self, lock_file: str, timeout: float = -1,
                 *, force_break_timeout: float = -1, poll_interval: float = 0.2):
        self.lock_file = os.path.join(tempfile.gettempdir(), lock_file)
        self.timeout = timeout
        self.force_break_timeout = force_break_timeout
        self.poll_interval = poll_interval

        self._lock_counter = 0
        self._token = None

    def _read_lock_info(self):
        try:
            with open(self.lock_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def _try_acquire_once(self, token):
        lock_info = {
            'token': token,
            'pid': os.getpid(),
            'hostname': socket.gethostname(),
            'created_at': time.time(),
        }

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        fd = os.open(self.lock_file, flags)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(lock_info, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    def _force_break(self, waited_seconds):
        info = self._read_lock_info() or {}
        try:
            os.remove(self.lock_file)
        except FileNotFoundError:
            return
        except PermissionError:
            # 极少数情况下文件正被其他操作占用，继续等待重试即可
            return

        holder = []
        if info.get('pid') is not None:
            holder.append(f"pid={info['pid']}")
        if info.get('hostname'):
            holder.append(f"host={info['hostname']}")
        holder_text = ', '.join(holder) or 'unknown holder'
        logger.warning('Force break autogui lock after %.1fs: %s (%s)',
                       waited_seconds, self.lock_file, holder_text)

    def acquire(self, timeout=None, poll_interval=None):
        if self._lock_counter > 0:
            self._lock_counter += 1
            return self

        timeout = self.timeout if timeout is None else timeout
        poll_interval = self.poll_interval if poll_interval is None else poll_interval
        start = time.monotonic()

        while True:
            token = uuid.uuid4().hex
            try:
                self._try_acquire_once(token)
            except FileExistsError:
                waited_seconds = time.monotonic() - start

                if self.force_break_timeout >= 0 and waited_seconds >= self.force_break_timeout:
                    self._force_break(waited_seconds)
                    time.sleep(min(poll_interval, 0.1))
                    continue

                if timeout >= 0 and waited_seconds >= timeout:
                    raise Timeout(self.lock_file)

                time.sleep(poll_interval)
            else:
                self._token = token
                self._lock_counter = 1
                return self

    def release(self):
        if self._lock_counter == 0:
            return

        self._lock_counter -= 1
        if self._lock_counter > 0:
            return

        token = self._token
        self._token = None

        info = self._read_lock_info()
        if info is None:
            return

        if info.get('token') != token:
            logger.warning('Skip releasing autogui lock because ownership changed: %s', self.lock_file)
            return

        try:
            os.remove(self.lock_file)
        except FileNotFoundError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def get_autogui_lock(lock_file='autogui.lock', timeout=-1, force_break_timeout=60):
    """
    创建 UI 自动化全局锁。

    :param lock_file: 锁文件名，实际保存在系统临时目录
    :param timeout: 正常等待获取锁的最长时间，-1 表示无限等待
    :param force_break_timeout: 等待超过该秒数后，强制删除旧锁并重新抢锁；-1 表示禁用
    """
    return XlFileLock(lock_file=lock_file, timeout=timeout,
                      force_break_timeout=force_break_timeout)


if __name__ == "__main__":
    try:
        with get_autogui_lock(timeout=5, force_break_timeout=2):
            print("Lock acquired, doing some work...")
            time.sleep(3)
            print("Work done.")
    except Timeout:
        print("Failed to acquire lock within the timeout.")
