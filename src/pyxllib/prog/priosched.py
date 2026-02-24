#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/12

import threading
import time
import subprocess
import heapq
import datetime
from queue import PriorityQueue, Empty
from typing import Optional, Union, List, Callable
from dataclasses import dataclass, field

from pyxllib.prog.lazyimport import lazy_import

logger = lazy_import("from loguru import logger")
croniter = lazy_import("from croniter import croniter")


@dataclass(order=True)
class TaskItem:
    """
    用于优先队列的任务包装对象。
    注意：PriorityQueue 是最小堆，所以 priority 存储时取负值，以便权重大的排在前面。
    为了保证相同权重的任务先进先出，引入 arrival_time。
    """

    priority: int  # 实际存储的是 -weight
    arrival_time: float
    task_id: int = field(compare=False)
    cmd: Union[str, List[str]] = field(compare=False)
    timeout: Optional[float] = field(compare=False, default=None)
    cron: Optional[str] = field(compare=False, default=None)
    interval: Optional[float] = field(compare=False, default=None)
    # 用于 Cron/Interval 任务的下一次计算基准时间
    next_base_time: Optional[datetime.datetime] = field(compare=False, default=None)

    @property
    def weight(self):
        return -self.priority


class LinearPriorityScheduler:
    """
    线性优先级调度器

    特点：
    1. 强制线性执行：同一时间只有一个任务在运行。
    2. 优先级抢占：每次任务执行完，都会从队列中取权重最高（weight）的任务执行。
    3. 非抢占式运行：高优先级任务不会打断当前正在运行的任务，必须等待当前任务完成。
    4. 支持 Cron 定时任务和一次性延时任务。
    5. 轻量级设计，仅依赖标准库和 croniter（可选）。
    """

    def __init__(self):
        # 就绪队列：存放所有已经到时间，等待执行的任务
        # 元素格式：TaskItem
        self._ready_queue = PriorityQueue()

        # 计划任务堆：存放还没到时间的任务 (run_timestamp, task_item)
        # 使用 heapq 维护，按时间排序
        self._scheduled_tasks = []
        self._scheduled_lock = threading.Lock()

        # 核心线程控制
        self._shutdown_event = threading.Event()
        self._wakeup_event = threading.Event()  # 用于唤醒调度循环
        self._worker_thread = None

        # 状态标记
        self._current_process = None
        self._task_counter = 0  # 用于生成简单的 task_id

    def start(self, background=True):
        """
        启动调度器
        :param background: 是否在后台线程运行
        """
        if self._worker_thread and self._worker_thread.is_alive():
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 调度器已在运行中。")
            return

        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True, name="LinearPriorityScheduler")
        self._worker_thread.start()
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LinearPriorityScheduler 调度器已启动。")
        
        if not background:
            self._worker_thread.join()

    def stop(self, wait=True):
        """
        停止调度器
        :param wait: 是否等待调度线程完成
        """
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在停止 LinearPriorityScheduler 调度器...")
        self._shutdown_event.set()
        self._wakeup_event.set()  # 唤醒线程以便它能检测到退出信号
        
        if wait and self._worker_thread:
            self._worker_thread.join()
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LinearPriorityScheduler 调度器已停止。")

    def add_task(
        self,
        cmd: Union[str, List[str]],
        weight: int = 100,
        timeout: Optional[float] = None,
        cron: Optional[str] = None,
        interval: Optional[float] = None,
        start_time: Optional[Union[datetime.datetime, float]] = None,
    ):
        """
        添加任务

        :param cmd: 执行的命令，可以是字符串或列表
        :param weight: 权重，默认100。权重越大，优先级越高。
        :param timeout: 超时时间（秒）。如果不设置，则无限等待。
        :param cron: Cron 表达式，例如 "*/5 * * * *"。
        :param interval: 循环间隔（秒）。设置此参数可实现秒级循环任务。
        :param start_time: 指定开始时间。可以是 datetime 对象或 timestamp。如果不指定且没有 cron，则立即放入就绪队列。
        """
        with self._scheduled_lock:
            self._task_counter += 1
            task_id = self._task_counter

        task = TaskItem(
            priority=-weight,  # 权重取反，适配最小堆
            arrival_time=time.time(),
            task_id=task_id,
            cmd=cmd,
            timeout=timeout,
            cron=cron,
            interval=interval,
        )

        now = datetime.datetime.now()
        run_at = None

        # 1. 计算运行时间
        if cron:
            # 如果是 Cron 任务，计算下一次运行时间
            # 注意：croniter 需要 base_time
            base = start_time if isinstance(start_time, datetime.datetime) else now
            iter = croniter(cron, base)
            run_at = iter.get_next(datetime.datetime).timestamp()
            task.next_base_time = datetime.datetime.fromtimestamp(run_at)
        elif interval:
            # 如果是间隔任务
            if start_time:
                # 如果指定了开始时间，就按开始时间排
                if isinstance(start_time, datetime.datetime):
                    run_at = start_time.timestamp()
                    task.next_base_time = start_time
                else:
                    run_at = float(start_time)
                    task.next_base_time = datetime.datetime.fromtimestamp(run_at)
            else:
                # 没指定开始时间，立即执行，并记录当前时间为基准，用于计算下一次
                run_at = None
                task.next_base_time = now
        elif start_time:
            # 指定了开始时间
            if isinstance(start_time, datetime.datetime):
                run_at = start_time.timestamp()
            else:
                run_at = float(start_time)
        else:
            # 立即执行
            run_at = None

        # 2. 放入对应的容器
        if run_at and run_at > time.time():
            # 放入计划任务堆
            with self._scheduled_lock:
                heapq.heappush(self._scheduled_tasks, (run_at, task))
            # logger.debug(f"任务 {task_id} 已计划于 {datetime.datetime.fromtimestamp(run_at)} 执行")
        else:
            # 立即放入就绪队列
            self._ready_queue.put(task)
            # logger.debug(f"任务 {task_id} 已添加到就绪队列 (权重={weight})")

        # 3. 唤醒调度器
        # 无论是加入了就绪队列，还是加入了计划列表（可能比当前等待的任务更早），都应该唤醒检查
        self._wakeup_event.set()

    def _run_loop(self):
        """调度器主循环"""
        while not self._shutdown_event.is_set():
            # 1. 检查计划任务，将到期的移动到就绪队列
            now_ts = time.time()
            with self._scheduled_lock:
                while self._scheduled_tasks and self._scheduled_tasks[0][0] <= now_ts:
                    run_ts, task = heapq.heappop(self._scheduled_tasks)
                    # 更新到达时间，确保相同权重的任务，先被调度的先执行
                    task.arrival_time = now_ts
                    self._ready_queue.put(task)
                    # logger.debug(f"计划任务 {task.task_id} 已移动到就绪队列。")

            # 2. 尝试从就绪队列取任务
            try:
                # 非阻塞获取，如果为空则捕获异常
                task = self._ready_queue.get_nowait()
                self._execute_task(task)

                # 如果是 Cron/Interval 任务，执行完（或开始执行后）需要计算下一次
                # 策略：这里选择在任务取出后立即安排下一次，这样周期更稳定，不受任务执行时长影响
                # 如果希望任务执行完再计算下一次，可以移到 _execute_task 之后
                if task.cron or task.interval:
                    self._reschedule_task(task)

            except Empty:
                # 3. 如果就绪队列为空，决定休眠多久
                wait_seconds = self._calculate_wait_time()
                if wait_seconds > 0:
                    # logger.debug(f"No ready tasks. Waiting for {wait_seconds:.2f}s...")
                    self._wakeup_event.wait(wait_seconds)
                    self._wakeup_event.clear()
                else:
                    # 如果不需要等待（可能有立即到期的计划任务），稍微 yield 一下避免死循环占满 CPU
                    # 但理论上 _calculate_wait_time 返回 0 意味着有任务马上到了
                    pass

    def _calculate_wait_time(self):
        """计算距离下一个计划任务还有多久"""
        with self._scheduled_lock:
            if not self._scheduled_tasks:
                return 3600 * 24  # 如果没有计划任务，默认睡很久，直到被 add_task 唤醒

            next_run_ts = self._scheduled_tasks[0][0]
            now_ts = time.time()
            wait = next_run_ts - now_ts
            return max(0, wait)

    def _reschedule_task(self, task: TaskItem):
        """重新调度 Cron/Interval 任务"""
        try:
            # 基于上一次的计划执行时间计算下一次，防止时间漂移
            # 如果没有 next_base_time，则用当前时间
            base = task.next_base_time if task.next_base_time else datetime.datetime.now()
            
            if task.cron:
                iter = croniter(task.cron, base)
                next_dt = iter.get_next(datetime.datetime)
            elif task.interval:
                # 间隔循环
                next_dt = base + datetime.timedelta(seconds=task.interval)
            else:
                return

            next_ts = next_dt.timestamp()

            # 创建新任务实例（为了保持状态纯洁，虽然 TaskItem 是可变的，但最好新建）
            # 注意：priority 保持不变
            new_task = TaskItem(
                priority=task.priority,
                arrival_time=time.time(),  # 暂时占位，实际放入 ready queue 时会更新
                task_id=self._task_counter + 1,  # 也可以复用 ID，或者自增
                cmd=task.cmd,
                timeout=task.timeout,
                cron=task.cron,
                interval=task.interval,
                next_base_time=next_dt,
            )
            # 这里我简单处理，不自增 ID 了，复用 ID 方便追踪是同一个 Cron 任务
            new_task.task_id = task.task_id

            with self._scheduled_lock:
                heapq.heappush(self._scheduled_tasks, (next_ts, new_task))
            
            # logger.debug(f"任务 {task.task_id} 已重新调度于 {next_dt}")
            
        except Exception as e:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 任务 {task.task_id} 重新调度失败: {e}")

    def _execute_task(self, task: TaskItem):
        """执行单个任务"""
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 任务 {task.task_id} 启动")
        
        try:
            # 兼容字符串和列表形式的 cmd
            cmd = task.cmd
            shell = isinstance(cmd, str)

            self._current_process = subprocess.Popen(
                cmd,
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",  # 显式指定编码，防止中文乱码
                errors="replace",
            )

            try:
                stdout, stderr = self._current_process.communicate(timeout=task.timeout)
                return_code = self._current_process.returncode

                if return_code == 0:
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 任务 {task.task_id} 完成。")
                else:
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 任务 {task.task_id} 失败，返回码 {return_code}。")
                    if stderr:
                        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 任务 {task.task_id} 错误:\n{stderr.strip()}")
                        
            except subprocess.TimeoutExpired:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 任务 {task.task_id} 在 {task.timeout}s 后超时。正在终止...")
                self._current_process.kill()
                self._current_process.communicate()
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 任务 {task.task_id} 已终止。")
                
        except Exception as e:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 执行任务 {task.task_id} 时出错: {e}")
        finally:
            self._current_process = None


if __name__ == "__main__":
    import sys

    # 初始化调度器
    scheduler = LinearPriorityScheduler()
    scheduler.start()

    # 辅助函数：根据平台生成模拟耗时的命令
    def get_sleep_cmd(seconds, msg):
        if sys.platform == "win32":
            # Windows ping -n N 大约耗时 N-1 秒，所以要 +1
            # 但为了简单，这里直接用 python -c 来模拟耗时，更准确跨平台
            return [sys.executable, "-c", f"import time; time.sleep({seconds})"]
        else:
            return f"sleep {seconds}"
    
    # 由于 croniter 标准不支持秒级，且为了演示简单直观，
    # 我们这里直接使用新增的 interval 参数来演示自动定时任务。
    # 这样就不需要外部线程 task_producer 了。
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] >>> 系统启动，添加自动循环任务...")
    
    # 任务1 (耗时10s, 权重100, 每60秒执行一次)
    scheduler.add_task(get_sleep_cmd(10, "任务1"), weight=100, interval=60)
    
    # 任务2 (耗时20s, 权重100, 每60秒执行一次)
    scheduler.add_task(get_sleep_cmd(20, "任务2"), weight=100, interval=60)
    
    # 任务3 (耗时30s, 权重100, 每60秒执行一次)
    scheduler.add_task(get_sleep_cmd(30, "任务3"), weight=100, interval=60)
    
    # 任务4 (耗时5s, 权重200, 高优, 每30秒执行一次)
    # 我们给它设定一个 start_time，让它在稍后一点开始，模拟错峰
    start_delay = datetime.datetime.now() + datetime.timedelta(seconds=5)
    scheduler.add_task(get_sleep_cmd(5, "任务4(高优)"), weight=200, interval=30, start_time=start_delay)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.stop()
