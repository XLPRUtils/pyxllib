#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 18:40


from pyxllib.prog.specialist.common import *
from pyxllib.prog.specialist.xllog import *
from pyxllib.prog.specialist.browser import *
from pyxllib.prog.specialist.bc import *
from pyxllib.prog.specialist.tictoc import *

import concurrent.futures
import functools
import os
import re
from statistics import mean
import subprocess
from threading import Thread
import time
from typing import Type, Literal, Dict, Tuple, Container, Any

from pyxllib.prog.lazyimport import lazy_import

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lazy_import('from tqdm import tqdm')

try:
    import requests
except ModuleNotFoundError:
    requests = lazy_import('requests')

try:
    from humanfriendly import parse_size
except ModuleNotFoundError:
    parse_size = lazy_import('from humanfriendly import parse_size')

try:
    from pydantic import BaseModel
except ModuleNotFoundError:
    BaseModel = lazy_import('from pydantic import BaseModel')

from pyxllib.prog.newbie import human_readable_size
from pyxllib.prog.pupil import get_installed_packages, aligned_range, percentage_and_value
from pyxllib.prog.xlenv import XlEnv
from pyxllib.file.specialist import cache_file


def mtqdm(func, iterable, *args, max_workers=1, check_per_seconds=0.01, **kwargs):
    """ 对tqdm的封装，增加了多线程的支持

    这里名称前缀多出的m有multi的意思

    :param max_workers: 默认是单线程，改成None会自动变为多线程
        或者可以自己指定线程数
        注意，使用负数，可以用对等绝对值数据的“多进程”
    :param smoothing: tqdm官方默认值是0.3
        这里关掉指数移动平均，直接计算整体平均速度
        因为对我个人来说，大部分时候需要严谨地分析性能，得到整体平均速度，而不是预估当前速度
    :param mininterval: 官方默认值是0.1，表示显示更新间隔秒数
        这里不用那么频繁，每秒更新就行了~~
    :param check_per_seconds: 每隔多少秒检查队列
        有些任务，这个值故意设大一点，可以减少频繁的队列检查时间，提高运行速度
    整体功能类似Iterate
    """

    # 0 个人习惯参数
    kwargs['smoothing'] = kwargs.get('smoothing', 0)
    kwargs['mininterval'] = kwargs.get('mininterval', 1)

    if max_workers == 1:
        # 1 如果只用一个线程，则不使用concurrent.futures.ThreadPoolExecutor，能加速
        for x in tqdm(iterable, *args, **kwargs):
            func(x)
    else:
        # 2 默认的多线程运行机制，出错是不会暂停的；这里对原函数功能进行封装，增加报错功能
        error = False

        def wrap_func(x):
            nonlocal error
            try:
                func(x)
            except Exception as e:
                error = e

        # 3 多线程/多进程 和 进度条 功能的结合
        if max_workers > 1:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers)
            for x in tqdm(iterable, *args, **kwargs):
                while executor._work_queue.qsize():
                    if check_per_seconds:
                        time.sleep(check_per_seconds)
                executor.submit(wrap_func, x)
                if error:
                    raise error
        else:
            executor = concurrent.futures.ProcessPoolExecutor(-max_workers)
            for x in tqdm(iterable, *args, **kwargs):
                # while executor._call_queue.pending_work_items:
                #     if check_per_seconds:
                #         time.sleep(check_per_seconds)
                executor.submit(wrap_func, x)
                if error:
                    raise error

        executor.shutdown()


def estimate_package_size(package):
    """ 估计一个库占用的存储大小 """

    # 将cache文件存储到临时目录中，避免重复获取网页
    def get_size(package):
        r = requests.get(f'https://pypi.org/project/{package}/#files')
        if r.status_code == 404:
            return '(0 MB'  # 找不到的包默认按0MB计算
        else:
            return r.text

    s = cache_file(package + '.pypi', lambda: get_size(package))
    # 找出所有包大小，计算平均值作为这个包大小的预估
    # 注意，这里进位是x1000，不是x1024
    v = mean(list(map(parse_size, re.findall(r'\((\d+(?:\.\d+)?\s*\wB(?:ytes)?)', s))) or [0])
    return v


def estimate_pip_packages(*, print_mode=False):
    """ 检查pip list中包的大小，从大到小排序

    :param print_mode:
        0，不输出，只返回运算结果，[(package_name, package_size), ...]
        1，输出最后的美化过的运算表格
        2，输出中间计算过程
    """

    def printf(*args, **kwargs):
        # dm表示mode增量
        if print_mode > 1:
            print(*args, **kwargs)

    packages = get_installed_packages()
    package_sizes = []
    for package_name in packages:
        package_size = estimate_package_size(package_name)
        package_sizes.append((package_name, package_size))
        printf(f"{package_name}: {human_readable_size(package_size)}")

    package_sizes.sort(key=lambda x: (-x[1], x[0]))
    if print_mode > 0:
        if print_mode > 1: print('- ' * 20)
        for package_name, package_size in package_sizes:
            print(f"{package_name}: {human_readable_size(package_size)}")
    return package_sizes


class ProgressBar:
    """ 对运行可能需要较长时间的任务，添加进度条显示

    # 示例用法
    with ProgressBar(100) as pb:
        for i in range(100):
            time.sleep(0.1)  # 模拟耗时工作
            pb.progress = i + 1  # 更新进度
    """

    def __init__(self, total):
        self.total = total  # 总进度
        self.progress = 0  # 当前进度
        self.stop_flag = False  # 停止标志

    def __enter__(self):
        # 启动进度显示线程
        self.progress_thread = Thread(target=self.display_progress)
        self.progress_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 强制将进度设置为100%
        self.progress = self.total
        # 停止进度显示线程
        self.stop_flag = True
        self.progress_thread.join()

    def display_progress(self):
        with tqdm(total=self.total) as pbar:
            while not self.stop_flag:
                pbar.n = self.progress
                pbar.refresh()
                time.sleep(1)
            pbar.n = self.progress
            pbar.refresh()


class BitMaskTool:
    """ 二进制位掩码工具

    概念术语
        bitval，每一个位上的具体取值，0或1
        bitsum，所有位上的取值的和，即二进制数的十进制表示
    """

    def __init__(self, bit_names=None, bitsum_counter=None):
        """ 初始化 BitMaskTool 对象

        :param list bit_names: 每一位功能开启时，显示的标签名。未输入时，填充0,1,2,3...注意总数，要按域宽对齐
        :param dict/list bitsum_counter: 各种值出现的次数，可以是一个字典或者一个包含出现次数的列表
        """
        # 1 每一位功能开启时，显示的标签名。未输入时，填充0,1,2,3...注意总数，要按域宽对齐
        if bit_names is None:
            bit_names = list(aligned_range(len(bit_names)))
        self.bit_names = bit_names
        # 2 各种值出现的次数
        if isinstance(bitsum_counter, (list, tuple)):
            bitsum_counter = Counter(bitsum_counter)
        self.bitsum_counter = bitsum_counter or {}

    def get_bitsum_names(self, bitsum):
        """ 从bitsum得到names的拼接

        >> get_bitsum_names(3)
        '语义定位,图表'
        """
        if not isinstance(bitsum, int):
            try:
                bitsum = int(bitsum)
            except ValueError:
                bitsum = 0

        tags = []
        for i, k in enumerate(self.bit_names):
            if (1 << i) & bitsum:
                tags.append(k)
        return ','.join(tags)

    def count_bitsum_relations(self, target_bitsum, relation='='):
        """ 计算特定关系的 bitsum 数量

        :param int target_bitsum: 目标 bitsum
        :param str relation: 关系类型，可以是 '=', '⊂', '⊃'
            假设bitval对应的bit_names为n1,n2,n3,n4。
            那么bitsum相当于是bitval的一个集合
            比如a={n1,n3,n4}，b={n1,n3}，因为a完全包含b，所以认为a⊃b，或者a⊋b、b⊂a、b⊊a
        :return int: 符合条件的 bitsum 数量
        """
        count = 0
        if relation == '=':
            # 直接计算等于 target_bitsum 的数量
            count = self.bitsum_counter.get(target_bitsum, 0)
        elif relation == '⊂':
            # 计算所有被 target_bitsum 包含的 bitsum 的数量
            for bitsum, num in self.bitsum_counter.items():
                if bitsum and bitsum & target_bitsum == bitsum:
                    count += num
        elif relation == '⊃':
            # 计算所有包含 target_bitsum 的 bitsum 的数量
            for bitsum, num in self.bitsum_counter.items():
                if bitsum & target_bitsum == target_bitsum:
                    count += num
        return count

    def check_bitflag(self, max_bitsum_len=None, reletion='=',
                      filter_zero=False, sort_by=None, *,
                      min_bitsum_len=0):
        """ 检查并返回 bitsum 关系的 DataFrame

        :param int max_bitsum_len: 最大 bitsum 长度
        :param str reletion: 关系类型，可以是 '=', '⊂', '⊃'
            支持输入多个字符，表示要同时计算多种关系
        :param bool filter_zero: 是否过滤掉零值
        :param None|str sort_by: 排序字段
            None, 默认排序
            count, 按照数量从大到小排序
            bitsum, 按照 bitsum 从小到大排序
        :param int min_bitsum_len: 最小 bitsum 长度
        :return: 包含 bitsum 关系的 DataFrame
        """
        from itertools import combinations

        total = sum(self.bitsum_counter.values())
        rows, columns = [], ['类型', '名称', '百分比.次数']
        rows.append([-1, '总计', total])

        if max_bitsum_len is None:
            max_bitsum_len = len(self.bit_names)

        bitvals = [(1 << i) for i in range(len(self.bit_names))]
        for m in range(min_bitsum_len, max_bitsum_len + 1):
            for comb in combinations(bitvals, m):
                bitsum = sum(comb)
                count = self.count_bitsum_relations(bitsum, relation=reletion)
                if filter_zero and count == 0:
                    continue
                rows.append([f'{reletion}{bitsum}',
                             self.get_bitsum_names(bitsum),
                             count])

        if sort_by == 'count':
            rows.sort(key=lambda x: x[2], reverse=True)
        elif sort_by == 'bitsum':
            rows.sort(key=lambda x: int(x[0][1:]) if isinstance(x[0], str) else x[0])

        df = pd.DataFrame.from_records(rows, columns=columns)
        df['百分比.次数'] = percentage_and_value(df['百分比.次数'], 2, total=total)
        return df

    def report(self):
        """ 生成统计报告 """
        html_content = []

        html_content.append('<h1>1 包含每一位bitval特征的数量</h1>')
        df1 = self.check_bitflag(1, '⊃')
        html_content.append(df1.to_html())

        html_content.append('<h1>2 每一种具体bitsum组合的数量</h1>')
        df2 = self.check_bitflag(reletion='=', filter_zero=True, sort_by='bitsum')
        html_content.append(df2.to_html())

        return '\n'.join(html_content)


def loguru_setup_jsonl_logfile(logger, log_dir, rotation_size="10 MB"):
    """
    给loguru的日志器添加导出文件的功能，使用jsonl格式

    :param logger: 日志记录器，一般是from loguru import logger的logger
    :param log_dir: 存储日志的目录，因为有多个文件，这里要输入的是所在的目录
    :param rotation_size: 文件多大后分割
    :return:
    """
    from datetime import datetime

    os.makedirs(log_dir, exist_ok=True)  # 自动创建日志目录

    # 日志文件名匹配的正则表达式，格式为 年月日_时分秒.log
    log_filename_pattern = re.compile(r"(\d{8}_\d{6})\.jsonl")

    # 找到最新的日志文件
    def find_latest_log_file(log_dir):
        log_files = []
        for file in os.listdir(log_dir):
            if log_filename_pattern.match(file):
                log_files.append(file)

        if log_files:
            # 根据时间排序，选择最新的日志文件
            log_files.sort(reverse=True)
            return os.path.join(log_dir, log_files[0])
        return None

    # 检查是否有未写满的日志文件
    latest_log_file = find_latest_log_file(log_dir)

    if latest_log_file:
        log_path = latest_log_file
    else:
        # 生成新的日志文件名
        log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jsonl"
        log_path = os.path.join(log_dir, log_filename)

    # 配置 logger，写入日志文件，设置旋转条件，使用 JSON 序列化
    logger.add(log_path, rotation=rotation_size, serialize=True)

    # 输出初始化成功信息
    logger.info(f"日志系统已初始化，日志文件路径：{log_path}")


class XlBaseModel(BaseModel):
    @classmethod
    def init(cls, *args, **kwargs):
        """ 支持位置参数的初始化，按field字段定义的先后顺序赋值 """
        if args:
            # 将位置参数按顺序映射到字段
            field_names = list(cls.model_fields.keys())
            args_dict = dict(zip(field_names, args))
            kwargs.update(args_dict)
        return cls(**kwargs)

    @classmethod
    def parse(cls, data=None):
        """ 支持类似这样的使用方式 params = PopenParams.parse(input_params)，确保有时候默认参数是None也支持转换到model """
        if data is None:
            return cls()
        else:
            return cls.model_validate(data)

    def update_valid(self,
                     data: dict = None,
                     *,
                     exclude_values: Container[Any] = (None,),
                     **kwargs):
        """ 智能更新模型字段

        将 data (通常是 locals()) 和 kwargs 中的数据合并到当前模型中。
        会自动过滤掉 exclude_values 中包含的值（默认过滤 None）。
        会自动忽略模型中不存在的字段（安全注入）。

        Args:
            data: 字典数据源
            exclude_values: 需要被忽略的值的集合。默认为 (None,)。
                            如果你想允许更新 None，可以传入空元组 ()。
            **kwargs: 显式关键字参数 (优先级高于 data)

        Returns:
            self: 支持链式调用
        """
        # 1. 确定数据源优先级：kwargs > data
        if data:
            # 这里的逻辑是：先处理 data，后处理 kwargs (覆盖前者)
            # 为了遍历效率，我们将两者合并逻辑放在循环里
            sources = [data, kwargs]
        else:
            sources = [kwargs]

        # 2. 遍历合并
        for source in sources:
            if not source:
                continue

            # 只遍历模型定义的字段 (安全性：防止 locals() 中的 self 等变量污染)
            for field_name in self.model_fields:
                if field_name not in source:
                    continue

                val = source[field_name]

                # 3. 核心过滤逻辑 (可配置化)
                if val in exclude_values:
                    continue

                # 4. 赋值 (Pydantic 是对象，必须用 setattr)
                setattr(self, field_name, val)

        return self


def resolve_params(*models: Type[BaseModel], mode: Literal['strict', 'extra', 'pass'] = 'strict'):
    """
    一个智能解析与Pydantic模型的参数注入装饰器

    Args:
        *models: 一个或多个 Pydantic 模型类，用于解析参数。
        mode (str, optional):
            - 'strict' (默认): 任何未被模型显式字段认领的关键字参数，都会
              抛出 TypeError。这是最安全的模式。
            - 'extra': 尝试将未认领的参数分配给第一个支持 extra='allow' 的
              模型。如果没有这样的模型，则行为同 'strict'。
            - 'pass': 未认领的参数将原封不动地传递给被装饰的函数。

    详细文档：https://www.yuque.com/xlpr/pyxllib/resolve_params
    
    1. 问题背景：函数参数的“混乱之治”
    在日常的 Python 编程中，我们经常会遇到需要接收大量、可选配置参数的函数。这种函数通常依赖 **kwargs 来接收参数，虽然灵活，但也带来了诸多痛点：
    ● 缺乏类型提示与自动补全：IDE 无法提示 **kwargs 里可以传哪些参数，也无法进行类型检查。
    ● 可读性差：函数签名 def my_func(**kwargs) 像一个“黑洞”，你必须深入阅读其内部实现，才能了解它到底支持哪些配置。
    ● 容易出错：手误拼错参数名（例如 shel=True 而非 shell=True）在运行时不会立即报错，可能会导致难以追踪的 bug。
    ● 参数分组困难：如果一个函数同时需要处理“进程配置”和“网络配置”，这两组参数混杂在 **kwargs 中，会显得非常杂乱。
    我们需要一种既能保持调用灵活性，又能让函数定义清晰、类型安全、易于维护的解决方案。

    2. 解决思路：声明式、可配置的参数解析
    我们的核心思想是：将零散的配置项，聚合到逻辑清晰、自我描述的 Pydantic 模型中，并利用 @resolve_params 装饰器作为“智能适配器”。
    @resolve_params 扮演着“智能参数总管”的角色。它位于用户和你的核心业务逻辑之间，负责：
    1. 拦截 用户传入的各种形式的参数 (*args 和 **kwargs)。
    2. 智能解析 这些参数，并将它们自动填充到你预先定义的 Pydantic 模型中。
    3. 注入 已经实例化、验证通过的模型对象，让你在函数内部可以干净、安全地使用。
    它优雅地将函数的 外部调用接口（灵活、易用） 与 内部实现接口（严谨、类型安全） 分离开来。
    """

    # --- 0. 预计算阶段 (Pre-computation) ---
    if not models:
        return lambda func: func

    # 建立各种快速查找表
    model_class_to_name = {m: m.__name__ for m in models}
    model_name_to_class = {m.__name__: m for m in models}
    registered_model_types = tuple(models)

    field_to_model_map = {}
    for model_cls in models:
        for field_name in model_cls.model_fields:
            field_to_model_map[field_name] = model_cls

    extra_model_cls = next((m for m in models if m.model_config.get('extra') == 'allow'), None)

    def decorator(func):

        # --- 定义内部辅助函数 (Sub-modules) ---
        def _distribute_pos_args(raw_args: tuple) -> Tuple[list, Dict[str, dict]]:
            """步骤1: 处理位置参数 (*args)"""
            pos_args = []
            # 初始化收集器：{'WaitParams': {}, 'LocateParams': {}}
            collected = {name: {} for name in model_name_to_class.keys()}

            for arg in raw_args:
                if isinstance(arg, registered_model_types):
                    # 如果是注册模型的实例，直接拆包认领
                    name = model_class_to_name[type(arg)]
                    collected[name].update(arg.model_dump(exclude_unset=True))
                else:
                    pos_args.append(arg)
            return pos_args, collected

        def _distribute_kwargs(raw_kwargs: dict, collected: Dict[str, dict]) -> dict:
            """步骤2: 处理关键字参数 (**kwargs) - 核心分发逻辑"""
            unassigned = {}

            for key, value in raw_kwargs.items():
                # [A] 显式模型透传 (Convention: Key == ModelName)
                if key in model_name_to_class and isinstance(value, model_name_to_class[key]):
                    collected[key].update(value.model_dump(exclude_unset=True))
                    continue

                # [B] 扁平字段映射 (Field -> Model)
                target_cls = field_to_model_map.get(key)
                if target_cls:
                    model_name = model_class_to_name[target_cls]
                    collected[model_name][key] = value
                else:
                    # [C] 暂存未知参数
                    unassigned[key] = value

            return unassigned

        def _apply_mode_policy(unassigned: dict, collected: Dict[str, dict]) -> dict:
            """步骤3: 根据 mode 处理剩余未知参数"""
            if not unassigned:
                return {}

            # 策略：extra - 尝试塞入支持 extra 的模型
            if mode == 'extra' and extra_model_cls:
                target_name = model_class_to_name[extra_model_cls]
                collected[target_name].update(unassigned)
                return {}  # 已被完全消化

            # 策略：strict - 报错
            if mode == 'strict' or (mode == 'extra' and not extra_model_cls):
                raise TypeError(
                    f"{func.__name__}() got unexpected keyword arguments: {list(unassigned.keys())}"
                )

            # 策略：pass - 透传
            return unassigned  # mode == 'pass'

        def _instantiate_models(collected: Dict[str, dict]) -> dict:
            """步骤4: 将字典数据转换为 Pydantic 实例"""
            instances = {}
            for name, data in collected.items():
                try:
                    # model_validate 负责校验和默认值填充
                    instance = model_name_to_class[name](**data)
                    instances[name] = instance
                except Exception as e:
                    raise ValueError(
                        f"Argument validation failed for model '{name}' in '{func.__name__}': {e}"
                    ) from e
            return instances

        # --- 主装饰器逻辑 ---
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1. 分离：位置参数 vs 模型数据
            func_pos_args, collected_data = _distribute_pos_args(args)

            # 2. 分发：关键字参数 -> 模型字段 / 未知参数
            unassigned_kwargs = _distribute_kwargs(kwargs, collected_data)

            # 3. 决策：处理未知参数 (报错、合并或直通)
            final_passthrough_kwargs = _apply_mode_policy(unassigned_kwargs, collected_data)

            # 4. 构造：生成最终的模型实例
            model_instances = _instantiate_models(collected_data)

            # 合并最终参数 (直通参数 + 模型实例)
            final_kwargs = {**final_passthrough_kwargs, **model_instances}

            return func(*func_pos_args, **final_kwargs)

        return wrapper

    return decorator
