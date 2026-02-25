#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/11/07 14:04

import os
import re
import time

from humanfriendly import format_size
import pandas as pd

from pyxllib.prog.sys import check_install_package
from pyxllib.prog.run import run_once


@run_once
def _nvml_init():
    check_install_package('pynvml')

    import pynvml
    pynvml.nvmlInit()


class NvmDevice:
    """
    TODO 增加获得多张卡的接口
    """

    def __init__(self, *, set_cuda_visible=True):
        """ 获得各个gpu内存使用信息

        :param set_cuda_visible: 是否根据 环境变量 CUDA_VISIBLE_DEVICES 重新计算gpu的相对编号
        """
        _nvml_init()

        import pynvml

        records = []
        columns = ['origin_id',  # 原始id编号
                   'total',  # 总内存
                   'used',  # 已使用
                   'free']  # 剩余空间

        try:
            # 2 每张gpu卡的绝对、相对编号
            if set_cuda_visible and 'CUDA_VISIBLE_DEVICES' in os.environ:
                idxs = re.findall(r'\d+', os.environ['CUDA_VISIBLE_DEVICES'])
                idxs = [int(v) for v in idxs]
            else:
                cuda_num = pynvml.nvmlDeviceGetCount()
                idxs = list(range(cuda_num))  # 如果不限定，则获得所有卡的信息

            # 3 获取每张候选gpu卡的内存使用情况
            for i in idxs:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                records.append([i, meminfo.total, meminfo.used, meminfo.free])
        except (FileNotFoundError, pynvml.nvml.NVMLError_LibraryNotFound) as e:
            # 注意，找不到nvml.dll文件，不代表没有gpu卡~
            pass

        self.stat = pd.DataFrame.from_records(records, columns=columns)

    def get_most_free_gpu_id(self, minimum_free_byte=-1, *, reverse=False):
        """ 获得当前剩余空间最大的gpu的id，没有则返回None

        :param minimum_free_byte: 最少需要剩余的空闲字节数，少于这个值则找不到gpu，返回None
        """
        gpu_id, most_free = None, minimum_free_byte
        for idx, row in self.stat.iterrows():
            if row['free'] > most_free:
                gpu_id, most_free = idx, row['free']
            # 个人习惯，从后往前找空闲最大的gpu，这样也能尽量避免使用到0卡
            #   所以≥就更新，而不是>才更新
            if reverse and row['free'] >= most_free:
                gpu_id, most_free = idx, row['free']
        return gpu_id

    def get_free_gpu_ids(self, minimum_free_byte=10 * 1024 ** 3):
        """ 获取多个有剩余空间的gpu id

        :param minimum_free_byte: 默认值至少需要10G
        """
        gpu_ids = []
        for idx, row in self.stat.iterrows():
            if row['free'] >= minimum_free_byte:
                gpu_ids.append(idx)
        return gpu_ids


def auto_set_visible_device(reverse=False):
    """ 自动设置环境变量为可见的某一张单卡

    CUDA_VISIBLE_DEVICES

    对于没有gpu的机子，可以自动设置CUDA_VISIBLE_DEVICES=''空值
    """
    name = 'CUDA_VISIBLE_DEVICES'
    if name not in os.environ:
        gpu_id = NvmDevice().get_most_free_gpu_id(reverse=reverse)
        if gpu_id is not None:
            os.environ[name] = str(gpu_id)
    if name in os.environ:
        print('{}={}'.format(name, os.environ[name]))


def get_current_gpu_useage(card_id=None):
    """ 查当前gpu最大使用率

    :param card_id:
        int, 查单卡
        str, 逗号隔开的多张卡
        list|tuple, 查多张卡
        None, 查所有卡
    """
    _nvml_init()

    import pynvml

    # 1 要检查哪些卡
    if isinstance(card_id, int):
        idxs = [card_id]
    elif isinstance(card_id, str):
        idxs = card_id.split(',')
    elif isinstance(card_id, (list, tuple)):
        idxs = card_id
    else:
        cuda_num = pynvml.nvmlDeviceGetCount()
        idxs = list(range(cuda_num))  # 如果不限定，则获得所有卡的信息

    # 2 获取每张候选gpu卡的显存使用情况
    cur_max_memory = 0
    for i in idxs:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        cur_max_memory = max(cur_max_memory, meminfo.used)
    return cur_max_memory


def watch_gpu_maximun(card_id=None, interval_seconds=0.1):
    """ 查gpu使用峰值

    :param interval_seconds: 查询间隔
    """
    max_memory = 0
    while True:
        cur = get_current_gpu_useage(card_id)
        max_memory = max(max_memory, cur)
        print(f'\rmax_memory={format_size(max_memory, binary=True)} '
              f'current_memory={format_size(cur, binary=True)}', end='')
        time.sleep(interval_seconds)


if __name__ == '__main__':
    import fire

    fire.Fire()
