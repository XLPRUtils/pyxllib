# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import numpy as np
import paddle
import signal
import random

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import copy
from paddle.io import Dataset, DataLoader, BatchSampler, DistributedBatchSampler
import paddle.distributed as dist

from pyxlpr.ppocr.data.imaug import transform, create_operators
# 可以在ppocr/data目录下新增脚本，添加自己的数据格式
# 个人想法，到时候新增的数据格式类，都统一放到一个文件xl_dataset里，方便整理和分享
from pyxlpr.ppocr.data.simple_dataset import *  # 这里扩展了一些自己的基础数据格式
from pyxlpr.ppocr.data.lmdb_dataset import LMDBDataSet
from pyxlpr.ppocr.data.pgnet_dataset import PGDataSet
from pyxlpr.ppocr.data.pubtab_dataset import PubTabDataSet

# 或者不在ppocr.data这里加也可以，重点是能导入特定接口范式的类，让这里eval能取到即可。

__all__ = ['build_dataloader', 'transform', 'create_operators']


def term_mp(sig_num, frame):
    """ kill all child processes
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)


def build_dataloader(config, mode, device, logger, seed=None):
    config = copy.deepcopy(config)

    support_dict = [
        'SimpleDataSet', 'LMDBDataSet', 'PGDataSet', 'PubTabDataSet', 'XlSimpleDataSet'
    ]
    module_name = config[mode]['dataset']['name']
    assert module_name in support_dict, Exception(
        'DataSet only support {}'.format(support_dict))
    # 这里eval没有安全隐患，因为会提前判断module_name要属于support_dict中的值
    assert mode in ['Train', 'Eval', 'Test'
                    ], "Mode should be Train, Eval or Test."

    # 自定义数据格式类，除了初始化，似乎没看到还有其他框架性的约束
    dataset = eval(module_name)(config, mode, logger, seed)
    loader_config = config[mode]['loader']
    # loader的必填参数
    batch_size = loader_config['batch_size_per_card']
    drop_last = loader_config['drop_last']
    shuffle = loader_config['shuffle']
    num_workers = loader_config['num_workers']
    # 可选参数
    # 注意这里实现机制策略与d2的不同，d2习惯是把这个参数加入到初始默认配置字典中，而ppocr是在函数运算中智能判断。
    if 'use_shared_memory' in loader_config.keys():
        use_shared_memory = loader_config['use_shared_memory']
    else:
        # use_shared_memory (bool) - 是否使用共享内存来提升子进程将数据放入进程间队列的速度，
        # 该参数尽在多进程模式下有效(即 num_workers > 0 )，
        # 请确认机器上有足够的共享内存空间(如Linux系统下 /dev/shm/ 目录空间大小)再设置此参数。
        # 默认为True。
        use_shared_memory = True
    if mode == "Train":
        # Train会用多卡机制分配BatchSampler，当然，如果外部设置了单卡也可以，单卡是特殊的多卡机制。
        # Distribute data to multiple cards
        batch_sampler = DistributedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)
    else:
        # 非Train阶段，强制使用单卡处理。
        # Distribute data to single card
        batch_sampler = BatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        places=device,
        num_workers=num_workers,
        return_list=True,
        use_shared_memory=use_shared_memory)

    # 看到用了signal库，好奇~~ https://www.jianshu.com/p/e0a69beb98bb
    # support exit using ctrl+c
    signal.signal(signal.SIGINT, term_mp)
    signal.signal(signal.SIGTERM, term_mp)

    return data_loader


def build_dataset(config, mode, logger, seed=None):
    """ ckz: 有时候不需要获得loader，只要dataset即可 """
    config = copy.deepcopy(config)

    support_dict = [
        'SimpleDataSet', 'LMDBDataSet', 'PGDataSet', 'PubTabDataSet', 'XlSimpleDataSet'
    ]
    module_name = config[mode]['dataset']['name']
    assert module_name in support_dict, Exception(
        'DataSet only support {}'.format(support_dict))
    assert mode in ['Train', 'Eval', 'Test'
                    ], "Mode should be Train, Eval or Test."

    dataset = eval(module_name)(config, mode, logger, seed)

    return dataset
