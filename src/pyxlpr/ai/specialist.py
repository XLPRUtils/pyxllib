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
import numpy as np

from pyxllib.prog.pupil import check_install_package, run_once


class ClasEvaluater:
    def __init__(self, gt, pred, *, names=None):
        """
        :param names: 可以将id映射到对应的明文名称
            list，跟id对应名称names[id]
            dict，使用key映射规则
        """
        assert len(gt) == len(pred), f'gt={len(gt)}和pred={len(pred)}数量不匹配'
        self.gt, self.pred = gt, pred
        self.total = len(gt)
        self.names = names

    @classmethod
    def from_pairs(cls, pairs):
        """
        :param list|tuple pairs: 一个列表 [(y1, y_hat1), (y2, y_hat2), ...]
            每个元素还是一个长度2的列表，常见格式是[[0, 0], [1,2], ...]
                第一个值是gt的类别y，第二个值是程序预测的类别y_hat
        """
        gt, pred = list(zip(*pairs))  # 这里内部的实现算法很多都是分解开的
        return cls(gt, pred)

    def n_correct(self):
        """ 类别正确数量 """
        return sum([y == y_hat for y, y_hat in zip(self.gt, self.pred)])

    def accuracy(self):
        """ 整体的正确率精度（等价于f1_score的micro） """
        return round(self.n_correct() / self.total, 4)

    def crosstab(self):
        """ 各类别具体情况交叉表 """
        # TODO 用names转明文？
        df = pd.DataFrame.from_dict({'gt': self.gt, 'pred': self.pred})
        return pd.crosstab(df['gt'], df['pred'])

    def f1_score(self, average='weighted'):
        """ 多分类任务是用F1分值 https://zhuanlan.zhihu.com/p/64315175

        :param average:
            weighted：每一类都算出f1，然后（按样本数）加权平均
            macro：每一类都算出f1，然后求平均值（样本不均衡下，有的类就算只出现1次，也会造成极大的影响）
            micro：按二分类形式直接计算全样本的f1，等价于accuracy
            all：我自己扩展的格式，会返回三种结果的字典值
        """
        check_install_package('sklearn', 'scikit-learn')
        from sklearn.metrics import f1_score

        if average == 'all':
            return {f'f1_{k}': self.f1_score(k) for k in ('weighted', 'macro', 'micro')}
        else:
            return round(f1_score(self.gt, self.pred, average=average), 4)


class ComputingReceptiveFields:
    """ 计算感受野大小的工具
    https://distill.pub/2019/computing-receptive-fields/#return-from-solving-receptive-field-size

    除了这里实现的基础版本的感受野大小计算，论文还能计算具体的区间位置、不规则图形等情况
    """

    @classmethod
    def computing(cls, network):
        """ 基础的计算工具

        network = [['Conv2D', 5, 1],
                   ['MaxPool2D', 2, 2],
                   ['Conv2D', 3, 1],
                   ['MaxPool2D', 2, 2],
                   ['Conv2D', 3, 1],
                   ['MaxPool2D', 2, 2]]
        df = computing(network)
        """

        # 0 基本配置数据表，由外部输入
        columns = ['name', 'kernel_size', 'stride']
        df = pd.DataFrame.from_records(network, columns=columns)

        # 1 感受野
        n = len(df)
        df['receptive'] = 1
        for i in range(n - 2, -1, -1):
            x = df.loc[i + 1]
            df.loc[i, 'receptive'] = x['stride'] * x['receptive'] + (x['kernel_size'] - x['stride'])

        return df

    @classmethod
    def from_paddle_layers(cls, network):
        """ 先对一些基础的常见类型做些功能接口，复杂的情况以后有需要再说吧 """

        import paddle

        ls = []
        for x in network.sublayers():
            if isinstance(x, paddle.nn.layer.conv.Conv2D):
                ls.append(['Conv2D', x._kernel_size[0], x._stride[0]])
            elif isinstance(x, paddle.nn.layer.pooling.MaxPool2D):
                ls.append(['MaxPool2D', x.ksize, x.ksize])
            else:  # 其他先不考虑，跳过
                pass

        return cls.computing(ls)


def show_feature_map(feature_map, show=True, *, pading=5):
    """ 显示特征图 """
    from pyxllib.xlcv import xlcv

    a = np.array(feature_map)
    a = a - a.min()
    m = a.max()
    if m:
        a = (a / m) * 255
    a = a.astype('uint8')

    if a.ndim == 3:
        a = xlcv.concat(list(a), pad=pading)
    elif a.ndim == 4:
        a = xlcv.concat([list(x) for x in a], pad=pading)

    if show:
        xlcv.show(a)

    return a


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
