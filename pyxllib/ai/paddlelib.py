#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/11/05 09:01

"""
pp是paddlepaddle的缩写
"""

import os
import logging

from tqdm import tqdm

import paddle

from pyxllib.xl import XlPath
from pyxllib.ai.specialist import ClasEvaluater


class SequenceDataset(paddle.io.Dataset):
    def __init__(self, samples, transform=None):
        super().__init__()
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = self.samples[index]
        if self.transform:
            x = self.transform(x)
        return x


def build_testdata_loader(samples, transform=None, *args, **kwargs):
    """ 简化的一个创建paddle的DataLoader的函数。主要用于简化部署阶段的推理。

    :param samples: list类型的输入格式
    """
    import paddle.fluid.dataloader.fetcher
    # 暂时不知道怎么关闭这个警告，先用暴力方法
    paddle.fluid.dataloader.fetcher._WARNING_TO_LOG = False

    if isinstance(samples, paddle.io.DataLoader):
        return samples
    elif isinstance(samples, paddle.io.Dataset):
        dataset = samples
    else:
        dataset = SequenceDataset(samples, transform)

    return paddle.io.DataLoader(dataset, *args, **kwargs)


class ImageClasPredictor:
    """ 图像分类框架的预测器 """

    def __init__(self, model, params_file, *, transform=None, classes=None):
        self.model = model
        self.model.load_dict(paddle.load(params_file))
        self.model.eval()

        self.transform = transform
        # 如果输入该字段，会把下标id自动转为明文类名
        self.classes = classes

    def pred_batch(self, samples, verbose=0, batch_size=None):
        """ 默认是进行批量识别，如果只识别单个，可以用pred

        :param samples: 要识别的数据，支持类list的列表，或Dataset、DataLoader
        :param verbose: 调试级别，0表示直接预测类别，1会显示进度条，2则是返回每个预测在各个类别的概率
        :param batch_size: 默认按把imgs整个作为一个批次前传，如果数据量很大，可以使用该参数切分batch
        :return:
        """
        import paddle.nn.functional as F

        if not batch_size: batch_size = len(samples)
        data_loader = build_testdata_loader(samples, self.transform, batch_size=batch_size)

        logits = []
        for inputs in tqdm(data_loader, desc='预测：', disable=verbose < 1):
            logits.append(self.model(inputs))
        logits = paddle.concat(logits, axis=0)

        if verbose < 2:
            idx = logits.argmax(1)
            if self.classes:
                idx = [self.classes[x] for x in idx]
            return idx
        elif verbose == 2:
            prob = F.softmax(logits, axis=1).tolist()
            for i, item in enumerate(prob):
                prob[i] = [round(x, 4) for x in item]  # 保留4位小数就够了
            return prob
        else:
            raise ValueError

    def __call__(self, *args, **kwargs):
        return self.pred_batch(*args, **kwargs)

    def pred(self, img, *args, **kwargs):
        return self.pred_batch([img], *args, **kwargs)[0]


class ClasAccuracy(paddle.metric.Metric):
    """ 分类问题的精度 """

    def __init__(self, num_classes=None, *, verbose=0):
        """
        Args:
            num_classes: 其实这个参数不输也没事~~
            verbose:
                0，静默
                1，reset的时候，输出f1指标
                2，reset的时候，还会输出crosstab
        """
        super(ClasAccuracy, self).__init__()
        self.num_classes = num_classes
        self.total = 0
        self.count = 0
        self.gt = []
        self.pred = []
        self.verbose = verbose

    def name(self):
        return 'acc'

    def update(self, x, y):
        x = x.argmax(axis=1)
        y = y.reshape(-1)
        cmp = (x == y)
        self.count += cmp.sum()
        self.total += len(cmp)
        self.gt += y.tolist()
        self.pred += x.tolist()

    def accumulate(self):
        return self.count / self.total

    def reset(self):
        if self.verbose:
            a = ClasEvaluater(self.gt, self.pred)
            print(a.f1_score('all'))
            if self.verbose > 1:
                print(a.crosstab())
        self.count = 0
        self.total = 0
        self.gt = []
        self.pred = []


class VisualAcc(paddle.callbacks.Callback):
    def __init__(self, logdir, experimental_name, *, save_model_with_input=None):
        """
        :param logdir:
        :param experimental_name:
        :param save_model_with_input: 默认不存储模型结构，当开启该参数时，
        """
        from pyxllib.prog.pupil import check_install_package
        check_install_package('visualdl')
        from visualdl import LogWriter

        super().__init__()
        # 这样奇怪地加后缀，是为了字典序后，每个实验的train显示在eval之前
        d = XlPath(logdir) / (experimental_name + '_train')
        # if d.exists(): shutil.rmtree(d)
        self.write = LogWriter(logdir=str(d))
        d = XlPath(logdir) / (experimental_name + '_val')
        # if d.exists(): shutil.rmtree(d)
        self.eval_writer = LogWriter(logdir=str(d))
        self.eval_times = 0

        self.save_model_with_input = save_model_with_input

    def on_epoch_end(self, epoch, logs=None):
        self.write.add_scalar('acc', step=epoch, value=logs['acc'])
        self.write.flush()

    def on_eval_end(self, logs=None):
        self.eval_writer.add_scalar('acc', step=self.eval_times, value=logs['acc'])
        self.eval_writer.flush()
        self.eval_times += 1
