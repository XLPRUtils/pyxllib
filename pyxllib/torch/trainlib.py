#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/10/18 16:06


"""

常见的训练操作的代码封装

"""

from pyxllib.debug import *

import torch
from torch import nn, optim


class TrainingModelBase:
    def __init__(self):
        self.log = get_xllog()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def loss_values_stat(cls, loss_vales):
        """ 一组loss损失的统计分析

        :param loss_vales: 一次info中，多份batch产生的误差数据
        """
        if not loss_vales:
            raise ValueError

        data = np.array(loss_vales, float)
        n, sum_ = len(data), data.sum()
        mean, std = data.mean(), data.std()
        msg = f'total_loss={sum_:.3f}, mean±std={mean:.3f}±{std:.3f}({max(data):.3f}->{min(data):.3f})'
        return msg

    @classmethod
    def sample_size(cls, data):
        """ 单个样本占用的空间大小，返回字节数 """
        x, label = data.dataset[0]  # 取第0个样本作为参考
        return getasizeof(x.numpy()) + getasizeof(label)


class TrainingClassifyModel(TrainingModelBase):
    """ 对pytorch（分类）模型的训练、测试等操作的进一步封装 """

    def __init__(self, model, data_dir=None, batch_size=None,
                 optimizer=None, loss_func=None):

        super().__init__()
        self.log.info(f'initialize. use device={self.device}.')

        self.data_dir = data_dir if data_dir else 'D:/data'
        self.batch_size = batch_size if batch_size else 500
        self.train_data = self.get_train_data()
        self.test_data = self.get_test_data()
        self.train_data_number, self.test_data_number = len(self.train_data.dataset), len(self.test_data.dataset)
        self.log.info(f'get data, train_data_number={self.train_data_number}(batch={len(self.train_data)}), '
                      f'test_data_number={self.test_data_number}(batch={len(self.test_data)}), batch_size={self.batch_size}')

        self.model = model.to(self.device)
        self.optimizer = optimizer if optimizer else optim.SGD(model.parameters(), lr=0.01)
        self.loss_func = loss_func if loss_func else nn.CrossEntropyLoss().to(self.device)

    def get_train_data(self):
        """ 子类必须实现的接口函数 """
        raise NotImplementedError

    def get_test_data(self):
        """ 子类必须实现的接口函数 """
        raise NotImplementedError

    def training_one_epoch(self):
        # 1 检查模式
        tt = TicToc()
        if not self.model.training:
            self.model.train(True)

        # 2 训练一轮
        loss_values = []
        for x, label in self.train_data:
            # 每个batch可能很大，所以每个batch依次放到cuda，而不是一次性全放入
            x, label = x.to(self.device), label.to(self.device)

            logits = self.model(x)
            loss = self.loss_func(logits, label)
            loss_values.append(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 3 训练阶段只看loss，不看实际预测准确度，默认每个epoch都会输出
        elapsed_time = tt.tocvalue()
        return elapsed_time, loss_values

    def test_accuracy(self, data, prefix=''):
        """ 测试验证集等数据上的精度 """
        # 1 eval模式
        if self.model.training:
            self.model.train(False)

        # 2 关闭梯度，可以节省显存和加速
        with torch.no_grad():
            tt = TicToc()

            # 2.1 有时候在训练阶段的batch_size太小，导致预测阶段速度太慢，所以把batch_size改一下
            sample_size = self.sample_size(data)
            # 用1G内存作为参考，每次能加载的样本数量上限；上限控制在2000条以内
            batch_size = min(math.floor(1024 ** 3 / sample_size), 2000)

            # 2.2 预测结果，计算正确率
            loss, correct, number = [], 0, len(data.dataset)
            for x, label in torch.utils.data.DataLoader(data.dataset, batch_size=batch_size):
                x, label = x.to(self.device), label.to(self.device)
                logits = self.model(x)
                loss.append(self.loss_func(logits, label))
                correct += logits.argmax(dim=1).eq(label).sum().item()  # 预测正确的数量
            elapsed_time, mean_loss = tt.tocvalue(), np.mean(loss, dtype=float)
            info = f'{prefix} accuracy={correct}/{number} ({correct / number:.0%})\t' \
                   f'mean_loss={mean_loss:.3f}\telapsed_time={elapsed_time:.0f}s\tbatch_size={batch_size}'
            self.log.info(info)

    def training(self, epochs=20,
                 log_interval=1, test_interval=5,
                 save_interval=None):
        """ 主要训练接口
        :param epochs: 训练代数，输出时从1开始编号
        :param log_interval: 每隔几个epoch输出当前epoch的训练情况，损失值
        :param test_interval: 每隔几个epoch进行一次正确率测试（训练阶段只能看到每轮epoch中多个batch的平均损失）
        :param save_interval: 每隔几个epoch保存一次模型（未实装）
        :return:
        """
        for epoch in range(1, epochs + 1):
            elapsed_time, loss_values = self.training_one_epoch()
            if log_interval and epoch % log_interval == 0:
                msg = self.loss_values_stat(loss_values)
                self.log.info(f'training_epoch={epoch}, elapsed_time={elapsed_time:.0f}s\t{msg}')
            if test_interval and epoch % test_interval == 0:
                self.test_accuracy(self.train_data, 'train_data')
                self.test_accuracy(self.test_data, ' test_data')
