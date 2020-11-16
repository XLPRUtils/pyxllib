#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/10/18 16:06


"""

常见的训练操作的代码封装

"""
from abc import ABC

from pyxllib.debug import *

import torch
from torch import nn, optim
import torch.utils.data

import torchvision
from torchvision import transforms

# 可视化工具
try:
    import visdom
except ModuleNotFoundError:
    subprocess.run(['pip', 'install', 'visdom'])
    import visdom


class Visdom(visdom.Visdom, metaclass=SingletonForEveryInitArgs):
    """

    visdom文档： https://www.yuque.com/code4101/pytorch/visdom
    """

    def __init__(
            self,
            server='http://localhost',
            endpoint='events',
            port=8097,
            base_url='/',
            ipv6=True,
            http_proxy_host=None,
            http_proxy_port=None,
            env='main',
            send=True,
            raise_exceptions=None,
            use_incoming_socket=True,
            log_to_filename=None):
        self.is_connection = is_url_connect(f'{server}:{port}')

        if self.is_connection:
            super().__init__(server, endpoint, port, base_url, ipv6,
                             http_proxy_host, http_proxy_port, env, send,
                             raise_exceptions, use_incoming_socket, log_to_filename)
        else:
            get_xllog().info('visdom server not support')

        self.plot_windows = set()

    def __bool__(self):
        return self.is_connection

    def one_batch_images(self, imgs, targets, title='one_batch_image', *, nrow=8, padding=2):
        self.images(imgs, nrow=nrow, padding=padding,
                    win=title, opts={'title': title, 'caption': str(targets)})

    def _check_plot_win(self, win, update=None):
        # 记录窗口是否为本次执行程序时第一次初始化，并且据此推导update是首次None，还是复用append
        if update is None:
            if win in self.plot_windows:
                update = 'append'
            else:
                update = None
        self.plot_windows.add(win)
        return update

    def _refine_opts(self, opts=None, *, title=None, legend=None, **kwargs):
        if opts is None:
            opts = {}
        if title and 'title' not in opts: opts['title'] = title
        if legend and 'legend' not in opts: opts['legend'] = legend
        for k, v in kwargs.items():
            if k not in opts:
                opts[k] = v
        return opts

    def loss_line(self, loss_values, epoch, win='loss', *, title=None, update=None):
        """ 损失函数曲线

        横坐标是epoch
        """
        # 1 记录窗口是否为本次执行程序时第一次初始化
        if title is None: title = win
        update = self._check_plot_win(win, update)

        # 2 画线
        xs = np.linspace(epoch - 1, epoch, num=len(loss_values) + 1)
        self.line(loss_values, xs[1:], win=win, opts={'title': title, 'xlabel': 'epoch'},
                  update=update)

    def plot_line(self, y, x, win, *, opts=None,
                  title=None, legend=None, update=None):
        # 1 记录窗口是否为本次执行程序时第一次初始化
        if title is None: title = win
        update = self._check_plot_win(win, update)

        # 2 画线
        self.line(y, x, win=win, update=update,
                  opts=self._refine_opts(opts, title=title, legend=legend, xlabel='epoch'))


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, labelfile, label_transform):
        """ 超轻量级的Dataset类，一般由外部ProjectData类指定每行label的转换规则 """
        self.labels = labelfile.read().splitlines()
        self.label_transform = label_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.label_transform(self.labels[idx])


class TrainerBase:
    def __init__(self, model, datasets, *,
                 save_dir=None,
                 batch_size=None,
                 optimizer=None, loss_func=None):
        self.log = get_xllog()
        self.device = get_device()
        self.save_dir = Path(save_dir) if save_dir else Path()  # 没有指定数据路径则以当前工作目录为准
        self.model = model
        self.datasets = datasets

    @classmethod
    def loss_values_stat(cls, loss_vales):
        """ 一组loss损失的统计分析

        :param loss_vales: 一次info中，多份batch产生的误差数据
        """
        if not loss_vales:
            raise ValueError

        data = np.array(loss_vales, dtype=float)
        n, sum_ = len(data), data.sum()
        mean, std = data.mean(), data.std()
        msg = f'total_loss={sum_:.3f}, mean±std={mean:.3f}±{std:.3f}({max(data):.3f}->{min(data):.3f})'
        return msg

    @classmethod
    def sample_size(cls, data):
        """ 单个样本占用的空间大小，返回字节数 """
        x, label = data.dataset[0]  # 取第0个样本作为参考
        return getasizeof(x.numpy()) + getasizeof(label)

    def save_model_state(self, file):
        """ 保存模型参数值
        一般存储model.state_dict，而不是直接存储model，确保灵活性

        # TODO 和path结合，增加if_exists参数
        """
        p = Path(file, root=self.save_dir)
        p.ensure_dir(pathtype='file')
        torch.save(self.model.state_dict(), str(p))

    def load_model_state(self, file):
        """ 读取模型参数值 """
        p = Path(file, root=self.save_dir)
        self.model.load_state_dict(torch.load(str(p), map_location=self.device))

    def get_train_data(self):
        train_loader = torch.utils.data.DataLoader(
            ImageDirectionDataset(self.data_dir, mode='train'),
            batch_size=self.batch_size, shuffle=True, num_workers=8)
        return train_loader

    def get_val_data(self):
        val_loader = torch.utils.data.DataLoader(
            self.datasets(self.data_dir, mode='val'),
            batch_size=self.batch_size, shuffle=True, num_workers=8)
        return val_loader


class ClassificationTrainer(TrainerBase):
    """ 对pytorch（分类）模型的训练、测试等操作的进一步封装

    # TODO log变成可选项，可以关掉
    """

    def __init__(self, model, *, data_dir=None, save_dir=None,
                 batch_size=None, optimizer=None, loss_func=None):

        super().__init__(save_dir=save_dir)
        self.log.info(f'initialize. use_device={self.device}.')

        self.model = model.to(self.device)
        self.optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=0.01)
        self.loss_func = loss_func if loss_func else nn.CrossEntropyLoss().to(self.device)
        self.log.info('model parameters size: ' + str(sum(map(lambda p: p.numel(), self.model.parameters()))))

        self.data_dir = Path(data_dir) if data_dir else Path()  # 没有指定数据路径则以当前工作目录为准
        self.log.info(f'data_dir={self.data_dir}, save_dir={self.save_dir}')

        self.batch_size = batch_size if batch_size else 500
        self.train_data = self.get_train_data()
        self.val_data = self.get_val_data()
        self.train_data_number, self.test_data_number = len(self.train_data.dataset), len(self.val_data.dataset)
        self.log.info(f'get data, train_data_number={self.train_data_number}(batch={len(self.train_data)}), '
                      f'test_data_number={self.test_data_number}(batch={len(self.val_data)}), batch_size={self.batch_size}')

    def viz_data(self):
        """ 用visdom显示样本数据

        TODO 增加一些自定义格式参数
        TODO 不能使用\n、\r\n、<br/>实现文本换行，有时间可以研究下，结合nrow、图片宽度，自动推算，怎么美化展示效果
        """
        viz = Visdom()
        if not viz: return

        x, label = next(iter(self.train_data))
        viz.one_batch_images(x, label, 'train data')

        x, label = next(iter(self.val_data))
        viz.one_batch_images(x, label, 'val data')

    def training_one_epoch(self):
        # 1 检查模式
        if not self.model.training:
            self.model.train(True)

        # 2 训练一轮
        loss_values = []
        for x, label in self.train_data:
            # 每个batch可能很大，所以每个batch依次放到cuda，而不是一次性全放入
            x, label = x.to(self.device), label.to(self.device)

            logits = self.model(x)
            if isinstance(logits, tuple):
                logits = logits[0]  # 如果返回是多个值，一般是RNN等层有其他信息，先只取第一个参数值就行了
            loss = self.loss_func(logits, label)
            loss_values.append(float(loss))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 3 训练阶段只看loss，不看实际预测准确度，默认每个epoch都会输出
        return loss_values

    def calculate_accuracy(self, data, prefix=''):
        """ 测试验证集等数据上的精度 """
        # 1 eval模式
        if self.model.training:
            self.model.train(False)

        # 2 关闭梯度，可以节省显存和加速
        with torch.no_grad():
            tt = TicToc()

            # 预测结果，计算正确率
            loss, correct, number = [], 0, len(data.dataset)
            for x, label in data:
                x, label = x.to(self.device), label.to(self.device)
                logits = self.model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss.append(self.loss_func(logits, label))
                correct += logits.argmax(dim=1).eq(label).sum().item()  # 预测正确的数量
            elapsed_time, mean_loss = tt.tocvalue(), np.mean(loss, dtype=float)
            accuracy = correct / number
            info = f'{prefix} accuracy={correct}/{number} ({accuracy:.2%})\t' \
                   f'mean_loss={mean_loss:.3f}\telapsed_time={elapsed_time:.0f}s'
            self.log.info(info)
            return accuracy

    def training(self, epochs=20, *, start_epoch=0,
                 log_interval=1,
                 test_interval=0, save_interval=0):
        """ 主要训练接口

        :param epochs: 训练代数，输出时从1开始编号
        :param start_epoch: 直接从现有的第几个epoch的模型读取参数
            使用该参数，需要在self.save_dir有对应名称的model文件
        :param log_interval: 每隔几个epoch输出当前epoch的训练情况，损失值
        :param test_interval: 每隔几个epoch进行一次正确率测试（训练阶段只能看到每轮epoch中多个batch的平均损失）
        :param save_interval: 每隔几个epoch保存一次模型
        :return:
        """
        # 1 参数
        tag = self.model.__class__.__name__
        epoch_time_tag = f'elapsed_time' if log_interval == 1 else f'{log_interval}*epoch_time'
        viz = Visdom()
        if test_interval == 0 and save_interval: test_interval = save_interval

        # 2 加载之前的模型继续训练
        if start_epoch:
            self.load_model_state(f'{tag} epoch={start_epoch}.pth')

        # 3 训练
        tt = TicToc()
        for epoch in range(start_epoch + 1, epochs + 1):
            loss_values = self.training_one_epoch()
            if viz: viz.loss_line(loss_values, epoch, 'train_loss')
            if log_interval and epoch % log_interval == 0:
                msg = self.loss_values_stat(loss_values)
                elapsed_time = tt.tocvalue(restart=True)
                self.log.info(f'epoch={epoch}, {epoch_time_tag}={elapsed_time:.0f}s\t{msg}')
            if test_interval and epoch % test_interval == 0:
                accuracy1 = self.calculate_accuracy(self.train_data, 'train_data')
                accuracy2 = self.calculate_accuracy(self.val_data, '  val_data')
                if viz: viz.plot_line([[accuracy1, accuracy2]], [epoch], 'accuracy', legend=['train', 'val'])
            if save_interval and epoch % save_interval == 0:
                self.save_model_state(f'{tag} epoch={epoch}.pth')


def get_classification_func(model, state_file, func):
    """ 工厂函数，生成一个分类器函数

    用这个函数做过渡的一个重要目的，也是避免重复加载模型

    :param model: 模型结构
    :param state_file: 存储参数的文件
    :param func: 模型结果的处理器，默认
    :return: 返回的函数结构见下述cls_func
    """
    model.load_state_dict(torch.load(str(state_file), map_location=get_device()))

    def cls_func(x):
        """
        :param x: 输入可以是路径、np.ndarray、PIL图片等，都为转为batch结构的tensor
        :return: 输入如果只有一张图片，则返回一个结果
            否则会存在list，返回一个batch的多个结果
        """
        pass

    return cls_func
