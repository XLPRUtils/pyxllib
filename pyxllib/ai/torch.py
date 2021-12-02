#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 23:10


from pyxllib.xlcv import *

import torch
from torch import nn, optim
import torch.utils.data

import torchvision
from torchvision import transforms
from easydict import EasyDict

# 把pytorch等常用的导入写了
import torch.utils.data
from torchvision.datasets import VisionDataset

from pyxllib.ai.specialist import ClasEvaluater

__base = """
"""


class NvmDevice:
    """
    TODO 增加获得多张卡的接口
    """

    def __init__(self, *, set_cuda_visible=True):
        """ 获得各个gpu内存使用信息

        :param set_cuda_visible: 是否根据 环境变量 CUDA_VISIBLE_DEVICES 重新计算gpu的相对编号
        """
        import pynvml

        records = []
        columns = ['origin_id',  # 原始id编号
                   'total',  # 总内存
                   'used',  # 已使用
                   'free']  # 剩余空间

        try:
            # 1 初始化
            pynvml.nvmlInit()

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

    def get_most_free_gpu_id(self, minimum_free_byte=-1):
        """ 获得当前剩余空间最大的gpu的id，没有则返回None

        :param minimum_free_byte: 最少需要剩余的空闲字节数，少于这个值则找不到gpu，返回None
        """
        gpu_id, most_free = None, minimum_free_byte
        for idx, row in self.stat.iterrows():
            # 个人习惯，从后往前找空闲最大的gpu，这样也能尽量避免使用到0卡
            #   所以≥就更新，而不是>才更新
            if row['free'] >= most_free:
                gpu_id, most_free = idx, row['free']
        return gpu_id

    def get_most_free_gpu_device(self):
        gpu_id = self.get_most_free_gpu_id()
        if gpu_id is not None:
            return torch.device(f'cuda:{gpu_id}')

    def get_free_gpu_ids(self, minimum_free_byte=10 * 1024 ** 3):
        """ 获取多个有剩余空间的gpu id

        :param minimum_free_byte: 默认值至少需要10G
        """
        gpu_ids = []
        for idx, row in self.stat.iterrows():
            if row['free'] >= minimum_free_byte:
                gpu_ids.append(idx)
        return gpu_ids


def get_device():
    """ 自动获得一个可用的设备
    """
    return NvmDevice().get_most_free_gpu_device() or torch.device('cpu')


__data = """
"""


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, labelfile, label_transform, maxn=None):
        """ 超轻量级的Dataset类，一般由外部ProjectData类指定每行label的转换规则 """
        self.labels = File(labelfile).read().splitlines()
        self.label_transform = label_transform

        self.number = len(self.labels)
        if maxn: self.number = min(self.number, maxn)

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        return self.label_transform(self.labels[idx])


class InputDataset(torch.utils.data.Dataset):
    def __init__(self, raw_in, transform=None, *, y_placeholder=...):
        """ 将非list、tuple数据转为list，并生成一个dataset类的万用类
        :param raw_in:
        """
        if not isinstance(raw_in, (list, tuple)):
            raw_in = [raw_in]

        self.data = raw_in
        self.transform = transform
        self.y_placeholder = y_placeholder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)

        if self.y_placeholder is not ...:
            return x, self.y_placeholder
        else:
            return x


__model = """
"""


class LeNet5(nn.Module):
    """ https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320 """

    def __init__(self, n_classes):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, n_classes),
        )

    def forward(self, batched_inputs):
        device = next(self.parameters()).device

        x = batched_inputs[0].to(device)
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        if self.training:
            y = batched_inputs[1].to(device)
            return nn.functional.cross_entropy(logits, y)
        else:
            return logits.argmax(dim=1)


__train = """
"""


class Trainer:
    """ 对pytorch模型的训练、测试等操作的进一步封装

    # TODO log变成可选项，可以关掉
    """

    def __init__(self, log_dir, device, data, model, optimizer,
                 loss_func=None, pred_func=None, accuracy_func=None):
        # 0 初始化成员变量
        self.log_dir, self.device = log_dir, device
        self.data, self.model, self.optimizer = data, model, optimizer
        if loss_func: self.loss_func = loss_func
        if pred_func: self.pred_func = pred_func
        if accuracy_func: self.accuracy_func = accuracy_func

        # 1 日志
        timetag = datetime.datetime.now().strftime('%Y%m%d.%H%M%S')
        # self.curlog_dir = Dir(self.log_dir / timetag)  # 本轮运行，实际log位置，是存放在一个子目录里
        self.curlog_dir = Dir(self.log_dir)
        self.curlog_dir.ensure_dir()
        self.log = get_xllog(log_file=self.curlog_dir / 'log.txt')
        self.log.info(f'1/4 log_dir={self.curlog_dir}')

        # 2 设备
        self.log.info(f'2/4 use_device={self.device}')

        # 3 数据
        self.train_dataloader = self.data.get_train_dataloader()
        self.val_dataloader = self.data.get_val_dataloader()
        self.train_data_number = len(self.train_dataloader.dataset)
        self.val_data_number = len(self.val_dataloader.dataset)
        self.log.info(f'3/4 get data, train_data_number={self.train_data_number}(batch={len(self.train_dataloader)}), '
                      f'val_data_number={self.val_data_number}(batch={len(self.val_dataloader)}), '
                      f'batch_size={self.data.batch_size}')

        # 4 模型
        parasize = sum(map(lambda p: p.numel(), self.model.parameters()))
        self.log.info(
            f'4/4 model parameters size: {parasize}* 4 Bytes per float ≈ {humanfriendly.format_size(parasize * 4)}')

        # 5 其他辅助变量
        self.min_total_loss = math.inf  # 目前epoch中总损失最小的值（训练损失，训练过程损失）
        self.min_train_loss = math.inf  # 训练集损失
        self.max_val_accuracy = 0  # 验证集精度

    @classmethod
    def loss_func(cls, model_out, y):
        """ 自定义损失函数 """
        # return loss
        raise NotImplementedError

    @classmethod
    def pred_func(cls, model_out):
        """ 自定义模型输出到预测结果 """
        # return y_hat
        raise NotImplementedError

    @classmethod
    def accuracy_func(cls, y_hat, y):
        """ 自定义预测结果和实际标签y之间的精度

        返回"正确的样本数"（在非分类任务中，需要抽象出这个数量关系）
        """
        # return accuracy
        raise NotImplementedError

    def loss_values_stat(self, loss_vales):
        """ 一组loss损失的统计分析

        :param loss_vales: 一次batch中，多份样本产生的误差数据
        :return: 统计信息文本字符串
        """
        if not loss_vales:
            raise ValueError

        data = np.array(loss_vales, dtype=float)
        n, sum_ = len(data), data.sum()
        mean, std = data.mean(), data.std()
        msg = f'total_loss={sum_:.3f}, mean±std={mean:.3f}±{std:.3f}({max(data):.3f}->{min(data):.3f})'
        if sum_ < self.min_total_loss:
            self.min_total_loss = sum_
            msg = '*' + msg
        return msg

    @classmethod
    def sample_size(cls, data):
        """ 单个样本占用的空间大小，返回字节数 """
        x, label = data.dataset[0]  # 取第0个样本作为参考
        return getasizeof(x.numpy()) + getasizeof(label)

    def save_model_state(self, file, if_exists='error'):
        """ 保存模型参数值
        一般存储model.state_dict，而不是直接存储model，确保灵活性

        # TODO 和path结合，增加if_exists参数
        """
        f = File(file, self.curlog_dir)
        if f.exist_preprcs(if_exists=if_exists):
            f.ensure_parent()
            torch.save(self.model.state_dict(), str(f))

    def load_model_state(self, file):
        """ 读取模型参数值

        注意load和save的root差异！ load的默认父目录是在log_dir，而save默认是在curlog_dir！
        """
        f = File(file, self.log_dir)
        self.model.load_state_dict(torch.load(str(f), map_location=self.device))

    def viz_data(self):
        """ 用visdom显示样本数据

        TODO 增加一些自定义格式参数
        TODO 不能使用\n、\r\n、<br/>实现文本换行，有时间可以研究下，结合nrow、图片宽度，自动推算，怎么美化展示效果
        """
        from visdom import Visdom

        viz = Visdom()
        if not viz: return

        x, label = next(iter(self.train_dataloader))
        viz.one_batch_images(x, label, 'train data')

        x, label = next(iter(self.val_dataloader))
        viz.one_batch_images(x, label, 'val data')

    def training_one_epoch(self):
        # 1 检查模式
        if not self.model.training:
            self.model.train(True)

        # 2 训练一轮
        loss_values = []
        for x, y in self.train_dataloader:
            # 每个batch可能很大，所以每个batch依次放到cuda，而不是一次性全放入
            x, y = x.to(self.device), y.to(self.device)

            y_hat = self.model(x)
            loss = self.loss_func(y_hat, y)
            loss_values.append(float(loss))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 3 训练阶段只看loss，不看实际预测准确度，默认每个epoch都会输出
        return loss_values

    def calculate_accuracy(self, dataloader):
        """ 测试验证集等数据上的精度 """
        # 1 eval模式
        if self.model.training:
            self.model.train(False)

        # 2 关闭梯度，可以节省显存和加速
        with torch.no_grad():
            tt = TicToc()

            # 预测结果，计算正确率
            loss, correct, number = [], 0, len(dataloader.dataset)
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                model_out = self.model(x)
                loss.append(self.loss_func(model_out, y))
                y_hat = self.pred_func(model_out)
                correct += self.accuracy_func(y_hat, y)  # 预测正确的数量
            elapsed_time, mean_loss = tt.tocvalue(), np.mean(loss, dtype=float)
            accuracy = correct / number
            info = f'accuracy={correct:.0f}/{number} ({accuracy:.2%})\t' \
                   f'mean_loss={mean_loss:.3f}\telapsed_time={elapsed_time:.0f}s'
            return accuracy, mean_loss, info

    def train_accuracy(self):
        accuracy, mean_loss, info = self.calculate_accuracy(self.train_dataloader)
        info = 'train ' + info
        if mean_loss < self.min_train_loss:
            # 如果是best ever，则log换成debug模式输出
            self.log.debug('*' + info)
            self.min_train_loss = mean_loss
        else:
            self.log.info(info)
        return accuracy

    def val_accuracy(self, save_model=None):
        """
        :param save_model: 如果验证集精度best ever，则保存当前epoch模型
            如果精度不是最好的，哪怕指定save_model也不会保存的
        :return:
        """
        accuracy, mean_loss, info = self.calculate_accuracy(self.val_dataloader)
        info = '  val ' + info
        if accuracy > self.max_val_accuracy:
            self.log.debug('*' + info)
            if save_model:
                self.save_model_state(save_model, if_exists='replace')
            self.max_val_accuracy = accuracy
        else:
            self.log.info(info)
        return accuracy

    def training(self, epochs, *, start_epoch=0, log_interval=1):
        """ 主要训练接口

        :param epochs: 训练代数，输出时从1开始编号
        :param start_epoch: 直接从现有的第几个epoch的模型读取参数
            使用该参数，需要在self.save_dir有对应名称的model文件
        :param log_interval: 每隔几个epoch输出当前epoch的训练情况，损失值
            每个几轮epoch进行一次监控
            且如果总损失是训练以来最好的结果，则会保存模型
                并对训练集、测试集进行精度测试
        TODO 看到其他框架，包括智财的框架，对保存的模型文件，都有更规范的一套命名方案，有空要去学一下
        :return:
        """
        from visdom import Visdom

        # 1 配置参数
        tag = self.model.__class__.__name__
        epoch_time_tag = f'elapsed_time' if log_interval == 1 else f'{log_interval}*epoch_time'
        viz = Visdom()  # 其实这里不是用原生的Visdom，而是我封装过的，但是我封装的那个也没太大作用意义，删掉了

        # 2 加载之前的模型继续训练
        if start_epoch:
            self.load_model_state(f'{tag} epoch{start_epoch}.pth')

        # 3 训练
        tt = TicToc()
        for epoch in range(start_epoch + 1, epochs + 1):
            loss_values = self.training_one_epoch()
            # 3.1 训练损失可视化
            if viz: viz.loss_line(loss_values, epoch, 'train_loss')
            # 3.2 显示epoch训练效果
            if log_interval and epoch % log_interval == 0:
                # 3.2.1 显示训练用时、训练损失
                msg = self.loss_values_stat(loss_values)
                elapsed_time = tt.tocvalue(restart=True)
                info = f'epoch={epoch}, {epoch_time_tag}={elapsed_time:.0f}s\t{msg.lstrip("*")}'
                # 3.2.2 截止目前训练损失最小的结果
                if msg[0] == '*':
                    self.log.debug('*' + info)
                    # 3.2.2.1 测试训练集、验证集上的精度
                    accuracy1 = self.train_accuracy()
                    accuracy2 = self.val_accuracy(save_model=f'{tag} epoch{epoch}.pth')
                    # 3.2.2.2 可视化图表
                    if viz: viz.plot_line([[accuracy1, accuracy2]], [epoch], 'accuracy', legend=['train', 'val'])
                else:
                    self.log.info(info)


@deprecated(reason='推荐使用XlPredictor实现')
def gen_classification_func(model, *, state_file=None, transform=None, pred_func=None,
                            device=None):
    """ 工厂函数，生成一个分类器函数

    用这个函数做过渡的一个重要目的，也是避免重复加载模型

    :param model: 模型结构
    :param state_file: 存储参数的文件
    :param transform: 每一个输入样本的预处理函数
    :param pred_func: model 结果的参数的后处理
    :return: 返回的函数结构见下述 cls_func
    """
    if state_file: model.load_state_dict(torch.load(str(state_file), map_location=get_device()))
    model.train(False)
    device = device or get_device()
    model.to(device)

    def cls_func(raw_in):
        """
        :param raw_in: 输入可以是路径、np.ndarray、PIL图片等，都为转为batch结构的tensor
            im，一张图片路径、np.ndarray、PIL图片
            [im1, im2, ...]，多张图片清单
        :return: 输入如果只有一张图片，则返回一个结果
            否则会存在list，返回多个结果
        """
        dataset = InputDataset(raw_in, transform)
        # TODO batch_size根据device空间大小自适应设置
        xs = torch.utils.data.DataLoader(dataset, batch_size=8)
        res = None
        for x in xs:
            # 每个batch可能很大，所以每个batch依次放到cuda，而不是一次性全放入
            x = x.to(device)
            y = model(x)
            if pred_func: y = pred_func(y)
            res = y if res is None else (res + y)
        return res

    return cls_func


class XlPredictor:
    """ 生成一个类似函数用法的推断功能类

    这是一个通用的生成器，不同的业务可以继承开发，进一步设计细则

    这里默认写的结构是兼容detectron2框架的分类模型，即model.forward：
        输入：list，第1个是batch_x，第2个是batch_y
        输出：training是logits，eval是（batch）y_hat
    """

    def __init__(self, model, state_file=None, device=None, *, batch_size=1, y_placeholder=...):
        """
        :param model: 基于d2框架的模型结构
        :param state_file: 存储权重的文件
            一般写某个本地文件路径
            也可以写url地址，会下载存储到临时目录中
            可以不传入文件，直接给到初始化好权重的model，该模式常用语训练阶段的model
        :param batch_size: 支持每次最多几个样本一起推断
            具体运作细节参见 XlPredictor.inputs2loader的解释
            TODO batch_size根据device空间大小自适应设置
        :param y_placeholder: 参见XlPredictor.inputs2loader的解释
        """
        # 默认使用model所在的device
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device

        if state_file is not None:
            if is_url(state_file):
                state_file = download(state_file, XlPath.tempdir() / 'xlpr')
            state = torch.load(str(state_file), map_location=self.device)
            if 'model' in state:
                state = state['model']
            model = model.to(device)
            model.load_state_dict(state)

        self.model = model
        self.model.train(False)

        self.batch_size = batch_size
        self.y_placeholder = y_placeholder

        self.transform = self.build_transform()
        self.target_transform = self.build_target_transform()

    @classmethod
    def build_transform(cls):
        """ 单个数据的转换规则，进入模型前的读取、格式转换

        为了效率性能，建议比较特殊的不同初始化策略，可以额外定义函数接口，例如：def from_paths()
        """
        return None

    @classmethod
    def build_target_transform(cls):
        """ 单个结果的转换的规则，模型预测完的结果，到最终结果的转换方式

        一些简单的情况直接返回y即可，但还有些复杂的任务可能要增加后处理
        """
        return None

    def inputs2loader(self, raw_in, *, batch_size=None, y_placeholder=..., sampler=None, **kwargs):
        """ 将各种类列表数据，转成torch.utils.data.DataLoader类型

        :param raw_in: 各种类列表数据格式，或者单个数据，都为转为batch结构的tensor
            torch.util.data.DataLoader
                此时XlPredictor自定义参数全部无效：transform、batch_size、y_placeholder，sampler
                因为这些在loader里都有配置了
            torch.util.data.Dataset
                此时可以定制扩展的参数有：batch_size，sampler
            [data1, data2, ...]，列表表示批量处理多个数据
                此时所有配置参数均可用：transform、batch_size、y_placeholder, sampler
                通常是图片文件路径清单
                    XlPredictor原生并没有扩展图片读取功能，但可以通过transform增加CvPrcs.read来支持
            single_data，单个数据
                通常是单个图片文件路径，注意transfrom要增加xlcv.read或xlpil.read来支持路径读取
                注意：有时候单个数据就是list格式，此时需要麻烦点，再套一层list避免歧义
        :param batch_size: 支持每次最多几个样本一起推断
            TODO batch_size根据device空间大小自适应设置
        :param y_placeholder: 常见的model.forward，是只输入batch_x就行，这时候就默认值处理机制就行
            但我从d2框架模仿的写法，forward需要补一个y的真实值，输入是[batch_x, batch_y]
            实际预测数据可能没有y，此时需要填充一个batch_y=None来对齐，即设置y_placeholder=None
            或者y_placeholder=0，则所有的y用0作为占位符填充
            不过用None、0、False这些填充都很诡异，容易误导开发者，建议需要设置的时候使用-1

            如果读者写的model.forward前传机制不同，本来batch_inputs就只输入x没有y，则这里不用设置y_placeholder参数
        :param sampler: 有时候只是要简单抽样部分数据测试下，可以设置该参数
            比如random.sample(range(10), 5)：可以从前10个数据中，无放回随机抽取5个数据
        """
        if isinstance(raw_in, torch.utils.data.DataLoader):
            loader = raw_in
        else:
            if not isinstance(raw_in, torch.utils.data.Dataset):
                y_placeholder = first_nonnone([y_placeholder, self.y_placeholder], lambda x: x is not ...)
                dataset = InputDataset(raw_in, self.transform, y_placeholder=y_placeholder)
            else:
                if not isinstance(raw_in, (list, tuple)):
                    raw_in = [raw_in]
                dataset = raw_in
            batch_size = first_nonnone([batch_size, self.batch_size])
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, **kwargs)

        return loader

    def forward(self, loader, *, info=False, return_gt=True):
        """ 前向传播

        改功能是__call__的子部分，常在train、eval阶段单独调用
        因为eval阶段，已经有预设好的train_loader、val_loader，不需要使用inputs2loader智能生成一个loader

        :param torch.utils.data.DataLoader loader: 标准的DataLoader类型，每次能获取[batch_x, batch_y]
        :param info: 有时候数据量比较大，可能会需要看推断进度条
        :param return_gt: 注意跟__call__的不同，这里默认是True，__call__默认是False
            前者常用于评价阶段，后者常用于部署阶段，应用场景不同，常见配置有区别
        :return:
            return_gt=True（默认）：[(y1, y_hat1), (y2, y_hat2), ...]
            return_gt=False：[y_hat1, y_hat2, ...]
        """
        preds = []
        with torch.no_grad():
            for batched_inputs in tqdm(loader, 'eval batch', disable=not info):
                # 有的模型forward里没有处理input的device问题，则需要在这里使用self.device设置
                # batched_inputs = batched_inputs.to(self.device)  # 这一步可能不应该写在这里，还是先注释掉吧
                batch_y = self.model(batched_inputs).tolist()
                if self.target_transform:
                    batch_y = [self.target_transform(y) for y in batch_y]
                if return_gt:
                    gt = batched_inputs[1].tolist()
                    preds += list(zip(*[gt, batch_y]))
                else:
                    preds += batch_y
        return preds

    def __call__(self, raw_in, *, batch_size=None, y_placeholder=...,
                 info=False, return_gt=False):
        """ 前传推断结果

        :param batch_size: 具体运行中可以重新指定batch_size
        :param return_gt: 使用该功能，必须确保每次loader都含有[x,y]，可能是raw_in自带，也可以用y_placeholder设置默认值
            单样本：y, y_hat
            多样本：[(y1, y_hat1), (y2, y_hat2), ...]
        :return:
            单样本：y_hat
            多样表：[y_hat1, y_hat2, ...]

        根据不同model结构特殊性
        """
        loader = self.inputs2loader(raw_in, batch_size=batch_size, y_placeholder=y_placeholder)
        preds = self.forward(loader, info=info, return_gt=return_gt)
        # 返回结果，单样本的时候作简化
        if len(preds) == 1 and not isinstance(raw_in, (list, tuple, set)):
            return preds[0]
        else:
            return preds


def setup_seed(seed):
    """ 完整的需要设置的随机数种子

    不过个人实验有时候也不一定有用~~
    还是有可能各种干扰因素导致模型无法复现
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TrainingSampler:
    """ 摘自detectron2，用来做无限循环的抽样
    我这里的功能做了简化，只能支持单卡训练，原版可以支持多卡训练

    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle

    def __iter__(self):
        g = torch.Generator()
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g).tolist()
            else:
                yield from torch.arange(self._size).tolist()


class ZcPredictor:
    """ 智财ocrwork框架的封装接口

    这个本来是特用功能，不应该放这里的，反正也没啥不可公开的技术细节，为了使用方便就放这了
    """

    def __init__(self, config_file, *, gpu=None, batch_size=None, opts=None):
        """
        :param config_file: 支持输入配置文件路径，或者字符串格式的配置参数值
        :param gpu: 默认可以不设，会挑选当前最大剩余的一张卡
            注意配置文件中也有gpu参数，在该接口模式下会被弃用
        :param batch_size: 每次能同时识别的最大图片数
            注意config_file里也有batch_size，不过那是训练用的参数，跟这没必然联系，部署最好额外设置batch_size
            该参数可以不设，默认每次传入多少张图，就同时多少张进行批处理
        :param opts: 除了配置文件的参数，可以自设字典，覆盖更新配置参数值，常用的参数有
        """
        # 1 配置参数
        if isinstance(config_file, str) and config_file[-5:].lower() == '.yaml':
            deploy_path = os.environ.get('OCRWORK_DEPLOY', '.')  # 支持在环境变量自定义：部署所用的配置、模型所在目录
            config_file = os.path.join(deploy_path, config_file)
            f = open(config_file, "r")
        elif isinstance(config_file, str):
            f = io.StringIO(config_file)
        else:
            raise TypeError
        prepare_args = EasyDict(list(yaml.load_all(f, Loader=yaml.FullLoader))[0])
        f.close()

        # 2 特殊配置参数
        opts = opts or {}
        if gpu is not None:
            opts['gpu'] = str(gpu)
        if 'gpu' not in opts:
            # gpu没设置的时候，默认找一个空闲最大的显卡
            opts['gpu'] = NvmDevice().get_most_free_gpu_id()
        if 'gpu' in opts:  # 智财的配置好像必须要写字符串
            opts['gpu'] = str(opts['gpu'])
        prepare_args.update(opts)

        # 3 初始化各组件
        self.prepare_args = prepare_args
        self.batch_size = batch_size
        self.transform = lambda x: xlcv.read(x, 1)  # 默认统一转cv2的图片格式
        # self.transform = lambda x: PilPrcs.read(x, 1)  # 也可以使用pil图片格式

    def forward(self, imgs):
        raise NotImplemented('子类必须实现forward方法')

    def __call__(self, raw_in, *, batch_size=None, progress=False):
        """ 智财的框架，dataloader默认不需要对齐，重置collate_fn
        （其实不是不需要对齐，而是其augument组件会处理）

        :return: 以多个结果为例
            preds结果是list
            pred = preds[0]
                pred也是list，是第0张图的所有检测框，比如一共8个
                    每个框是 4*2 的numpy矩阵（整数）
        """
        # 1 判断长度
        if not getattr(raw_in, '__len__', None):
            imgs = [raw_in]
        else:
            imgs = raw_in
        n = len(imgs)
        batch_size = first_nonnone([batch_size, self.batch_size, n])

        # 2 一段一段处理
        preds = []
        t = tqdm(desc='forward', total=n, disable=not progress)
        for i in range(0, n, batch_size):
            inputs = imgs[i:i + batch_size]
            preds += self.forward([self.transform(img) for img in inputs])
            t.update(len(inputs))

        # 3 返回结果，单样本的时候作简化
        if len(preds) == 1 and not getattr(raw_in, '__len__', None):
            return preds[0]
        else:
            return preds
