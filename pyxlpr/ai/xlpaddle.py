#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/11/05 09:01

"""
pp是paddlepaddle的缩写
"""

import os
import sys
import logging
import random
import shutil
import re

from tqdm import tqdm
import numpy as np
import pandas as pd
import humanfriendly

import paddle
import paddle.inference as paddle_infer

from pyxllib.algo.pupil import natural_sort
from pyxllib.xl import XlPath, browser
from pyxllib.xlcv import xlcv
from pyxlpr.ai.specialist import ClasEvaluater, show_feature_map


def __1_数据集():
    pass


class SequenceDataset(paddle.io.Dataset):
    def __init__(self, samples, labels=None, transform=None):
        super().__init__()
        self.samples = samples
        self.labels = labels
        # if self.labels is not None:  # 对np.array类型无法直接使用len，从通用角度看，这个参数可以不设
        #     assert len(self.samples) == len(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = self.samples[index]
        if self.transform:
            x = self.transform(x)

        if self.labels is not None:
            return x, self.labels[index]
        else:
            return x


def build_testdata_loader(samples, *, labels=None, transform=None, **kwargs):
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
        dataset = SequenceDataset(samples, labels, transform)

    return paddle.io.DataLoader(dataset, **kwargs)


class ImageClasDataset(paddle.io.Dataset):
    """ 常用的分类数据集格式 """

    def __init__(self, num_classes, samples, ratio=None, *,
                 use_img_augment=False, seed=4101, class_names=None):
        """ 直接按root下的目录数进行分类，注意如果train和val是分开的，目录结构要一直，否则自带的类别id编号会不匹配

        :param num_classes: 类别数
        :param list samples: 样本清单，每个条目有[图片路径, 类别id]
        :param int|float|list|tuple ratio: 取数据的比例，默认全取，可以输入一个区间，指定取哪个部分
            这个操作会设置seed，确保每次随机打乱后选取的结果相同
        :param list class_names: 表示id从0开始依次取，对应的类别昵称
        """
        super().__init__()

        if ratio is not None:
            if isinstance(ratio, (int, float)):
                # 每个类别取得样本区间
                if ratio > 0:
                    left, right = 0, ratio
                else:
                    left, right = (1 + ratio), 1
            else:
                left, right = ratio

            # 初始化，按类别分好组
            random.seed(seed)
            groups = [[] for i in range(num_classes)]
            for file, label in samples:
                groups[label].append(file)

            # 每个类别选取部分数据
            samples = []
            for label, files in enumerate(groups):
                n = len(files)
                random.shuffle(files)
                files2 = files[int(left * n):int(right * n)]
                samples += [[f, label] for f in files2]

        self.samples = samples
        self.num_classes = num_classes
        self.class_names = class_names
        self.use_img_augment = use_img_augment

    @classmethod
    def from_folder(cls, root, ratio=None, *, class_mode=1, **kwargs):
        """ 从类别目录式的数据，构造图像分类数据集

        :param root: 数据根目录
        :param ratio: 每个类别取多少样本量
        :param class_mode: 类别限定方法。注意空目录也会标记为1个类。
            0，一般是读取没有label标签的测试集，所有的类别，统一用0占位
            1，root下每个直接子目录是一个类别，每个类别目录里如果有嵌套目录，都会归为可用图片
            2，root下每个目录均被视为一个类别，这些类别在目录结构上虽然有嵌套结构，但在模型上先用线性类别模式处理
        """

        def run_mode0():
            samples = list(XlPath(root).glob_images('**/*'))
            return samples, []

        def run_mode1():
            samples, class_names = [], []
            dirs = sorted(XlPath(root).glob_dirs())
            for i, d in enumerate(dirs):
                class_names.append(d.name)
                for f in d.glob_images('**/*'):
                    samples.append([f, i])
            return samples, class_names

        def run_mode2():
            samples, class_names = [], []
            dirs = sorted(XlPath(root).rglob_dirs())
            for i, d in enumerate(dirs):
                class_names.append(d.name)
                for f in d.glob_images():
                    samples.append([f, i])
            return samples, class_names

        func = {0: run_mode0, 1: run_mode1, 2: run_mode2}[class_mode]
        samples, class_names = func()
        return cls(len(class_names), samples, ratio, class_names=class_names, **kwargs)

    @classmethod
    def from_label(cls, label_file, root=None, ratio=None, **kwargs):
        """ 从标注文件初始化 """
        label_file = XlPath(label_file)
        lines = label_file.read_text().splitlines()
        if root is None:
            root = label_file.parent
        else:
            root = XlPath(root)

        samples, class_names = [], set()
        for line in lines:
            if not line:
                continue
            path, label = line.split('\t')
            class_names.add(label)
            samples.append([root / path, int(label)])

        class_names = natural_sort(list(class_names))

        return cls(len(class_names), samples, ratio=ratio, class_names=class_names, **kwargs)

    def __len__(self):
        return len(self.samples)

    def save_class_names(self, outfile):
        """ 保存类别文件 """
        class_names = self.class_names
        if not class_names:
            class_names = list(map(str, range(self.num_classes)))
        outfile = XlPath(outfile)
        if not outfile.parent.is_dir():
            os.makedirs(outfile.parent)
        outfile.write_text('\n'.join(class_names))

    @classmethod
    def img_augment(cls, img):
        """ 自带的一套默认的增广、数据处理方案。实际应用建议根据不同任务做扩展调整。
        """
        import albumentations as A
        h, w, c = img.shape
        # 如果进行随机裁剪，则h, w的尺寸变化
        h = random.randint(int(h * 0.7), h)
        w = random.randint(int(w * 0.7), w)
        transform = A.Compose([
            A.RandomCrop(width=w, height=h, p=0.8),
            A.CoarseDropout(),  # 随机噪声遮挡
            A.RandomSunFlare(p=0.1),  # 随机强光
            A.RandomShadow(p=0.1),  # 随机阴影
            A.RGBShift(p=0.1),  # RGB波动
            A.Blur(p=0.1),  # 模糊
            A.RandomBrightnessContrast(p=0.2),  # 随机调整图片明暗
        ])
        return transform(image=img)['image']

    @classmethod
    def transform(cls, x):
        """ 自带的一种默认的图片预处理方案，实际应用建议根据不同任务做扩展调整。
        """
        import paddle.vision.transforms.functional as F
        img = xlcv.read(x)
        img = F.resize(img, (256, 256))  # 将图片尺寸统一，方便按batch训练。但resnet并不强制输入图片尺寸大小。
        img = np.array(img, dtype='float32') / 255.
        img = img.transpose([2, 0, 1])
        return img

    def __getitem__(self, index):
        file, label = self.samples[index]
        img = xlcv.read(file)
        if self.use_img_augment:
            img = self.img_augment(img)
        img = self.transform(img)
        return img, np.array(label, dtype='int64')


def __2_模型结构():
    pass


def check_network(x):
    """ 检查输入的模型x的相关信息 """
    msg = '总参数量：'
    msg += str(sum([p.size for p in x.parameters()]))
    msg += ' | ' + ', '.join([f'{p.name}={p.size}' for p in x.parameters()])
    print(msg)


def model_state_dict_df(model, *, browser=False):
    """ 统计模型中所有的参数

    :param browser: 不单纯返回统计表，而是用浏览器打开，展示更详细的分析报告

    详细见 w211206周报
    """
    ls = []
    # 摘选ParamBase中部分成员属性进行展示
    columns = ['var_name', 'name', 'shape', 'size', 'dtype', 'trainable', 'stop_gradient']

    state_dict = model.state_dict()  # 可能会有冗余重复

    used = set()
    for k, v in state_dict.items():
        # a 由于state_dict的机制，self.b=self.a，a、b都是会重复获取的，这时候不应该重复计算参数量
        # 但是后面计算存储文件大小的时候，遵循原始机制冗余存储计算空间消耗
        param_id = id(v)
        if param_id in used:
            continue
        else:
            used.add(param_id)
        # b msg
        msg = [k]
        for col_name in columns[1:]:
            msg.append(getattr(v, col_name, None))
        ls.append(msg)
    df = pd.DataFrame.from_records(ls, columns=columns)

    def html_content(df):
        import io

        content = f'<pre>{model}' + '</pre><br/>'
        content += df.to_html()
        total_params = sum(df['size'])
        content += f'<br/>总参数量：{total_params}'

        f = io.BytesIO()
        paddle.save(state_dict, f)
        content += f'<br/>文件大小：{humanfriendly.format_size(len(f.getvalue()))}'
        return content

    if browser:
        browser.html(html_content(df))

    return df


def __3_损失():
    pass


def __4_优化器():
    pass


def __5_评价指标():
    pass


class ClasAccuracy(paddle.metric.Metric):
    """ 分类问题的精度 """

    def __init__(self, num_classes=None, *, print_mode=0):
        """
        :param num_classes: 其实这个参数不输也没事~~
        :param print_mode:
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
        self.print_mode = print_mode

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
        if self.print_mode:
            a = ClasEvaluater(self.gt, self.pred)
            print(a.f1_score('all'))
            if self.print_mode > 1:
                print(a.crosstab())
        self.count = 0
        self.total = 0
        self.gt = []
        self.pred = []


class VisualAcc(paddle.callbacks.Callback):
    def __init__(self, logdir, experimental_name, *, reset=False, save_model_with_input=None):
        """
        :param logdir: log所在根目录
        :param experimental_name: 实验名子目录
        :param reset: 是否重置目录
        :param save_model_with_input: 默认不存储模型结构
        """
        from pyxllib.prog.pupil import check_install_package
        check_install_package('visualdl')
        from visualdl import LogWriter

        super().__init__()
        # 这样奇怪地加后缀，是为了字典序后，每个实验的train显示在eval之前
        d = XlPath(logdir) / (experimental_name + '_train')
        if reset and d.exists(): shutil.rmtree(d)
        self.write = LogWriter(logdir=str(d))
        d = XlPath(logdir) / (experimental_name + '_val')
        if reset and d.exists(): shutil.rmtree(d)
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


def __6_集成():
    pass


class XlModel(paddle.Model):
    def __init__(self, network, **kwargs):
        """

        """
        super(XlModel, self).__init__(network, **kwargs)
        self.save_dir = None
        self.train_data = None
        self.eval_data = None
        self.test_data = None
        self.callbacks = []

    def get_save_dir(self):
        """
        注意 self.save_dir、self.get_save_dir()各有用途
            self.save_dir获取原始配置，可能是None，表示未设置，则在某些场合默认不输出文件
            self.get_save_dir()，有些场合显示指定要输出文件了，则需要用这个接口获得一个明确的目录
        """
        if self.save_dir is None:
            return XlPath('.')
        else:
            return self.save_dir

    def set_save_dir(self, save_dir):
        """
        :param save_dir: 模型等保存的目录，有时候并不想保存模型，则可以不设
            如果在未设置save_dir情况下，仍使用相关读写文件功能，默认在当前目录下处理
        """
        # 相关数据的保存路径
        self.save_dir = XlPath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

    def set_dataset(self, train_data=None, eval_data=None, test_data=None):
        if train_data:
            self.train_data = train_data
            if isinstance(train_data, ImageClasDataset):
                self.train_data.save_class_names(self.get_save_dir() / 'class_names.txt')  # 保存类别昵称文件
        if eval_data:
            self.eval_data = eval_data
        if test_data:
            self.test_data = test_data
            # TODO 可以扩展一些能自动处理测试集的功能
            #   不过考虑不同性质的任务，这个不太好封装，可能要分图像分类，目标检测的分类处理
            #   但这样一来就等于重做一遍PaddleDet等框架了，不太必要

    def try_load_params(self, relpath='final.pdparams'):
        # TODO 怎么更好地resume训练？回复学习率等信息？虽然目前直接加载权重重新训练也没大碍。
        pretrained_model = self.get_save_dir() / relpath
        if pretrained_model.is_file():
            self.network.load_dict(paddle.load(str(pretrained_model)))

    def prepare_clas_task(self, optimizer=None, loss=None, metrics=None, amp_configs=None,
                          use_visualdl=None):
        """ 分类模型的一套默认的优化器、损失、测评配置

        TODO 这套配置不一定是最泛用的，需要进行更多研究

        :param use_visualdl: 是否使用visualdl
            支持输入str类型，作为自定义路径名
            否则每次实验，会自增一个编号，生成 e0001、e0002、e0003、...
        """
        from paddle.optimizer import Momentum
        from paddle.regularizer import L2Decay

        if optimizer is None:
            optimizer = Momentum(learning_rate=0.01,
                                 momentum=0.9,
                                 weight_decay=L2Decay(1e-4),
                                 parameters=self.network.parameters())
        if loss is None:
            loss = paddle.nn.CrossEntropyLoss()
        if metrics is None:
            metrics = ClasAccuracy(print_mode=2)  # 自定义可以用crosstab检查的精度类

        self.prepare(optimizer, loss, metrics, amp_configs)

        # 但有设置save_dir的时候，默认开启可视化
        if use_visualdl is None and self.save_dir is not None:
            use_visualdl = True

        if use_visualdl:
            p = self.save_dir or XlPath('.')
            if not isinstance(use_visualdl, str):
                num = max([int(re.search(r'\d+', x.stem).group())
                           for x in p.glob_dirs()
                           if re.match(r'e\d+_', x.stem)], default=0) + 1
                use_visualdl = f'e{num:04}'
            self.callbacks.append(VisualAcc(p / 'visualdl', use_visualdl))

    def train(self,
              epochs=1,
              batch_size=1,
              eval_freq=1000,  # 每多少次epoch进行精度验证，可以调大些，默认就是不验证了。反正目前机制也没有根据metric保存最优模型的操作。
              log_freq=1000,  # 每轮epoch中，每多少step显示一次日志，可以调大些
              save_freq=1000,  # 每多少次epoch保存模型。可以调大些，默认就只保存final了。
              verbose=2,
              drop_last=False,
              shuffle=True,
              num_workers=0,
              callbacks=None,
              accumulate_grad_batches=1,
              num_iters=None,
              ):
        """ 对 paddle.Model.fit的封装

        简化了上下游配置
        修改了一些参数默认值，以更符合我实际使用中的情况
        """
        train_data = self.train_data
        eval_data = self.eval_data

        callbacks = callbacks or []
        if self.callbacks:
            callbacks += self.callbacks

        super(XlModel, self).fit(train_data, eval_data, batch_size, epochs, eval_freq, log_freq,
                                 self.save_dir, save_freq, verbose, drop_last, shuffle, num_workers,
                                 callbacks, accumulate_grad_batches, num_iters)

        # 判断最后是否要再做一次eval：有验证集 + 原本不是每次epoch都预测 + 正好最后次epochs结束是eval周期结束
        # 此时paddle.Model.fit机制是恰好不会做eval的，这里做个补充
        if eval_data and eval_freq != 1 and (epochs % eval_freq == 0):
            self.evaluate(eval_data)

        # TODO 要再写个metric测评？这个其实就是evaluate，不用重复写吧。

    def save_static_network(self, *, data_shape=None):
        """ 导出静态图部署模型 """
        if data_shape is None:
            data_shape = [1, 3, 256, 256]
            # TODO 可以尝试从train_data、eval_data等获取尺寸
        data = paddle.zeros(data_shape, dtype='float32')
        paddle.jit.save(paddle.jit.to_static(self.network), str(self.get_save_dir() / 'infer/inference'), [data])


def __7_部署():
    pass


class ImageClasPredictor:
    """ 图像分类框架的预测器 """

    def __init__(self, model, *, transform=None, class_names=None):
        self.model = model
        self.transform = transform
        # 如果输入该字段，会把下标id自动转为明文类名
        self.class_names = class_names

    @classmethod
    def from_dynamic(cls, model, params_file=None, **kwargs):
        """ 从动态图初始化 """
        if params_file:
            model.load_dict(paddle.load(params_file))
        model.eval()
        return cls(model, **kwargs)

    @classmethod
    def from_static(cls, pdmodel, pdiparams, **kwargs):
        """ 从静态图初始化 """
        # 创建配置对象，并根据需求配置
        config = paddle_infer.Config(pdmodel, pdiparams)
        device = paddle.get_device()

        if device.startswith('gpu'):
            config.enable_use_gpu(0, device.split(':')[1])

        # 根据Config创建预测对象
        predictor = paddle_infer.create_predictor(config)

        def model(x):
            """ 静态图的使用流程会略麻烦一点

            以及为了跟动态图的上下游衔接，需要统一格式
                输入的tensor x 需要改成 np.array
                输出的np.array 需要改成 tensor

            TODO 关于这里动静态图部署的代码，可能有更好的组织形式，这个以后继续研究吧~~
            """
            # 获取输入的名称
            input_names = predictor.get_input_names()
            # 获取输入handle
            x_handle = predictor.get_input_handle(input_names[0])
            x_handle.copy_from_cpu(x.numpy())
            # 运行预测引擎
            predictor.run()
            # 获得输出名称
            output_names = predictor.get_output_names()
            # 获得输出handle
            output_handle = predictor.get_output_handle(output_names[0])
            output_data = output_handle.copy_to_cpu()  # return numpy.ndarray
            return paddle.Tensor(output_data)

        return cls(model, **kwargs)

    @classmethod
    def from_modeldir(cls, root, *, dynamic_net=None, **kwargs):
        """ 从特定的目录结构中初始化部署模型
        使用固定的配置范式，我自己常用的训练目录结构

        :param dynamic_net: 输入动态图模型类型，初始化动态图

        注：使用这个接口初始化，在目录里必须要有个class_names.txt文件来确定类别数
            否则请用更底层的from_dynamic、from_static精细配置
        """
        root = XlPath(root)
        class_names_file = root / 'class_names.txt'
        assert class_names_file.is_file(), f'{class_names_file} 必须要有类别昵称配置文件，才知道类别数'
        class_names = class_names_file.read_text().splitlines()

        if dynamic_net:
            clas = ImageClasPredictor.from_dynamic(dynamic_net(num_classes=len(class_names)),
                                                   str(root / 'final.pdparams'),
                                                   class_names=class_names,
                                                   **kwargs)
        else:
            clas = cls.from_static(str(root / 'infer/inference.pdmodel'),
                                   str(root / 'infer/inference.pdiparams'),
                                   class_names=class_names,
                                   **kwargs)

        return clas

    def pred_batch(self, samples, batch_size=None, *, return_mode=0, print_mode=0):
        """ 默认是进行批量识别，如果只识别单个，可以用pred

        :param samples: 要识别的数据，支持类list的列表，或Dataset、DataLoader
        :param return_mode: 返回值细粒度，0表示直接预测类别，1则是返回每个预测在各个类别的概率
        :param print_mode: 0 静默运行，1 显示进度条
        :param batch_size: 默认按把imgs整个作为一个批次前传，如果数据量很大，可以使用该参数切分batch
        :return:
        """
        import paddle.nn.functional as F

        if not batch_size: batch_size = len(samples)
        data_loader = build_testdata_loader(samples, transform=self.transform, batch_size=batch_size)

        logits = []
        for inputs in tqdm(data_loader, desc='预测：', disable=not print_mode):
            logits.append(self.model(inputs))
            # if sys.version_info.minor >= 8:  # v0.1.62.2 paddlelib bug，w211202
            #     break
        logits = paddle.concat(logits, axis=0)

        if return_mode == 0:
            idx = logits.argmax(1).tolist()
            if self.class_names:
                idx = [self.class_names[x] for x in idx]
            return idx
        elif return_mode == 1:
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
