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

from tqdm import tqdm
import numpy as np
import pandas as pd
import humanfriendly

import paddle
import paddle.inference as paddle_infer

from pyxllib.xl import XlPath, browser
from pyxllib.xlcv import xlcv
from pyxlpr.ai.specialist import ClasEvaluater, show_feature_map


class SequenceDataset(paddle.io.Dataset):
    def __init__(self, samples, labels=None, transform=None):
        super().__init__()
        self.samples = samples
        self.labels = labels
        if self.labels:
            assert len(self.samples) == len(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = self.samples[index]
        if self.transform:
            x = self.transform(x)

        if self.labels:
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
        """ 从静态图初始化
        使用固定的配置范式，我自己常用的训练目录结构

        :param dynamic_net: 输入动态图模型类型，初始化动态图
        """
        root = XlPath(root)
        class_names_file = root / 'class_names.txt'
        class_names = class_names_file.read_text().splitlines() if class_names_file.is_file() else None

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
    def __init__(self, logdir, experimental_name, *, save_model_with_input=None):
        """
        :param logdir: log所在根目录
        :param experimental_name: 实验名子目录
        :param save_model_with_input: 默认不存储模型结构
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


class ImageClasDataset(paddle.io.Dataset):
    """ 常用的分类数据集格式 """

    def __init__(self, num_classes, samples, ratio=None, *, use_img_augment=False, seed=4101, class_names=None):
        """ 直接按root下的目录数进行分类，注意如果train和val是分开的，目录结构要一直，否则自带的类别id编号会不匹配

        :param num_classes: 类别数
        :param list samples: 样本清单，每个条目有[图片路径, 类别id]
        :param int|float|list|tuple ratio: 取数据的比例，默认全取，可以输入一个区间，指定取哪个部分
            这个操作会设置seed，确保每次随机打乱后选取的结果相同
        :param list class_names: 表示id从0开始依次取，对应的类别昵称
        """
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
    def from_folder(cls, root, ratio=None, *, class_mode=0, **kwargs):
        """ 从类别目录式的数据，构造图像分类数据集

        :param root: 数据根目录
        :param ratio: 每个类别取多少样本量
        :param class_mode: 类别限定方法。注意空目录也会标记为1个类。
            0，root下每个直接子目录是一个类别，每个类别目录里如果有嵌套目录，都会归为可用图片
            1，root下每个目录均被视为一个类别，这些类别在目录结构上虽然有嵌套结构，但在模型上先用线性类别模式处理
        """

        def run_mode0():
            samples, class_names = [], []
            dirs = sorted(XlPath(root).glob_dirs())
            for i, d in enumerate(dirs):
                class_names.append(d.name)
                for f in d.glob_images('**/*'):
                    samples.append([f, i])
            return samples, class_names

        def run_mode1():
            samples, class_names = [], []
            dirs = sorted(XlPath(root).rglob_dirs())
            for i, d in enumerate(dirs):
                class_names.append(d.name)
                for f in d.glob_images():
                    samples.append([f, i])
            return samples, class_names

        func = {0: run_mode0, 1: run_mode1}[class_mode]
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

        class_names = sorted(list(class_names))

        return cls(len(class_names), samples, ratio=ratio, class_names=class_names, **kwargs)

    def __len__(self):
        return len(self.samples)

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
    def _transform(cls, x):
        import paddle.vision.transforms.functional as F
        img = xlcv.read(x)
        img = cls.img_augment(img)  # 数据增广
        img = F.resize(img, (256, 256))  # 将图片尺寸统一，方便按batch训练。但resnet并不强制输入图片尺寸大小。
        img = np.array(img, dtype='float32') / 255.
        img = img.transpose([2, 0, 1])
        return img

    def __getitem__(self, index):
        file, label = self.samples[index]
        img = xlcv.read(file)
        if self.use_img_augment:
            img = self.img_augment(img)
        img = self._transform(img)
        return img, np.array(label, dtype='int64')
