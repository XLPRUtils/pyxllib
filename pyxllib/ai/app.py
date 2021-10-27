#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/07/08 16:16

import torch
from torch import nn
import torchvision

from pyxllib.cv.expert import CvImg
from pyxllib.ai.torch import XlPredictor


class ImageDirectionModelV2(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=2),
            nn.BatchNorm2d(10, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 5, kernel_size=3),
            nn.BatchNorm2d(5, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3),
            nn.BatchNorm2d(5, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=19220, out_features=n_classes),
        )

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        # dprint(x.shape)
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        if self.training:
            return logits
        else:
            return logits.argmax(dim=1)


def get_imagedirection_predictor(state_file=None, device=None, batch_size=1):
    """
    :param state_file: 权重文件
        默认会自动下载，可以不输入
        如果输入url会自动下载
        也可以指定本地权重文件

        这里只是该任务比较简单，所以内置权重文件了，
        在某些复杂项目中，可以设为必选参数

        目前内置的模型，效果并不高，在内部测试集中精度只有57%，
        但模型比较小才300kb，可以用于演示demo、测试
    :param device: 计划在哪个设备运行
        默认会自动找一个空闲最大的gpu设备
        没有gpu则默认使用cpu模式执行
        可以自定义 'cpu', 'cuda', 'cuda:1'
    :param batch_size: 允许多少样本同时识别
    :return: 生成一个分类器"函数"，支持单张、多张图识别，支持路径、cv2、pil格式图片
        函数返回值
            0，正向
            1，顺时针旋转了90度
            2，顺时针旋转了180度
            3，顺时针旋转了270度

    补充说明:
        这相当于是一个最简接口示例，有些模型比较复杂，可以增加初始化用的配置文件等

    """
    # 1 确定本地权重文件路径，没有则预制了一个网络上的模型，会自动下载
    if state_file is None:
        state_file = 'https://gitee.com/code4101/TestData/raw/master/ImageDirectionModelV2%20epoch=17.pth'

    # 2 初始化分类器 （不一定都要用XlPredictor框架实现，但最终提供的接口希望都跟get_imagedirection_func这样简洁）
    #   模型结构，可以借助配置文件来初始化，但接口要可能简单。
    #   权重初始化，参考前面自动获取权重文件的方法，简化交接过程。
    #   device可选项，默认自动查找一个本机设备
    #   batch_size可选项，默认为1，后续pred还可以重新设定。
    pred = XlPredictor(ImageDirectionModelV2(), state_file, device=device, batch_size=batch_size)

    # 自定义预处理器，即pred(datas)中的data要经过怎样的预处理，再传入model.forward
    def img_transform(arg):
        # 输入的参数可以是路径、opencv图片、pil图片
        # 然后会转为灰度图、resize、to_tensor
        img = CvImg.read(arg, 0).resize2((512, 512))
        return torchvision.transforms.functional.to_tensor(img)

    pred.transform = img_transform

    # pred.target_transform  还可以指定对model.forward结果的y_hat进行后处理

    return pred


class ContentTypeModel(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5, padding=2),
            nn.BatchNorm2d(10, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 5, kernel_size=3),
            nn.BatchNorm2d(5, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3),
            nn.BatchNorm2d(5, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=180, out_features=n_classes),
        )

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)

        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        if self.training:
            return logits
        else:
            return logits.argmax(dim=1)


def get_contenttype_predictor(state_file=None, device=None, batch_size=1, text=None):
    """

    :param text: 将结果类别映射到字符串

    Returns:
        0, handwriting，手写体
        1, printed，印刷体
        2, seal，印章

    """
    # 1 确定本地权重文件路径，没有则预制了一个网络上的模型，会自动下载
    if state_file is None:
        state_file = 'https://gitee.com/code4101/TestData/raw/master/ContentTypeModel%20epoch=15.pth'

    # 2 初始化分类器 （不一定都要用XlPredictor框架实现，但最终提供的接口希望都跟get_imagedirection_func这样简洁）
    pred = XlPredictor(ContentTypeModel(3), state_file, device=device, batch_size=batch_size)

    # 自定义预处理器，即pred(datas)中的data要经过怎样的预处理，再传入model.forward
    def img_transform(arg):
        # 输入的参数可以是路径、opencv图片、pil图片
        # 然后会转为灰度图、resize、to_tensor
        img = CvImg.read(arg, 1).resize2((64, 64))
        return torchvision.transforms.functional.to_tensor(img)

    pred.transform = img_transform

    if text:
        pred.target_transform = lambda idx: text[idx]

    return pred
