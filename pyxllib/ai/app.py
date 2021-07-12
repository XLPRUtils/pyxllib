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
        if self.training:
            x = self.feature_extractor(x)
            x = torch.flatten(x, 1)
            # dprint(x.shape)
            logits = self.classifier(x)
            return logits
        else:
            # 这个如果用 XlPredictor 框架，输入是一个list
            x = self.feature_extractor(x[0])
            x = torch.flatten(x, 1)
            # dprint(x.shape)
            logits = self.classifier(x)
            return logits.argmax(dim=1)


def get_imagedirection_func(state_file=None, device=None, batch_size=1):
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
        img = CvImg(arg, 0).resize((512, 512)).im
        return torchvision.transforms.functional.to_tensor(img)

    pred.transform = img_transform

    # pred.target_transform  还可以指定对model.forward结果的y_hat进行后处理

    return pred
