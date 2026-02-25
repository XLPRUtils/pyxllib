#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/11/07 14:04

import pandas as pd


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
