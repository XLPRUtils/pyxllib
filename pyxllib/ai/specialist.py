#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/11/07 14:04

import pandas as pd

from pyxllib.prog.pupil import check_install_package


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
    def from_paddle_sequential(cls, network):
        """ 先对一些基础的常见类型做些功能接口，复杂的情况以后有需要再说吧 """

        import paddle

        ls = []
        for x in network:
            if isinstance(x, paddle.nn.layer.conv.Conv2D):
                ls.append(['Conv2D', x._kernel_size[0], x._stride[0]])
            elif isinstance(x, paddle.nn.layer.pooling.MaxPool2D):
                ls.append(['MaxPool2D', x.ksize, x.ksize])
            else:  # 其他先不考虑，跳过
                pass

        return cls.computing(ls)
