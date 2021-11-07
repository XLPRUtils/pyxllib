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
