#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/02/22 10:29

""" 对icdar2013的三种测评方法的接口封装

官方原版处理两个 zip 文件，这里扩展支持目录、内存对象
"""

import re

from pyxllib.xl import File, Dir, shorten


class IcdarEval:
    """
    >>> gt = {'1.abc': [[158, 128, 411, 181], [443, 128, 450, 169]], '2': [[176, 189, 456, 274]]}
    >>> dt = {'1.abc': [[158, 128, 411, 185], [443, 120, 450, 169]]}
    >>> ie = IcdarEval(gt, dt)  # 除了内存格式，也兼容原来的zip文件、目录初始化方法
    >>> ie.icdar2013()
    {'precision': 1.0, 'recall': 0.6667, 'hmean': 0.8}
    >>> ie.deteval()
    {'precision': 1.0, 'recall': 0.6667, 'hmean': 0.8}
    >>> ie.iou()
    {'precision': 1.0, 'recall': 0.6667, 'hmean': 0.8, 'AP': 0}
    """

    def __init__(self, gt, dt):
        """ 输入gt和dt文件

        官方原版是支持 【zip文件】，必须要遵循官方原版所有的规则
            压缩包里的文件名格式为： gt_img_1.txt， res_img_1.txt
        我这里扩展，也支持输入 【目录】，注意这种操作格式，除了文件名也要完全遵守官方的规则
            这里文件名降低要求，只匹配出第一个出现的数值
        还扩展了内存操作方式，这个格式比官方简洁，不需要遵循官方琐碎的规则，只需要
            gt是一个dict
                key写图片名或id编号都可以
                value写若干个定位框，例如 [[xmin1, ymin1, xmax1, ymax1], [xmin2, ymin2, xmax2, ymax2], ...]
            dt同gt，注意key要对应

        icdar系列指标，原本是用于文本检测效果的评测，也可以扩展应用到一般性的检测任务
            icdar只考虑单类，不考虑多类别问题，如果要加入类别问题，可以修改key达到更精细的分组效果

        附，官方原版格式说明
            {'1': b'38,43,...', '2':, b'...', ...}
            key是图片编号1,2,3...233，其实改成其他各种key也行，就是一个分组概念
            value是匹配效果，使用bytes格式，用\r\n作为换行符分开每个检测框
                对gt而言，存储x1,y1,x2,y2,label，最后必须要有个label值
                对dt而言，存储x1,y1,x2,y2
        因为我这里底层做了扩展，所以从IcdarEval入口调用的测评，都是转成了我新的字典数据结构来预测的
        """
        self.gt = self.init_label(gt)
        self.dt = self.init_label(dt)

    @classmethod
    def init_label(cls, label):
        if isinstance(label, dict):
            # 如果是字典，信任其是按照官方格式来标注的
            # {'16000,1': b'566,227,673,261,0\n682,210,945,260,0', '16001,1': ...
            return label
        elif isinstance(label, (str, File)) and str(label)[-4:].lower() == '.zip':
            # 官方原版的 zip 文件初始化方法
            return label
        elif Dir.safe_init(label):
            # 输入是目录，则按照数字编号大小顺序依次读数数据
            d = Dir(label)
            res = dict()
            for f in d.select('*.txt').subfiles():
                k = re.search(r'\d+', f.stem).group()
                res[k] = f.read(mode='b')
            return res
        else:
            raise TypeError(shorten(label))

    def _eval(self, evaluate_method, default_evaluation_params, update_params):
        eval_params = default_evaluation_params()
        if update_params:
            eval_params.update(update_params)
        eval_data = evaluate_method(self.gt, self.dt, eval_params)
        # eval_data字典还存有'per_sample'的每张图片详细数据
        res = {k: round(v, 4) for k, v in eval_data['method'].items()}  # 只保留4位小数，看起来比较舒服
        return res

    def icdar2013(self, params=None):
        from pyxllib.data.icdar import evaluate_method, default_evaluation_params
        return self._eval(evaluate_method, default_evaluation_params, params)

    def deteval(self, params=None):
        from pyxllib.data.icdar import evaluate_method, default_evaluation_params
        return self._eval(evaluate_method, default_evaluation_params, params)

    def iou(self, params=None):
        from pyxllib.data.icdar.iou import evaluate_method, default_evaluation_params
        return self._eval(evaluate_method, default_evaluation_params, params)
