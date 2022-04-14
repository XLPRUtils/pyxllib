#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/03/03 09:35

""" 主要含有功能

结果分数测评
结果调试、可视化分析
"""

import pandas as pd

from pyxllib.data.coco import *


class KieResParser:
    """ Key Information Extraction Result Parser

    关键信息提取结果的分析工具
    """

    def __init__(self, df, keys):
        """
        :param df: 必须要有一列file文件标识，用来进行分组判断，每一组的识别效果
        :param keys: 信息种类，有哪些键
            for k in keys，都有对应名称的 dt_{k},  gt_{k}  属性列，测试集场合，可以没有 gt 列
        """
        assert 'file' in df.columns
        self.df = df
        self.keys = keys
        self.add_diff_columns()

    def add_diff_columns(self):
        """ 使用该功能，必须要有对应的gt列 """
        for k in self.keys:
            diff_col = f'diff_{k}'
            if diff_col not in self.df.columns:
                dt_col, gt_col = f'dt_{k}', f'gt_{k}'
                self.df[diff_col] = [StrDiffType.difftype(row[dt_col], row[gt_col]) for idx, row in self.df.iterrows()]

    def check_file(self):
        """ 按每份文档为单位查看效果

        TODO
            1、支持查看单文件
            2、支持筛选key，只对比查看部分key
            3、支持显示key名称
        """
        pc = PairContent(Dir.TEMP / 'dt.txt', Dir.TEMP / 'gt.txt')
        for i, (idx, item) in enumerate(self.df.iterrows(), start=1):
            pc.add(f'{i}、{item.file}')
            for k in self.keys:
                pc.add(item[f'dt_{k}'], item[f'gt_{k}'])
            pc.add('')
        pc.bcompare(wait=False)

    def stat_difftype(self):
        """ 分析dt各种情况下检测效果、质量 """
        items = []
        index = []
        for i in StrDiffType.typename.keys():
            vals = []
            for k in self.keys:
                vals.append(sum(self.df[f'diff_{k}'] == i))
            if sum(vals):
                items.append(vals)
                index.append(i)
        df = pd.DataFrame.from_records(items, columns=self.keys, index=index)
        return df

    def check_key(self, key):
        """ 检查某个特定key的识别效果 """
        n = len(self.df)  # 总条目数
        pc = PairContent(Dir.TEMP / f'dt-{key}.txt', Dir.TEMP / f'gt-{key}.txt')
        dt_col, gt_col, diff_col = f'dt_{key}', f'gt_{key}', f'diff_{key}'
        for difftype, items in self.df.groupby(diff_col):
            m = len(items)
            pc.add(f'{difftype}、{StrDiffType.typename[difftype]}（{m}/{n}≈{m / n:.2%}）')
            for _, x in items.iterrows():
                f = x['file'] + ', '
                pc.add(f + x[dt_col], f + x[gt_col])
            pc.add('')
        pc.bcompare(wait=False)
