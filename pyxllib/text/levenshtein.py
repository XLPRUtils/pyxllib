#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 17:01

from pyxllib.prog.pupil import check_install_package

# 这个需要C++14编译器 https://download.microsoft.com/download/5/f/7/5f7acaeb-8363-451f-9425-68a90f98b238/visualcppbuildtools_full.exe
# 在需要的时候安装，防止只是想用pyxllib很简单的功能，但是在pip install阶段处理过于麻烦
# MatchSimString计算编辑距离需要
check_install_package('Levenshtein', 'python-Levenshtein')

import Levenshtein
import pandas as pd

from pyxllib.prog.specialist import dataframe_str
from pyxllib.text.pupil import briefstr


class MatchSimString:
    """匹配近似字符串

    mss = MatchSimString()

    # 1 添加候选对象
    mss.append_candidate('福州+厦门2018初数暑假讲义-请录入-快乐学习\初一福厦培优-测试用')
    mss.append_candidate('2018_快乐数学_六年级_秋季_第01讲_圆柱与圆锥_教案（教师版）')
    mss.append_candidate('删除所有标签中间多余的空白')

    # 2 需要匹配的对象1
    s = '奕本初一福周厦门培油'

    idx, sim = mss.match(s)
    print('匹配目标：', mss[idx])  # 匹配目标： 福州+厦门2018初数暑假讲义-请录入-快乐学习\初一福厦培优-测试用
    print('相似度：', sim)         # 相似度： 0.22

    # 3 需要匹配的对象2
    s = '圆柱与【圆锥】_教案空白版'

    idx, sim = mss.match(s)
    print('匹配目标：', mss[idx])  # 2018_快乐数学_六年级_秋季_第01讲_圆柱与圆锥_教案（教师版）
    print('相似度：', sim)         # 相似度： 0.375

    如果append_candidate有传递2个扩展信息参数，可以索引获取：
    mss.ext_value[idx]
    """

    def __init__(self, method=briefstr):
        self.preproc = method
        self.origin_str = list()  # 原始字符串内容
        self.key_str = list()  # 对原始字符串进行处理后的字符
        self.ext_value = list()  # 扩展存储一些信息

    def __getitem__(self, item):
        return self.origin_str[item]

    def __delitem__(self, item):
        del self.origin_str[item]
        del self.key_str[item]
        del self.ext_value[item]

    def __len__(self):
        return len(self.key_str)

    def append_candidate(self, k, v=None):
        self.origin_str.append(k)
        if callable(self.preproc):
            k = self.preproc(k)
        self.key_str.append(k)
        self.ext_value.append(v)

    def match(self, s):
        """跟候选字符串进行匹配，返回最佳匹配结果
        """
        idx, sim = -1, 0
        for i in range(len(self)):
            k, v = self.key_str[i], self.ext_value[i]
            sim_ = Levenshtein.ratio(k, s)
            if sim_ > sim:
                sim = sim_
                idx = i
            i += 1
        return idx, sim

    def match_test(self, s, count=-1, showstr=lambda x: x[:50]):
        """输入一个字符串s，和候选项做近似匹配

        :param s: 需要进行匹配的字符串s
        :param count: 只输出部分匹配结果
            -1：输出所有匹配结果
            0 < count < 1：例如0.4，则只输出匹配度最高的40%结果
            整数：输出匹配度最高的count个结果
        :param showstr: 字符串显示效果
        """
        # 1 计算编辑距离，存储结果到res
        res = []
        n = len(self)
        for i in range(n):
            k, v = self.key_str[i], self.ext_value[i]
            sim = Levenshtein.ratio(k, s)
            res.append([i, v, sim, showstr(k)])  # 输出的时候从0开始编号
            i += 1

        # 2 排序、节选结果
        res = sorted(res, key=lambda x: -x[2])
        if 0 < count < 1:
            n = max(1, int(n * count))
        elif isinstance(count, int) and count > 0:
            n = min(count, n)
        res = res[:n]

        # 3 输出
        df = pd.DataFrame.from_records(res, columns=('序号', '标签', '编辑距离', '内容'))
        s = dataframe_str(df)
        s = s.replace('\u2022', '')  # texstudio无法显示会报错的字符
        print(s)
