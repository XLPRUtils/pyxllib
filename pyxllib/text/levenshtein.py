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

from collections import defaultdict
from more_itertools import chunked
import warnings

import Levenshtein
import numpy as np
import pandas as pd

from pyxllib.prog.pupil import run_once
from pyxllib.prog.specialist import dataframe_str
from pyxllib.text.pupil import briefstr

# 忽略特定的警告
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="sklearn.cluster._agglomerative",
                        lineno=1005)


@run_once('str')
def get_levenshtein_similar(x, y):
    """ 缓存各字符串之间的编辑距离 """
    return Levenshtein.ratio(x, y)


class MatchSimString:
    """ 匹配近似字符串

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
        self.origin_str = []  # 原始字符串内容
        self.key_str = []  # 对原始字符串进行处理后的字符
        self.ext_value = []  # 扩展存储一些信息

    def __getitem__(self, item):
        return self.origin_str[item]

    def __delitem__(self, item):
        del self.origin_str[item]
        del self.key_str[item]
        del self.ext_value[item]

    def __len__(self):
        return len(self.key_str)

    def get_similarity(self, x, y):
        """ 计算两对数据之间的相似度 """
        pass

    def append_candidate(self, k, v=None):
        self.origin_str.append(k)
        if callable(self.preproc):
            k = self.preproc(k)
        self.key_str.append(k)
        self.ext_value.append(v)

    def match(self, s):
        """ 跟候选字符串进行匹配，返回最佳匹配结果
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

    def match_many(self, s, count=1):
        """跟候选字符串进行匹配，返回多个最佳匹配结果
        :param str s: 待匹配的字符串
        :param int count: 需要返回的匹配数量
        :return: 匹配结果列表，列表中的元素为(idx, sim)对
        """
        scores = [(i, Levenshtein.ratio(self.key_str[i], s)) for i in range(len(self))]
        # 根据相似度排序并返回前count个结果
        return sorted(scores, key=lambda x: x[1], reverse=True)[:count]

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

    def agglomerative_clustering(self, threshold=0.5):
        """ 对内部字符串进行层次聚类

        :param threshold: 可以理解成距离的阈值，距离小于这个阈值的字符串会被聚为一类
            值越小，分出的类别越多越细
        """
        check_install_package('sklearn', 'scikit-learn')
        from sklearn.cluster import AgglomerativeClustering

        # 1 给每个样本标类别
        distance_matrix = np.zeros((len(self), len(self)))
        for i in range(len(self)):
            for j in range(i + 1, len(self)):
                # 我们需要距离，所以用1减去相似度
                distance = 1 - Levenshtein.ratio(self.key_str[i], self.key_str[j])
                distance_matrix[i, j] = distance_matrix[j, i] = distance

        # 进行层次聚类
        clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                             distance_threshold=threshold,
                                             linkage='complete')
        labels = clustering.fit_predict(distance_matrix)

        return labels

    def display_clusters(self, threshold=0.5):
        """ 根据agglomerative_clustering的结果，显示各个聚类的内容 """

        labels = self.agglomerative_clustering(threshold=threshold)
        cluster_dict = defaultdict(list)

        # 组织数据到字典中
        for idx, label in enumerate(labels):
            cluster_dict[label].append(self.origin_str[idx])

        # 按标签排序并显示
        result = {}
        for label, items in sorted(cluster_dict.items(), key=lambda x: -len(x[1])):
            result[label] = items

        return result


class HierarchicalMatchSimString(MatchSimString):
    """ 在面对数据量很大的候选数据情况下，建议使用这个层次聚类后的匹配方法 """

    def __init__(self, method=briefstr):
        super().__init__(method)
        self.groups = dict()

    def get_center_sample(self, indices=None):
        """ 输入一组下标，计算中心样本，未输入参数值的时候，则在全量样本里找 """
        if indices is None:
            indices = range(len(self))

        # 用于存储之前计算的结果
        cached_results = {}

        def get_similarity(i, j):
            """ 获取两个索引的相似度，利用缓存来避免重复计算 """
            if (i, j) in cached_results:
                return cached_results[(i, j)]
            sim_val = Levenshtein.ratio(self.key_str[i], self.key_str[j])
            cached_results[(i, j)] = cached_results[(j, i)] = sim_val
            return sim_val

        center_idx = max(indices, key=lambda x: sum(get_similarity(x, y) for y in indices))
        return center_idx

    def merge_group(self, indices, threshold=0.5, strategy='center'):
        """ 对输入的indexs清单，按照threshold的阈值进行合并
        返回的是一个字典，key是代表性样本，value是同组内的数据编号

        :param strategy: 代表样本的挑选策略
            center，中心样本
            first，第一个样本
        """
        check_install_package('sklearn', 'scikit-learn')
        from sklearn.cluster import AgglomerativeClustering

        # 1 给每个样本标类别
        n = len(indices)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # 我们需要距离，所以用1减去相似度
                distance = 1 - Levenshtein.ratio(self.key_str[indices[i]], self.key_str[indices[j]])
                distance_matrix[i, j] = distance_matrix[j, i] = distance

        # 进行层次聚类
        clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                             distance_threshold=threshold,
                                             linkage='complete')
        labels = clustering.fit_predict(distance_matrix)

        # 2 分组字典
        cluster_dict = defaultdict(list)
        # 组织数据到字典中
        for i, label in enumerate(labels):
            cluster_dict[label].append(indices[i])

        # 3 改成代表样本映射到一组里，并且按照样本数从多到少排序
        result = {}
        for label, items in sorted(cluster_dict.items(), key=lambda x: -len(x[1])):
            if strategy == 'first':
                representative = items[0]
            elif strategy == 'center':
                # 使用局部索引计算平均距离
                local_indices = [i for i, idx in enumerate(indices) if idx in items]
                sub_matrix = distance_matrix[np.ix_(local_indices, local_indices)]
                avg_distances = sub_matrix.mean(axis=1)
                representative_idx = np.argmin(avg_distances)
                representative = items[representative_idx]
            else:
                raise ValueError(f'Invalid strategy: {strategy}')
            result[representative] = items

        return result

    def init_groups(self, threshold=0.5, batch_size=1000):
        """
        :param threshold: 按照阈值进行分组，在这个距离内的都会归到一组
        :param batch_size: 因为数据可能太大，不可能一次性全量两两比较，这里可以分batch处理
            这样虽然结果不太精确，但能大大减小运算量
        """
        # 1 最开始每个样本都是一个组
        groups = {i: [i] for i in range(len(self))}
        new_groups = {}

        # 2 不断合并，直到没有组数变化
        while len(groups) > 1:
            for indices in chunked(groups.keys(), batch_size):
                # 对于这里返回的字典，原groups里的values也要对应拼接的
                indices2 = self.merge_group(indices, threshold=threshold)
                for idx, idxs in indices2.items():
                    # 获取原始分组中的索引
                    original_idxs = [groups[original_idx] for original_idx in idxs]
                    # 展平列表并分配到新分组中
                    new_groups[idx] = [item for sublist in original_idxs for item in sublist]

            # 如果分组没有发生变化，退出循环
            if len(new_groups) == len(groups):
                break

            groups = new_groups
            new_groups = {}

        # 3 按数量从多到少排序
        new_groups = {}
        for label, items in sorted(groups.items(), key=lambda x: -len(x[1])):
            new_groups[label] = items  # 暂用第一个出现的作为代表

        self.groups = new_groups
        return self.groups
