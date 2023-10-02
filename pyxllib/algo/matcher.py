#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2023/09/16

from pyxllib.prog.pupil import check_install_package
# check_install_package('sklearn', 'scikit-learn')

# 这个需要C++14编译器 https://download.microsoft.com/download/5/f/7/5f7acaeb-8363-451f-9425-68a90f98b238/visualcppbuildtools_full.exe
# 在需要的时候安装，防止只是想用pyxllib很简单的功能，但是在pip install阶段处理过于麻烦
# 字符串计算编辑距离需要
# check_install_package('Levenshtein', 'python-Levenshtein')

from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs:")
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="sklearn.cluster._agglomerative",
                        lineno=1005)

from more_itertools import chunked

from tqdm import tqdm

try:
    import numpy as np
except ModuleNotFoundError:
    pass

try:  # 如果不使用字符串编辑距离相关的功能，那这个包导入失败也没关系
    import Levenshtein
except ModuleNotFoundError:
    pass

try:  # 层次聚类相关的功能
    from sklearn.cluster import AgglomerativeClustering
except ModuleNotFoundError:
    pass


class DataMatcher:
    """ 泛化的匹配类，对任何类型的数据进行匹配 """

    def __init__(self, *, cmp_key=None):
        """
        :param cmp_key: 当设置该值时，表示data中不是整个用于比较，而是有个索引列
        """
        self.cmp_key = cmp_key
        self.data = []  # 用于匹配的数据

    def __getitem__(self, i):
        return self.data[i]

    def __delitem__(self, i):
        del self.data[i]

    def __len__(self):
        return len(self.data)

    def compute_similarity(self, x, y):
        """ 计算两个数据之间的相似度，这里默认对字符串使用编辑距离 """
        if self.cmp_key:
            x = x[self.cmp_key]
        ratio = Levenshtein.ratio(x, y)
        return ratio

    def add_candidate(self, data):
        """添加候选数据"""
        self.data.append(data)

    def find_best_matches(self, item, top_n=1, print_mode=0):
        """ 找到与给定数据项最匹配的候选项。

        :param item: 需要匹配的数据项。
        :param top_n: 返回的最佳匹配数量。
        :return: 一个包含(index, similarity)的元组列表，代表最佳匹配。
        """
        # 计算所有候选数据的相似度
        similarities = [(i, self.compute_similarity(candidate, item))
                        for i, candidate in tqdm(enumerate(self.data), disable=not print_mode)]

        # 按相似度降序排序
        sorted_matches = sorted(similarities, key=lambda x: x[1], reverse=True)

        return sorted_matches[:top_n]

    def find_best_match_items(self, item, top_n=1):
        """ 直接返回匹配的数据内容，而不是下标和相似度 """
        matches = self.find_best_matches(item, top_n=top_n)
        return [self.data[m[0]] for m in matches]

    def find_best_match(self, item):
        """ 返回最佳匹配 """
        matches = self.find_best_matches(item, top_n=1)
        return matches[0]

    def find_best_match_item(self, item):
        """ 直接返回匹配的数据内容，而不是下标和相似度 """
        items = self.find_best_match_items(item)
        return items[0]

    def agglomerative_clustering(self, threshold=0.5):
        """ 对内部字符串进行层次聚类

        :param threshold: 可以理解成距离的阈值，距离小于这个阈值的字符串会被聚为一类
            值越小，分出的类别越多越细
        """
        # 1 给每个样本标类别
        distance_matrix = np.zeros((len(self), len(self)))
        for i in range(len(self)):
            for j in range(i + 1, len(self)):
                # 我们需要距离，所以用1减去相似度
                distance = 1 - self.compute_similarity(self.data[i], self.data[j])
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
            cluster_dict[label].append(self.data[idx])

        # 按标签排序并显示
        result = {}
        for label, items in sorted(cluster_dict.items(), key=lambda x: -len(x[1])):
            result[label] = items

        return result

    def get_center_sample(self, indices=None):
        """ 获取一个数据集的中心样本

        :param indices: 数据项的索引列表。如果为None，则考虑所有数据。
        :return: 中心样本的索引。
        """
        if indices is None:
            indices = range(len(self.data))

        cached_results = {}

        def get_similarity(i, j):
            """ 获取两个索引的相似度，利用缓存来避免重复计算 """
            if (i, j) in cached_results:
                return cached_results[(i, j)]
            sim_val = self.compute_similarity(self.data[indices[i]], self.data[indices[j]])
            cached_results[(i, j)] = cached_results[(j, i)] = sim_val
            return sim_val

        center_idx = max(indices, key=lambda x: sum(get_similarity(x, y) for y in indices))
        return center_idx


class GroupedDataMatcher(DataMatcher):
    """ 对数据量特别大的情况，我们可以先对数据进行分组，然后再对每个分组进行匹配 """

    def __init__(self):
        """ 初始化一个分组数据匹配器 """
        super().__init__()
        # 父类有个data(list)存储了所有数据，这里self.groups只存储数据的下标
        self.groups = dict()

    def _sort_groups(self):
        """ 按照组员数量从多到少排序groups """
        new_groups = {}
        for rep, items in sorted(self.groups.items(), key=lambda x: -len(x[1])):
            new_groups[rep] = items
        self.groups = new_groups

    def merge_group(self, indices, threshold=0.5, strategy='center'):
        """ 对输入的索引进行合并，根据阈值生成分组

        :param indices: 数据项的索引列表。
        :param threshold: 两个数据项的距离小于此阈值时，它们被认为是相似的。
        :param strategy: 选择组代表的策略，可以是'center'或'first'。
        :return: 一个字典，键是代表性数据项的索引，值是相似数据项的索引列表。
        """
        # 1 给每个样本标类别
        n = len(indices)
        if n == 1:
            return {indices[0]: indices}

        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distance = 1 - self.compute_similarity(self.data[indices[i]], self.data[indices[j]])
                distance_matrix[i, j] = distance_matrix[j, i] = distance

        clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                             distance_threshold=threshold,
                                             linkage='average')
        labels = clustering.fit_predict(distance_matrix)

        # 2 分组字典
        cluster_dict = defaultdict(list)
        for i, label in enumerate(labels):
            cluster_dict[label].append(indices[i])

        # 3 改成代表样本映射到一组里，并且按照样本数从多到少排序
        result = {}
        for label, items in sorted(cluster_dict.items(), key=lambda x: -len(x[1])):
            if strategy == 'first':
                representative = items[0]
            elif strategy == 'center':
                local_indices = [i for i, idx in enumerate(indices) if idx in items]
                sub_matrix = distance_matrix[np.ix_(local_indices, local_indices)]
                avg_distances = sub_matrix.mean(axis=1)
                representative_idx = np.argmin(avg_distances)
                representative = items[representative_idx]
            else:
                raise ValueError(f'Invalid strategy: {strategy}')
            result[representative] = items

        return result

    def init_groups(self, threshold=0.5, batch_size=1000, print_mode=0):
        """ 初始化数据的分组

        :param threshold: 两个数据项的距离小于此阈值时，它们被认为是相似的。
            这里写成1的话，一般就是故意特地把类别只分成一类
        :param batch_size: 由于数据可能很大，可以使用批量处理来减少计算量。
        :return: 一个字典，键是代表性数据项的索引，值是相似数据项的索引列表。
        """
        # 1 最开始每个样本都是一个组
        groups = {i: [i] for i in range(len(self.data))}
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

            if print_mode:
                print(f'Groups number: {len(new_groups)}')

            groups = new_groups
            new_groups = {}

        self.groups = groups
        self._sort_groups()
        return self.groups

    def split_large_groups(self, max_group_size, threshold=0.5):
        """ 对于样本数过多的类，进行进一步的拆分

        :param max_group_size: 一个组内的最大样本数，超过这个数就会被进一步拆分。
        :param threshold: 用于拆分的阈值，两个数据项的距离小于此阈值时，它们被认为是相似的。
        :return: 返回拆分后的分组。
        """

        refined_groups = {}
        for rep, items in self.groups.items():
            if len(items) > max_group_size:
                # 该组样本数超过阈值，需要进一步拆分
                sub_groups = self.merge_group(items, threshold)
                refined_groups.update(sub_groups)
            else:
                # 该组样本数在阈值范围内，保持不变
                refined_groups[rep] = items

        self.groups = refined_groups
        self._sort_groups()
        return refined_groups

    def merge_small_groups(self, min_group_size=10):
        """ 将样本数较小的组合并成一个大组

        :param min_group_size: 一个组的最小样本数，低于这个数的组将被合并。
        :return: 返回合并后的分组。
        """

        merged_group = []
        preserved_groups = {}

        for rep, items in self.groups.items():
            if len(items) < min_group_size:
                # 该组样本数低于阈值，将其添加到待合并的大组中
                merged_group.extend(items)
            else:
                # 该组样本数大于等于阈值，保留原状
                preserved_groups[rep] = items

        if merged_group:
            rep_item = self.merge_group(merged_group, 1)
            for rep, items in rep_item.items():
                preserved_groups[rep] = items

        self.groups = preserved_groups
        self._sort_groups()
        return preserved_groups
