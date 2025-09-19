#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2023/11/05

""" 基于jieba库的一些文本处理功能 """

from collections import Counter
import re

from pyxllib.prog.lazyimport import lazy_import

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lazy_import('from tqdm import tqdm')

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import('pandas')

try:
    import jieba
    import jieba.posseg as pseg
except ModuleNotFoundError:
    jieba = lazy_import('jieba')
    pseg = lazy_import('jieba.posseg')

try:
    from simhash import Simhash
except ModuleNotFoundError:
    Simhash = lazy_import('from simhash import Simhash')

from pyxllib.prog.pupil import DictTool, run_once
from pyxllib.file.specialist import XlPath
from pyxllib.algo.stat import update_dataframes_to_excel


def jieba_add_words(words):
    for w in words:
        jieba.add_word(w)


def jieba_del_words(words):
    for w in words:
        jieba.del_word(w)


@run_once('str')
def jieba_cut(text):
    return tuple(jieba.cut(text))


@run_once('str')
def pseg_cut(text):
    return tuple(pseg.cut(text))


def _count_word_frequency(texts, function_word=True):
    """ 统计关键词出现频数 (主要是协助计算tf-idf)

    :param texts: 输入字符串列表
    :param function_word: 是否要统计虚词
    :return: 一个dict
        key: 分词名称
        values: [x, y]，x是出现总频数，y是这个词在多少篇文章中出现过

    >>> _count_word_frequency(['正正正正', '正反正', '反反反反'])
    {'正正': [1, 1], '反反': [2, 1]}

    原没有过滤词性的结果：{'正正': [2, 1], '正': [1, 1], '反正': [1, 1], '反反': [2, 1]}
    """

    d = dict()
    for text in tqdm(texts, '词频统计'):
        wordflags = list(pseg.cut(text))
        words = set()
        for word, flag in wordflags:
            # 虚词不做记录
            if (not function_word) and flag in ('uj', 'd', 'p', 'c', 'u', 'xc'):
                continue
            words.add(word)
            if word not in d:
                d[word] = [0, 0]
            d[word][0] += 1
        for word in words:
            d[word][1] += 1
    return d


def analyse_tf_idf(texts, outfile=None, sheet_name='tf-idf', *, function_word=True):
    """ 分析tf-idf值

    :param list[str] texts: 多份文件的文本内容
    :return: 一个DataFrame数据

    这个算法jieba可能有些自带库可以搞，但是自己写一下也不难啦
    注意我这里返回的tf-idf中，是放大了总频数倍的，这样显示的数值大一点，看起来舒服~
    """
    from math import log10

    frequency = _count_word_frequency(texts, function_word)
    DictTool.isub(frequency, [' ', '\t', '\n'])

    n = len(texts)
    sum_frequency = sum([v[0] for v in frequency.values()])

    li = []
    for k, v in frequency.items():
        idf = log10(n / v[1])
        # idf = 1
        li.append([k, v[0], v[0] / sum_frequency, v[1], idf, v[0] * idf])
    df = pd.DataFrame.from_records(li, columns=('词汇', '频数', '频率', '出现该词文章数', 'idf', 'tf-idf'))
    df.sort_values(by='tf-idf', ascending=False, inplace=True)

    if outfile:
        update_dataframes_to_excel(outfile, {sheet_name: df})

    return df


class TextClassifier:
    def __init__(self, texts=None):
        """ 文本分类器

        :param list[str] texts: 文本内容
        """

        self.texts = []
        self.tfidf = {}
        self.vecs = []  # 每份文本对应的向量化表达
        self.default_tfidf = 1  # 如果没有计算tf-idf，可以全部默认用权重1

        if texts:
            for text in texts:
                self.texts.append(text)

    def get_text_tf(self, text, *,
                    function_word_weight=0.2,
                    normalize=True,
                    ingore_words=(' ', '\t', '\n'),
                    add_flag=False):
        """ 这里可以定制提取text关键词的算法

        :param function_word_weight: 这里可以自定义功能性词汇权重，一般是设一个小数降低权重

        一般是定制一些过滤规则，比如过滤掉一些词性，或者过滤掉一些词
        """
        ct = Counter()

        # 1 初步的分词，以及是否要过滤虚词
        wordflags = list(pseg_cut(text))
        for word, flag in wordflags:
            if flag in ('uj', 'd', 'p', 'c', 'u', 'xc', 'x'):
                if add_flag:
                    ct[word + ',' + flag] += function_word_weight
                else:
                    ct[word] += function_word_weight
            else:
                if add_flag:
                    ct[word + ',' + flag] += 1
                else:
                    ct[word] += 1

        # 2 归一化一些词
        if normalize:
            ct2 = Counter()
            for k, v in ct.items():
                # 如果需要对一些词汇做归一化，也可以这里设置
                k = re.sub(r'\d', '0', k)  # 把数字都换成0
                ct2[k] += v
            ct = ct2

        # 3 过滤掉一些词
        if ingore_words:
            for k in ingore_words:
                if k in ct:
                    del ct[k]

        return ct

    def compute_tfidf(self, outfile=None, sheet_name='tf-idf', normalize=False, function_word_weight=0.2,
                      add_flag=False):
        """ 重算tfidf表 """
        from math import log10

        # 1 统计频数和出现该词的文章数
        d = dict()
        for text in tqdm(self.texts, '词频统计'):
            ct = self.get_text_tf(text, normalize=normalize, function_word_weight=function_word_weight,
                                  add_flag=add_flag)
            for k, v in ct.items():
                if k not in d:
                    d[k] = [0, 0]
                d[k] = [d[k][0] + v, d[k][1] + 1]

        # 2 计算tfidf
        n = len(self.texts)
        sum_tf = sum([v[0] for v in d.values()])
        ls = []
        for k, v in d.items():
            idf = log10(n / v[1])
            # idf = 1
            ls.append([k, v[0], v[0] / sum_tf, v[1], idf, v[0] * idf])

        df = pd.DataFrame.from_records(ls, columns=('词汇', '频数', '频率', '出现该词文章数', 'idf', 'tf-idf'))
        df.sort_values(by='tf-idf', ascending=False, inplace=True)

        # 3 保存到文件
        if outfile:
            update_dataframes_to_excel(outfile, {sheet_name: df})

        self.tfidf = {row['词汇']: row['tf-idf'] for idx, row in df.iterrows()}
        self.default_tfidf = df.loc[len(df) - 1]['tf-idf']  # 最后条的权重作为其他未见词的默认权重

        return df

    def normalization(self, d):
        """ 向量归一化

        输入一个类字典结构表示的向量，对向量做归一化处理
        """
        length = sum([v * v for v in d.values()]) ** 0.5  # 向量长度
        return {k: v / length for k, v in d.items()}

    def get_text_vec(self, text):
        """ 获取文本的向量化表达

        :param str text: 文本内容
        """
        ct = self.get_text_tf(text)
        vec = {k: v * self.tfidf.get(k, self.default_tfidf) for k, v in ct.items()}
        vec = self.normalization(vec)
        return vec

    def compute_vecs(self):
        """ 重置向量化表达 """
        vecs = []
        for text in tqdm(self.texts, desc='query向量化'):
            vecs.append(self.get_text_vec(text))
        self.vecs = vecs
        return vecs

    def cosine_similar(self, x, y):
        """ 两个向量的余弦相似度，值越大越相似

        这里是简化的，只算两个向量的点积，请确保输入的都是单位长度的向量
        注意这里x和y都是稀疏矩阵的存储形式，传入的是dict结构
        """
        keys = x.keys() & y.keys()  # 求出x和y共有的键值
        return sum([x[k] * y[k] for k in keys])

    def find_similar_vec(self, x, maxn=10):
        """ 找与x最相近的向量，返回下标和相似度

        :pamra x: 待查找的对象
        :param maxn: 返回最相近的前maxn个对象
        """
        if isinstance(x, str):
            x = self.get_text_vec(x)

        # todo 使用并行计算？或者其实也可以向量化，但向量化是稀疏矩阵，挺占空间的
        sims = [(i, self.cosine_similar(x, v)) for i, v in enumerate(self.vecs)]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:maxn]

    def refine_vecs(self):
        """ 优化向量数据，去掉权重小余0.0001的维度 """
        # 1 计算每个向量的长度
        vecs = []
        for vec in tqdm(self.vecs, '优化向量'):
            vec = [(k, v) for k, v in vec.items()]
            vec.sort(key=lambda x: x[1], reverse=True)
            vec2 = {}
            for k, v in vec:
                if v < 0.0001:
                    break
                vec2[k] = round(v, 4)
            vecs.append(vec2)

        self.vecs = vecs
        return self.vecs
