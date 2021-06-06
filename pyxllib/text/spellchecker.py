#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 17:00

import subprocess

# 0 安装库和导入库
#   spellchecker模块主要有两个类，SpellChecker和WordFrequency
#       WordFrequency是一个词频类
#       一般导入SpellChecker就行了：from spellchecker import SpellChecker
try:  # 拼写检查库，即词汇库
    from spellchecker import SpellChecker
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'pyspellchecker'])
    from spellchecker import SpellChecker

from pyxllib.debug.pupil import dprint


class MySpellChecker(SpellChecker):
    """
    拼写检查
    190923周一21:54，源自 完形填空ocr 识别项目
    """

    def __init__(self, language="en", local_dictionary=None, distance=2, tokenizer=None, case_sensitive=False,
                 df=None):
        from collections import defaultdict, Counter

        # 1 原初始化功能
        super(MySpellChecker, self).__init__(language=language, local_dictionary=local_dictionary,
                                             distance=distance, tokenizer=tokenizer,
                                             case_sensitive=case_sensitive)

        # 2 自己要增加一个分析用的字典
        self.checkdict = defaultdict(Counter)
        for k, v in self.word_frequency._dictionary.items():
            self.checkdict[k][k] = v

        # 3 如果输入了一个df对象要进行更新
        if df: self.update_by_dataframe(df)

    def update_by_dataframe(self, df, weight_times=1):
        """
        :param df: 这里的df有要求，是DataFrame对象，并且含有这些属性列：old、new、count
        :param weight_times: 对要加的count乘以一个倍率
        :return:
        """
        # 1 是否要处理大小写
        #   如果不区分大小写，需要对df先做预处理，全部转小写
        #   而大小写不敏感的时候，self.word_frequency._dictionary在init时已经转小写，不用操心
        if not self._case_sensitive:
            df.loc[:, 'old'] = df.loc[:, 'old'].str.lower()
            df.loc[:, 'new'] = df.loc[:, 'new'].str.lower()

        # 2 df对self.word_frequency._dictionary、self.check的影响
        d = self.word_frequency._dictionary
        for index, row in df.iterrows():
            old, new, count = row['old'].decode(), row['new'].decode(), row['count'] * weight_times
            d[old] += count if old == new else -count
            # if row['id']==300: dprint(old, new, count)
            self.checkdict[old][new] += count

        # 3 去除d中负值的key
        self.word_frequency.remove_words([k for k in d.keys() if d[k] <= 0])

    def _ensure_term(self, term):
        if term not in self.checkdict:
            d = {k: self.word_frequency._dictionary[k] for k in self.candidates(term)}
            self.checkdict[term] = d

    def correction(self, term):
        # 1 本来就是正确的
        w = term if self._case_sensitive else term.lower()
        if w in self.word_frequency._dictionary: return term

        # 2 如果是错的，且是没有记录的错误情况，则做一次候选项运算
        self._ensure_term(w)

        # 3 返回权重最大的结果
        res = max(self.checkdict[w], key=self.checkdict[w].get)
        val = self.checkdict[w].get(res)
        if val <= 0: res = '^' + res  # 是一个错误单词，但是没有推荐修改结果，就打一个^标记
        return res

    def correction_detail(self, term):
        """更加详细，给出所有候选项的纠正

        >> a.correction_detail('d')
        [('d', 9131), ('do', 1), ('old', 1)]
        """
        w = term if self._case_sensitive else term.lower()
        self._ensure_term(w)
        ls = [(k, v) for k, v in self.checkdict[w].items()]
        ls = sorted(ls, key=lambda x: x[1], reverse=True)
        return ls


def demo_myspellchecker():
    # 类的初始化大概要0.4秒
    a = MySpellChecker()

    # sql的加载更新大概要1秒
    # hsql = HistudySQL('ckz', 'tr_develop')
    # df = hsql.query('SELECT * FROM spell_check')
    # a.update_by_dataframe(df)

    # dprint(a.correction_detail('d'))
    # dprint(a.correction_detail('wrod'))  # wrod有很多种可能性，但word权重是最大的
    # dprint(a.correction_detail('ckzckzckzckzckzckz'))  # wrod有很多种可能性，但word权重是最大的
    # dprint(a.correction('ckzckzckzckzckzckz'))  # wrod有很多种可能性，但word权重是最大的
    dprint(a.correction_detail('ike'))
    dprint(a.correction_detail('dean'))
    dprint(a.correction_detail('stud'))
    dprint(a.correction_detail('U'))


def demo_spellchecker():
    """演示如何使用spellchecker库
    官方介绍文档 pyspellchecker · PyPI: https://pypi.org/project/pyspellchecker/
    190909周一15:58，from 陈坤泽
    """
    # 1 创建对象
    # 可以设置语言、大小写敏感、拼写检查的最大距离
    #   默认'en'英语，大小写不敏感
    spell = SpellChecker()
    # 如果是英语，SpellChecker会自动加载语言包site-packages\spellchecker\resources\en.json.gz，大概12万个词汇，包括词频权重
    d = spell.word_frequency  # 这里的d是WordFrequency对象，其底层用了Counter类进行数据存储
    dprint(d.unique_words, d.total_words)  # 词汇数，权重总和

    # 2 修改词频表 spell.word_frequency
    dprint(d['ckz'])  # 不存在的词汇直接输出0
    d.add('ckz')  # 可以添加ckz词汇的一次词频
    d.load_words(['ckz', 'ckz', 'lyb'])  # 可以批量添加词汇
    dprint(d['ckz'], d['lyb'])  # d['ckz']=3  d['lyb']=1
    d.load_words(['ckz'] * 100 + ['lyb'] * 500)  # 可以用这种技巧进行大权重的添加
    dprint(d['ckz'], d['lyb'])  # d['ckz']=103  d['lyb']=501

    # 同理，去除也有remove和remove_words两种方法
    d.remove('ckz')
    # d.remove_words(['ckz', 'lyb'])  # 不过注意不能删除已经不存在的key（'ckz'），否则会报KeyError
    dprint(d['ckz'], d['lyb'])  # d['ckz']=0  d['lyb']=501
    # remove是完全去除单词，如果只是要减权重可以访问底层的_dictionary对象操作
    d._dictionary['lyb'] -= 100  # 当然不太建议直接访问下划线开头的成员变量~~
    dprint(d['lyb'])  # ['lyb']=401

    # 还可以按阈值删除词频不超过设置阈值的词汇
    d.remove_by_threshold(5)

    # 3 spell的基本功能
    # （1）用unknown可以找到可能拼写错误的单词，再用correction可以获得最佳修改意见
    misspelled = spell.unknown(['something', 'is', 'hapenning', 'here'])
    dprint(misspelled)  # misspelled<set>={'hapenning'}

    for word in misspelled:
        # Get the one `most likely` answer
        dprint(spell.correction(word))  # <str>='happening'
        # Get a list of `likely` options
        dprint(spell.candidates(word))  # <set>={'henning', 'happening', 'penning'}

    # 注意默认的spell不区分大小写，如果词库存储了100次'ckz'
    #   此时判断任意大小写形式组合的'CKZ'都是返回原值
    #   例如 spell.correction('ckZ') => 'ckZ'

    # （2）可以通过修改spell.word_frequency影响correction的计算结果
    dprint(d['henning'], d['happening'], d['penning'])
    # d['henning']<int>=53    d['happening']<int>=4538    d['penning']<int>=23
    d._dictionary['henning'] += 10000
    dprint(spell.correction('hapenning'))  # <str>='henning'

    # （3）词汇在整个字典里占的权重
    dprint(spell.word_probability('henning'))  # <float>=0.0001040741914298211
