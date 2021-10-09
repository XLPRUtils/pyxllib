#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 16:57

from pyxllib.prog.pupil import check_install_package

check_install_package('ahocorasick', 'pyahocorasick')

from collections import Counter
import re

import ahocorasick


def make_automaton(words):
    """ 根据输入的一串words模式，生成一个AC自动机 """
    a = ahocorasick.Automaton()
    for index, word in enumerate(words):
        a.add_word(word, (index, word))
    a.make_automaton()
    return a


def count_words(content, word, scope=2, exclude=None):
    # 1 统计所有词汇出现次数
    c = Counter()
    c += Counter(re.findall(f'.{{,{scope}}}{word}.{{,{scope}}}', content))
    # 2 排除掉不处理的词 （注意因为这里每句话都已经是被筛选过的，所以处理比较简单，并不需要复杂到用区间集处理）
    if exclude:
        new_c = Counter()
        a = make_automaton(exclude)  # 创建AC自动机
        for k in c.keys():
            if not next(a.iter(k), None):
                # 如果k没匹配到需要排除的词汇，则拷贝到新的计数器
                new_c[k] = c[k]
        c = new_c
    return c
