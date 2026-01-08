#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/05/10

import re

import tiktoken

from pyxllib.prog.newbie import SingletonForEveryInitArgs


class TicToken(metaclass=SingletonForEveryInitArgs):
    def __init__(self, for_model='gpt-4'):
        """ gpt4，默认使用的就是cl100k_base """
        self.for_model = for_model
        self.enc = tiktoken.encoding_for_model(for_model)

    def encode(self, text):
        """ 将文本转换为token """
        return self.enc.encode(text)

    def count_tokens(self, text, max_length=20000, num_segments=4, *, delete_space=False):
        """ 统计文本的token数量，对于超过max_length的文本，采用估算方法 """
        if delete_space:
            text = re.sub(r'\s+', '', text)

        if len(text) <= max_length:
            return len(self.encode(text))

        # 对长文本进行采样估计
        total_length = len(text)
        sample_size = max_length // num_segments
        interval = (total_length - sample_size) // (num_segments - 1)
        samples = [text[i * interval: i * interval + sample_size] for i in range(num_segments)]
        sample_tokens = sum(len(self.encode(sample)) for sample in samples)
        estimated_tokens = sample_tokens * total_length // max_length
        return estimated_tokens
