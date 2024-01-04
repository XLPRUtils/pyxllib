#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2023/07/13 14:26

from pyxllib.prog.pupil import check_install_package

# check_install_package('transformers', 'transformers')

import ast
from collections import OrderedDict
from collections import Counter
import contextlib
import copy
import datetime
import heapq
import html
import json
import math
import random
import re
from urllib.parse import unquote
import io
import logging
import warnings

from jinja2 import Template
from openpyxl import Workbook
import pandas as pd
import requests
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError:
    pass

from pyxllib.prog.pupil import OutputLogger
from pyxllib.prog.specialist import browser, TicToc
from pyxllib.algo.pupil import ValuesStat
from pyxllib.file.specialist import XlPath, JsonlDataFile, JsonlDataDir, TwinDirs, ensure_localdir
from pyxllib.file.xlsxlib import extract_workbook_summary


def __1_生成提问数据():
    pass


class Tokenizer:
    _tokenizer = None

    @classmethod
    def get_tokenizer(cls):
        """ 获取tokenizer，第一次调用时进行初始化 """

        if cls._tokenizer is None:
            # 根本没必要每次都尝试连接官网，本地有就不要老是sb的尝试连接huggingface
            # 而且官网连接也不稳，这里换成我自己的服务器中转
            # gpt2_dir = XlPath.tempdir() / 'huggingface_gpt2'
            # ensure_localdir(gpt2_dir, 'https://xmutpriu.com/download/huggingface_gpt2.zip')
            # Tokenizer._tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_dir)
            # 240103周三21:23，hx给过的新评测模型
            gpt2_dir = XlPath.tempdir() / 'Atom-CL-SS'
            ensure_localdir(gpt2_dir, 'https://xmutpriu.com/download/Atom-CL-SS.zip')
            cls._tokenizer = AutoTokenizer.from_pretrained(gpt2_dir, trust_remote_code=True)
        return cls._tokenizer

    @classmethod
    def tokenize(cls, paragraph, max_length=500):
        """ 对段落进行tokenize

        :param str paragraph: 待分词的段落
        :param int max_length: 单次处理的最大分词数，为了防止超过GPT2的限制，默认设置为500
        :return list: 分词后的列表

        >>> Tokenizer.tokenize('Hello, world! 汉字 123.14 35')
        ['Hello', ',', 'Ġworld', '!', 'Ġæ', '±', 'ī', 'åŃ', 'Ĺ', 'Ġ123', '.', '14', 'Ġ35']
        """
        tokenizer = cls.get_tokenizer()

        # 对段落进行切分
        paragraph_slices = [paragraph[i:i + max_length] for i in range(0, len(paragraph), max_length)]

        # 对每个切分的子段进行分词，并将结果拼接在一起
        tokens = []
        for slice in paragraph_slices:
            tokens += tokenizer.tokenize(slice)

        return tokens

    @classmethod
    def count_tokens(cls, paragraph, max_length=500):
        """ 获取段落的token数量

        :param str paragraph: 待分词的段落
        :param int max_length: 单次处理的最大分词数，为了防止超过GPT2的限制，默认设置为500
        :return int: token的数量

        >>> Tokenizer.count_tokens('Hello, world!')
        5
        """
        return len(cls.tokenize(paragraph, max_length))


def print_statistics(data, indent_level=1):
    """ 计算字符串长度，并且计算关键的一些token数

    :param data: data应该是一个嵌套结构，表示会话与消息
    """
    fmts = ['g', '.0f', '.0f', 'd', 'd']
    stat_len = ValuesStat([len(str(x)) for x in data])

    indent = '\t' * indent_level
    print(f'{indent} {stat_len.summary(fmts)}')


def check_conversation_lengths(all_texts, n_values=(4, 4),
                               compute_tokens=False, ids=None):
    """ 分析会话长度 """

    # 0 预处理
    for i, texts in enumerate(all_texts):
        if isinstance(texts, str):
            all_texts[i] = [texts]

    # 如果没有提供ID，则使用默认的range(n)
    if ids is None:
        ids = list(range(len(all_texts)))

    # 处理n_values的重叠
    if sum(n_values) >= len(all_texts):
        n_values = [len(all_texts), 0]  # 将所有数据视为最短数据，不再考虑最长数据

    # 1 消息长度统计
    fmts = [None, '.0f', '.0f', 'd', 'd']
    lengths = [len(t) for texts in all_texts for t in texts]
    print(f'1、消息长度统计 {ValuesStat(lengths).summary(fmts)}')

    # 2 每组会话消息数目
    ct = Counter(len(texts) for texts in all_texts)
    sorted_ct = {k: v for k, v in sorted(ct.items(), key=lambda x: x[0])}
    print(f'2、每组消息数目: {sorted_ct}')

    # 3 找出消息总长度最短和最长的会话
    total_lengths = [(i, sum(len(t) for t in texts)) for i, texts in enumerate(all_texts)]
    shortest_indices = [item[0] for item in heapq.nsmallest(n_values[0], total_lengths, key=lambda x: x[1])]
    longest_indices = [item[0] for item in heapq.nlargest(n_values[1], total_lengths, key=lambda x: x[1])]
    longest_indices = longest_indices[::-1]  # 从小到大排序

    parts = []
    if shortest_indices:
        parts.append(', '.join(map(str, [ids[i] for i in shortest_indices])))
    if longest_indices:
        parts.append(', '.join(map(str, [ids[i] for i in longest_indices])))
    print(f'3、最短最长会话的id:', ', ..., '.join(parts))

    # 4 计算token
    if compute_tokens:
        # 4.1 代表性样本的tokens数
        s_texts = [' '.join([x for x in all_texts[i]]) for i in shortest_indices]
        l_texts = [' '.join([x for x in all_texts[i]]) for i in longest_indices]

        s_lens = [[len(x), Tokenizer.count_tokens(x)] for x in s_texts]
        l_lens = [[len(x), Tokenizer.count_tokens(x)] for x in l_texts]

        parts = []
        if s_lens:
            parts.append(', '.join(map(str, [x[1] for x in s_lens])))
        if l_lens:
            parts.append(', '.join(map(str, [x[1] for x in l_lens])))
        # 仅计算3中代表性样本
        print(f'4、tokens数量:', ', ..., '.join(parts))

        # 4.2 token的比率规律
        ratios = []
        for x in s_lens + l_lens:
            ratios.append(x[1] / x[0])
        fmts = [None, '.0%', '.0%', '.0%', '.0%']
        print(f'token/len比率统计 {ValuesStat(ratios).summary(fmts)}')
        # 比率越大，代表越接近中文场景，汉字越多，要注意len的控制不要让token某些场合超出长度


def set_template(s, *args, **kwargs):
    """ todo 这个名字会不会太容易冲突了？ """
    return Template(s.strip(), *args, **kwargs)


def set_meta_template(s, meta_start='[[', meta_end=']]', **kwargs):
    """ 支持预先用某些格式渲染后，再返回标准渲染模板 """
    t = Template(s.strip(), variable_start_string=meta_start,
                 variable_end_string=meta_end).render(**kwargs)
    return Template(t)


class StyleParser:
    def __init__(self, text):
        # 使用正则表达式拆分文本，并获取权重和风格
        self.styles = []
        self.weights = []
        matches = re.findall(r'<风格变换\d+(\s+(\d+))?[^>]*>\s*(.*?)\s*(?=<风格变换|$)', text, re.DOTALL)
        for match in matches:
            self.styles.append(match[2])
            # 提取权重
            weight = match[1]
            if weight:
                self.weights.append(int(weight))
            else:
                self.weights.append(100)  # 默认权重

    def random_pick(self):
        """ 随机选择一个风格，并返回其下标和内容

        :return tuple: (下标, 风格内容)

        >>> sp = StyleParser("...")  # 按照之前的格式传入一个字符串
        >>> index, style = sp.random_pick()  # 随机选择一个风格
        """
        index = random.choices(range(len(self.styles)), weights=self.weights, k=1)[0]
        return index, self.styles[index]


class GptChatJsonl(JsonlDataFile):
    """ GPT问答批量执行脚本的jsonl生成、读取器 """

    def __init__(self, file=None, num_records=None, *, start_id=None):
        from datetime import datetime

        super().__init__(file, num_records)
        if start_id is None:
            # 230821周一02:02，原本只有日期标记，后面发现这样id很容易出现重复，还是加上小时分钟更不容易引起一些没必要的麻烦
            today = datetime.now().strftime("%Y%m%d%H%M")
            self.start_id = int(today + "000000")
        else:
            self.start_id = start_id

    def read_jsonl(self, file):
        """ 从一个文件加载数据
        """
        self.records = XlPath(file).read_jsonl()
        try:
            self.start_id = self.records[-1]['id']
        except KeyError:
            pass

    def split_and_add_prompt(self, text, max_word_length=None, prompt=None):
        """
        :param text: 要插入的文本（纯文本，不能是字典格式的 {'content': ...}）
        :param max_word_length: 如果设置了该值，则会对输入内容进行长度控制，拆成多个片段
            之前测试，全英文长度大概在32500，中文在5000内
        :param prompt: max_word_length开启时才会生效，每个part要附加的提示规则
            可以写个字符串，默认每段都是开头加这个提示
            也可以参考gen_prompt2,一些生成函数写法进行自定义
        :return:
        """
        # 0 如果没有输入max_word_length，就不用特地处理了
        if max_word_length is None:
            return [text]

        # 1 工具函数
        def gen_prompt1(n, i, text):
            """ 一共n条，当前第i条的，当前内容是text """
            if n == 1:
                return text
            if n > 1:
                if i == 0:
                    return f'【注意，由于本次提问过长，这里拆分成{n}个片段分开输入，目前是第{1}个片段，你只需暂时回复"收到"即可】\n' + text
                elif i < n - 1:
                    return f'【这是{n}个片段中的第{i + 1}个片段，先回复"收到"即可】\n' + text
                else:
                    return f'【这是{n}个片段的最后一个片段，请开始回复内容】\n' + text

        def gen_prompt2(n, i, text):
            return prompt + text

        if prompt is None:
            gen_prompt = gen_prompt1
        elif isinstance(prompt, str):
            gen_prompt = gen_prompt2
        else:  # callable
            gen_prompt = prompt

        # 2 拆分重拼接
        # 首先要重新调整max_word_length，在确定要拆分为几个片段的情况下，尽量保证这些片段之间的均匀性
        num = len(text) // max_word_length + 1
        max_word_length = math.ceil(len(text) / num)

        # 2.1 检查是否有超长单行文本，要提前切分成多行
        lines = text.rstrip().split('\n')
        new_lines = []
        for line in lines:
            if len(line) < max_word_length:
                new_lines.append(line)
            else:  # 单行就已经爆限制长度的比较特别
                n = max_word_length - 10  # 将这个长文本按照n的长度再拆分成多个片段加入new_lines
                parts = [line[i:i + n] for i in range(0, len(line), n)]
                new_lines += parts

        # 2.2 拼接new_lines
        fragments = []
        current_fragment = []
        current_fragment_total_length = 0
        for line in new_lines:
            if current_fragment_total_length + len(line) <= max_word_length:
                current_fragment.append(line)
                current_fragment_total_length += len(line)
            else:
                fragments.append('\n'.join(current_fragment))
                current_fragment = [line]
                current_fragment_total_length = len(line)
        if current_fragment:
            fragments.append('\n'.join(current_fragment))

        n = len(fragments)
        fragments = [gen_prompt(n, i, x).strip() for i, x in enumerate(fragments)]

        for i, fragment in enumerate(fragments):
            fragment = {"content": fragment}
            fragments[i] = fragment
        return fragments

    def split_texts(self, texts, max_word_length=None, prompt=None):
        """ 长对话自动拆分成多轮对话 """
        new_texts = []
        for text in texts:
            pure_text = text['content']
            new_texts += self.split_and_add_prompt(pure_text, max_word_length=max_word_length, prompt=prompt)
            if 'file_paths' in text:  # 如果有文件，自动放在最后一轮插入
                new_texts[-1]['file_paths'] = text['file_paths']
        return new_texts

    def add_record(self, texts, *, extra=None,
                   record_id=0, max_word_length=None, prompt=None):
        """
        :param texts:
            str -> list[str]，可以只输入一个str，默认一轮对话
            list[str] -> list[{'content': ..., 'file_paths': [...]}]
                content: 文本内容
                file_paths: 注意可以设置本地电脑其他来源，会自动移到该任务的upload_files里
        :param record_id: 可以自定义这个session的id
        :param max_word_length: 是否设置一个约束长度，自动切分会话中太长的消息
            gpt4是8192个token，大概len就是8192/0.6=13653，一般建议如果要设就设10000左右
        :param prompt: 自动分段后
            None，自动配置的一套提示
            '', 不用提示
        :return:
        """
        # 1 变成标准的list + 字典结构，方便后面统一处理
        if not isinstance(texts, list):
            texts = [texts]

        for i, text in enumerate(texts):
            if isinstance(text, str):
                texts[i] = {'content': text}

        # 2 如果设置了每次最大会话长度，要进行拆分
        if max_word_length:
            texts = self.split_texts(texts, max_word_length=max_word_length, prompt=prompt)

        for i, text in enumerate(texts):
            texts[i]['content'] = text['content'].strip()

        # 3 添加会话conversation
        self.start_id += 1
        item = {'id': str(record_id or self.start_id),  # 要转成字符串类型，不然容易出问题
                'text': texts,
                'first_text_length': len(texts[0]['content'])}
        if extra:
            item['extra'] = extra
        self.records.append(item)
        return item

    def fix_file_paths(self, save_dir):
        """ 修正records中设置的file_paths

        这些路径可能在设置的时候图方便，设置的是非项目目录下的路径
        这个函数会对这些路径进行修正，为了修正，需要输入一个该jsonl所保存的目录位置
        """
        save_dir = XlPath(save_dir)
        for i, record in tqdm(enumerate(self.records), desc='修复文件路径'):
            dst_dir = save_dir / 'upload_files' / str(record['id'])
            for j, text in enumerate(record['text']):
                for k, fn in enumerate(text.get('file_paths', [])):
                    src_file = XlPath(fn)
                    src_file2 = src_file.as_posix()
                    if src_file2.startswith(f'upload_files/{record["id"]}/'):
                        continue
                    dst_file = dst_dir / src_file.name
                    dst_file2 = dst_file.relpath(save_dir).as_posix()
                    if src_file.is_file():
                        if src_file2 != dst_file2:
                            dst_dir.mkdir(parents=True, exist_ok=True)
                            src_file.copy(dst_file, if_exists='replace')
                    else:  # 既然设置了，原文件目录应该在
                        raise FileNotFoundError(f'{src_file}')
                    text['file_paths'][k] = dst_file2

    def clean_file_paths(self):
        """ 清除records中的file_paths
        一般用于把一些相关文件移到对应会话后，实际提问gpt的时候并不上传文件
        """
        for x in self.records:
            for t in x['text']:
                if 'file_paths' in t:
                    del t['file_paths']

    def find_indices_by_qlength(self):
        """ 返回提问(q,question)内容从短到长的数据下标 """
        lens = [(i, len(''.join([t['content'] for t in x['text']]))) for i, x in enumerate(self.records)]
        # 根据长度进行排序，得到的元组的第一个元素为原列表的下标，第二个元素为对应的长度
        sorted_lens = sorted(lens, key=lambda x: x[1])
        # 取出排序后的下标
        sorted_indexs = [i for i, _ in sorted_lens]
        return sorted_indexs

    def browse_record(self, index=None, paths=None, **kwargs):
        """ 检查第i次会话的内容
        """
        # 如果未提供索引，则尝试使用查询参数找到第一个匹配的记录
        if index is None:
            index = self.find_index(paths, **kwargs)
            if index is None:
                raise ValueError('No matching record found')
        session = self.records[index]

        # 构建HTML内容
        html_content = "<html><body>"

        # 输出除了text和all_answers以外的所有键值信息
        html_content += "<h2>会话信息：</h2>"
        html_content += "<ul>"
        for key, value in session.items():
            if key not in ["text", "all_answers"]:
                html_content += f"<li>{html.escape(key)}: {html.escape(str(value))}</li>"
        html_content += "</ul>"

        # 输出text和all_answers的内容
        texts = self.get_text_texts(session.get("text", []))
        all_answers = self.get_all_answers_texts(session.get("all_answers", []))

        max_length = max(len(texts), len(all_answers))
        for idx in range(max_length):
            html_content += f"<h3>第{idx + 1}次询问：</h3>"
            if idx < len(texts):
                html_content += f"<pre>{html.escape(texts[idx])}</pre>"
            if idx < len(all_answers):
                html_content += f"<h3>第{idx + 1}次回答：</h3>"
                html_content += f"<pre>{html.escape(str(all_answers[idx]))}</pre>"

        html_content += "</body></html>"
        html_file = (XlPath.tempdir() / (str(session.get('id', index)) + '.html'))
        html_file.write_text(html_content)
        browser.html(html_file)

        # 返回HTML字符串
        return html_content

    def get_text_texts(self, text):
        """ 从text字段获得所有的文本内容
        因为里面可能是dict
        """
        ls = []
        for t in text:
            if isinstance(t, str):
                ls.append(t)
            else:
                if "file_path" in t:
                    ls.append(("filep_path=" + str(t["file_path"]) + "\n\n") + t["content"])
                else:
                    ls.append(t["content"])
        return ls

    def get_all_answers_texts(self, all_answers):
        ls = []
        for t in all_answers:
            if isinstance(t, dict):
                t = json.dumps(t, ensure_ascii=False, indent=2)
            ls.append(str(t))
        return ls

    def check(self):
        """ 检查会话、消息长度等信息 """
        # 1 提问的内容
        all_texts = [self.get_text_texts(session.get('text', []))
                     for session in self.records]
        print('【提问的内容】')
        check_conversation_lengths(all_texts,
                                   compute_tokens=True,
                                   ids=[x['id'] for x in self.records])

        # 2 回复的内容
        all_texts = [self.get_all_answers_texts(session.get('all_answers', []))
                     for session in self.records]
        # 过滤空值，并相应地更新ids
        filtered_texts = [(text, session['id']) for text, session in zip(all_texts, self.records) if text]
        all_texts, ids = zip(*filtered_texts) if filtered_texts else ([], [])
        if all_texts:
            print('【回复的内容】')
            check_conversation_lengths(all_texts,
                                       compute_tokens=True,
                                       ids=ids)

    def filter_records_without_answers(self):
        """ 过滤掉没有 'all_answers' 字段的sessions """

        # 输出过滤前的sessions数量
        print(f"过滤前的sessions数量：{len(self.records)}")

        # 使用列表推导式过滤出包含 'all_answers' 字段的sessions
        self.records = [s for s in self.records
                        if (''.join(map(str, s.get('all_answers', []))))]

        # 输出过滤后的sessions数量
        print(f"过滤后的sessions数量：{len(self.records)}")

    @classmethod
    def _parse_single_record_answer_contents(cls, record):
        """ 注意本函数不做record备份 """
        for answer in record.get('all_answers', []):
            if isinstance(answer, dict) and 'contents' in answer:
                n = len(answer['contents'])
                for i in range(n - 1, -1, -1):
                    message = answer['contents'][i]['message']
                    if message and 'content' in message and 'error' not in message:
                        break
                else:
                    answer['contents'] = ''
                    continue

                content = message['content']
                if 'parts' in content:
                    content = '\n'.join(content['parts'])
                else:
                    content = content['text']
                answer['contents'] = content

    @classmethod
    def _parse_single_record_answer_downloads(cls, record):
        for answer in record.get('all_answers', []):
            if 'downloads' in answer:
                for i, link in enumerate(answer['downloads']):
                    m = re.search(r'filename%3D(.+?)&sig=', link)
                    if m:
                        answer['downloads'][i] = unquote(unquote(m.group(1)))

    @classmethod
    def parse_single_record_answer(cls, record):
        cls._parse_single_record_answer_contents(record)
        cls._parse_single_record_answer_downloads(record)

    def parse_answer_contents(self):
        """ 简化解释器返回结果中，contents的结构信息 """
        for record in self.records:
            self._parse_single_record_answer_contents(record)

    def parse_answer_downloads(self):
        """ 解析，简化下载链接的表达形式 """
        for record in self.records:
            self._parse_single_record_answer_downloads(record)

        # 目录里的文件名也同理做精简
        for f in self.infile.parent.glob_files():
            if f.name.startswith('OpenAI-download-'):
                f.rename2(f.parent / re.sub(r'OpenAI-download-\d+-', '', f.name),
                          if_exists='replace')

    def filter_to_rechat(self, check_func, rechat_path=None):
        """ 筛选失败的数据到一个新的目录，常用于对chatted数据筛选出未成功的样例，上池子重跑
        这个不是简单的找出得不到all_answers的，而是可以很精细，包含复杂post、verify的情况

        :param check_func: 一个函数，接收一个record，返回True或False
            True，表示这个record是对的
            False，表示这个record是错的，要挑选出来
        :param rechat_path: 把挑选出来的数据放到新路径
        """
        if rechat_path is None:
            rechat_path = XlPath(self.infile.parent.as_posix() + '_rechat/in.jsonl')

        rechat_path = XlPath(rechat_path)
        td = TwinDirs(self.infile.parent, rechat_path.parent)

        gcj = type(self)()
        for record in self.records:
            if not check_func(record):
                record2 = {}
                for k in ['id', 'text', 'first_text_length', 'extra']:
                    record2[k] = record[k]
                gcj.records.append(record2)
                for x in record['text']:
                    if 'file_path' in x:
                        td.copy_file(td.src_dir / x['file_path'])

        gcj.save(rechat_path)
        return gcj

    def update_from_rechat(self, check_func, rechat_path=None):
        """ 从另一份rechat的数据，更新回主master数据

        :param check_func: 原chatted没过，但是rechatted通过的，需要把数据更新过来
        :param rechat_path: 注意只能传路径，因为可能涉及到文件操作，需要知道目录所在
            依据这个文件里的record记录更新回self
        """
        if rechat_path is None:
            rechat_path = XlPath(self.infile.parent.as_posix() + '_rechat') / 'out.jsonl'

        rechat_path = XlPath(rechat_path)
        td = TwinDirs(rechat_path.parent, self.infile.parent)

        id2index = {x['id']: i for i, x in enumerate(self.records)}

        gcj = type(self)(rechat_path)
        gcj.parse_answer_contents()
        gcj.parse_answer_downloads()

        # 需要处理下下载链接名称
        self.parse_answer_downloads()
        gcj.parse_answer_downloads()

        for y in gcj.records:
            index = id2index[y['id']]
            x = self.records[index]
            if not check_func(x) and check_func(y):
                # 先把x相关的数据删掉
                if 'all_answers' in x:
                    for answer in x['all_answers']:
                        for fname in answer.get('downloads', []):
                            (XlPath(self.infile.parent) / fname).delete()
                # 再把y拷贝过来
                for answer in y['all_answers']:
                    for fname in answer.get('downloads', []):
                        td.copy_file(td.src_dir / fname)
                self.records[index] = y
        return gcj


GptQuestionJsonl = GptChatJsonl  # 名称向下兼容


def __2_数据后处理():
    """ 一些常用的文本、后处理功能也放到这里 """


def try_eval_json(resp_json):
    try:
        resp_json = ast.literal_eval(resp_json)
        if isinstance(resp_json, dict):
            resp_json = resp_json[resp_json.keys()[0]]
    except:
        pass
    return resp_json


def try_load_json(resp_json):
    if isinstance(resp_json, str):
        try:
            resp_json = json.loads(resp_json)
            if isinstance(resp_json, dict):
                resp_json = resp_json[resp_json.keys()[0]]
        except:
            pass
    return resp_json


def try_parse_json(resp_json):
    if isinstance(resp_json, dict):
        try:
            resp_json = '\n'.join(resp_json['contents'][-1]['message']['content'].get('parts', []))
        except TypeError:
            return ''

    resp_json = try_eval_json(resp_json)
    if isinstance(resp_json, str):
        return try_load_json(resp_json)
    return resp_json


def extract_code_blocks_from_md(markdown_text, *, sort_by_length=False):
    """ 可以输入str，也可以输入list[str]

    :param sort_by_length: 按代码长度从短到长排序
        常用在比较确信有效代码段应该只有一段，但是有些短小的片段有干扰
        此时可以排序后，选取最长的一个代码片段作为正确代码
    """
    if isinstance(markdown_text, str):
        markdown_text = [markdown_text]

    matches = []
    pattern = re.compile(r'^```[^\n]*\n(.+?)\n^```', re.MULTILINE | re.DOTALL)
    for text in markdown_text:
        matches += pattern.findall(text)

    if sort_by_length:
        matches = sorted(matches, key=len)

    return matches


def extract_airscript_code_from_answers(all_answers):
    """ 从多轮回答的最后一次回答中提取求解代码 """
    contents = all_answers[-1]['contents']
    text = contents[-1]['text']
    code_blocks = extract_code_blocks_from_md(text, sort_by_length=True)

    if code_blocks:
        return code_blocks[-1]
    else:
        return ''


def merge_answers_contents(answers):
    """ 对一组answers结果中，相同type的contents进行合并 """
    for answer in answers:
        contents = []
        for content in answer['contents']:
            if len(contents) == 0:
                contents.append(content)
            else:
                if contents[-1]['type'] == content['type']:
                    contents[-1]['text'] += '\n' + content['text']
                else:
                    contents.append(content)
        answer['contents'] = contents


def refine_content_title(content, tag, dst_title=None):
    """ 将内容中的标题描述形式标准化

    :param tag: 原标题相关字符
    :param content: 文本内容
    :param dst_title: 目标标题格式
    :return: 处理后的字符串
    """
    if dst_title is None:
        dst_title = f'<{tag}>'
    content_lines = content.splitlines()
    chars_str = re.compile(tag.replace(':', '[:的]?'))
    chinese_chars = re.compile(r'[\u4e00-\u9fa5]')

    res = []
    for line in content_lines:
        # 使用正则表达式查找匹配的部分
        new_line = chars_str.sub('', line)
        if new_line != line and not chinese_chars.search(new_line):
            res.append(dst_title)
        else:
            # 如果不满足条件，不进行替换
            res.append(line)
    return '\n'.join(res)


def refine_block_name(record, block_names, preproc=None):
    """ 优化模块的标题名，方便后续结构化提取数据

    感觉这个系列解析是比较通用的，就放在标准库中
    """
    # if preproc is None:
    #     def preproc(x):
    #         return x

    for answer in record['all_answers']:
        for content in answer['contents']:
            if content['type'] == 'text':
                text = old_text = content['text']
                if preproc is not None:
                    text = preproc(text)

                for block_name in block_names:
                    text = refine_content_title(text, block_name)
                text = refine_content_title(text, '---', '')
                # 一般不要直接修改原数据，但post里会有备份，所以这里verify可以直接修改了
                # if 'answer' not in curr_record['extra']:
                #     curr_record['extra']['answer'] = []
                # curr_record['extra']['answer'].append(text)
                content['text'] = text
                # 可以借助bc调试
                # bcompare(old_text, text)


def extract_block_content(record, block_name):
    """ 从record的all_answers中，从后往前检索 <block_name> 的内容，
    返回第一个匹配结果，如果找不到则返回空字符串
    """
    for answer in record['all_answers'][::-1]:
        for content in answer['contents'][::-1]:
            if content['type'] == 'text':
                matches = list(re.finditer(rf'^<{block_name}>\n((.|\n)+?)(?=^<.+?>\n)',
                                           content['text'] + '\n<test>\n',  # 末尾补一个<test>，方便对齐
                                           flags=re.MULTILINE))
                if matches:
                    s = matches[-1].group(1).strip()
                    blocks = extract_code_blocks_from_md(s, sort_by_length=True)
                    if blocks:
                        return blocks[-1]
                    if s:
                        return s
    return ''  # 提取不到


def __3_生成最后训练用的数据():
    pass


def texts2train_record(texts):
    """ user和assistant的轮询对话，转为训练集格式 """
    messages = []
    for i, text in enumerate(texts):
        role = 'assistant' if i % 2 else 'user'
        messages.append({'role': role, 'content': text})
    return {'messages': messages}


class GptTrainJsonl(JsonlDataFile):
    """
    record: dict
        messages: list
          dict: role='user', content=...
          dict: role='assistant', content=...
    """

    def analyze_text_length(self):
        # 1 先将数据统计到df
        ls = []
        columns = ['role', 'content']
        for x in self.records:
            for t in x['messages']:
                ls.append([t['role'], t['content']])
        df = pd.DataFrame.from_records(ls, columns=columns)

        # 2 再从df筛选出不同的统计数据
        print('【user和assistant】')
        print_statistics(df['content'])
        print('【user】')
        print_statistics(df[df['role'] == 'user']['content'])
        print('【assistant】')
        print_statistics(df[df['role'] == 'assistant']['content'])

    def check(self):
        """ 检查会话、消息长度等信息 """
        # 1. 提取'user'角色的content
        user_texts = [[message['content']
                       for message in record['messages']
                       if message['role'] == 'user']
                      for record in self.records]
        if not user_texts:
            print('空数据')
            return

        print('【User的内容】')
        check_conversation_lengths(user_texts, compute_tokens=True,
                                   # 因为一般是使用JLineViewer进行查看，跟那个软件对称使用1开始编号
                                   ids=list(range(1, len(user_texts) + 1)))

        # 2. 提取'assistant'角色的content
        assistant_texts = [[message['content']
                            for message in record['messages']
                            if message['role'] == 'assistant']
                           for record in self.records]
        print('【Assistant的内容】')
        check_conversation_lengths(assistant_texts, compute_tokens=True,
                                   ids=list(range(1, len(assistant_texts) + 1)))

        # 3. 将整个record视为一个完整的会话
        full_conversations = [' '.join([message['content'] for message in record['messages']])
                              for record in self.records]
        print('【完整的会话】')
        check_conversation_lengths(full_conversations, compute_tokens=True,
                                   ids=list(range(1, len(full_conversations) + 1)))

    def browse_record(self, index=None, paths=None, **kwargs):
        """ 显示第i次会话的内容 """
        # 如果未提供索引，则尝试使用查询参数找到第一个匹配的记录
        if index is None:
            index = self.find_index(paths, **kwargs)
            if index is None:
                raise ValueError('No matching record found')
        session = self.records[index]

        # 构建HTML内容
        html_content = "<html><body>"

        # 输出除了messages以外的所有键值信息
        html_content += "<h2>会话信息：</h2>"
        html_content += "<ul>"
        for key, value in session.items():
            if key != "messages":
                html_content += f"<li>{html.escape(key)}: {html.escape(str(value))}</li>"
        html_content += "</ul>"

        # 输出messages的内容
        messages = session.get("messages", [])

        for idx, message in enumerate(messages):
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            html_content += f"<h3>第{(idx // 2) + 1}次{role}的发言：</h3>"
            html_content += f"<pre>{html.escape(content)}</pre>"

        html_content += "</body></html>"
        html_file = (XlPath.tempdir() / (f'session_{index}.html'))  # 创建临时文件名，防止覆盖现有文件
        html_file.write_text(html_content)
        browser.html(html_file)  # 在浏览器中打开HTML文件

        # 或者返回HTML字符串
        return html_content

    def add_record(self, texts):
        messages = []
        for i, text in enumerate(texts):
            role = 'assistant' if i % 2 else 'user'
            messages.append({'role': role, 'content': text})
        self.records.append({'messages': messages})

    def add_from_texts(self, texts):
        record = texts2train_record(texts)
        self.records.append(record)


def __4_综合集成类():
    pass


class GptChatDir:
    """ 一个目录，包含了一个任务的所有数据，包括in、out、post等文件 """

    def __init__(self, root=None, lines_per_file=10000):
        if root is None:
            root = self.__class__.__name__.lower()

        self.root = root = XlPath(root)
        self.lines_per_file = lines_per_file

        self.chat_file = root / 'in.jsonl'
        self.chatted_file = root / 'out.jsonl'
        self.post_file = root / 'post.jsonl'
        self.verify_file = root / 'verify.jsonl'
        self.train_file = root / 'train.jsonl'

        # 如果有目录文件，会优先以目录为准。如果没有，则会从单文件拆分创建。
        self.update_dir()

        self.upload_files_dir = root / 'upload_files'
        self.download_files_dir = root / 'download_files'

        # todo 把 1chat 改名 in，2chatted 改名 out
        # for f in self.root.glob_files('*1chat*.jsonl'):
        #     f.rename2(f.parent / 'in.jsonl')

        # for dir_path in [self.root, self.upload_files_dir, self.download_files_dir]:
        for dir_path in [self.root]:
            if not dir_path.is_dir():
                dir_path.mkdir(parents=True, exist_ok=True)

        # 这个类经常要并发处理，不能把一个不能序列化的类放到这里~
        # self.logger = OutputLogger(log_file=self.root / 'log.txt')

    def update_dir(self):
        """ 目录结构有些更新后，一些成员变量要跟着改变 """
        # 如果有目录文件，会优先以目录为准。如果没有，则会从单文件拆分创建。
        self.chat_dir = JsonlDataDir.init_from_file(self.chat_file, self.lines_per_file)
        self.chatted_dir = JsonlDataDir.init_from_file(self.chatted_file, self.lines_per_file)
        self.post_dir = JsonlDataDir.init_from_file(self.post_file, self.lines_per_file)
        self.verify_dir = JsonlDataDir.init_from_file(self.verify_file, self.lines_per_file)
        self.train_dir = JsonlDataDir.init_from_file(self.train_file, self.lines_per_file)

    def summary_records(self):
        """ 一些统计信息 """
        # 1 chat信息
        gcd1 = self.chatted_dir or self.chat_dir
        if not gcd1:
            print('请确认是否有生成初始的chat数据')
            return

        print(f'【{self.root.name}】')
        texts = [len(x['text']) for x in gcd1.yield_record()]
        n, m = len(texts), sum(texts)
        print(f'1、chat：{n}条会话*{m / n:.2g}条消息')
        gcj1 = GptChatJsonl(gcd1.files[0])  # 统计一个文件就够了，不然太多了
        gcj1.check_records()
        print()

        # 2 chatted信息
        filter_records = [x for x in gcd1.yield_record() if 'all_answers' in x]
        if filter_records:
            print(f'2、chatted：已获得{len(filter_records)}条会话')
        else:
            print('2、chatted：暂未获得生成数据')

        # 3 post信息
        if self.post_dir:
            print(f'3、post：{self.post_dir.count_records()}条会话')

        # 4 verify（这一步有时候会集成到post中）
        if self.verify_dir:
            print(f'4、verify：{self.verify_dir.count_records()}条会话')

        # 5 train 生成的训练数据
        # print('5、train：')
        # gtj = GptTrainJsonl(self.train_file)
        # gtj.analyze_text_length()

    def summary_downloads(self):
        """ 统计下载的文件情况 """
        print('【每个目录文件数量】')
        files_each_dir = []
        for d in self.download_files_dir.glob_dirs():
            files_each_dir.append(len(list(d.rglob_files())))
        print(ValuesStat(files_each_dir).summary())
        print(Counter(files_each_dir))

        print('【每个文件大小】')
        filesizes_each_dir = []
        for d in self.download_files_dir.glob_dirs():
            for f in d.rglob_files():
                filesizes_each_dir.append(f.size())
        print(ValuesStat(filesizes_each_dir).summary())

    def create_chat(self):
        """ 生成chat数据，具体内容方式跟业务有关 """
        raise NotImplementedError

    def browse_chatted_record(self, index=None, paths=None, **kwargs):
        """ 显示第i次会话的内容 """
        f = self.chatted_file if self.chatted_file.is_file() else self.chat_file
        return GptChatJsonl(f, 100).browse_record(index, paths, **kwargs)

    def chatted2post_record(self, chatted_record):
        """ 后处理，解析

        一般会保留基本的all_answers结果，供检查上游一些基本情况
        然后把一些结构化结果存储到extra字段

        :return: 会返回新的dict结构的一个post_record，如果解析失败，会返回None
        """
        # 0 基本情况判断
        if 'all_answers' not in chatted_record:
            return

        post_record = copy.deepcopy(chatted_record)

        # 1 删掉一些没卵用的字段
        for name in ['all_questions', 'first_text_length', 'second_text_length']:
            if name in post_record:
                del post_record[name]

        # 2 解析all_answers：这个结构太复杂，进行内容整理精简
        # 2.1 contents：这个结构太复杂，搁这俄罗斯套娃呢~ 稍微精简下更方便后处理
        for k, answer in enumerate(post_record['all_answers']):
            if isinstance(answer, dict) and 'contents' in answer:
                new_contents = []
                for i, x in enumerate(answer['contents']):
                    if not x['message']:
                        # Error in message stream
                        # print(f'{post_record["id"]} answer[{k}] contents[{i}] message为空')
                        continue

                    content = x['message']['content']
                    tp = content['content_type']
                    new_content = {'type': content['content_type']}
                    if tp == 'text':
                        new_content['text'] = '\n'.join(content['parts'])
                    elif tp == 'code':
                        new_content['text'] = content['text']
                    elif tp == 'execution_output':
                        new_content['text'] = content['text']
                    elif tp == 'system_error':
                        continue
                    else:
                        print(f'{post_record["id"]} answer[{k}] contents[{i}] content_type={tp} 未见类型')
                        continue

                    new_contents.append(new_content)
                answer['contents'] = new_contents
            elif isinstance(answer, str):  # 普通模式也转成解释器风格，方便统一处理
                post_record['all_answers'][k] = {'contents': [{'type': 'text',
                                                               'text': answer}]}

        # 2.2 downloads：下载链接精简下，并把关联的文件也顺带整理一下
        for answer in post_record['all_answers']:
            if 'downloads' not in answer:
                continue
            for i, link in enumerate(answer['downloads']):
                m = re.search(r'filename%3D(.+?)&sig=', link)
                if m:
                    answer['downloads'][i] = str(post_record['id']) + '/' + unquote(unquote(m.group(1)))
                # 对应的文件不存在的不要，有数据超过50M的也不要
                file = self.download_files_dir / link
                if not file.exists() and file.size() > 50 * 1024 * 1024:
                    return

            # 理论上下载的文件不应该有重复，虽然不知道为什么会拿到重复，但去掉重复比较好
            answer['downloads'] = list(OrderedDict.fromkeys(answer['downloads']))

        # 2.3 删掉answer里其他没用的字段
        for answer in post_record['all_answers']:
            for name in ['created', 'message_id', 'conversation_id', 'end_turn']:
                if name in answer:
                    del answer[name]

        # 返回处理结果
        return post_record

    @staticmethod
    def post2verify_record(post_record):
        """ 这个一般是要具体任务定制的，没有通用操作方式

        注意，如果要使用create_verify的多进程功能，这个函数必须是静态的，并且里面也不能使用其他"类静态方法"
            否则写成类方法或对象方法都可以

        """
        raise NotImplementedError

    def verify2train_record(self, verify_record):
        """ 这个一般是要具体任务定制的，没有通用操作方式 """
        raise NotImplementedError

    def organize_downloaded_files(self):
        # 把下载的文件整理的更清晰些
        for f in tqdm(list(self.root.glob_files('OpenAI-download-*')),
                      desc='整理下载的文件'):
            new_name = re.sub(r'OpenAI-download-\d+-', '', f.name)
            new_name = new_name.replace('-', '/', 1)
            try:
                (self.download_files_dir / new_name).parent.mkdir(exist_ok=True)
                f.rename2(self.download_files_dir / new_name, if_exists='replace')
            except FileExistsError as e:
                # 有的文件会移动不了
                print(e)

        # 会剩一些特殊的处理不了的文件，可以看一眼后手动删掉
        # 这些相关的records，默认的chatted2post_record会把这些记录过滤掉

    def create_post(self, **kwargs):
        """ 建议初步跑的时候，先串行debug，等比较稳定后，再开并发跑
        """
        if 'dst_dir' not in kwargs:
            kwargs['dst_dir'] = self.post_dir.root
        self.chatted_dir.process_each_record(self.chatted2post_record, **kwargs)
        self.post_dir.update_subfiles()
        num1, num2 = self.chatted_dir.count_records(), self.post_dir.count_records()
        print(f'chatted有{num1}条，转换post有{num2}条，转换率{num2 / num1:.2%}')

    def create_verify(self, **kwargs):
        """ 有时候create_verify是有cpu密集运算场景的，可以开多进程
        """
        if 'dst_dir' not in kwargs:
            kwargs['dst_dir'] = self.verify_dir.root
        self.post_dir.process_each_record(self.post2verify_record, **kwargs)
        self.verify_dir.update_subfiles()
        num1, num2 = self.post_dir.count_records(), self.verify_dir.count_records()
        num1 = num1 or -1
        print(f'post有{num1}条，转换verify有{num2}条，转换率{num2 / num1:.2%}')

    def refine_verify(self, print_mode=1, **kwargs):
        """ 重复检查verify数据

        这个函数可以重复执行，但前提是self.post2verify_record里的设计有增量规则部分
        """
        self.verify_dir.process_each_record(self.post2verify_record, print_mode=print_mode,
                                            inplace=True, desc='refine_verify', **kwargs)

    @classmethod
    def texts2train_record(cls, texts):
        """ user和assistant的轮询对话，转为训练集格式 """
        messages = []
        for i, text in enumerate(texts):
            role = 'assistant' if i % 2 else 'user'
            messages.append({'role': role, 'content': text})
        return {'messages': messages}

    def create_train(self, **kwargs):
        if 'dst_dir' not in kwargs:
            kwargs['dst_dir'] = self.train_dir.root
        self.post_dir.process_each_record(self.verify2train_record, **kwargs)
        self.train_dir.update_subfiles()

    def check_chatted_record(self, chatted_record):
        """ 检查chatted数据的有效性 """
        x = chatted_record
        x = self.chatted2post_record(x)
        # x = self.post2verify_record(x)
        # 针对verify可以再进一步定制规则
        return bool(x)

    def create_rechat(self, rechat_path):
        """ 筛选失败的数据到一个新的目录，常用于对chatted数据筛选出未成功的样例，上池子重跑

        :param rechat_path: 把挑选出来的数据放到新路径
        """
        gcd = GptChatDir(rechat_path)
        f = open(gcd.chat_file, 'w', encoding='utf-8')

        for record in tqdm(self.chatted_dir.yield_record(), '检查待重新生成的问题'):
            if not self.check_chatted_record(record):
                continue
            # 否则把这个条目放到rechat，准备拿去重新提问
            if 'error' in record:
                del record['error']
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            # 如果有文件，也要对应移动
            src_dir = self.upload_files_dir / str(record['id'])
            if src_dir.is_dir():
                src_dir.copy(gcd.upload_files_dir / src_dir.name, if_exists='skip')

        f.close()
        return gcd

    def update_chatted(self, rechat_path):
        """ 从另一个rechat数据，更新数据条目过来

        self依然叫src，rechat叫dst，虽然其实数据是从rechat更新流向self

        注意：这个函数还没有比较严格地进行调试~
        """
        # 1 读取有效记录
        gcd = GptChatDir(rechat_path)
        gcd.organize_downloaded_files()
        # 请确保内存充足哦，这个函数会从rechat的chatted读取所有通过的记录保存起来
        dst_records = {}
        for record in gcd.chatted_dir.yield_record():
            # 找到有all_answers的挑出来
            post_record = self.chatted2post_record(record)
            if post_record:
                dst_records[record['id']] = record

        # 2 更新记录
        def update_each_record(x):
            if x['id'] in dst_records:
                # 除了返回record，还得拷贝目录数据呢
                # 上传的目录一般没变，但最好重置下
                src_dir = self.upload_files_dir / x['id']
                dst_dir = gcd.upload_files_dir / x['id']
                dst_dir.copy(src_dir, if_exists='replace')
                # 下载的目录
                src_dir = self.download_files_dir / x['id']
                dst_dir = gcd.download_files_dir / x['id']
                dst_dir.copy(src_dir, if_exists='replace')
                return dst_records[x['id']]
            else:
                return x

        self.chatted_dir.update_each_record(update_each_record)


def __5_bdchat():
    """ 百度相关api """


class BaiduChatbot:
    def __init__(self, api_key, secret_key, file_path=None):
        self.API_KEY = api_key
        self.SECRET_KEY = secret_key
        self.ACCESS_TOKEN = self._get_access_token()
        self.base_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token="
        self.file_path = file_path  # 文件路径为可选参数

    def _get_access_token(self):
        """
        使用 AK，SK 生成鉴权签名（Access Token）
        :return: access_token，或是None(如果错误)
        """
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.API_KEY,
            "client_secret": self.SECRET_KEY
        }
        return str(requests.post(url, params=params).json().get("access_token"))

    def chat(self, user_message):
        """ 向Baidu API发送用户消息并返回API的回复
        注意user_message的token不要超过3k
        """
        url = self.base_url + self.ACCESS_TOKEN
        payload = json.dumps({
            "messages": [{"role": "user", "content": user_message}]
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=payload)
        response_json = response.json()
        response_json['user_message'] = user_message
        response_json['timestamp'] = datetime.datetime.now().isoformat()

        # 如果指定了文件路径，自动保存记录
        if self.file_path:
            self._save_to_file(response_json)

        return response_json.get('result', '')

    def _save_to_file(self, response):
        with open(self.file_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(response, ensure_ascii=False) + '\n')
