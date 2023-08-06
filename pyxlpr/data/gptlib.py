#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2023/07/13 14:26
import ast
import json
import re

from pyxllib.prog.pupil import check_install_package

check_install_package('transformers', 'transformers')

import html
import random

import pandas as pd
from transformers import GPT2TokenizerFast

from pyxllib.prog.specialist import browser
from pyxllib.algo.pupil import ValuesStat
from pyxllib.text.pupil import strwidth
from pyxllib.file.specialist import XlPath, JsonlDataFile


def __1_生成提问数据():
    pass


class Tokenizer:
    _tokenizer = None

    @classmethod
    def get_tokenizer(cls):
        """ 获取tokenizer，第一次调用时进行初始化 """

        if Tokenizer._tokenizer is None:
            Tokenizer._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        return Tokenizer._tokenizer

    @classmethod
    def tokenize(cls, paragraph, max_length=500):
        """ 对段落进行tokenize

        :param str paragraph: 待分词的段落
        :param int max_length: 单次处理的最大分词数，为了防止超过GPT2的限制，默认设置为500
        :return list: 分词后的列表

        >>> Tokenizer.tokenize('Hello, world! 汉字 123.14 35')
        ['Hello', ',', 'Ġworld', '!', 'Ġæ', '±', 'ī', 'åŃ', 'Ĺ', 'Ġ123', '.', '14', 'Ġ35']
        """
        tokenizer = Tokenizer.get_tokenizer()

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


def print_statistics(data, indent_level=2, price_base=0.0015):
    """
    :param price_base: 每1K token对应的美元单价
    """
    data = list(data)
    for i, x in enumerate(data):
        if isinstance(x, dict):
            x = x.get('content', str(x))
        data[i] = html.unescape(x)

    # 使用 ValuesStat 类统计数据并输出摘要
    stat_len = ValuesStat([len(x) for x in data])
    stat_strwith = ValuesStat([strwidth(x) for x in data])
    # 算token的机制很慢，只能抽查一部分估算
    samples = random.sample(data, min(len(data), 500))
    stat_tokens = ValuesStat([Tokenizer.count_tokens(x) for x in samples])

    fmts = ['g', '.0f', '.0f', 'd', 'd']

    indent = "\t" * indent_level
    print(f"{indent}     len {stat_len.summary(fmts)}")
    print(f"{indent}strwidth {stat_strwith.summary(fmts)}")
    # 官方gpt3.5价格，/1000是除1K token，*7.1388是美元兑换人民币基本价格（浮动，不定期更新）
    price = stat_tokens.mean * len(data) / 1000 * price_base * 7.1388
    print(f"{indent}  tokens {stat_tokens.summary(fmts)} gpt3_price=￥{price:.0f}")


class GptQuestionJsonl(JsonlDataFile):
    """ GPT问答批量执行脚本的jsonl生成、读取器 """

    def __init__(self, file=None, *, start_id=None):
        from datetime import date
        if start_id is None:
            today = date.today().strftime("%Y%m%d")
            self.start_id = int(today + "00000000")
        else:
            self.start_id = start_id

        super().__init__()
        if file is not None:
            self.read_jsonl(file)

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
        :param text: 要插入的文本
        :param max_word_length: 如果设置了该值，则会对输入内容进行长度控制，拆成多个片段
            之前测试，全英文长度大概在32500，中文在5000内
        :param prompt: max_word_length开启时才会生效，每个part要附加的提示规则
            可以写个字符串，默认每段都是开头加这个提示
            也可以参考gen_prompt2,一些生成函数写法进行自定义
        :return:
        """
        if max_word_length is None:
            return [text]

        def gen_prompt1(n, i, text):
            """ 一共n条，当前第i条的，当前内容是text """
            if n == 1:
                return text
            if n > 1:
                if i == 0:
                    return f'【注意，由于本次提问过长，这里拆分成{n}个片段分开输入，目前是第{1}个片段，你只需暂时回复"收到"即可】' + text
                elif i < n - 1:
                    return f'【这是{n}个片段中的第{i + 1}个片段，先回复"收到"即可】' + text
                else:
                    return f'【这是{n}个片段的最后一个片段，请开始回复内容】' + text

        def gen_prompt2(n, i, text):
            return prompt + text

        if prompt is None:
            gen_prompt = gen_prompt1
        elif isinstance(prompt, str):
            gen_prompt = gen_prompt2
        else:  # callable
            gen_prompt = prompt

        fragments = []
        current_fragment = ''

        lines = text.rstrip().split('\n')
        new_lines = []
        for line in lines:
            if len(line) < max_word_length:
                new_lines.append(line)
            else:
                n = max_word_length - 10
                parts = [line[i:i + n] for i in range(0, len(line), n)]
                new_lines += parts

        for line in new_lines:
            if len(current_fragment) + len(line) <= max_word_length:
                current_fragment += line + '\n'
            else:
                fragments.append(current_fragment)
                current_fragment = line
        if current_fragment:
            fragments.append(current_fragment)

        n = len(fragments)
        fragments = [gen_prompt(n, i, x).rstrip() for i, x in enumerate(fragments)]
        return fragments

    def split_texts(self, texts, max_word_length=None, prompt=None):
        """ 长对话自动拆分成多轮对话 """
        if isinstance(texts, str):
            texts = [texts]
        new_texts = []
        for text in texts:
            new_texts += self.split_and_add_prompt(text, max_word_length=max_word_length, prompt=prompt)
        texts = new_texts
        return texts

    def add_record(self, texts, *, file_path=None,
                   record_id=0, max_word_length=None, prompt=None):
        """
        :param texts:
            可以输入list，原本配置的多轮对话
            也可以只输入一个str，默认一轮对话
        :param record_id: 可以自定义这个session的id
        :param max_word_length: 是否设置一个约束长度，自动切分会话
        :param prompt: 自动分段后
            None，自动配置的一套提示
            '', 不用提示
        :return:
        """
        self.start_id += 1

        if isinstance(texts, str):
            texts = [texts]

        if max_word_length:
            texts = self.split_texts(texts, max_word_length=max_word_length, prompt=prompt)

        texts = [x.strip() for x in texts]
        if file_path:
            t = texts[-1]
            texts[-1] = {'content': t, 'file_path': XlPath(file_path).name}
        else:
            for i, x in enumerate(texts):
                texts[i] = {'content': x}
        item = {'id': record_id or self.start_id,
                'text': texts,
                'first_text_length': len(texts[0])
                if isinstance(texts[0], str) else len(texts[0]['content'])}
        self.records.append(item)
        return item

    def find_indices_by_qlength(self):
        """ 返回提问(q,question)内容从短到长的数据下标 """
        lens = [(i, len(''.join(x['text']))) for i, x in enumerate(self.records)]
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
                    ls.append(("filep_path=" + t["file_path"] + "\n\n") + t["content"])
                else:
                    ls.append(t["content"])
        return ls

    def get_all_answers_texts(self, all_answers):
        ls = []
        for t in all_answers:
            ls.append(str(t))
        return ls

    def check_records(self):
        # 单次 QA 的长度信息
        qa_texts = []
        qa_answers = []
        for session in self.records:
            texts = self.get_text_texts(session.get("text", []))
            all_answers = self.get_all_answers_texts(session.get("all_answers", []))
            qa_texts.extend(texts)
            qa_answers.extend(all_answers)

        print("单次 QA 的长度信息:")
        print('\ttext（提问）')
        print_statistics(qa_texts)
        if qa_answers:
            print('\tall_answers（回答）')
            print_statistics(qa_answers, price_base=0.002)

        # 单次 session 的长度信息
        session_texts = []
        session_answers = []
        for session in self.records:
            texts = self.get_text_texts(session.get("text", []))
            all_answers = self.get_all_answers_texts(session.get("all_answers", []))
            session_texts.append("".join(texts))
            session_answers.append("".join(all_answers))

        print("单次 session 的长度信息(如果只有单轮qa，则统计跟上面是一样的):")
        print('\ttext（提问）')
        print_statistics(session_texts)
        print('\tall_answers（回答）')
        print_statistics(session_answers, price_base=0.002)

    def filter_records_without_answers(self):
        """ 过滤掉没有 'all_answers' 字段的sessions """

        # 输出过滤前的sessions数量
        print(f"过滤前的sessions数量：{len(self.records)}")

        # 使用列表推导式过滤出包含 'all_answers' 字段的sessions
        self.records = [s for s in self.records
                        if (''.join(map(str, s.get('all_answers', []))))]

        # 输出过滤后的sessions数量
        print(f"过滤后的sessions数量：{len(self.records)}")


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


def extract_code_blocks_from_md(markdown_text):
    """ 可以输入str，也可以输入list[str] """
    if isinstance(markdown_text, str):
        markdown_text = [markdown_text]

    matches = []
    pattern = re.compile(r'^```[^\n]*\n(.+?)^```', re.MULTILINE | re.DOTALL)
    for text in markdown_text:
        matches += pattern.findall(text)
    return matches


def __3_生成最后训练用的数据():
    pass


class GptTrainJsonl(JsonlDataFile):
    """
    record: dict
        messages: list
          dict: role='user', content=...
          dict: role='assistant', content=...
    """

    def analyze_text_length(self):
        ls = []
        columns = ['role', 'content']
        for x in self.records:
            for t in x['messages']:
                ls.append([t['role'], t['content']])
        df = pd.DataFrame.from_records(ls, columns=columns)

        print('【user和assistant】')
        print_statistics(df['content'])
        print('【user】')
        print_statistics(df[df['role'] == 'user']['content'])
        print('【assistant】')
        print_statistics(df[df['role'] == 'assistant']['content'])

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
