#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2023/07/13 14:26
import copy

from pyxllib.prog.pupil import check_install_package
from joblib import Parallel, delayed

check_install_package('transformers', 'transformers')

import ast
import json
import re
import html
import random
from urllib.parse import unquote
from collections import OrderedDict

import pandas as pd
from transformers import GPT2TokenizerFast
from tqdm import tqdm
from openpyxl import Workbook

from pyxllib.prog.specialist import browser, TicToc
from pyxllib.algo.pupil import ValuesStat
from pyxllib.file.specialist import XlPath, JsonlDataFile, TwinDirs


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


def print_statistics(data, indent_level=1, price_base=0.0015, max_samples=500):
    """
    :param price_base: 每1K token对应的美元单价
    """
    data = list(data)
    for i, x in enumerate(data):
        if isinstance(x, dict):
            x = x.get('content', str(x))
        data[i] = html.unescape(x)

    # 使用 ValuesStat 类统计数据并输出摘要
    # stat_len = ValuesStat([len(x) for x in data])
    # stat_strwith = ValuesStat([strwidth(x) for x in data])
    # 算token的机制很慢，只能抽查一部分估算
    samples = random.sample(data, min(len(data), max_samples))
    stat_tokens = ValuesStat([Tokenizer.count_tokens(x) for x in samples])

    fmts = ['g', '.0f', '.0f', 'd', 'd']

    indent = "\t" * indent_level
    # print(f"{indent}     len {stat_len.summary(fmts)}")
    # print(f"{indent}strwidth {stat_strwith.summary(fmts)}")
    # 官方gpt3.5价格，/1000是除1K token，*7.1388是美元兑换人民币基本价格（浮动，不定期更新）
    price = stat_tokens.mean * len(data) / 1000 * price_base * 7.1388
    print(f"{indent}  tokens {stat_tokens.summary(fmts)} gpt3_price=￥{price:.0f}")


class GptChatJsonl(JsonlDataFile):
    """ GPT问答批量执行脚本的jsonl生成、读取器 """

    def __init__(self, file=None, num_records=None, *, start_id=None):
        from datetime import date

        super().__init__(file, num_records)
        if start_id is None:
            today = date.today().strftime("%Y%m%d")
            self.start_id = int(today + "00000000")
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
        new_texts = []
        for text in texts:
            pure_text = text['content']
            new_texts += self.split_and_add_prompt(pure_text, max_word_length=max_word_length, prompt=prompt)
            if 'file_path' in text:  # 如果有文件，自动放在最后一轮插入
                new_texts[-1]['file_path'] = XlPath(text['file_path']).name
        return new_texts

    def add_record(self, texts, *, extra=None,
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
        # 1 变成标准的list + 字典结构，方便后面统一处理
        if isinstance(texts, str):
            texts = [texts]

        for i, text in enumerate(texts):
            if isinstance(text, str):
                texts[i] = {'content': text}

        # 2 如果设置了每次最大会话长度，要进行拆分
        if max_word_length:
            texts = self.split_texts(texts, max_word_length=max_word_length, prompt=prompt)

        self.start_id += 1

        for i, text in enumerate(texts):
            texts[i]['content'] = text['content'].strip()

        # 3 添加会话conversation
        item = {'id': record_id or self.start_id,
                'text': texts,
                'first_text_length': len(texts[0]['content'])}
        if extra:
            item['extra'] = extra
        self.records.append(item)
        return item

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

    def check_records(self):
        # 单次 QA 的长度信息
        qa_texts = []
        qa_answers = []
        for session in self.records:
            texts = self.get_text_texts(session.get("text", []))
            all_answers = self.get_all_answers_texts(session.get("all_answers", []))
            qa_texts.extend(texts)
            qa_answers.extend(all_answers)

        print("消息messages长度统计信息:")
        print('\t提问', end='\t')
        print_statistics(qa_texts)
        if qa_answers:
            print('\t回答', end='\t')
            print_statistics(qa_answers, price_base=0.002)

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

    def add_record(self, texts):
        messages = []
        for i, text in enumerate(texts):
            role = 'assistant' if i % 2 else 'user'
            messages.append({'role': role, 'content': text})
        self.records.append({'messages': messages})


def __4_综合集成类():
    pass


class GptChatDir:
    """ 一个目录，包含了一个任务的所有数据，包括in、out、post等文件 """

    def __init__(self, root):
        self.root = root = XlPath(root)

        self.chat_file = root / 'in.jsonl'
        self.chatted_file = root / 'out.jsonl'
        self.post_file = root / 'post.jsonl'
        self.verify_file = root / 'verify.jsonl'
        self.train_file = root / 'train.jsonl'

        self.upload_files_dir = root / 'upload_files'
        self.download_files_dir = root / 'download_files'

        # 把 1chat 改名 in，2chatted 改名 out
        # for f in self.root.glob_files('*1chat*.jsonl'):
        #     f.rename2(f.parent / 'in.jsonl')

        for dir_path in [self.root, self.upload_files_dir, self.download_files_dir]:
            if not dir_path.is_dir():
                dir_path.mkdir(parents=True, exist_ok=True)

    def summary(self):
        """ 一些统计信息 """
        # 1 chat信息
        if self.chatted_file.is_file():
            gcj1 = GptChatJsonl(self.chatted_file)
        elif self.chat_file.is_file():
            gcj1 = GptChatJsonl(self.chat_file)
        else:
            print('请确认是否有生成初始的chat数据')
            return

        print(f'【{self.root.name}】')
        n = len(gcj1.records)
        m = sum([len(x['text']) for x in gcj1.records])
        print(f'1、chat：{n}条会话*{m / n:.2g}条消息')
        gcj1.check_records()
        print()

        # 2 chatted信息
        filter_records = [x for x in gcj1.records if 'all_answers' in x]
        if filter_records:
            print(f'2、chatted：已获得{len(filter_records)}条会话')
        else:
            print('2、chatted：暂未获得生成数据')

        # 3 post信息
        if filter_records and not self.post_file.is_file():  # 待生成后处理文件
            gcj2 = GptChatJsonl()
            gcj2.infile = self.post_file
            gcj2.records = filter_records
            gcj2.parse_answer_contents()
            gcj2.parse_answer_downloads()
            gcj2.save()
        elif self.post_file.is_file():  # 已经有后处理文件
            gcj2 = GptChatJsonl(self.post_file)
        else:
            return

        print(f'3、post：{len(gcj2.records)}条会话')

        # 4 verify（这一步有时候会集成到post中）

        # 5 train 生成的训练数据
        print('5、train：')
        gtj = GptTrainJsonl(self.train_file)
        gtj.analyze_text_length()

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
        for answer in post_record['all_answers']:
            if isinstance(answer, dict) and 'contents' in answer:
                new_contents = []
                for i, x in enumerate(answer['contents']):
                    if not x['message']:
                        continue
                    content = x['message']['content']
                    new_content = {'type': content['content_type']}
                    if content['content_type'] == 'text':
                        new_content['text'] = '\n'.join(content['parts'])
                    elif content['content_type'] == 'code':
                        new_content['text'] = content['text']
                    elif content['content_type'] == 'execution_output':
                        new_content['text'] = content['text']
                    else:
                        raise ValueError('未见类型')

                    new_contents.append(new_content)
                answer['contents'] = new_contents

        # 2.2 downloads：下载链接精简下，并把关联的文件也顺带整理一下
        for answer in post_record['all_answers']:
            if 'downloads' not in answer:
                continue
            for i, link in enumerate(answer['downloads']):
                m = re.search(r'filename%3D(.+?)&sig=', link)
                if m:
                    answer['downloads'][i] = str(post_record['id']) + '-' + unquote(unquote(m.group(1)))
            # 理论上下载的文件不应该有重复，虽然不知道为什么会拿到重复，但去掉重复比较好
            answer['downloads'] = list(OrderedDict.fromkeys(answer['downloads']))

        # 2.3 删掉answer里其他没用的字段
        for answer in post_record['all_answers']:
            for name in ['created', 'message_id', 'conversation_id', 'end_turn']:
                if name in answer:
                    del answer[name]

        # 返回处理结果
        return post_record

    def create_post(self, n_jobs=1):
        """ 建议初步跑的时候，先串行debug，等比较稳定后，再开并发跑 """
        # 1 把下载的文件整理的更清晰些
        with TicToc('整理下载的文件'):
            for f in self.root.glob_files('OpenAI-download-*'):
                new_name = re.sub(r'OpenAI-download-\d+-', '', f.name)
                # gpt同一次会话是有重名文件的，如果使用replace请慎重
                f.rename2(self.download_files_dir / new_name, if_exists='replace')

        # 2 把chatted数据解析为post格式
        with TicToc('读取chatted数据'):
            gcj1 = GptChatJsonl(self.chatted_file)
            gcj2 = GptChatJsonl()

        for x in tqdm(gcj1.records):
            y = self.chatted2post_record(x)

            if y:
                gcj2.records.append(y)

        # def func(x):
        #     y = self.chatted2post_record(x)
        #     if y:
        #         gcj2.records.append(y)
        #
        # pl = Parallel(n_jobs=n_jobs, backend='threading', timeout=5)
        # pl(delayed(func)(x) for x in tqdm(gcj1.records))

        # 3 保存后处理文件
        gcj2.save(self.post_file)

    def create_train(self):
        raise NotImplementedError
