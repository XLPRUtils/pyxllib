#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2023/07/13 14:26


from pyxllib.prog.pupil import check_install_package
from joblib import Parallel, delayed

check_install_package('transformers', 'transformers')

import ast
import json
import re
import html
import random
import copy
from urllib.parse import unquote
from collections import OrderedDict
from collections import Counter
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

import pandas as pd
from transformers import GPT2TokenizerFast
from tqdm import tqdm
from openpyxl import Workbook
from jinja2 import Template
# import pathos
# import pathos.multiprocessing as multiprocessing

from pyxllib.prog.pupil import OutputLogger
from pyxllib.prog.specialist import browser, TicToc
from pyxllib.algo.pupil import ValuesStat
from pyxllib.file.specialist import XlPath, JsonlDataFile, JsonlDataDir, TwinDirs


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


def set_template(s, *args, **kwargs):
    return Template(s.strip(), *args, **kwargs)


def set_meta_template(s, meta_start='[[', meta_end=']]', **kwargs):
    """ 支持预先用某些格式渲染后，再返回标准渲染模板 """
    t = Template(s.strip(), variable_start_string=meta_start,
                 variable_end_string=meta_end).render(**kwargs)
    return Template(t)


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
            str -> list[str]，可以只输入一个str，默认一轮对话
            list[str] -> list[{'content': ..., 'file_paths': [...]}]
                content: 文本内容
                file_paths: 注意可以设置本地电脑其他来源，会自动移到该任务的upload_files里
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
        for i, record in enumerate(self.records):
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
                        raise FileNotFoundError
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


def extract_code_blocks_from_md(markdown_text, *, sort_by_length=False):
    """ 可以输入str，也可以输入list[str]

    :param sort_by_length: 按代码长度从短到长排序
        常用在比较确信有效代码段应该只有一段，但是有些短小的片段有干扰
        此时可以排序后，选取最长的一个代码片段作为正确代码
    """
    if isinstance(markdown_text, str):
        markdown_text = [markdown_text]

    matches = []
    pattern = re.compile(r'^```[^\n]*\n(.+?)^```', re.MULTILINE | re.DOTALL)
    for text in markdown_text:
        matches += pattern.findall(text)

    if sort_by_length:
        matches = sorted(matches, key=len)

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


def process_file(processing_func, input_file, output_dir=None,
                 thread_num=1, mininterval=None, num_records=None,
                 json_encoder=None):
    """ 处理指定的文件

    :param processing_func: 用于处理记录的函数
    :param input_file: 待处理的输入文件
    :param output_dir: 输出文件的目录
    :param thread_num: 使用的线程数
    :param mininterval: tqdm的更新间隔
    :param num_records: 每个文件最多提取多少条目，用于小批量运行调试
    """
    # 当目标文件已存在时，视为已经处理过，不再重复处理。如果想重跑，可以删掉目标文件即可
    if output_dir is not None:
        dst_file = output_dir / input_file.name
        if dst_file.is_file():
            return

    data_input = JsonlDataFile(input_file, num_records=num_records)
    data_output = JsonlDataFile()

    if thread_num == 1:
        for record in tqdm(data_input.records, total=len(data_input.records),
                           desc=f'Processing {input_file.name}',
                           mininterval=mininterval):
            result = processing_func(record)
            if result:
                data_output.records.append(result)
    else:
        with ThreadPool(thread_num) as pool:
            for y in tqdm(pool.imap(processing_func, data_input.records),
                          total=len(data_input.records),
                          desc=f'Processing {input_file.name}',
                          mininterval=mininterval):
                if y:
                    data_output.records.append(y)

    if output_dir is not None:
        data_output.save(dst_file, json_encoder=json_encoder)


class GptChatDir:
    """ 一个目录，包含了一个任务的所有数据，包括in、out、post等文件 """

    def __init__(self, root, lines_per_file=10000):
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
                        # self.logger.print(f'{post_record["id"]} answer[{k}] contents[{i}] message为空')
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
                        self.logger.print(f'{post_record["id"]} answer[{k}] contents[{i}] content_type={tp} 未见类型')
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

    def post2verify_record(self, post_record):
        """ 这个一般是要具体任务定制的，没有通用操作方式

        注意，如果有些record不想重复verify，可以在类里其他地方预设存储一些已经处理过的结果
            然后在这个函数中引用
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

    @classmethod
    def process_files(cls, processing_func, input_files, output_dir,
                      *, process_num=1, thread_num=1, num_records=None,
                      json_encoder=None):
        if process_num == 1:  # 单进程
            for file in input_files:
                process_file(processing_func, file, output_dir,
                             thread_num=thread_num, num_records=num_records,
                             json_encoder=json_encoder)
        elif isinstance(process_num, int):  # 多进程
            with multiprocessing.Pool(process_num) as pool:
                pool.starmap(process_file,
                         [(processing_func, file, output_dir,
                           thread_num, process_num * 3, num_records, json_encoder)
                          for file in input_files])
        elif isinstance(process_num, (list, tuple)):  # 多进程，但是不同进程"不同构"
            # 这个功能还不是很完善，设计的不太好，暂不推荐使用。但基本原理差不多是这样的，放在这里做个参考。
            process_functions = process_num
            with multiprocessing.Pool(len(process_functions)) as pool:
                pool.starmap(lambda process_func, file:
                         process_func(processing_func, file, output_dir,
                                      thread_num, process_num * 3, num_records,
                                      json_encoder),
                         zip(process_functions, input_files))
        else:
            raise TypeError

    def create_post(self, *, reset=False, num_records=None, process_num=1, thread_num=1):
        """ 建议初步跑的时候，先串行debug，等比较稳定后，再开并发跑

        :param num_records: 每个文件最多提取多少条目，用于小批量运行调试
        :param int process_num: 分文件多进程执行
        :param int thread_num: 每个文件里的多线程执行数
        """
        input_files = self.chatted_dir.files
        if num_records:
            input_files = input_files[:1]  # 使用num_records的时候，会只跑一个文件
        output_dir = self.post_dir.root
        processing_func = self.chatted2post_record
        if reset:
            output_dir.delete()

        input_num = len(input_files)
        print(f'【create_post】后处理 {input_num}个文件待处理')

        self.process_files(processing_func, input_files, output_dir,
                           process_num=process_num, thread_num=thread_num,
                           num_records=num_records)

        self.update_dir()

    def create_verify(self, *, reset=False, num_records=None, process_num=1, thread_num=1,
                      json_encoder=None):
        """ 有时候create_verify是有cpu密集运算场景的，可以开多进程
        """
        input_files = self.post_dir.files
        if num_records:
            input_files = input_files[:1]  # 使用num_records的时候，会只跑一个文件
        output_dir = self.verify_dir.root
        processing_func = self.post2verify_record
        if reset:
            output_dir.delete()

        input_num = len(input_files)
        print(f'【create_verify】得到更准确或精确后处理的验证集 {input_num}个文件待处理')

        self.process_files(processing_func, input_files, output_dir,
                           process_num=process_num, thread_num=thread_num,
                           num_records=num_records, json_encoder=json_encoder)

        self.update_dir()

    @classmethod
    def texts2train_record(cls, texts):
        """ user和assistant的轮询对话，转为训练集格式 """
        messages = []
        for i, text in enumerate(texts):
            role = 'assistant' if i % 2 else 'user'
            messages.append({'role': role, 'content': text})
        return {'messages': messages}

    def create_train(self, *, reset=False, num_records=None, process_num=1, thread_num=1):
        input_files = self.verify_dir.files
        if num_records:
            input_files = input_files[:1]  # 使用num_records的时候，会只跑一个文件
        output_dir = self.train_dir.root
        processing_func = self.verify2train_record
        if reset:
            output_dir.delete()

        input_num = len(input_files)
        print(f'【create_train】生成训练集数据 {input_num}个文件待处理')

        self.process_files(processing_func, input_files, output_dir,
                           process_num=process_num, thread_num=thread_num,
                           num_records=num_records)

        self.update_dir()

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
