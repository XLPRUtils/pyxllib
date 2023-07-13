#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2023/07/13 14:26

import json

from pyxllib.prog.specialist import browser
from pyxllib.algo.pupil import ValuesStat
from pyxllib.text.pupil import strwidth
from pyxllib.file.specialist import XlPath


class GPTQADataBatch:
    """ GPT问答批量执行脚本的jsonl生成、读取器 """

    def __init__(self, file=None, *, start_id=None):
        from datetime import date
        if start_id is None:
            today = date.today().strftime("%Y%m%d")
            self.start_id = int(today + "00000000")
        else:
            self.start_id = start_id

        self.sessions = []

        if file is not None:
            self.read_jsonl(file)

    def read_jsonl(self, file):
        """ 从一个文件加载数据
        """
        self.sessions = XlPath(file).read_jsonl()
        try:
            self.start_id = self.sessions[-1]['id']
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

    def set_session(self, texts, *, id_=0, max_word_length=None, prompt=None):
        """
        :param texts:
            可以输入list，原本配置的多轮对话
            也可以只输入一个str，默认一轮对话
        :param id_: 可以自定义这个session的id
        :param max_word_length: 是否设置一个约束长度，自动切分会话
        :param prompt: 自动分段后
            None，自动配置的一套提示
            '', 不用提示
        :return:
        """
        self.start_id += 1

        if isinstance(texts, str):
            texts = [texts]

        new_texts = []
        for text in texts:
            new_texts += self.split_and_add_prompt(text, max_word_length=max_word_length, prompt=prompt)
        texts = new_texts

        item = {'id': id_ or self.start_id,
                'text': texts, 'first_text_length': len(texts[0])}
        self.sessions.append(item)

    def browser_session(self, i):
        """ 检查第i次会话的内容
        """
        session = self.sessions[i]

        # 构建HTML内容
        html_content = "<html><body>"

        # 输出除了text和all_answers以外的所有键值信息
        html_content += "<h2>会话信息：</h2>"
        html_content += "<ul>"
        for key, value in session.items():
            if key not in ["text", "all_answers"]:
                html_content += f"<li>{key}: {value}</li>"
        html_content += "</ul>"

        # 输出text和all_answers的内容
        text = session.get("text", [])
        all_answers = session.get("all_answers", [])

        max_length = max(len(text), len(all_answers))
        for idx in range(max_length):
            html_content += f"<h3>第{idx + 1}次询问：</h3>"
            if idx < len(text):
                html_content += f"<pre>{text[idx]}</pre>"
            if idx < len(all_answers):
                html_content += f"<h3>第{idx + 1}次回答：</h3>"
                html_content += f"<pre>{all_answers[idx]}</pre>"

        html_content += "</body></html>"
        html_file = (XlPath.tempdir() / (str(session.get('id', i)) + '.html'))
        html_file.write_text(html_content)
        browser.html(html_file)

        # 或者返回HTML字符串
        return html_content

    def check_sessions(self):
        def print_statistics(data, indent_level=2):
            # 使用 ValuesStat 类统计数据并输出摘要
            stat_len = ValuesStat([len(x) for x in data])
            stat_strwith = ValuesStat([strwidth(x) for x in data])

            indent = "\t" * indent_level
            print(f"{indent}     len {stat_len.summary()}")
            print(f"{indent}strwidth {stat_strwith.summary()}")

        # 单次 QA 的长度信息
        qa_texts = []
        qa_answers = []
        for session in self.sessions:
            texts = session.get("text", [])
            all_answers = session.get("all_answers", [])
            qa_texts.extend(texts)
            qa_answers.extend(all_answers)

        print("单次 QA 的长度信息:")
        print('\ttext')
        print_statistics(qa_texts)
        if qa_answers:
            print('\tall_answers')
            print_statistics(qa_answers)

        # 单次 session 的长度信息
        session_texts = []
        session_answers = []
        for session in self.sessions:
            texts = session.get("text", [])
            all_answers = session.get("all_answers", [])
            session_texts.append("".join(texts))
            session_answers.append("".join(all_answers))

        print("单次 session 的长度信息(如果只有单轮qa，则统计跟上面是一样的):")
        print('\ttext')
        print_statistics(session_texts)
        print('\tall_answers')
        print_statistics(session_answers)

    def save_jsonl(self, filename):
        content = '\n'.join([json.dumps(x, ensure_ascii=False) for x in self.sessions])
        XlPath(filename).write_text(content)
