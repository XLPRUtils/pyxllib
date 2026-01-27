#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/20 11:46


import re
import tokenize
import io

import fire

from pyxllib.text.nestenv import PyNestEnv


def remove_interaction_chars(text):
    """ 去掉复制的一段代码中，前导的“>>>”标记 """
    # 这个算法可能还不够严谨，实际应用中再逐步写鲁棒
    # ">>> "、"... "
    lines = [line[4:] for line in text.splitlines()]
    return '\n'.join(lines)


def sort_import(text):
    def cmp(line):
        """ 将任意一句import映射为一个可比较的list对象

        :return: 2个数值
            1、模块优先级
            2、import在前，from在后
        """
        name = re.search(r'(?:import|from)\s+(\S+)', line).group(1)
        for i, x in enumerate('stdlib prog algo text file cv data extend'.split()):
            name = name.replace('pyxllib.' + x, f'{i:02}')
        for i, x in enumerate('pyxllib pyxlpr xlproject'.split()):
            name = name.replace(x, f'~{i:02}')
        for i, x in enumerate('newbie pupil specialist expert'.split()):
            name = name.replace('.' + x, f'{i:02}')

        # 忽略大小写
        return [name.lower(), not line.startswith('import')]

    def sort_part(m):
        parts = PyNestEnv(m.group()).imports().strings()
        parts = [p.rstrip() + '\n' for p in parts]
        parts.sort(key=cmp)
        return ''.join(parts)

    res = PyNestEnv(text).imports().sub(sort_part, adjacent=True)  # 需要邻接，分块处理
    return res


def rename_identifier(text, old_name, new_name):
    """ 标识符重命名
    """
    ne = PyNestEnv(text).identifier(old_name)
    new_text = ne.replace(new_name)
    return new_text


def refine_quotes(text):
    """ 将双引号字符串转换为单引号字符串，保留三引号

    使用 tokenize 模块进行解析，正确处理嵌套结构。
    """
    try:
        tokens = list(tokenize.tokenize(io.BytesIO(text.encode('utf-8')).readline))
    except tokenize.TokenError:
        # 解析失败（如代码不完整）则保持原样
        return text

    # 记录待替换的范围 (start, end) 和新内容，后续统一从后往前替换
    replacements = []

    # Python 3.12+ f-string 相关 token 类型
    FSTRING_START = getattr(tokenize, 'FSTRING_START', -1)
    FSTRING_MIDDLE = getattr(tokenize, 'FSTRING_MIDDLE', -1)
    FSTRING_END = getattr(tokenize, 'FSTRING_END', -1)

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok.type == tokenize.STRING:
            s = tok.string
            # 排除三引号
            if s.endswith('"""') or s.endswith("'''"):
                i += 1
                continue

            # 解析前缀、引号类型和内容
            m = re.match(r'^(?P<prefix>(?i:(?:fr|rf|br|rb|[ubrf]))?)(?P<quote>["\'])(?P<body>.*)(?P=quote)$', s, re.DOTALL)
            if not m:
                i += 1
                continue

            prefix, quote, body = m.group('prefix'), m.group('quote'), m.group('body')

            if quote == '"':
                # 如果内容包含单引号，则不转换
                if "'" in body:
                    i += 1
                    continue

                if 'r' in prefix.lower():
                    # 原始字符串：直接替换引号
                    replacements.append((tok.start, tok.end, f"{prefix}'{body}'"))
                else:
                    # 普通字符串：转换 \" 为 "
                    def _refine_body_func(match):
                        g = match.group()
                        return '"' if g == '\\"' else g

                    new_body = re.sub(r'\\.|[^\\]', _refine_body_func, body, flags=re.DOTALL)
                    replacements.append((tok.start, tok.end, f"{prefix}'{new_body}'"))

        elif tok.type == FSTRING_START:
            # 处理 f-string (Python 3.12+)
            start_tok = tok
            s = start_tok.string

            # 仅处理非三引号的双引号 f-string
            if not s.endswith('"') or s.endswith('"""'):
                i += 1
                continue

            # 寻找匹配的 FSTRING_END，并检查中间内容是否包含单引号
            depth, j = 1, i + 1
            relevant_middles, can_convert = [], True

            while j < len(tokens):
                t = tokens[j]
                if t.type == FSTRING_START:
                    depth += 1
                elif t.type == FSTRING_END:
                    depth -= 1
                    if depth == 0: break
                elif t.type == FSTRING_MIDDLE and depth == 1:
                    if "'" in t.string: can_convert = False
                    relevant_middles.append(j)
                j += 1

            if can_convert and depth == 0 and j < len(tokens):
                # 转换开始、结束和中间部分
                replacements.append((start_tok.start, start_tok.end, s[:-1] + "'"))
                replacements.append((tokens[j].start, tokens[j].end, "'"))

                is_raw = 'r' in s.lower()
                for mid_idx in relevant_middles:
                    mt = tokens[mid_idx]
                    if not is_raw:
                        new_mid = mt.string.replace('\\"', '"')
                        if new_mid != mt.string:
                            replacements.append((mt.start, mt.end, new_mid))

        i += 1

    # 倒序应用替换，避免偏移量失效
    lines = text.splitlines(keepends=True)
    replacements.sort(key=lambda x: (x[0], x[1]), reverse=True)

    result_lines = list(lines)
    for start, end, new_text in replacements:
        s_row, s_col = start
        e_row, e_col = end
        s_idx, e_idx = s_row - 1, e_row - 1

        if s_idx == e_idx:
            # 单行替换
            line = result_lines[s_idx]
            result_lines[s_idx] = line[:s_col] + new_text + line[e_col:]
        else:
            # 多行替换：更新起始行，并清空后续行
            start_line_pre = result_lines[s_idx][:s_col]
            end_line_post = result_lines[e_idx][e_col:]
            result_lines[s_idx] = start_line_pre + new_text + end_line_post
            for i in range(s_idx + 1, e_idx + 1):
                result_lines[i] = ""

    return "".join(result_lines)
