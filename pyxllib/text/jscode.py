#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽

import re


class JSCommentRemover:
    """ js代码注释删除器，使用了比较严谨的规则，但也不能说特别完美 """

    def __init__(self, js_source_code):
        self.js_source_code = js_source_code + '\n\n             '  # Adding buffer to prevent out-of-bound errors
        self.js_code_length = len(js_source_code)
        self.index_pointer = -1
        self.comment_placeholder = ''  # Placeholder for comments
        self.js_without_comment = ''

    def _handle_normal_quote(self, delimiter='"'):
        while self.index_pointer < self.js_code_length:
            self.index_pointer += 1
            char = self.js_source_code[self.index_pointer]
            self.js_without_comment += char

            if char == delimiter:
                return
            elif char == '\\':
                self.index_pointer += 1
                if self.js_source_code[self.index_pointer + 1] == delimiter:
                    pass
                self.js_without_comment += self.js_source_code[self.index_pointer]
            elif char == '\n':
                raise ValueError(f'Syntax Error: Missing right string delimiter: {delimiter}')

    def _handle_template_string(self, delimiter='`'):
        while self.index_pointer < self.js_code_length:
            self.index_pointer += 1
            char = self.js_source_code[self.index_pointer]
            self.js_without_comment += char

            if char == delimiter:
                return
            elif char == '\\':
                self.index_pointer += 1
                self.js_without_comment += self.js_source_code[self.index_pointer]
            elif char == '$' and self.js_source_code[self.index_pointer + 1] == '{':
                self.index_pointer += 1
                self.js_without_comment += '{'
                self._nested_expression_inside_template_string()

    def _nested_expression_inside_template_string(self):
        while self.index_pointer < self.js_code_length:
            self.index_pointer += 1
            char = self.js_source_code[self.index_pointer]
            if char == '}':
                self.js_without_comment += char
                return
            elif char in '"\'':
                self.js_without_comment += char
                self._handle_normal_quote(delimiter=char)
            elif char == "`":
                self.js_without_comment += char
                self._handle_template_string()
            elif char == '/':
                self._handle_slash()

    def _handle_slash(self):
        next_char = self.js_source_code[self.index_pointer + 1]
        if next_char == '/':
            self._single_line_comment()
            self.js_without_comment += self.comment_placeholder
        elif next_char == '*':
            self.js_without_comment += self.comment_placeholder
            self.index_pointer += 1
            self._multi_line_comment()
        elif bool(re.search(r'[\n\r(\[\{=+,;]([\t]*|\\[\n\r])*[\t]*$', self.js_without_comment)):
            self.js_without_comment += '/'
            self._regular_expression()
        else:
            self.js_without_comment += '/'

    def _single_line_comment(self):
        while self.js_source_code[self.index_pointer] not in '\n\r' and self.index_pointer < self.js_code_length:
            self.index_pointer += 1
        self.js_without_comment += '\n'

    def _multi_line_comment(self):
        while not (
                self.js_source_code[self.index_pointer] == '*' and self.js_source_code[self.index_pointer + 1] == '/'):
            self.index_pointer += 1
        self.index_pointer += 1

    def _regular_expression(self):
        while self.js_source_code[self.index_pointer + 1] != '/':
            self.index_pointer += 1
            self.js_without_comment += self.js_source_code[self.index_pointer]
            if self.js_source_code[self.index_pointer] == '\\':
                self.index_pointer += 1
                self.js_without_comment += self.js_source_code[self.index_pointer]

    def remove_comments(self):
        while self.index_pointer < self.js_code_length:
            self.index_pointer += 1
            char = self.js_source_code[self.index_pointer]
            if char == '/':
                self._handle_slash()
                continue
            elif char in '"\'':
                self.js_without_comment += char
                self._handle_normal_quote(delimiter=char)
            elif char == "`":
                self.js_without_comment += char
                self._handle_template_string()
            elif char == "S" and self.js_code_length - self.index_pointer > 11 and \
                    self.js_source_code[self.index_pointer:self.index_pointer + 11] == 'String.raw`':
                self.js_without_comment += char
                self.index_pointer += 10
                self.js_without_comment += 'tring.raw`'
                self._handle_template_string()
            else:
                self.js_without_comment += char
        return self.js_without_comment


def remove_js_comments(js_source_code):
    remover = JSCommentRemover(js_source_code)
    return remover.remove_comments()
