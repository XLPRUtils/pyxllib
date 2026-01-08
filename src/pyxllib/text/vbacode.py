#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2023/10/20

import re


class VBACodeFixer:
    @classmethod
    def simplify_code(cls, code_text):
        """ 代码简化，去掉一些冗余写法 """
        code_text = re.sub(r'ActiveSheet\.(Range|Rows|Columns|Cells)', r'\1', code_text)
        code_text = re.sub(r'(\w+)\.(Row|Column)\s*\+\s*\1\.\2s\.Count\s*-\s*1', r'\1.\2End', code_text)

        return 1, code_text.strip()
