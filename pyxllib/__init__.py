#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 10:00


"""
厦门理工模式识别团队通用python代码工具库

注意为了避免循环嵌套引用，代码逻辑清晰，请尽量不要在此文件写代码

文档：https://www.yuque.com/xlpr/pyxllib/home/edit
"""

import sys

if sys.version_info.major == 3 and sys.version_info.minor >= 8:
    import importlib.metadata

    version = importlib.metadata.version('pyxllib')
