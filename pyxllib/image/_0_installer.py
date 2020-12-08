#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/12/08

import subprocess

try:
    import PIL
except ModuleNotFoundError:
    subprocess.run(['pip', 'install', 'pillow'])
    import PIL

try:
    import fitz
except ModuleNotFoundError:
    subprocess.run(['pip', 'install', 'PyMuPdf'])
    import fitz
