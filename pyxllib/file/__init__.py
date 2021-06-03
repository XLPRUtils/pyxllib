#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/08/14 21:52


"""
最基础常用的一些功能

basic中依赖的三方库有直接写到 requirements.txt 中
（其他debug、cv等库的依赖项都是等使用到了才加入）

且basic依赖的三方库，都确保是体积小
    能快速pip install
    及pyinstaller -F打包生成的exe也不大的库
"""

from pyxllib.file.base import *
from pyxllib.file.file import *
from pyxllib.file.dir import *
