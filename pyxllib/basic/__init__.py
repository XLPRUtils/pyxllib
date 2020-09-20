#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/08/14 21:52


"""
最基础常用的一些功能

basic中依赖的三方库有直接写到 requirements.txt 中
（其他debug、cv等库的依赖项都是等使用到了才加入）

且basic依赖的三方库，都确保是体积小
    能快速pip install
    及pyinstaller -F打包生成的exe也不大的库
"""

# 1 文本处理等一些基础杂项功能
from pyxllib.basic._1_strlib import *
# 2 时间相关工具
from pyxllib.basic._2_timelib import *
# 3 文件、路径工具
from pyxllib.basic._3_pathlib import *
# 4 调试工具，Iterate等一些高级通用功能
from pyxllib.basic._4_loglib import *
# 5 目录工具
from pyxllib.basic._5_dirlib import *
