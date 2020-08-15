#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/08/14 21:52


"""
最基础常用的一些功能

basic中依赖的三方库有直接写到 requirements.txt 中
（其他debug、image等库的依赖项都是等使用到了才加入）

且basic依赖的三方库，都确保是体积小
    能快速pip install
    及pyinstaller -F打包生成的exe也不大的库
"""

# 1 时间相关工具
from .pytictoc import TicToc
from .timer import Timer
from .arrow_ import Datetime

# 2 调试1
from .dprint import *

# 3 文本
from .strlib import *

# 4 文件、目录工具
from .judge import *
from .chardet_ import *
from .qiniu_ import *
from .pathlib_ import Path
from .dirlib import *

# 5 其他工具
from .jsondata import *
