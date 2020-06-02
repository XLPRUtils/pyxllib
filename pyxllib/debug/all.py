#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/06/02 11:06


"""加载debug包下的所有功能
懒人使用，一键全导入，不推荐工程性较强的项目这样使用
"""


from pyxllib.debug.pytictoc import *
from pyxllib.debug.timer import *
from pyxllib.debug.arrow_ import *

from pyxllib.debug.dprint import *

from pyxllib.debug.strlib import *
from pyxllib.debug.judge import *
from pyxllib.debug.chardet_ import *

from pyxllib.debug.qiniu_ import *
from pyxllib.debug.pathlib_ import *
from pyxllib.debug.dirlib import *

from pyxllib.debug.typelib import *

from pyxllib.debug.chrome import *
from pyxllib.debug.showdir import *
from pyxllib.debug.bcompare import *

from pyxllib.debug.main import *


if __name__ == '__main__':
    TicToc.process_time(f'{__file__} 启动准备共用时')
    # D:/slns/pyxllib/pyxllib/debug/all.py 启动准备共用时 0.516 秒

    tictoc = TicToc(__file__)

    tictoc.toc()
