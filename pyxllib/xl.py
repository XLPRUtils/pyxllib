#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 22:08

""" pyxllib常用功能
"""

from pyxllib.prog.lazyimport import lazy_import

try:
    import fire
except ModuleNotFoundError:
    fire = lazy_import('fire')

from pyxllib.file.packlib import *

from pyxllib.prog.newbie import *
from pyxllib.prog.pupil import *
from pyxllib.prog.specialist import *
from deprecated import deprecated

from pyxllib.algo.newbie import *
from pyxllib.algo.pupil import *
from pyxllib.algo.specialist import *

from pyxllib.text.newbie import *
from pyxllib.text.pupil import *
from pyxllib.text.specialist import *

from pyxllib.file.newbie import *
from pyxllib.file.pupil import *
from pyxllib.file.specialist import *

if __name__ == '__main__':
    # 直接运行的话，支持开放出所有函数类接口
    fire.Fire()
