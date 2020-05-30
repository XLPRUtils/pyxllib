#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/30


import os


from pyxllib.extend.pytictoc import TicToc
from pyxllib.extend.arrow_ import Datetime


if __name__ == '__main__':
    # TicToc.process_time(f'{dformat()}启动准备共用时')
    tictoc = TicToc(__file__)
    # os.chdir(Path.DESKTOP)

    print(os.path.isfile(Datetime()))

    tictoc.toc()
