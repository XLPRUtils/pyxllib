#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/08/14 22:20


import subprocess

try:
    from pympler import asizeof
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'pympler'])
    from pympler import asizeof

try:
    import lxml
except ModuleNotFoundError:
    # 好像很多人这个库都装不上，好奇怪~~在命令行装就可以
    subprocess.run(['pip3', 'install', 'lxml'])
    import lxml

try:
    import bs4
except:
    subprocess.run(['pip3', 'install', 'beautifulsoup4'])

try:
    import numpy as np
except:
    subprocess.run(['pip3', 'install', 'numpy'])

try:
    import pandas as pd
except:
    subprocess.run(['pip3', 'install', 'Jinja2'])
    subprocess.run(['pip3', 'install', 'pandas>=0.23.4'])
    import pandas as pd
