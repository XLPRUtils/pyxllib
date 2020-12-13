#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/12/08

import subprocess

try:
    import ahocorasick
except ModuleNotFoundError:
    subprocess.run(['pip', 'install', 'pyahocorasick'])
    import ahocorasick

try:
    import humanfriendly
except ModuleNotFoundError:
    subprocess.run(['pip', 'install', 'humanfriendly'])
    import humanfriendly
