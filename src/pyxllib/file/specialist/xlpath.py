#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/02/24

"""
Deprecated: specialist.xlpath 模块已废弃，请使用 pyxllib.file.xlpath
"""

import warnings

warnings.warn("pyxllib.file.specialist.xlpath is deprecated, please use pyxllib.file.xlpath instead.",
              DeprecationWarning, stacklevel=2)

from pyxllib.file.xlpath import get_encoding, cache_file, XlPath, GetEtag, get_etag, StreamJsonlWriter
from pyxllib.file.fmt import refinepath
from pyxllib.file.dirglob import reduce_dir_depth
