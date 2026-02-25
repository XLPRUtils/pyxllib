#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/09 11:02

"""
Deprecated: specialist.download 模块已废弃，请使用 pyxllib.file.downloader
"""

import warnings

warnings.warn("pyxllib.file.specialist.download is deprecated, please use pyxllib.file.downloader instead.",
              DeprecationWarning, stacklevel=2)

from pyxllib.file.downloader import download_file, read_from_ubuntu, download, ensure_localfile, ensure_localdir, get_font_file
