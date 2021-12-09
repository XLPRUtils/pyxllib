#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/08 22:53

"""
TODO 写一些图片相似度相关功能
"""

from pyxllib.prog.pupil import check_install_package

check_install_package('imagehash', 'ImageHash')

import imagehash
