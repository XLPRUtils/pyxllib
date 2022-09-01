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
import numpy as np

from pyxllib.cv.xlpillib import xlpil


def get_init_hash():
    """ 获得一个初始、空哈希值 """
    return imagehash.ImageHash(np.zeros([8, 8]).astype(bool))


def phash(image, *args, **kwargs):
    """ 修改了官方接口，这里输入的image支持泛用格式
    """
    im = xlpil.read(image)
    return imagehash.phash(im, *args, **kwargs)


def dhash(image, *args, **kwargs):
    """ 修改了官方接口，这里输入的image支持泛用格式
    """
    im = xlpil.read(image)
    return imagehash.dhash(im, *args, **kwargs)
