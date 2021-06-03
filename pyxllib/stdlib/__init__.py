#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/01/13 15:29

"""
对标准库或一些第三方库，进行的功能扩展
    也有可能对一些bug进行了修改

有些是小的库，直接把源码搬过来了
    有些是较大的库，仍然要（会自动在需要使用时 pip install）安装

改了底层标准库一些功能，修复一些bug，或者提升功能兼容性、强度

zipfile: py3.6在windows处理zip，解压中文文件会乱码，要改一个编码
    这个在py3.8中也没有修复，但是py3.8的zipfile更新了不少内容，有时间我要重新整理过来
onepy: 做了些中文注解，其他修改了啥我也忘了~~可能是有改源码功能的
"""

# from pyxllib.stdlib import zipfile
# from pyxllib.stdlib import shutil
