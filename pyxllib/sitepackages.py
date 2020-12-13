#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 09:36


r"""
除了使用pip install pyxllib，
还可以下载最新源代码，然后用该脚本将pyxllib配置到site-packages里，便于索引使用
"""


import os
import site


def add_project_to_sitepackages(project_path = os.path.dirname(__file__)):
    """ 将制定项目配置到sitepackages目录
    :param project_path: 项目路径
        默认本脚本所在的目录为项目路径

    >> add_project_to_sitepackages('D:/pyxllib/pyxllib')
    """
    pth_file = os.path.join(site.getsitepackages()[-1], os.path.basename(project_path) + '.pth')
    with open(pth_file, 'w') as f:
        f.write(os.path.dirname(project_path))


if __name__ == '__main__':
    add_project_to_sitepackages()
