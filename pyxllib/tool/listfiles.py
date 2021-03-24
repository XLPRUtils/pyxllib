#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/03/23 15:28

""" 列出目录下所有文件清单的命令行工具 """

from pyxllib.basic import *

try:
    import fire
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'fire'])
    import fire


def listfiles(dirpath):
    res = []
    root = dirpath
    for path, subdirs, files in os.walk(root):
        for name in files:
            nm = os.path.join(os.path.relpath(path, root), name)
            nm = nm.replace('\\', '/')  # 哪怕是在win，也转成linux的路径表达风格
            res.append(nm)
    return res


if __name__ == '__main__':
    # python -m pyxllib.tool.listfiles /home/datasets/textGroup/PubLayNet/publaynet/test2
    fire.Fire(listfiles)  # fire库会自动输出返回值
