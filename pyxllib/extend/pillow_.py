#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/30 22:05


import io
import subprocess


import requests

try:
    import PIL
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'pillow'])
    import PIL

from PIL import Image


def get_img_content(in_):
    """获取in_代表的图片的二进制数据
    :param in_: 可以是本地文件，也可以是图片url地址
    """
    from pyxllib.util.judge import is_url, is_file

    # 1、取不同来源的数据
    if is_url(in_):
        content = requests.get(in_).content
        img = Image.open(io.BytesIO(content))
    elif is_file(in_):
        with open(in_, 'rb') as f:
            content = f.read()
        img = Image.open(in_)
    else:
        raise ValueError

    # 2、如果是RGBA类型，要把透明底变成白色
    # img.mode: https://pillow.readthedocs.io/en/5.1.x/handbook/concepts.html#concept-modes
    if img.mode in ('RGBA', 'P'):
        # 判断图片mode模式，如果是RGBA或P等可能有透明底，则和一个白底图片合成去除透明底
        background = Image.new('RGBA', img.size, (255, 255, 255))
        # composite是合成的意思。将右图的alpha替换为左图内容
        img = Image.alpha_composite(background, img.convert('RGBA')).convert('RGB')
        file = io.BytesIO()
        img.save(file, 'PNG')
        content = file.getvalue()
    # file = Path('a.png', root=Path.TEMP).write(content)
    # chrome(str(file))

    return content
