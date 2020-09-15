#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/08/14 22:20

import subprocess

try:
    import PIL
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'pillow'])
    import PIL

try:
    from get_image_size import get_image_size
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'opsdroid-get-image-size'])
    from get_image_size import get_image_size

try:
    import fitz
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'PyMuPdf'])
    import fitz

try:
    import cv2
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'opencv-python'])
    import cv2

try:
    import shapely
except ModuleNotFoundError:
    try:
        subprocess.run(['conda', 'install', 'shapely'])
        import shapely
    except FileNotFoundError:
        # 这个库用pip安装是不够的，正常要用conda，有些dll才会自动配置上
        subprocess.run(['pip', 'install', 'shapely'])
        import shapely
