#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/11/15 10:04

import requests
import subprocess

try:
    import PIL
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'pillow'])
    import pil

try:
    import get_image_size
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'opsdroid-get-image-size'])
    import get_image_size

try:
    import cv2
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'opencv-python'])
    import cv2

from pyxllib.cv.cvprcs import *
from pyxllib.cv.pilprcs import *
from pyxllib.cv.cvimg import *
