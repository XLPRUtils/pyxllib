#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 22:23

from pyxllib.xl import *
from pyxllib.algo.geo import *
from pyxllib.cv.expert import *

# 把自定义的一些功能嵌入到PIL.Image.Image类中。
# 因为pyxllib.xlcv设计初衷本就是为了便捷而牺牲工程性。
# 如果这步没有给您“惊喜”而是“惊吓”，
# 可以使用 from pyxllib.cv.expert import * 代替 from pyxllib.xlcv import *。
# 然后显式使用 xlpil.imsize(im) 来代替 im.imsize 等用法。
xlpil.enchant()
