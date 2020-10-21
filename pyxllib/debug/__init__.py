#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/06/02 15:48

"""调试功能，通用底层功能

文中所有前缀4个下划线_的是模块划分标记，且
    前4个下划线_是一级结构
    前8个下划线_是二级结构

相关文档： https://blog.csdn.net/code4101/article/details/83269101
"""

____main = """
这里会加载好主要功能，但并不会加载所有功能
"""

from ._0_installer import *
from ._1_typelib import *
from ._2_chrome import *
from ._3_showdir import *
from ._4_bcompare import *

____other = """
"""


def render_echart(ob, name, show=False):
    """ 渲染显示echart图表

    https://www.yuque.com/xlpr/pyxllib/render_echart

    :param ob: 一个echart图表对象
    :param name: 存储的文件名
    :param show: 是否要立即在浏览器显示
    :return: 存储的文件路径
    """
    # 如果没有设置页面标题，则默认采用文件名作为标题
    if not ob.page_title or ob.page_title == 'Awesome-pyecharts':
        ob.page_title = name
    f = ob.render(Path(f'{name}.html', root=Path.TEMP).fullpath)
    if show: chrome(f)
    return f


if __name__ == '__main__':
    pass
