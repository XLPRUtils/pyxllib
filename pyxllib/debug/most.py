#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/01 17:20

from pyxllib.basic.most import *
from pyxllib.debug.type import *
from pyxllib.debug.browser import *
from pyxllib.debug.showdir import *
from pyxllib.debug.bc import *

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
    f = ob.render(str(File(f'{name}.html', Dir.TEMP)))
    if show: browser(f)
    return f


if __name__ == '__main__':
    pass
