#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/05/26

from pyxllib.prog.pupil import check_install_package

# 一个xpath解析库
check_install_package('jinja2')

import jinja2
from jinja2 import Template, Environment

from pyxllib.file.specialist import XlPath


def set_template(s, *args, **kwargs):
    """ todo 这个名字会不会太容易冲突了？ """
    return Template(s.strip(), *args, **kwargs)


def set_meta_template(s, meta_start='[[', meta_end=']]', **kwargs):
    """ 支持预先用某些格式渲染后，再返回标准渲染模板 """
    t = Template(s.strip(), variable_start_string=meta_start,
                 variable_end_string=meta_end).render(**kwargs)
    return Template(t)


def get_jinja_template(name, **kwargs):
    template = Environment(**kwargs).from_string((XlPath(__file__).parent / f'templates/{name}').read_text())
    return template
