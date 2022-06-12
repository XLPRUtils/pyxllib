#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/06/09 16:26

"""
针对PostgreSQL封装的工具
"""

from pyxllib.prog.pupil import check_install_package

check_install_package('psycopg')

import psycopg
# import psycopg2
# psycopg2.extensions.connection


class Connection(psycopg.connection):
    pass
