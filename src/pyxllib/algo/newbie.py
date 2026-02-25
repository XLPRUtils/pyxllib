#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

"""
Deprecated: newbie 模块已废弃，请使用新的模块导入功能。
"""

import warnings

warnings.warn("pyxllib.algo.newbie is deprecated, please use specific modules in pyxllib.algo instead.",
              DeprecationWarning, stacklevel=2)

from pyxllib.algo.stat_lite import vector_compare, round_unit
from pyxllib.algo.convert import int2excel_col_name, excel_col_name2int, cvsecs
from pyxllib.algo.struct import gentuple
