#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/09/18 22:16

import traceback

from pyxllib.basic.pathlib_ import *

XLLOG_CONF_FILE = 'xllog.yaml'


def get_xllog():
    """ 获得pyxllib库的日志类

    由于日志类可能要读取yaml配置文件，需要使用Path类，所以实现代码先放在pathlib_.py

    TODO 类似企业微信机器人的机制怎么设？或者如何配置出问题发邮件？
    """
    import logging

    if 'pyxllib.xllog' in logging.root.manager.loggerDict:
        # 1 判断xllog是否已存在，直接返回
        pass
    elif os.path.isfile(XLLOG_CONF_FILE):
        # 2 若不存在，尝试在默认位置是否有自定义配置文件，读取配置文件来创建
        import logging.config
        data = Path(XLLOG_CONF_FILE).read()
        if isinstance(data, dict):
            # 推荐使用yaml的字典结构，格式更简洁清晰
            logging.config.dictConfig(data)
        else:
            # 但是普通的conf配置文件也支持
            logging.config.fileConfig(XLLOG_CONF_FILE)
    else:
        # 3 否则生成一个非常简易版的xllog
        # TODO 不同级别能设不同的格式（颜色）？
        xllog = logging.getLogger('pyxllib.xllog')
        xllog.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S'))
        xllog.addHandler(ch)
    return logging.getLogger('pyxllib.xllog')


def format_exception(e):
    return ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
