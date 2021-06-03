#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 22:54

import os
import socket

HOSTNAME = socket.getfqdn()


def get_username():
    return os.path.split(os.path.expanduser('~'))[-1]
