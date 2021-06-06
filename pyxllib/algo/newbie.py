#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

def xywh2ltrb(p):
    return [p[0], p[1], p[0] + p[2], p[1] + p[3]]


def ltrb2xywh(p):
    return [p[0], p[1], p[2] - p[0], p[3] - p[1]]
