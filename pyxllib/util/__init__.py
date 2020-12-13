#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2018/09/19 19:41

"""
一、一些通用的扩展组件

二、对标准库或一些第三方库，进行的功能扩展
    也有可能对一些bug进行了修改

有些是小的库，直接把源码搬过来了
    有些是较大的库，仍然要（会自动在需要使用时 pip install）安装

zipfile: py3.6在windows处理zip，解压中文文件会乱码，要改一个编码
    这个在py3.8中也没有修复，但是py3.8的zipfile更新了不少内容，有时间我要重新整理过来
onepy: 做了些中文注解，其他修改了啥我也忘了~~可能是有改源码功能的
pyautogui: 封装扩展了自己的一个 AutoGui 类
"""
