#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 10:00


"""
厦门理工模式识别团队通用python代码工具库

注意为了避免循环嵌套引用，代码逻辑清晰，请尽量不要在此文件写代码

目前已有的模块清单
01、ai，人工智能
02、algo，算法
03、cv，计算机视觉
04、data，特殊数据处理、可视化
05、debug，调试
06、example，测试代码片段
07、excel，电子表格
08、file，文件处理
09、gui，图形界面开发
10、prog，程序开发、工程化组件
11、robot，机器人流程自动化
12、stdlib，魔改标准库、三方库
13、text，文本处理
14、time，时间
15、tool，工具

模块整理计划
1、0.0.x->0.1.x，把basic,debug,cv,util模式拆的更加精细，15个模块
    暂时仍使用 from pyxllib.xx import * 的导入模式
2、一个一个模块精简，尽量剥离对三方依赖小的功能出来
"""

VERSION = '0.1.1'
