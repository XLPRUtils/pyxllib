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
06、ex，example的缩写，测试代码片段
07、excel，电子表格
08、file，文件处理
09、gui，图形界面开发
10、prog，程序开发、工程化组件
11、robot，机器人流程自动化
12、stdlib，魔改标准库、三方库
13、text，文本处理
14、time，时间
15、tool，工具

子模块命名规范含义，譬如小学、初中、高中、大学、读研的过程：
newbie，不用库，新手，幼儿园
pupil：可用标准库，学生，小学
specialist：常用三方库，行家，初高中
    以上级别代码库默认都会包含在 pyxllib.xl 里直接使用
    numpy、pandas在我这里归在这个级别里
expert，该领域常用包，专家，大学
    例如 cv 里基本都要用 opencv、pil
    例如 excel 里基本都要用 openpyxl 处理
master：精细功能，大师，研究生
grandmaster，备用，宗师，博士，一般用不到

以上级别主要在一些功能比较少、杂时，统一默认的命名模式，
    但有比较精细的功能模块时，以功能命名更佳

TODO 模块整理计划
1、0.0.x->0.1.x，把basic,debug,cv,util模式拆的更加精细，15个模块
2、一个一个模块精简，尽量剥离对三方依赖小的功能到basic、advance模块
"""

VERSION = '0.1.2'
