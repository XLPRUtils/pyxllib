#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/03/16 09:19


"""一些python通用功能的性能测试
    虽然其实大部分场合其实都是相通的
    有时候test测试代码，其实也是演示如何使用的demo

demo：示例代码，注重演示
test：测试代码，注重分析功能稳定性
perf：性能测试，注重分析代码的运行效率
"""

from code4101py.util.main import *
import socket


def demo_timer():
    """该函数也可以用来测电脑性能
    代码中附带的示例结果是我在自己小米笔记本上的测试结果
    Intel（R） Core（TM） i7-10510U CPU@ 1.80GHz 2.30 GHz，15G 64位
    """
    print('1、普通用法（循环5*1000万次用时）')
    timer = Timer('循环', start_now=True)  # 也可以写 timer = Timer('循环')
    for _ in range(5):
        for _ in range(10 ** 7):
            pass
    timer.stop_and_report()
    # 循环 用时: 0.727s

    print('2、循环多轮计时（循环5*1000万次用时）')
    timer = Timer()
    for _ in range(5):
        timer.start()  # 该句可以省略
        for _ in range(10 ** 7):
            pass
        timer.stop()  # 每轮结束时标记
    timer.stop_and_report()
    # 总耗时: 0.771s	均值标准差: 0.154±0.005s	总数: 5	最小值: 0.148s	最大值: 0.162s

    print('3、with上下文用法')
    with Timer('循环'):
        for _ in range(5):
            for _ in range(10 ** 6):
                pass
    # 循环 用时: 0.081s

    print('4、with上下文用法2')
    with Timer('循环') as t:
        for _ in range(5):
            for _ in range(10 ** 6):
                pass
            t.stop()
    # 循环 总耗时: 0.082s	均值标准差: 0.016±0.000s	总数: 5	最小值: 0.016s	最大值: 0.017s

    print('5、可以配合dformat输出定位信息')
    with Timer(dformat()):
        for _ in range(5):
            for _ in range(10 ** 6):
                pass
    # [04]demolib.py/59:      用时: 0.087s


def demo_system():
    """主要是测试一些系统变量值，顺便再演示一次Timer用法"""
    def demo_pc_messages():
        """演示如何获取当前操作系统的PC环境数据"""
        # fqdn：fully qualified domain name
        print('1、socket.getfqdn() :', socket.getfqdn())  # 完全限定域名，可以理解成pcname，计算机名
        # 注意py的很多标准库功能本来就已经处理了不同平台的问题，尽量用标准库而不是自己用sys.platform作分支处理
        print('2、sys.platform     :', sys.platform)  # 运行平台，一般是win32和linux
        li = os.getenv('PATH').split(os.path.pathsep)  # 环境变量名PATH，win中不区分大小写，linux中区分大小写必须写成PATH
        print("3、os.getenv('PATH'):", f'数量={len(li)},', pprint.pformat(li, 4))

    def demo_executable_messages():
        """演示如何获取被执行程序相关的数据"""
        print('1、sys.path      :', f'数量={len(sys.path)},', pprint.pformat(sys.path, 4))  # import绝对位置包的搜索路径
        print('2、sys.executable:', sys.executable)  # 当前被执行脚本位置
        print('3、sys.version   :', sys.version)  # python的版本
        print('4、os.getcwd()   :', os.getcwd())  # 获得当前工作目录
        print('5、gettempdir()  :', tempfile.gettempdir())  # 临时文件夹位置

    timer = Timer('demo_system')
    print('>>> demo_pc_messages()')
    demo_pc_messages()
    print('>>> demo_executable_messages()')
    demo_executable_messages()
    timer.stop_and_report()


def test_re():
    """ 正则re模块相关功能测试
    """
    # 190103周四
    # py的正则[ ]语法，可以用连字符-匹配一个区间内的字符，
    # 例如数字0-9（你是不是蠢，不会用\d么），还有a-z、A-Z（\w），甚至①-⑩，但是一-十，注意'四'匹配不到
    dprint(re.sub(r'[一-十]', '', '一二三四五六七八九十'))
    # [05]demolib.py/98: re.sub(r'[一-十]', '', '一二三四五六七八九十')<str>='四'

    # 200319周四14:11，匹配顺序与内容有关，先出现的先匹配，而与正则里or语法参数顺序无关
    print(re.findall(r'(<(a|b)>.*?</\2>)', '<a><b></b></a>'))
    print(re.findall(r'(<(b|a)>.*?</\2>)', '<a><b></b></a>'))
    # 结果都是： [('<a><b></b></a>', 'a')]
    # TODO 200323周一17:22，其实是模式不够复杂，在特殊场景下，可选条件的前后顺序是有影响的
