#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/30


"""配置数据，调试功能，通用底层功能
    通用工具包util的核心基础模块

文中所有前缀4个下划线_的是模块划分标记，且
    前4个下划线_是一级结构
    前8个下划线_是二级结构

代码共分五个大模块
    time: 时间相关功能
    str: 文本相关处理功能
    file: 路径、文件相关处理功能
    debug: 调试相关功能更
    other: 其他常用的功能组件

相关文档： https://blog.csdn.net/code4101/article/details/83269101
"""


# 用import导入一些常用包
import pathlib
import collections
import datetime
import enum
import filecmp
import html
import inspect
import io
import logging
import math
import os
import pickle
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap


import pandas as pd  # 该库加载需要0.22秒
import requests


try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'BeautifulSoup4'])
    from bs4 import BeautifulSoup


____time = """
时间相关工具
"""

from pyxllib.extend.pytictoc import TicToc
from pyxllib.util.timer import Timer
from pyxllib.extend.arrow_ import Datetime


____file = """
文件相关
"""

# Dir的代码还需要好好重新整理下
from pyxllib.util.dirlib import Path, Dir


____debug = """
调试相关功能

TODO dprint是基于我个人目前的经验实现的，我还是要找个时间系统学习下python正规的日志是怎么做的
    有些功能可能有现成标准库可以实现，不用我自己搞一套
    以及我可能还要扩展高亮、写入文件日志等功能

showdir: 查阅一个对象的成员变量及成员方法
    为了兼容linux输出df时也能对齐，有几个中文域宽处理相关的函数
chrome: viewfiles是基础组件，负责将非文本对象文本化后写到临时文件，
    调用外部程序时，是否wait，以及找不到程序时的报错提示。
"""


def getasizeof(*objs, **opts):
    """获得所有类的大小，底层用pympler.asizeof实现"""
    try:
        from pympler import asizeof
    except ModuleNotFoundError:
        subprocess.run(['pip3', 'install', 'pympler'])
        from pympler import asizeof
    try:
        res = asizeof.asizeof(*objs, **opts)
    # except TypeError:  # sqlalchemy.exc.InvalidRequestError
    except:
        res = -1
    return res


def getmembers(object, predicate=None):
    """自己重写改动的 inspect.getmembers"""
    from inspect import isclass, getmro, types

    if isclass(object):
        mro = (object,) + getmro(object)
    else:
        mro = ()
    results = []
    processed = set()
    names = dir(object)
    # :dd any DynamicClassAttributes to the list of names if object is a class;
    # this may result in duplicate entries if, for example, a virtual
    # attribute with the same name as a DynamicClassAttribute exists
    try:
        for base in object.__bases__:
            for k, v in base.__dict__.items():
                if isinstance(v, types.DynamicClassAttribute):
                    names.append(k)
    except AttributeError:
        pass
    for key in names:
        # First try to get the value via getattr.  Some descriptors don't
        # like calling their __get__ (see bug #1785), so fall back to
        # looking in the __dict__.
        try:
            value = getattr(object, key)
            # handle the duplicate key
            if key in processed:
                raise AttributeError
        # except AttributeError:
        except:  # 加了这种异常获取，190919周四15:14，sqlalchemy.exc.InvalidRequestError
            dprint(key)  # 抓不到对应的这个属性
            for base in mro:
                if key in base.__dict__:
                    value = base.__dict__[key]
                    break
            else:
                # could be a (currently) missing slot member, or a buggy
                # __dir__; discard and move on
                continue

        if not predicate or predicate(value):
            results.append((key, value))
        processed.add(key)
    results.sort(key=lambda pair: pair[0])
    return results


def showdir(c, *, to_html=None, printf=True):
    """查看类信息
    会罗列出类c的所有成员方法、成员变量，并生成一个html文
    :param c: 要处理的对象
    :param to_html:
        win32上默认True，用chrome、explorer打开
        linux上默认False，直接输出到控制台
    :param printf:
        默认是True，会输出到浏览器或控制条
        设为False则不输出
    """
    # 1、输出类表头
    res = []
    object_name = func_input_message(2)['argnames'][0]
    if to_html is None:
        to_html = sys.platform == 'win32'
    newline = '<br/>' if to_html else '\n'

    t = f'==== 对象名称：{object_name}，类继承关系：{inspect.getmro(type(c))}，' \
        + f'内存消耗：{sys.getsizeof(c)}（递归子类总大小：{getasizeof(c)}）Byte ===='

    if to_html:
        res.append('<p>')
        t = html.escape(t) + '</p>'
    res.append(t + newline)

    # 2、html的样式精调
    def df2str(df):
        if to_html:
            df = df.applymap(str)  # 不转成文本经常有些特殊函数会报错
            df.index += 1  # 编号从1开始
            # pd.options.display.max_colwidth = -1  # 如果临时需要显示完整内容
            t = df.to_html()
            table = BeautifulSoup(t, 'lxml')
            table.thead.tr['bgcolor'] = 'LightSkyBlue'  # 设置表头颜色
            ch = 'A' if '成员变量' in table.tr.contents[3].string else 'F'
            table.thead.tr.th.string = f'编号{ch}{len(df)}'
            t = table.prettify()
        else:
            # 直接转文本，遇到中文是会对不齐的，但是showdir主要用途本来就是在浏览器看的，这里就不做调整了
            t = dataframe_str(df)
        return t

    # 3、添加成员变量和成员函数
    # 成员变量
    members = getmembers(c)
    methods = filter(lambda m: not callable(getattr(c, m[0])), members)
    ls = []
    for ele in methods:
        k, v = ele
        if k.endswith(r'________'):  # 这个名称的变量是我代码里的特殊标记，不显示
            continue
        attr = getattr(c, k)
        if isinstance(attr, enum.IntFlag):  # 对re.RegexFlag等枚举类输出整数值
            v = typename(attr) + '，' + str(int(attr)) + '，' + str(v)
        else:
            try:
                text = str(v)
            except:
                text = '取不到str值'
            v = typename(attr) + '，' + text
        ls.append([k, v])
    df = pd.DataFrame.from_records(ls, columns=['成员变量', '描述'])
    res.append(df2str(df) + newline)

    # 成员函数
    methods = filter(lambda m: callable(getattr(c, m[0])), members)
    df = pd.DataFrame.from_records(methods, columns=['成员函数', '描述'])
    res.append(df2str(df) + newline)
    res = newline.join(res)

    # 4、使用chrome.exe浏览或输出到控制台
    #   这里底层可以封装一个chrome函数来调用，但是这个chrome需要依赖太多功能，故这里暂时手动简单调用
    if to_html:
        filename = writefile('', object_name, suffix='.html', root=Path.TEMP)
        with open(filename, 'w') as f:
            f.write(ensure_gbk(res))
            try:
                subprocess.run(['chrome.exe', filename])
            except FileNotFoundError:
                subprocess.run(['explorer', filename], shell=True)
                logging.warning('启动chrome.exe失败，可能是没有安装谷歌浏览器或配置环境变量。')
    else:  # linux环境直接输出表格
        print(res)

    return res


def intersection_split(a, b):
    """输入两个对象a,b，可以是dict或set类型，list等

    会分析出二者共有的元素值关系
    返回值是 ls1, ls2, ls3, ls4，大部分是list类型，但也有可能遵循原始情况是set类型
        ls1：a中，与b共有key的元素值
        ls2：a中，独有key的元素值
        ls3：b中，与a共有key的元素值
        ls4：b中，独有key的元素值
    """
    # 1、获得集合的key关系
    keys1 = set(a)
    keys2 = set(b)
    keys0 = keys1 & keys2  # 两个集合共有的元素

    # 2、组合出ls1、ls2、ls3、ls4

    def split(t, s, ks):
        """原始元素为t，集合化的值为s，共有key是ks"""
        if isinstance(t, (set, list, tuple)):
            return ks, s - ks
        elif isinstance(t, dict):
            ls1 = sorted(map(lambda x: (x, t[x]), ks), key=lambda x: natural_sort_key(x[0]))
            ls2 = sorted(map(lambda x: (x, t[x]), s - ks), key=lambda x: natural_sort_key(x[0]))
            return ls1, ls2
        else:
            dprint(type(s))  # s不是可以用来进行集合规律分析的类型
            raise ValueError

    ls1, ls2 = split(a, keys1, keys0)
    ls3, ls4 = split(b, keys2, keys0)
    return ls1, ls2, ls3, ls4


def dict2list(d: dict, *, nsort=False):
    """字典转n*2的list
    :param d: 字典
    :param nsort:
        True: 对key使用自然排序
        False: 使用d默认的遍历顺序
    :return:
    """
    ls = list(d.items())
    if nsort:
        ls = sorted(ls, key=lambda x: natural_sort_key(str(x[0])))
    return ls


def bcompare(oldfile, newfile=None, basefile=None, wait=True, named=True):
    """调用Beyond Compare软件对比两段文本（请确保有把BC的bcompare.exe加入环境变量）

    :param oldfile:
    :param newfile:
    :param basefile: 一般用于冲突合并是，oldfile、newfile共同依赖的旧版本
    :param wait: 见viewfiles的kwargs参数解释
        一般调用bcompare的时候，默认值wait=True，python是打开一个子进程并等待bcompare软件关闭后，才继续执行后续代码。
        如果你的业务场景并不需要等待bcompare关闭后执行后续python代码，可以设为wait=False，不等待。
    :param named:
        True，如果没有文件名，使用输入的变量作为文件名
        False，使用oldfile、newfile作为文件名
    :return: 程序返回被修改的oldfile内容
        注意如果没有修改，或者wait=False，会返回原始值
    这在进行调试、检测一些正则、文本处理算法是否正确时，特别方便有用
    >> bcompare('oldfile.txt', 'newfile.txt')

    180913周四：如果第1、2个参数都是set或都是dict，会进行特殊的文本化后比较
    """
    # 1、如果oldfile和newfile都是dict、set、list、tuple，则使用特殊方法文本化
    #   如果两个都是list，不应该提取key后比较，所以限制第1个类型必须是dict或set，然后第二个类型可以适当放宽条件
    if not oldfile: oldfile = str(oldfile)
    if isinstance(oldfile, (dict, set)) and isinstance(newfile, (dict, set, list, tuple)):
        t = [prettifystr(li) for li in intersection_split(oldfile, newfile)]
        oldfile = f'【共有部分】，{t[0]}\n\n【独有部分】，{t[1]}'
        newfile = f'【共有部分】，{t[2]}\n\n【独有部分】，{t[3]}'

    # 2、获取文件扩展名ext
    if Path(oldfile).is_file():
        ext = Path(oldfile).suffix
    elif Path(newfile).is_file():
        ext = Path(newfile).suffix
    elif Path(basefile).is_file():
        ext = Path(basefile).suffix
    else:
        ext = '.txt'  # 默认为txt文件

    # 3、生成所有文件
    ls = []
    names = func_input_message()['argnames']
    if not names[0]:
        names = ('oldfile.txt', 'newfile.txt', 'basefile.txt')

    def func(file, d):
        if file is not None:
            if Path(file).is_file():
                ls.append(file)
            else:
                ls.append(writefile(file, names[d] + ext, root=Path.TEMP))

    func(oldfile, 0)
    func(newfile, 1)
    func(basefile, 2)  # 注意这里不要写names[2]，因为names[2]不一定有存在

    # 4、调用程序（并计算外部操作时间）
    viewfiles('BCompare.exe', *ls, wait=wait)
    return Path(ls[0]).read()


____other = """
"""


class SingletonForEveryInitArgs(type):
    """Python单例模式(Singleton)的N种实现 - 知乎: https://zhuanlan.zhihu.com/p/37534850

    注意！注意！注意！重要的事说三遍！
    我的单例类不是传统意义上的单例类。
    传统意义的单例类，不管用怎样不同的初始化参数创建对象，永远都只有最初的那个对象类。
    但是我的单例类，为每种不同的参数创建形式，都构造了一个对象。
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        tag = f'{cls}{args}{kwargs}'  # id加上所有参数的影响来控制单例类
        # dprint(tag)
        if tag not in cls._instances:
            cls._instances[tag] = super(SingletonForEveryInitArgs, cls).__call__(*args, **kwargs)
        return cls._instances[tag]


def digit2weektag(d):
    """输入数字1~7，转为“周一~周日”

    >>> digit2weektag(1)
    '周一'
    >>> digit2weektag('7')
    '周日'
    """
    d = int(d)
    if 1 <= d <= 7:
        return '周' + '一二三四五六日'[d - 1]
    else:
        raise ValueError


if __name__ == '__main__':
    timer = Timer(start_now=True)

    timer.stop_and_report()
