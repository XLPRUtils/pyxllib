#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/01


import enum
import html

import pandas as pd
from bs4 import BeautifulSoup

from pyxllib.basic import *
from pyxllib.debug._1_typelib import dataframe_str
from pyxllib.debug._2_browser import getasizeof, browser


def getmembers(object, predicate=None):
    """自己重写改动的 inspect.getmembers

    TODO 这个实现好复杂，对于成员，直接用dir不就好了？
    """
    from inspect import isclass, getmro
    import types

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

    查阅一个对象的成员变量及成员方法
    为了兼容linux输出df时也能对齐，有几个中文域宽处理相关的函数

    :param c: 要处理的对象
    :param to_html:
        win32上默认True，用chrome、explorer打开
        linux上默认False，直接输出到控制台
    :param printf:
        默认是True，会输出到浏览器或控制条
        设为False则不输出
    """
    # 1 输出类表头
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

    # 2 html的样式精调
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

    # 3 添加成员变量和成员函数
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

    # 4 使用chrome.exe浏览或输出到控制台
    #   这里底层可以封装一个chrome函数来调用，但是这个chrome需要依赖太多功能，故这里暂时手动简单调用
    if to_html:
        filename = File(object_name, Dir.TEMP, suffix='.html'). \
            write(ensure_gbk(res), if_exists='delete').to_str()
        browser(filename)
    else:  # linux环境直接输出表格
        print(res)

    return res
