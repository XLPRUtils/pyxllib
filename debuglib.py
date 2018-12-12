#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽(chenkz@histudy.com)
# @Data   : 181211周二19:25
# @Version: 1.0.0

"""
通用工具包util的核心基础模块

文中所有后8个下划线_的是模块划分标记，且
    前8个下划线_是一级结构
    前16个下划线_是二级结构

重点推荐前三个模块的功能，这些功能的使用应该是非常高频的：
    Timer：计时器
    dprint、dformat：很好用的调试小工具
    showdir：查阅一个对象的成员变量及成员方法
其他一些模块使用频率不是那么高，读者根据自己需求摘选使用：
    file：文件复制删除等基本操作，读写内容、备份
    str：一些字符串化处理函数、装饰器
    chrome：调用第三方程序来查阅内容，包括bcompare文件比较
最后的common是一些零散的杂项函数，或者一些临时需要放到该公共底层文件的基础函数

"________file________"等字符串里有每个模块更详细的文档介绍，或demo示例函数: demo_timer()

相关文档： https://blog.csdn.net/code4101/article/details/83269101
"""

# 用import导入一些常用包
from copy import deepcopy
from pathlib import Path
import chardet
import collections
import datetime
import enum
import filecmp
import html
import inspect
import io
import logging
import os
import pickle
import pprint
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import timeit

try:
    import pandas as pd
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'pandas'])
    subprocess.run(['pip3', 'install', 'Jinja2'])  # pandas和showdir结合时有些文本输出功能需要
    import pandas as pd

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'BeautifulSoup4'])
    from bs4 import BeautifulSoup

try:
    import pyperclip  # mydatetagtool需要
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'pyperclip'])
    import pyperclip


________Timer_______ = """
"""


class Timer:
    """计时器类
    支持with语法调用

    代码实现参考了程明明的CmTimer： https://github.com/MingMingCheng/CmCode/blob/master/CmLib/Basic/CmTimer.h
    """

    def __init__(self, title=None, *, start_now=False):
        """
        :param title: 计时器名称
        :param start_now: 在构造函数时就开启计时器
        """
        # 不同的平台应该使用的计时器不同，这个直接用timeit中的配置最好
        self.default_timer = timeit.default_timer
        self.title = title
        self.start_clock = 0
        self.cumulative_clock = 0
        self.n_starts = 0
        if start_now:
            self.start()

    def start(self):
        self.n_starts += 1
        self.start_clock = self.default_timer()

    def stop(self):
        self.cumulative_clock += self.default_timer() - self.start_clock

    def reset(self):
        """重置计时器，清除所有累计值"""
        self.cumulative_clock = 0

    def report(self, prefix=''):
        pre = '' if self.title is None else f'{self.title} '
        print(f'{prefix}{pre}CumuTime: {self.time_in_seconds():.3f}s, '
              f'#run: {self.n_starts}, AvgTime: {self.avg_time():.3f}s')

    def stop_and_report(self, prefix=''):
        self.stop()
        self.report(prefix)

    def time_in_seconds(self):
        """计时器记录的总耗时"""
        return self.cumulative_clock

    def avg_time(self):
        """Timer对象可以多次start和stop，该函数可以返回多次统计的平均时间"""
        if self.n_starts:
            return self.time_in_seconds() / self.n_starts
        else:
            return 0

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        self.stop_and_report()


def demo_timer():
    """该函数也可以用来测电脑性能"""
    print('1、普通用法（循环5*1000万次用时）')
    timer = Timer('循环', start_now=True)
    for _ in range(5):
        for _ in range(10 ** 7):
            pass
    timer.stop_and_report()

    print('2、循环多轮计时（循环5*1000万次用时）')
    timer = Timer()
    for _ in range(5):
        timer.start()
        for _ in range(10 ** 7):
            pass
        timer.stop_and_report()

    print('3、with上下文用法')
    with Timer('循环'):
        for _ in range(5):
            for _ in range(10 ** 6):
                pass

    # 1、普通用法（循环5*1000万次用时）
    # 循环 CumuTime: 0.834s, #run: 1, AvgTime: 0.834s
    # 2、循环多轮计时（循环5*1000万次用时）
    # CumuTime: 0.157s, #run: 1, AvgTime: 0.157s
    # CumuTime: 0.314s, #run: 2, AvgTime: 0.157s
    # CumuTime: 0.470s, #run: 3, AvgTime: 0.157s
    # CumuTime: 0.626s, #run: 4, AvgTime: 0.157s
    # CumuTime: 0.787s, #run: 5, AvgTime: 0.157s
    # 3、上下文用法
    # 循环 CumuTime: 0.080s, #run: 1, AvgTime: 0.080s


def demo_system():
    def demo_pc_messages():
        """演示如何获取当前操作系统的PC环境数据"""
        # fqdn，fully qualified domain name
        print('1、socket.getfqdn() :', socket.getfqdn())  # 完全限定域名，可以理解成pcname，计算机名
        # 注意py的很多标准库功能本来就已经处理了不同平台的问题，尽量用标准库而不是自己用sys.platform作分支处理
        print('2、sys.platform     :', sys.platform)  # 运行平台，一般是win32和linux
        li = os.getenv('PATH').split(os.path.pathsep)  # 环境变量PATH，win中不区分大小写，linux中区分大小写，所以必须写大写
        print("3、os.getenv('PATH'):", f'len={len(li)},', pprint.pformat(li, 4))

    def demo_executable_messages():
        """演示如何获取被执行程序相关的数据"""
        print('1、sys.path      :', f'len={len(sys.path)},', pprint.pformat(sys.path, 4))  # import绝对位置包的搜索路径
        print('2、sys.executable:', sys.executable)  # 当前被执行脚本位置
        print('3、sys.version   :', sys.version)  # python的版本
        print('4、os.getcwd()   :', os.getcwd())  # 获得当前工作目录
        print('5、gettempdir()  :', tempfile.gettempdir())  # 临时文件夹位置

    # TODO 增加性能测试？
    timer = Timer('demo_system', start_now=True)
    print('>>> demo_pc_messages()')
    demo_pc_messages()
    print('>>> demo_executable_messages()')
    demo_executable_messages()
    timer.stop_and_report()


________dprint________ = """
监控变量数值
"""


def brieftype(c):
    """简化输出的type类型
    >>> brieftype(123)
    'int'
    """
    s = str(type(c))[8:-2]
    return s


def func_input_message(depth=2) -> dict:
    """假设调用了这个函数的函数叫做f，这个函数会获得
        调用f的时候输入的参数信息，返回一个dict，键值对为
            fullfilename：完整文件名
            filename：文件名
            funcname：所在函数名
            lineno：代码所在行号
            comment：尾巴的注释

            argnames：变量名（list），这里的变量名也有可能是一个表达式
            types：变量类型（list），如果是表达式，类型指表达式的运算结果类型
            argvals：变量值（list）

        这样以后要加新的键值对也很方便

        :param depth: 需要分析的层级
            0，当前func_input_message函数的参数输入情况
            1，调用func_input_message的函数 f 参数输入情况
            2，调用f的函数 g ，参数输入情况

        参考： func_input_message 的开发使用具体方法可以参考dformat函数
        花絮：inspect可以获得函数签名，也可以获得一个函数各个参数的输入值，但我想要展现的是原始表达式，
            例如func(a)，以func(1+2)调用，inpect只能获得“a=3”，但我想要的是“1+2=3”的效果
    """
    res = {}
    # 1、找出调用函数的代码
    ss = inspect.stack()
    frameinfo = ss[depth]
    arginfo = inspect.getargvalues(ss[depth - 1][0])
    if arginfo.varargs:
        origin_args = arginfo.locals[arginfo.varargs]
    else:
        origin_args = list(map(lambda x: arginfo.locals[x], arginfo.args))

    res['fullfilename'] = frameinfo.filename
    res['filename'] = os.path.basename(frameinfo.filename)
    res['funcname'] = frameinfo.function
    res['lineno'] = frameinfo.lineno

    if frameinfo.code_context:
        code_line = frameinfo.code_context[0].strip()
    else:  # 命令模式无法获得代码，是一个None对象
        code_line = ''

    funcname = ss[depth - 1].function  # 调用的函数名
    # 这一行代码不一定是从“funcname(”开始，所以要用find找到开始位置
    code = code_line[code_line.find(funcname + '(') + len(funcname):]

    # 2、先找到函数的()中参数列表，需要以')'作为分隔符分析
    ls = code.split(')')
    logo, i = True, 1
    while logo and i <= len(ls):
        # 先将'='做特殊处理，防止字典类参数导致的语法错误
        s = ')'.join(ls[:i]).replace('=', '+') + ')'
        try:
            compile(s, '<string>', 'single')
        except SyntaxError:
            i += 1
        else:  # 正常情况
            logo = False
    code = ')'.join(ls[:i])[1:]

    # 3、获得注释
    # 这个注释实现的不是很完美，不过影响应该不大，还没有想到比较完美的解决方案
    t = ')'.join(ls[i:])
    comment = t[t.find('#'):] if '#' in t else ''
    res['comment'] = comment

    # 4、获得变量名
    ls = code.split(',')
    n = len(ls)
    argnames = list()
    i, j = 0, 1
    while j <= n:
        s = ','.join(ls[i:j])
        try:
            compile(s.lstrip(), '<string>', 'single')
        except SyntaxError:
            j += 1
        else:  # 没有错误的时候执行
            argnames.append(s.strip())
            i = j
            j = i + 1

    # 5、获得变量值和类型
    res['argvals'] = origin_args
    res['types'] = list(map(brieftype, origin_args))

    if not argnames:  # 如果在命令行环境下调用，argnames会有空，需要根据argvals长度置空名称
        argnames = [''] * len(res['argvals'])
    res['argnames'] = argnames

    return res


def dformat(*args, depth=2,
            delimiter=' ' * 4,
            strfunc=repr,
            fmt='{filename}/{funcname}/{lineno}: {argmsg}    {comment}',
            subfmt='{name}<{tp}>={val}'):
    """
    :param args:  需要检查的表达式
        这里看似没有调用，其实在func_input_message用inspect会提取到args的信息
    :param depth: 处理对象
        默认值2，即处理dformat本身
        2以下值没意义
        2以上的值，可以不传入args参数
    :param delimiter: 每个变量值展示之间的分界
    :param strfunc: 对每个变量值的文本化方法，常见的有repr、str
    :param fmt: 展示格式，除了func_input_message中的关键字，新增
        argmsg：所有的「变量名=变量值」，或所有的「变量名<变量类型>=变量值」，或自定义格式，采用delimiter作为分界符
    :param subfmt: 自定义每个变量值对的显示形式
        name，变量名
        val，变量值
        tp，变量类型
    :return: 返回格式化好的文本字符串
    """
    res = func_input_message(depth)
    ls = [subfmt.format(name=name, val=strfunc(val), tp=tp)
          for name, val, tp in zip(res['argnames'], res['argvals'], res['types'])]
    res['argmsg'] = delimiter.join(ls)
    return fmt.format(**res)


def dprint(*args, **kwargs):
    """
    # 故意写的特别复杂，测试在极端情况下是否能正确解析出表达式
    >> re.sub(str(dprint(1, b, a, "aa" + "bb)", "a[,ba\nbb""b", [2, 3])), '', '##')  # 注释 # 注
    corelib.py/<module>/304: 1<int>=1    b<int>=2    a<int>=1    "aa" + "bb)"<str>='aabb)'
                                "a[,ba\nbb""b"<str>='a[,ba\nbbb'    ##')  # 注释 # 注
    """
    print(dformat(depth=3, **kwargs))


________showdir________ = """
为了兼容linux输出df时也能对齐，有几个中文域宽处理相关的函数
"""


def east_asian_len(s, ambiguous_width=None):
    from pandas.compat import east_asian_len as func
    if ambiguous_width is None:
        ambiguous_width = 2 if sys.platform == 'win32' else 1
    return func(s, ambiguous_width=ambiguous_width)


def east_asian_shorten(s, width=50, placeholder='...'):
    """考虑中文情况下的域宽截断

    :param s: 要处理的字符串
    :param width: 宽度上限，仅能达到width-1的宽度
    :param placeholder: 如果做了截断，末尾补足字符

    >>> east_asian_shorten('a啊b'*4, 5, '...')
    'a...'
    >>> east_asian_shorten('a啊b'*4, 11)
    'a啊ba啊...'
    >>> east_asian_shorten('a啊b'*4, 16, '...')
    'a啊ba啊ba啊b...'
    >>> east_asian_shorten('a啊b'*4, 18, '...')
    'a啊ba啊ba啊ba啊b'
    """
    width -= 1
    # 1、如果输入的width比placeholder还短
    t = east_asian_len(placeholder)
    if width <= t:
        return placeholder[:width]

    # 2、计算长度
    width -= t
    # 用textwrap的折行功能，尽量不删除文本，采用width的2倍一般够后面用了
    s = textwrap.shorten(s, width * 2, placeholder='')
    n = len(s)
    if east_asian_len(s) <= width:
        return s

    # 3、截取s
    try:
        s = s.encode('gbk')[:width].decode('gbk', errors='ignore')
    except UnicodeEncodeError:
        i, count = 0, t
        while i < n and count <= width:
            if ord(s[i]) > 127:
                count += 2
            else:
                count += 1
            i += 1
        s = s[:i]

    return s + placeholder


def dataframe_str(df, *args, ambiguous_as_wide=None, shorten=True):
    """输出DataFrame
    DataFrame可以直接输出的，这里是增加了对中文字符的对齐效果支持

    :param df: DataFrame数据结构
    :param args: option_context格式控制
    :param ambiguous_as_wide: 是否对①②③这种域宽有歧义的设为宽字符
        win32平台上和linux上①域宽不同，默认win32是域宽2，linux是域宽1
    :param shorten: 是否对每个元素提前进行字符串化并控制长度在display.max_colwidth以内
        因为pandas的字符串截取遇到中文是有问题的，可以用我自定义的函数先做截取
        默认开启，不过这步比较消耗时间

    >> df = pd.DataFrame({'哈哈': ['a'*100, '哈\n①'*10, 'a哈'*100]})
                                                        哈哈
        0  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...
        1   哈 ①哈 ①哈 ①哈 ①哈 ①哈 ①哈 ①哈 ①哈 ①...
        2  a哈a哈a哈a哈a哈a哈a哈a哈a哈a哈a哈a哈a哈a哈a哈a...
    """
    if ambiguous_as_wide is None:
        ambiguous_as_wide = sys.platform == 'win32'
    with pd.option_context('display.unicode.east_asian_width', True,  # 中文输出必备选项，用来控制正确的域宽
                           'display.unicode.ambiguous_as_wide', ambiguous_as_wide,
                           'max_columns', 20,  # 最大列数设置到20列
                           'display.width', 200,  # 最大宽度设置到200
                           *args):
        if shorten:  # applymap可以对所有的元素进行映射处理，并返回一个新的df
            df = df.applymap(lambda x: east_asian_shorten(str(x), pd.options.display.max_colwidth))
        s = str(df)
    return s


def getasizeof(*objs, **opts):
    """获得所有类的大小，底层用pympler.asizeof实现"""
    try:
        from pympler import asizeof
    except ModuleNotFoundError:
        subprocess.run(['pip3', 'install', 'pympler'])
        from pympler import asizeof
    return asizeof.asizeof(*objs, **opts)


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
    methods = filter(lambda m: not callable(getattr(c, m[0])), inspect.getmembers(c))
    ls = []
    for ele in methods:
        k, v = ele
        if k.endswith(r'________'):  # 这个名称的变量是我代码里的特殊标记，不显示
            continue
        attr = getattr(c, k)
        if isinstance(attr, enum.IntFlag):  # 对re.RegexFlag等枚举类输出整数值
            v = brieftype(attr) + '，' + str(int(attr)) + '，' + str(v)
        else:
            v = brieftype(attr) + '，' + str(v)
        ls.append([k, v])
    df = pd.DataFrame.from_records(ls, columns=['成员变量', '描述'])
    res.append(df2str(df) + newline)

    # 成员函数
    methods = filter(lambda m: callable(getattr(c, m[0])), inspect.getmembers(c))
    df = pd.DataFrame.from_records(methods, columns=['成员函数', '描述'])
    res.append(df2str(df) + newline)

    res = newline.join(res)

    # 4、使用chrome.exe浏览或输出到控制台
    #   这里底层可以封装一个chrome函数来调用，但是这个chrome需要依赖太多功能，故这里暂时手动简单调用
    if to_html:
        filename = os.path.join(tempfile.gettempdir(), object_name + '.html')
        with open(filename, 'w') as f:
            f.write(res)
            try:
                subprocess.run(['chrome.exe', filename])
            except FileNotFoundError:
                subprocess.run(['explorer', filename], shell=True)
                logging.warning('启动chrome.exe失败，可能是没有安装谷歌浏览器或配置环境变量。')
    else:  # linux环境直接输出表格
        print(res)

    return res


________file________ = """
主要是为了提供readfile、wrritefile函数
与普通的读写文件相比，有以下优点：
1、智能识别pkl等特殊格式文件的处理
2、智能处理编码
3、目录不存在自动创建
4、自动备份旧文件，而不是强制覆盖写入

其他相关文件处理组件：isfile、get_encoding、ensure_folders
以及同时支持文件或文件夹的对比复制删除等操作的函数：filescmp、filesdel、filescopy
文件备份相关：filebackup_time、filebackup
"""


def isfile(fn):
    """判断输入的是不是合法的路径格式，且存在确实是一个文件"""
    try:
        return os.path.isfile(fn)
    except ValueError:  # 出现文件名过长的问题
        return False
    except TypeError:  # 输入不是字符串类型
        return False


def get_encoding(bstr):
    """输入二进制字符串，返回文件编码，并会把GB2312改为GBK
    :return: 'utf-8' 或 'GBK'

    备注：又想正常情况用chardet快速识别，又想异常情况暴力编码试试，代码就写的这么又臭又长了~~
    """
    # 1、读取编码
    detect = None
    if isinstance(bstr, bytes):  # 如果输入是一个二进制字符串流则直接识别
        detect = chardet.detect(bstr)
        encoding = detect['encoding']
    elif isfile(bstr):  # 如果是文件，则按二进制打开
        # 如果输入是一个文件名则进行读取
        with open(bstr, 'rb') as f:  # 以二进制读取文件，注意二进制没有\r\n的值
            bstr = f.read()
        encoding = get_encoding(bstr)
    else:  # 其他类型不支持
        return 'utf-8'
    # 检测结果存储在encoding

    # 2、智能适应优化，最终应该只能是gbk、utf8两种结果中的一种
    if encoding in ('ascii', 'utf-8', 'ISO-8859-1'):
        # 对ascii类编码，理解成是utf-8编码；ISO-8859-1跟ASCII差不多
        encoding = 'utf-8'
    elif encoding in ('GBK', 'GB2312'):
        encoding = 'GBK'
    elif bstr.strip():  # 如果bstr非空
        # 进入这个if分支算是比较异常的情况，会输出原识别结果detect
        try:  # 先尝试utf8编码，如果编码成功则认为是utf8
            bstr.decode('utf8')
            encoding = 'utf-8'
            dprint(detect)  # chardet编码识别异常，根据文件内容已优化为utf8编码
        except UnicodeDecodeError:
            try:  # 否则尝试gbk编码
                bstr.decode('gbk')
                encoding = 'GBK'
                dprint(detect)  # chardet编码识别异常，根据文件内容已优化为gbk编码
            except UnicodeDecodeError:  # 如果两种都有问题
                encoding = 'utf-8'
                dprint(detect)  # 警告：chardet编码识别异常，已强制使用utf8处理
    else:
        encoding = 'utf-8'

    return encoding


def readfile(fn=None, *, encoding=None):
    """根据文件后缀名，自动识别需要转换的对象
    pkl、txt等
    """
    if fn is None:
        return sys.stdin.read()  # 注意在PyCharm输入是按 Ctrl + D 结束，在cmd是按Ctrl + Z 结束
    elif isfile(fn):  # 如果存在这样的文件，那就读取文件内容
        # 获得文件扩展名，并统一转成小写
        _, ext = os.path.splitext(fn)
        ext = ext.lower()
        if ext == '.pkl':  # pickle库
            with open(fn, 'rb') as f:
                return pickle.load(f)
        else:
            with open(fn, 'rb') as f:
                bstr = f.read()
            if not encoding: encoding = get_encoding(bstr)
            s = bstr.decode(encoding=encoding, errors='ignore')
            if '\r' in s: s = s.replace('\r\n', '\n')  # 如果用\r\n作为换行符会有一些意外不好处理
            return s
    else:  # 非文件对象返回原值
        return fn


def ensure_folders(*dsts):
    """当目录不存在的时候，自动创建目录，包括中间目录
    dst
        输入'A/B/C/'、'A/B/C/a.txt'，均会创建A/B/C目录
        但'A/B/C'，只会创建A/B目录

    可以使用 ensure_folders('A/', 'B/') 同时确保存在A, B两个目录
    """
    for dst in dsts:
        folder = os.path.split(dst)[0]  # path.split可以拆分目录名和文件名
        if folder and not os.path.exists(folder):
            # dprint(dst, folder)
            os.makedirs(os.path.dirname(dst))  # os.makedirs可以创建中间目录


def filescmp(f1, f2, shallow=True):
    """相同返回True，有差异返回False

    :param f1: 待比较的第1个文件（文件夹）
    :param f2: 待比较的第2个文件（文件夹）
    :param shallow: 默认True，即是利用os.stat()返回的基本信息进行比较
        例如其中的文件大小，但修改时间等是不影响差异判断的
        如果设为False，则会打开比较具体内容，速度会慢一点
    """
    if os.path.isfile(f1):
        cmp = filecmp.cmp(f1, f2, shallow)
    else:  # TODO：文件比较功能其实还有瑕疵，并不能准确的比出差异，后续要优化
        cmp = filecmp.dircmp(f1, f2, shallow)
    return cmp


def filesdel(*dsts):
    """删除文件或文件夹"""
    for d in dsts:
        if os.path.exists(d):
            if os.path.isfile(d):
                os.remove(d)
            else:
                # os.removedirs(d)  # 这个只能移除空文件夹
                shutil.rmtree(d)


def filescopy(src, dst, root_dir=None):
    """会自动添加不存在的目录的拷贝"""
    # TODO: 添加目录拷贝功能
    if root_dir:
        src = os.path.join(root_dir, src)
        dst = os.path.join(root_dir, dst)
    ensure_folders(dst)  # 确保目标目录存在
    if os.path.exists(src):
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        else:
            try:
                shutil.copytree(src, dst)
            except FileExistsError:
                dprint(Path(os.curdir).resolve(), src, dst)  # warning：目标文件夹已存在，不做复制


def filebackup_time(s):
    """在FileBackUpTime上做了优化
    备份文件都遵循特定的命名规范
    如果是文件，是：'chePre 171020-153959.tex'
    如果是目录，是：'figs 171020-153959'
    通过后缀分析，可以判断这是不是一个备份文件

    >>> filebackup_time('chePre 171020-153959.tex')
    '171020-153959'
    >>> filebackup_time('figs 171020-153959')
    '171020-153959'
    >>> filebackup_time('figs 171020')
    ''
    """
    name, _ = os.path.splitext(s)  # 去除扩展名
    if len(name) < 14:
        return ''
    name = name[-13:]
    g = re.match(r'(\d{6}-\d{6})', name)
    if g:
        return g.group(1)
    return ''


def filebackup(filename, remove_origin_file=False, tail=None):
    """对fileName这个文件末尾添加时间戳备份文件
    也可以使用自定义标记tail

    :param filename: 需要处理的文件或文件夹
        如果不存在，函数直接结束，返回None
    :param remove_origin_file: 是否删除原文件
    :param tail: 自定义添加后缀
        tail为None时，默认添加特定格式的时间戳

    # TODO：有个小bug，如果在不同时间实际都是相同一个文件，也会被不断反复备份
    #    如果想解决这个，就要读取目录下最近的备份文件对比内容了
    """
    # 1、导入包，以及判断文件是否存在，不存在则返回None
    from os.path import exists, getmtime, splitext

    if not exists(filename):
        return None

    # 2、计算出新名称
    if not tail:
        tail = time.strftime(' %y%m%d-%H%M%S', time.localtime(getmtime(filename)))  # 时间戳
    name, ext = splitext(filename)
    targetname = name + tail + ext

    # 3、如果目标已经存在，要进行一些比较判断
    if exists(targetname):
        # 文件和文件夹分别处理
        if filescmp(filename, targetname):  # 文件是一样的就不用处理了
            pass
        else:  # 如果文件确实是不一样的，则进行二次备份
            filebackup(targetname, remove_origin_file=True)

    # 4、如果没有目标文件，生成
    if not exists(targetname):
        # 上一步可能会删除targetname，所以这里要重新判断是否存在
        if remove_origin_file:
            os.rename(filename, targetname)
        else:
            filescopy(filename, targetname)

    # 5、补充处理，是否删除原文件
    if remove_origin_file and exists(filename):
        filesdel(filename)


def writefile(ob, fn=None, *, encoding=None, backup=True, ext=None, temp=False):
    """往文件fn写入ob

    :param ob: 写入的文件内容，如果要写文本文件且ob不是文本对象，只会进行简单的字符串化
    :param fn: 写入的文件名，不设置时，默认写入到「临时文件夹」，随机生成无后缀文件名
    :param encoding: 强制写入的编码
    :param backup: 如果写入的文件已存在，是否进行备份
    :param ext: 用来写真实文件后缀类型，例如txt文件名强制使用ext='pkl'编译
    :param temp: 将文件写到临时文件夹
    :return: 返回写入的文件名，这个主要是在写临时文件时有用
    """
    # 1、确定写入的文件名
    if temp:
        folder = tempfile.gettempdir()
    else:
        folder = os.getcwd()
    # 如果fn是绝对路径，则无论temp、folder是什么值都不受影响
    #   如果fn是相对路径，则会放置到相应folder里
    if fn:
        fn = os.path.join(folder, fn)
    else:  # 利用tempfile生成一个文件名，但是并不用其自动保存功能，毕竟在windows上也是有bug的
        fn = tempfile.mktemp(dir=folder)
    # 确保文件夹存在
    ensure_folders(os.path.dirname(fn) + '/')

    # 2、将目标内容写到一个临时文件
    if not ext: _, ext = os.path.splitext(fn)
    tf = tempfile.mktemp()
    if ext == '.pkl':  # pickle库
        with open(tf, 'wb') as f:
            pickle.dump(ob, f)
    else:  # 其他类型认为是文本类型
        # 如果没有输入encoding控制，那就以原文件的编码为准啦
        if not encoding: encoding = get_encoding(fn)
        with open(tf, 'w', errors='ignore', encoding=encoding) as f:
            f.write(str(ob))

    # 3、备份相关处理
    exist = isfile(fn)
    same = exist and filescmp(fn, tf)
    if backup and exist and not same: filebackup(fn)
    if not exist or not same: filescopy(tf, fn)
    filesdel(tf)
    return fn


________str_______ = """
"""


class StrDecorator:
    """将函数的返回值字符串化，仅调用朴素的str字符串化

    装饰器开发可参考： https://mp.weixin.qq.com/s/Om98PpncG52Ba1ZQ8NIjLA
    """

    def __init__(self, func):
        self.func = func  # 使用self.func可以索引回原始函数名称
        self.last_raw_res = None  # last raw result，上一次执行函数的原始结果

    def __call__(self, *args, **kwargs):
        self.last_raw_res = self.func(*args, **kwargs)
        return str(self.last_raw_res)


def prettifystr(s):
    """对一个对象用更友好的方式字符串化

    :param s: 输入类型不做限制，会将其以友好的形式格式化
    :return: 格式化后的字符串
    """
    title = ''
    if isinstance(s, str):
        pass
    elif isinstance(s, collections.Counter):  # Counter要按照出现频率显示
        s = s.most_common()
        title = f'collections.Counter长度：{len(s)}\n'
        df = pd.DataFrame.from_records(s, columns=['value', 'count'])
        s = dataframe_str(df)
    elif isinstance(s, (list, tuple)):  # 每个元素占一行，并带有编号输出
        title = f'{brieftype(s)}长度：{len(s)}\n'
        s = pprint.pformat(s)
    elif isinstance(s, (dict, set)):  # 每个元素占一行，并带有编号输出
        title = f'{brieftype(s)}长度：{len(s)}\n'
        s = pprint.pformat(s)
    else:  # 其他的采用默认的pformat
        s = pprint.pformat(s)
    return title + s


class PrettifyStrDecorator:
    """将函数的返回值字符串化（调用 prettifystr 美化）"""

    def __init__(self, func):
        self.func = func  # 使用self.func可以索引回原始函数名称
        self.last_raw_res = None  # last raw result，上一次执行函数的原始结果

    def __call__(self, *args, **kwargs):
        self.last_raw_res = self.func(*args, **kwargs)
        return prettifystr(self.last_raw_res)


class PrintDecorator:
    """将函数返回结果直接输出"""

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        s = self.func(*args, **kwargs)
        print(s)
        return s  # 输出后仍然会返回原函数运行值


def natural_sort_key(key):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    return [convert(c) for c in re.split('([0-9]+)', str(key))]


def natural_sort(ls):
    """自然排序"""
    return sorted(ls, key=natural_sort_key)


________chrome_______ = """
viewfiles是基础组件，负责将非文本对象文本化后写到临时文件，
调用外部程序时，是否wait，以及找不到程序时的报错提示。
"""


def pathext(fn):
    """得到文件的扩展名"""
    return os.path.splitext(fn)[1]


def viewfiles(procname, *files, **kwargs):
    """调用procname相关的文件程序打开files
    :param procname: 程序名
    :param files: 一个文件名参数清单，每一个都是文件路径，或者是字符串等可以用writefile转成文件的路径
    :param kwargs:
        save: 如果True，则会按时间保存文件名；否则采用特定名称，每次运行就会把上次的覆盖掉
        wait: 是否等待当前进程结束后，再运行后续py代码
        filename: 控制写入的文件名
        TODO：根据不同软件，这里还可以扩展很多功能
    :param kwargs:
        wait:
            True：在同一个进程中执行子程序，即会等待bc退出后，再进入下一步
            False：在新的进程中执行子程序

    细节：注意bc跟其他程序有比较大不同，建议使用专用的bcompare函数
    目前已知可以扩展多文件的有：chrome、notepad++、texstudio

    >> ls = list(range(100))
    >> viewfiles('notepad++', ls, save=True)
    """
    # 1、生成文件名
    ls = []  # 将最终所有绝对路径文件名存储到ls
    save = kwargs.get('save')

    basename = ext = None
    if 'filename' in kwargs and kwargs['filename']:
        basename, ext = os.path.splitext(kwargs['filename'])

    for i, t in enumerate(files):
        if isfile(t):
            ls.append(t)
        else:
            # 如果要保存，则设为None，程序会自动按时间戳存储，否则设为特定名称的文件，下次运行就会把上次的覆盖了
            bn = None if save else f'file{i+1}'
            if basename:
                bn = basename
            ls.append(writefile(t, bn+ext, temp=True))

    # 2、调用程序（并计算外部操作时间）
    timer = Timer(start_now=True)
    try:
        if kwargs.get('wait'):
            subprocess.run([procname, *ls])
        else:
            subprocess.Popen([procname, *ls])
    except FileNotFoundError:
        raise FileNotFoundError(f'未找到程序：{procname}。请检查是否有安装及设置了环境变量。')
    timer.stop()
    return timer.time_in_seconds()


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
            return s, s - ks
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
    程序返回bc外部操作的时间

    :param oldfile:
    :param newfile:
    :param basefile: 一般用于冲突合并是，oldfile、newfile共同依赖的旧版本
    :param wait: 见viewfiles的kwargs参数解释
    :param named:
        True，如果没有文件名，使用输入的变量作为文件名
        False，使用oldfile、newfile作为文件名

    这在进行调试、检测一些正则、文本处理算法是否正确时，特别方便有用
    >> bcompare('oldfile.txt', 'newfile.txt')

    180913周四：如果第1、2个参数都是set或都是dict，会进行特殊的文本化后比较
    """
    # 1、如果oldfile和newfile都是dict、set、list、tuple，则使用特殊方法文本化
    #   如果两个都是list，不应该提取key后比较，所以限制第1个类型必须是dict或set，然后第二个类型可以适当放宽条件
    if isinstance(oldfile, (dict, set)) and isinstance(newfile, (dict, set, list, tuple)):
        t = list(map(prettifystr, intersection_split(oldfile, newfile)))
        oldfile = f'共有部分，{t[0]}\n\n独有部分，{t[1]}'
        newfile = f'共有部分，{t[2]}\n\n独有部分，{t[3]}'

    # 2、获取文件扩展名ext
    if isfile(oldfile): ext = pathext(oldfile)
    elif isfile(newfile): ext = pathext(newfile)
    elif isfile(basefile): ext = pathext(basefile)
    else: ext = '.txt'  # 默认为txt文件

    # 3、生成所有文件
    ls = []
    names = func_input_message()['argnames']
    if not names[0]:
        names = ('oldfile.txt', 'newfile.txt', 'basefile.txt')
    if oldfile is not None:
        if isfile(oldfile):
            ls.append(oldfile)
        else:
            ls.append(writefile(oldfile, names[0]+ext, temp=True))
    if newfile is not None:
        if isfile(newfile):
            ls.append(newfile)
        else:
            ls.append(writefile(newfile, names[1]+ext, temp=True))
    if basefile is not None:
        if isfile(basefile):
            ls.append(basefile)
        else:
            ls.append(writefile(basefile, names[2]+ext, temp=True))

    # 4、调用程序（并计算外部操作时间）
    return viewfiles('BCompare.exe', *ls, wait=wait)


def chrome(file, filename=None, **kwargs):
    r"""使用谷歌浏览器查看内容，详细用法见底层函数viewfiles
    >> chrome(r'C:\Users\kzche\Desktop\b.xml')  # 使用chrome查看文件内容
    >> chrome('aabb')  # 使用chrome查看一个字符串值
    >> chrome([123, 456])  # 使用chrome查看一个变量值
    """
    if isinstance(file, dict):
        file = dict2list(file, nsort=True)
        file = pd.DataFrame.from_records(file, columns=('key', 'value'), **kwargs)

    if isinstance(file, (list, tuple)):
        try:  # 能转DataFrame处理的转DateFrame
            file = pd.DataFrame.from_records(file, **kwargs)
        except TypeError:  # TypeError: 'int' object is not iterable
            pass

    if isinstance(file, pd.DataFrame):  # DataFrame在网页上有更合适的显示效果
        df = file
        if not filename: filename = 'a.html'
        if not filename.endswith('.html'): filename += '.html'
        viewfiles('chrome.exe', df.to_html(), filename=filename)
    else:
        viewfiles('chrome.exe', file, filename=filename)


________common_______ = """
"""


def strfind(fullstr, objstr, *, start=None, times=0, overlap=False):
    """进行强大功能扩展的的字符串查找函数。
    TODO 性能有待优化

    :param fullstr: 原始完整字符串
    >>> strfind('aabbaabb', 'bb')  # 函数基本用法
    2

    :param objstr: 需要查找的目标字符串，可以是一个list或tuple
    >>> strfind('bbaaaabb', 'bb') # 查找第1次出现的位置
    0
    >>> strfind('aabbaabb', 'bb', times=1) # 查找第2次出现的位置
    6
    >>> strfind('aabbaabb', 'cc') # 不存在时返回-1
    -1
    >>> strfind('aabbaabb', ['aa', 'bb'], times=2)
    4

    :param start: 起始查找位置。默认值为0，当times<0时start的默认值为-1。
    >>> strfind('aabbaabb', 'bb', start=2) # 恰好在起始位置
    2
    >>> strfind('aabbaabb', 'bb', start=3)
    6
    >>> strfind('aabbaabb', ['aa', 'bb'], start=5)
    6

    :param times: 定位第几次出现的位置，默认值为0，即从前往后第1次出现的位置。
        如果是负数，则反向查找，并返回的是目标字符串的起始位置。
    >>> strfind('aabbaabb', 'aa', times=-1)
    4
    >>> strfind('aabbaabb', 'aa', start=5, times=-1)
    4
    >>> strfind('aabbaabb', 'aa', start=3, times=-1)
    0
    >>> strfind('aabbaabb', 'bb', start=7, times=-1)
    6

    :param overlap: 重叠情况是否重复计数
    >>> strfind('aaaa', 'aa', times=1)  # 默认不计算重叠部分
    2
    >>> strfind('aaaa', 'aa', times=1, overlap=True)
    1

    """

    def nonnegative_min_value(*arr):
        """计算出最小非负整数，如果没有非负数，则返回-1"""
        arr = tuple(filter(lambda x: x >= 0, arr))
        return min(arr) if arr else -1

    def nonnegative_max_value(*arr):
        """计算出最大非负整数，如果没有非负数，则返回-1"""
        arr = tuple(filter(lambda x: x >= 0, arr))
        return max(arr) if arr else -1

    # 1、根据times不同，start的初始默认值设置方式也不同
    if times < 0 and start is None:
        start = len(fullstr) - 1  # 反向查找start设到末尾字符-1
    if start is None:
        start = 0  # 正向查找start设为0
    p = -1  # 记录答案位置，默认找不到

    # 2、单串匹配
    if isinstance(objstr, str):  # 单串匹配
        offset = 1 if overlap else len(objstr)  # overlap影响每次偏移量

        # A、正向查找
        if times >= 0:
            p = start - offset
            for _ in range(times + 1):
                p = fullstr.find(objstr, p + offset)
                if p == -1:
                    return -1

        # B、反向查找
        else:
            p = start + offset + 1
            for _ in range(-times):
                p = fullstr.rfind(objstr, 0, p - offset)
                if p == -1:
                    return -1

    # 3、多模式匹配（递归调用，依赖单串匹配功能）
    else:
        # A、正向查找
        if times >= 0:
            p = start - 1
            for _ in range(times + 1):
                # 把每个目标串都找一遍下一次出现的位置，取最近的一个
                #   因为只找第一次出现的位置，所以overlap参数传不传都没有影响
                # TODO 需要进行性能对比分析，有必要的话后续可以改AC自动机实现多模式匹配
                ls = tuple(map(lambda x: strfind(fullstr, x, start=p + 1, overlap=overlap), objstr))
                p = nonnegative_min_value(*ls)
                if p == -1:
                    return -1

        # B、反向查找
        else:
            p = start + 1
            for _ in range(-times):  # 需要循环处理的次数
                # 使用map对每个要查找的目标调用strfind
                ls = tuple(map(lambda x: strfind(fullstr, x, start=p - 1, times=-1, overlap=overlap), objstr))
                p = nonnegative_max_value(*ls)
                if p == -1:
                    return -1

    return p


class Stdout:
    """重定向标准输出流，切换print标准输出位置
    使用with语法调用
    """

    def __init__(self, path=None, mode='w'):
        """
        :param path: 可选参数
            如果是一个合法的文件名，在__exit__时，会将结果写入文件
            如果不合法不报错，只是没有功能效果
        :param mode: 写入模式
            'w': 默认模式，直接覆盖写入
            'a': 追加写入
        """
        self.origin_stdout = sys.stdout
        self._path = path
        self._mode = mode
        self.strout = io.StringIO()
        self.result = None

    def __enter__(self):
        sys.stdout = self.strout
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.origin_stdout
        self.result = str(self)

        # 如果输入的是一个合法的文件名，则将中间结果写入
        if not self._path:
            return

        try:
            with open(self._path, self._mode) as f:
                f.write(self.result)
        except TypeError as e:
            logging.exception(e)
        except FileNotFoundError as e:
            logging.exception(e)

        self.strout.close()

    def __str__(self):
        """在这个期间获得的文本内容"""
        if self.result:
            return self.result
        else:
            return self.strout.getvalue()


def str2date(s):
    """
    >>> str2date('180823')
    datetime.date(2018, 8, 23)
    """
    year = int(s[:2])
    year = 2000 + year if year < 50 else 1900 + year  # 小于50默认是20xx年，否则认为是19xx年
    d = datetime.date(year, int(s[2:4]), int(s[4:]))
    return d


def shorten(s, width=200, placeholder='...'):
    """
    >>> shorten('aaa', 10)
    'aaa'
    >>> shorten('hell world! 0123456789 0123456789', 11)
    'hell world!'
    >>> shorten("Hello  world!", width=12)
    'Hello world!'
    >>> shorten("Hello  world!", width=11)
    'Hello world'

    textwrap.shorten有placeholder参数，但我这里暂时还没用这个参数值

    我在textwrap.shorten使用中发现了一个bug，所以才打算自己写一个shorten的：
    >>> textwrap.shorten('0123456789 0123456789', 11)  # 全部字符都被折叠了
    '[...]'
    >>> shorten('0123456789 0123456789', 11)  # 自己写的shorten
    '0123456789 '
    """
    s = re.sub(r'\s+', ' ', str(s))
    n = len(s)
    if n > width:
        s = s[:width]
    return s

    # return textwrap.shorten(str(s), width)


def strwidth(s):
    """string width
    中英字符串实际宽度
    >>> strwidth('ab')
    2
    >>> strwidth('a⑪中⑩')
    7

    ⑩等字符的宽度还是跟字体有关的，不过在大部分地方好像都是域宽2，目前算法问题不大
    """
    try:
        res = len(s.encode('gbk'))
    except UnicodeEncodeError:
        count = len(s)
        for x in s:
            if ord(x) > 127:
                count += 1
        res = count
    return res


def strwidth_proc(s, fmt='r', chinese_char_width=1.8):
    """ 此函数主要用于每个汉字域宽是w=1.8的情况

    为了让字符串域宽为一个整数，需要补充中文空格，会对原始字符串进行修改。
    故返回值有2个，第1个是修正后的字符串s，第2个是实际宽度w。

    :param s: 一个字符串
    :param fmt: 目标对齐格式
    :param chinese_char_width: 每个汉字字符宽度
    :return: (s, w)
        s: 修正后的字符串值s
        w: 修正后字符串的实际宽度

    >>> strwidth_proc('哈哈a')
    ('　　　哈哈a', 10)
    """
    # 1、计算一些参数值
    s = str(s)  # 确保是字符串类型
    l1 = len(s)
    l2 = strwidth(s)
    y = l2 - l1  # 中文字符数
    x = l1 - y  # 英文字符数
    # ch = chr(12288)  # 中文空格
    ch = chr(12288)  # 中文空格
    w = x + y * chinese_char_width  # 当前字符串宽度
    # 2、计算需要补充t个中文空格
    error = 0.05  # 允许误差范围
    t = 0  # 需要补充中文字符数
    while error < w % 1 < 1 - error:  # 小数部分超过误差
        t += 1
        w += chinese_char_width
    # 3、补充中文字符
    if t:
        if fmt == 'r':
            s = ch * t + s
        elif fmt == 'l':
            s = s + ch * t
        else:
            s = ch * (t - t // 2) + s + ch * (t // 2)
    return s, int(w)


def listalign(ls, fmt='r', *, width=None, fillchar=' ', prefix='', suffix='', chinese_char_width=2):
    """文档： https://blog.csdn.net/code4101/article/details/80985218（不过文档有些过时了）
    listalign列表对齐
    py3中str的len是计算字符数量，例如len('ab') --> 2， len('a中b') --> 3。
    但在对齐等操作中，是需要将每个汉字当成宽度2来处理，计算字符串实际宽度的。
    所以我们需要开发一个strwidth函数，效果： strwidth('ab') --> 2，strwidth('a中b') --> 4。

    :param ls:
        要处理的列表，会对所有元素调用str处理，确保全部转为string类型
            且会将换行符转为\n显示
    :param fmt: （format）
        l: left，左对齐
        c: center，居中
        r: right，右对齐
        多个字符: 扩展fmt长度跟ls一样，每一个元素单独设置对齐格式。如果fmt长度小于ls，则扩展的格式按照fmt[-1]设置
    :param width:
        None或者设置值小于最长字符串: 不设域宽，直接按照最长的字符串为准
    :param fillchar: 填充字符
    :param prefix: 添加前缀
    :param suffix: 添加后缀
    :param chinese_char_width: 每个汉字字符宽度

    :return:
        对齐后的数组ls，每个元素会转为str类型

    >>> listalign(['a', '哈哈', 'ccd'])
    ['   a', '哈哈', ' ccd']
    >>> listalign(['a', '哈哈', 'ccd'], chinese_char_width=1.8)
    ['        a', '　　　哈哈', '      ccd']
    """
    # 1、处理fmt数组
    if len(fmt) == 1:
        fmt = [fmt] * len(ls)
    elif len(fmt) < len(ls):
        fmt = list(fmt) + [fmt[-1]] * (len(ls) - len(fmt))

    # 2、算出需要域宽
    if chinese_char_width == 2:
        strs = list(map(lambda x: str(x).replace('\n', r'\n'), ls))  # 存储转成字符串的元素
        lens = list(map(strwidth, strs))  # 存储每个元素的实际域宽
    else:
        strs = []  # 存储转成字符串的元素
        lens = []  # 存储每个元素的实际域宽
        for i, t in enumerate(ls):
            t, n = strwidth_proc(t, fmt[i], chinese_char_width)
            strs.append(t)
            lens.append(n)
    w = max(lens)
    if width and isinstance(width, int) and width > w:
        w = width

    # 3、对齐操作
    for i, s in enumerate(strs):
        if fmt[i] == 'r':
            strs[i] = fillchar * (w - lens[i]) + strs[i]
        elif fmt[i] == 'l':
            strs[i] = strs[i] + fillchar * (w - lens[i])
        elif fmt[i] == 'c':
            t = w - lens[i]
            strs[i] = fillchar * (t - t // 2) + strs[i] + fillchar * (t // 2)
        strs[i] = prefix + strs[i] + suffix
    return strs


def len_in_dim2(arr):
    """计算类List结构在第2维上的长度

    >>> len_in_dim2([[1,1], [2], [3,3,3]])
    3
    """
    if not isinstance(arr, (list, tuple)):
        raise TypeError('类型错误，不是list构成的二维数组')

    # 找出元素最多的列
    column_num = 0
    for i, item in enumerate(arr):
        if isinstance(item, (list, tuple)):  # 该行是一个一维数组
            column_num = max(column_num, len(item))
        else:  # 如果不是数组，是指单个元素，当成1列处理
            column_num = max(column_num, 1)

    return column_num


def ensure_array(arr, default_value=''):
    """对一个由list、tuple组成的二维数组，确保所有第二维的列数都相同

    >>> ensure_array([[1,1], [2], [3,3,3]])
    [[1, 1, ''], [2, '', ''], [3, 3, 3]]
    """
    max_cols = len_in_dim2(arr)
    if max_cols == 1:
        return arr
    dv = str(default_value)
    a = [[]] * len(arr)
    for i, ls in enumerate(arr):
        if isinstance(ls, (list, tuple)):
            t = list(arr[i])
        else:
            t = [ls]  # 如果不是数组，是指单个元素，当成1列处理
        a[i] = t + [dv] * (max_cols - len(t))  # 左边的写list，是防止有的情况是tuple，要强制转list后拼接
    return a


def swap_rowcol(a, *, ensure_arr=False, default_value=''):
    """矩阵行列互换

    注：如果列数是不均匀的，则会以最小列数作为行数

    >>> swap_rowcol([[1,2,3], [4,5,6]])
    [[1, 4], [2, 5], [3, 6]]
    """
    if ensure_arr:
        a = ensure_array(a, default_value)
    # 这是非常有教学意义的行列互换实现代码
    return list(map(list, zip(*a)))


def int2excel_col_name(d):
    """
    >>> int2excel_col_name(1)
    'A'
    >>> int2excel_col_name(28)
    'AB'
    >>> int2excel_col_name(100)
    'CV'
    """
    s = []
    while d:
        t = (d - 1) % 26
        s.append(chr(65 + t))
        d = (d - 1) // 26
    return ''.join(reversed(s))


def excel_col_name2int(s):
    """
    >>> excel_col_name2int('A')
    1
    >>> excel_col_name2int('AA')
    27
    >>> excel_col_name2int('AB')
    28
    """
    d = 0
    for ch in s:
        d = d * 26 + (ord(ch) - 64)
    return d


def int2myalphaenum(n):
    """
    :param n: 0~52的数字
    """
    if 0 <= n <= 52:
        return '_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[n]
    else:
        dprint(n)  # 不在处理范围内的数值
        raise ValueError


def gentuple(n, tag):
    """有点类似range函数，但生成的数列更加灵活
    :param n:
        数组长度
    :param tag:
        整数，从指定整数开始编号
        int类型，从指定数字开始编号
            0，从0开始编号
            1，从1开始编号
        'A'，用Excel的形式编号
        tuple，按枚举值循环显示
            ('A', 'B')：循环使用A、B编号

    >>> gentuple(4, 'A')
    ('A', 'B', 'C', 'D')
    """
    a = [''] * n
    if isinstance(tag, int):
        for i in range(n):
            a[i] = i + tag
    elif tag == 'A':
        a = tuple(map(lambda x: int2excel_col_name(x + 1), range(n)))
    elif isinstance(tag, (list, tuple)):
        k = len(tag)
        a = tuple(map(lambda x: tag[x % k], range(n)))
    return a


"""4、arralign数组对齐
"""


def ensure_gbk(s):
    """检查一个字符串的所有内容是否能正常转为gbk，
    如果不能则ignore掉不能转换的部分"""
    try:
        s.encode('gbk')
    except UnicodeEncodeError:
        origin_s = s
        s = s.encode('gbk', errors='ignore').decode('gbk')
        dprint(origin_s, s)  # 字符串存在无法转为gbk的字符
    return s


def funcmsg(func):
    """输出函数func所在的文件、函数名、函数起始行"""
    return f'函数名：{func.__name__}，来自文件：{func.__code__.co_filename}，所在行号={func.__code__.co_firstlineno}'


class GrowingList(list):
    """可变长list"""

    def __init__(self, default_value=None):
        super().__init__(self)
        self.default_value = default_value

    def __getitem__(self, index):
        if index >= len(self):
            self.extend([self.default_value] * (index + 1 - len(self)))
        return list.__getitem__(self, index)

    def __setitem__(self, index, value):
        if index >= len(self):
            self.extend([self.default_value] * (index + 1 - len(self)))
        list.__setitem__(self, index, value)


def arr_hangclear(arr, depth=None):
    """ 清除连续相同值，简化表格内容
    >> arr_hangclear(arr, depth=2)
    原表格：
        A  B  D
        A  B  E
        A  C  E
        A  C  E
    新表格：
        A  B  D
              E
           C  E
              E

    :param arr: 二维数组
    :param depth: 处理列上限
        例如depth=1，则只处理第一层
        depth=None，则处理所有列

    >>> arr_hangclear([[1, 2, 4], [1, 2, 5], [1, 3, 5], [1, 3, 5]])
    [[1, 2, 4], ['', '', 5], ['', 3, 5], ['', '', 5]]
    >>> arr_hangclear([[1, 2, 4], [1, 2, 5], [2, 2, 5], [1, 2, 5]])
    [[1, 2, 4], ['', '', 5], [2, 2, 5], [1, 2, 5]]
    """
    m = depth if depth else len_in_dim2(arr) - 1
    a = deepcopy(arr)

    # 算法原理：从下到上，从右到左判断与上一行重叠了几列数据
    for i in range(len(arr) - 1, 0, -1):
        for j in range(m):
            if a[i][j] == a[i - 1][j]:
                a[i][j] = ''
            else:
                break
    return a


def digit2weektag(d):
    """输入数字1~7，转为“周一~周日”

    >>> digit2weektag(1)
    '周一'
    >>> digit2weektag('7')
    '周日'
    """
    d = int(d)
    if 1 <= d <= 7:
        return '周' + '一二三四五六日'[d-1]
    else:
        raise ValueError


def mydatetag(s=None):
    """未输入s时，返回当前值
    >>> mydatetag(180826)
    '180826周日'
    >>> mydatetag('180823')
    '180823周四'

    >> mydatetag()
    '180823周四'
    """
    # 1、类型检测
    if s is None:
        d = datetime.date.today()
        s = d.strftime('%y%m%d')
    elif isinstance(s, str) and len(s) == 6:
        d = str2date(s)
    elif isinstance(s, int) and len(str(s)) == 6:
        s = str(s)
        d = str2date(s)
    else:
        raise TypeError

    # 2、计算出周几
    wd = digit2weektag(d.isoweekday())

    return s + wd


def mydatetimetag():
    """获取当前时间的tag标签，例如：

    >> mydatetimetag()
    '180823周四16:06'
    """
    return mydatetag() + datetime.datetime.now().strftime('%H:%M')


def userdatetimetag(username=None):
    """
    >> userdatetimetag()
    '坤泽，180823周四16:07'
    """
    if username is None: username = socket.getfqdn()
    return username + '，' + mydatetimetag()


def mydatetagtool(s=None, username=None):
    """有输入值的时候，识别其日期；无输入值的时候，返回当前用户、时间
    结果会同时存储一份到剪切板
    """
    if s:
        s = mydatetag(s)
    else:
        s = userdatetimetag(username)

    pyperclip.copy(s)
    return s


def arr2table(arr, rowmerge=False):
    """数组转html表格代码
    :param arr:  需要处理的数组
    :param rowmerge: 行单元格合并
    :return: html文本格式的<table>
    """
    n = len(arr)
    m = len_in_dim2(arr)
    res = ['<table border="1"><tbody>']
    for i, line in enumerate(arr):
        res.append('<tr>')
        for j, ele in enumerate(line):
            if rowmerge:
                if ele != '':
                    cnt = 1
                    while i + cnt < n and arr[i + cnt][j] == '':
                        for k in range(j - 1, -1, -1):
                            if arr[i + cnt][k] != '':
                                break
                        else:
                            cnt += 1
                            continue
                        break
                    if cnt > 1:
                        res.append(f'<td rowspan="{cnt}">{ele}</td>')
                    else:
                        res.append(f'<td>{ele}</td>')
                elif j == m - 1:
                    res.append(f'<td>{ele}</td>')
            else:
                res.append(f'<td>{ele}</td>')
        res.append('</tr>')
    res.append('</tbody></table>')
    return ''.join(res)


if __name__ == '__main__':
    timer = Timer(start_now=True)

    timer.stop_and_report()
