#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 11:00

import builtins
import inspect
import json
import logging
import os
import re
import traceback
from collections import Counter

from pyxllib.prog.basic import typename
from pyxllib.prog.fmt import print2string
from pyxllib.prog.time import utc_now2, utc_timestamp


def funcmsg(func):
    """输出函数func所在的文件、函数名、函数起始行"""
    # showdir(func)
    if not hasattr(func, '__name__'):  # 没有__name__属性表示这很可能是一个装饰器去处理原函数了
        if hasattr(func, 'func'):  # 我的装饰器常用func成员存储原函数对象
            func = func.func
        else:
            return f'装饰器：{type(func)}，无法定位'
    return f'函数名：{func.__name__}，来自文件：{func.__code__.co_filename}，所在行号={func.__code__.co_firstlineno}'


def set_global_var(name, value):
    """ 设置某个全局变量，一般用在一些特殊的需要跨作用域进行调试的场景
    切勿！切勿！切勿用于正式功能，否则会导致难以维护控制的功能代码

    为了避免和某些关键的全局变量名冲突，这里的变量命令统一会加上pyxllib_的前缀
    """
    g = globals()
    name = f'pyxllib_{name}'
    g[name] = value


def get_global_var(name, default_value=None):
    """ 获取某个全局变量的值 """
    g = globals()
    name = f'pyxllib_{name}'
    if name not in g:
        g[name] = default_value
    return g[name]


def func_input_message(depth=2) -> dict:
    """假设调用了这个函数的函数叫做f，这个函数会获得
        调用f的时候输入的参数信息，返回一个dict，键值对为
            fullfilename：完整文件名
            filename：文件名
            funcname：所在函数名
            lineno：代码所在行号
            comment：尾巴的注释
            depth：深度
            funcnames：整个调用过程的函数名，用/隔开，例如...

            argnames：变量名（list），这里的变量名也有可能是一个表达式
            types：变量类型（list），如果是表达式，类型指表达式的运算结果类型
            argvals：变量值（list）

        这样以后要加新的键值对也很方便

        :param depth: 需要分析的层级
            0，当前func_input_message函数的参数输入情况
            1，调用func_input_message的函数 f 参数输入情况
            2，调用 f 的函数 g ，g的参数输入情况

        参考： func_input_message 的具体使用方法可以参考 dformat 函数
        细节：inspect可以获得函数签名，也可以获得一个函数各个参数的输入值，但我想要展现的是原始表达式，
            例如func(a)，以func(1+2)调用，inpect只能获得“a=3”，但我想要的是“1+2=3”的效果
    """
    res = {}
    # 1 找出调用函数的代码
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
    res['depth'] = len(ss)
    ls_ = list(map(lambda x: x.function, ss))
    # ls.reverse()
    res['funcnames'] = '/'.join(ls_)

    if frameinfo.code_context:
        code_line = frameinfo.code_context[0].strip()
    else:  # 命令模式无法获得代码，是一个None对象
        code_line = ''

    funcname = ss[depth - 1].function  # 调用的函数名
    # 这一行代码不一定是从“funcname(”开始，所以要用find找到开始位置
    code = code_line[code_line.find(funcname + '(') + len(funcname):]

    # 2 先找到函数的()中参数列表，需要以')'作为分隔符分析
    # TODO 可以考虑用ast重实现
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

    # 3 获得注释
    # 这个注释实现的不是很完美，不过影响应该不大，还没有想到比较完美的解决方案
    t = ')'.join(ls[i:])
    comment = t[t.find('#'):] if '#' in t else ''
    res['comment'] = comment

    # 4 获得变量名
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

    # 5 获得变量值和类型
    res['argvals'] = origin_args
    res['types'] = list(map(typename, origin_args))

    if not argnames:  # 如果在命令行环境下调用，argnames会有空，需要根据argvals长度置空名称
        argnames = [''] * len(res['argvals'])
    res['argnames'] = argnames

    return res


def dformat(*args, depth=2,
            delimiter=' ' * 4,
            strfunc=repr,
            fmt='[{depth:02}]{filename}/{lineno}: {argmsg}',
            subfmt='{name}<{tp}>={val}'):
    r"""
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
        旧版还用过这种格式： '{filename}/{funcname}/{lineno}: {argmsg}    {comment}'
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
    r"""
    # 故意写的特别复杂，测试在极端情况下是否能正确解析出表达式
    >> a, b = 1, 2
    >> re.sub(str(dprint(1, b, a, "aa" + "bb)", "a[,ba\nbb""b", [2, 3])), '', '##')  # 注释 # 注
    [08]<doctest debuglib.dprint[1]>/1: 1<int>=1    b<int>=2    a<int>=1    "aa" + "bb)"<str>='aabb)'    "a[,ba\nbb""b"<str>='a[,ba\nbbb'    [2, 3]<list>=[2, 3]    ##')  # 注释 # 注
    '##'
    """
    print(dformat(depth=3, **kwargs))


# dprint会被注册进builtins，可以在任意地方直接使用
setattr(builtins, 'dprint', dprint)


class DPrint:
    """ 用来存储上下文相关变量，进行全局性地调试

    TODO 需要跟logging库一样，可以获取不同名称的配置
        可以进行很多扩展，比如输出到stderr还是stdout
    """

    watch = {}

    @classmethod
    def reset(cls):
        cls.watch = {}

    @classmethod
    def format(cls, watch2, show_type=False, sep=' '):
        """
        :param watch2: 必须也是字典类型
        :param show_type: 是否显示每个数值的类型
        :param sep: 每部分的间隔符
        """
        msg = []
        input_msg = func_input_message(2)
        filename, lineno = input_msg['filename'], input_msg['lineno']
        msg.append(f'{filename}/{lineno}')

        watch3 = cls.watch.copy()
        watch3.update(watch2)
        for k, v in watch3.items():
            if k.startswith('$'):
                # 用 $ 修饰的不显示变量名，直接显示值
                msg.append(f'{v}')
            else:
                if show_type:
                    msg.append(f'{k}<{typename(v)}>={repr(v)}')
                else:
                    msg.append(f'{k}={repr(v)}')

        return sep.join(msg)


def format_exception(e, mode=3):
    if mode == 1:
        # 仅获取异常类型的名称
        text = ''.join(traceback.format_exception_only(type(e), e)).strip()
    elif mode == 2:
        # 获取异常类型的名称和附加的错误信息
        text = f"{type(e).__name__}: {e}"
    elif mode == 3:
        text = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    else:
        raise ValueError
    return text


class OutputLogger(logging.Logger):
    """
    我在jupyter写代码，经常要print输出一些中间结果。
    但是当结果很多的时候，又要转存到文件里保存起来查看。
    要保存到文件时，和普通的print写法是不一样的，一般要新建立一个ls = []的变量。
    然后print改成ls.append操作，会很麻烦。

    就想着能不能自己定义一个类，支持.print方法，不仅能实现正常的输出控制台的功能。
    也能在需要的时候指定文件路径，会自动将结果存储到文件中。
    """

    def __init__(self, name='OutputLogger', *, log_file=None, log_mode='a', output_to_console=True):
        """
        :param str name: 记录器的名称。默认为 'OutputLogger'。
        :param log_file: 日志文件的路径。默认为 None，表示不输出到文件。
        :param bool output_to_console: 是否输出到命令行。默认为 True。
        """
        super().__init__(name)

        self.output_to_console = output_to_console
        self.log_file = log_file

        self.setLevel(logging.INFO)
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s/%(lineno)d - %(message)s',
                                      '%Y-%m-%d %H:%M:%S')

        # 提前重置为空文件
        if log_file is not None:
            if not os.path.isfile(log_file) or log_mode == 'w':
                with open(log_file, 'w', encoding='utf8') as f:
                    f.write('')

        # 创建文件日志处理器
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # 日志文件是最详细级别都记录
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)

        # 创建命令行日志处理器
        if output_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)  # 有些太详细的问题，不想写在控制台，而是写到文件
            console_handler.setFormatter(formatter)
            self.addHandler(console_handler)

        # 只输出到控制台：标准库的print
        # 只输出到文件：debug, info
        # 同时输出到控制台和文件：warning, error, critical, print

    def print(self, *args, **kwargs):
        """ 使用print机制，会同时输出到控制台和日志文件 """
        msg = print2string(*args, **kwargs)

        if self.output_to_console:
            print(msg, end='')

        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(msg)

        return msg

    def tprint(self, *args, **kwargs):
        """ 带时间戳的print """
        self.print(utc_now2(), *args, **kwargs)

    def log_json(self, data):
        """ 类似print，但是是把数据按照json的格式进行记录整理，更加结构化，方便后期处理 """
        data['time'] = utc_timestamp()
        msg = json.dumps(data, ensure_ascii=False, default=str)
        self.print(msg)


def check_counter(data, top_n=10):
    """ 将一个数据data转为Counter进行频数分析 """
    # 1 如果是list、tuple类型，需要转counter
    if isinstance(data, (list, tuple)):
        data = Counter(data)
    if not isinstance(data, Counter):
        raise ValueError(f'输入的数据类型不对，应该是Counter类型，而不是{typename(data)}')

    # 2 列出出现次数最多的top_n条目
    # 打印基本统计信息
    total_items = sum(data.values())
    print(f"总条目数: {total_items}")

    if top_n > 0:
        top_items = data.most_common(top_n)
        max_n = len(data)
        print(f"出现次数最多的{min(top_n, max_n)}/{max_n}条数据（频率）:")
        for item, count in top_items:
            print(f"\t{item}\t{count}")

    # 3 打印基本统计信息
    # 对原始Counter的计数值进行再计数
    count_frequencies = Counter(data.values())

    # 打印各计数值出现的次数
    print("各计数值出现的次数，频率的频率（频率分布）:")
    for count, frequency in count_frequencies.most_common():
        print(f"\t{count}\t{frequency}")


def tprint(*args, **kwargs):
    """ 带时间戳的print """
    print(utc_now2(), *args, **kwargs)


def loguru_setup_jsonl_logfile(sink, *, rotation='50 MB', retention='30 days', **kwargs):
    """
    配置 Loguru 日志记录器，使其以 JSONL 格式记录日志。

    参数:
    - sink: 日志文件的路径或文件对象。
    - rotation: 日志文件的轮转策略，默认为 '50 MB'。
    - retention: 日志文件的保留策略，默认为 '30 days'。
    - **kwargs: 传递给 `logger.add` 的其他关键字参数。

    返回:
    - logger: 配置好的 Loguru 日志记录器。
    """
    from loguru import logger

    def json_formatter(record):
        # 创建一个新的字典，包含我们需要的所有字段
        log_record = {
            "time": record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "level": record["level"].name,
            "message": record["message"],
            # "file": record["file"].name,  # 'NoneType' object has no attribute 'name'
            # "line": record["line"],
            "function": record["function"],
            # "module": record["module"],
            # "process": record["process"].name,
            # "thread": record["thread"].name,
            "extra": record["extra"]
        }
        if record["file"]:
            log_record["file"] = record["file"].name
        if record["line"]:
            log_record["line"] = record["line"]
        if record["module"]:
            log_record["module"] = record["module"]
        # 如果有异常信息，添加到记录中
        if record["exception"]:
            log_record["exception"] = str(record["exception"])

        # 将字典转换为 JSON 字符串，并添加换行符
        return json.dumps(log_record, ensure_ascii=False) + "\n"

    # 移除默认的 handler
    # logger.remove()

    # 添加新的 handler，使用自定义的 JSON 格式化器
    # 2024-02-27: 修复 Loguru KeyError: '"time"' 问题
    # 当使用 format=function 时，Loguru 期望函数返回格式化后的字符串，但它内部处理机制似乎会再次尝试格式化
    # 正确的做法是直接让 format 函数返回处理好的 JSON 字符串，但在 add 时不应该再有其他干扰
    # 参考 loguru 文档：sink 可以是函数，也可以是文件路径。如果是文件路径，format 参数可以是字符串或函数。
    # 这里的问题可能是 json_formatter 返回的内容包含了类似 {time} 的字符串，被 loguru 再次解析了。
    # 实际上，loguru 的 format 参数如果是一个函数，它应该接受 record 并返回字符串。
    
    # 经过排查，问题在于 loguru 的 format 参数如果返回的字符串中包含 {}，loguru 可能会尝试对其进行 .format() 操作
    # 但在这里我们返回的是 JSON 字符串，其中必然包含 {}。
    # 解决方法：在返回的 JSON 字符串中，将 { 和 } 转义，或者使用 serialize=True 参数（如果 loguru 支持直接序列化）
    # 但 loguru 的 serialize=True 是默认的 JSON 格式，我们想要自定义格式。
    
    # 另一种思路：sink 作为一个函数，在函数内部写入文件。但这里 sink 是文件路径。
    
    # 重新审视错误：KeyError: '"time"'。这说明 format_map 正在尝试解析我们返回的 JSON 字符串中的 key。
    # 因为 JSON 字符串里有 "time": "...", loguru 把它当成了格式化占位符。
    # 所以我们需要转义返回字符串中的花括号。
    
    def json_formatter_safe(record):
        json_str = json_formatter(record)
        # 1. 转义花括号，避免 loguru 再次格式化
        # 2. 转义 < 和 >，避免 loguru 将其解析为颜色标签 (如 <green>, <module>)
        #    loguru 的颜色标签格式是 <color>，如果 json 中包含类似 <module> 的内容，会报错
        #    解决方法：将 < 替换为 \<
        return json_str.replace("{", "{{").replace("}", "}}").replace("<", "\\<")

    logger.add(
        sink,
        format=json_formatter_safe,
        rotation=rotation,
        retention=retention,
        **kwargs
    )

    return logger
