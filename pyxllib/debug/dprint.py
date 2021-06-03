#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 20:35

import inspect
import os

from pyxllib.time import TicToc
from pyxllib.prog import typename

____dprint = """
调试相关功能

TODO 高亮格式？
"""


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
            fmt='[{depth:02}]{filename}/{lineno}: {argmsg}    {comment}',
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


def demo_dprint():
    """这里演示dprint常用功能
    """
    # 1 查看程序是否运行到某个位置
    dprint()
    # [05]dprint.py/169:      意思：这是堆栈的第5层，所运行的位置是 dprint.py文件的第169行

    # 2 查看变量、表达式的 '<类型>' 和 ':值'
    a, b, s = 1, 2, 'ab'
    dprint(a, b, a ^ b, s * 2)
    # [05]dprint.py/174: a<int>=1    b<int>=2    a ^ b<int>=3    s*2<str>='abab'

    # 3 异常警告
    b = 0
    if b:
        c = a / b
    else:
        c = 0
        dprint(a, b, c)  # b=0不能作为除数，c默认值暂按0处理
    # [05]dprint.py/183: a<int>=1    b<int>=0    c<int>=0    # b=0不能作为除数，c默认值暂按0处理

    # 4 如果想在其他地方使用dprint的格式内容，可以调底层dformat函数实现
    with TicToc(dformat(fmt='[{depth:02}]{fullfilename}/{lineno}: {argmsg}')):
        for _ in range(10 ** 7):
            pass
    # [04]D:\slns\pyxllib\pyxllib\debug\dprint.py/187:  0.173 秒.
