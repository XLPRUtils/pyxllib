#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/09/18 22:16

import concurrent.futures
import inspect
import os
import traceback
import math
import sys

from pyxllib.basic._1_strlib import shorten
from pyxllib.basic._3_pathlib import Path

XLLOG_CONF_FILE = 'xllog.yaml'

____dprint = """
调试相关功能

TODO 高亮格式？
"""


def typename(c):
    """简化输出的type类型
    >>> typename(123)
    'int'
    """
    return str(type(c))[8:-2]


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


def demo_dprint():
    """这里演示dprint常用功能
    """
    from ._2_timelib import TicToc

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


____xllog = """
"""


def get_xllog():
    """ 获得pyxllib库的日志类

    由于日志类可能要读取yaml配置文件，需要使用Path类，所以实现代码先放在pathlib_.py

    TODO 类似企业微信机器人的机制怎么设？或者如何配置出问题发邮件？
    """
    import logging

    if 'pyxllib.xllog' in logging.root.manager.loggerDict:
        # 1 判断xllog是否已存在，直接返回
        pass
    elif os.path.isfile(XLLOG_CONF_FILE):
        # 2 若不存在，尝试在默认位置是否有自定义配置文件，读取配置文件来创建
        import logging.config
        data = Path(XLLOG_CONF_FILE).read()
        if isinstance(data, dict):
            # 推荐使用yaml的字典结构，格式更简洁清晰
            logging.config.dictConfig(data)
        else:
            # 但是普通的conf配置文件也支持
            logging.config.fileConfig(XLLOG_CONF_FILE)
    else:
        # 3 否则生成一个非常简易版的xllog
        # TODO 不同级别能设不同的格式（颜色）？
        xllog = logging.getLogger('pyxllib.xllog')
        xllog.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S'))
        xllog.addHandler(ch)
    return logging.getLogger('pyxllib.xllog')


def format_exception(e):
    return ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))


____iterate = """
"""


class Iterate:
    """ 迭代器类，用来封装一些特定模式的for循环操作

    TODO 双循环，需要内部两两对比的迭代功能
    """

    def __init__(self, items):
        # 没有总长度倒也能接受，关键是可能要用start、end切片，所以还是先转成tuple更方便操作
        self.items = tuple(items)
        self.n_items = len(self.items)
        self.format_width = math.ceil(math.log10(self.n_items + 1))
        self.xllog = get_xllog()

    def _format_pinterval(self, pinterval=None):
        if isinstance(pinterval, str) and pinterval.endswith('%'):
            # 百分比的情况，重算出间隔元素数
            return int(round(self.n_items * float(pinterval[:-1]) / 100))
        else:  # 其他格式暂不解析，按原格式处理
            return pinterval

    def _step1_check_number(self, pinterval, func):
        if pinterval:
            sys.stdout.flush()  # 让逻辑在前的标准输出先print出来，但其实这句也不一定能让print及时输出的~~可能会被日志提前抢输出了
            self.xllog.info(f"使用 {func} 处理 {self.n_items} 个数据 {shorten(str(self.items), 30)}")

    def _step2_check_range(self, start, end):
        if start:
            self.xllog.info(f"使用start参数，只处理≥{start}的条目")
        else:
            start = 0
        if end:
            # 这里空格是为了对齐，别删
            self.xllog.info(f"使用 end 参数，只处理<{end}的条目")
        else:
            end = len(self.items)
        return start, end

    def _step3_executor(self, pinterval, max_workers):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers)
        if executor._max_workers != 1:
            if pinterval:
                self.xllog.info(f'多线程执行，当前迭代所用线程数：{executor._max_workers}')
        return executor

    def _step4_iter(self, i, pinterval, executor):
        if pinterval and (i or pinterval == 1) and i % pinterval == 0:
            message = f' {self.items[i]}' if pinterval == 1 else ''
            self.xllog.info(f'{i:{self.format_width}d}/{self.n_items}={i / self.n_items:6.2%}{message}')
        # 队列中没有新任务时，才放入新任务，这样能确保pinterval的输出能反应实时情况，而不是一下全部进入队列，把for循环跑完了
        while executor._work_queue.qsize(): pass

    def _step5_finish(self, pinterval, interrupt, executor):
        executor.shutdown()
        if not interrupt and pinterval:
            self.xllog.info(f'{self.n_items:{self.format_width}d}/{self.n_items}='
                            f'{self.n_items / self.n_items:6.2%} 完成迭代')
            sys.stderr.flush()

    def run(self, func, start=0, end=None, pinterval=None, max_workers=1, interrupt=True):
        """
        :param func: 对每个item执行的功能
        :param start: 跳过<start的数据，只处理>=start编号以上
        :param end: 只处理 < end 的数据
        :param pinterval: 每隔多少条目输出进度日志，默认不输出进度日志（但是错误日志依然会输出）
            TODO 支持按百分比进度显示？  例如每20%，pinterval='20%'
            TODO 支持按指定时间间隔显示？ 例如每15秒，pinterval='15s'
        :param max_workers: 默认线程数，默认1，即串行
        :type max_workers: int, None
        :param interrupt: 出现错误时是否中断，默认True会终止程序，否则只会输出错误日志
        :return:
        """

        # 1 统一的参数处理部分
        pinterval = self._format_pinterval(pinterval)
        self._step1_check_number(pinterval, func)
        start, end = self._step2_check_range(start, end)
        error = False
        executor = self._step3_executor(pinterval, max_workers)

        # 2 封装的子处理部分
        def wrap_func(func, i):
            nonlocal error
            item = self.items[i]
            try:
                func(item)
            except Exception as e:
                error = True
                self.xllog.error(f'💔idx={i}运行出错：{item}\n{format_exception(e)}')

        # 3 执行迭代
        for i in range(start, end):
            self._step4_iter(i, pinterval, executor)
            executor.submit(wrap_func, func, i)
            if interrupt and error: break
        self._step5_finish(pinterval, interrupt and error, executor)

    def run_pair(self, func, start=0, end=None, pinterval=None, max_workers=1, interrupt=True):
        """ 对items两两运算
            func(x, y) 等同于 func(y, x)，不重复运算

        :param start: 这里的start、end特指第一层迭代器i的取值范围

        TODO starti, endi, startj, endj，i和j支持单独设置遍历区间？
        """
        # 1 统一的参数处理部分
        pinterval = self._format_pinterval(pinterval)
        self._step1_check_number(pinterval, func)
        start, end = self._step2_check_range(start, end)
        error = False
        executor = self._step3_executor(pinterval, max_workers)

        # 2 封装的子处理部分
        def wrap_func(func, i, j):
            nonlocal error
            item1, item2 = self.items[i], self.items[j]
            try:
                func(item1, item2)
            except Exception as e:
                error = True
                self.xllog.error(f'💔idxs=({i},{j})运行出错：{item1},{item2}\n{format_exception(e)}')

        # 3 执行迭代
        for i in range(start, end):
            self._step4_iter(i, pinterval, executor)
            for j in range(i + 1, self.n_items):
                executor.submit(wrap_func, func, i, j)
                if interrupt and error: break
        self._step5_finish(pinterval, interrupt and error, executor)

    def run_pair2(self, func, start=0, end=None, pinterval=None, max_workers=1, interrupt=True):
        """ 对items两两运算
            func(x, y) 不同于 func(y, x)，需要全量运算

        :param start: 这里的start、end特指第一层迭代器i的取值范围
        """
        # 1 统一的参数处理部分
        pinterval = self._format_pinterval(pinterval)
        self._step1_check_number(pinterval, func)
        start, end = self._step2_check_range(start, end)
        error = False
        executor = self._step3_executor(pinterval, max_workers)

        # 2 封装的子处理部分
        def wrap_func(func, i, j):
            nonlocal error
            item1, item2 = self.items[i], self.items[j]
            try:
                func(item1, item2)
            except Exception as e:
                error = True
                self.xllog.error(f'💔idxs=({i},{j})运行出错：{item1},{item2}\n{format_exception(e)}')

        # 3 执行迭代
        for i in range(start, end):
            self._step4_iter(i, pinterval, executor)
            for j in range(self.n_items):
                if j == i: continue
                executor.submit(wrap_func, func, i, j)
                if interrupt and error: break
        self._step5_finish(pinterval, interrupt and error, executor)


class RunOnlyOnce:
    """ 被装饰的函数，不同的参数输入形式，只会被执行一次，

    重复执行时会从内存直接调用上次相同参数调用下的运行的结果
    可以使用reset成员函数重置，下一次调用该函数时则会重新执行

    文档：https://www.yuque.com/xlpr/pyxllib/RunOnlyOnce

    使用好该装饰器，可能让一些动态规划dp、搜索问题变得更简洁，
    以及一些配置文件操作，可以做到只读一遍
    """

    def __init__(self, func, distinct_args=True):
        """
        :param func: 封装的函数
        :param distinct_args: 默认不同输入参数形式，都会保存一个结果
            设为False，则不管何种参数形式，函数就真的只会保存第一次运行的结果
        """
        self.func = func
        self.distinct_args = distinct_args
        self.results = {}

    def __call__(self, *args, **kwargs):
        tag = f'{args}{kwargs}' if self.distinct_args else ''
        # TODO 思考更严谨，考虑了值类型的tag标记
        #   目前的tag规则，是必要不充分条件。还可以使用id，则是充分不必要条件
        #   能找到充要条件来做是最好的，不行的话，也应该用更严谨的充分条件来做
        # TODO kwargs的顺序应该是没影响的，要去掉顺序干扰
        if tag not in self.results:
            self.results[tag] = self.func(*args, **kwargs)
        return self.results[tag]

    def reset(self):
        self.results = {}
