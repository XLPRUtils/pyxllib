#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/09/20


import datetime
import math
import re
import time
import timeit

import arrow

from humanfriendly import format_timespan

from pyxllib.basic.str import shorten, natural_sort, listalign

____tictoc = """
基于 pytictoc 代码，做了些自定义扩展

原版备注：
Module with class TicToc to replicate the functionality of MATLAB's tic and toc.
Documentation: https://pypi.python.org/pypi/pytictoc
__author__       = 'Eric Fields'
__version__      = '1.4.0'
__version_date__ = '29 April 2017'
"""


class TicToc:
    """
    Replicate the functionality of MATLAB's tic and toc.

    #Methods
    TicToc.tic()       #start or re-start the timer
    TicToc.toc()       #print elapsed time since timer start
    TicToc.tocvalue()  #return floating point value of elapsed time since timer start

    #Attributes
    TicToc.start     #Time from timeit.default_timer() when t.tic() was last called
    TicToc.end       #Time from timeit.default_timer() when t.toc() or t.tocvalue() was last called
    TicToc.elapsed   #t.end - t.start; i.e., time elapsed from t.start when t.toc() or t.tocvalue() was last called
    """

    def __init__(self, title=''):
        """Create instance of TicToc class."""
        self.start = timeit.default_timer()
        self.end = float('nan')
        self.elapsed = float('nan')
        self.title = title

    def tic(self):
        """Start the timer."""
        self.start = timeit.default_timer()

    def toc(self, msg='', restart=False):
        """
        Report time elapsed since last call to tic().

        Optional arguments:
            msg     - String to replace default message of 'Elapsed time is'
            restart - Boolean specifying whether to restart the timer
        """
        self.end = timeit.default_timer()
        self.elapsed = self.end - self.start
        # print(f'{self.title} {msg} {self.elapsed:.3f} 秒.')
        print(f'{self.title} {msg} elapsed {format_timespan(self.elapsed)}.')
        if restart:
            self.start = timeit.default_timer()

    def tocvalue(self, restart=False):
        """
        Return time elapsed since last call to tic().

        Optional argument:
            restart - Boolean specifying whether to restart the timer
        """
        self.end = timeit.default_timer()
        self.elapsed = self.end - self.start
        if restart:
            self.start = timeit.default_timer()
        return self.elapsed

    @staticmethod
    def process_time(msg='time.process_time():'):
        """计算从python程序启动到目前为止总用时"""
        print(f'{msg} {format_timespan(time.process_time())}.')

    def __enter__(self):
        """Start the timer when using TicToc in a context manager."""
        if self.title == '__main__':
            from pyxllib.basic.log import get_xllog
            get_xllog().info(f'time.process_time(): {format_timespan(time.process_time())}.')
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """On exit, print time elapsed since entering context manager."""
        from pyxllib.basic.log import get_xllog

        elapsed = self.tocvalue()
        xllog = get_xllog()

        if exc_tb is None:
            xllog.info(f'{self.title} finished in {format_timespan(elapsed)}.')
        else:
            xllog.info(f'{self.title} interrupt in {format_timespan(elapsed)},')


____timer = """

"""


class ValuesStat:
    """ 一串数值的相关统计分析 """

    def __init__(self, values):
        self.values = values
        self.n = len(values)
        self.sum = sum(values)
        # np有标准差等公式，但这是basic底层库，不想依赖太多第三方库，所以手动实现
        if self.n:
            self.mean = self.sum / self.n
            self.std = math.sqrt((sum([(x - self.mean) ** 2 for x in values]) / self.n))
            self.min, self.max = min(values), max(values)
        else:
            self.mean = self.std = self.min = self.max = float('nan')

    def __len__(self):
        return self.n

    def summary(self, valfmt='g'):
        """ 输出性能分析报告，data是每次运行得到的时间数组

        :param valfmt: 数值显示的格式
            g是比较智能的一种模式
            也可以用 '.3f'表示保留3位小数

            也可以传入长度5的格式清单，表示 [和、均值、标准差、最小值、最大值] 一次展示的格式
        """
        if isinstance(valfmt, str):
            valfmt = [valfmt] * 5

        if self.n > 1:  # 有多轮，则应该输出些参考统计指标
            ls = [f'总和: {self.sum:{valfmt[0]}}', f'均值标准差: {self.mean:{valfmt[1]}}±{self.std:{valfmt[2]}}',
                  f'总数: {self.n}', f'最小值: {self.min:{valfmt[3]}}', f'最大值: {self.max:{valfmt[4]}}']
            return '\t'.join(ls)
        elif self.n == 1:  # 只有一轮，则简单地输出即可
            return f'{self.sum:{valfmt[0]}}'
        else:
            raise ValueError


class Timer:
    """分析性能用的计时器类，支持with语法调用
    必须显示地指明每一轮的start()和end()，否则会报错
    """

    def __init__(self, title=''):
        """
        :param title: 计时器名称
        """
        # 不同的平台应该使用的计时器不同，这个直接用timeit中的配置最好
        self.default_timer = timeit.default_timer
        # 标题
        self.title = title
        self.data = []
        self.start_clock = float('nan')

    def start(self):
        self.start_clock = self.default_timer()

    def stop(self):
        self.data.append(self.default_timer() - self.start_clock)

    def report(self, msg=''):
        """ 报告目前性能统计情况
        """
        msg = f'{self.title} {msg}'
        n = len(self.data)

        if n >= 1:
            print(msg, '用时(秒) ' + ValuesStat(self.data).summary(valfmt='.3f'))
        elif n == 1:
            sum_ = sum(self.data)
            print(f'{msg} 用时: {sum_:.3f}s')
        else:  # 没有统计数据，则补充执行一次stop后汇报
            print(f'{msg} 暂无计时信息')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.report()


def performit(title, stmt="pass", setup="pass", repeat=1, number=1, globals=None):
    """ 在timeit.repeat的基础上，做了层封装

    200920周日15:33，简化函数，该函数不再获得执行结果，避免重复运行

    :param title: 测试标题、名称功能
    :return: 返回原函数单次执行结果
    """
    data = timeit.repeat(stmt=stmt, setup=setup, repeat=repeat, number=number, globals=globals)
    print(title, '用时(秒) ' + ValuesStat(data).summary(valfmt='.3f'))
    return data


def perftest(title, stmt="pass", repeat=1, number=1, globals=None, res_width=None, print_=True):
    """ 与performit的区别是，自己手动循环，记录程序运行结果

    :param title: 测试标题、名称功能
    :param res_width: 运行结果内容展示的字符上限数
    :param print_: 输出报告
    :return: 返回原函数单次执行结果

    这里为了同时获得表达式返回值，就没有用标注你的timeit.repeat实现了
    """
    # 1 确保stmt是可调用对象
    if callable(stmt):
        func = stmt
    else:
        code = compile(stmt, '', 'eval')

        def func():
            return eval(code, globals)

    # 2 原函数运行结果（这里要先重载stdout）
    data = []
    res = ''
    for i in range(repeat):
        start = time.time()
        for j in range(number):
            res = func()
        data.append(time.time() - start)

    # 3 报告格式
    if res_width is None:
        # 如果性能报告比较短，只有一次测试，那res_width默认长度可以高一点
        res_width = 50 if len(data) > 1 else 200
    if res is None:
        res = ''
    else:
        res = '运行结果：' + shorten(str(res), res_width)
    if print_:
        print(title, '用时(秒) ' + ValuesStat(data).summary(valfmt='.3f'), res)

    return data


class PerfTest:
    """ 这里模仿了unittest的机制

    v0.0.38 重要改动，将number等参数移到perf操作，而不是类初始化中操作，继承使用上会更简单
    """

    def perf(self, number=1, repeat=1, globals=None):
        """

        :param number: 有些代码运算过快，可以多次运行迭代为一个单元
        :param number: 对单元重复执行次数，最后会计算平均值、标准差
        """
        # 1 找到所有perf_为前缀，且callable的函数方法
        funcnames = []
        for k in dir(self):
            if k.startswith('perf_'):
                if callable(getattr(self, k)):
                    funcnames.append(k)

        # 2 自然排序
        funcnames = natural_sort(funcnames)
        funcnames2 = listalign([fn[5:] for fn in funcnames], 'r')
        for i, funcname in enumerate(funcnames):
            perftest(funcnames2[i], getattr(self, funcname),
                     number=number, repeat=repeat, globals=globals)


____arrow = """
"""


class Datetime(arrow.Arrow):
    r"""时间戳类

    包含功能：
        各种日期格式 -> 标准日期格式 -> 各种类型的输出格式
        以及日期间的运算

    TODO 这个库的初始化接口写的太灵活了，需要一定的限制，否则会造成使用者的模糊
    """

    def __init__(self, *argv, fold=0):
        r"""超高智能日期识别，能处理常见的各种输入类型来初始化一个标准时间戳值

        :param fold: 暂时也没研究这个参数有什么用，就是为了兼容性先挂着

        # 建议使用 Datetime.now()
        >> Datetime()  # 没有参数的时候，初始化为当前时间
        <Datetime [2020-03-22T01:00:07.035481+00:00]>

        # 正常初始化方法
        >>> Datetime(2017, 2, 5, 8, 26, 58, 861782)  # 标准的时间初始化方法
        <Datetime [2017-02-05T08:26:58.861782+08:00]>
        >>> Datetime(2017)  # 可以省略月、日等值
        <Datetime [2017-01-01T00:00:00+08:00]>

        # 建议使用 Datetime.strptime
        >>> Datetime('2019年3月6日', '%Y年%m月%d日')  # 指定格式
        <Datetime [2019-03-06T00:00:00+08:00]>
        >>> Datetime('w200301周日', 'w%y%m%d周日')  # 周日必须写全，有缺失会报ValueError
        <Datetime [2020-03-01T00:00:00+08:00]>

        >>> Datetime(180213)
        <Datetime [2018-02-13T00:00:00+08:00]>
        >>> Datetime('180213')
        <Datetime [2018-02-13T00:00:00+08:00]>

        >>> Datetime('2015-06-15_22-19-01_HDR.jpg')
        <Datetime [2015-06-15T22:19:01+08:00]>
        >>> Datetime('IMG_20150615_2219011234_HDR.jpg')
        <Datetime [2015-06-15T22:19:01.001234+08:00]>
        >>> Datetime('_2015.6.15_22:19:02')
        <Datetime [2015-06-15T22:19:02+08:00]>
        """
        dt = None
        # 1 没有参数则默认当前运行时间
        if not argv:
            dt = datetime.datetime.now()
        if not dt and isinstance(argv[0], datetime.date):
            dt = datetime.datetime(argv[0].year, argv[0].month, argv[0].day)
        if not dt and isinstance(argv[0], (datetime.datetime, arrow.Arrow, Datetime)):
            dt = argv[0]
        if not dt and isinstance(argv[0], float):
            dt = datetime.datetime.fromtimestamp(argv[0])
        # 2 优先尝试用标准的datetime初始化方法
        if not dt:
            dt = Datetime._datetime(argv)
        # 3 如果上述解析不了，且argv恰好为两个参数，则判断为使用strptime初始化
        if not dt and len(argv) == 2:
            dt = datetime.datetime.strptime(str(argv[0]), argv[1])
        # 4 判断是否我个人特用的六位日期标记
        if not dt:
            dt = Datetime._six_digits_date(argv[0])
        # 5 如果仍然解析不了，开始使用一个智能推导算法
        if not dt:
            dt = Datetime._parse_time_string(argv[0])
        # 6 最后任何解析方案都失败，则报错
        if not dt:
            print(f'无法解析输入的时间标记 argv: {argv}，现重置为当前时间点')
            dt = datetime.datetime.now()

        super().__init__(dt.year, dt.month, dt.day,
                         dt.hour, dt.minute, dt.second, dt.microsecond,
                         'local', fold=getattr(dt, 'fold', 0))

    @classmethod
    def strptime(cls, data_strnig, format):
        raise NotImplementedError

    @staticmethod
    def _datetime(argv):
        args, n = list(argv), len(argv)
        if n < 3:  # 若没填写月、日，默认1月、1日
            args = args + [1] * (3 - n)
        try:
            return datetime.datetime(*args)
        except (ValueError, TypeError) as e:
            return None

    @staticmethod
    def _six_digits_date(s):
        """主要是我个人常用的日期标注格式
        """
        s, dt = str(s), None
        if re.match(r'\d{6}$', s):
            year = int(s[:2])
            year = 2000 + year if year < 50 else 1900 + year  # 小于50默认是20xx年，否则认为是19xx年
            dt = Datetime._datetime([year, int(s[2:4]), int(s[4:])])
        return dt

    @staticmethod
    def _parse_time_string(s):
        r"""只对2000~2099年的时间点有效
        """
        data, break_flag = [], False

        def parse(pattern, left=None, right=None):
            """通用底层解析器"""
            nonlocal break_flag, s
            if break_flag: return

            m = re.search(pattern, s)
            if m:
                d = int(m.group())
                if left and d < left:
                    break_flag = True
                elif right and d > right:
                    break_flag = True
                else:
                    data.append(d)
                    s = s[m.end():]
            else:
                break_flag = True

        parse(r'20\d{2}')
        parse(r'\d\d?', 1, 12)  # 有连续两个数组就获得两个，否则获得一个也行
        parse(r'\d\d?', 1, 31)  # 这样其实不严谨，有的月份不到31天，不过反正_datetime会返回None代表处理失败的
        parse(r'\d\d?', 0, 23)
        parse(r'\d\d?', 0, 59)
        parse(r'\d\d?', 0, 59)
        parse(r'\d{1,6}')  # microsecond

        return Datetime._datetime(data)

    def replace(self, **kwargs):
        """
        'weekday'，1~7分别代表周一到周日

        >>> Datetime(180826).replace(weekday=1).strftime('%y%m%d周%k')
        '180820周一'
        >>> Datetime(180826).replace(weekday=3).strftime('%y%m%d周%k')
        '180822周三'
        """
        a = self
        if 'weekday' in kwargs:
            if set(kwargs.keys()) & {'year', 'month', 'day'}:
                raise ValueError('weekday参数不能混合年月日的修改使用')
            a = a - a.isoweekday() + kwargs['weekday']  # 先减去当前星期几，再加上1~7的一个目标星期
            del kwargs['weekday']
        if kwargs:
            a = a.replace(**kwargs)
        return Datetime(a)

    def __add__(self, other):
        """加减数值时，单位按天处理"""
        if isinstance(other, (int, float)):
            other = datetime.timedelta(other)
        return Datetime(super().__add__(other))

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = datetime.timedelta(other)
        return Datetime(super().__sub__(other))

    def shift(self, **kwargs):
        """shift 有点像游标卡尺，可以左右两边进行加减移位操作，加减的对象可以是年月日时分秒和星期。

        >>> a = Datetime(2018, 8, 24)
        >>> a.shift(months=-1)
        <Datetime [2018-07-24T00:00:00+08:00]>
        >>> a.shift(months=-1).format("YYYYMM")
        '201807'
        >>> a.shift(weeks=1)
        <Datetime [2018-08-31T00:00:00+08:00]>
        """
        return Datetime(super().shift(**kwargs))

    def to(self, tz):
        """to 可以将一个本地时区转换成其它任意时区，例如：

        >>> a = Datetime(2020, 6, 1, 17)
        >>> a.to("utc")
        <Datetime [2020-06-01T09:00:00+08:00]>
        >>> a.to("utc").to("local")
        <Datetime [2020-06-01T09:00:00+08:00]>
        >>> a.to("America/New_York")
        <Datetime [2020-06-01T05:00:00+08:00]>
        """
        return Datetime(super().to(tz))

    def humanize(self, other=None, locale="zh", only_distance=False, granularity="auto"):
        r""" humanize 方法是相对于当前时刻表示为“多久以前”的一种可读行字符串形式，
                默认是英文格式，指定 locale 可显示相应的语言格式。

        >>> a = Datetime().shift(hours=-6)
        >>> a.humanize()
        '6小时前'
        >>> a.humanize(locale='en')
        '6 hours ago'

        这是从 https://mp.weixin.qq.com/s/DqD_PmrspMeytloV_o54IA 摘录的Arrow的文档
            并不是该Datetime类本身的文档，但意思差不多，可以参考
        """
        return super().humanize(other, locale, only_distance, granularity)

    def strftime(self, fmt='%Y/%m/%d'):
        r"""做了简单格式拓展

        官方支持的符号： https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior

        我自己扩展的功能：
        %k，周几：一、二、三、四、五、六、日

        >>> dt = Datetime(2020, 2, 28, 9, 10, 52)
        >>> dt.strftime('%y%m%d')
        '200228'
        >>> dt.strftime('%Y/%m/%d')
        '2020/02/28'
        >>> dt.strftime('%y%m%d周%k')  # 我自己最常用的格式
        '200228周五'
        """
        # 先用占位符替代中文，和我扩展的%k等标记
        fmt1 = re.sub(r'([\u4e00-\u9fa5，。；？（）【】、①-⑨]|%k)', r'{placeholder}', fmt)
        tag = super().strftime(fmt1)
        if fmt1 != fmt:
            texts = re.findall(r'([\u4e00-\u9fa5，。；？（）【】、①-⑨]|%k)', fmt)
            for i in range(len(texts)):
                tag = tag.replace('{placeholder}', texts[i], 1)
            tag = tag.replace('%k', '日一二三四五六'[self.isoweekday() % 7])
        return tag
