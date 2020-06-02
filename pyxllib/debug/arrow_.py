#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/31


import datetime
import os
import re
import subprocess


try:
    import arrow
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'arrow'])
    import arrow


class Datetime(arrow.Arrow):
    r"""时间戳类

    包含功能：
        各种日期格式 -> 标准日期格式 -> 各种类型的输出格式
        以及日期间的运算

    TODO 这个库的初始化接口写的太灵活了，需要一定的限制，否则会造成使用者的模糊
    """

    def __init__(self, *argv):
        r"""超高智能日期识别，能处理常见的各种输入类型来初始化一个标准时间戳值

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
        # 1、没有参数则默认当前运行时间
        if not argv:
            dt = datetime.datetime.now()
        if not dt and isinstance(argv[0], datetime.date):
            dt = datetime.datetime(argv[0].year, argv[0].month, argv[0].day)
        if not dt and isinstance(argv[0], (datetime.datetime, arrow.Arrow, Datetime)): dt = argv[0]
        if not dt and isinstance(argv[0], float): dt = datetime.datetime.fromtimestamp(argv[0])
        # 2、优先尝试用标准的datetime初始化方法
        if not dt:
            dt = Datetime._datetime(argv)
        # 3、如果上述解析不了，且argv恰好为两个参数，则判断为使用strptime初始化
        if not dt and len(argv) == 2:
            dt = datetime.datetime.strptime(str(argv[0]), argv[1])
        # 4、判断是否我个人特用的六位日期标记
        if not dt:
            dt = Datetime._six_digits_date(argv[0])
        # 5、如果仍然解析不了，开始使用一个智能推导算法
        if not dt:
            dt = Datetime._parse_time_string(argv[0])
        # 6、最后任何解析方案都失败，则报错
        if not dt:
            print(f'无法解析输入的时间标记 argv: {argv}，现重置为当前时间点')
            dt = datetime.datetime.now()

        super(Datetime, self).__init__(dt.year, dt.month, dt.day,
                                       dt.hour, dt.minute, dt.second, dt.microsecond,
                                       'local')

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
        return Datetime(super(Datetime, self).__add__(other))

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = datetime.timedelta(other)
        return Datetime(super(Datetime, self).__sub__(other))

    def shift(self, **kwargs):
        """shift 有点像游标卡尺，可以左右两边进行加减移位操作，加减的对象可以是年月日时分秒和星期。
        TODO 这文档什么鬼，输出还有Arrow，改成Datatime呀
        >>> a = Datetime(2018, 8, 24)
        >>> a.shift(months=-1)
        <Datetime [2018-07-24T00:00:00+08:00]>
        >>> a.shift(months=-1).format("YYYYMM")
        '201807'
        >>> a.shift(weeks=1)
        <Datetime [2018-08-31T00:00:00+08:00]>
        """
        return Datetime(super(Datetime, self).shift(**kwargs))

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
        return Datetime(super(Datetime, self).to(tz))

    def humanize(self, other=None, locale="zh", only_distance=False, granularity="auto"):
        r""" humanize 方法是相对于当前时刻表示为“多久以前”的一种可读行字符串形式，
                默认是英文格式，指定 locale 可显示相应的语言格式。
        >>> a = Datetime.now().shift(hours=-6)
        >>> a.humanize()
        '6小时前'
        >>> a.humanize(locale='en')
        '6 hours ago'

        这是从 https://mp.weixin.qq.com/s/DqD_PmrspMeytloV_o54IA 摘录的Arrow的文档
            并不是该Datetime类本身的文档，但意思差不多，可以参考
        """
        return super(Datetime, self).humanize(other, locale, only_distance, granularity)

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
        tag = super(Datetime, self).strftime(fmt1)
        if fmt1 != fmt:
            texts = re.findall(r'([\u4e00-\u9fa5，。；？（）【】、①-⑨]|%k)', fmt)
            for i in range(len(texts)):
                tag = tag.replace('{placeholder}', texts[i], 1)
            tag = tag.replace('%k', '日一二三四五六'[self.isoweekday() % 7])
        return tag


def demo_datetime():
    from pyxllib.debug.dprint import dprint

    dprint(Datetime.now())
    # [05]arrow_.py/247: Datetime.now()<__main__.Datetime>=<Datetime [2020-06-01T16:54:45.365788+08:00]>

    # 获取文件的创建时间，st_ctime获得的事timestamp格式 1579529472.2958975
    dprint(Datetime(os.stat(__file__).st_ctime))
    # [05]arrow_.py/251: Datetime.now()<__main__.Datetime>=<Datetime [2020-06-01T16:54:45.365788+08:00]>


if __name__ == '__main__':
    demo_datetime()
