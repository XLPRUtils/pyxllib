#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 23:28

"""

旧的Datetime类已删除，原来稍微比较有用的一个功能，是

替换周几，这个算法可以参考：
a = a - a.isoweekday() + kwargs['weekday']  # 先减去当前星期几，再加上1~7的一个目标星期

"""

import datetime
import re


def parse_datetime(*argv):
    """ 解析字符串日期时间

    解析算法偏私人应用，规则虽然多，但反而变得混乱隐晦，
    实际开发能明确使用具体接口，尽量还是用具体接口。

    # 建议使用 Datetime.strptime
    >>> parse_datetime('2019年3月6日', '%Y年%m月%d日')  # 指定格式
    datetime.datetime(2019, 3, 6, 0, 0)
    >>> parse_datetime('w200301周日', 'w%y%m%d周日')  # 周日必须写全，有缺失会报ValueError
    datetime.datetime(2020, 3, 1, 0, 0)

    >>> parse_datetime(180213)
    datetime.datetime(2018, 2, 13, 0, 0)
    >>> parse_datetime('180213')
    datetime.datetime(2018, 2, 13, 0, 0)

    # 以下只对2000~2099年的时间点有效
    >>> parse_datetime('2015-06-15_22-19-01_HDR.jpg')
    datetime.datetime(2015, 6, 15, 22, 19, 1)
    >>> parse_datetime('IMG_20150615_2219011234_HDR.jpg')
    datetime.datetime(2015, 6, 15, 22, 19, 1, 1234)
    >>> parse_datetime('_2015.6.15_22:19:02')
    datetime.datetime(2015, 6, 15, 22, 19, 2)
    """

    def _datetime(argv):
        args, n = list(argv), len(argv)
        if n < 3:  # 若没填写月、日，默认1月、1日
            args = args + [1] * (3 - n)
        try:
            return datetime.datetime(*args)
        except (ValueError, TypeError) as e:
            return None

    def _six_digits_date(s):
        """主要是我个人常用的日期标注格式
        """
        s, dt = str(s), None
        if re.match(r'\d{6}$', s):
            year = int(s[:2])
            year = 2000 + year if year < 50 else 1900 + year  # 小于50默认是20xx年，否则认为是19xx年
            dt = _datetime([year, int(s[2:4]), int(s[4:])])
        return dt

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

        return _datetime(data)

    dt = None
    # 1 没有参数则默认当前运行时间
    if not argv:
        dt = datetime.datetime.now()
    if not dt and isinstance(argv[0], float):
        dt = datetime.datetime.fromtimestamp(argv[0])
    # 2 如果上述解析不了，且argv恰好为两个参数，则判断为使用strptime初始化
    if not dt and len(argv) == 2:
        dt = datetime.datetime.strptime(str(argv[0]), argv[1])
    # 3 判断是否我个人特用的六位日期标记
    if not dt:
        dt = _six_digits_date(argv[0])
    # 4 如果仍然解析不了，开始使用一个智能推导算法
    if not dt:
        dt = _parse_time_string(argv[0])
    # 5 最后任何解析方案都失败，则返回None

    return dt


def parse_timedelta(s):
    """ 解析字符串所代表的时间差

    >>> parse_timedelta('38:18')
    datetime.timedelta(seconds=2298)
    >>> parse_timedelta('03:55')
    datetime.timedelta(seconds=235)
    >>> parse_timedelta('1:34:25')
    datetime.timedelta(seconds=5665)
    """
    parts = s.split(':')[::-1]
    d = {k: int(v) for k, v in zip(['seconds', 'minutes', 'hours'], parts)}
    td = datetime.timedelta(**d)
    return td

