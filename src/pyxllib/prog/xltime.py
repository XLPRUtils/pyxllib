#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 23:28

"""

xltime.py 是一个全能型时间处理工具库，旨在简化 Python 原生 datetime 繁琐的操作。它通过封装 NbTime 库，为开发者提供了一个支持无限链式调用、智能格式识别和人性化时间计算的统一接口 XlTime。
XlTime兼容原本datetime、NbTime的所有功能。

# 1 datetime具有的基础功能

Python 原生 datetime 模块提供了时间处理的基础能力。

### 1.1 获取与创建
- 获取当前时间：datetime.now()
- 构造指定日期：datetime(2025, 1, 1, 12, 0)
- 从时间戳转换：datetime.fromtimestamp(1700000000)
- 组合日期与时间：datetime.combine(date_obj, time_obj)

### 1.2 解析与格式化
- 字符串转对象：datetime.strptime('2025-01-01', '%Y-%m-%d')
- 对象转字符串：dt.strftime('%Y-%m-%d %H:%M:%S')

### 1.3 常用属性与方法
- 时间分量：.year, .month, .day, .hour, .minute, .second, .microsecond
- 星期索引：.weekday()（周一为0）
- 日期部分：.date()
- 时间部分：.time()

### 1.4 时间差计算 (timedelta)
- 创建时间差：timedelta(days=1, hours=2)
- 时间加减：dt + timedelta(days=7)
- 两个时间相减：dt2 - dt1 -> 返回 timedelta 对象

# 2 NbTime简介

NbTime 类作为面向对象的时间处理工具，支持无限链式调用，并作为万能适配器处理各种时间类型的转换。
这里这里是介绍XlTime父类NbTime的功能，最终实际使用要用XlTime作为统一处理接口。

## 2.1 初始化与入参支持
NbTime 可自动识别多种类型的入参，无需用户手动转换。
- 无参（当前时间）：NbTime()
- 时间戳：自动识别 10 位（秒）或 13 位（毫秒）时间戳。例：NbTime(1709192429)
- Datetime 对象：NbTime(datetime.datetime.now())
- 字符串：无需指定格式，自动识别 ISO、紧凑型或自然语言格式。例：NbTime('2023-05-06 01:02:03.886')
- 其他对象：支持 arrow.Arrow、DateTimeValue 及嵌套的 NbTime 对象。

## 2.2 时区管理
NbTime 默认处理 Aware datetime（带时区的时间），支持时区设置与转换。
- 初始化设置：NbTime(time_zone='Asia/Shanghai')
- 时区转换 (to_tz)：生成一个指定时区的新 NbTime 对象。例：NbTime(...).to_tz('UTC+8')
- 全局默认设置：NbTime.set_default_time_zone('UTC+8')

## 2.3 时间计算 (shift)
支持链式时间推移计算，返回新的 NbTime 对象。
- NbTime().shift(days=3)
- NbTime().shift(hours=1, minutes=10)
- 链式操作：NbTime().shift(days=-1).shift(hours=2)

## 2.4 格式化与输出
- 内置属性：.datetime, .timestamp, .timestamp_millisecond, .datetime_str, .date_str, .time_str, .arrow
- 字符串格式化：NbTime().get_str('%Y%m%d')
- 全局默认格式：NbTime.set_default_formatter(NbTime.FORMATTER_DATETIME_NO_ZONE)

## 2.5 比较与人性化显示
- 比较：支持标准运算符 (>, <, ==) 直接比较 NbTime 对象。
- 人性化显示：将时间差转换为自然语言。例：NbTime().shift(days=-3).humanize() -> '3 days ago'

# 3 XlTime进一步扩展的功能

详见文件下具体的功能接口。
"""

import typing
import datetime
import re
import logging

from pyxllib.prog.lazyimport import lazy_import

try:
    from fastcore.basics import GetAttr
except ModuleNotFoundError:
    GetAttr = lazy_import("from fastcore.basics import GetAttr", "fastcore")

try:
    import nb_time
    from nb_time import NbTime

    nb_time.logger.setLevel(logging.CRITICAL)  # 把官方的警告日志关掉
    NbTime.set_default_time_zone("UTC+8")
except ModuleNotFoundError:
    NbTime = lazy_import("from nb_time import NbTime", "nb-time")

try:
    from fastcore.utils import GetAttr
except ModuleNotFoundError:
    GetAttr = lazy_import("from fastcore.utils import GetAttr", "fastcore")


def parse_datetime(*argv):
    """解析字符串日期时间

    解析算法偏私人应用，规则虽然多，但反而变得混乱隐晦，
    实际开发能明确使用具体接口，尽量还是用具体接口。

    一般用XlTime初始化自带的格式解析已经够用了。

    # 建议使用 Datetime.strptime
    >>> parse_datetime('2019年3月6日', '%Y年%m月%d日')  # 指定格式
    datetime.datetime(2019, 3, 6, 0, 0)
    >>> parse_datetime('w200301周日', 'w%y%m%d周日')  # 周日必须写全，有缺失会报ValueError
    datetime.datetime(2020, 3, 1, 0, 0)

    >>> parse_datetime('2019.3.6 22:30:40')
    datetime.datetime(2019, 3, 6, 22, 30, 40)

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
        """主要是我个人常用的日期标注格式"""
        s, dt = str(s), None
        if re.match(r"\d{6}$", s):
            year = int(s[:2])
            year = 2000 + year if year < 50 else 1900 + year  # 小于50默认是20xx年，否则认为是19xx年
            dt = _datetime([year, int(s[2:4]), int(s[4:])])
        return dt

    def _parse_time_string(s):
        r"""只对2000~2099年的时间点有效"""
        data, break_flag = [], False

        def parse(pattern, left=None, right=None):
            """通用底层解析器"""
            nonlocal break_flag, s
            if break_flag:
                return

            m = re.search(pattern, s)
            if m:
                d = int(m.group())
                if left and d < left:
                    break_flag = True
                elif right and d > right:
                    break_flag = True
                else:
                    data.append(d)
                    s = s[m.end() :]
            else:
                break_flag = True

        parse(r"20\d{2}")
        parse(r"\d\d?", 1, 12)  # 有连续两个数组就获得两个，否则获得一个也行
        parse(r"\d\d?", 1, 31)  # 这样其实不严谨，有的月份不到31天，不过反正_datetime会返回None代表处理失败的
        parse(r"\d\d?", 0, 23)
        parse(r"\d\d?", 0, 59)
        parse(r"\d\d?", 0, 59)
        parse(r"\d{1,6}")  # microsecond

        return _datetime(data)

    dt = None
    # 1 没有参数则默认当前运行时间
    if not argv:
        dt = datetime.datetime.now()
    if not dt and isinstance(argv[0], datetime.datetime):
        dt = argv[0]
    if not dt and isinstance(argv[0], datetime.date):
        # 要转成datetime类型，time部分默认00:00:00
        dt = datetime.datetime.combine(argv[0], datetime.time())
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
    """解析字符串所代表的时间差

    >>> parse_timedelta('38:18')
    datetime.timedelta(seconds=2298)
    >>> parse_timedelta('03:55')
    datetime.timedelta(seconds=235)
    >>> parse_timedelta('1:34:25')
    datetime.timedelta(seconds=5665)
    """
    parts = s.split(":")[::-1]
    d = {k: int(v) for k, v in zip(["seconds", "minutes", "hours"], parts)}
    td = datetime.timedelta(**d)
    return td


class XlTime(GetAttr, NbTime):
    """基于NbTime，进一步扩展我自己的XlTime的常用功能"""

    _default = "datetime"

    def __1_基础操作(self):
        pass

    def fix_hints(self):
        """使用GetAttr的时候，是以"组合:有一个"的逻辑来模拟实现了"继承:是一个"的效果
        这在动态运行上很好用，但静态代码分析上，pycharm等IDE就识别不了了，
        此时可以运行这个fix_hints补丁就能修复静态提示了：self = self.fix_hints()

        这个可以看作是一个调试，脚手架功能，实际运行中删掉这一层是没有任何影响的
        所以如果在某个高频运算中用到了这个操作，真的担心这一点点性能开销，可以在生产环境中删掉
        只需要在调试、需要跳转，查看成员的时候临时打开就好

        详细文档：https://www.yuque.com/xlpr/pyxllib/fix_hints
        """

        # Hint写出期望IDE认为的类继承关系功能
        class Hint(XlTime, datetime.datetime):
            pass

        # typing.cast不会做任何实际功能上的变动，只是纯粹给IDE的静态分析提示这里可以当成Hint处理
        return typing.cast(Hint, self)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} [{self.datetime_str}] ({self.time_zone_str})>"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.datetime_str}] ({self.time_zone_str})>"

    def __sub__(self, other) -> datetime.timedelta:
        assert isinstance(other, NbTime), f"类型不匹配 {type(other)}"
        return self.datetime - other.datetime

    def __2_自己常用的其他偏移(self):
        pass

    def add_days(self, days):
        """增加指定天数，支持负数"""
        return self.shift(days=days)

    def monday(self):
        """移动到本周周一"""
        return self.shift(days=-self.datetime.weekday())

    def next_week(self):
        """移动到下一周"""
        return self.shift(days=7)

    def prev_week(self):
        """移动到上一周"""
        return self.shift(days=-7)

    def __3_其他功能():
        pass

    def week_idx(self, anchor_date=None):
        """计算当前日期距离 anchor_date 是第几周

        :param anchor_date: 锚点日期
            支持 str('2024-01-01') 或 datetime/XlTime 对象
            None: 默认以 1992-01-06（周一）为锚点，且该日记为第1周
        :return int: 连续的周索引，可以是负数

        >>> XlTime('2024-01-01').week_idx('2024-01-01')
        0
        >>> XlTime('2023-10-23').week_idx('2024-01-01')
        -10
        >>> XlTime('1992-01-06').week_idx()
        1
        >>> XlTime('2026-01-13').week_idx()
        1776
        """
        # 1. 确定锚点日期
        offset = 0
        if anchor_date is None:
            anchor_date = "1992-01-06"
            offset = 1

        # 统一转换为 date 对象
        base = self.__class__(anchor_date).datetime.date()

        # 2. 获取当前日期
        current = self.datetime.date()

        # 3. 计算周数
        return (current - base).days // 7 + offset

    def weektag(self):
        """
        :return: 周标签名，例如 'w250414'，表示所属周的周一是2025年4月14日
        """
        monday = self.monday()  # 获取本周周一
        tag = "w" + monday.strftime("%y%m%d")
        return tag

    def daytag(self):
        """
        :return: 日标签名，例如 '250415周二'，表示当天的日期标记，一般是语雀周报中使用
        """
        ch = "一二三四五六日"[self.dt.weekday()]
        tag = self.dt.strftime("%y%m%d") + "周" + ch
        return tag

    def week_daytags(self):
        # 循环获得本周每天的daytag
        monday = self.monday()
        tags = [monday.add_days(i).daytag() for i in range(7)]
        return tags


if __name__ == "__main__":
    pass
