#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/02 11:09


from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from pyxllib.basic import *


def east_asian_len(s, ambiguous_width=None):
    import pandas.io.formats.format as fmt
    return fmt.EastAsianTextAdjustment().len(s)


def east_asian_shorten(s, width=50, placeholder='...'):
    """考虑中文情况下的域宽截断

    :param s: 要处理的字符串
    :param width: 宽度上限，仅能达到width-1的宽度
    :param placeholder: 如果做了截断，末尾补足字符

    # width比placeholder还小
    >>> east_asian_shorten('a', 2)
    'a'
    >>> east_asian_shorten('a啊b'*4, 3)
    '..'
    >>> east_asian_shorten('a啊b'*4, 4)
    '...'

    >>> east_asian_shorten('a啊b'*4, 5, '...')
    'a...'
    >>> east_asian_shorten('a啊b'*4, 11)
    'a啊ba啊...'
    >>> east_asian_shorten('a啊b'*4, 16, '...')
    'a啊ba啊ba啊b...'
    >>> east_asian_shorten('a啊b'*4, 18, '...')
    'a啊ba啊ba啊ba啊b'
    """
    # 一、如果字符串本身不到width设限，返回原值
    s = textwrap.shorten(s, width * 3, placeholder='')  # 用textwrap的折行功能，尽量不删除文本
    n = east_asian_len(s)
    if n < width: return s

    # 二、如果输入的width比placeholder还短
    width -= 1
    m = east_asian_len(placeholder)
    if width <= m:
        return placeholder[:width]

    # 三、需要添加 placeholder
    # 1 计算长度
    width -= m

    # 2 截取s
    try:
        s = s.encode('gbk')[:width].decode('gbk', errors='ignore')
    except UnicodeEncodeError:
        i, count = 0, m
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
    import pandas as pd

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


def prettifystr(s):
    """对一个对象用更友好的方式字符串化

    :param s: 输入类型不做限制，会将其以友好的形式格式化
    :return: 格式化后的字符串
    """
    title = ''
    if isinstance(s, str):
        pass
    elif isinstance(s, Counter):  # Counter要按照出现频率显示
        li = s.most_common()
        title = f'collections.Counter长度：{len(s)}\n'
        # 不使用复杂的pd库，先简单用pprint即可
        # df = pd.DataFrame.from_records(s, columns=['value', 'count'])
        # s = dataframe_str(df)
        s = pprint.pformat(li)
    elif isinstance(s, (list, tuple)):
        title = f'{typename(s)}长度：{len(s)}\n'
        s = pprint.pformat(s)
    elif isinstance(s, (dict, set)):
        title = f'{typename(s)}长度：{len(s)}\n'
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


def dict2df(d):
    """dict类型转DataFrame类型"""
    name = typename(d)
    if isinstance(d, Counter):
        li = d.most_common()
    else:
        li = dict2list(d, nsort=True)
    return pd.DataFrame.from_records(li, columns=(f'{name}-key', f'{name}-value'))


def list2df(li):
    if li and isinstance(li[0], (list, tuple)):  # 有两维时按表格显示
        df = pd.DataFrame.from_records(li)
    else:  # 只有一维时按一列显示
        df = pd.DataFrame(pd.Series(li), columns=(typename(li),))
    return df


def try2df(arg):
    """尝试将各种不同的类型转成dataframe"""
    if isinstance(arg, dict):
        df = dict2df(arg)
    elif isinstance(arg, (list, tuple)):
        df = list2df(arg)
    elif isinstance(arg, pd.Series):
        df = pd.DataFrame(arg)
    else:
        df = arg
    return df


class NestedDict:
    """ 字典嵌套结构相关功能

    TODO 感觉跟 pprint 的嵌套识别美化输出相关，可能有些代码是可以结合简化的~~
    """

    @classmethod
    def has_subdict(cls, data, include_self=True):
        """是否含有dict子结构
        :param include_self: 是否包含自身，即data本身是一个dict的话，也认为has_subdict是True
        """
        if include_self and isinstance(data, dict):
            return True
        elif isinstance(data, (list, tuple, set)):
            for v in data:
                if cls.has_subdict(v):
                    return True
        return False

    @classmethod
    def to_html_table(cls, data, max_items=10):
        """ 以html表格套表格的形式，展示一个嵌套结构数据

        :param data: 数据
        :param max_items: 项目显示上限，有些数据项目太多了，要精简下
            设为假值则不设上限
        :return:

        TODO 这个速度有点慢，怎么加速？
        """

        def tohtml(d):
            if max_items:
                df = try2df(d)
                if len(df) > max_items:
                    n = len(df)
                    return df[:max_items].to_html(escape=False) + f'... {n - 1}'
                else:
                    return df.to_html(escape=False)
            else:
                return try2df(d).to_html(escape=False)

        if not cls.has_subdict(data):
            res = str(data)
        elif isinstance(data, dict):
            if isinstance(data, Counter):
                d = data
            else:
                d = dict()
                for k, v in data.items():
                    if cls.has_subdict(v):
                        v = cls.to_html_table(v, max_items=max_items)
                    d[k] = v
            res = tohtml(d)
        else:
            li = [cls.to_html_table(x, max_items=max_items) for x in data]
            res = tohtml(li)

        return res.replace('\n', ' ')


class KeyValuesCounter:
    """ 各种键值对出现次数的统计
    会递归找子字典结构，但不存储结构信息，只记录纯粹的键值对信息

    应用场景：对未知的json结构，批量读取后，显示所有键值对的出现情况
    """

    def __init__(self):
        self.kvs = defaultdict(Counter)

    def add(self, data, max_value_length=100):
        """
        :param max_value_length: 添加的值，进行截断，防止有些值太长
        """
        if not NestedDict.has_subdict(data):
            return
        elif isinstance(data, dict):
            for k, v in data.items():
                if NestedDict.has_subdict(v):
                    self.add(v)
                else:
                    self.kvs[k][shorten(str(v), max_value_length)] += 1
        else:  # 否则 data 应该是个可迭代对象，才可能含有dict
            for x in data:
                self.add(x)

    def to_html_table(self, max_items=10):
        return NestedDict.to_html_table(self.kvs, max_items=max_items)


class MatchBase:
    """ 匹配类

    MatchBase(ys, cmp_func).matches(xs, least_score)
    """

    def __init__(self, ys, cmp_func):
        self.ys = list(ys)
        self.cmp_func = cmp_func

    def __getitem__(self, idx):
        return self.ys[idx]

    # def __del__(self, idx):
    #     del self.ys[idx]

    def __len__(self):
        return len(self.ys)

    def match(self, x, k=1):
        """ 匹配一个对象

        :param x: 待匹配的一个对象
        :param k: 返回次优的几个结果
        :return:
            当k=1时，返回 (idx, score)
            当k>1时，返回类似matchs的return结构
        """
        scores = [self.cmp_func(x, y) for y in self.ys]
        if k == 1:
            score = max(scores)
            idx = scores.index(score)
            return idx, score
        else:
            idxs = np.argsort(scores)
            idxs = idxs[::-1][:k]
            return [(idx, scores[idx]) for idx in idxs]

    def matches(self, xs, least_score=sys.float_info.epsilon):
        """ 同时匹配多个对象

        :param xs: 要匹配的一组对象
        :param least_score: 分数必须要不小于least_score才能算匹配，否则属于找不到匹配项
        :return: 为每个x找到一个最佳的匹配y，存储其下标和对应的分值
            [(idx0, score0), (idx1, score1), ...]  长度 = len(xs)

        匹配到的y不会放回，如果是可放回的，可以自己用match进行列表推导式直接计算
        ms = [self.match(x) for x in xs]
        """
        m = len(self.ys)
        used = set()

        res = []
        for x in xs:
            ms = self.match(x, k=m)
            for idx, score in ms:
                if score < least_score:
                    res.append((-1, score))
                    break
                if idx not in used:
                    used.add(idx)
                    res.append((idx, score))
                    break

        return res
