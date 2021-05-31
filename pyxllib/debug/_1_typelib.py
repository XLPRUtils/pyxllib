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


class TypeConvert:
    @classmethod
    def dict2list(cls, d: dict, *, nsort=False):
        """ 字典转n*2的list

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

    @classmethod
    def dict2df(cls, d):
        """dict类型转DataFrame类型"""
        name = typename(d)
        if isinstance(d, Counter):
            li = d.most_common()
        else:
            li = cls.dict2list(d, nsort=True)
        return pd.DataFrame.from_records(li, columns=(f'{name}-key', f'{name}-value'))

    @classmethod
    def list2df(cls, li):
        if li and isinstance(li[0], (list, tuple)):  # 有两维时按表格显示
            df = pd.DataFrame.from_records(li)
        else:  # 只有一维时按一列显示
            df = pd.DataFrame(pd.Series(li), columns=(typename(li),))
        return df

    @classmethod
    def list2df(cls, li):
        if li and isinstance(li[0], (list, tuple)):  # 有两维时按表格显示
            df = pd.DataFrame.from_records(li)
        else:  # 只有一维时按一列显示
            df = pd.DataFrame(pd.Series(li), columns=(typename(li),))
        return df

    @classmethod
    def try2df(cls, arg):
        """尝试将各种不同的类型转成dataframe"""
        if isinstance(arg, dict):
            df = cls.dict2df(arg)
        elif isinstance(arg, (list, tuple)):
            df = cls.list2df(arg)
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
                df = TypeConvert.try2df(d)
                if len(df) > max_items:
                    n = len(df)
                    return df[:max_items].to_html(escape=False) + f'... {n - 1}'
                else:
                    return df.to_html(escape=False)
            else:
                return TypeConvert.try2df(d).to_html(escape=False)

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


class MatchPairs:
    """ 匹配类，对X,Y两组数据中的x,y等对象按照cmp_func的规则进行相似度配对

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
            当 k = 1 时，返回 (idx, score)
            当 k > 1 时，返回 [(idx1, score1), (idx2, score2), ...]
        """
        scores = [self.cmp_func(x, y) for y in self.ys]
        if k == 1:
            score = max(scores)
            idx = scores.index(score)
            return idx, score
        else:
            # 按权重从大到小排序
            idxs = np.argsort(scores)
            idxs = idxs[::-1][:k]
            return [(idx, scores[idx]) for idx in idxs]

    def matches(self, xs):
        """ 对xs中每个元素都找到一个最佳匹配对象

        注意：这个功能是支持ys中的元素被重复匹配的，而且哪怕相似度很低，也会返回一个最佳匹配结果
            如果想限定相似度，或者不支持重复匹配，请到隔壁使用 matchpairs

        :param xs: 要匹配的一组对象
        :return: 为每个x找到一个最佳的匹配y，存储其下标和对应的分值
            [(idx0, score0), (idx1, score1), ...]  长度 = len(xs)

        >>> m = MatchPairs([1, 5, 8, 9, 2], lambda x,y: 1-abs(x-y)/max(x,y))
        >>> m.matches([4, 6])  # 这里第1个值是下标，所以分别是对应5、8
        [(1, 0.8), (1, 0.8333333333333334)]

        # 要匹配的对象多于实际ys，则只会返回前len(ys)个结果
        # 这种情况建议用matchpairs功能实现，或者实在想用就对调xs、ys
        >>> m.matches([4, 6, 1, 2, 9, 4, 5])
        [(1, 0.8), (1, 0.8333333333333334), (0, 1.0), (4, 1.0), (3, 1.0), (1, 0.8), (1, 1.0)]
        """
        return [self.match(x) for x in xs]


def matchpairs(xs, ys, cmp_func, least_score=sys.float_info.epsilon, *,
               key=None, index=False):
    r""" 匹配两组数据

    :param xs: 第一组数据
    :param ys: 第二组数据
    :param cmp_func: 所用的比较函数，值越大表示两个对象相似度越高
    :param least_score: 允许匹配的最低分，默认必须要大于0
    :param key: 是否需要对xs, ys进行映射后再传入 cmp_func 操作
    :param index: 返回的不是原值，而是下标
    :return: 返回结构[(x1, y1, score1), (x2, y2, score2), ...]，注意长度肯定不会超过min(len(xs), len(ys))

    注意：这里的功能①不支持重复匹配，②任何一个x,y都有可能没有匹配到
        如果每个x必须都要有一个匹配，或者支持重复配对，请到隔壁使用 MatchPairs

    TODO 这里很多中间步骤结果都是很有分析价值的，能改成类，然后支持分析中间结果？
    TODO 这样全量两两比较是很耗性能的，可以加个参数草算，不用精确计算的功能？

    >>> xs, ys = [4, 6, 1, 2, 9, 4, 5], [1, 5, 8, 9, 2]
    >>> cmp_func = lambda x,y: 1-abs(x-y)/max(x,y)
    >>> matchpairs(xs, ys, cmp_func)
    [(1, 1, 1.0), (2, 2, 1.0), (9, 9, 1.0), (5, 5, 1.0), (6, 8, 0.75)]
    >>> matchpairs(ys, xs, cmp_func)
    [(1, 1, 1.0), (5, 5, 1.0), (9, 9, 1.0), (2, 2, 1.0), (8, 6, 0.75)]
    >>> matchpairs(xs, ys, cmp_func, 0.9)
    [(1, 1, 1.0), (2, 2, 1.0), (9, 9, 1.0), (5, 5, 1.0)]
    >>> matchpairs(xs, ys, cmp_func, 0.9, index=True)
    [(2, 0, 1.0), (3, 4, 1.0), (4, 3, 1.0), (6, 1, 1.0)]
    """
    # 0 实际计算使用的是 xs_, ys_
    if key:
        xs_ = [key(x) for x in xs]
        ys_ = [key(y) for y in ys]
    else:
        xs_, ys_ = xs, ys

    # 1 计算所有两两相似度
    n, m = len(xs), len(ys)
    all_pairs = []
    for i in range(n):
        for j in range(m):
            score = cmp_func(xs_[i], ys_[j])
            if score >= least_score:
                all_pairs.append([i, j, score])
    # 按分数权重排序，如果分数有很多相似并列，就只能按先来后到排序啦
    all_pairs = sorted(all_pairs, key=lambda v: (-v[2], v[0], v[1]))

    # 2 过滤出最终结果
    pairs = []
    x_used, y_used = set(), set()
    for p in all_pairs:
        i, j, score = p
        if i not in x_used and j not in y_used:
            if index:
                pairs.append((i, j, score))
            else:
                pairs.append((xs[i], ys[j], score))
            x_used.add(i)
            y_used.add(j)

    return pairs


class Groups:
    def __init__(self, data):
        """ 分组

        :param data: 输入字典结构直接赋值
            或者其他结构，会自动按相同项聚合

        TODO 显示一些数值统计信息，甚至图表
        TODO 转文本表达，方便bc比较
        """
        if not isinstance(data, dict):
            new_data = dict()
            # 否要要转字典类型，自动从1~n编组
            for k, v in enumerate(data, start=1):
                new_data[k] = v
            data = new_data
        self.data = data  # 字典存原数据
        self.ctr = Counter({k: len(x) for k, x in self.data.items()})  # 计数
        self.stat = ValuesStat(self.ctr.values())  # 综合统计数据

    def __repr__(self):
        ls = []
        for i, (k, v) in enumerate(self.data.items(), start=1):
            ls.append(f'{i}, {k}：{v}')
        return '\n'.join(ls)

    @classmethod
    def groupby(cls, ls, key, ykey=None):
        """
        :param ls: 可迭代等数组类型
        :param key: 映射规则，ls中每个元素都会被归到映射的key组上
            Callable[Any, 不可变类型]
            None，未输入时，默认输入的ls已经是分好组的数据
        :param ykey: 是否对分组后存储的内容y，也做一个函数映射
        :return: dict
        """
        data = defaultdict(list)
        for x in ls:
            k = key(x)
            if ykey:
                x = ykey(x)
            data[k].append(x)
        return cls(data)


class PathGroups(Groups):
    """ 按文件名（不含后缀）分组的相关功能 """

    @classmethod
    def groupby(cls, files, key=lambda x: os.path.splitext(str(x))[0], ykey=lambda y: y.suffix[1:]):
        """
        :param files: 用Dir.select选中的文件、目录清单
        :param key: D:/home/datasets/textGroup/SROIE2019+/data/task3_testcrop/images/X00016469670
        :param ykey: ['jpg', 'json', 'txt']
        :return: dict
            1, task3_testcrop/images/X00016469670：['jpg', 'json', 'txt']
            2, task3_testcrop/images/X00016469671：['jpg', 'json', 'txt']
            3, task3_testcrop/images/X51005200931：['jpg', 'json', 'txt']
        """
        return super().groupby(files, key, ykey)

    def select_group(self, judge):
        """ 对于某一组，只要该组有满足judge的元素则保留该组 """
        data = {k: v for k, v in self.data.items() if judge(k, v)}
        return type(self)(data)

    def select_group_which_hassuffix(self, pattern, flags=re.IGNORECASE):
        def judge(k, values):
            for v in values:
                m = re.match(pattern, v, flags=flags)
                if m and len(m.group()) == len(v):
                    # 不仅要match满足，还需要整串匹配，比如jpg就必须是jpg，不能是jpga
                    return True
            return False
        return self.select_group(judge)

    def select_group_which_hasimage(self, pattern=r'jpe?g|png|bmp', flags=re.IGNORECASE):
        """ 只保留含有图片格式的分组数据 """
        return self.select_group_which_hassuffix(pattern, flags)

    def find_files(self, name, *, count=-1):
        """ 找指定后缀的文件

        :param name: 支持 '1.jpg', 'a/1.jpg' 等格式
        :param count: 返回匹配数量上限，-1表示返回所有匹配项
        :return: 找到第一个匹配项后返回
            找的有就返回File对象
            找没有就返回None

        注意这个功能是大小写敏感的，如果出现大小写不匹配
            要么改文件名本身，要么改name的格式

        TODO 如果有大量的检索，这样每次遍历会很慢，可能要考虑构建后缀树来处理
        """
        ls = []
        stem, ext = os.path.splitext(name)
        stem = '/' + stem.replace('\\', '/')
        ext = ext[1:]
        for k, v in self.data.items():
            if k.endswith(stem) and ext in v:
                ls.append(File(k, suffix=ext))
                if len(ls) >= count:
                    break
        return ls
