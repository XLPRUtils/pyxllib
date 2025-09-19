#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/02 11:09

from collections import defaultdict, Counter
import copy
import re
import sys

from pyxllib.prog.lazyimport import lazy_import

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import('pandas')

try:
    from more_itertools import unique_everseen
except ModuleNotFoundError:
    unique_everseen = lazy_import('from more_itertools import unique_everseen')

from pyxllib.prog.newbie import typename
from pyxllib.algo.pupil import natural_sort_key
from pyxllib.text.pupil import shorten, east_asian_shorten


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


class JsonStructParser:
    """ 类json数据格式的结构解析

    【名称定义】
    item: 一条类json的数据条目
    path: 用类路径的格式，表达item中某个数值的索引。例如
        /a/b/3/c: 相当于 item['a']['b'][3]['c']
        有一些特殊的path，例如容器类会以/结尾： /a/
        以及一般会带上数值的类型标记，区分度更精确：/a/=dict
    pathx: 泛指下述中某种格式
        pathlist: list, 一条item对应的扁平化的路径
        pathstr/struct: paths拼接成一个str
        pathdict: paths重新组装成一个dict字典(未实装，太难写，性价比也低)
    """

    default_cfg = {'include_container': True,  # 包含容器（dict、list）的路径
                   'value_type': True,  # 是否带上后缀：数值的类型
                   # 可以输入一个自定义的路径合并函数 path,type=merge_path(path,type)。
                   # 一般是字典中出现不断变化的数值id，格式不统一，使用一定的规则，可以将path几种相近的冗余形式合并。
                   # 也可以设True，默认会将数值类统一为0。
                   'merge_path': False,
                   }

    @classmethod
    def _get_item_path_types(cls, item, prefix=''):
        """
        :param item: 类json结构的数据，可以含有类型： dict, list(tuple), int, str, bool, None
            结点类型
                其中 dict、list称为 container 容器类型
                其他int、str称为数值类型
            结构
                item 可以看成一个树形结构
                其中数值类型可以视为叶子结点，其他容器类是非叶子结点

        瑕疵
        1、如果key本身带有"/"，会导致混乱
        2、list的下标转为0123，和字符串类型的key会混淆，和普通的字典key也会混淆
        """
        path_types = []
        if isinstance(item, dict):
            path_types.append([prefix + '/', 'dict'])
            for k in sorted(item.keys()):  # 实验表明，在这里对字典的键进行排序就行，最后总的paths不要排序，不然结构会乱
                v = item[k]
                path_types.extend(cls._get_item_path_types(v, f'{prefix}/{k}'))
        elif isinstance(item, (list, tuple)):
            path_types.append([prefix + '/', type(item).__name__])
            for k, v in enumerate(item):
                path_types.extend(cls._get_item_path_types(v, f'{prefix}/{k}'))
        else:
            path_types.append([prefix, type(item).__name__])
        return path_types

    @classmethod
    def get_item_pathlist(cls, item, prefix='', **kwargs):
        """ 获得字典的结构标识
        """
        # 1 底层数据
        cfg = copy.copy(cls.default_cfg)
        cfg.update(kwargs)
        paths = cls._get_item_path_types(item, prefix)

        # 2 配置参数
        if cfg['merge_path']:
            if callable(cfg['merge_path']):
                func = cfg['merge_path']
            else:
                def func(p, t):
                    return re.sub(r'\d+', '0', p), t

            # 保序去重
            paths = list(unique_everseen(map(lambda x: func(x[0], x[1]), paths)))

        if not cfg['include_container']:
            paths = [pt for pt in paths if (pt[0][-1] != '/')]

        if cfg['value_type']:
            paths = ['='.join(pt) for pt in paths]
        else:
            paths = [pt[0] for pt in paths]

        return paths

    @classmethod
    def get_item_pathstr(cls, item, prefix='', **kwargs):
        paths = cls.get_item_pathlist(item, prefix, **kwargs)
        return '\n'.join(paths)

    @classmethod
    def get_items_struct2cnt(cls, items, **kwargs):
        # 1 统计每种结构出现的次数
        struct2cnt = Counter()
        for item in items:
            pathstr = cls.get_item_pathstr(item, **kwargs)
            struct2cnt[pathstr] += 1
        # 2 按照从多到少排序
        struct2cnt = Counter(dict(sorted(struct2cnt.items(), key=lambda item: -item[1])))
        return struct2cnt

    @classmethod
    def get_items_structdf(cls, items, **kwargs):
        """ 分析一组题目里，出现了多少种不同的json结构 """
        # 1 获取原始数据，初始化
        struct2cnt = cls.get_items_struct2cnt(items, **kwargs)
        m = len(struct2cnt)

        # 2 path2cnt
        path2cnt = Counter()
        for struct in struct2cnt.keys():
            path2cnt.update({path: struct2cnt[struct] for path in struct.splitlines()})
        paths = sorted(path2cnt.keys(), key=lambda path: re.split(r'/=', path))
        path2cnt = {path: path2cnt[path] for path in paths}

        # 3 生成统计表
        ls = []
        columns = ['path', 'total'] + [f'struct{i}' for i in range(1, m + 1)]
        for path, cnt in path2cnt.items():
            row = [path, cnt]
            for struct, cnt in struct2cnt.items():
                t = cnt if path in struct else 0
                row.append(t)
            ls.append(row)

        df = pd.DataFrame.from_records(ls, columns=columns)
        return df

    @classmethod
    def get_itemgroups_structdf(cls, itemgroups, **kwargs):
        """ 分析不同套数据间的json结构区别

        这里为了减少冗余开发，直接复用get_items_structdf
            虽然会造成一些冗余功能，
        """
        # 1 统计所有paths出现情况
        n = len(itemgroups)
        d = dict()
        for i, gs in enumerate(itemgroups):
            for x in gs:
                paths = cls.get_item_pathlist(x, **kwargs)
                for p in paths:
                    if p not in d:
                        d[p] = [0] * n
                    d[p][i] += 1

        # 排序
        paths = sorted(d.keys(), key=lambda path: re.split(r'/=', path))

        # 2 统计表
        ls = []
        columns = ['path', 'total'] + [f'group{i}' for i in range(1, n + 1)]
        for path in paths:
            vals = d[path]
            row = [path, sum(vals)] + vals
            ls.append(row)

        df = pd.DataFrame.from_records(ls, columns=columns)
        return df
