#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 11:00

import copy
import json
import random
import re
import sys
from collections import defaultdict, Counter
from typing import Dict, Any, Type, TypeVar, Union

from pyxllib.prog.lazyimport import lazy_import

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import('pandas')

try:
    from more_itertools import unique_everseen
except ModuleNotFoundError:
    unique_everseen = lazy_import('from more_itertools import unique_everseen')

try:
    from pydantic import BaseModel, ValidationError
except ModuleNotFoundError:
    BaseModel = lazy_import('from pydantic import BaseModel')

from pyxllib.prog.basic import typename
from pyxllib.algo.sort import natural_sort_key
from pyxllib.text.base import shorten
from pyxllib.text.format import east_asian_shorten


def convert_to_json_compatible(d, custom_converter=None):
    """ 递归地将字典等类型转换为JSON兼容格式。对于非标准JSON类型，使用自定义转换器或默认转换为字符串。

    :param d: 要转换的字典或列表。
    :param custom_converter: 自定义转换函数，用于处理非标准JSON类型的值。
    :return: 转换后的字典或列表。

    todo 这个函数想法是好的，但总感觉精确性中，总容易有些问题的，需要更多的考察测试
    """
    if custom_converter is None:
        custom_converter = str

    if isinstance(d, dict):  # defaultdict呢？
        return {k: convert_to_json_compatible(v, custom_converter) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_to_json_compatible(v, custom_converter) for v in d]
    elif isinstance(d, (int, float, str, bool)) or d is None:
        return d
    else:
        return custom_converter(d)


class DictTool:
    @classmethod
    def json_loads(cls, label, default=None):
        """ 尝试从一段字符串解析为字典

        :param default: 如果不是字典时的处理策略
            None，不作任何处理
            str，将原label作为defualt这个键的值来存储
        :return: s为非字典结构时返回空字典

        >>> DictTool.json_loads('123', 'text')
        {'text': '123'}
        >>> DictTool.json_loads('[123, 456]', 'text')
        {'text': '[123, 456]'}
        >>> DictTool.json_loads('{"a": 123}', 'text')
        {'a': 123}
        """
        labelattr = dict()
        try:
            data = json.loads(label)
            if isinstance(data, dict):
                labelattr = data
        except json.decoder.JSONDecodeError:
            pass
        if not labelattr and isinstance(default, str):
            labelattr[default] = label
        return labelattr

    @classmethod
    def or_(cls, *args):
        """ 合并到新字典

        左边字典有的key，优先取左边，右边不会覆盖。
        如果要覆盖效果，直接用 d1.update(d2)功能即可。

        :return: args[0] | args[1] | ... | args[-1].
        """
        res = {}
        cls.ior(res, *args)
        return res

    @classmethod
    def ior(cls, dict_, *args):
        """ 合并到第1个字典

        :return: dict_ |= (args[0] | args[1] | ... | args[-1]).

        220601周三15:45，默认已有对应key的话，值是不覆盖的，如果要覆盖，直接用update就行了，不需要这个接口
            所以把3.9的|=功能关掉
        """
        # if sys.version_info.major == 3 and sys.version_info.minor >= 9:
        #     for x in args:
        #         dict_ |= x
        # else:  # 旧版本py手动实现一个兼容功能
        for x in args:
            for k, v in x.items():
                # 220729周五21:21，又切换成dict_有的不做替换
                if k not in dict_:
                    dict_[k] = v
                # dict_[k] = v

    @classmethod
    def sub(cls, dict_, keys):
        """ 删除指定键值（不存在的跳过，不报错）

        inplace subtraction

        :param keys: 可以输入另一个字典，也可以输入一个列表表示要删除的键值清单

        :return: dict2 = dict_ - keys
        """
        if isinstance(keys, dict):
            keys = keys.keys()

        return {k: v for k, v in dict_.items() if k not in keys}

    @classmethod
    def isub(cls, dict_, keys):
        """ 删除指定键值（不存在的跳过，不报错）

        inplace subtraction

        keys可以输入另一个字典，也可以输入一个列表表示要删除的键值清单

        效果相当于 dict_ -= keys
        """
        if isinstance(keys, dict):
            keys = keys.keys()

        for k in keys:
            if k in dict_:
                del dict_[k]


def shuffle_dict_keys(d):
    keys = list(d.keys())
    random.shuffle(keys)
    d = {k: d[k] for k in keys}
    return d


def is_valid_identifier(name):
    """ 判断是否是合法的标识符 """
    return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name)


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
                           'display.max_columns', 20,  # 最大列数设置到20列
                           'display.width', 200,  # 最大宽度设置到200
                           *args):
        if shorten:  # applymap可以对所有的元素进行映射处理，并返回一个新的df
            if hasattr(df, 'map'):
                df = df.map(lambda x: east_asian_shorten(str(x), pd.options.display.max_colwidth))
            else:
                df = df.applymap(lambda x: east_asian_shorten(str(x), pd.options.display.max_colwidth))
        s = str(df)
    return s


def to_list(d, nsort=False):
    if isinstance(d, dict):
        items = list(d.items())
        if nsort:
            items.sort(key=lambda x: natural_sort_key(str(x[0])))
        return items
    return list(d)


def to_df(d):
    return pd.DataFrame(d)


class TypeConvert:
    @classmethod
    def dict2list(cls, d: dict, *, nsort=False):
        """ 字典转n*2的list """
        return to_list(d, nsort=nsort)

    @classmethod
    def dict2df(cls, d):
        """dict类型转DataFrame类型"""
        return to_df(d)

    @classmethod
    def list2df(cls, li):
        """list类型转DataFrame类型"""
        return to_df(li)

    @classmethod
    def try2df(cls, arg):
        """尝试将各种不同的类型转成dataframe"""
        return to_df(arg)


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
    def to_html_table(cls, data, max_items=-1, width=200):
        """ 以html表格套表格的形式，展示一个嵌套结构数据

        :param data: 数据
        :param max_items: 项目显示上限，有些数据项目太多了，要精简下
            设为 -1 则不设上限
        :param width: 基础元素显示上限
        :return:

        TODO 这个速度有点慢，怎么加速？
        """

        def tohtml(d):
            if isinstance(d, dict):
                # 字典转 DataFrame，键作为索引
                df = pd.DataFrame.from_dict(d, orient='index', columns=['Value'])
            else:
                # 列表/元组等转 DataFrame
                df = pd.DataFrame(d, columns=['Value'])

            if max_items is not None and max_items != -1 and len(df) > max_items:
                n = len(df)
                return df[:max_items].to_html(escape=False) + f'... {n}'
            else:
                return df.to_html(escape=False)

        if not cls.has_subdict(data):
            res = shorten(str(data), width=width)
        elif isinstance(data, dict):
            if isinstance(data, Counter):
                d = data
            else:
                d = dict()
                for k, v in data.items():
                    if cls.has_subdict(v):
                        v = cls.to_html_table(v, max_items=max_items, width=width)
                    else:
                        v = shorten(str(v), width=width)
                    d[k] = v
            res = tohtml(d)
        else:
            li = []
            for x in data:
                if cls.has_subdict(x):
                    li.append(cls.to_html_table(x, max_items=max_items, width=width))
                else:
                    li.append(shorten(str(x), width=width))
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

    def to_html_table(self, max_items=10, width=200):
        return NestedDict.to_html_table(self.kvs, max_items=max_items, width=width)


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


class BitMaskTool:
    """ 位掩码工具 """

    def __init__(self, names):
        self.names = names
        self.name2mask = {name: 1 << i for i, name in enumerate(names)}

    def to_mask(self, names):
        """ 将名称列表转换为掩码 """
        mask = 0
        for name in names:
            mask |= self.name2mask[name]
        return mask

    def to_names(self, mask):
        """ 将掩码转换为名称列表 """
        names = []
        for i, name in enumerate(self.names):
            if mask & (1 << i):
                names.append(name)
        return names


class XlBaseModel(BaseModel):
    """
    1. 提供一个dict函数，可以将对象转换为dict
    """

    def dict(self, **kwargs):
        return self.model_dump(**kwargs)

    @classmethod
    def parse_obj(cls, obj):
        return cls.model_validate(obj)


T = TypeVar("T", bound=BaseModel)


def resolve_params(params: Union[Dict[str, Any], T], model_cls: Type[T]) -> T:
    """
    解析参数并返回指定的 Pydantic 模型实例。

    :param params: 输入参数，可以是字典或 Pydantic 模型实例。
    :param model_cls: 目标 Pydantic 模型类。
    :return: 目标 Pydantic 模型实例。
    :raises TypeError: 如果输入参数类型不匹配。
    :raises ValidationError: 如果参数校验失败。
    """
    if isinstance(params, dict):
        # 如果是字典，直接用 parse_obj 解析
        return model_cls.model_validate(params)
    elif isinstance(params, model_cls):
        # 如果已经是目标类型的实例，直接返回
        return params
    else:
        raise TypeError(f"Expected dict or {model_cls.__name__}, got {type(params).__name__}")
