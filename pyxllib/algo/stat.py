#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 23:04

""" 统计方面的功能

主要是pandas、表格运算
"""

import sys
from collections import defaultdict, Counter

import pandas as pd

from pyxllib.prog.pupil import dprint, typename
from pyxllib.file.specialist import XlPath

pd.options.display.unicode.east_asian_width = True  # 优化中文输出对齐问题
try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception as e:
    pass


def treetable(childreds, parents, arg3=None, nodename_colname=None):
    """ 输入childres子结点id列表，和parents父结点id列表

    两个列表长度必须相等
    文档：http://note.youdao.com/noteshare?id=126200f45d301fcb4364d06a0cae8376

    有两种调用形式
    >> treetable(childreds, parents)  --> DataFrame  （新建df）
    >> treetable(df, child_colname, parent_colname)  --> DataFrame （修改后的df）

    返回一个二维列表
        新的childreds （末尾可能回加虚结点）
        新的parents
        函数会计算每一行childred对应的树排序后的排序编号order
        以及每个节点深度depth

    >> ls1 = [6, 2, 4, 5, 3], ls2 = [7, 1, 2, 2, 1], treetable(ls1, ls2)
          child_id   parent_id   depth     tree_order    tree_struct
        5        7     root        1           1         = = 7
        0        6        7        2           2         = = = = 6
        6        1     root        1           3         = = 1
        1        2        1        2           4         = = = = 2
        2        4        2        3           5         = = = = = = 4
        3        5        2        3           6         = = = = = = 5
        4        3        1        2           7         = = = = 3
    """
    # 0 参数预处理
    if isinstance(childreds, pd.DataFrame):
        df = childreds
        child_colname = parents
        parent_colname = arg3
        if not arg3: raise TypeError
        childreds = df[child_colname].tolist()
        parents = df[parent_colname].tolist()
    else:
        df = None

    # 1 建立root根节点，确保除了root其他结点都存在记录
    lefts = set(parents) - set(childreds)  # parents列中没有在childreds出现的结点
    cs, ps = list(childreds), list(parents)

    if len(lefts) == 0:
        # b_left为空一定有环，b_left不为空也不一定是正常的树
        raise ValueError('有环，不是树结构')
    elif len(lefts) == 1:  # 只有一个未出现的结点，那么它既是根节点
        root = list(lefts)[0]
    else:  # 多个父结点没有记录，则对这些父结点统一加一个root父结点
        root = 'root'
        allnode = set(parents) | set(childreds)  # 所有结点集合
        while root in allnode: root += '-'  # 一直在末尾加'-'，直到这个结点是输入里未出现的
        # 添加结点
        lefts = list(lefts)
        lefts.sort(key=lambda x: parents.index(x))
        for t in lefts:
            cs.append(t)
            ps.append(root)

    n = len(cs)
    depth, tree_order, len_childs = [-1] * n, [-1] * n, [0] * n

    # 2 构造父结点-孩子结点的字典dd
    dd = defaultdict(list)
    for i in range(n): dd[ps[i]] += [i]

    # 3 dfs
    cnt = 1

    def dfs(node, d):
        """找node的所有子结点"""
        nonlocal cnt
        for i in dd.get(node, []):
            tree_order[i], depth[i], len_childs[i] = cnt, d, len(dd[cs[i]])
            cnt += 1
            dfs(cs[i], d + 1)

    dfs(root, 1)

    # 4 输出格式
    tree_struct = list(map(lambda i: f"{'_ _ ' * depth[i]}{cs[i]}" + (f'[{len_childs[i]}]' if len_childs[i] else ''),
                           range(n)))

    if df is None:
        ls = list(zip(cs, ps, depth, tree_order, len_childs, tree_struct))
        df = pd.DataFrame.from_records(ls, columns=('child_id', 'parent_id',
                                                    'depth', 'tree_order', 'len_childs', 'tree_struct'))
    else:
        k = len(df)
        df = df.append(pd.DataFrame({child_colname: cs[k:], parent_colname: ps[k:]}), sort=False, ignore_index=True)
        if nodename_colname:
            tree_struct = list(
                map(lambda i: f"{'_ _ ' * depth[i]}{cs[i]} {df.iloc[i][nodename_colname]}"
                              + (f'[{len_childs[i]}]' if len_childs[i] else ''), range(n)))
        df['depth'], df['tree_order'], df['len_childs'], df['tree_struct'] = depth, tree_order, len_childs, tree_struct
    df.sort_values('tree_order', inplace=True)  # 注意有时候可能不能排序，要维持输入时候的顺序
    return df


def treetable_flatten(df, *, reverse=False, childid_colname='id', parentid_colname='parent_id', format_colname=None):
    """ 获得知识树横向展开表：列为depth-3, depth-2, depth-1，表示倒数第3级、倒数第2级、倒数第1级

    :param df: DataFrame数据
    :param reverse:
        False，正常地罗列depth1、depth2、depth3...等结点信息
        True，反向列举所属层级，即显示倒数第1层parent1，然后是倒数第2层parent2...
    :param childid_colname: 孩子结点列
    :param parentid_colname: 父结点列
    :param format_colname: 显示的数值
        None，默认采用 childid_colname 的值
        str，某一列的名称，采用那一列的值（可以实现设置好格式）
    :return:
    """
    # 1 构造辅助数组
    if format_colname is None: format_colname = parentid_colname
    parentid = dict()  # parentid[k] = v， 存储结点k对应的父结点v
    nodeval = dict()  # nodeval[k] = v，  存储结点k需要显示的数值情况
    if len(df[df.index.duplicated()]):
        dprint(len(set(df.index)), len(df.index))  # 有重复index
        raise ValueError

    for idx, row in df.iterrows():
        parentid[row[childid_colname]] = row[parentid_colname]
        nodeval[row[childid_colname]] = str(row[format_colname])

    # 2 每个结点往上遍历出所有父结点
    parents = []
    for idx, row in df.iterrows():
        ps = [nodeval[row[childid_colname]]]  # 包含结点自身的所有父结点名称
        p = row[parentid_colname]
        while p in parentid:
            ps.append(nodeval[p])
            p = parentid[p]
        parents.append(ps)
    num_depth = max(map(len, parents), default=0)

    # 3 这里可以灵活调整最终要显示的格式效果
    df['parents'] = parents
    if reverse:
        for j in range(num_depth, 0, -1): df[f'depth-{j}'] = ''
        for idx, row in df.iterrows():
            for j in range(1, len(row.parents) + 1):
                df.loc[idx, f'depth-{j}'] = row.parents[j - 1]
    else:
        for j in range(num_depth): df[f'depth{j}'] = ''
        for idx, row in df.iterrows():
            for j in range(len(row.parents)):
                df.loc[idx, f'depth{j}'] = row.parents[-j - 1]
    df.drop('parents', axis=1, inplace=True)
    return df


def write_dataframes_to_excel(outfile, dataframes, order_mode='序号'):
    """ 将多个DataFrame表格写入一个Excel文件，并添加序号列

    :param str outfile: 输出的Excel文件路径
    :param dict dataframes: 包含要保存的DataFrame的字典，键为sheet名，值为DataFrame
    :param str order_mode: 序号模式，可选值为 'default' 或 '序号'，默认为 '序号'

    >> write_dataframes_to_excel('test.xlsx', {'images': df1, 'annotations': df2})

    # TODO 存成表格后，可以使用openpyxl等库再打开表格精修

    实现上，尽可能在一些常见结构上，进行一些格式美化。但对费常规结构，就保留df默认排版效果，不做特殊处理。
    """
    with pd.ExcelWriter(str(outfile), engine='xlsxwriter') as writer:
        head_format = writer.book.add_format({'font_size': 12, 'font_color': 'blue',
                                              'align': 'left', 'valign': 'vcenter'})
        for sheet_name, df in dataframes.items():
            if df.index.nlevels == 1 and df.columns.nlevels == 1:
                if order_mode == '序号':
                    # 写入带有序号列的数据表格
                    if '序号' not in df.columns:
                        df = df.copy()
                        df.insert(0, '序号', range(1, len(df) + 1))
                else:
                    df = df.reset_index()
                    df.columns = ['_index'] + list(df.columns[1:])
                df.to_excel(writer, sheet_name=sheet_name, freeze_panes=(1, 0), index=False)
            else:
                # 写入普通的数据表格
                df.to_excel(writer, sheet_name=sheet_name, freeze_panes=(1, df.index.nlevels))

            # 设置表头格式
            if df.columns.nlevels == 1:
                start = df.index.nlevels
                if start == 1:
                    start = 0
                for col_num, value in enumerate(df.columns, start=start):
                    writer.sheets[sheet_name].write(0, col_num, value, head_format)


def read_dataframes_from_excel(infile):
    """ 从Excel文件读取多个DataFrame表格

    :param str infile: Excel文件路径
    :return: 包含读取的DataFrame的字典，键为工作表名，值为DataFrame
    :rtype: dict

    注意这个函数不太适用于与读取多级index和多级columns的情况，建议遇到这种情况，手动读取，
        read_excel可以设置header=[0,1]、index=[0,1,2]的形式来定制表头所在位置。
    """
    dataframes = {}
    with pd.ExcelFile(infile) as xls:
        sheet_names = xls.sheet_names
        for sheet_name in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if '_index' in df.columns:
                df = df.drop('_index', axis=1)
            dataframes[sheet_name] = df
    return dataframes


def update_dataframes_to_excel(outfile, dataframes, order_mode='序号'):
    """ 更新xlsx文件中的sheets数据 """
    outfile = XlPath(outfile)
    if outfile.is_file():
        data = read_dataframes_from_excel(outfile)
    else:
        data = {}
    data.update(dataframes)
    write_dataframes_to_excel(outfile, data, order_mode)


def xlpivot(df, index=None, columns=None, values=None):
    """ 对pandas进行封装的数据透视表功能

    :param df: 数据表
    :param index: 行划分方式
    :param columns: 列划分方式
    :param values: 显示的值
        Callable[items, value]：输出一个函数
    :return: 数据透视表的表格

    使用示例：
    def func(items):  # 输入匹配的多行数据
        x = items.iloc[0]
        return f'{x["precision"]:.0f}，{x["recall"]:.0f}，{x["hmean"]:.2f}，{x["fps"]}'  # 返回显示的值
    >> df2 = xlpivot(df, ['model_type'], ['dataset', 'total_frame'], {'precision，recall，hmean，fps': func})
    """

    # 1 将分组的格式标准化
    def reset_groups(keys):
        if isinstance(keys, (list, tuple)):
            return list(keys)
        elif keys:
            return [keys]
        else:
            return []

    index_, columns_ = reset_groups(index), reset_groups(columns)

    # 2 目标值的格式标准化
    if callable(values):
        values_ = {'values': values}
    elif isinstance(values, dict):
        values_ = values
    else:
        raise TypeError

    # 3 分组
    keys = index_ + columns_
    dfgp = df.groupby(keys)
    data = defaultdict(list)
    for ks, items in dfgp:
        # 要存储分组（keys）相关的值
        if len(keys) == 1:
            data[keys[0]].append(ks)
        else:
            for i, k in enumerate(keys):
                data[k].append(ks[i])
        # 再存储生成的值
        for k, func in values_.items():
            data[k].append(func(items))
    df2 = pd.DataFrame.from_dict(data)

    # 4 可视化表格
    if index and columns:
        view_table = df2.pivot(index=index, columns=columns, values=list(values_.keys()))
    elif index:
        view_table = df2.set_index(index_)
    else:  # 只有columns，没有index
        view_table = df2.set_index(index_).T
    return view_table


def count_key_combinations(df, col_names, count_col_name='count'):
    """ 统计列出的几个列名，各种组合出现的次数

    :param df:
    :param col_names: ['a', 'b', 'c']
    :param count_col_name: 新增的统计出现次数的列名，默认count
    :return: 新的次数统计的df表格

    这个功能跟 SqlCodeGenerator 的 keys_count、one2many很像，是可以代替这两个功能的
    """
    from collections import Counter

    # 0 参数处理
    if isinstance(col_names, str):
        col_names = [col_names]

    # 1 统计每种组合出现的次数
    cols = [df[name] for name in col_names]
    ct = Counter(tuple(zip(*cols)))

    # 2 生成新的df的统计表
    ls = []
    for k, v in ct.most_common():
        ls.append([*k, v])
    df2 = pd.DataFrame.from_records(ls, columns=list(col_names) + [count_col_name])
    return df2


def pareto_accumulate(weights, accuracy=0.01, *, print_mode=False, value_unit_type='K'):
    """ 帕累托累计

    可以用来分析主要出现的权重、频次
    二八法则，往往20%的数据，就能解决80%的问题

    :param weights: 一组权重数据
    :param accuracy: 累计精度，当统计到末尾时，可能有大量权重过小的数值
        此时不频繁进行累计权重计算，而是当更新权重累计达到accuracy，才会更新一个记录点
        注意这是全量数据综合的百分比，所以最小更新量就是1%
    :param print_mode: 是否直接展示可视化结果
    :return: [(累计数值数量, ≥当前阈值, 累计权重), ...]

    >>> pareto_accumulate([1, 2, 3, 9, 8, 7, 4, 6, 5])
    [(1, 9, 9), (2, 8, 17), (3, 7, 24), (4, 6, 30), (5, 5, 35), (6, 4, 39), (7, 3, 42), (8, 2, 44), (9, 1, 45)]
    >>> pareto_accumulate([1, 2, 3, 9, 8, 7, 4, 6, 5], 0.1)
    [(1, 9, 9), (2, 8, 17), (3, 7, 24), (4, 6, 30), (5, 5, 35), (7, 3, 42), (9, 1, 45)]
    """
    # 1 基础数据计算
    points = []
    weights = sorted(weights, reverse=True)

    total = sum(weights)
    accuracy = total * accuracy

    acc = 0
    delta = 0
    for i, w in enumerate(weights, start=1):
        acc += w
        delta += w
        if delta >= accuracy:
            points.append((i, w, acc))
            delta = 0
    if delta:
        points.append((len(weights), weights[-1], acc))

    # 2 结果展示
    def fmt(p):
        from pyxllib.prog.newbie import human_readable_number
        ls = [f'{human_readable_number(p[0], "万")}条≥{human_readable_number(p[1])}',
              f'{human_readable_number(p[2], value_unit_type)}({p[2] / total_size:.0%})']
        return '，'.join(map(str, ls))

    total_size = points[-1][2]
    labels = [fmt(p) for p in points]

    pts = [[p[0], p[2]] for p in points]

    if print_mode:
        if sys.platform == 'win32':
            from pyxllib.data.echarts import Line
            from pyxllib.prog.specialist import browser

            x = Line()
            x.add_series('帕累托累积权重', pts, labels=labels, label={'position': 'right'})
            browser(x)
        else:
            print(*labels, sep='\n')

    return pts, labels


class XlDataFrame(pd.DataFrame):
    def check_dtypes(self):
        """ 检查数据类型
        第1列是列名，第2列是原本dtypes显示的类型看，第3列是我扩展的统计的实际数据类型
        """
        d = self.dtypes
        ls = [[k, d[k], Counter([typename(x) for x in v]).most_common()] for k, v in self.iteritems()]
        df = pd.DataFrame.from_records(ls, columns=['name', 'type', 'detail'])
        return df


class ModifiableRow:
    def __init__(self, df, index):
        self.df = df
        self.index = index

    def __getitem__(self, item):
        return self.df.at[self.index, item]

    def __setitem__(self, key, value):
        self.df.at[self.index, key] = value


def print_full_dataframe(df):
    """
    临时设置以完整显示DataFrame的内容

    :param pd.DataFrame df: 需要完整显示的DataFrame
    """
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 1000,
                           'display.max_colwidth', None):
        print(df)

    pd.options('display.max_rows', 60)


def custom_fillna(df, default_fill_value='', numeric_fill_value=None, specific_fill=None):
    """ 使用更多灵活性填充DataFrame中的NaN值。

    :param pandas.DataFrame df: 需要处理的DataFrame。
    :param str default_fill_value: 非数值列中NaN的默认填充值。
    :param numeric_fill_value: 数值列中NaN的填充值，如果不指定，则默认为None。
    :param dict specific_fill: 指定列名及其NaN的填充值，如果不指定，则默认为None。
    :return: 已根据指定标准填充NaN值的pandas.DataFrame。

    >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [None, 'x', 'y'], 'C': [None, None, None]})
    >>> custom_fillna(df, 'filled', 0, {'C': 'special'})
    """
    for column in df.columns:
        # 检查列是否在specific_fill中指定；如果是，则使用指定的值填充。
        if specific_fill and column in specific_fill:
            df[column] = df[column].fillna(specific_fill[column])
        # 如果列是数值型且指定了numeric_fill_value，则使用numeric_fill_value填充。
        elif numeric_fill_value is not None and pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(numeric_fill_value)
        # 否则，对非数值列使用default_fill_value进行填充。
        elif pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].fillna(default_fill_value)
        # 可以在这里添加更多条件，以处理其他数据类型，如datetime。
    return df


def dataframe_to_list(df):
    """将DataFrame转换为列表结构，第一行是表头，其余是数据"""
    # 获取表头（列名）作为第一个列表元素
    headers = df.columns.tolist()

    # 获取数据行，每一行作为一个列表，然后将所有这些列表收集到一个大列表中
    data_rows = df.values.tolist()

    # 将表头和数据行合并成最终的列表
    result_list = [headers] + data_rows

    return result_list
