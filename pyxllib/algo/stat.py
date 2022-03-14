#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 23:04

""" 统计方面的功能

主要是pandas、表格运算
"""

from collections import defaultdict
import pandas as pd

from pyxllib.debug.pupil import dprint


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


def dataframes_to_excel(outfile, dataframes):
    """ 将多个dataframe表格写入一个excel文件

    >> dataframes_to_excel('test.xlsx', {'images': df1, 'annotations': df2})
    """
    with pd.ExcelWriter(str(outfile)) as writer:
        # 标题比正文11号大1号，并且蓝底白字，左对齐，上下垂直居中
        # head_format = writer.book.add_format({'font_size': 12, 'font_color': 'blue',
        #                                       'align': 'left', 'valign': 'vcenter'})
        for k, v in dataframes.items():
            # 设置首行冻结
            v.to_excel(writer, sheet_name=k, freeze_panes=(1, 0))
            # 首行首列特殊标记  （这个不能随便加，对group、mul index、含名称index等场合不适用）
            # writer.sheets[k].write('A1', '_order', head_format)
            # 特殊标记第一行的格式
            # writer.sheets[k].set_row(0, cell_format=head_format)  # 这个不知道为什么不起作用~~


def xlpivot(df, index=None, columns=None, values=None):
    """ 对pandas进行封装的数据透视表功能

    :param df: 数据表
    :param index: 行划分方式
    :param columns: 列划分方式
    :param values: 显示的值
        Callable[items, value]：输出一个函数
    :return: 数据透视表的表格
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
