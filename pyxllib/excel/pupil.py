#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 23:04

from collections import defaultdict
import pandas as pd


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
