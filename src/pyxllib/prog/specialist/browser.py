#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 22:43

import builtins
import sys

from loguru import logger

from pyxllib.prog.lazyimport import lazy_import

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import("pandas")


from pyxllib.prog.browser import browser, view_files
from pyxllib.prog.viewobj import view_obj
from pyxllib.prog.specialist.common import (
    NestedDict,
    KeyValuesCounter,
)
from pyxllib.file.specialist.dirlib import File, Dir


# -----------------------------------------------------------------------------
# 1. 其他工具
# -----------------------------------------------------------------------------


def browse_json(f):
    """可视化一个json文件结构

    :param f: json文件路径
    """
    data = File(f).read()
    # 使用NestedDict.to_html_table转成html的嵌套表格代码，存储到临时文件夹
    htmlfile = File("chrome_json.html", root=Dir.TEMP).write(NestedDict.to_html_table(data))
    # 展示html文件内容
    browser(htmlfile)


def browse_jsons_kv(fd, files="**/*.json", encoding=None, max_items=10, max_value_length=100):
    """demo_keyvaluescounter，查看目录下json数据的键值对信息

    :param fd: 目录
    :param files: 匹配的文件格式
    :param encoding: 文件编码
    :param max_items: 项目显示上限，有些数据项目太多了，要精简下
            设为假值则不设上限
    :param max_value_length: 添加的值，进行截断，防止有些值太长
    :return: None
    """
    kvc = KeyValuesCounter()
    d = Dir(fd)
    for p in d.select_files(files):
        data = p.read(encoding=encoding, mode=".json")
        kvc.add(data, max_value_length=max_value_length)
    p = File("demo_keyvaluescounter.html", Dir.TEMP)
    p.write(kvc.to_html_table(max_items=max_items), if_exists="replace")
    browser(p.to_str())


def check_repeat_filenames(dir, key="stem", link=True):
    """检查目录下文件结构情况的功能函数

    https://www.yuque.com/xlpr/pyxllib/check_repeat_filenames

    :param dir: 目录Dir类型，也可以输入路径，如果没有files成员，则默认会获取所有子文件
    :param key: 以什么作为行分组的key名称，基本上都是用'stem'，偶尔可能用'name'
        遇到要忽略 -eps-to-pdf.pdf 这种后缀的，也可以自定义处理规则
        例如 key=lambda p: re.sub(r'-eps-to-pdf', '', p.stem).lower()
    :param bool link: 默认True会生成文件超链接
    :return pd.DataFrame: 一个df表格，行按照key的规则分组，列默认按suffix扩展名分组
    """
    # 1 智能解析dir参数
    if not isinstance(dir, Dir):
        dir = Dir(dir)
    if not dir.subs:
        dir = dir.select("**/*", type_="file")

    # 2 辅助函数，智能解析key参数
    if isinstance(key, str):

        def extract_key(p):
            return getattr(p, key).lower()
    elif callable(key):
        extract_key = key
    else:
        raise TypeError

    # 3 制作df表格数据
    columns = ["key", "suffix", "filename"]
    li = []
    for f in dir.subs:
        p = File(f)
        li.append([extract_key(p), p.suffix.lower(), f])
    df = pd.DataFrame.from_records(li, columns=columns)

    # 4 分组
    def joinfile(files):
        if len(files):
            if link:
                return ", ".join([f"<a href='{dir / f}' target='_blank'>{f}</a>" for f in files])
            else:
                return ", ".join(files)
        else:
            return ""

    groups = df.groupby(["key", "suffix"]).agg({"filename": joinfile})
    groups.reset_index(inplace=True)
    view_table = groups.pivot(index="key", columns="suffix", values="filename")
    view_table.fillna("", inplace=True)

    # 5 判断每个key的文件总数
    count_df = df.groupby("key").agg({"filename": "count"})
    view_table = pd.concat([view_table, count_df], axis=1)
    view_table.rename({"filename": "count"}, axis=1, inplace=True)

    browser(view_table, to_html_args={"escape": not link})
    return df


if __name__ == "__main__":
    import fire

    fire.Fire()
