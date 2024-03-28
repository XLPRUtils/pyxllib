#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/03/28

"""
处理数据文件常用的拆分逻辑

半定制化的功能组件
"""

from collections import defaultdict, Counter
import datetime
import re

from pyxllib.prog.pupil import check_counter, tprint
from pyxllib.file.specialist import XlPath, refinepath


def get_timestamp(fmt='%Y%m%d%H%M%S'):
    return datetime.datetime.now().strftime(fmt)


def process_file_base(file_func, file, correct_tag, error_tag,
                      *,
                      correct_dir=None, error_dir=None,
                      correct_file_name=None, error_file_name=None,
                      split_group_func=None,
                      reset=False):
    """ 处理单个文件数据，默认将两套运行结果放在同目录的两个文件中

    :param file_func: 对于自定义的处理函数
        返回支持1个参数，2个参数。1个参数时，默认没有错误的记录数据。
        对于数据，支持list[str]或list[dict]等多种格式
    :param split_group_func: 对正确的数据，按照某种规则进一步分组
    """

    # 0 工具函数
    def remove_old_file(new_file):
        # 如果目录下有同名函数，仅最后的时间戳不同的旧文件，删除
        stem = re.sub(r'_\d{14}$', '', new_file.stem)
        fmt_str = fr'{stem}_\d{{14}}{new_file.suffix}'
        for f in new_file.parent.glob(f'{stem}_*{new_file.suffix}'):
            if re.match(fmt_str, f.name):
                f.delete()

    def write_file(_dir, name, tag, data):
        if name is None:
            suffix = file.suffix if isinstance(data[0], str) else '.jsonl'
            name = f'{file.stem}_{tag}{len(data)}_{get_timestamp()}{suffix}'
        _file = _dir / name
        _file.parent.mkdir(exist_ok=True, parents=True)
        if reset:  # 找到旧后缀的文件，删除，排除时间戳差异的文件检索
            remove_old_file(_file)

        if isinstance(data[0], str):
            _file.write_text('\n'.join(data))
        else:
            _file.write_jsonl(data)

    # 1 处理数据
    file = XlPath(file)
    res = file_func(file)
    if isinstance(res, tuple) and len(res) == 2:
        correct_data, error_data = res
    else:
        correct_data = res
        error_data = None

    # 2 计算标准文件名
    if correct_data:
        correct_dir = file.parent if correct_dir is None else XlPath(correct_dir)
        suffix = file.suffix if isinstance(correct_data[0], str) else '.jsonl'
        if split_group_func:  # 使用split_group_func的时候，split_group_func不生效
            group_data = defaultdict(list)
            for x in correct_data:
                group_data[split_group_func(x)].append(x)

            for k, correct_data in group_data.items():
                correct_file_name2 = (f'{file.stem}_{refinepath(k)}_'
                                      f'{correct_tag}{len(correct_data)}_{get_timestamp()}{suffix}')
                write_file(correct_dir, correct_file_name2, correct_tag, correct_data)
        else:
            write_file(correct_dir, correct_file_name, correct_tag, correct_data)
    if error_data:
        error_dir = file.parent if error_dir is None else XlPath(error_dir)
        write_file(error_dir, error_file_name, error_tag, error_data)


def process_dir_base(file_func, dir_path, correct_tag, error_tag,
                     *, pattern='*', correct_dir=None, error_dir=None, reset=False,
                     **kwargs):
    """ 处理一个目录下的所有文件
    """
    dir_path = XlPath(dir_path)
    correct_dir = XlPath(correct_dir) if correct_dir else (dir_path.parent / f'{dir_path.name}_{correct_tag}')
    error_dir = XlPath(error_dir) if error_dir else (dir_path.parent / f'{dir_path.name}_{error_tag}')

    files = list(dir_path.rglob(pattern))
    for idx, file in enumerate(files, start=1):
        tprint(f'处理第{idx}/{len(files)}个文件: {file.name} ==> {correct_tag}')
        process_file_base(file_func, file, correct_tag, error_tag,
                          correct_dir=correct_dir, error_dir=error_dir, reset=reset,
                          **kwargs)


def process_path(file_func, tag, path, **kwargs):
    """ 对单文件，或者目录处理的封装 """
    path = XlPath(path)

    if isinstance(file_func, str):  # 用命令行等接口的时候，输入可能是字符串名
        file_func = globals()[file_func]

    # 有些特殊参数，是预设给func使用的
    func_args = {}
    if 'remove_repeat_mode' in kwargs:
        func_args['remove_repeat_mode'] = kwargs.pop('remove_repeat_mode')

    if func_args:
        file_func2 = lambda x: file_func(x, **func_args)
    else:
        file_func2 = file_func

    if path.is_file():
        process_file_base(file_func2, path, tag, f'{tag}error', **kwargs)
    elif path.is_dir():
        process_dir_base(file_func2, path, tag, f'{tag}error', **kwargs)


class Analyzer:
    """ 分析器，一般用来写一些不影响主体功能，协助调试，分析数据特征的功能 """

    def __init__(self):
        self.counters = defaultdict(Counter)

    def count_once(self, tag, key):
        # tag相当于需要分组统计的名称
        self.counters[tag][key] += 1

    def check_counter(self):
        # 检查各种特征出现的次数
        for name, ct in self.counters.items():
            print(f'【{name}】')
            check_counter(ct)


def head_data(infile, num=1000, file_size=50):
    """ 获取infile头部部分数据

    :param file_size: 除了数量限制外，同时限制文件大小不超过file_size MB
        这里不是精确解法，而是简化大概的写法
    """
    res = []
    total_size = 0
    for x in XlPath(infile).yield_line(end=num):
        sz = len(x)
        total_size += sz
        if total_size > file_size * 1024 * 1024:
            break
        res.append(x)

    return res


def remove_repeat_base(infile,
                       get_etags_func,
                       exists_etags=set(),  # trick: 这里需要全局记录已出现过的etag
                       ):
    """ 去除重复数据，基础函数

    :param get_etags_func: 每道题目的etag，用于判断是否重复
        可能用于判断的etags数量不止一个，可以返回多个
        返回一个值的版本也可以，会自动转换为list
    """
    data = XlPath(infile).read_jsonl()
    src_len = len(data)

    data2 = []
    for x in data:
        etags = get_etags_func(x)
        if isinstance(etags, str):
            etags = [etags]
        if any(etag in exists_etags for etag in etags):
            continue
        data2.append(x)
        exists_etags.update(set(etags))

    new_len = len(data2)
    print(f'去重后剩余 {new_len}/{src_len} ≈ {new_len / src_len:.2%} 的数据')
    return data2
