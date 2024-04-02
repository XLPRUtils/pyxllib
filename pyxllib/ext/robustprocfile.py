#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/03/28

"""
处理数据文件常用的拆分逻辑

半定制化的功能组件
"""

from pyxllib.prog.pupil import check_install_package

check_install_package('joblib')

from collections import defaultdict, Counter
import datetime
import re
import json

from tqdm import tqdm
from joblib import Parallel, delayed

from pyxllib.prog.pupil import check_counter, tprint, typename
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


class CacheJsonlFile:
    """ 流式存储文件 """

    def __init__(self, parent_dir, prefix_stem, tag, batch_size=2000):
        self.parent_dir = parent_dir
        self.prefix_stem = prefix_stem
        self.tag = tag
        self.cache_text_lines = []
        self.batch_size = batch_size

        # 缓存时的文件名
        self.cache_file = XlPath(parent_dir) / f'{prefix_stem}_{tag}.cache.jsonl'
        self.total = 0

    def append(self, data):
        for x in data:
            if isinstance(x, str):
                self.cache_text_lines.append(x)
            else:
                self.cache_text_lines.append(json.dumps(x, ensure_ascii=False))
        if len(self.cache_text_lines) >= self.batch_size:
            self.flush()

    def flush(self):
        """ 刷新，将当前缓存写入文件 """
        if self.cache_text_lines:
            if self.total == 0:  # 第一次写入时，删除旧缓存文件
                self.cache_file.delete()

            self.total += len(self.cache_text_lines)
            self.parent_dir.mkdir(exist_ok=True, parents=True)
            with open(self.cache_file, 'a', encoding='utf8') as f:
                f.write('\n'.join(self.cache_text_lines) + '\n')
            self.cache_text_lines = []

    def save_all(self):
        """ 最终存储的文件名 """
        self.flush()
        if self.cache_file.is_file():
            dst_file = self.cache_file.with_stem(f'{self.prefix_stem}_{self.tag}{self.total}_{get_timestamp()}')
            self.cache_file.rename(dst_file)


class CacheJsonlGroupFiles:
    def __init__(self, parent_dir, prefix_stem, tag, group_func=None, batch_size=2000):
        self.files = {}

        self.parent_dir = parent_dir
        self.prefix_stem = prefix_stem
        self.tag = tag
        self.group_func = group_func
        self.batch_size = batch_size

    def append(self, data):
        groups = defaultdict(list)
        if self.group_func:
            for x in data:
                groups[self.group_func(x)].append(x)
        else:
            groups[''] = data

        for k, v in groups.items():
            if k not in self.files:
                subtag = f'{k}_{self.tag}' if k not in ('', None) else self.tag
                self.files[k] = CacheJsonlFile(self.parent_dir, self.prefix_stem, subtag, self.batch_size)
            self.files[k].append(v)

    def save_all(self):
        for file in self.files.values():
            file.save_all()


def process_single_file(root,
                        infile,
                        tag,
                        row_func,
                        *,
                        cur_idx=1,
                        total=None,
                        if_exists='skip',
                        group_func=None,
                        batch_size=2000,
                        debug=False):
    # 1 缓存路径
    srcdir = XlPath(root)
    infile = XlPath.init(infile, root=srcdir)
    relpath = infile.relative_to(srcdir)

    dstdir = srcdir.parent / f'{srcdir.name}_{tag}' / relpath.parent
    errdir = srcdir.parent / f'{srcdir.name}_{tag}error' / relpath.parent

    # 2 判断是不是已处理过的文件
    # stem = re.split(r'\d+_\d{14}', relpath.stem)[0]
    stem = relpath.stem + f'_{tag}'
    dstfiles = list(dstdir.glob_files(f'{stem}*.jsonl'))
    errfiles = list(errdir.glob_files(f'{stem}*.jsonl'))

    # 需要进一步过滤
    def is_exists_old_file(f):
        if f.name.endswith('.cache.jsonl'):
            return False
        stem2 = f.name[len(stem):]
        if re.match(r'(error)?\d+_\d{14}.jsonl$', stem2):
            return True

    # cache文件不算，不用管
    check_files = [f for f in (dstfiles + errfiles) if is_exists_old_file(f)]

    if check_files:
        if if_exists == 'skip':
            return
        elif if_exists == 'overwrite':
            for f in check_files:
                f.delete()
        elif if_exists == 'error':
            raise FileExistsError(f'目标文件已存在：{check_files}')
        else:
            return

    # 3 处理数据
    if debug:  # 如果开启调试模式，则关闭分组功能
        group_func = None
    dstcgf = CacheJsonlGroupFiles(dstdir, infile.stem, tag, group_func, batch_size)
    errcgf = CacheJsonlGroupFiles(errdir, infile.stem, tag + 'error', batch_size=batch_size)
    if debug:  # 如果开启调试模式，则单独分错误文件
        errcgf = dstcgf

    for line in tqdm(infile.yield_line(), disable=total):
        # todo 出错的数据，应该添加错误信息，也存成一个新的jsonl格式
        row = row_func(line)
        if row and (not isinstance(row, dict) or row.get('status', 'ok') == 'ok'):
            # 必须要有内容，但如果内容有个status字段且不是'ok'，也不能要
            dstcgf.append([row])
        else:
            # 注意旧版存储的事line，但是新版的存储成row了
            # 但如果row确实没内容，还是兼容旧版，存储line
            errcgf.append([row or line])

    dstcgf.save_all()
    errcgf.save_all()
    if total:
        tprint(f'处理第{cur_idx}/{total}个文件：{infile.name} ==> {tag}')
    else:
        tprint(f'处理文件：{infile.name} ==> {tag}')


class StructureAnalyzer:
    @classmethod
    def item_to_json(cls, x, depth):
        """ 获得字典结构的签名

        todo 这个项目之后，可以对这个函数进一步优化精简，作为以后解析结构的一个通用工具
        """
        if depth <= 0:
            return typename(x)

        if isinstance(x, dict):
            d = {}
            keys = sorted(x.keys())
            for k in keys:
                d[k] = cls.item_to_json(x[k], depth - 1)
        elif isinstance(x, list):
            d = []
            for k in x:
                d.append(cls.item_to_json(k, depth - 1))
        else:
            d = typename(x)

        return d

    @classmethod
    def item_to_str(cls, x, depth):
        res = cls.item_to_json(x, depth)
        return json.dumps(res)

    @classmethod
    def group_items(cls, items, depth):
        ct = Counter()
        groups = defaultdict(list)
        for x in items:
            desc = cls.item_to_str(x, depth)
            ct[desc] += 1
            groups[desc].append(x)
        # 按照值的数量对groups排序
        groups = dict(sorted(groups.items(), key=lambda x: -len(x[1])))
        return groups

    @classmethod
    def get_items_structures(cls, items, savefile=None):
        """ 获取jsonl数据的结构分布

        :param list[json] items: 一组json数据
        :return list[json]: 统计每种结构的分布数量，按树形结构，从多到少排序展示
        """

        def add_group(parent, items, depth, res):
            tag = parent['depth'] if parent else ''
            groups = cls.group_items(items, depth)
            for i, (desc, items2) in enumerate(groups.items(), start=1):
                if desc == parent['desc']:  # 再细化的结果跟父结点相同时，不再加深层级
                    continue

                d = {}
                d['depth'] = f'{tag}-{i}' if tag else f'{i}'
                d['count'] = len(items2)
                d['desc'] = desc
                d['structure'] = cls.item_to_json(items2[0], depth)
                res.append(d)

                add_group(d, items2, depth + 1, res)

        res = []  # 初始化结果列表
        add_group({'depth': '', 'desc': ''}, items, 1, res)  # 从空标签开始递归

        if savefile:
            XlPath(savefile).write_jsonl(res)

        return res


def process_batch_files(srcdir,
                        dsttag,
                        line_convert_func,
                        pattern='*.jsonl',
                        if_exists='skip',
                        processes_num=1,
                        group_func=None,
                        batch_size=2000,
                        debug=False,
                        ):
    """ 通用批处理函数

    :param srcdir: 输入待处理的目录或文件
    :param pattern: 文件名检索逻辑，如 '*'，'*.jsonl', '*_std*.jsonl'等
    :param if_exists: 如果目标文件已存在，如何处理。'skip'跳过不重复处理，'overwrite'覆盖重新运行，'error'抛出报错
    :param processes_num: 并发处理的进程数
    :param batch_size: 每个文件缓存的条目数，超过这个数目后就会先写入文件
    """
    # 1 检索文件
    srcdir = XlPath(srcdir)
    if srcdir.is_file():
        files = [srcdir.name]
        srcdir = srcdir.parent
    else:
        files = [f.relpath(srcdir) for f in srcdir.rglob_files(pattern)]

    file_num = len(files)

    # 2 并发处理多个文件
    backend = 'loky' if processes_num > 1 else 'sequential'
    tasks = []
    for i, f in enumerate(files, start=1):
        task = delayed(process_single_file)(srcdir,
                                            f,
                                            dsttag,
                                            line_convert_func,
                                            cur_idx=i,
                                            total=file_num,
                                            if_exists=if_exists,
                                            group_func=group_func,
                                            batch_size=batch_size,
                                            debug=debug)
        tasks.append(task)

    Parallel(n_jobs=processes_num, backend=backend)(tasks)
