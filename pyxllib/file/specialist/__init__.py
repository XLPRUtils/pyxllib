#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 17:46

from itertools import islice
import multiprocessing
import multiprocessing.dummy

from joblib import Parallel, delayed

from pyxllib.file.specialist.filelib import *
from pyxllib.file.specialist.dirlib import *
from pyxllib.file.specialist.download import *


def merge_jsonl(*infiles):
    data = []
    for f in infiles:
        data += XlPath(f).read_jsonl()
    return data


class JsonlDataFile:
    """ 通用的jsonl文件处理类 """

    def __init__(self, filepath=None, num_records=None):
        """
        从指定的jsonl文件中读取数据。可以选择读取全部数据或只读取前N条数据。

        :param str filepath: jsonl文件的路径
        :param int num_records: 指定读取的记录数量，如果为None则读取全部数据
        """
        self.infile = None
        self.records = []

        if filepath is not None:
            filepath = XlPath(filepath)
            if '?k' in filepath.name:  # 如果文件名中有'?'，则需要进行模式匹配检索
                new_name = filepath.name.replace('?k', '*')
                filepaths = list(filepath.parent.glob(new_name))
                if filepaths:
                    filepath = filepaths[0]  # 找到第1个匹配的文件
                    self.infile = XlPath(filepath)
            else:
                self.infile = filepath

        if self.infile and self.infile.is_file():  # 机制上文件也可能不存在的，有可能只是一个预设目录~
            if num_records is None:
                # 读取全部数据
                if self.infile.is_file():
                    self.records = self.infile.read_jsonl()
            else:
                # 只读取部分数据
                self.read_partial_records(num_records)

    def __len__(self):
        return len(self.records)

    def yield_record(self, start=0, end=None, step=1, batch_size=None):
        """ 返回指定区间的记录

        :param int start: 起始记录索引，默认为0
        :param int end: 结束记录索引，默认为None（读取到记录末尾）
        :param int step: 步长，默认为1
        :param int batch_size: 每批返回的记录数，如果为None，则逐记录返回
        """
        total_records = len(self.records)  # 获取总记录数

        # 处理负索引
        if start < 0 or (end is not None and end < 0):
            if start < 0:
                start = total_records + start
            if end is not None and end < 0:
                end = total_records + end

        iterator = islice(self.records, start, end, step)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            if batch_size is None:
                yield from batch
            else:
                yield batch

    def read_partial_records(self, num_records):
        """ 从jsonl文件中只读取指定数量的记录 """
        if self.infile and self.infile.is_file():
            try:
                lines = next(self.infile.yield_line(batch_size=num_records))
                for line in lines:
                    self.records.append(json.loads(line))
            except StopIteration:
                self.records = []

    def save(self, outfile=None, ensure_ascii=False, json_encoder=None):
        """ 将当前数据保存到指定的jsonl文件中 """
        if outfile is None:  # 默认保存回原文件
            outfile = self.infile
        p = XlPath(outfile)

        # 如果文件名包含'?k'，则替换'?'为self.records的数量
        if m := re.search(r'\?k', p.name):
            n = len(self.records)
            if n < 500:
                replace_str = f'{n}'  # 数量小于500，直接给出数量
            else:
                v = int(round(n / 1000))  # 数量大于等于500，以"千"为单位'k'，四舍五入计算
                replace_str = f'{v}k'
            # 用新字符串替换原来的字符串
            new_name = re.sub(r'\?k', replace_str, p.name)
            p = p.with_name(new_name)  # 更改文件名

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_jsonl(self.records, ensure_ascii=ensure_ascii, default=json_encoder)

    def browse_record(self, index=None, paths=None, **kwargs):
        """ 在浏览器中显示指定记录的内容 """
        from pyxllib.prog.specialist import browser

        # 如果未提供索引，则尝试使用查询参数找到第一个匹配的记录
        if index is None:
            index = self.find_index(paths, **kwargs)
            if index is None:
                raise ValueError('No matching record found')

        record = self.records[index]
        html_content = ['<html><body><pre>',
                        json.dumps(record, ensure_ascii=False, indent=4),
                        '</pre></body></html>']
        html_file = (XlPath.tempdir() / f'{self.__class__.__name__}_{index}.html')
        html_file.write_text('\n'.join(html_content))
        browser.html(html_file)

    def browse_records(self, indices=None, paths=None, **kwargs):
        """ 在浏览器中显示所有匹配的记录 """
        from pyxllib.prog.specialist import browser

        if indices is None:
            indices = list(self.find_indexs(paths, **kwargs))
        if not indices:
            raise ValueError('No matching records found')

        html_content = ['<html><body><h1>Matching Records: {}</h1>'.format(len(indices))]

        for index in indices:
            record = self.records[index]
            html_content.extend([
                '<h2>Record {}</h2>'.format(index),
                '<pre>',
                json.dumps(record, ensure_ascii=False, indent=4),
                '</pre>'
            ])

        html_content.append('</body></html>')
        html_file = (XlPath.tempdir() / f'{self.__class__.__name__}_matched.html')
        html_file.write_text('\n'.join(html_content))
        browser.html(html_file)

    def find_indexs(self, paths=None, **kwargs):
        """ 查找满足特定条件的记录的索引，返回所有匹配的结果 """
        paths = paths or {}

        for i, record in enumerate(self.records):
            # 检查kwargs中的所有条件
            for key, value in kwargs.items():
                if callable(value):
                    if not value(record.get(key)):
                        break
                elif record.get(key) != value:
                    break
            else:
                # 检查paths中的所有条件
                for path, value in paths.items():
                    try:
                        actual_value = eval(f'record{path}')
                    except Exception:
                        break

                    if callable(value):
                        if not value(actual_value):
                            break
                    elif actual_value != value:
                        break
                else:
                    # 如果记录满足所有条件，则返回它的索引
                    yield i

    def find_index(self, paths=None, **kwargs):
        """
        :param dict paths: 在比较复杂场景，无法使用kwargs定位的时候可以用这个规则
            key: 检索范式
            value: 需要满足的值
            示例： find_index({"['messages'][0]['role']": 'user'})
        :param kwargs: 直接子结点名称和对应的值
            示例：find_index(id=2023071320000003)

        补充说明：
            1、paths和kwargs可以组合使用，表示必须同时满足二者里限定的所有规则
            2、value可以写一个 def func(v)->bool的函数，输入对应的value，返回是否满足条件
        """
        return next(self.find_indexs(paths, **kwargs), None)

    def add_record_basic(self, **kwargs):
        """ 最基础的添加一个条目的接口 """
        record = kwargs
        self.records.append(record)
        return record

    @classmethod
    def read_from_files(cls, src_files):
        """ 从多个文件中读取并合并数据，并返回新的JsonlDataFile实例 """
        merged_records = []
        for file in src_files:
            jsonl_file = cls(file)
            merged_records.extend(jsonl_file.records)
        # 创建新的实例并返回
        new_instance = cls()
        new_instance.records = merged_records
        return new_instance

    @classmethod
    def read_from_dir(cls, src_dir):
        """ 从一个目录下的所有jsonl文件中读取并合并数据，并返回新的JsonlDataFile实例 """
        src_dir = XlPath(src_dir)
        src_files = [str(file_path) for file_path in src_dir.glob('*.jsonl')]
        return cls.read_from_files(src_files)

    def __add__(self, other):
        """ 实现类加法操作，合并两个JsonlDataFile的records """
        if not isinstance(other, JsonlDataFile):
            raise TypeError(f'Unsupported operand type: {type(other)}')
        result = JsonlDataFile()
        result.records = self.records + other.records
        return result

    def __iadd__(self, other):
        """ 实现原地加法操作，即 += """
        if not isinstance(other, JsonlDataFile):
            raise TypeError(f'Unsupported operand type: {type(other)}')
        self.records += other.records
        return self

    def process_each_record(self, func, *,
                            timeout=None,
                            inplace=False, print_mode=0, threads_num=1,
                            **kwargs):
        """ 对records中的每个record应用函数func，可以选择是否在原地修改，以及是否显示进度条

        :param function func: 对record进行处理的函数，应接受一个record作为参数并返回处理后的record，如果返回None则删除该record
        :param bool inplace: 是否在原地修改records，如果为False（默认），则创建新的JsonlDataFile并返回
        :param int print_mode: 是否显示处理过程的进度条，0表示不显示（默认），1表示显示
        :return JsonlDataFile or None: 如果inplace为False，则返回新的JsonlDataFile，否则返回None
        :param int threads_num: 线程数，默认为1，即单线程

        遍历self.records，对每个record执行func函数，如果func返回None，则不包含该record到新的records中。
        """
        backend = 'threading' if threads_num != 1 else 'sequential'
        parallel = Parallel(n_jobs=threads_num, backend=backend,
                            timeout=timeout, return_as='generator')
        tasks = [delayed(func)(record) for record in self.records]
        new_records = []
        for y in tqdm(parallel(tasks), disable=not print_mode,
                      total=len(self.records), **kwargs):
            if y:
                new_records.append(y)

        if inplace:
            self.records = new_records

        return new_records

    def update_each_record(self, func, print_mode=0, threads_num=1, timeout=None):
        """ 遍历并对原始数据进行更改 """
        return self.process_each_record(func, inplace=True,
                                        timeout=timeout,
                                        print_mode=print_mode,
                                        threads_num=threads_num)


class JsonlDataDir:
    """ 注意这个类开发目标，应该是尽量去模拟JsonDataFile，让下游工作更好衔接统一 """

    def __init__(self, root):
        """ 一般用来处理较大的jsonl文件，将其该放到一个目录里，拆分成多个jsonl文件

        注意待处理的文件名是依照 01.jsonl, 02.jsonl,... 的格式识别的，不要改动这个规则
        """
        self.root = XlPath(root)
        self.files = []
        for f in self.root.glob_files('*.jsonl'):
            if re.match(r'_?\d+$', f.stem):  # 目前先用'_?'兼容旧版，但以后应该固定只匹配_\d+
                self.files.append(f)

    def __bool__(self):
        if self.root.is_dir() and self.files:
            return True
        else:
            return False

    def count_records(self):
        total = 0
        for f in self.files:
            total += len(JsonlDataFile(f).records)
        return total

    def check(self, title=''):
        """ 检查一些数据状态 """
        print(title, '文件数:', len(self.files), '条目数:', self.count_records())

    @classmethod
    def init_from_file(cls, file, lines_per_file=10000):
        """ 从一个jsonl文件初始化一个JsonlDataDir对象 """
        file = XlPath(file)
        dst_dir = file.parent / file.stem
        if not dst_dir.is_dir() and file.is_file():
            file.split_to_dir(lines_per_file, dst_dir)
        c = cls(dst_dir)
        return c

    def rearrange(self, lines_per_file=10000):
        """ 重新整理划分文件

        :param int lines_per_file: 每个文件的行数
        """
        output_dir = self.root

        # 使用临时文件名前缀，以便在处理完成后更改为最终的文件名
        temp_prefix = 'temp_'

        new_file_count = 0
        new_file = None
        line_count = 0

        # 计算总行数以确定文件名的前导零数量
        total_lines = sum(1 for file in self.files for _ in file.open('r', encoding='utf-8'))
        num_digits = len(str((total_lines + lines_per_file - 1) // lines_per_file))

        for file in self.files:
            with file.open('r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    if line_count == 0:
                        if new_file is not None:
                            new_file.close()
                        new_file_name = f'{temp_prefix}{new_file_count:0{num_digits}d}.jsonl'
                        new_file_path = output_dir / new_file_name
                        new_file = new_file_path.open('w', encoding='utf-8')
                        new_file_count += 1

                    new_file.write(line)
                    line_count += 1

                    if line_count == lines_per_file:
                        line_count = 0

        if new_file is not None:
            new_file.close()

        # 删除旧文件
        for file in self.files:
            os.remove(file)

        # 将临时文件名更改为最终的文件名
        for temp_file in output_dir.glob(f'{temp_prefix}*.jsonl'):
            final_name = temp_file.name[len(temp_prefix) - 1:]
            temp_file.rename(output_dir / final_name)

    def yield_record(self, batch_size=None):
        """ 返回数据记录

        :param int batch_size: 每批返回的记录数，如果为None，则逐条返回
        """
        for i, file in enumerate(self.files):
            data = file.read_jsonl()
            iterator = iter(data)
            while True:
                batch = list(islice(iterator, batch_size))
                if not batch:
                    break
                if batch_size is None:
                    yield from batch
                else:
                    yield batch

    def process_each_file(self, func, *, print_mode=0,
                          processes_num=1,
                          desc='process_each_file',
                          **kwargs):
        backend = 'loky' if processes_num != 1 else 'sequential'
        parallel = Parallel(n_jobs=processes_num, backend=backend, return_as='generator')
        tasks = [delayed(func)(file) for file in self.files]
        list(tqdm(parallel(tasks), disable=not print_mode,
                  total=len(self.files), desc=desc, **kwargs))

    def process_each_record(self, func, *, inplace=False,
                            print_mode=2, desc=None, timeout=None,
                            processes_num=1, threads_num=1,
                            dst_dir=None):
        """ 封装的对每个record进行操作的函数

        :param int processes_num: 进程数，每个文件为单独一个进程
        :param int threads_num: 线程数，每个文件处理的时候使用几个线程
        :param print_mode: 0 不显示，1 只显示文件数进度，2 显示文件内处理进度
        """

        def func2(file):
            # print(file)
            jdf = JsonlDataFile(file)
            stem = jdf.infile.stem
            jdf.process_each_record(func, inplace=inplace,
                                    timeout=timeout,
                                    print_mode=print_mode == 2,
                                    threads_num=threads_num,
                                    mininterval=processes_num * 3,
                                    desc=stem)

            if inplace:
                jdf.save()

            if dst_dir:
                jdf.save(XlPath(dst_dir) / jdf.infile.name)

        self.process_each_file(func2, processes_num=processes_num,
                               print_mode=print_mode == 1, desc=desc)

    def save(self, dst_path=None):
        """ 将数据合并到一个jsonl文件中 """
        if not dst_path:
            dst_path = self.root.parent / f'{self.root.name}.jsonl'
        dst_path = XlPath(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with dst_path.open('w', encoding='utf8') as f:
            for file in tqdm(self.files, desc=f'合并文件并保存 {dst_path.name}'):
                with file.open('r', encoding='utf8') as f2:
                    for line in f2:
                        if line.strip():  # 不存储空行
                            f.write(line)
