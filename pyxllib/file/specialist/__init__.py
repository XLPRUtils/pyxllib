#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 17:46
import re

from pyxllib.prog.pupil import check_install_package

check_install_package('joblib', 'joblib>=1.3.2')

from collections import OrderedDict
import sqlite3

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

    def yield_group(self, key, sort_mode='keep'):
        """ 分组提取数据

        :param key: 一个函数，对record的映射，通过这个映射规则来分组
        :param sort_mode:
            keep: 保留原本的相对顺序
            id: 按照id的值进行排序
            sort: 按照key的值进行排序
        """
        # 1 创建一个默认字典来保存分组
        grouped_data = OrderedDict()

        records = self.records
        if sort_mode == 'id':
            records = sorted(records, key=lambda x: x['id'])

        # 2 对数据进行分组
        for record in records:
            k = key(record)
            if k not in grouped_data:
                grouped_data[k] = [record]
            else:
                grouped_data[k].append(record)

        # 3 将分组的数据重新排序并合并为一个新列表
        # 并且在这里可以进行一些分组信息的计算
        if sort_mode == 'sort':
            grouped_data = {k: grouped_data[k] for k in sorted(grouped_data.keys())}

        # 4 返回分组后的数据
        yield from grouped_data.values()

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
                            inplace=False,
                            timeout=None,
                            print_mode=0,
                            threads_num=1,
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

        if print_mode:
            parallel = Parallel(n_jobs=threads_num, backend=backend,
                                timeout=timeout, return_as='generator')
            tasks = [delayed(func)(record) for record in self.records]
            new_records = []
            for y in tqdm(parallel(tasks), total=len(self.records), **kwargs):
                if y:
                    new_records.append(y)
        else:
            parallel = Parallel(n_jobs=threads_num, backend=backend, timeout=timeout)
            tasks = [delayed(func)(record) for record in self.records]
            new_records = parallel(tasks)
            new_records = [y for y in new_records if y]

        if inplace:
            self.records = new_records

        return new_records

    def update_each_record(self, func,
                           timeout=None,
                           print_mode=0,
                           threads_num=1):
        """ 遍历并对原始数据进行更改 """
        return self.process_each_record(func,
                                        inplace=True,
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
        self.update_subfiles()

    def update_subfiles(self):
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
            total += f.get_total_lines(skip_blank=True)
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

    def _rearrange_group(self, lines_per_file=10000,
                         group_key=None, sort_mode='keep',
                         print_mode=1):
        # 1 使用sqlite3存储数据和分组信息
        # 创建一个临时文件来作为SQLite数据库
        temp_db_file = self.root / 'data.sqlite3'
        temp_db_file.delete()

        # 使用临时文件创建SQLite数据库连接
        conn = sqlite3.connect(temp_db_file)
        cursor = conn.cursor()

        # 创建一个临时表来存储jsonl数据
        cursor.execute('CREATE TABLE records (id INTEGER PRIMARY KEY AUTOINCREMENT,'
                       'data TEXT, group_key TEXT)')
        # 给group_key添加索引
        cursor.execute('CREATE INDEX idx_group_key ON records(group_key)')

        # 从jsonl文件加载数据到SQLite数据库
        commit_interval = 2000  # 多少记录执行一次commit
        count = 0
        for record in tqdm(self.yield_record(), desc='计算每个record分组', disable=not print_mode):
            count += 1
            group = group_key(record) if group_key else count
            group = str(group)
            cursor.execute('INSERT INTO records (data, group_key) VALUES (?, ?)',
                           (json.dumps(record, ensure_ascii=False), group))
            if count % commit_interval == 0:
                conn.commit()
        conn.commit()

        # 2 查询数据库以进行排序和分组，并将结果写入新的jsonl文件
        new_file_count = 0
        lines_written = 0
        current_file = None
        sort_sql = ''
        if sort_mode == 'id':
            sort_sql = 'ORDER BY id'
        elif sort_mode == 'sort':
            sort_sql = f'ORDER BY {group_key}'

        for group, in tqdm(cursor.execute('SELECT DISTINCT group_key FROM records').fetchall(),
                           desc='提取每一组数据',
                           disable=not print_mode):
            query = f'SELECT data FROM records WHERE group_key = ? {sort_sql}'
            cursor.execute(query, (group,))

            if current_file is None or lines_written >= lines_per_file:
                if current_file:
                    current_file.close()
                new_file_name = f'temp_{new_file_count}.jsonl'
                new_file_path = self.root / new_file_name
                current_file = new_file_path.open('w', encoding='utf-8')
                new_file_count += 1
                lines_written = 0

            while True:
                row = cursor.fetchone()
                if row is None:
                    break

                current_file.write(row[0] + '\n')
                lines_written += 1

        if current_file:
            current_file.close()

        # 3 关闭数据库连接并删除临时文件
        conn.close()
        temp_db_file.delete()

        # 4 删除旧文件，重命名新文件
        for f in self.files:
            f.delete()

        widths = len(str(new_file_count))
        for temp_file in self.root.glob('temp_*.jsonl'):
            n = int(re.search(r'\d+', temp_file.name).group())
            temp_file.rename(self.root / f'_{n:0{widths}}.jsonl')

    def rearrange(self, lines_per_file=10000, group_key=None,
                  sort_mode='keep', print_mode=1):
        """ 重新整理划分文件

        :param int lines_per_file: 每个文件的行数
        :param func group_key: 用来分组的函数，确保相同key的数据会被分到同一个文件里
        :param str sort_mode:
            keep: 保留原本的相对顺序
            id: 按照id的值进行排序
            sort: 按照key的值进行排序
        """
        if group_key is not None or sort_mode != 'keep':
            return self._rearrange_group(lines_per_file, group_key, sort_mode, print_mode)

        output_dir = self.root
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

    def yield_group(self, key, sort_mode='keep'):
        """ 分组提取数据

        :param key: 一个函数，对record的映射，通过这个映射规则来分组

        注意：这个分组只会对每个分文件单独执行，不会全局性质检索
            一般要用self.rearrange对全局的文件进行检索重排后再使用这个函数
        """
        for filepath in self.files:
            jdf = JsonlDataFile(filepath)
            yield from jdf.yield_group(key, sort_mode)

    def process_each_file(self, func=None, *,
                          print_mode=0, desc='process_each_file',
                          processes_num=1,
                          **kwargs):
        backend = 'loky' if processes_num != 1 else 'sequential'
        if print_mode:
            parallel = Parallel(n_jobs=processes_num, backend=backend, return_as='generator')
            tasks = [delayed(func)(file) for file in self.files]
            list(tqdm(parallel(tasks), total=len(self.files), desc=desc, **kwargs))
        else:
            parallel = Parallel(n_jobs=processes_num, backend=backend)
            tasks = [delayed(func)(file) for file in self.files]
            parallel(tasks)

    def process_each_record(self, func, *,
                            inplace=False, reset=False,
                            print_mode=2, desc=None,
                            timeout=None,
                            processes_num=1, threads_num=1,
                            dst_dir=None, json_encoder=None):
        """ 封装的对每个record进行操作的函数

        :param func: 外部传入的处理函数
        :param inplace: 是否修改原始数据
        :param reset: 是否重新处理已经处理过的文件
        :param print_mode:
            0 不显示
            1 只显示文件数进度
            2（默认） 显示更详细的文件内处理进度
        :param desc: print_mode=1的进度条标题
        :param timeout: 超时时间，但有些场合会使用不了（比如linux的子进程里不能使用singal）
            在用不了的场合，可以使用requests自带的timeout等各种机制来限时
        :param processes_num: 进程数，每个文件为单独一个进程
        :param threads_num: 线程数，每个文件处理的时候使用几个线程
        :param dst_dir: 要保存到的目标目录，未设置的时候不保存
        :param json_encoder: 有些不是标准的json数据结构，如何进行处理，有需要的时候一般会设置成str
        """
        files_num = len(self.files)

        def process_jsonl_file(srcfile):
            # 1 如果没有reset，且dstfile存在，则不处理
            srcfile = XlPath(srcfile)
            if dst_dir:
                dstfile = XlPath(dst_dir) / srcfile.name
            else:
                dstfile = None
            if not reset and dstfile and dstfile.is_file():
                return

            # 2 跑特定文件里的条目
            jdf = JsonlDataFile(srcfile)
            new_records = jdf.process_each_record(func,
                                                  inplace=inplace,
                                                  print_mode=print_mode == 2,
                                                  desc=f'{jdf.infile.name}/{files_num}',
                                                  timeout=timeout,
                                                  threads_num=threads_num,
                                                  mininterval=processes_num * 3,
                                                  )

            # 3 是否修改原文件，是否保存到dst_dir
            if inplace:
                jdf.save()

            if dstfile:
                jdf = JsonlDataFile()
                jdf.records = new_records
                jdf.save(dstfile, json_encoder=json_encoder)

        self.process_each_file(process_jsonl_file,
                               processes_num=processes_num,
                               print_mode=print_mode == 1, desc=desc)

    def process_each_group(self, func, group_key, sort_mode='keep', *,
                           inplace=False, reset=False,
                           print_mode=1, desc=None,
                           processes_num=1,
                           dst_dir=None,
                           json_encoder=None):
        """ 封装的对每组records的处理

        todo 230909周六14:00，还有些细节功能可能不完善，比如内部的进度条，多线程等，等后续使用的时候慢慢优化
        """

        def process_jsonl_file(srcfile):
            # 1 如果没有reset，且dstfile存在，则不处理
            srcfile = XlPath(srcfile)
            if dst_dir:
                dstfile = XlPath(dst_dir) / srcfile.name
            else:
                dstfile = None
            if not reset and dstfile and dstfile.is_file():
                return

            # 2 跑特定文件里的条目
            jdf = JsonlDataFile(srcfile)
            new_records = []
            for records in jdf.yield_group(group_key, sort_mode):
                records2 = func(records)
                if records2:
                    new_records.extend(records2)

            # 3 是否修改原文件，是否保存到dst_dir
            if inplace:
                jdf.records = new_records
                jdf.save()

            if dstfile:
                jdf = JsonlDataFile()
                jdf.records = new_records
                jdf.save(dstfile, json_encoder=json_encoder)

        self.process_each_file(process_jsonl_file,
                               processes_num=processes_num,
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

    def clear(self):
        for f in self.files:
            f.delete()
