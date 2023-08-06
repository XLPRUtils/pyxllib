#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 17:46


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

    def __init__(self, filepath=None):
        """ 从指定的jsonl文件中读取数据 """
        self.infile = None
        self.records = []

        if filepath is not None:
            self.infile = XlPath(filepath)
            if self.infile.is_file():
                self.records = self.infile.read_jsonl()

    def save(self, outfile=None, ensure_ascii=False):
        """ 将当前数据保存到指定的jsonl文件中 """
        if outfile is None:  # 默认保存回原文件
            outfile = self.infile
        p = XlPath(outfile)

        # 如果文件名包含'_？k_'，则替换'？'为self.records的数量
        if m := re.search(r'_(\？)k_', p.name):
            n = len(self.records)
            if n < 500:
                replace_str = f'_{n}_'  # 数量小于500，直接给出数量
            else:
                v = int(round(n / 1000))  # 数量大于等于500，以"千"为单位'k'，四舍五入计算
                replace_str = f'_{v}k_'
            # 用新字符串替换原来的字符串
            new_name = re.sub(r'_(\？)k_', replace_str, p.name)
            p = p.with_name(new_name)  # 更改文件名

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_jsonl(self.records, ensure_ascii=ensure_ascii)

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

    def split_to_dir(self, num_split_files, dst_dir=None):
        """ 将数据拆分到多个文件中，如果提供了目标目录，则将拆分的文件保存到目标目录，否则保存到当前工作目录 """
        if dst_dir is None:
            # 如果未提供目标目录，则拆分的文件保存到当前工作目录
            dst_dir = self.infile.parent / f"{self.infile.stem}_split"
        else:
            # 如果提供了目标目录，将拆分的文件保存到目标目录
            dst_dir = XlPath(dst_dir)

        dst_dir.mkdir(parents=True, exist_ok=True)
        chunk_size = len(self.records) // num_split_files
        # 格式化字符串，使得编号部分的长度和总文件数量的位数相同
        filename_format = "{:0" + str(len(str(num_split_files))) + "d}"
        split_files = []  # 用于保存拆分的文件路径
        for i in range(num_split_files):
            chunk_records = self.records[i * chunk_size:(i + 1) * chunk_size]
            outfile = dst_dir / f"{self.infile.stem}_{filename_format.format(i)}.jsonl"
            XlPath(outfile).write_jsonl(chunk_records, ensure_ascii=False)
            split_files.append(str(outfile))
        # 返回拆分的文件路径列表
        return split_files

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

    def apply_function_to_records(self, func, inplace=False, print_mode=0):
        """ 对records中的每个record应用函数func，可以选择是否在原地修改，以及是否显示进度条

        :param function func: 对record进行处理的函数，应接受一个record作为参数并返回处理后的record，如果返回None则删除该record
        :param bool inplace: 是否在原地修改records，如果为False（默认），则创建新的JsonlDataFile并返回
        :param int print_mode: 是否显示处理过程的进度条，0表示不显示（默认），1表示显示
        :return JsonlDataFile or None: 如果inplace为False，则返回新的JsonlDataFile，否则返回None

        遍历self.records，对每个record执行func函数，如果func返回None，则不包含该record到新的records中。

        >>> data_file = JsonlDataFile()
        >>> data_file.records = [{'a': 1}, {'b': 2}, {'c': 3}]
        >>> func = lambda x: {k: v * 2 for k, v in x.items()} if 'a' in x else None  # 定义一个只处理含有'a'的record并将其值翻倍的函数
        >>> new_data_file = data_file.apply_function_to_records(func, print_mode=1)
        >>> new_data_file.records
        [{'a': 2}]
        >>> data_file.records  # 原始的data_file并没有被修改
        [{'a': 1}, {'b': 2}, {'c': 3}]
        """
        records = self.records
        if print_mode == 1:
            records = tqdm(records)

        new_records = [func(record) for record in records if func(record) is not None]

        if inplace:
            self.records = new_records
        else:
            new_data_file = JsonlDataFile()
            new_data_file.records = new_records
            return new_data_file
