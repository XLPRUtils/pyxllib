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
            self.records = self.infile.read_jsonl()

    def save(self, outfile=None, ensure_ascii=False):
        """ 将当前数据保存到指定的jsonl文件中 """
        if outfile is None:  # 默认保存回原文件
            outfile = self.infile
        p = XlPath(outfile)
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
