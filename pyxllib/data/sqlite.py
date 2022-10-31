#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/04/12 08:59

import json
import sqlite3

import pandas as pd


class SqlBase:
    """ Sql语法通用的功能 """

    def __1_库(self):
        pass

    def __2_表格(self):
        pass

    def get_table_names(self):
        """ 获得所有表格名 """
        raise NotImplementedError

    def create_table(self, table_name, column_descs):
        """ 【DDL增】

        :param table_name:
        :param column_descs:
            str, 正常的列格式描述，例如 'c1 text, c2 blob'
            dict, k是列名，v是一个具体的值，供分析参考格式类型
        :return:
        """
        # 1 列数据格式智能分析
        if not isinstance(column_descs, str):
            descs = []
            for k, v in column_descs.items():
                t = self.autotype(v)
                descs.append(f'{k} {t}')
            column_descs = ','.join(descs)

        # 2 新建表格
        self.execute(f'CREATE TABLE {table_name}({column_descs})')

    def get_column_names(self, table_name):
        """ 获得一个表格中所有字段名 """
        raise NotImplementedError

    def ensure_column(self, table_name, col_name, col_type='', *, col_ref_val=None, **kwargs):
        """ 【DDL改】添加字段

        :param table_name:
        :param col_name:
        :param col_type:
        :param col_ref_val: 可以通过具体的数值，智能判断列数据格式
        :return:
        """
        if col_name in self.get_column_names(table_name):
            return

        if col_ref_val:
            col_type = self.autotype(col_ref_val)
        if not col_type:
            col_type = 'text'

        self.execute(f'ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}')

    def delete_column(self, table_name, col_name):
        """ 删除表格中的一个列

        其实直接删除也不麻烦：ALTER TABLE {table_name} DROP COLUMN {col_name}
        这个封装主要是鲁棒性考虑，只有列存在时才会删除
        """
        if col_name in self.get_column_names(table_name):
            self.execute(f'ALTER TABLE {table_name} DROP COLUMN {col_name}')

    def create_index(self, index_name, table_name, cols):
        if not isinstance(cols, str):
            cols = ','.join(map(str, cols))
        self.execute(f'CREATE INDEX {index_name} ON {table_name}(cols)')

    def keep_top_n_rows(self, table_name, num, col_name='id'):
        """ 只保留一小部分数据，常用来做lite、demo数据示例文件

        :param col_name: 参照的列名
        """
        self.execute(f'DELETE FROM {table_name} WHERE {col_name} NOT IN'
                     f'(SELECT {col_name} FROM {table_name} LIMIT {num})')
        self.commit()

    def clear_table(self, table_name):
        """ 【DDL删】清空表格内容 """
        self.execute(f'DELETE FROM {table_name}')

    def count_all_talbe_rows(self):
        """ 统计所有表格的数据行数 """
        names = self.get_table_names()
        ls = []
        for name in names:
            n = self.execute(f'SELECT count(*) FROM {name}').fetchone()[0]
            ls.append([name, n])
        ls.sort(key=lambda x: -x[1])
        return ls

    def __3_execute(self):
        pass

    def exec_col(self, *args, **kwargs):
        """ 获得第1列的值，注意这个方法跟select_col很像，但更泛用，优先推荐使用exec_col

        >> self.exec_col('SELECT id FROM skindata')
        """
        for row in self.execute(*args, **kwargs):
            yield row[0]

    def __4_数据类型(self):
        pass

    @classmethod
    def cvt_type(cls, val):
        """ py一些内存对象，需要进行适当的格式转换，才能存储到sql中
        """
        raise NotImplementedError

    @classmethod
    def cvt_types(cls, vals):
        """ 批量转换类型
        """
        return [cls.cvt_type(v) for v in vals]

    @classmethod
    def autotype(cls, val):
        """ 自动判断一个py内存对象应该以什么类型进行存储 """
        raise NotImplementedError

    def __5_增删改查(self):
        pass

    def update_row(self, table_name, cols, where, *, commit=False):
        """ 【改】更新数据

        虽然名称是update_row，但where条件满足时，是有可能批量替换多行的

        :param dict cols: 要更新的字段及值
        :param dict where: 怎么匹配到对应记录
        :param commit: 建议减小commit频率，会极大降低性能
        :return:

        >> xldb.update('xlapi', {'input': d}, {'id': x['id']})
        """
        kvs = ','.join([f'{k}=%s' for k in cols.keys()])
        ops = ' AND '.join([f'{k}=%s' for k in where.keys()])
        vals = list(cols.values()) + list(where.values())
        self.execute(f'UPDATE {table_name} SET {kvs} WHERE {ops}', self.cvt_types(vals))
        if commit:
            self.commit()

    def select_col(self, table_name, col):
        """ 获得一列数据，常使用的功能，所以做了一个封装
        """
        return [x[0] for x in self.execute(f'SELECT {col} FROM {table_name}').fetchall()]

    def group_count(self, table_name, cols, count_column_name='cnt'):
        """ 【查】分组统计各组值组合出现次数
        分析{table}表中，{cols}出现的种类和次数，按照出现次数从多到少排序

        :param str|list cols: 输入逗号','隔开的字符串，比如
            con.group_count('gpu_trace', 'host_name,total_memory')
            后记：写list也行，会自动join为字符串
        """
        if not isinstance(cols, str):
            cols = ','.join(map(str, cols))
        sql = f'SELECT {cols}, COUNT(*) {count_column_name} FROM {table_name} ' \
              f'GROUP BY {cols} ORDER BY {count_column_name} DESC'
        records = self.execute(sql).fetchall()
        df = pd.DataFrame.from_records(records, columns=cols.split(',') + [count_column_name])
        return df


class Connection(sqlite3.Connection, SqlBase):
    """
    DDL - 数据定义语言
        CREATE	创建一个新的表，一个表的视图，或者数据库中的其他对象。
        ALTER	修改数据库中的某个已有的数据库对象，比如一个表。
        DROP	删除整个表，或者表的视图，或者数据库中的其他对象。
    DML - 数据操作语言
        INSERT	创建一条记录。
        UPDATE	修改记录。
        DELETE	删除记录。
    DQL - 数据查询语言
        SELECT	从一个或多个表中检索某些记录。
    """

    def __1_库(self):
        pass

    def vacuum(self):
        """ 删除数据后，文件不会直接减小，需要使用vacuum来实际压缩文件占用空间 """
        self.execute('vacuum')  # 不用 commit

    def __2_表格(self):
        pass

    def get_table_names(self):
        """ 获得所有表格名 """
        return [x[0] for x in self.execute("SELECT name FROM sqlite_master WHERE type='table'")]

    def has_table(self, table_name):
        res = self.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'").fetchone()
        return bool(res)

    def get_column_names(self, table_name):
        """ 【查】表格有哪些字段
        """
        names = [item[1] for item in self.execute(f'PRAGMA table_info({table_name})')]
        return names

    def __3_execute(self):
        pass

    def exec_nametuple(self, *args, **kwargs):
        cur = self.cursor()
        cur.row_factory = sqlite3.Row
        return cur.execute(*args, **kwargs)

    def exec_dict(self, *args, **kwargs):
        """ execute基础上，改成返回值为dict类型 """

        def dict_factory(cursor, row):
            d = {}
            for idx, col in enumerate(cursor.description):
                d[col[0]] = row[idx]
            return d

        cur = self.cursor()  # todo 不关是不是不好？如果出错了是不是会事务未结束导致无法修改表格结构？是否有auto close的机制？
        cur.row_factory = dict_factory
        return cur.execute(*args, **kwargs)

    def __4_数据类型(self):
        pass

    @classmethod
    def cvt_type(cls, val):
        if isinstance(val, (dict, list, tuple)):
            val = json.dumps(val, ensure_ascii=False)
        return val

    @classmethod
    def autotype(cls, val):
        if isinstance(val, str):
            return 'text'
        elif isinstance(val, (int, bool)):
            return 'integer'
        elif isinstance(val, float):
            return 'real'
        else:  # 其他dict、list等类型，可以用json.dumps或str转文本存储
            return 'text'

    def __5_增删改查(self):
        pass

    def insert_row(self, table_name, cols, if_exists='IGNORE'):
        """ 【增】插入新数据

        :param table_name:
        :param cols: 一般是用字典表示的要插入的值
        :param if_exists: 如果已存在的处理策略
            IGNORE，跳过
            REPLACE，替换
        :return:

        注意加了 OR IGNORE，支持重复数据自动忽略插入
        """
        ks = ','.join(cols.keys())
        vs = ','.join('?' * (len(cols.keys())))
        self.execute(f'INSERT OR {if_exists} INTO {table_name}({ks}) VALUES ({vs})', self.cvt_types(cols.values()))
