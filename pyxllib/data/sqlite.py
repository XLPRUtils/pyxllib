#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/04/12 08:59

import copy
import json
import re
import sqlite3
import warnings
from collections import defaultdict

import pandas as pd

# 旧版的pandas警告
warnings.filterwarnings('ignore', message="pandas only support SQLAlchemy connectable")
# 新版的pandas警告多了个's‘
warnings.filterwarnings('ignore', message="pandas only supports SQLAlchemy connectable")


class SqlBuilder:
    def __init__(self, table=''):
        self.table = table
        self._select = []
        self._set = []
        self._where = []
        self._order_by = []
        self._group_by = []
        self._limit = None  # 限制最多读多少条数据
        self._offset = None  # 从第几条数据开始读

    def copy(self):
        # 拷贝一个当前状态的副本sql
        return copy.deepcopy(self)

    def __1_组件(self):
        pass

    def from_table(self, table):
        self.table = table
        return self

    def select(self, *columns):
        self._select.extend(columns)
        return self

    def set(self, *columns):
        self._set.extend(columns)
        return self

    def where(self, condition):
        if isinstance(condition, (list, tuple)):
            self._where.extend(condition)
        elif isinstance(condition, str):
            self._where.append(condition)
        else:
            raise ValueError(f'不支持的where条件类型{type(condition)}')

        return self

    def where_in(self, column, values):
        if values is None:
            return self

        if isinstance(values, str):
            values = [values]
        values_str = ', '.join(f"'{str(value)}'" for value in values)
        if len(values) == 1:
            self._where.append(f"{column} = {values_str[0]}")
        else:
            self._where.append(f"{column} IN ({values_str})")
        return self

    def where_or(self, *conditions):
        """ 输入的这一批条件，作为OR组合后成为一个整体条件
        """
        self._where.append(f"({' OR '.join(conditions)})")
        return self

    def where_mod(self, column, divisor, remainder):
        """ 输入的column列的值对divisor取余，筛选出余数为remainder的记录
        """
        condition = f"({column} % {divisor} = {remainder})"
        self._where.append(condition)
        return self

    def where_mod2(self, desc):
        """ 使用一种特殊的格式化标记来设置规则

        :param desc: 例如 'id%2=1'
        """
        if not desc:
            return
        column, divisor_remainder = desc.split('%')
        divisor, remainder = map(int, divisor_remainder.split('='))
        return self.where_mod(column, divisor, remainder)

    def group_by(self, *columns):
        self._group_by.extend(columns)
        return self

    def order_by(self, *columns):
        self._order_by.extend(columns)
        return self

    def limit(self, limit, offset=None):
        self._limit = limit
        self._offset = offset
        return self

    def __2_build_初级命令(self):
        pass

    def build_select(self, *columns):
        if columns:
            columns = self._select + list(columns)
        else:
            columns = self._select

        sql = [f"SELECT {', '.join(columns) or '*'}", f"FROM {self.table}"]
        if self._where:
            sql.append(f"WHERE {' AND '.join(self._where)}")
        if self._group_by:
            sql.append(f"GROUP BY {', '.join(self._group_by)}")
        if self._order_by:
            sql.append(f"ORDER BY {', '.join(self._order_by)}")
        if self._limit is not None:
            limit_clause = f"LIMIT {self._limit}"
            if self._offset is not None:
                limit_clause += f" OFFSET {self._offset}"
            sql.append(limit_clause)
        return '\n'.join(sql)

    def build_count(self):
        sql = [f"SELECT COUNT(*)", f"FROM {self.table}"]
        if self._where:
            sql.append(f"WHERE {' AND '.join(self._where)}")
        if self._group_by:
            sql.append(f"GROUP BY {', '.join(self._group_by)}")
        return '\n'.join(sql)

    def build_update(self):
        sql = [f"UPDATE {self.table}"]
        if self._set:
            sql.append(f"SET {', '.join(self._set)}")
        if self._where:
            sql.append(f"WHERE {' AND '.join(self._where)}")
        return '\n'.join(sql)

    def __3_build_中级命令(self):
        pass

    def build_check_data_type(self, column):
        """ 检查column的数据类型 """
        sql = SqlBuilder('information_schema.columns')
        sql.select(f"data_type")
        sql.where(f"table_name='{self.table}' AND column_name='{column}'")
        return sql.build_select()

    def build_group_count(self, columns, count_column_name='cnt'):
        sql = SqlBuilder(self.table)
        if isinstance(columns, (list, tuple)):
            columns = ', '.join(columns)
        sql.select(columns, f"COUNT(*) {count_column_name}")
        sql.group_by(columns)
        sql.order_by(f'{count_column_name} DESC')
        sql._where = self._where.copy()
        return sql.build_select()


class SqlBase:
    """ Sql语法通用的功能 """

    def __init__(self, *args, **kwargs):
        self._commit_cache = defaultdict(list)

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
        self.execute(f'CREATE INDEX {index_name} ON {table_name}({cols})')

    def create_index2(self, table_name, cols):
        """ 创建一个简单的索引，索引名字自动生成 """
        if not isinstance(cols, str):
            cols = ','.join(map(str, cols))
        self.execute(f'CREATE INDEX idx_{table_name}_{cols.replace(",", "_")} ON {table_name}({cols})')

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

    def exec2one(self, *args, **kwargs):
        """ 获得第1行的值 """
        try:
            return self.execute(*args, **kwargs).fetchone()[0]
        except TypeError:
            return None

    def exec2row(self, *args, **kwargs):
        """ 获得第1行的值 """
        return self.execute(*args, **kwargs).fetchone()

    def exec2col(self, *args, **kwargs):
        """ 获得第1列的值 """
        return [row[0] for row in self.execute(*args, **kwargs).fetchall()]

    # 兼容旧接口
    exec_col = exec2col

    def exec2df(self, *args, **kwargs):
        """ 获得pandas.DataFrame类型的返回值 """
        return pd.read_sql(*args, self, **kwargs)

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

    def commit_base(self, commit_type, query, params=None):
        """
        :param commit_type:
            -1，先真正缓存在本地
            False，传统的事务机制，虽然不会更新数据，但每一条依然会连接数据库，其实速度回挺慢的
            True，传统的事务机制，但每条都作为独立事务，直接更新了
        """
        if commit_type == -1:
            self._commit_cache[query].append(params)
        elif commit_type is False:
            self.execute(query, params)
        elif commit_type is True:
            self.execute(query, params)
            self.commit()

    def commit_all(self):
        if not self._commit_cache:
            self.commit()
            return

        for query, params in self._commit_cache.items():
            cur = self.cursor()
            cur.executemany(query, params)
            cur.close()
            self.commit()

        self._commit_cache = defaultdict(list)

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
        self.commit_base(commit,
                         f'UPDATE {table_name} SET {kvs} WHERE {ops}',
                         self.cvt_types(vals))

    def delete_row(self, table_name, where, *, commit=False):
        """ 【删】删除数据

        :param dict where: 怎么匹配到对应记录
        :param commit: 建议减小commit频率，会极大降低性能
        :return:
        """
        ops = ' AND '.join([f'{k}=%s' for k in where.keys()])
        vals = list(where.values())
        self.commit_base(commit,
                         f'DELETE FROM {table_name} WHERE {ops}',
                         self.cvt_types(vals))

    def select_col(self, table_name, col):
        """ 获得一列数据，常使用的功能，所以做了一个封装

        注意，"exec"前缀的方法一般返回的是迭代器，而"select"前缀获得一般是直接的全部列表、结果
        """
        return self.exec2col(f'SELECT {col} FROM {table_name}')

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

    def get_count_by_altering_query(self, data_query: str) -> int:
        """
        从给定的SQL SELECT查询中获取行数计数。这个方法通过修改原始的SELECT查询，
        将其转换为一个COUNT查询来实现计数。这种方法特别适用于在获取大量数据之前，
        需要预估数据量的场景。

        问题背景：
        在进行大规模数据处理前，了解数据的规模可以帮助进行更有效的资源分配和性能优化。
        传统的做法是分两步执行：首先计算数据总量，然后再执行实际的数据提取。
        这个函数旨在通过单个查询来简化这一流程，减少数据库的负载和响应时间。

        实现机制：
        函数首先使用正则表达式识别出SQL查询的FROM关键词，这是因为无论SELECT查询的复杂程度如何，
        计数的核心都是保留FROM及其后面的表和条件语句。然后，它构造一个新的COUNT查询，
        替换原始查询中的SELECT部分。最后，函数执行这个新的查询并返回结果。

        :param data_query (str): 原始的SQL SELECT查询字符串。
        :return int: 查询结果的行数。

        示例:
        >> sql = SqlBase()
        >> query = "SELECT id, name FROM users WHERE active = True"
        >> count = sql.get_count_by_altering_query(query)
        >> print(count)
        45

        注意:
        - 这个函数假设输入的是合法的SQL SELECT查询。
        - 函数依赖于数据库连接的execute方法能够正确执行转换后的COUNT查询。
        - 在一些复杂的SQL查询中，特别是包含子查询、特殊函数或复杂的JOIN操作时，
          请确保转换后的计数查询仍然有效。
        """
        # 使用正则表达式定位'FROM'（考虑各种大小写情况），并确保它前后是空格或语句的开始/结束
        match = re.search(r'\bFROM\b', data_query, flags=re.IGNORECASE)
        if match:
            from_index = match.start()
            count_query = 'SELECT COUNT(*) ' + data_query[from_index:]  # 构造计数查询
            try:
                result = self.execute(count_query).fetchone()  # 执行查询
                return result[0] if result else 0  # 返回计数结果
            except Exception as e:
                print(f"Error executing count query: {e}")
                return 0
        else:
            print("No 'FROM' keyword found in the data query.")
            return 0

    def get_column_data_type(self, table_name, col_name):
        """ 获取表格中某一列的数据类型 """
        return self.exec2one(SqlBuilder(table_name).build_check_data_type(col_name))


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

    def exec2nametuple(self, *args, **kwargs):
        cur = self.cursor()
        cur.row_factory = sqlite3.Row
        return cur.execute(*args, **kwargs)

    def exec2dict(self, *args, **kwargs):
        """ execute基础上，改成返回值为dict类型 """

        def dict_factory(cursor, row):
            d = {}
            for idx, col in enumerate(cursor.description):
                d[col[0]] = row[idx]
            return d

        cur = self.cursor()  # todo 不关是不是不好？如果出错了是不是会事务未结束导致无法修改表格结构？是否有auto close的机制？
        cur.row_factory = dict_factory
        return cur.execute(*args, **kwargs)

    # 兼容老版本
    exec_dict = exec2dict

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
