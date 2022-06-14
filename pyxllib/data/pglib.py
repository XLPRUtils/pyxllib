#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/06/09 16:26

"""
针对PostgreSQL封装的工具
"""

from pyxllib.prog.pupil import check_install_package

check_install_package('psycopg')

import json

import psycopg
import psycopg.rows

from pyxllib.data.sqlite import SqlBase


class Connection(psycopg.Connection, SqlBase):
    def exec_col(self, query, params=None, *, prepare=None, binary=False):
        """ 获得第1列的值，注意这个方法跟select_col很像，但更泛用，优先推荐使用exec_col
        """
        for row in self.execute(query, params, prepare=prepare, binary=binary):
            yield row[0]

    def exec_nametuple(self, *args, **kwargs):
        cur = self.cursor(row_factory=psycopg.rows.namedtuple_row)
        return cur.execute(*args, **kwargs)

    def exec_dict(self, *args, **kwargs):
        cur = self.cursor(row_factory=psycopg.rows.dict_row)
        return cur.execute(*args, **kwargs)

    def __增(self):
        pass

    @classmethod
    def cvt_type(cls, val):
        if isinstance(val, dict):
            val = json.dumps(val, ensure_ascii=False)
        # 注意list数组类型读、写都会自动适配py
        return val

    @classmethod
    def cvt_types(cls, vals):
        return [cls.cvt_type(v) for v in vals]

    def insert_row(self, table_name, cols, *, on_conflict='DO NOTHING', commit=True):
        """ 【增】插入新数据

        :param dict cols: 用字典表示的要插入的值
        :param on_conflict: 如果已存在的处理策略
            DO NOTHING，跳过不处理
            REPLACE，这是个特殊标记，会转为主键冲突也仍然更新所有值
            (id) DO UPDATE SET host_name=EXCLUDED.host_name, nick_name='abc'
                也可以写复杂的处理算法规则，详见 http://postgres.cn/docs/12/sql-insert.html
                比如这里是插入的id重复的话，就把host_name替换掉，还可以指定nick_name替换为'abc'
                注意前面的(id)是必须要输入的
        """
        ks = ','.join(cols.keys())
        vs = ','.join(['%s'] * (len(cols.keys())))
        query = f'INSERT INTO {table_name}({ks}) VALUES ({vs})'
        params = self.cvt_types(cols.values())

        if on_conflict == 'REPLACE':
            on_conflict = f'ON CONFLICT ON CONSTRAINT {table_name}_pkey DO UPDATE SET ' + \
                          ','.join([f'{k}=EXCLUDED.{k}' for k in cols.keys()])
        else:
            on_conflict = f'ON CONFLICT {on_conflict}'
        query += f' {on_conflict}'

        self.execute(query, params)
        if commit:
            self.commit()

    def insert_rows(self, table_name, keys, ls, *, on_conflict='DO NOTHING', commit=True):
        """ 【增】插入新数据

        :param str keys: 要插入的字段名，一个字符串，逗号,隔开属性值
        :param list[list] ls: n行m列的数组

        >> con.insert_rows('hosts2', 'id,host_name,nick_name', [[1, 'test5', 'dcba'], [11, 'test', 'aabb']])
        """
        n, m = len(ls), len(keys.split(','))
        vs = ','.join(['%s'] * m)
        query = f'INSERT INTO {table_name}({keys}) VALUES ' + ','.join([f'({vs})'] * n)
        params = []
        for cols in ls:
            params += self.cvt_types(cols)

        if on_conflict == 'REPLACE':
            on_conflict = f'ON CONFLICT ON CONSTRAINT {table_name}_pkey DO UPDATE SET ' + \
                          ','.join([f'{k.strip()}=EXCLUDED.{k.strip()}' for k in keys.split(',')])
        else:
            on_conflict = f'ON CONFLICT {on_conflict}'
        query += f' {on_conflict}'

        self.execute(query, params)
        if commit:
            self.commit()

    def __删(self):
        pass

    def __改(self):
        pass

    def update(self, table_name, cols, where, *, commit=True):
        """ 【改】更新数据

        :param dict cols: 要更新的字段及值
        :param dict where: 怎么匹配到对应记录
        :return:
        """
        kvs = ','.join([f'{k}=%s' for k in cols.keys()])
        ops = ' AND '.join([f'{k}=%s' for k in where.keys()])
        vals = list(cols.values()) + list(where.values())
        self.execute(f'UPDATE {table_name} SET {kvs} WHERE {ops}', self.cvt_types(vals))
        if commit:
            self.commit()

    def __查(self):
        pass

    def has_table(self, table_name, schemaname='public'):
        query = ["SELECT EXISTS (SELECT FROM pg_tables ",
                 f"WHERE schemaname='{schemaname}' AND tablename='{table_name}')"]
        res = self.execute(' '.join(query))
        return res.fetchone()[0]
