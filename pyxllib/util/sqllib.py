#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/06/03 09:52


import math
import subprocess

import pandas as pd

try:
    from bidict import bidict
except ModuleNotFoundError:
    subprocess.run(['pip', 'install', 'bidict'])
    from bidict import bidict

try:
    import sqlalchemy
except ModuleNotFoundError:
    subprocess.run(['pip', 'install', 'sqlalchemy'])
    subprocess.run(['pip', 'install', 'mysqlclient'])
    import sqlalchemy

from pyxllib.basic import TicToc, dformat, dprint, Path


SQL_LIB_ACCOUNT_FILE = Path(__file__).parent / 'sqllibaccount.pkl'


def create_account_df(file='sqllibaccount.pkl'):
    """请在这里设置您个人的账户密码，并在运行完后，销毁明文信息"""
    df = pd.DataFrame.from_records([
        ['ckz', 'rm.sbsql.rds.aliyuncs.com', '', '', 'dddddd'],
        ['ckzlocal', '0.0.0.0', '', '', 'eeeeee'],
    ], columns=['index_name', 'host', 'port', 'user', 'passwd'])
    df['port'] = df['port'].replace('', '3306')  # 没写端口的默认值
    df['user'] = df['user'].replace('', 'root')  # 没写用户名的默认值
    df['passwd'] = df['passwd'].replace('', '123456')  # 没写密码的默认值
    df.set_index('index_name', inplace=True)
    Path(file).write(df)


class SqlEngine:
    """mysql 通用基础类
    """

    def __init__(self, alias=None, database=None, *,
                 user='root', passwd='123456', host=None, port='3306',
                 connect_timeout=None, account_file_path=None):
        """ 初始化需要连接数据库

        :param alias: 数据库的简化别名，为了方便快速调用及隐藏明文密码
            使用该参数将会覆盖掉已有的user、passwd、host、port参数值
            例如我自己设置的别名有：
                ckz，我自己阿里云上的个人数据库
                ckzlocal，本PC开的数据库
        :param account_file_path: 使用alias时才有效
            该参数指定存储账号信息的pkl文件所在位置，注意pkl的格式必须用类似下述的代码方式生成
            默认从与该脚本同目录下的 sqllibaccount.pkl 文件获取

        :param database: 数据库名称
            例如在快乐做教研时一些相关数据库名：
                tr，教研
                tr_develop，教研开发数据
                tr_test，教研测试数据

        :param connect_timeout: 连接超时时等待秒数
            如果设置，建议2秒以上

        :return:
        """

        # 1 读取地址、账号信息
        if alias:
            if account_file_path is None:
                account_file_path = Path(SQL_LIB_ACCOUNT_FILE)
            # dprint(alias,account_file_path)
            record = Path(account_file_path).read().loc[alias]  # 从文件读取账号信息
            user, passwd, host, port = record.user, record.passwd, record.host, record.port

        # 2 '数据库类型+数据库驱动名称://用户名:口令@机器地址:端口号/数据库名'
        address = f'mysql+mysqldb://{user}:{passwd}@{host}:{port}/{database}?charset=utf8mb4'
        # 3 存储成员
        self.alias, self.database = alias, database
        connect_args = {"connect_timeout": connect_timeout} if connect_timeout else {}
        self.engine = sqlalchemy.create_engine(address, connect_args=connect_args)

    def query(self, sql, index_col=None, coerce_float=True, params=None,
              parse_dates=None, columns=None, chunksize=None):
        """本质上就是pd.read_sql函数

        pd.read_sql()知道这些就够用了 - 漫步量化 - CSDN博客:
        https://blog.csdn.net/The_Time_Runner/article/details/86601988

        官方文档：pandas.read_sql — pandas 0.25.1 documentation:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html

        :param sql: 命令字符串，可以写%s设置
            这里我做了扩展：
                单语句: '...'
                多语句: ['...', '...']，默认\n隔开
        :param index_col: 设为索引列
        :param coerce_float: 设为float类型的列
        :param params: 和sql命令相关，具体语法规则与所用的引擎相关，例如我这里是用sqlalchemy，支持用法
            'SELECT point_name FROM tr_point LIMIT %s', params=(5,))   # list、tuple等列举每个%s的值
            'SELECT point_name FROM tr_point LIMIT %(n)s', params={'n', 5})  # 是用dict关联命名参数
        :param parse_dates: 转为datetime类型的列
        :param columns: 要选取的列。一般没啥用，因为在sql命令里面一般就指定要选择的列了
        :param chunksize: 如果提供了一个整数值，那么就会返回一个generator，每次输出的行数就是提供的值的大小
        :return: DataFrame类型的数据
        """
        # 1 合并sql命令
        if isinstance(sql, str): sql = [sql]
        sql = '\n'.join(sql)

        # 含有 % 的特殊符号要转义
        # import sqlalchemy
        # sql = sqlalchemy.text(sql)

        # 2 解析结果
        res = pd.read_sql(sql, self.engine, index_col=index_col, coerce_float=coerce_float, params=params,
                          parse_dates=parse_dates, columns=columns, chunksize=chunksize)
        return res

    def execute(self, statement, *multiparams, **params):
        """本质上就是sqlalchemy.con.execute的封装

        可以这样使用：
        hsql.execute('UPDATE spell_check SET count=:count WHERE old=:old AND new=:new',
                      count=count[0] + add, old=old, new=new)
        """
        # 1 解析sql命令
        if isinstance(statement, str): statement = [statement]
        statement = '\n'.join(statement)
        statement = sqlalchemy.text(statement)

        # 2 如果设置了getdf参数
        res = self.engine.execute(statement, *multiparams, **params)
        return res

    def insert_from_df(self, df, table_name, patch_size=100, if_exists='append'):
        """将df写入con数据库的table_name表格

        190731周三18:51，TODO
        可以先用：df.to_sql('formula_stat', HistudySQL('dev', 'tr_develop').con, if_exists='replace')
        191017周四10:21，目前这函数改来改去，都还没严格测试呢~~

        这个函数开发要参考：DataFrame.to_sql()
            是因为其con参数好像不支持pymysql

        :param df: DataFrane类型表格数据
        :param table_name: 要写入的表格名
        :param patch_size: 每轮要写入的数量
            如果df很大，是无法一次性用sql语句写入的，一般要分批写
            patch_size是设置每批导入多少条数据
        :param if_exists: {'fail', 'replace', 'append'}, default 'append'
                How to behave if the table already exists.

                * fail: Raise a ValueError.
                * replace: Drop the table before inserting new values.
                * append: Insert new values to the existing table.
        """
        con = self.engine
        # TODO 增加表格是否存在的判断；我这个函数本质上只能往已存在的表格插入数据
        if if_exists == 'append':
            pass
        elif if_exists == 'replace':
            con.query(f'TRUNCATE TABLE {table_name}')
        elif if_exists == 'fail':
            raise ValueError('表格已存在')
        else:
            raise NotImplementedError

        # 1 删除table中不支持的df的列
        cols = pd.read_sql(f'SHOW COLUMNS FROM {table_name}', con)['Field']
        cols = list(set(df.columns) & set(cols))
        df = df[cols]

        # 2 将df每一行数据转成mysql语句文本
        data = []  # data[i]是第i条数据的sql文本

        # 除了nan，bool值、None值都能正常转换
        def func(x):
            # s = con.escape(str(x))
            s = x
            if s == 'nan': s = 'NULL'  # nan转为NULL
            return s

        for idx, row in df.iterrows():
            t = ', '.join(map(func, row))
            data.append('(' + t + ')')

        # 3 分批导入
        columns = '( ' + ', '.join(cols) + ' )'
        for j in range(0, math.ceil(len(data) / patch_size)):
            subdata = ',\n'.join(data[j * patch_size:(j + 1) * patch_size])
            con.execute("INSERT IGNORE INTO :a :b VALUES :c",
                        a=table_name, b=columns, c=subdata)
        con.commit()  # 更新后才会起作用


class SqlCodeGenerator:
    @staticmethod
    def keys_count(table, keys):
        codes = [f'-- 分析{table}表中，{keys}出现的种类和次数，按照出现次数从多到少排序',
                 f'SELECT {keys}, COUNT(*) cnt FROM {table} GROUP BY {keys} ORDER BY cnt DESC']
        return '\n'.join(codes)

    @staticmethod
    def one2many(table, keys, vars):
        codes = [f'-- 分析{table}表中，{keys}构成的键，对应{vars}构成的值，是否有一对多的关系，按多到少排序',
                 f'SELECT {keys}, COUNT(DISTINCT {vars}) cnt',
                 f'FROM {table} GROUP BY {keys}',
                 'HAVING cnt > 1 ORDER BY cnt DESC']
        return '\n'.join(codes)


def demo_sqlengine():
    db = SqlEngine('ckz', 'runoob')
    df = db.query('SELECT * FROM apps')
    print(df)


class MultiEnumTable:
    """多份枚举表的双向映射
        目前是用来做数据库表中的枚举值映射，但实际可以通用于很多地方

    >>> met = MultiEnumTable()
    >>> met.add_enum_table('subject', [5, 8, 6], ['语文', '数学', '英语'])
    >>> met.add_enum_table_from_dict('grade', {1: '小学', 2: '初中', 3: '高中'})

    >>> met['subject'][6]
    '英语'
    >>> met['subject'].inverse['英语']
    6

    >>> met.decode('subject', 5)
    '语文'
    >>> met.encode('subject', '数学')
    8

    >>> met.decodes('grade', [1, 3, 3, 2, 1])
    ['小学', '高中', '高中', '初中', '小学']
    >>> met.encodes('grade', ['小学', '高中', '大学', '初中', '小学'])
    [1, 3, None, 2, 1]
    """

    def __init__(self):
        self.enum_tables = dict()

    def __getitem__(self, table):
        return self.enum_tables[table]

    def add_enum_table(self, table, ids, values):
        """增加一个映射表"""
        self.enum_tables[table] = bidict({k: v for k, v in zip(ids, values)})

    def add_enum_table_from_dict(self, table, d):
        self.enum_tables[table] = bidict({k: v for k, v in d.items()})

    def set_alias(self, table, alias):
        """已有table的其他alias别名
        :param alias: list
        """
        for a in alias:
            self.enum_tables[a] = self.enum_tables[table]

    def decode(self, table, id_, default=None):
        """转明文"""
        return self.enum_tables[table].get(id_, default)

    def encode(self, table, value, default=None):
        """转id"""
        return self.enum_tables[table].inverse.get(value, default)

    def decodes(self, table, ids, default=None):
        d = self.enum_tables[table]
        return [d.get(k, default) for k in ids]

    def encodes(self, table, values, default=None):
        d = self.enum_tables[table].inverse
        return [d.get(v, default) for v in values]


def adjust_repeat_data(li, suffix='+'):
    """ 分析序列li里的值，对出现重复的值进行特殊标记去重
    :param li: list，每个元素值一般是str
    :param suffix: 通过增加什么后缀来去重
    :return: 新的无重复数值的li

    >>> adjust_repeat_data(['a', 'b', 'a', 'c'])
    ['a', 'b', 'a+', 'c']
    """
    res = []
    values = set()
    for x in li:
        while x in values:
            x += suffix
            # print(x)
        res.append(x)
        values.add(x)

    return res


if __name__ == '__main__':
    TicToc.process_time(f'{dformat()}启动准备共用时')
    tictoc = TicToc(__file__)

    # create_account_df()
    # demo_sqlengine()

    tictoc.toc()
