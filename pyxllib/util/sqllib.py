#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/06/03 09:52


import subprocess


import pandas as pd


try:
    import sqlalchemy
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'sqlalchemy'])
    subprocess.run(['pip3', 'install', 'mysqlclient'])
    import sqlalchemy


from pyxllib.debug.pytictoc import TicToc
from pyxllib.debug.dprint import dformat, dprint
from pyxllib.debug.pathlib_ import Path


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
        # 1、读取地址、账号信息
        if alias:
            if account_file_path is None:
                account_file_path = (Path(__file__).parent / 'sqllibaccount.pkl')
            record = Path(account_file_path).read().loc[alias]  # 从文件读取账号信息
            user, passwd, host, port = record.user, record.passwd, record.host, record.port
        # 2、'数据库类型+数据库驱动名称://用户名:口令@机器地址:端口号/数据库名'
        address = f'mysql+mysqldb://{user}:{passwd}@{host}:{port}/{database}?charset=utf8mb4'
        # 3、存储成员
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
        # 1、合并sql命令
        if isinstance(sql, str): sql = [sql]
        sql = '\n'.join(sql)

        # 含有 % 的特殊符号要转义
        # import sqlalchemy
        # sql = sqlalchemy.text(sql)

        # 2、解析结果
        res = pd.read_sql(sql, self.engine, index_col=index_col, coerce_float=coerce_float, params=params,
                          parse_dates=parse_dates, columns=columns, chunksize=chunksize)
        return res

    def execute(self, statement, *multiparams, **params):
        """本质上就是sqlalchemy.con.execute的封装

        可以这样使用：
        hsql.execute('UPDATE spell_check SET count=:count WHERE old=:old AND new=:new',
                      count=count[0] + add, old=old, new=new)
        """
        # 1、解析sql命令
        if isinstance(statement, str): statement = [statement]
        statement = '\n'.join(statement)
        statement = sqlalchemy.text(statement)

        # 2、如果设置了getdf参数
        res = self.engine.execute(statement, *multiparams, **params)
        return res


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


if __name__ == '__main__':
    TicToc.process_time(f'{dformat()}启动准备共用时')
    tictoc = TicToc(__file__)

    # create_account_df()
    demo_sqlengine()

    tictoc.toc()

