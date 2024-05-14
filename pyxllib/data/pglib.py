#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/06/09 16:26

"""
针对PostgreSQL封装的工具
"""

import os
import sys

from pyxllib.prog.pupil import check_install_package
from pyxllib.file.specialist import XlPath, ensure_localdir

# 使用binary安装，好像就不需要自己配置依赖了
# if sys.platform == 'win32':
#     # windows系统中pg的默认安装位置
#     # https://www.yuque.com/xlpr/pyxllib/install_psycopg
#     pgdll_dir = XlPath('C:/Program Files/PostgreSQL/14/bin')
#     if not pgdll_dir.is_dir():  # 换一个自定义目录继续检查
#         pgdll_dir = XlPath.userdir() / '.xlpr/pgdll'
#     if not pgdll_dir.is_dir():
#         print('没有PostgreSQL对应的dll文件，自动下载：')
#         ensure_localdir(pgdll_dir, 'https://xmutpriu.com/download/pgdll_v14.zip')
#     os.environ['PATH'] += ';' + pgdll_dir.as_posix()

check_install_package('psycopg', 'psycopg[binary]')

import io
from collections import Counter
import json
import json
import textwrap
import datetime
import re

from tqdm import tqdm

import psycopg
import psycopg.rows

from pyxllib.prog.newbie import round_int
from pyxllib.prog.pupil import utc_now, utc_timestamp
from pyxllib.prog.specialist import XlOsEnv
from pyxllib.algo.pupil import ValuesStat2
from pyxllib.file.specialist import get_etag
from pyxllib.data.sqlite import SqlBase, SqlBuilder


class Connection(psycopg.Connection, SqlBase):

    def __init__(self, *args, **kwargs):
        psycopg.Connection.__init__(self, *args, **kwargs)
        SqlBase.__init__(self, *args, **kwargs)

    def __1_库(self):
        pass

    def get_db_activities(self):
        """
        检索当前数据库的活动信息。
        """
        sql = """
        SELECT pid, datname, usename, state, query, age(now(), query_start) AS "query_age"
        FROM pg_stat_activity
        WHERE state = 'active'
        """
        return self.exec2dict(sql).fetchall()

    def __2_表格(self):
        pass

    def get_table_names(self):
        # cmd = "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'"
        cmd = "SELECT tablename FROM pg_tables WHERE schemaname='public'"
        return [x[0] for x in self.execute(cmd)]

    def has_table(self, table_name, schemaname='public'):
        query = ["SELECT EXISTS (SELECT FROM pg_tables ",
                 f"WHERE schemaname='{schemaname}' AND tablename='{table_name}')"]
        res = self.execute(' '.join(query))
        return res.fetchone()[0]

    def get_column_names(self, table_name):
        """ 【查】表格有哪些字段
        """
        cmd = f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}'"
        return self.exec2col(cmd)

    def ensure_column(self, table_name, col_name, *args, comment=None, **kwargs):
        super(Connection, self).ensure_column(table_name, col_name, *args, **kwargs)
        if comment:
            self.set_column_comment(table_name, col_name, comment)

    def set_column_comment(self, table_name, col_name, comment):
        # 这里comment按%s填充会报错。好像是psycopg库的问题，这种情况因为没有表格字段参照类型，会不知道要把comment转成字符串。
        # 然后使用py的f字符串填充的话，要避免comment有引号等特殊字符，就用了特殊的转义标记$zyzf$
        # 暂时找不到更优雅的方式~~ 而且这个问题，可能以后其他特殊的sql语句也会遇到。。。
        self.execute(f"COMMENT ON COLUMN {table_name}.{col_name} IS $zyzf${comment}$zyzf$")
        self.commit()

    def reset_table_item_id(self, table_name, item_id_name=None, counter_name=None):
        """ 重置表格的数据的id值，也重置计数器

        :param item_id_name: 表格的自增id字段名，有一套我自己风格的自动推算算法
        :param counter_name: 计数器字段名，如果有的话，也会重置
        """
        # 1 重置数据的编号
        if item_id_name is None:
            m = re.match(r'(.+?)(_table)?$', table_name)
            item_id_name = m.group(1) + '_id'

        sql = f"""WITH cte AS (
    SELECT {item_id_name}, ROW_NUMBER() OVER (ORDER BY {item_id_name}) AS new_{item_id_name}
    FROM {table_name}
)
UPDATE {table_name}
SET {item_id_name} = cte.new_{item_id_name}
FROM cte
WHERE {table_name}.{item_id_name} = cte.{item_id_name}"""
        self.execute(sql)  # todo 这种sql写法好像偶尔有bug会出问题
        self.commit()

        # 2 重置计数器
        if counter_name is None:
            counter_name = f'{table_name}_{item_id_name}_seq'

        # 找到目前最大的id值
        max_id = self.exec2one(f'SELECT MAX({item_id_name}) FROM {table_name}')
        self.execute(f'ALTER SEQUENCE {counter_name} RESTART WITH {max_id + 1}')
        self.commit()

    def __3_execute(self):
        pass

    def exec_nametuple(self, *args, **kwargs):
        cur = self.cursor(row_factory=psycopg.rows.namedtuple_row)
        data = cur.execute(*args, **kwargs)
        # cur.close()
        return data

    def exec2dict(self, *args, **kwargs):
        cur = self.cursor(row_factory=psycopg.rows.dict_row)
        data = cur.execute(*args, **kwargs)
        # cur.close()
        return data

    def exec2dict_batch(self, sql, batch_size=1000, **kwargs):
        """ 分批返回数据的版本

        :return:
            第1个值，是一个迭代器，看起来仍然能一条一条返回，实际后台是按照batch_size打包获取的
            第2个值，是数据总数
        """
        if not isinstance(sql, SqlBuilder):
            raise ValueError('暂时只能搭配SQLBuilder使用')

        num = self.exec2one(sql.build_count())
        cur = self.exec2dict(sql.build_select(), **kwargs)

        def yield_row():
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                yield from rows

        return yield_row(), num

    exec_dict = exec2dict

    def __4_数据类型(self):
        pass

    @classmethod
    def cvt_type(cls, val):
        if isinstance(val, (dict, list)):
            val = json.dumps(val, ensure_ascii=False)
        # 注意list数组类型读、写都会自动适配py
        return val

    @classmethod
    def autotype(cls, val):
        if isinstance(val, str):
            return 'text'
        elif isinstance(val, int):
            return 'int4'  # int2、int8
        elif isinstance(val, bool):
            return 'boolean'
        elif isinstance(val, float):
            return 'float4'
        elif isinstance(val, dict):
            return 'jsonb'
        else:  # 其他list等类型，可以用json.dumps或str转文本存储
            return 'text'

    def __5_增删改查(self):
        pass

    def insert_row(self, table_name, cols, *, on_conflict='DO NOTHING', commit=False):
        """ 【增】插入新数据

        :param dict cols: 用字典表示的要插入的值
        :param on_conflict: 如果已存在的处理策略
            DO NOTHING，跳过不处理
            REPLACE，这是个特殊标记，会转为主键冲突也仍然更新所有值
            (id) DO UPDATE SET host_name=EXCLUDED.host_name, nick_name='abc'
                也可以写复杂的处理算法规则，详见 http://postgres.cn/docs/12/sql-insert.html
                比如这里是插入的id重复的话，就把host_name替换掉，还可以指定nick_name替换为'abc'
                注意前面的(id)是必须要输入的

        注意：有个常见需求，是想插入后返回对应的id，但是这样就需要知道这张表自增的id字段名
            以及还是很难获得插入后的id值，可以默认刚插入的id是最大的，但是这样并不安全，有风险
            建议还是外部自己先计算全表最大的id值，自己实现自增，就能知道插入的这条数据的id了
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

        self.commit_base(commit, query, params)

    def insert_rows(self, table_name, keys, ls, *, on_conflict='DO NOTHING', commit=False):
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

        self.commit_base(commit, query, params)

    def __6_高级统计(self):
        pass

    def get_column_valuesstat(self, table_name, column, percentile_count=5,
                              filter_condition=None, by_data=False):
        """ 获得指定表格的某个字段的统计特征ValuesStat2对象

        :param table_name: 表名
        :param column: 用于计算统计数据的字段名
        :param percentile_count: 分位数的数量，例如 3 表示只计算中位数
        :param by_data: 是否获得原始数据
            默认只获得统计特征，不获得原始数据
        """

        def init_from_db_data():
            sql = SqlBuilder(table_name)
            if filter_condition:
                sql.where(filter_condition)
            values = self.exec2col(sql.build_select(column))
            if data_type == 'numeric':
                values = [x and float(x) for x in values]
            return ValuesStat2(raw_values=values, data_type=data_type)

        def init_from_db():
            # 1 构建基础的 SQL 查询
            sql = SqlBuilder(table_name)
            sql.select("COUNT(*) AS total_count")
            sql.select(f"COUNT({column}) AS non_null_count")
            sql.select(f"MIN({column}) AS min_value")
            sql.select(f"MAX({column}) AS max_value")
            if data_type and 'timestamp' in data_type:
                percentile_type = 'PERCENTILE_DISC'
                # todo 其实时间类也可以"泛化"一种平均值、标准差算法的，这需要获取全量数据，然后自己计算
            elif data_type == 'text':
                percentile_type = 'PERCENTILE_DISC'
            else:  # 默认是正常的数值类型
                sql.select(f"SUM({column}) AS total_sum")
                sql.select(f"AVG({column}) AS average")
                sql.select(f"STDDEV({column}) AS standard_deviation")
                percentile_type = 'PERCENTILE_CONT'

            percentiles = []
            # 根据分位点的数量动态添加分位数计算
            if percentile_count > 2:
                step = 1 / (percentile_count - 1)
                percentiles = [(i * step) for i in range(1, percentile_count - 1)]
                for p in percentiles:
                    sql.select(f"{percentile_type}({p:.2f}) WITHIN GROUP (ORDER BY {column}) "
                               f"AS percentile_{int(p * 100)}")

            if filter_condition:
                sql.where(filter_condition)

            row = self.exec2dict(sql.build_select()).fetchone()

            # 2 统计展示
            x = ValuesStat2(data_type=data_type)
            x.raw_n = row['total_count']
            x.n = row['non_null_count']
            if not x.n:
                return x

            x.sum = row.get('total_sum', None)
            x.mean = row.get('average', None)
            x.std = row.get('standard_deviation', None)

            # 如果计算了分位数，填充相应属性
            x.dist = [row['min_value']] + [row[f"percentile_{int(p * 100)}"] for p in percentiles] + [row['max_value']]
            if data_type == 'numeric':
                x.dist = [float(x) for x in x.dist]

            return x

        data_type = self.get_column_data_type(table_name, column)
        if by_data:
            vs = init_from_db_data()
        else:
            vs = init_from_db()

        return vs


"""
【关于为什么XlprDb要和pglib合一个文件】
好处
1、代码集中管理，方便开发效率

坏处 - 对治
1、导入代码多，效率低
    - 导入代码其实很快的，不差这一点点
2、仅用简单功能的时候，会引入多余复杂功能
    - 如果单拉一个xlprdb.py文件，依然不能头部import复杂库，因为XlprDb也存在需要精简使用的场景
        既然都要有精简的部分，干脆都把代码弄到一起得了
        然后复杂包都在成员函数里单独import
"""


class XlprDb(Connection):
    """ xlpr统一集中管理的一个数据库

    为了一些基础的数据库功能操作干净，尽量不要在全局导入过于复杂的包，可以在每个特定功能里导特定包
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seckey = ''

    @classmethod
    def set_conninfo(cls, conninfo, seckey=''):
        """ 提前将登录信息加密写到环境变量中，这样代码中就没有明文的账号密码信息

        :param conninfo:
        :param seckey: 如果要获得数据库里较重要的密码等信息，需要配置key值，否则默认可以不设

        使用后，需要重启IDE重新加载环境变量
        并且本句明文代码需要删除
        """
        # TODO 目前只设一个账号，后续可以扩展支持多个账号指定配置
        # conninfo = 'postgresql://postgres:yourpassword@172.16.170.110/xlpr'
        return XlOsEnv.persist_set('XlprDbAccount', {'conninfo': conninfo, 'seckey': seckey}, encoding=True)

    @classmethod
    def connect(cls, conninfo='', seckey='', *,
                autocommit=False, row_factory=None, context=None, **kwargs) -> 'XlprDb':
        """ 因为要标记 -> XlprDb，IDE才会识别出类别，有自动补全功能
        但在类中写@classmethod，无法标记 -> XlprDb，所以就放外面单独写一个方法了
        """
        d = XlOsEnv.get('XlprDbAccount', decoding=True)
        if conninfo == '':
            conninfo = d['conninfo']
        if seckey == '' and isinstance(d, dict) and 'seckey' in d:
            seckey = d['seckey']
        # 注意这里获取的是XlprDb类型
        con = super(XlprDb, cls).connect(conninfo, autocommit=autocommit, row_factory=row_factory, context=context,
                                         **kwargs)
        con.seckey = seckey
        return con

    @classmethod
    def connect2(cls, name, passwd, database=None, ip_list=None, **kwargs):
        """ 简化的登录方式，并且自带优先尝试局域网连接，再公网连接

        一般用在jupyter等对启动速度没有太高要求的场合，因为不断尝试localhost等需要耗费不少时间
        """
        if database is None:
            database = name

        xldb = None
        if ip_list is None:
            ip_list = ['localhost', '172.16.170.136', 'xmutpriu.com']
        for ip in ip_list:
            try:
                xldb = cls.connect(f'postgresql://{name}:{passwd}@{ip}/{database}', **kwargs)
                break
            except psycopg.OperationalError:
                pass

        return xldb

    def __1_hosts相关数据表操作(self):
        pass

    def update_host(self, host_name, accounts=None, **kwargs):
        """ 更新一台服务器的信息

        :param dict accounts: 账号信息，记得要列全，比如
            {'root': '123456', 'chenkunze': '654321'}
        """
        if not self.execute('SELECT EXISTS (SELECT FROM hosts WHERE host_name=%s)', (host_name,)).fetchone()[0]:
            self.insert_row('hosts', {'host_name': host_name})
        if kwargs:
            self.update_row('hosts', kwargs, {'host_name': host_name})
        if accounts:
            self.execute('UPDATE hosts SET accounts=pgp_sym_encrypt(%s, %s) WHERE host_name=%s',
                         (json.dumps(accounts, ensure_ascii=False), self.seckey, host_name))
        self.commit()

    def set_host_account(self, host_name, user_name, passwd):
        """ 设置某台服务器的一个账号密码

        >> xldb.set_host_account('titan2', 'chenkunze', '123456')

        """
        # 读取旧的账号密码字典数据
        d = self.execute("SELECT pgp_sym_decrypt(accounts, %s)::jsonb FROM hosts WHERE host_name=%s",
                         (self.seckey, host_name)).fetchone()[0]
        # 修改字典数据
        d[user_name] = str(passwd)
        # 将新的字典数据写回数据库
        self.execute('UPDATE hosts SET accounts=pgp_sym_encrypt(%s, %s) WHERE host_name=%s',
                     (json.dumps(d, ensure_ascii=False), self.seckey, host_name))
        self.commit()

    def login_ssh(self, host_name, user_name, map_path=None, **kwargs) -> 'XlSSHClient':
        """ 通过数据库里的服务器数据记录，直接登录服务器 """
        from pyxllib.ext.unixlib import XlSSHClient

        if host_name.startswith('g_'):
            host_ip = self.execute("SELECT host_ip FROM hosts WHERE host_name='xlpr0'").fetchone()[0]
            pw, port = self.execute('SELECT (pgp_sym_decrypt(accounts, %s)::jsonb)[%s]::text, frpc_port'
                                    ' FROM hosts WHERE host_name=%s',
                                    (self.seckey, user_name, host_name[2:])).fetchone()
        else:
            port = 22
            host_ip, pw = self.execute('SELECT host_ip, (pgp_sym_decrypt(accounts, %s)::jsonb)[%s]::text'
                                       ' FROM hosts WHERE host_name=%s',
                                       (self.seckey, user_name, host_name)).fetchone()

        if map_path is None:
            if sys.platform == 'win32':
                map_path = {'C:/': '/'}
            else:
                map_path = {'/': '/'}

        return XlSSHClient(host_ip, user_name, pw[1:-1], port=port, map_path=map_path, **kwargs)

    def __2_xlapi相关数据表操作(self):
        """
        files，存储二进制文件的表
            etag，文件、二进制对应的etag值
            meta，可以存储不同数据类型一些特殊的属性，比如图片可以存储高、宽
                但因为用户使用中，为了提高速度，以及减少PIL等依赖，执行中不做计算
                可以其他途径使用定期脚本自动处理
        xlapi，底层api调用的记录统计
            input，所有输入的参数
                mode，使用的算法接口
                etag，涉及到大文件数据的，以打包后的文件的etag做记录
            output，运行结果
            elapse_ms，调用函数的用时，不含后处理转换格式的时间
        xlserver，服务端记录的使用情况
        """
        pass

    def insert_row2files(self, buffer, *, etag=None, **kwargs):
        """

        为了运算效率考虑，除了etag需要用于去重，是必填字段
        其他字段默认先不填充计算
        """
        # 1 已经有的不重复存储
        if etag is None:
            etag = get_etag(buffer)

        res = self.execute('SELECT etag FROM files WHERE etag=%s', (etag,)).fetchone()
        if res:  # 已经做过记录的，不再重复记录
            return

        # 2 没有的图，做个备份
        kwargs['etag'] = etag
        kwargs['data'] = buffer
        kwargs['fsize_kb'] = round_int(len(buffer) / 1024)

        self.insert_row('files', kwargs)
        self.commit()

    def get_xlapi_record(self, **input):
        res = self.execute('SELECT id, output FROM xlapi WHERE input=%s', (self.cvt_type(input),)).fetchone()
        if res:
            _id, output = res
            output['xlapi_id'] = _id
            return output

    def insert_row2xlapi(self, input, output, elapse_ms, *, on_conflict='(input) DO NOTHING'):
        """ 往数据库记录当前操作内容

        :return: 这个函数比较特殊，需要返回插入的条目的id值
        """
        if on_conflict == 'REPLACE':
            on_conflict = "(input) DO UPDATE " \
                          "SET output=EXCLUDED.output, elapse_ms=EXCLUDED.elapse_ms, update_time=EXCLUDED.update_time"

        input = json.dumps(input, ensure_ascii=False)
        self.insert_row('xlapi', {'input': input, 'output': output,
                                  'elapse_ms': elapse_ms, 'update_time': utc_timestamp(8)},
                        on_conflict=on_conflict)
        self.commit()
        return self.execute('SELECT id FROM xlapi WHERE input=%s', (input,)).fetchone()[0]

    def insert_row2xlserver(self, request, xlapi_id=0, **kwargs):
        kw = {'remote_addr': request.headers.get('X-Real-IP', request.remote_addr),
              'route': '/'.join(request.base_url.split('/')[3:]),
              'update_time': utc_timestamp(8),
              'xlapi_id': xlapi_id}
        kw.update(kwargs)
        print(kw)  # 监控谁在用api
        self.insert_row('xlserver', kw, commit=True)

    def __3_host_trace相关可视化(self):
        """ TODO dbview 改名 host_trace """
        pass

    def __dbtool(self):
        pass

    def record_host_usage(self, cpu=True, gpu=True, disk=False):
        """ 记录服务器各种状况，存储到PG数据库

        TODO 并行处理
        TODO 功能还可以增加：gpu显卡温度、硬盘读写速率检查、网络上传下载带宽
        """
        # 1 服务器列表
        host_names = self.exec2col('SELECT host_name FROM hosts WHERE id > 1 ORDER BY id')
        host_cpu_gb = {h: v for h, v in self.execute('SELECT host_name, cpu_gb FROM hosts')}

        # 2 去所有服务器取使用情况
        for i, host_name in enumerate(host_names, start=1):
            print('-' * 20, i, host_name, '-' * 20)
            try:
                ssh = self.login_ssh(host_name, 'root', relogin=5, relogin_interval=0.2)
                status = {}
                if cpu:
                    data = ssh.check_cpu_usage(print_mode=True)
                    status['cpu'] = {k: v[0] for k, v in data.items()}
                    status['cpu_memory'] = {k: round(v[1] * host_cpu_gb[host_name] / 100, 2) for k, v in data.items()}
                if gpu:
                    status['gpu_memory'] = ssh.check_gpu_usage(print_mode=True)
                if disk:
                    # 检查磁盘空间会很慢，如果超时可以跳过。（设置超时6小时）
                    status['disk_memory'] = ssh.check_disk_usage(print_mode=True, timeout=60 * 60 * 6)
            except Exception as e:
                status = {'error': f'{str(type(e))[8:-2]}: {e}'}
                print(status)

            if status:
                self.insert_row('host_trace',
                                {'host_name': host_name, 'status': status, 'update_time': utc_timestamp(8)},
                                commit=True)
            print()

    def _get_host_trace_total(self, mode, title, yaxis_name, date_trunc, recent, host_attr):
        # CREATE INDEX ON gpu_trace (update_time);  -- update_time最好建个索引
        ls = self.execute(textwrap.dedent(f"""\
        WITH cte1 AS (  -- 筛选近期内的数据，并且时间做trunc处理
            SELECT host_name, (status)['{mode}'], date_trunc('{date_trunc}', update_time) ttime
            FROM host_trace WHERE update_time > %s AND (status ? '{mode}')
        ), cte2 AS (  -- 每个时间里每个服务器的多条记录取平均
            SELECT ttime, cte1.host_name, jsonb_div(jsonb_deep_sum(status), count(*)) status
            FROM cte1 GROUP BY ttime, cte1.host_name
        )  -- 接上每台服务器显存总值，并且分组得到每个时刻总情况
        SELECT ttime, {host_attr}, jsonb_deep_sum(status)
        FROM cte2 JOIN hosts ON cte2.host_name = hosts.host_name
        GROUP BY ttime ORDER BY ttime"""), ((utc_now(8) - recent).isoformat(timespec='seconds'),)).fetchall()
        return self._create_stack_chart(title, ls, yaxis_name=yaxis_name)

    def _get_host_trace_per_host(self, hostname, mode, title, yaxis_name, date_trunc, recent, host_attr):
        ls = self.execute(textwrap.dedent(f"""\
        WITH cte1 AS (
            SELECT (status)['{mode}'], date_trunc('{date_trunc}', update_time) ttime
            FROM host_trace WHERE host_name='{hostname}' AND update_time > %s AND (status ? '{mode}')
        ), cte2 AS (
            SELECT ttime, jsonb_div(jsonb_deep_sum(status), count(*)) status
            FROM cte1 GROUP BY ttime
        )
        SELECT ttime, {host_attr}, jsonb_deep_sum(status)
        FROM cte2 JOIN hosts ON hosts.host_name='{hostname}'
        GROUP BY ttime ORDER BY ttime"""), ((utc_now(8) - recent).isoformat(timespec='seconds'),)).fetchall()
        if not ls:  # 有的服务器可能数据是空的
            return '', 0
        return self._create_stack_chart(title, ls, yaxis_name=yaxis_name)

    def _create_stack_chart(self, title, ls, *, yaxis_name=''):
        """ 创建展示表

        :param title: 表格标题
        :param list ls: n*3，第1列是时间，第2列是总值，第3列是每个用户具体的数据
        """
        from pyecharts.charts import Line

        map_user_name = {}
        for ks, v in self.execute('SELECT account_names, name FROM users'):
            for k in ks:
                map_user_name[k] = v

        # 1 计算涉及的所有用户以及使用总量
        all_users_usaged = Counter()
        last_time = None
        for x in ls:
            hours = 0 if last_time is None else ((x[0] - last_time).total_seconds() / 3600)
            last_time = x[0]
            for k, v in x[2].items():
                if k == '_total':
                    continue
                all_users_usaged[map_user_name.get(k, k)] += v * hours

        # ls里的姓名也要跟着更新
        for i, x in enumerate(ls):
            ct = Counter()
            for k, v in x[2].items():
                ct[map_user_name.get(k, k)] += v
            ls[i] = (x[0], x[1], ct)

        # 2 转图表可视化
        def to_list(values):
            return [(x[0], v) for x, v in zip(ls, values)]

        def pretty_val(v):
            return round_int(v) if v > 100 else round(v, 2)

        chart = Line()
        chart.set_title(title)
        chart.options['xAxis'][0].update({'min': ls[0][0], 'type': 'time',
                                          # 'minInterval': 3600 * 1000 * 24,
                                          'name': '时间', 'nameGap': 50, 'nameLocation': 'middle'})
        chart.options['yAxis'][0].update({'name': yaxis_name, 'nameGap': 50, 'nameLocation': 'middle'})
        # 目前是比较暴力的方法调整排版，后续要研究是不是能更自动灵活些
        chart.options['legend'][0].update({'top': '6%', 'icon': 'pin'})
        chart.options['grid'] = [{'top': 55 + len(all_users_usaged) * 4, 'containLabel': True}]
        chart.options['tooltip'].opts.update({'axisPointer': {'type': 'cross'}, 'trigger': 'item'})

        chart.add_series(f'total{pretty_val(ls[0][1]):g}', to_list([x[1] for x in ls]), areaStyle={})
        for user, usaged in all_users_usaged.most_common():
            usaged = usaged / ((ls[-1][0] - ls[0][0]).total_seconds() / 3600 + 1e-9)
            chart.add_series(f'{user}{pretty_val(usaged):g}',
                             to_list([x[2].get(user, 0) for x in ls]),
                             areaStyle={}, stack='Total', emphasis={'focus': 'series'})

        return '<body>' + chart.render_embed() + '</body>', sum(all_users_usaged.values())

    def dbview_cpu(self, recent=datetime.timedelta(days=1), date_trunc='hour'):
        from pyxllib.data.echarts import render_echart_html

        args = ['CPU核心数（比如4核显示是400%）', date_trunc, recent, 'sum(hosts.cpu_number)*100']

        htmltexts = [
            '<a target="_blank" href="https://www.yuque.com/xlpr/data/hnpb2g?singleDoc#"> 《服务器监控》工具使用文档 </a>']
        res = self._get_host_trace_total('cpu', 'XLPR服务器 CPU 使用近况', *args)
        htmltexts.append(res[0])

        hosts = self.execute('SELECT host_name, nick_name FROM hosts WHERE gpu_gb > 0').fetchall()
        host_stats = []
        for i, (hn, nick_name) in enumerate(hosts, start=1):
            name = f'{hn}，{nick_name}' if nick_name else hn
            host_stats.append(self._get_host_trace_per_host(hn, 'cpu', f'{name}', *args))
        host_stats.sort(key=lambda x: -x[1])  # 按使用量，从多到少排序
        htmltexts += [x[0] for x in host_stats]

        self.commit()

        h = render_echart_html('cpu', body='<br/>'.join(htmltexts))
        return h

    def dbview_cpu_memory(self, recent=datetime.timedelta(days=1), date_trunc='hour'):
        from pyxllib.data.echarts import render_echart_html

        args = ['内存（单位：GB）', date_trunc, recent, 'sum(hosts.cpu_gb)']

        htmltexts = [
            '<a target="_blank" href="https://www.yuque.com/xlpr/data/hnpb2g?singleDoc#"> 《服务器监控》工具使用文档 </a>']
        res = self._get_host_trace_total('cpu_memory', 'XLPR服务器 内存 使用近况', *args)
        htmltexts.append(res[0])

        hosts = self.execute('SELECT host_name, nick_name FROM hosts WHERE gpu_gb > 0').fetchall()
        host_stats = []
        for i, (hn, nick_name) in enumerate(hosts, start=1):
            name = f'{hn}，{nick_name}' if nick_name else hn
            host_stats.append(self._get_host_trace_per_host(hn, 'cpu_memory', f'{name}', *args))
        host_stats.sort(key=lambda x: -x[1])  # 按使用量，从多到少排序
        htmltexts += [x[0] for x in host_stats]

        self.commit()

        h = render_echart_html('cpu_memory', body='<br/>'.join(htmltexts))
        return h

    def dbview_disk_memory(self, recent=datetime.timedelta(days=30), date_trunc='day'):
        """ 查看disk硬盘使用近况
        """
        from pyxllib.data.echarts import render_echart_html

        args = ['硬盘（单位：GB）', date_trunc, recent, 'sum(hosts.disk_gb)']

        htmltexts = [
            '<a target="_blank" href="https://www.yuque.com/xlpr/data/hnpb2g?singleDoc#"> 《服务器监控》工具使用文档 </a>']
        res = self._get_host_trace_total('disk_memory', 'XLPR服务器 DISK硬盘 使用近况', *args)
        htmltexts.append(res[0])
        htmltexts.append('注：xlpr4（四卡）服务器使用du计算/home大小有问题，未统计在列<br/>')

        hosts = self.execute('SELECT host_name, nick_name FROM hosts WHERE gpu_gb > 0').fetchall()
        host_stats = []
        for i, (hn, nick_name) in enumerate(hosts, start=1):
            name = f'{hn}，{nick_name}' if nick_name else hn
            host_stats.append(self._get_host_trace_per_host(hn, 'disk_memory', f'{name}', *args))
        host_stats.sort(key=lambda x: -x[1])  # 按使用量，从多到少排序
        htmltexts += [x[0] for x in host_stats]

        self.commit()

        h = render_echart_html('disk_memory', body='<br/>'.join(htmltexts))
        return h

    def dbview_gpu_memory(self, recent=datetime.timedelta(days=30), date_trunc='day'):
        """ 查看gpu近使用近况

        TODO 这里有可以并行处理的地方，但这个方法不是很重要，不需要特地去做加速占用cpu资源
        """
        from pyxllib.data.echarts import render_echart_html

        args = ['显存（单位：GB）', date_trunc, recent, 'sum(hosts.gpu_gb)']

        htmltexts = [
            '<a target="_blank" href="https://www.yuque.com/xlpr/data/hnpb2g?singleDoc#"> 《服务器监控》工具使用文档 </a>']
        res = self._get_host_trace_total('gpu_memory', 'XLPR八台服务器 GPU显存 使用近况', *args)
        htmltexts.append(res[0])

        hosts = self.execute('SELECT host_name, nick_name FROM hosts WHERE gpu_gb > 0').fetchall()
        host_stats = []
        for i, (hn, nick_name) in enumerate(hosts, start=1):
            name = f'{hn}，{nick_name}' if nick_name else hn
            host_stats.append(self._get_host_trace_per_host(hn, 'gpu_memory', f'{name}', *args))
        host_stats.sort(key=lambda x: -x[1])  # 按使用量，从多到少排序
        htmltexts += [x[0] for x in host_stats]

        self.commit()

        h = render_echart_html('gpu_memory', body='<br/>'.join(htmltexts))
        return h

    def __4_一些数据更新操作(self):
        """ 比如一些扩展字段，在调用api的时候为了性能并没有进行计算，则可以这里补充更新 """

    def update_files_dhash(self, print_mode=True):
        """ 更新files表中的dhash字段值 """
        from pyxllib.cv.imhash import dhash
        from pyxllib.xlcv import xlpil

        # 获取总图片数
        total_count = self.execute("SELECT COUNT(*) FROM files WHERE dhash IS NULL").fetchall()[0][0]

        # 执行查询语句，获取dhash为NULL的记录

        # 初始化进度条
        progress_bar = tqdm(total=total_count, disable=not print_mode)

        while True:
            result = self.execute("SELECT id, data FROM files WHERE dhash IS NULL LIMIT 5")
            if not result:
                break
            for row in result:
                file_id = row[0]
                im = xlpil.read_from_buffer(row[1])
                computed_dhash = str(dhash(im))
                self.update_row('files', {'dhash': computed_dhash}, {'id': file_id})
                progress_bar.update(1)
            self.commit()

    def append_history(self, table_name, where, backup_keys, *,
                       can_merge=None,
                       update_time=None,
                       commit=False):
        """ 为表格添加历史记录，请确保这个表有一个jsonb格式的historys字段

        这里每次都会对关键字段进行全量备份，没有进行高级的优化。
        所以只适用于一些历史记录功能场景。更复杂的还是需要另外自己定制。

        :param table_name: 表名
        :param where: 要记录的id的规则，请确保筛选后记录是唯一的
        :param backup_keys: 需要备份的字段名
        :param can_merge: 在某些情况下，history不需要非常冗余地记录，可以给定与上一条合并的规则
            def can_merge(last, now):
                "last是上一条字典记录，now是当前要记录的字典数据，
                返回True，则用now替换last，并不新增记录"
                ...

        :param update_time: 更新时间，如果不指定则使用当前时间
        """
        # 1 获得历史记录
        ops = ' AND '.join([f'{k}=%s' for k in where.keys()])
        historys = self.exec2one(f'SELECT historys FROM {table_name} WHERE {ops}', list(where.values())) or []
        if historys:
            status1 = historys[-1]
        else:
            status1 = {}

        # 2 获得新记录
        if update_time is None:
            update_time = utc_timestamp()
        status2 = self.exec2dict(f'SELECT {",".join(backup_keys)} FROM {table_name} WHERE {ops}',
                                 list(where.values())).fetchone()
        status2['update_time'] = update_time

        # 3 添加历史记录
        if can_merge is None:
            def can_merge(status1, status2):
                for k in backup_keys:
                    if status1.get(k) != status2.get(k):
                        return False
                return True

        if historys and can_merge(status1, status2):
            historys[-1] = status2
        else:
            historys.append(status2)

        self.update_row(table_name, {'historys': historys}, where, commit=commit)
