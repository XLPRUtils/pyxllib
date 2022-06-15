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

from collections import Counter
import json
import json
import textwrap

import psycopg
import psycopg.rows

from pyxllib.prog.pupil import utc_timestamp
from pyxllib.prog.specialist import XlOsEnv
from pyxllib.file.specialist import XlPath
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
        XlOsEnv.persist_set('XlprDbAccount', {'conninfo': conninfo, 'seckey': seckey}, encoding=True)

    def update_host(self, host_name, accounts=None, **kwargs):
        """ 更新一台服务器的信息

        :param dict accounts: 账号信息，记得要列全，比如
            {'root': '123456', 'chenkunze': '654321'}
        """
        if not self.execute('SELECT EXISTS (SELECT FROM hosts WHERE host_name=%s)', (host_name,)).fetchone()[0]:
            self.insert_row('hosts', {'host_name': host_name})
        if kwargs:
            self.update('hosts', kwargs, {'host_name': host_name})
        if accounts:
            self.execute('UPDATE hosts SET accounts=pgp_sym_encrypt(%s, %s) WHERE host_name=%s',
                         (json.dumps(accounts, ensure_ascii=False), self.seckey, host_name))
        self.commit()

    def get_host_account(self, host_name, user_name):
        pw = self.execute(f'SELECT (pgp_sym_decrypt(accounts, %s)::jsonb)[%s]::text FROM hosts WHERE host_name=%s',
                          (self.seckey, user_name, host_name)).fetchone()[0]
        return pw[1:-1]

    def __已有表格封装的一些操作(self):
        pass

    def insert_gpu_trace(self, host_name, total_memory, user_usage):
        data = (host_name, round(total_memory, 2), json.dumps(user_usage, ensure_ascii=False), utc_timestamp(8))
        self.execute("""INSERT INTO gpu_trace(host_name, total_memory, used_memory, update_time)"""
                     """VALUES(%s, %s, %s, %s)""", data)

    def insert_disk_trace(self, host_name, total_memory, user_usage):
        data = (host_name, round(total_memory, 2), json.dumps(user_usage, ensure_ascii=False), utc_timestamp(8))
        self.execute("""INSERT INTO disk_trace(host_name, total_memory, used_memory, update_time)"""
                     """VALUES(%s, %s, %s, %s)""", data)

    def __dbtool(self):
        pass

    def dbview_gpustat(self, days=7):
        """ 查看gpu近一周使用情况

        TODO 这里有可以并行处理的地方，但这个方法不是很重要，不需要特地去做加速占用cpu资源
        """
        from pyxllib.data.echarts import Line, render_echart_html, get_render_body

        map_user_name = {x[0]: x[1] for x in self.execute('SELECT account_name, name FROM users')}

        def create_chart(ls, title):
            # 1 计算涉及的所有用户以及使用总量
            all_users_usaged = Counter()
            last_time = None
            for x in ls:
                hours = 0 if last_time is None else ((x[0] - last_time).total_seconds() / 3600)
                last_time = x[0]
                for k, v in x[2].items():
                    all_users_usaged[k] += v * hours

            # print(all_users_usaged.most_common())

            # 2 转图表可视化
            def to_list(values):
                return [(x[0], v) for x, v in zip(ls, values)]

            chart = Line()
            chart.set_title(title)
            chart.options['xAxis'][0].update({'min': ls[0][0], 'type': 'time',
                                              # 'minInterval': 3600 * 1000 * 24,
                                              'name': '时间', 'nameGap': 50, 'nameLocation': 'middle'})
            chart.options['yAxis'][0].update({'name': '显存（单位：GB）', 'nameGap': 50, 'nameLocation': 'middle'})
            # 目前是比较暴力的方法调整排版，后续要研究是不是能更自动灵活些
            chart.options['legend'][0].update({'top': '6%', 'icon': 'pin'})
            chart.options['grid'] = [{'top': 100, 'containLabel': True}]
            chart.options['tooltip'].opts.update({'axisPointer': {'type': 'cross'}, 'trigger': 'item'})

            chart.add_series(f'total{ls[0][1]:.2f}', to_list([x[1] for x in ls]), areaStyle={})
            for user, usaged in all_users_usaged.most_common():
                usaged = usaged / ((ls[-1][0] - ls[0][0]).total_seconds() / 3600)
                chart.add_series(f'{map_user_name.get(user, user)}{usaged:.2f}',
                                 to_list([x[2].get(user, 0) for x in ls]),
                                 areaStyle={}, stack='Total', emphasis={'focus': 'series'})

            return get_render_body(chart), sum(all_users_usaged.values())

        def get_total(title):
            # CREATE INDEX ON gpu_trace (update_time);  -- update_time最好建个索引
            ls = self.execute(textwrap.dedent(f"""\
            WITH cte1 AS (  -- 筛选一周内的数据，并且时间只精确到小时
                SELECT host_name, total_memory, used_memory, date_trunc('hour', update_time) htime
                FROM gpu_trace
                WHERE update_time > (date_trunc('day', now() - interval '{days} days' + interval '8 hours'))
            ), cte2 AS (  -- 每小时每个服务器只保留一条记录
                SELECT DISTINCT ON (htime, host_name) *
                FROM cte1
            )
            SELECT htime, sum(total_memory) total_memory, jsonb_deep_sum(used_memory) used_memory
            FROM cte2 GROUP BY htime HAVING (COUNT(host_name)=8 OR htime > '2022-06-12')
            ORDER BY htime""")).fetchall()
            # 6月12日以前，脚本鲁棒性不够，只有完整统计了8台的才展示，6月12日后有兼容了，没取到的服务器就是宕机了，可以显示剩余服务器总情况
            return create_chart(ls, title)

        def get_host(hostname, title):
            ls = self.execute(textwrap.dedent(f"""\
            SELECT update_time, total_memory, used_memory
            FROM gpu_trace
            WHERE host_name='{hostname}' AND
                update_time > (date_trunc('day', now() - interval '{days} days' + interval '8 hours'))
            ORDER BY update_time""")).fetchall()
            return create_chart(ls, title)

        htmltexts = []
        htmltexts.append(get_total('XLPR八台服务器GPU显存资源最近使用情况')[0])

        hosts = self.execute('SELECT host_name, nick_name FROM hosts WHERE id > 1').fetchall()
        self.commit()
        host_stats = []
        for i, (hn, nick_name) in enumerate(hosts, start=1):
            name = f'{hn}，{nick_name}' if nick_name else hn
            host_stats.append(get_host(hn, f'{name}'))
        host_stats.sort(key=lambda x: -x[1])  # 按使用量，从多到少排序
        htmltexts += [x[0] for x in host_stats]

        h = render_echart_html('gpustat', body='<br/>'.join(htmltexts))
        return h


def connect_xlprdb(conninfo='', seckey='', *, autocommit=False, row_factory=None, context=None, **kwargs) -> XlprDb:
    """ 因为要标记 -> XlprDb，IDE才会识别出类别，有自动补全功能
    但在类中写@classmethod，无法标记 -> XlprDb，所以就放外面单独写一个方法了
    """
    d = XlOsEnv.get('XlprDbAccount', decoding=True)
    if conninfo == '':
        conninfo = d['conninfo']
    if seckey == '':
        seckey = d['seckey']
    con = super(XlprDb, XlprDb).connect(conninfo,
                                        autocommit=autocommit, row_factory=row_factory,
                                        context=context, **kwargs)
    con.seckey = seckey
    return con
