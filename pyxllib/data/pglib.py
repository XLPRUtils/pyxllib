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

if sys.platform == 'win32':
    # https://www.yuque.com/xlpr/pyxllib/install_psycopg

    # windows系统中pg的默认安装位置
    # TODO 更好的办法还是检查没有后，自动下载，这个得等新版的后端，从我们自己服务器下载
    os.environ['PATH'] += r";C:/Program Files/PostgreSQL/14/bin"

check_install_package('psycopg')

import io
from collections import Counter
import json
import json
import textwrap
import datetime
import time

import psycopg
import psycopg.rows

from pyxllib.prog.newbie import round_int
from pyxllib.prog.pupil import utc_now, utc_timestamp
from pyxllib.prog.specialist import XlOsEnv
from pyxllib.file.specialist import XlPath, get_etag
from pyxllib.data.sqlite import SqlBase


class Connection(psycopg.Connection, SqlBase):
    def exec_col(self, query, params=None, *, prepare=None, binary=False):
        """ 获得第1列的值，注意这个方法跟select_col很像，但更泛用，优先推荐使用exec_col

        >> self.exec_col('SELECT id FROM skindata')
        """
        for row in self.execute(query, params, prepare=prepare, binary=binary):
            yield row[0]

    def exec_nametuple(self, *args, **kwargs):
        cur = self.cursor(row_factory=psycopg.rows.namedtuple_row)
        data = cur.execute(*args, **kwargs)
        # cur.close()
        return data

    def exec_dict(self, *args, **kwargs):
        cur = self.cursor(row_factory=psycopg.rows.dict_row)
        data = cur.execute(*args, **kwargs)
        # cur.close()
        return data

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

        >> xldb.update('xlapi', {'input': d}, {'id': x['id']})
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

    @classmethod
    def connect(cls, conninfo='', seckey='', *,
                autocommit=False, row_factory=None, context=None, **kwargs) -> 'XlprDb':
        """ 因为要标记 -> XlprDb，IDE才会识别出类别，有自动补全功能
        但在类中写@classmethod，无法标记 -> XlprDb，所以就放外面单独写一个方法了
        """
        d = XlOsEnv.get('XlprDbAccount', decoding=True)
        if conninfo == '':
            conninfo = d['conninfo']
        if seckey == '':
            seckey = d['seckey']
        con = super(XlprDb, cls).connect(conninfo,
                                         autocommit=autocommit, row_factory=row_factory,
                                         context=context, **kwargs)
        con.seckey = seckey
        return con

    def __1_xlapi相关数据表操作(self):
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
        if res:
            return

        # 2 没有的图，做个备份
        kwargs['etag'] = etag
        kwargs['data'] = buffer
        kwargs['fsize_kb'] = round_int(len(buffer) / 1024)

        self.insert_row('files', kwargs)
        self.commit()

    def get_xlapi_record(self, **input):
        if 'options' in input and not input['options']:
            del input['options']
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

        return self.execute('SELECT id FROM xlapi WHERE input=%s', (input,)).fetchone()[0]

    def insert_row2xlserver(self, request, xlapi_id=0):
        kw = {'remote_addr': request.remote_addr,
              'token': request.headers.get('Token', None),
              'route': '/'.join(request.base_url.split('/')[3:]),
              'update_time': utc_timestamp(8),
              'xlapi_id': xlapi_id}
        print(kw)  # 监控谁在用api
        self.insert_row('xlserver', kw)

    def run_api(self, func, buffer, options=None, *,
                mode_name=None,
                use_exists_record=True,
                update_record=True,
                save_buffer_threshold_size=4 * 1024 ** 2):
        """ 配合database数据库的情况下，调用API功能

        :param func: 被封装的带执行的api函数
        :param buffer: 图片数据
        :param options: 执行api功能的配套采纳数
        :param mode_name: 可以指定存入数据库的功能名，默认用func的名称
        :param use_exists_record: 如果数据库里已有记录，是否直接复用
        :param update_record: 新结果是否记录到数据库
        :param save_buffer_threshold_size: buffer小余多少，才存储进数据库

        """

        # 1 预处理，参数标准化
        if options is None:
            options = {}
        options = {k: options[k] for k in sorted(options.keys())}  # 对参数进行排序，方便去重

        # 2 调百度识别接口
        if use_exists_record or update_record:  # 有开数据库，并且复用和更新至少有项开启了
            if mode_name is None:
                mode_name = func.__name__
            # 如果数据库里有处理过的记录，直接引用
            im_etag = get_etag(buffer)
            if use_exists_record:
                res = self.get_xlapi_record(mode=mode_name, etag=im_etag, options=options)
            else:
                res = None

            # 否则调用百度的接口识别
            # TODO 这里使用协程逻辑最合理但配置麻烦，需要func底层等做协程的适配支持
            #   使用多线程测试了并没有更快，也发现主要耗时是post，数据库不会花太多时间，就先不改动了
            #   等以后数据库大了，看运行是否会慢，可以再测试是否有必要弄协程
            if res is None or 'error_code' in res:
                tt = time.time()
                res = func(buffer, options)
                elapse_ms = round_int(1000 * (time.time() - tt))

                if len(buffer) < save_buffer_threshold_size:
                    self.insert_row2files(buffer, etag=im_etag, name='.jpg')
                if update_record:
                    input = {'mode': mode_name, 'image': im_etag}
                    if options:
                        input['options'] = options
                    xlapi_id = self.insert_row2xlapi(input, res, elapse_ms, on_conflict='REPLACE')
                    res['xlapi_id'] = xlapi_id

            if 'log_id' in res:  # 有xlapi_id的标记，就不用百度原本的log_id了
                del res['log_id']
        else:
            res = func(buffer, options)

        return res

    def __2_host_trace相关可视化(self):
        """ TODO dbview 改名 host_trace """
        pass

    def __dbtool(self):
        pass

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
        from pyxllib.data.echarts import Line, get_render_body

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

        return get_render_body(chart), sum(all_users_usaged.values())

    def dbview_cpu(self, recent=datetime.timedelta(days=1), date_trunc='hour'):
        from pyxllib.data.echarts import render_echart_html

        args = ['CPU核心数（比如4核显示是400%）', date_trunc, recent, 'sum(hosts.cpu_number)*100']

        htmltexts = []
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

        htmltexts = []
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

        htmltexts = []
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

        htmltexts = []
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
