#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/06/10 15:46

from collections import Counter
import textwrap

from pyxllib.prog.specialist import XlOsEnv
from pyxllib.file.specialist import XlPath

from pyxllib.data.pglib import Connection


class XlprDb(Connection):
    """ xlpr统一集中管理的一个数据库 """

    @classmethod
    def set_conninfo(cls, conninfo):
        """ 提前将登录信息加密写到环境变量中，这样代码中就没有明文的账号密码信息

        使用后，需要重启IDE重新加载环境变量
        并且本句明文代码需要删除
        """
        # TODO 目前只设一个账号，后续可以扩展支持多个账号指定配置
        # conninfo = 'postgresql://postgres:yourpassword@172.16.170.110/xlpr'
        XlOsEnv.persist_set('XlprDbAccount', conninfo, encoding=True)

    @classmethod
    def connect(cls, conninfo='', *, autocommit=False, row_factory=None, context=None, **kwargs):
        if conninfo == '':
            conninfo = XlOsEnv.get('XlprDbAccount', decoding=True)
        con = super(XlprDb, cls).connect(conninfo,
                                         autocommit=autocommit, row_factory=row_factory,
                                         context=context, **kwargs)
        return con

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
            ls = self.execute(textwrap.dedent(f"""\
            -- CREATE INDEX ON gpu_trace (update_time);  -- update_time最好建个索引
            WITH cte1 AS (  -- 筛选一周内的数据，并且时间只精确到小时
                SELECT host_name, total_memory, used_memory, date_trunc('hour', update_time) htime
                FROM gpu_trace
                WHERE update_time > (date_trunc('day', now() - interval '{days} days' + interval '8 hours'))
            ), cte2 AS (  -- 每小时每个服务器只保留一条记录
                SELECT DISTINCT ON (htime, host_name) *
                FROM cte1
            ) -- count>6，是允许一台宕机的情况下仍能统计展示
            SELECT htime, sum(total_memory) total_memory, jsonb_deep_sum(used_memory) used_memory
            FROM cte2 GROUP BY htime HAVING COUNT(host_name)>6;""")).fetchall()
            return create_chart(ls, title)

        def get_host(hostname, title):
            ls = self.execute(textwrap.dedent(f"""\
            SELECT update_time, total_memory, used_memory
            FROM gpu_trace
            WHERE host_name='{hostname}' AND
                update_time > (date_trunc('day', now() - interval '{days} days' + interval '8 hours'));""")).fetchall()
            return create_chart(ls, title)

        htmltexts = []
        htmltexts.append(get_total('XLPR八台服务器GPU显存资源最近使用情况')[0])

        hosts = self.execute('SELECT host_name, nick_name FROM hosts WHERE id > 1').fetchall()
        host_stats = []
        for i, (hn, nick_name) in enumerate(hosts, start=1):
            name = f'{hn}，{nick_name}' if nick_name else hn
            host_stats.append(get_host(hn, f'{name}'))
        host_stats.sort(key=lambda x: -x[1])  # 按使用量，从多到少排序
        htmltexts += [x[0] for x in host_stats]

        h = render_echart_html('gpustat', body='<br/>'.join(htmltexts))
        return h
