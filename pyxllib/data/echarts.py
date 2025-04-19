#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/02/09 11:14


"""
Apache ECharts: https://echarts.apache.org/zh/index.html
python版：pyechats的封装
"""

# import types

import pyecharts
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ChartType
from pyecharts import types
from pyecharts.charts import Bar, Line, Radar
from pyecharts.charts.chart import Chart

from pyxllib.prog.pupil import inject_members
from pyxllib.prog.specialist import TicToc
from pyxllib.file.specialist import XlPath


class XlChart(Chart):
    def set_title(self, title):
        self.set_global_opts(title_opts=pyecharts.options.TitleOpts(title=title))

    def add_series(self, name, data, *, type=None, color=None, labels=None,
                   **kwargs):
        """ 垃圾pyecharts，毁我青春~~

        很多图x都不是等间距的，pyecharts处理不等间距x好像有很大的不兼容问题
        需要手动添加x、y坐标位置

        :param list|tuple labels: 直接提供现成的文本标签

        """
        if type is None:
            type = self.__class__.__name__.lower()

        if labels:
            s = [[x[0][0], x[1]] for x in zip(data, labels)]
            fmt = JsCode(f"function(x){{var m = new Map({s}); return m.get(x.value[0]);}}")
            if 'label' not in kwargs:
                kwargs['label'] = opts.LabelOpts(is_show=True, formatter=fmt)
            elif isinstance(kwargs['label'], opts.LabelOpts):
                kwargs['label'].opts['formatter'] = fmt
            else:
                kwargs['label']['show'] = True
                kwargs['label']['formatter'] = fmt

        self._append_color(color)
        # self._append_legend(name, is_selected=True)
        self._append_legend(name)

        self.options.get('series').append(
            {
                'type': type,
                'name': name,
                'data': data,
                **kwargs,
            }
        )
        return self


inject_members(XlChart, Chart)


class XlLine(Line):
    @classmethod
    def from_dict(cls, yaxis, xaxis=None, *, title=None,
                  xaxis_name=None, yaxis_name=None, **kwargs):
        """ 查看一个数据的折线图

        :param list|dict yaxis: 列表数据，或者字典name: values表示的多组数据
        """
        c = cls()

        if isinstance(yaxis, (list, tuple)):
            yaxis = {'value': yaxis}
        if xaxis is None:
            xaxis = list(range(max([len(v) for v in yaxis.values()])))
        c.add_xaxis(xaxis)
        for k, v in yaxis.items():
            c.add_yaxis(k, v)

        configs = {
            'tooltip_opts': opts.TooltipOpts(trigger="axis"),
            **kwargs,
        }
        c.set_global_opts(tooltip_opts=opts.TooltipOpts(trigger="axis"))
        if title:
            configs['title_opts'] = opts.TitleOpts(title=title)
        if xaxis_name:
            configs['xaxis_opts'] = opts.AxisOpts(name=xaxis_name, axislabel_opts=opts.LabelOpts(rotate=45, interval=0))
        if yaxis_name:
            configs['yaxis_opts'] = opts.AxisOpts(name=yaxis_name)
        c.set_global_opts(**configs)
        return c


class XlBar(Bar):

    @classmethod
    def from_dict(cls, yaxis, xaxis=None, *, title=None,
                  xaxis_name=None, yaxis_name=None, **kwargs):
        """ 查看一个数据的条形图

        :param list|dict yaxis: 列表数据，或者字典name: values表示的多组数据
        """
        c = cls()

        if isinstance(yaxis, (list, tuple)):
            yaxis = {'value': yaxis}
        if xaxis is None:
            xaxis = list(range(max([len(v) for v in yaxis.values()])))
        c.add_xaxis(xaxis)
        for k, v in yaxis.items():
            c.add_yaxis(k, v)

        configs = {
            'tooltip_opts': opts.TooltipOpts(trigger="axis"),
            **kwargs,
        }
        c.set_global_opts(tooltip_opts=opts.TooltipOpts(trigger="axis"))
        if title:
            configs['title_opts'] = opts.TitleOpts(title=title)
        if xaxis_name:
            configs['xaxis_opts'] = opts.AxisOpts(name=xaxis_name)
        if yaxis_name:
            configs['yaxis_opts'] = opts.AxisOpts(name=yaxis_name)
        c.set_global_opts(**configs)
        return c

    @classmethod
    def from_list(cls, yaxis, xaxis=None, *, title=None):
        """
        >> browser(pyecharts.charts.Bar.from_list(numbers))
        """
        return cls.from_dict({'value': list(yaxis)}, xaxis=xaxis, title=title)

    @classmethod
    def from_data_split_into_groups(cls, data, groups, *, title=None):
        """根据给定的组数自动拆分数据并生成条形图
        :param list data: 数据清单
        :param int groups: 要拆分成的组数
        """
        # 找到最大值和最小值
        min_val, max_val = min(data), max(data)

        # 计算间隔
        interval = (max_val - min_val) / groups

        # 分组和标签
        group_counts = [0] * groups
        labels = []
        # todo 如果数据量特别大，这里应该排序后，再用特殊方法计算分组
        for value in data:
            index = min(int((value - min_val) / interval), groups - 1)
            group_counts[index] += 1

        for i in range(groups):
            labels.append(f"{min_val + interval * i:.2f}-{min_val + interval * (i + 1):.2f}")
        # t = cls.from_dict({'value': group_counts}, xaxis=labels, title=title)

        return cls.from_dict({'value': group_counts}, xaxis=labels, title=title)


class XlRadar(Radar):
    """ 雷达图 """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_idx = 0

    def add(
            self,
            series_name: str,
            data: types.Sequence[types.Union[opts.RadarItem, dict]],
            *,
            label_opts=None,
            color: types.Optional[str] = None,
            linestyle_opts=None,
            **kwargs
    ):
        """ 标准库(2.0.5版)的雷达图颜色渲染有问题，这里要增加一个修正过程 """
        if label_opts is None:
            label_opts = opts.LabelOpts(is_show=False)

        if linestyle_opts is None:
            linestyle_opts = opts.LineStyleOpts(color=self.colors[self.color_idx % len(self.colors)])
            self.color_idx += 1
        elif linestyle_opts.get('color') is None:
            linestyle_opts.update(color=self.colors[self.color_idx % len(self.colors)])
            self.color_idx += 1

        if color is None:
            color = linestyle_opts.get('color')

        return super(XlRadar, self).add(series_name, data,
                                        label_opts=label_opts,
                                        color=color,
                                        linestyle_opts=linestyle_opts,
                                        **kwargs)


inject_members(XlBar, Bar)


def render_echart_html(title='Awesome-pyecharts', body=''):
    from pyxllib.text.xmllib import get_jinja_template
    return get_jinja_template('echart_base.html').render(title=title, body=body)


# 绘制帕累托累计图
def draw_pareto_chart(data, accuracy=0.1, *, title='帕累托累积权重', value_unit_type='K'):
    from pyxllib.algo.stat import pareto_accumulate
    pts, labels = pareto_accumulate(data, accuracy=accuracy, value_unit_type=value_unit_type)
    x = Line()
    x.add_series(title, pts, labels=labels, label={'position': 'right'})
    x.set_global_opts(
        # x轴末尾要故意撑大一些，不然有部分内容会显示不全
        xaxis_opts=opts.AxisOpts(name='条目数', max_=int(float(f'{pts[-1][0] * 1.2:.2g}'))),
        yaxis_opts=opts.AxisOpts(name='累积和')
    )
    return x


if __name__ == '__main__':
    with TicToc(__name__):
        pass
