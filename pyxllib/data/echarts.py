#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/02/09 11:14


"""
Apache ECharts: https://echarts.apache.org/zh/index.html
python版：pyechats的封装
"""

from pyxllib.prog.pupil import check_install_package

check_install_package('pyecharts')

import pyecharts
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from pyecharts.charts import Bar, Line
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
        self._append_legend(name)
        # self._append_legend(name, is_selected=True)

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


class XlBar(Bar):

    @classmethod
    def from_dict(cls, yaxis, xaxis=None, *, title=None):
        """ 查看一个数据的条形图

        :param dict yaxis: 列表数据，或者字典name: values表示的多组数据
        """
        b = cls()

        if xaxis is None:
            xaxis = list(range(max([len(v) for v in yaxis.values()])))

        b.add_xaxis(xaxis)

        for k, v in yaxis.items():
            b.add_yaxis(k, v)

        if title:
            b.set_title(title)

        return b

    @classmethod
    def from_list(cls, yaxis, xaxis=None, *, title=None):
        """
        >> browser(pyecharts.charts.Bar.from_list(numbers))
        """
        return cls.from_dict({'value': list(yaxis)}, xaxis=xaxis, title=title)


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
