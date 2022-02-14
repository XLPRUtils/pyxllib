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

from pyxllib.prog.newbie import RunOnlyOnce
from pyxllib.prog.pupil import EnchantBase, EnchantCvt
from pyxllib.debug.specialist import TicToc, browser


class EnchantChart(EnchantBase):

    @classmethod
    @RunOnlyOnce
    def enchant(cls):
        from pyecharts.charts.chart import Chart
        names = cls.check_enchant_names([Chart])
        cls._enchant(Chart, names)

    @staticmethod
    def set_title(self, title):
        self.set_global_opts(title_opts=pyecharts.options.TitleOpts(title=title))


EnchantChart.enchant()


class EnchantBar(EnchantBase):

    @classmethod
    @RunOnlyOnce
    def enchant(cls):
        from pyecharts.charts import Bar
        names = cls.check_enchant_names([Bar])
        cls._enchant(Bar, names, EnchantCvt.staticmethod2classmethod)

    @staticmethod
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

    @staticmethod
    def from_list(cls, yaxis, xaxis=None, *, title=None):
        return cls.from_dict({'value': list(yaxis)}, xaxis=xaxis, title=title)


EnchantBar.enchant()

if __name__ == '__main__':
    with TicToc(__name__):
        pass
