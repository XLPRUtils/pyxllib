#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/30 20:18


import math
import timeit


class Timer:
    """分析性能用的计时器类，支持with语法调用
    必须显示地指明每一轮的start()和end()，否则会报错
    """

    def __init__(self, title=''):
        """
        :param title: 计时器名称
        """
        # 不同的平台应该使用的计时器不同，这个直接用timeit中的配置最好
        self.default_timer = timeit.default_timer
        # 标题
        self.title = title
        self.data = []
        self.start_clock = float('nan')

    def start(self):
        self.start_clock = self.default_timer()

    def stop(self):
        self.data.append(self.default_timer() - self.start_clock)

    def report(self, msg=''):
        """ 报告目前性能统计情况
        """
        msg = f'{self.title} {msg}'
        n = len(self.data)

        if n > 1:  # 有多轮，则应该输出些参考统计指标
            # numpy有标准差等公式，但这是debuglib底层库，不想依赖太多第三方库，所以手动实现
            sum_ = sum(self.data)
            mean = sum_ / n
            std = math.sqrt((sum([(x - mean) ** 2 for x in self.data]) / n))
            li = [f'{msg}总耗时: {sum_:.3f}s', f'均值标准差: {mean:.3f}±{std:.3f}s',
                  f'总数: {n}', f'最小值: {min(self.data):.3f}s', f'最大值: {max(self.data):.3f}s']
            print('\t'.join(li))
        elif n == 1:  # 只有一轮，则简单地输出耗时即可
            sum_ = sum(self.data)
            print(f'{msg} 用时: {sum_:.3f}s')
        else:  # 没有统计数据，则补充执行一次stop后汇报
            print(f'{msg} 暂无计时信息')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.report()