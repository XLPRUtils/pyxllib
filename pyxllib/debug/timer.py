#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/31


import math
import timeit


import numpy


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
            mean, std = numpy.mean(self.data), numpy.std(self.data)
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


def demo_timer():
    """该函数也可以用来测电脑性能
    代码中附带的示例结果是我在自己小米笔记本上的测试结果
    Intel（R） Core（TM） i7-10510U CPU@ 1.80GHz 2.30 GHz，15G 64位
    """
    from pyxllib.debug.dprint import dformat, dprint
    import numpy

    print('1、普通用法（循环5*1000万次用时）')
    timer = Timer('循环')
    timer.start()
    for _ in range(5):
        for _ in range(10 ** 7):
            pass
    timer.stop()
    timer.report()
    # 循环 用时: 0.727s

    print('2、循环多轮计时')
    timer = Timer('自己算均值标准差耗时')

    # 数据量=200是大概的临界值，往下自己算快，往上用numpy算快
    # 临界量时，每万次计时需要0.45秒。其实整体都很快影响不大，所以Timer最终统一采用numpy来运算。
    data = list(range(10))*20

    for _ in range(5):
        timer.start()  # 必须明确指定每次的 开始、结束 时间
        for _ in range(10 ** 4):
            n, sum_ = len(data), sum(data)
            mean1 = sum_ / n
            std1 = math.sqrt((sum([(x - mean1) ** 2 for x in data]) / n))
        timer.stop()  # 每轮结束时标记
    timer.report()
    # 自己算均值标准差耗时 总耗时: 2.214s	均值标准差: 0.443±0.008s	总数: 5	最小值: 0.435s	最大值: 0.459s
    dprint(mean1, std1)
    # [05]timer.py/97: mean1<float>=4.5    std1<float>=2.8722813232690143

    print('3、with上下文用法')
    with Timer('使用numpy算均值标准差耗时') as t:
        for _ in range(5):
            t.start()
            for _ in range(10 ** 4):
                mean2, std2 = numpy.mean(data), numpy.std(data)
            t.stop()
    # 主要就是结束会自动report，其他没什么太大差别
    # 使用numpy算均值标准差耗时 总耗时: 2.282s	均值标准差: 0.456±0.015s	总数: 5	最小值: 0.442s	最大值: 0.483s
    dprint(mean2, std2)
    # [05]timer.py/109: mean2<numpy.float64>=4.5    std2<numpy.float64>=2.8722813232690143

    print('4、可以配合dformat输出定位信息')
    with Timer(dformat()) as t:
        for _ in range(5):
            t.start()
            for _ in range(10 ** 6):
                pass
            t.stop()
    # [04]timer.py/113:      总耗时: 0.096s	均值标准差: 0.019±0.002s	总数: 5	最小值: 0.018s	最大值: 0.023s




if __name__ == '__main__':
    demo_timer()
