#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/03/16 09:19


"""一些python通用功能的性能测试
    虽然其实大部分场合其实都是相通的
    有时候test测试代码，其实也是演示如何使用的demo

demo：示例代码，注重演示
debug：调试代码，注重分析自己代码功能是否有bug
test：测试代码，注重分析功能稳定性
perf：性能测试，注重分析代码的运行效率
"""

from pyxllib.xl import *

____stdlib = """
标准库相关
"""


def test_re():
    """ 正则re模块相关功能测试
    """
    # 190103周四
    # py的正则[ ]语法，可以用连字符-匹配一个区间内的字符，
    # 例如数字0-9（你是不是蠢，不会用\d么），还有a-z、A-Z（\w），甚至①-⑩，但是一-十，注意'四'匹配不到
    dprint(re.sub(r'[一-十]', '', '一二三四五六七八九十'))
    # [05]demolib.py/98: re.sub(r'[一-十]', '', '一二三四五六七八九十')<str>='四'

    # 200319周四14:11，匹配顺序与内容有关，先出现的先匹配，而与正则里or语法参数顺序无关
    print(re.findall(r'(<(a|b)>.*?</\2>)', '<a><b></b></a>'))
    print(re.findall(r'(<(b|a)>.*?</\2>)', '<a><b></b></a>'))
    # 结果都是： [('<a><b></b></a>', 'a')]
    # TODO 200323周一17:22，其实是模式不够复杂，在特殊场景下，可选条件的前后顺序是有影响的


def perf_concurrent():
    import time
    import concurrent.futures

    def func():
        s = 0
        for i in range(1000):
            for j in range(1000):
                s += j ** 5
        return s

    start = time.time()
    for i in range(5):
        func()
    print(f'单线程 During Time: {time.time() - start:.3f} s')

    start = time.time()
    executor = concurrent.futures.ThreadPoolExecutor(4)
    for i in range(5):
        executor.submit(func)
    executor.shutdown()
    print(f'多线程 During Time: {time.time() - start:.3f} s')


____pyxllib = """
pyxllib库相关
"""


def demo_timer():
    """ 该函数也可以用来测电脑性能

    代码中附带的示例结果是我在自己小米笔记本上的测试结果
    Intel（R） Core（TM） i7-10510U CPU@ 1.80GHz 2.30 GHz，15G 64位
    """
    import math
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
    data = list(range(10)) * 20

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


def demo_dprint():
    """这里演示dprint常用功能
    """
    # 1 查看程序是否运行到某个位置
    dprint()
    # [05]dprint.py/169:      意思：这是堆栈的第5层，所运行的位置是 dprint.py文件的第169行

    # 2 查看变量、表达式的 '<类型>' 和 ':值'
    a, b, s = 1, 2, 'ab'
    dprint(a, b, a ^ b, s * 2)
    # [05]dprint.py/174: a<int>=1    b<int>=2    a ^ b<int>=3    s*2<str>='abab'

    # 3 异常警告
    b = 0
    if b:
        c = a / b
    else:
        c = 0
        dprint(a, b, c)  # b=0不能作为除数，c默认值暂按0处理
    # [05]dprint.py/183: a<int>=1    b<int>=0    c<int>=0    # b=0不能作为除数，c默认值暂按0处理

    # 4 如果想在其他地方使用dprint的格式内容，可以调底层dformat函数实现
    with TicToc(dformat(fmt='[{depth:02}]{fullfilename}/{lineno}: {argmsg}')):
        for _ in range(10 ** 7):
            pass
    # [04]D:\slns\pyxllib\pyxllib\debug\pupil.py/187:  0.173 秒.


def _test_getfile_speed():
    """
    遍历D盘所有文件(205066个) 用时0.65秒
    遍历D盘所有tex文件(7796个) 用时0.95秒
    有筛选遍历D盘所有文件(193161个) 用时1.19秒
    有筛选遍历D盘所有tex文件(4464个) 用时1.22秒
        + EnsureContent： 3.18秒，用list存储所有文本要 310 MB 开销，转str拼接差不多也是这个值
        + re.sub(r'\$.*?\$', r'', s)： 4.48秒
    """
    timer = Timer(start_now=True)
    ls = list(getfiles(r'D:\\'))
    timer.stop_and_report(f'遍历D盘所有文件({len(ls)}个)')

    timer = Timer(start_now=True)
    ls = list(getfiles(r'D:\\', '.tex'))
    timer.stop_and_report(f'遍历D盘所有tex文件({len(ls)}个)')

    timer = Timer(start_now=True)
    ls = list(mygetfiles(r'D:\\'))
    timer.stop_and_report(f'有筛选遍历D盘所有文件({len(ls)}个)')

    timer = Timer(start_now=True)
    ls = list(mygetfiles(r'D:\\', '.tex'))
    timer.stop_and_report(f'有筛选遍历D盘所有tex文件({len(ls)}个)')


____perf = """
"""


def check_os_status():
    """ 检查系统当前运行状态 """
    import time
    import psutil

    brief_str = []  # 简化显示

    # 1
    print(f'1 逻辑cpu数量：{psutil.cpu_count()}  \t{psutil.cpu_percent(1) / 100:-3.0%}')
    brief_str.append(f'{psutil.cpu_count()}({psutil.cpu_percent(1) / 100:.0%})')

    # 2
    m = psutil.virtual_memory()
    print(f'2 内存大小：{m.total / (1024 ** 3):.0f} GB \t{m.percent / 100:-3.0%}')
    brief_str.append(f'{m.total / (1024 ** 3):.0f}GB({m.percent / 100:.0%})')

    # 3
    disks = psutil.disk_partitions()
    used, total = 0, 0
    for d in disks:
        msg = psutil.disk_usage(d.mountpoint)
        used += msg.used
        total += msg.total
    used /= 1024 ** 4
    total /= 1024 ** 4
    print(f'3 磁盘空间：{total:.2f} TB\t{used / total:-3.0%}')
    brief_str.append(f'{total:.2f}TB({used / total:.0%})')
    print('/'.join(brief_str))

    # 4
    msg1 = psutil.disk_io_counters()
    sec = 5  # 统计几秒内的读写状态
    time.sleep(sec)
    msg2 = psutil.disk_io_counters()
    print(f'4 读写：', end='')
    for name in ['read_bytes', 'write_bytes']:
        value = getattr(msg2, name) - getattr(msg1, name)
        if name.endswith('_count'):
            print(f'{name}={value / sec:.0f} /s', end=' ')
        else:
            print(f'{name}={value / (1024 ** 2) / sec:.0f}MB/s', end=' ')
    print()

    # 5
    msg1 = psutil.net_io_counters()
    sec = 5  # 统计几秒内的读写状态
    time.sleep(sec)
    msg2 = psutil.net_io_counters()
    print(f'5 网络：', end='')
    for name in ['bytes_sent', 'bytes_recv']:
        value = getattr(msg2, name) - getattr(msg1, name)
        print(f'{name}={value / (1024 ** 2) / sec:.0f}MB/s', end=' ')
    print()
