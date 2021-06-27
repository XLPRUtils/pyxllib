#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/04 09:14


from pyxllib.xlai import *


def test_labeldata():
    os.chdir(r'D:\home\datasets\RealEstate2020\temp')

    # 1 初始化
    ld = LabelmeDataset(r'D:\home\datasets\RealEstate2020\temp')
    # 升级字典
    # ld.to_labelattrs()

    # 2 转coco
    gt_dict = ld.to_coco_gt_dict()
    # pprint.pprint(gt_dict)

    # 3 转表格
    # ld.to_excel('test.xlsx')

    # 4 coco对象
    cgd = CocoGtData(gt_dict)

    # 5 coco对象再转回labelme，礼尚往来
    ld2 = cgd.to_labelme(r'D:\home\datasets\RealEstate2020\temp2')
    ld2.writes()

    # 6 ld2再转回coco，梅开二度
    gt_dict = ld2.to_coco_gt_dict()
    pprint.pprint(gt_dict)


def test_tqdm():
    """ 下载情况下的进度显示怎么做 """
    with tqdm(unit="B", unit_scale=True, miniters=1, desc='a.test', leave=True) as t:
        t.total = 10000000
        for i in range(10):
            t.update(1000000)  # type: ignore
            time.sleep(1)


class ArithmeticPerf(PerfTest):
    """ 可能数值比较小，这个实验不太明显吧

    >> ArithmeticPerf().perf(number=10000, repeat=100)
    add 用时(秒) 总和: 1.266	均值标准差: 0.127±0.019	总数: 10	最小值: 0.094	最大值: 0.167 运行结果：121
    div 用时(秒) 总和: 1.281	均值标准差: 0.128±0.017	总数: 10	最小值: 0.097	最大值: 0.150 运行结果：1.75
    mul 用时(秒) 总和: 1.121	均值标准差: 0.112±0.012	总数: 10	最小值: 0.095	最大值: 0.131 运行结果：3388
    sub 用时(秒) 总和: 1.214	均值标准差: 0.121±0.015	总数: 10	最小值: 0.102	最大值: 0.150 运行结果：33
    """

    def perf_add(self):
        return 77 + 44

    def perf_sub(self):
        return 77 - 44

    def perf_mul(self):
        return 77 * 44

    def perf_div(self):
        return 77 / 44


class CallPerf(PerfTest):
    """ 奕本纠结嵌套太多产生的效率问题

    >> CallPerf().perf(number=100000, repeat=100)
    func1 用时(秒) 总和: 7.947	均值标准差: 0.079±0.019	总数: 100	最小值: 0.063	最大值: 0.172 运行结果：bc
    func2 用时(秒) 总和: 12.581	均值标准差: 0.126±0.038	总数: 100	最小值: 0.093	最大值: 0.268 运行结果：bc
    """

    def perf_func1(self):
        def a():
            def b():
                def c():
                    return 'abc'.replace('a', '')

                return c()

            return b()

        return a()

    def perf_func2(self):
        return re.sub(r'a', '', 'abc')


def 文件结构(dirpath, depth=0):
    from humanfriendly import format_size

    d = Dir(dirpath)
    print('\t' * depth + f'☐{d.name}, {format_size(d.size, binary=True)}')
    for dd in d.select_dirs('*'):
        # 不处理的一些目录
        if dd.name in ('__pycache__',):
            continue
        文件结构(dd, depth + 1)
    for f in d.select_files('*'):
        print('\t' * (depth + 1) + f'{f.name}, {format_size(f.size, binary=True)}')


if __name__ == '__main__':
    with TicToc(__name__):
        # test_labeldata()

        # 文件结构(r'C:\ProgramData\Anaconda3\Lib\site-packages\iopath')
        # 文件结构(r'D:\slns\detectron2\detectron2')

        pass
