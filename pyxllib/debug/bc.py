#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/01 18:13

import copy
import pandas as pd

from pyxllib.algo.group import intersection_split
from pyxllib.file.dir import File, Dir
from pyxllib.debug.type import prettifystr
from pyxllib.debug.browser import Explorer


class BCompare(Explorer):
    def __init__(self, app='BCompare', shell=False):
        super().__init__(app, shell)

    @classmethod
    def to_bcompare_files(cls, *args, files=None):
        """ 这个需要一次性获得所有的数据，才适合分析整体上要怎么获取对应的多个文件

        :param files: 每个arg对应的文件名，默认按 'left'、'right', 'base' 来生成
            也可以输入一个list[str]，表示多个args依次对应的文件名
            filename的长度可以跟args不一致，多的不用，少的自动生成
        """
        # 1 如果oldfile和newfile都是dict、set、list、tuple，则使用特殊方法文本化
        #   如果两个都是list，不应该提取key后比较，所以限制第1个类型必须是dict或set，然后第二个类型可以适当放宽条件
        # if not oldfile: oldfile = str(oldfile)
        if len(args) > 1 and isinstance(args[0], (dict, set)) and isinstance(args[1], (dict, set, list, tuple)):
            args = copy.copy(list(args))
            t = [prettifystr(li) for li in intersection_split(args[0], args[1])]
            args[0] = f'【共有部分】，{t[0]}\n\n【独有部分】，{t[1]}'
            args[1] = f'【共有部分】，{t[2]}\n\n【独有部分】，{t[3]}'

        # 2 参数对齐
        if not isinstance(files, (list, tuple)):
            files = [files]
        if len(files) < len(args):
            files += [None] * (len(args) - len(files))
        ref_names = ['left', 'right', 'base']

        # 3 将每个参数转成一个文件
        new_args = []
        default_suffix = None
        for i, arg in enumerate(args):
            f = File.safe_init(arg)
            if f:  # 是文件对象，且存在
                new_args.append(f)
                if not default_suffix:
                    default_suffix = f.suffix
            # elif isinstance(f, File):  # 文本内容也可能生成合法的伪路径，既然找不到还是统一按字符串对比差异好
            #     # 是文件对象，但不存在 -> 报错
            #     raise FileNotFoundError(f'{f}')
            else:  # 不是文件对象，要转存到文件
                if not files[i]:  # 没有设置文件名则生成一个
                    files[i] = File(ref_names[i], Dir.TEMP, suffix=default_suffix)
                files[i].write(arg)
                new_args.append(files[i])

        return new_args

    def __call__(self, *args, wait=True, files=None, sameoff=False, **kwargs):
        r"""
        :param wait:
            一般调用bcompare的时候，默认值wait=True，python是打开一个子进程并等待bcompare软件关闭后，才继续执行后续代码。
            如果你的业务场景并不需要等待bcompare关闭后执行后续python代码，可以设为wait=False，不等待。
        :param sameoff: 如果设为True，则会判断内容，两份内容如果相同则不打开bc
            这个参数一定程度会影响性能，非必要的时候不要开。
            或者在外部的时候，更清楚数据情况，可以在外部判断内容不重复再跑bcompare函数
            这个目前的实现策略，是读取文件重新判断的，会有性能开销，默认关闭该功能
        :return: 程序返回被修改的oldfile内容
            注意如果没有修改，或者wait=False，会返回原始值
        """
        files = self.to_bcompare_files(*args, files=files)

        if sameoff and len(files) > 1:
            if files[0].read() == files[1].read():
                return
        super().__call__(*([str(f) for f in files]), wait=wait, **kwargs)
        # bc软件操作中可能会修改原文内容，所以这里需要重新读取，不能用前面算过的结果
        return files[0].read()


bcompare = BCompare()


def modify_file(file, func, *, outfile=None, file_mode=None, debug=0):
    """ 对单个文件就行优化的功能函数

    这样一些底层函数功能可以写成数据级的接口，然后由这个函数负责文件的读写操作，并且支持debug比较前后内容差异

    :param outfile: 默认是对file原地操作，如果使用该参数，则将处理后的内容写入outfile文件
    :param file_mode: 指定文件读取类型格式，例如'.json'是json文件，读取为字典
    :param debug: 这个功能可以对照refine分级轮理解
        无outfile参数时，原地操作
            0，【直接原地操作】关闭调试，直接运行 （根据outfile选择原地操作，或者生成新文件）
            1，【进行审核】打开BC比较差异，左边原始内容，右边参考内容  （打开前后差异内容对比）
            -1，【先斩后奏】介于完全不调试和全部人工检查之间，特殊的差异比较。左边放原始文件修改后的结果，右边对照原内容。
        有outfile参数时
            0 | False，直接生成目标文件，如果outfile已存在会被覆盖
            1 | True，直接生成目标文件，但是会弹出bc比较前后内容差异 （相同内容不会弹出）
    """
    infile = File(file)
    data = infile.read(mode=file_mode)
    origin_content = str(data)
    new_data = func(data)

    isdiff = origin_content != str(new_data)
    if outfile is None:  # 原地操作
        if isdiff:  # 内容不同才会有相关debug功能，否则静默跳过就好
            if debug == 0:
                infile.write(new_data, mode=file_mode)  # 直接处理
            elif debug == 1:
                temp_file = File('refine_file', Dir.TEMP, suffix=infile.suffix).write(new_data)
                bcompare(infile, temp_file)  # 使用beyond compare软件打开对比查看
            elif debug == -1:
                temp_file = File('old_content', Dir.TEMP, suffix=infile.suffix)
                infile.copy(temp_file)
                infile.write(new_data, mode=file_mode)  # 把原文件内容替换了
                bcompare(infile, temp_file)  # 然后显示与旧内容进行对比
            else:
                raise ValueError(f'{debug}')
    else:
        outfile = File(outfile)
        outfile.write(new_data, mode=file_mode)  # 直接处理
        if debug and isdiff:
            bcompare(infile, outfile)

    return isdiff


class SetCmper:
    """ 集合两两比较 """

    def __init__(self, data):
        """
        :param data: 字典结构
            key: 类别名
            value: 该类别含有的元素（非set类型会自动转set）
        """
        self.data = copy.deepcopy(data)
        for k, v in self.data.items():
            if not isinstance(v, set):
                self.data[k] = set(self.data[k])

    def intersection(self):
        r""" 两两集合共有元素数量

        :return: df
            df对角线存储的是每个集合自身大小，df第i行第j列是第i个集合减去第j个集合的剩余元素数

        >>> s1 = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        >>> s2 = {1, 3, 5, 7, 8}
        >>> s3 = {2, 3, 5, 8}
        >>> df = SetCmper({'s1': s1, 's2': s2, 's3': s3}).intersection()
        >>> df
            s1  s2  s3
        s1   9   5   4
        s2   5   5   3
        s3   4   3   4
        >>> df.loc['s1', 's2']
        5
        """
        cats = list(self.data.keys())
        data = self.data
        n = len(cats)
        rows = []
        for i, c in enumerate(cats):
            a = data[c]
            row = [0] * n
            for j, d in enumerate(cats):
                if i == j:
                    row[j] = len(a)
                elif j < i:
                    row[j] = rows[j][i]
                elif j > i:
                    row[j] = len(a & data[d])
            rows.append(row)
        df = pd.DataFrame.from_records(rows, columns=cats)
        df.index = cats
        return df


class PairContent:
    """ 配对文本类，主要用于bc差异比较 """

    def __init__(self, left_file_name=None, right_file_name=None):
        self.left_file = File(left_file_name) if left_file_name else left_file_name
        self.right_file = File(right_file_name) if right_file_name else right_file_name
        self.left, self.right = [], []

    def add(self, lt, rt=None):
        """ rt不加，默认本轮内容是同lt """
        lt = str(lt)
        self.left.append(lt)
        rt = lt if rt is None else str(rt)
        self.right.append(rt)

    def bcompare(self, **kwargs):
        left, right = '\n'.join(self.left), '\n'.join(self.right)
        if self.left_file is not None:
            left = self.left_file.write(left)
        if self.right_file is not None:
            right = self.right_file.write(right)
        bcompare(left, right, **kwargs)
