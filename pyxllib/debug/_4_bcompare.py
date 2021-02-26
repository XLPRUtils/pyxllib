#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/01 18:13


from pyxllib.basic import func_input_message, dprint, natural_sort_key, File, refinepath, Dir
from pyxllib.debug._1_typelib import prettifystr
from pyxllib.debug._2_chrome import viewfiles

import copy
import pandas as pd


def intersection_split(a, b):
    """输入两个对象a,b，可以是dict或set类型，list等

    会分析出二者共有的元素值关系
    返回值是 ls1, ls2, ls3, ls4，大部分是list类型，但也有可能遵循原始情况是set类型
        ls1：a中，与b共有key的元素值
        ls2：a中，独有key的元素值
        ls3：b中，与a共有key的元素值
        ls4：b中，独有key的元素值
    """
    # 1 获得集合的key关系
    keys1 = set(a)
    keys2 = set(b)
    keys0 = keys1 & keys2  # 两个集合共有的元素

    # TODO 如果是字典，希望能保序

    # 2 组合出ls1、ls2、ls3、ls4

    def split(t, s, ks):
        """原始元素为t，集合化的值为s，共有key是ks"""
        if isinstance(t, (set, list, tuple)):
            return ks, s - ks
        elif isinstance(t, dict):
            ls1 = sorted(map(lambda x: (x, t[x]), ks), key=lambda x: natural_sort_key(x[0]))
            ls2 = sorted(map(lambda x: (x, t[x]), s - ks), key=lambda x: natural_sort_key(x[0]))
            return ls1, ls2
        else:
            dprint(type(s))  # s不是可以用来进行集合规律分析的类型
            raise ValueError

    ls1, ls2 = split(a, keys1, keys0)
    ls3, ls4 = split(b, keys2, keys0)
    return ls1, ls2, ls3, ls4


def bcompare(oldfile, newfile=None, basefile=None, wait=True, sameoff=False, oldfilename=None, newfilename=None):
    """ 调用Beyond Compare软件对比两段文本（请确保有把BC的bcompare.exe加入环境变量）

    :param oldfile:
    :param newfile:
    :param basefile: 一般用于冲突合并时，oldfile、newfile共同依赖的旧版本
    :param wait: 见viewfiles的kwargs参数解释
        一般调用bcompare的时候，默认值wait=True，python是打开一个子进程并等待bcompare软件关闭后，才继续执行后续代码。
        如果你的业务场景并不需要等待bcompare关闭后执行后续python代码，可以设为wait=False，不等待。
    :param sameoff: 如果设为True，则会判断内容，两份内容如果相同则不打开bc
        这个参数一定程度会影响性能，非必要的时候不要开。
        或者在外部的时候，更清楚数据情况，可以在外部判断内容不重复再跑bcompare函数。
    :param oldfilename: 强制指定旧的文件名，如果oldfile已经是一个文件路径，则不生效
    :param newfilename: 强制指定新的文件名，如果newfile已经是一个文件路径，则不生效
    :return: 程序返回被修改的oldfile内容
        注意如果没有修改，或者wait=False，会返回原始值
    这在进行调试、检测一些正则、文本处理算法是否正确时，特别方便有用
    >> bcompare('oldfile.txt', 'newfile.txt')

    180913周四：如果第1、2个参数都是set或都是dict，会进行特殊的文本化后比较
    """
    # 1 如果oldfile和newfile都是dict、set、list、tuple，则使用特殊方法文本化
    #   如果两个都是list，不应该提取key后比较，所以限制第1个类型必须是dict或set，然后第二个类型可以适当放宽条件
    if not oldfile: oldfile = str(oldfile)
    if isinstance(oldfile, (dict, set)) and isinstance(newfile, (dict, set, list, tuple)):
        t = [prettifystr(li) for li in intersection_split(oldfile, newfile)]
        oldfile = f'【共有部分】，{t[0]}\n\n【独有部分】，{t[1]}'
        newfile = f'【共有部分】，{t[2]}\n\n【独有部分】，{t[3]}'

    # 2 获取文件扩展名ext
    if File.safe_init(oldfile):
        ext = File(oldfile).suffix
    elif File.safe_init(newfile):
        ext = File(newfile).suffix
    elif File.safe_init(basefile):
        ext = File(basefile).suffix
    else:
        ext = '.txt'  # 默认为txt文件

    # 3 生成所有文件
    ls = []
    names = func_input_message()['argnames']
    if not names[0]:
        names = ('oldfile.txt', 'newfile.txt', 'basefile.txt')

    def func(file, d):
        if file is not None:
            p = File.safe_init(file)
            if p:
                ls.append(str(p))
            else:
                if d == 0 and oldfilename:
                    name = oldfilename
                elif d == 1 and newfilename:
                    name = newfilename
                else:
                    name = refinepath(names[d] + ext)
                ls.append(File(name, Dir.TEMP).write(file, if_exists='delete').to_str())

    func(oldfile, 0)
    func(newfile, 1)
    func(basefile, 2)  # 注意这里不要写names[2]，因为names[2]不一定有存在

    # 4 调用程序（并计算外部操作时间）
    if sameoff:
        if File(ls[0]).read() != File(ls[1]).read():
            viewfiles('BCompare.exe', *ls, wait=wait)
    else:
        viewfiles('BCompare.exe', *ls, wait=wait)
    return File(ls[0]).read()


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
