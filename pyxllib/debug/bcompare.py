#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/06/01 18:13


from pyxllib.debug.dprint import func_input_message, dprint
from pyxllib.debug.strlib import natural_sort_key, prettifystr
from pyxllib.debug.pathlib_ import Path
from pyxllib.debug.dirlib import refinepath
from pyxllib.debug.chrome import viewfiles


def intersection_split(a, b):
    """输入两个对象a,b，可以是dict或set类型，list等

    会分析出二者共有的元素值关系
    返回值是 ls1, ls2, ls3, ls4，大部分是list类型，但也有可能遵循原始情况是set类型
        ls1：a中，与b共有key的元素值
        ls2：a中，独有key的元素值
        ls3：b中，与a共有key的元素值
        ls4：b中，独有key的元素值
    """
    # 1、获得集合的key关系
    keys1 = set(a)
    keys2 = set(b)
    keys0 = keys1 & keys2  # 两个集合共有的元素

    # 2、组合出ls1、ls2、ls3、ls4

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


def bcompare(oldfile, newfile=None, basefile=None, wait=True, named=True):
    """调用Beyond Compare软件对比两段文本（请确保有把BC的bcompare.exe加入环境变量）

    :param oldfile:
    :param newfile:
    :param basefile: 一般用于冲突合并是，oldfile、newfile共同依赖的旧版本
    :param wait: 见viewfiles的kwargs参数解释
        一般调用bcompare的时候，默认值wait=True，python是打开一个子进程并等待bcompare软件关闭后，才继续执行后续代码。
        如果你的业务场景并不需要等待bcompare关闭后执行后续python代码，可以设为wait=False，不等待。
    :param named:
        True，如果没有文件名，使用输入的变量作为文件名
        False，使用oldfile、newfile作为文件名
    :return: 程序返回被修改的oldfile内容
        注意如果没有修改，或者wait=False，会返回原始值
    这在进行调试、检测一些正则、文本处理算法是否正确时，特别方便有用
    >> bcompare('oldfile.txt', 'newfile.txt')

    180913周四：如果第1、2个参数都是set或都是dict，会进行特殊的文本化后比较
    """
    # 1、如果oldfile和newfile都是dict、set、list、tuple，则使用特殊方法文本化
    #   如果两个都是list，不应该提取key后比较，所以限制第1个类型必须是dict或set，然后第二个类型可以适当放宽条件
    if not oldfile: oldfile = str(oldfile)
    if isinstance(oldfile, (dict, set)) and isinstance(newfile, (dict, set, list, tuple)):
        t = [prettifystr(li) for li in intersection_split(oldfile, newfile)]
        oldfile = f'【共有部分】，{t[0]}\n\n【独有部分】，{t[1]}'
        newfile = f'【共有部分】，{t[2]}\n\n【独有部分】，{t[3]}'

    # 2、获取文件扩展名ext
    if Path(oldfile).is_file():
        ext = Path(oldfile).suffix
    elif Path(newfile).is_file():
        ext = Path(newfile).suffix
    elif Path(basefile).is_file():
        ext = Path(basefile).suffix
    else:
        ext = '.txt'  # 默认为txt文件

    # 3、生成所有文件
    ls = []
    names = func_input_message()['argnames']
    if not names[0]:
        names = ('oldfile.txt', 'newfile.txt', 'basefile.txt')

    def func(file, d):
        if file is not None:
            if Path(file).is_file():
                ls.append(file)
            else:
                ls.append(Path(refinepath(names[d] + ext), root=Path.TEMP).write(file, if_exists='replace').fullpath)

    func(oldfile, 0)
    func(newfile, 1)
    func(basefile, 2)  # 注意这里不要写names[2]，因为names[2]不一定有存在

    # 4、调用程序（并计算外部操作时间）
    viewfiles('BCompare.exe', *ls, wait=wait)
    return Path(ls[0]).read()
