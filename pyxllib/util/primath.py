#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2018/11/09 ~ 2018/12/09

"""
生成小学数学题的通用工具箱

小学数学： Primary school mathematics
    primath含义：取第1个单词前3个字母pri，+math
"""

from code4101py.util.filelib import *
from code4101py.tex.nestlib import NestEnv

import random
from random import randint
import numpy
import numpy as np

from enum import Enum
import decimal
from decimal import Decimal
from fractions import Fraction
import math
from functools import reduce


class Version(Enum):
    LOCAL = 0
    WEB = 1


def get_prob(p):
    s = sum(p)
    p = map(lambda x: x / s, p)
    return tuple(p)


def get_non_order_key(*ls):
    """对ls里的内容进行排序后生成键值，确保相同元素的ls对应键值是一样的"""
    return str(natural_sort(ls))


def choice(a, p):
    """在数组a中，按照概率分布p来挑选一个元素
    即对每个元素的选取设置权重
    """
    if abs(sum(p) - 1) > 0.001:
        p = get_prob(p)
    t = numpy.random.choice(a, p=p)
    if isinstance(t, numpy.int32):
        t = int(t)
    return t


def shuffle(s):
    ls = list(s)
    random.shuffle(ls)
    return ls


class PickDecorator:
    """
    这函数很碉~~


    """
    maxn = 1000  # 每一个知识点最多生成的题目数
    threshold = 1000  # 产生的重复题达到这个阈值，程序强制停止继续生成

    def __init__(self, func):
        self.func = func
        self.res = []  # 函数的运行结果
        self.s = set()  # 去重检查器

    def run(self):
        if self.res:
            return
        random.seed(0)  # 固定随机数种子，让每次生成的结果都固定，方便避免重复题等问题
        numpy.random.seed(0)
        cnt = 0
        for ans in self.func():
            key = ans[0] if len(ans) == 2 else ans[2]  # 如果输入只有两个参数，以第0个参数作为key，否则以第2个参数为key
            if len(self.res) == PickDecorator.maxn or cnt == PickDecorator.threshold:
                break
            elif key not in self.s:
                a0, a1 = str(ans[0]), str(ans[1])
                try:
                    if latexeval(ans[0]) != latexeval(ans[1]):  # 智能判断运算结果是否正确
                        dprint(ans[0], latexeval(ans[0]), ans[1])  # 运算结果好像不对
                except SyntaxError:  # 报错就不管了
                    pass
                except NameError:
                    pass
                except AttributeError:
                    pass
                self.res.append(ans)
                self.s.add(key)
            else:
                if random.random() < 0.5:  # 有一半的概率旧值会被新值（虽然key相同）取代
                    for i in range(len(self.res)):
                        if len(ans) > 2 and len(self.res[i]) > 2 and self.res[i][2] == ans[2]:
                            # dprint(self.res[i], ans)
                            self.res[i] = ans
                            break
                cnt += 1
        self.res = list(map(lambda x: x[:2], self.res))

    def __call__(self, item=None):
        """如果取的数量少于总数，则用shuffle打乱"""
        self.run()
        res = self.res
        if isinstance(item, int):
            if item < len(res):
                res = shuffle(res)
            return res[:item]
        else:
            return res

    def __getitem__(self, item=None):
        self.run()
        res = self.res
        if isinstance(item, int):
            return res[item]
        elif isinstance(item, slice):
            try:
                if item.stop < len(res):
                    res = shuffle(res)
            except TypeError:
                pass
            return res[item]


class Tolatex:
    version = Version.LOCAL

    def __init__(self, filename):
        self.filename = filename
        self.content = []
        self.cur_section = 0
        self.cnt = 0

    def add_template(self, title, ls=None, *, fmt=None, linelen=1, nhang=0, knowledge=None):
        """建议给的ls是微过量版

        :param title: 本节名称
        :param ls: 生成的所有算式
        :param fmt: 题干答案等格式控制
            None，采用默认格式
            'empty'，不进行任何设置，由输入的ls完全决定
        :param linelen: 下划线长度
        """
        title = title.replace(' ', '')
        if ls:  # 输入算式时，处理的是四级标题
            # 1、采用自动编号的a.b值，而不是输入的明文编号
            self.cnt += 1
            title = re.sub(r'^[一-十四0-9\.]+、', '', title)  # 删除手动提供的编号
            title = f'{self.cur_section}.{self.cnt}、' + title  # 使用自动编号
            tex = [f'\\section{{{title}}}\n',
                   f'%<ch_template title={title} type=CONTENT>\n']

            # 2、生成带百分注的题目内容
            i = 1
            for stem, ans in ls:
                tex.append('%<topic type=tiankong>')

                if self.version == Version.LOCAL:
                    if fmt is None:
                        stem = f'${stem} = \\TK{{{ans}}}$'
                    elif fmt == 'empty':
                        stem = f'{stem} \\TK{{{ans}}}'
                    ans = f'%{ans}'
                else:  # WEB
                    if isinstance(stem, str): stem = NestEnv(stem).formula(invert=True).replace(lambda x: x.replace('\\%', '%'))
                    if isinstance(ans, str): ans = NestEnv(ans).formula(invert=True).replace(lambda x: x.replace('\\%', '%'))

                    if nhang:
                        if fmt is None:
                            stem = f'${stem}=$\n\n$\\nHang{{{nhang}}}$'
                        elif fmt == 'empty':
                            stem = f'{stem}\n\n$\\nHang{{{nhang}}}$'
                    else:
                        if fmt is None:
                            stem = f'${stem} = \\underline{{\\hspace{{{linelen}cm}}}}$'
                        elif fmt == 'empty':
                            stem = f'{stem} $\\underline{{\\hspace{{{linelen}cm}}}}$'

                tex.append('\\itemQ{}{}')
                tex.append(f'%<seq value={i}>\n%</seq>')
                if knowledge: tex.append(f'%<knowledge value={knowledge}>\n%</knowledge>')
                tex.append(f'%<stem>\n{stem}\n%</stem>\n%<answer>\n{ans}\n%</answer>')
                tex.append('%</topic>\n')
                i += 1
        else:  # 未输入算式时，处理的是三级标题
            title = re.sub(r'^Section\s*(\d+)、', '', title)  # 删除手写的section编号，使用程序自己计算的编号值
            self.cur_section += 1
            self.cnt = 0
            title = f'Section{self.cur_section}、' + title  # 使用自动编号
            tex = [f'\\chapter{{{title}}}\n', f'%<ch_template title={title} type=CONTENT>\n']
        tex.append(f'%</ch_template>\n')
        self.content.append('\n'.join(tex))

    def save(self):
        # 去除文档标题里的编号
        if re.match('\d+-\d+、', self.filename):
            title_ = re.sub('^\d+-\d+、(.*?)', r'\1', self.filename)
        else:
            title_ = self.filename
        # 生成整份讲义代码
        if self.version == Version.LOCAL:
            t = f'\n\n%<ch_title>\n%{title_}\n%</ch_title>\n'  # 本地测试版
        else:
            t = f'\n\n%<ch_title>\n{title_}\n%</ch_title>\n'  # 导入平台版
        head = '%\n\\documentclass[10pt,openany,hyperref]{ctexbook}' \
               + '\n\\input{../ConfChemStu}\n\n\n\\begin{document}\n' + t
        end = '\\end{document}'
        # 写入文件
        with open(self.filename + '.tex', 'w', encoding='utf8') as f:
            f.write(head + '\n'.join(self.content) + end)


class Carry(Enum):
    """加法和乘法都可以用到这个枚举类"""
    不进位 = 0
    不连续进位 = 1
    连续进位 = 2


class Abdication(Enum):
    """减法用得到"""
    不退位 = 0
    不连续退位 = 1
    连续退位 = 2


def extraction_digits(v):
    """
    >>> list(extraction_digits(123))
    [3, 2, 1]
    """
    while v:
        yield v % 10
        v //= 10


def idxdigit(a, i):
    """取一个数a第i位上的值
    >>> idxdigit(123, 0)
    3
    >>> idxdigit(123, 1)
    2
    """
    return a // (10**i) % 10


def 加法进位分类(a, b):
    """
    输入可以是int、float、Decimal类型

    >>> 加法进位分类(33, 11)
    <Carry.不进位: 0>
    >>> 加法进位分类(15, 15)
    <Carry.不连续进位: 1>
    >>> 加法进位分类(55, 55)
    <Carry.连续进位: 2>
    >>> 加法进位分类(505, 505)
    <Carry.不连续进位: 1>
    >>> 加法进位分类(43, 3)
    """
    # 0、预处理，先都变成整数
    a, b = Decimal(a), Decimal(b)
    n = max(len_decimal(a), len_decimal(b))  # 算出最长的小数位数
    a, b = int(a*10**n), int(b*10**n)

    # 1、
    a = list(extraction_digits(a))
    b = list(extraction_digits(b))
    n1, n2 = len(a), len(b)
    if n1 > n2:
        b.extend([0] * (n1 - n2))
    elif n1 < n2:
        a.extend([0] * (n2 - n1))

    # 2、
    cnt = 0
    ans = logo = 0
    for k1, k2 in zip(a, b):
        cnt += k1 + k2
        if cnt >= 10:
            if logo == 0:
                logo = 1
            elif logo == 1:
                return Carry.连续进位
        else:
            ans = max(ans, logo)
            logo = 0
        cnt //= 10
    else:
        ans = max(ans, logo)
    return Carry(ans)


def 乘法进位分类(a, b):
    """
    >>> 乘法进位分类(123, 33)
    <Carry.不进位: 0>
    >>> 乘法进位分类(105, 20)
    <Carry.不连续进位: 1>
    >>> 乘法进位分类(55, 22)
    <Carry.连续进位: 2>
    >>> 乘法进位分类(505, 2)
    <Carry.不连续进位: 1>
    """
    # 0、预处理，先都变成整数
    a, b = Decimal(a), Decimal(b)
    n = max(len_decimal(a), len_decimal(b))  # 算出最长的小数位数
    a, b = int(a*10**n), int(b*10**n)


    def 多位数乘一位数的进位(a, b):
        """
        >>> 多位数乘一位数的进位(123, 3)
        <Carry.不进位: 0>
        >>> 多位数乘一位数的进位(15, 2)
        <Carry.不连续进位: 1>
        >>> 多位数乘一位数的进位(55, 2)
        <Carry.连续进位: 2>
        """
        if b > a: a, b = b, a
        ans = logo = 0
        cnt = 0
        for t in extraction_digits(a):
            cnt += t * b
            if cnt >= 10:
                if logo == 0:
                    logo = 1
                elif logo == 1:
                    return Carry.连续进位
            else:
                ans = max(ans, logo)
                logo = 0
            cnt //= 10
        else:
            ans = max(ans, logo)
        return Carry(ans)

    if b > a: a, b = b, a
    logo = 0
    for t in extraction_digits(b):
        logo = max(logo, 多位数乘一位数的进位(a, t).value)
    return Carry(logo)


def 减法退位分类(a, b):
    """a-b的退位分类
    >>> 减法退位分类(33, 11)
    <Abdication.不退位: 0>
    >>> 减法退位分类(20, 15)
    <Abdication.不连续退位: 1>
    >>> 减法退位分类(155, 66)
    <Abdication.连续退位: 2>
    >>> 减法退位分类(1555, 606)
    <Abdication.不连续退位: 1>
    """
    # 0、预处理，先都变成整数
    a, b = Decimal(a), Decimal(b)
    n = max(len_decimal(a), len_decimal(b))  # 算出最长的小数位数
    a, b = int(a*10**n), int(b*10**n)

    # 1、
    a = list(extraction_digits(a))
    b = list(extraction_digits(b))
    n1, n2 = len(a), len(b)
    if n1 > n2:
        b.extend([0] * (n1 - n2))
    elif n1 < n2:
        a.extend([0] * (n2 - n1))

    # 2、
    cnt = 0
    ans = logo = 0
    for k1, k2 in zip(a, b):
        cnt = -1 if cnt + k1 < k2 else 0
        if cnt < 0:
            if logo == 0:
                logo = 1
            elif logo == 1:
                return Abdication.连续退位
        else:
            ans = max(ans, logo)
            logo = 0
    else:
        ans = max(ans, logo)
    return Abdication(ans)


def 重复题判断(root):
    """用于对一册书（一个目录下的多份tex）的重复性题目判断
    会对每道题生成一个key
    跟题干所用数字，和答案有关

    一本书里无重题并不是强要求，只是能做到更好，做不到影响也不大，只要每节不重题就行了

    这个算法不仅是一本书里判重，其实也能检查一节里的重题！！！

    示例视频： https://pan.baidu.com/s/1n1bAvwv3wjTP6qxQy-Nc4A
    """
    # 1、获取基础数值
    filename = []
    stem = []
    answer = []
    for f in getfiles(root, '.tex'):
        s = Path(f).read()
        f = os.path.basename(f)
        stems = re.findall(r'%<stem>\n(.*?)\n%</stem>', s, flags=re.DOTALL)
        n = len(stems)
        filename.extend([f]*n)
        stem.extend(stems)
        answers = re.findall(r'%<answer>[ ]*\n%?(.*?)\n%</answer>', s, flags=re.DOTALL)
        if len(answers) != len(stems):
            dprint(f, len(stem), len(answers))  # 该文件有题目没有answer
        answer.extend(answers)
    df = pd.DataFrame({'filename': filename, 'stem': stem, 'answer': answer})

    # 2、计算判重用的key和count
    def get_key(row):
        stem, ans = row['stem'], row['answer']
        d = re.findall('\d+\.?\d*|[+\-]', stem)
        d.sort()
        return str(d) + ans
    df['key'] = df.apply(get_key, axis=1)
    key_count = df['key'].value_counts()
    df['count'] = df.apply(lambda row: key_count[row['key']], axis=1)

    # 3、判重分析
    df = df[df['count'] > 1]  # 只留下有重复key数据
    df = df.set_index(['key', 'filename']).sort_index(level=0)
    chrome(df)


def text(a):
    if isinstance(a, Decimal):
        a = Fraction(a)
        a = Frac(a.numerator, a.denominator)
        return a.latex()
    else:
        return a


def pretty(d):
    """美化小数类型输出"""
    if isinstance(d, Decimal):
        return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()
    else:
        return d


def pretty2(d):
    """能转分数的转分数"""
    d = pretty(d)
    if isinstance(d, Decimal):
        d = Fraction(d)
        d = Frac(d.numerator, d.denominator)
        return '$' + str(d.latex()) + '$'
    return str(d)


def expand_add(v):
    """将v随机拆成两个数的和"""
    if isinstance(v, int) and v > 20:
        a = randint(10, v-10)
        return a, v - a
    elif isinstance(v, Decimal):
        n = len_decimal(v)
        a, b = expand_add(int(v * (10 ** n)))
        if n:
            t = Decimal('0.' + '0'*(n-1) + '1')
        else:
            t = 1
        return pretty(a * t), pretty(b * t)
    else:
        return


def expand_sub(v, maxv=999):
    """将v随机拆成两个数的差
    :param maxv: 被减数可能的最大值
    """
    if isinstance(v, int) and v < maxv - 10:
        a = randint(v+10, maxv)
        return a, a - v
    elif isinstance(v, float):
        return
    else:
        return

def expandn_addorsub(v, n=2, maxv=999):
    """
    :param v:
    :param n: 拆成n个数相加减，结果为v
        拆成加减的概率相同
    :param maxv:
    :return:
    """
    pass


def primefactors(n):
    """Generate all prime factors of n."""
    f = 2
    while f * f <= n:
        while not n % f:
            yield f
            n //= f
        f += 1
    if n > 1:
        yield n


def expand_mul(v):
    """将v随机拆成两个数的乘积，因数不能出现1
    拆分失败返回None
    """
    if isinstance(v, int) and v > 4:
        factors = list(primefactors(v))
        if len(factors) <= 1:
            return
        # 每个质因数有一半概率归到a
        a = 1
        for t in factors:
            if random.random() < 0.5:
                a *= t
        # 能不出现因数1尽量不要出现
        if a == 1:
            a *= factors[-1]
        elif a == v:
            a //= factors[-1]
        return a, v // a
    elif isinstance(v, float):
        return
    else:
        return


def expand_div(v, maxv=999):
    """将v随机拆成两个数的除法
    :param maxv: 被除数可能的最大值
    """
    if isinstance(v, int) and v > 1:
        if maxv // v < 2: return
        b = randint(2, maxv//v)
        return v*b, b
    elif isinstance(v, float):
        return
    else:
        return


def expand_random(v, maxv=999, depth=0, expand_add=expand_add, expand_sub=expand_sub,
                  expand_mul=expand_mul, expand_div=expand_div):  # 可以自定义加减乘除展开规则
    """随机展开一个四则运算式子，其运算结果是v，返回latex代码表达式
    depth可以设置层级
    失败会返回None
    """
    r = random.random()
    if r < 0.2:  # 出加法题
        t = expand_add(v)
        if not t: return
        a, b = t
        op = '+'
    elif r < 0.4:  # 出减法题
        t = expand_sub(v)
        if not t: return
        a, b = t
        op = '-'
    elif r < 0.7:  # 出乘法题（乘除的概率比加减高一些）
        t = expand_mul(v)
        if not t: return
        a, b = t
        op = '\\times'
    else:  # 出除法题
        t = expand_div(v)
        if not t: return
        a, b = t
        op = '\\div'

    if depth:
        left, right = '([{'[depth-1], ')]}'[depth-1]
        if random.random() < 0.5:  # 展开左边
            t = expand_random(a, maxv, depth-1)
            if not t: return
            a = f'{left}{t}{right}'
        else:  # 展开右边
            t = expand_random(b, maxv, depth-1)
            if not t: return
            b = f'{left}{t}{right}'

    return f'{a} {op} {b}'


def expand3_random(v, maxv=999,
                   expand_add=expand_add, expand_sub=expand_sub,
                   expand_mul=expand_mul, expand_div=expand_div):  # 可以自定义加减乘除展开规则
    """展开为3个数字的随机计算式: a o b o c
    """
    rand = random.random
    r = random.random()
    try:
        if r < 1/3:  # 两轮随机加减
            x, c, op2 = (*expand_add(v), '+') if rand() < 0.5 else (*expand_sub(v), '-')
            a, b, op1 = (*expand_add(x), '+') if rand() < 0.5 else (*expand_sub(x), '-')
        elif r < 2/3:  # 先随机加减，然后随机拆一个数变乘除
            x, y, op = (*expand_add(v), '+') if rand() < 0.5 else (*expand_sub(v), '-')
            if rand() < 0.5:  # 拆左边的数
                a, b, op1 = (*expand_mul(x), '\\times') if rand() < 0.5 else (*expand_div(x), '\\div')
                op2, c = op, y
            else:  # 拆右边的数
                b, c, op2 = (*expand_mul(y), '\\times') if rand() < 0.5 else (*expand_div(y), '\\div')
                op1, a = op, x
        else:  # 两轮随机乘除
            x, c, op2 = (*expand_mul(v), '\\times') if rand() < 0.5 else (*expand_div(v), '\\div')
            a, b, op1 = (*expand_mul(x), '\\times') if rand() < 0.5 else (*expand_div(x), '\\div')

        return f'{a} {op1} {b} {op2} {c}'
    except TypeError:
        return


def expand_random2(v, depth=0):
    """分数Frac版本
    """
    r = random.random()
    if r < 0.2:  # 出加法题
        t = v.expand_add()
        if not t: return
        a, b = t
        op = '+'
    elif r < 0.4:  # 出减法题
        t = v.expand_sub()
        if not t: return
        a, b = t
        op = '-'
    elif r < 0.7:  # 出乘法题（乘除的概率比加减高一些）
        t = v.expand_mul()
        if not t: return
        a, b = t
        op = '\\times'
    else:  # 出除法题
        t = v.expand_div()
        if not t: return
        a, b = t
        op = '\\div'

    if depth:
        if depth == 1:
            left, right = '\\left(', '\\right)'
        elif depth == 2:
            left, right = '\\left[', '\\right]'
        if random.random() < 0.5:
            t = expand_random2(a, depth-1)
            if not t: return
            a = f'{left}{t}{right}'
        else:
            t = expand_random2(b, depth-1)
            if not t: return
            b = f'{left}{t}{right}'

    if isinstance(a, Frac): a = a.latex3()
    if isinstance(b, Frac): b = b.latex3()

    return f'{a} {op} {b}'


class Expand:
    """前面的展开类，规则、命令太乱了，这里重构一个

    输入支持且仅支持：int 整数、 Decimal、float、str描述的小数

    重要技巧：定义对象后，可以自定义加减add、sub等展开规则，覆盖对象的原成员函数，实现自定制展开规则
    """
    def __init__(self, maxv=999):
        """
        :param maxv: 默认的最大取值
            不是严格意义的限制，当输入的数值很大时，算法会自适应扩充范围
        """
        self.maxv=maxv

    def add(self, v, n=2):
        """将数值v拆成n个数之间的运算（后面其他函数同名参数同理）
        这里是将v拆成n个数相加
        所有函数均以latex代码的字符串形式返回，
            要提取子部分的，可以用split对空格进行拆分，返回的字符串会用空格隔开各个项目
        """
        if isinstance(v, int):
            if n <= 1: return str(v)
            ls = [v]
            for i in range(n-1): ls.append(randint(0, v))
            ls.sort()
            for i in range(n-1, 0, -1): ls[i] = ls[i] - ls[i-1]  # 这步会生成和为v的一个随机数列
            # 如果有极端情况（含0值或较小值，尽可能做一些调整）
            while True:
                minv, maxv = min(ls), max(ls)
                if minv > 10 or maxv - minv < 2: break  # 最小值不比10小，或者里面的数极差小于2,，不用再调整优化
                #  否则将大的数随机分一部分给小的数，
                #    注意这里的分割值，会保证大的值仍然大于小的值，在后续index计算不会出错
                t = randint(1, maxv-minv-1)
                ls[ls.index(minv)] += t
                ls[ls.index(maxv)] -= t
            return ' + '.join(map(str, ls))
        elif isinstance(v, (float,Decimal,str)):
            v = Decimal(v)
            scale = 10**len_decimal(v)
            s = self.add(int(v*scale), n)
            s = re.sub(r'\d+', lambda m: str((Decimal(m.group())/scale).normalize()), str(s))
            return s
        else:
            return

    def sub(self, v, n=2):
        if isinstance(v, int):
            if n <= 1: return str(v)
            maxv = max(self.maxv, int(10**math.ceil(math.log10(v))))
            t = randint(min(maxv, v+2*n, v+10),  # 尽可能从大的地方往后取，
                        min(maxv, v*100))  # 也尽量不要偏离原值太远的地方
            return f'{t} - ' + self.add(t-v, n-1).replace('+', '-')  # 巧用加的拆分来实现减的拆分
        elif isinstance(v, (float, Decimal, str)):
            v = Decimal(v)
            scale = 10**len_decimal(v)
            s = self.sub(int(v*scale), n)
            s = re.sub(r'\d+', lambda m: str((Decimal(m.group())/scale).normalize()), str(s))
            return s
        else:
            return

    def addsub(self, v, n=2):
        """随机拆分成若干加加减混合运算"""
        if isinstance(v, int):
            if n <= 1: return str(v)
            s = f'{v}'
            for i in range(n-1):
                if random.random() < 0.5:  # 加减各一半概率
                    s = re.sub(r'^\d+', lambda m: str(self.add(int(m.group()))), s)
                else:
                    s = re.sub(r'^\d+', lambda m: str(self.sub(int(m.group()))), s)
            return s
        elif isinstance(v, (float, Decimal, str)):
            v = Decimal(v)
            scale = 10**len_decimal(v)
            s = self.addsub(int(v*scale), n)
            s = re.sub(r'\d+', lambda m: str((Decimal(m.group())/scale).normalize()), str(s))
            return s
        else:
            return

    def mul(self, v, n=2):
        if not v: return  # 0无法进行乘除拆分
        if isinstance(v, int):
            if n <= 1: return str(v)
            factors = list(primefactors(v))
            if len(factors) < n: return  # 因数不够拆分，返回None表示拆分失败
            # 每个位置至少拿一个因数
            ls = shuffle(factors)
            for i in range(n, len(ls)):  # 超出拆分部分的，每个随机添加到前面
                j = randint(0, n-1)
                ls[j] *= ls[i]
            return ' \\times '.join(map(str, ls[:n]))
        else:
            return

    def div(self, v, n=2):
        if not v: return  # 0无法进行乘除拆分
        if isinstance(v, int):
            if n <= 1: return str(v)
            if self.maxv//v < 2: return  # v的值太大了，没法拆分除法
            b = randint(2, self.maxv//v)
            a = b * v
            return f'{a} \\div ' + str(self.mul(b, n-1)).replace('\\times', '\\div')
        else:
            return

    def muldiv(self, v, n=2):
        """乘除混合运算"""
        if isinstance(v, int):
            if n <= 1: return str(v)
            s = f'{v}'
            for i in range(n-1):
                if random.random() < 0.5:  # 乘除各一半概率
                    s = re.sub(r'^\d+', lambda m: str(self.mul(int(m.group()))), s)
                else:
                    s = re.sub(r'^\d+', lambda m: str(self.div(int(m.group()))), s)
            if 'None' in s: return
            return s
        else:
            return


def get_decimal(n=None):
    """随机生成一个decimal对象
    小数部分可能一位也可能两位

    :param n: 控制生成的小数位数
        None，在1~99均匀生成
        1，一位小数
        2，两位小数
        3，三位小数
    """
    if n is None:
        t = randint(1, 99)
        s = f'{t:02d}'.rstrip('0')
    elif isinstance(n, int):
        # 生成特定位数的小数
        s = ''
        while len(s) != n:
            t = randint(1, 10 ** n - 1)
            s = f'{t:0{n}d}'
            s.strip('0')
    else:
        raise NotImplementedError
    a = Decimal('.' + s)
    return a


def get_decimal2(n=None):
    return randint(0, 9) + get_decimal(n)


def len_decimal(d):
    """decimal类小数部分的长度

    >>> d = Decimal('2.355'); len_decimal(d)
    3
    >>> d = Decimal('2.0'); len_decimal(d)
    0
    """
    return max(len(str(d % 1).strip('0').lstrip('.')), 0)


def 小数乘法凑整():
    """随机返回一个两元素的tuple

    第1个是decimal小数
    第2个是一个整数，二者相乘能“凑整”
    """
    d = {
        '0.125': [8],
        '0.2': [5],
        '0.25': [4, 8],
        '0.4': [5],
        '0.5': [2, 4, 6, 8],
        '0.6': [5],
        '0.8': [5],
    }
    k = random.choice(tuple(d.keys()))
    v = random.choice(d[k])
    return decimal.Decimal(k), v


def significant_figures(d):
    """计算一个Decimal对象的有效数字"""
    s = str(d).replace('.', '').strip('0')
    return len(s)


def df_row_filter(df, *funcs):
    ft = None
    for func in funcs:
        if ft:
            ft &= df.apply(func, axis=1)
        else:
            ft = df.apply(func, axis=1)
    return df[ft]


def gen_integer():
    """分母
        1/3概率： 2~9
        1/3概率： 10~20
        1/3概率： 21~99
    """
    t = np.random.choice(3)
    if t == 0:
        return np.random.choice(8) + 2
    elif t == 1:
        return np.random.choice(11) + 10
    else:
        return np.random.choice(79) + 21


def gen_integer1():
    """分母
        1/3概率： 1~9
        1/3概率： 10~20
        1/3概率： 21~99
    """
    t = np.random.choice(3)
    if t == 0:
        return np.random.choice(9) + 1
    elif t == 1:
        return np.random.choice(11) + 10
    else:
        return np.random.choice(79) + 21


class Frac:
    def __init__(self, m_=None, n_=None, *, can_one=False, integer=True, minv=None, maxv=None):
        """n分之m，m/n
        :param m_: 分子
        :param n_: 分母
        :param can_one: 是否可以生成值为1的分数，例如18/18
        :param integer: 分数实际值可以是整数，例如25/5
        :param minv: 最小值
        :param maxv: 最大值
        """
        # 1、支持用特殊类型数据初始化
        if isinstance(m_, Decimal):  # 用小数初始化
            m_ = Fraction(m_)
        if isinstance(m_, Fraction):  # 用标准分数类初始化
            self.m, self.n = m_.numerator, m_.denominator
            return

        # 2、否则按照普通规则随机生成一个分数
        if n_ is not None and m_ is not None:
            self.m, self.n = m_, n_
            return

        while True:
            m = gen_integer1() if m_ is None else m_  # 分子可以为1
            n = gen_integer() if n_ is None else n_  # 分母的初始化不能为1
            if not can_one:
                while n == m:  # 分子分母不能相同
                    n = gen_integer()

            self.m, self.n = m, n

            if integer is False and m % n == 0: continue
            if minv is not None and float(self) < minv: continue
            if maxv is not None and float(self) > maxv: continue
            break

    def defstr(self):
        """得到定义当前值的表达式"""
        return f'{type(self).__name__}({self.m}, {self.n})'

    def reduction(self):
        """约分
        该分数类不会主动进行约分，需要手动处理
        """
        if self.gcd() > 1:
            c = self.gcd()
            m, n = self.m // c, self.n // c
        else:
            m, n = self.m, self.n
        return Frac(m, n)

    def gcd(self):
        """计算分子和分母的最大公约数，如果分子为0，则会返回0"""
        if self.n:
            return math.gcd(self.n, self.m)
        else:
            return 0

    def maxv(self):
        """约分后的分子、分母最大值"""
        d = self.reduction()
        return max(d.n, d.m)

    def n0(self):
        """约分后的分母"""
        d = self.reduction()
        return d.n

    def latex(self):
        """将分数转为latex展示"""
        self = self.reduction()
        if self.n == 1:
            return str(self.m)
        elif self.m == 0:
            return '0'
        return f'\dfrac{{{self.m}}}{{{self.n}}}'

    def latex2(self):
        """将分数转为latex展示，假分数有一半的概率会转带分数"""
        self = self.reduction()
        if self.n != 1 and self.m > self.n and self.m % self.n and random.random() < 0.5:
            t, m = divmod(self.m, self.n)
            return f'{t}\dfrac{{{m}}}{{{self.n}}}'
        else:
            return self.latex()

    def latex3(self):
        """可以转为3位小数以内的数有1/3概率以小数显示"""
        d = self.decimal()
        n = len_decimal(d)
        self = self.reduction()
        if 0 < n < 4 and random.random() < 0.33:
            return str(d)
        else:
            return self.latex2()

    def latex4(self):
        """190802周五19:38，将分数转为latex展示
        先约分后显示
        """
        self = self.reduction()
        if self.n == 1:
            return str(self.m)
        elif self.m == 0:
            return '0'
        return f'\dfrac{{{self.m}}}{{{self.n}}}'

    def scale(self):
        """以比例的文本形式展现结果"""
        t = self.reduction()
        return f'{t.m}:{t.n}'

    def text(self, ops):
        if ops == '(d±d)':  # 拆成两个分数相加或相减的文本形式
            if random.random() < 0.5:
                a, b = self.expand_add()
                return f'\\left({a.latex2()} + {b.latex2()}\\right)'
            else:
                a, b = self.expand_sub()
                return f'\\left({a.latex2()} - {b.latex2()}\\right)'
        else:
            raise NotImplementedError

    def format(self, fmt=3):
        if fmt == 0:
            return str(self)
        elif fmt == 1:
            return self.latex()
        elif fmt == 2:
            return self.latex2()
        elif fmt == 3:
            return self.latex3()
        else:
            return repr(self)

    def decimal(self):
        """分数转小数"""
        return Decimal(self.m) / Decimal(self.n)

    def expand_add(self):
        """拆成两个分数相加，返回两个分数对象，分数的实际值可能是整数"""
        for _ in range(100):
            a = Frac(maxv=max(0.02, float(self)-0.01))
            b = self - a
            if not b: continue
            if b.maxv() < 100: break
        else:
            a = Frac(self.m, self.n*2)
            b = Frac(self.m, self.n*2)
        return a.reduction(), b.reduction()

    def expand_sub(self):
        """拆成两个分数相减，返回两个分数对象，分数的实际值可能是整数"""
        for _ in range(100):
            b = Frac()
            a = self + b
            if not a: continue
            if a.maxv() < 100: break
        else:
            b = Frac(randint(1, self.m), self.n)
            a = self + b
        if not a or not b: return None, None
        return a.reduction(), b.reduction()

    def expand_mul(self):
        for _ in range(100):
            a = Frac()
            b = self / a
            if not b: continue
            if b.maxv() < 100: break
        else:
            a = Frac()
            b = self / a
            return None
        return a.reduction(), b.reduction()

    def expand_div(self):
        for _ in range(100):
            b = Frac()
            a = self * b
            if a.maxv() < 100: break
        else:
            b = Frac()
            a = self * b
            return None
        return a.reduction(), b.reduction()

    def expands(self, ops, fmt=3):
        """指定ops操作顺序，不带括号的多项展开"""
        ls = []
        d = Frac(self.m, self.n)
        while len(ops):
            if ops[0] == '+':
                a, d = d.expand_add()
                ls.append(f'{a.format(fmt)} +')
                ops = ops[1:]
            elif ops[0] == '-':
                a, d = d.expand_sub()
                ls.append(f'{a.format(fmt)} -')
                ops = ops[1:]
            elif ops[:2] == '*+':
                a, d = d.expand_add()
                a, b = a.expand_mul()
                ls.append(f'{a.format(fmt)} \\times {b.format(fmt)} +')
                ops = ops[2:]
            elif ops[:2] == '*-':
                a, d = d.expand_sub()
                a, b = a.expand_mul()
                ls.append(f'{a.format(fmt)} \\times {b.format(fmt)} -')
                ops = ops[2:]
            elif ops[:2] == '/+':
                a, d = d.expand_add()
                a, b = a.expand_div()
                ls.append(f'{a.format(fmt)} \\div {b.format(fmt)} +')
                ops = ops[2:]
            elif ops[:2] == '/-':
                a, d = d.expand_sub()
                a, b = a.expand_div()
                ls.append(f'{a.format(fmt)} \\div {b.format(fmt)} -')
                ops = ops[2:]
            elif ops[0] == '*':
                a, d = d.expand_mul()
                ls.append(f'{a.format(fmt)} \\times')
                ops = ops[1:]
            elif ops[0] == '/':
                a, d = d.expand_div()
                ls.append(f'{a.format(fmt)} \\div')
                ops = ops[1:]
            elif ops[0] in '[()]':
                if ops[0] in ')]':
                    pass
                ls.append(ops[0])
                ops = ops[1:]
            else:
                raise NotImplementedError
        ls.append(f'{d.format(fmt)}')
        return ' '.join(ls)

    def expand_one(self, op):
        """按照运算符op展开，返回两个新的分数a,b，以及运算符op表达式的s"""
        if op == '+':
            a, b = self.expand_add()
            s = '+'
        elif op == '-':
            a, b = self.expand_sub()
            s = '-'
        elif op == '×':
            a, b = self.expand_mul()
            s = '\\times'
        else:
            a, b = self.expand_div()
            s = 'divs'
        return a, b, s

    def expand(self, cnt=None, ops=None, fmt=3):
        """
        :param cnt: 展开后包含的数字数量
        :param ops: 操作符
            如果提供了，长度必须为cnt-1，会按照ops提供的顺序运算符展开表达式
            如果未提供，则随机生成一串'+-×÷'
        :param fmt: 分数字符串化的方法，详见self.format(fmt)函数
        :return: 返回一段latex代码
        """
        # 0、参数初始化
        cnt = cnt if cnt and cnt > 0 else 2  # 默认展开一层
        if ops is None: ops = random.choices('+-×÷', k=cnt-1)
        assert len(ops) == cnt-1

        # 1、中括号展开
        if cnt >= 4 and random.random() < 0.33:  # 达到4个数可以出中括号题（中括号内部必带小括号）
            cnt1 = randint(3, cnt - 1)  # 左边的式子出中括号题
            cnt2 = cnt - cnt1  # 右边留下cnt2个数字再随机展开
            a, b, s = self.expand_one(ops[0])
            return f'[a.expands(ops[1:cnt1], fmt)]'
        # 2、小括号展开
        elif cnt >= 3 and random.random() < 0.33:  # 达到3个数可以出小括号题
            a, b, s = self.expand_one(ops[0])
        # 3、平凡展开一个数
        else:
            a, b, s = self.expand_one(ops[0])

    def __add__(self, other):
        """分数加法"""
        if isinstance(other, Frac):
            n = self.n * other.n // math.gcd(self.n, other.n)
            m = (n // self.n) * self.m + (n // other.n) * other.m
            if max(m, n) / math.gcd(m, n) > 99: return
            return Frac(m, n)
        else:
            return None

    def __radd__(self, other):
        """分数加法"""
        if isinstance(other, Frac):
            return self + other
        else:
            return None

    def __sub__(self, other):
        """分数减法"""
        if isinstance(other, Frac):
            n = self.n * other.n // math.gcd(self.n, other.n)
            m = (n // self.n) * self.m - (n // other.n) * other.m
            if m < 0: return
            if max(m, n) / math.gcd(m, n) > 99: return
            return Frac(m, n)
        else:
            return None

    def __rsub__(self, other):
        """分数减法"""

        if isinstance(other, Frac):
            return other - self
        else:
            return None

    def __mul__(self, other):
        """分数乘法"""
        if isinstance(other, int):
            m = self.m * other
            return Frac(m, self.n)
        elif isinstance(other, Frac):
            f = Frac(self.m * other.m, self.n * other.n)
            f = f.reduction()
            if max(f.m, f.n) > 99: return
            return f
        elif isinstance(other, Decimal):
            d = Fraction(other)
            return self * d
        elif isinstance(other, Fraction):
            return Frac(self.m * other.numerator, self.n * other.denominator)
        else:
            return None

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, int):
            n = self.n * other
            return Frac(self.m, n)
        elif isinstance(other, Frac):
            f = Frac(self.m * other.n, self.n * other.m)
            if max(f.m, f.n) / math.gcd(f.m, f.n) > 99: return
            return f
        elif isinstance(other, Decimal):
            d = Fraction(other)
            return self / d
        elif isinstance(other, Fraction):
            return Frac(self.m * other.denominator, self.n * other.numerator)
        else:
            return None

    def __rtruediv__(self, other):
        if isinstance(other, int):
            n = self.n * other
            return Frac(n, self.m)
        elif isinstance(other, Frac):
            f = Frac(self.m * other.n, self.n * other.m)
            if max(f.m, f.n) / math.gcd(f.m, f.n) > 99: return
            return f
        elif isinstance(other, Decimal):
            d = Fraction(other)
            return d / self
        elif isinstance(other, Fraction):
            return Frac(self.n * other.numerator, self.m * other.denominator)
        else:
            return None

    def __float__(self):
        return self.m / self.n

    def __lt__(self, other):
        if isinstance(other, Frac):
            return float(self) < float(other)
        elif isinstance(other, (int, float)):
            return float(self) < other
        else:
            return float(self) < float(other)

    def __eq__(self, other):
        if isinstance(other, Frac):
            return self.n == other.n and self.m == other.m
        return False

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.n == 1:
            return str(self.m)
        elif self.m == 0:
            return '0'
        else:
            return f'{self.m}/{self.n}'


def lcm(a, b):
    return a * b // math.gcd(a, b)


def lcms(*numbers):
    return reduce(lcm, numbers)


def gcds(*numbers):
    return reduce(math.gcd, numbers)


def expandformula(value, fmt, strfunc=None, genn=None, genf=None, gend=None):
    """按照fmt指定的格式，展开value的值

    :param value: 指定的答案数值
    :param fmt: 展开的式子形式
        '*+'，
        '(+)*'，带括号优先级，支持圆括号、方括号、花括号
        可以加入数字，n表示整数，f表示浮点数（Decimal），d表示分数（Fraction），如果没做数值类型限制，则默认与value相同
    :param strfunc: 可以自定义每种数值对象的字符串化方法
    :param genn: 随机生成一个整数的生成器
    :param genf: 随机生成一个浮点数的生成器
    :param gend: 随机生成一个分数的生成器
    :return: 展开失败会返回None
    """
    # 0、导入常用功能
    from random import randint
    from decimal import Decimal
    from fractions import Fraction

    # 1、基础函数
    def strfunc0(v): return str(v)

    def genn0(): return randint(2, 99)

    def genf0():
        t = randint(1, 99)
        a = Decimal('.' + f'{t:02d}'.rstrip('0')) + randint(0, 9)
        return a

    def gend0(): return Fraction(randint(1, 99), randint(2, 99))

    if not strfunc: strfunc = strfunc0
    if not genn: genn = genn0
    if not genf: genf = genf0
    if not gend: gend = gend0

    def expand(v, op):
        """按照指定的op操作展开数值v
        TODO：目前暂不支持op的操作符特殊数值类型指定功能
        """
        if '+' in op:
            return expand_add(v)
        elif '-' in op:
            return expand_sub(v)
        elif '*' in op:
            return expand_mul(v)
        elif '/' in op:
            return expand_div(v)
        else:
            raise NotImplementedError

    # 2、遍历fmt表达式
    ls = []
    try:
        while len(fmt):
            if re.match(r'\w?[+-]', fmt):  # 加减拆分
                m = re.match(r'\w?[+-]\w?', fmt)
                a, value = expand(value, m.group())
                op = '+' if '+' in m.group() else '-'
                ls.append(f'{strfunc(a)} {op}')
                fmt = fmt[m.end():]
            elif re.match(r'\w?[*/]\w?[+-]', fmt):  # 先乘除后加减，那么拆分的时候是要倒回来的，先加减拆分，再乘除拆分
                m = re.match(r'(\w?[*/]\w?)([+-]\w?)', fmt)
                a, value = expand(value, m.group(2))
                a, b = expand(a, m.group(1))
                op1 = '\\times' if '*' in m.group(1) else '\\div'
                op2 = '+' if '+' in m.group(2) else '-'
                ls.append(f'{strfunc(a)} {op1} {strfunc(b)} {op2}')
                fmt = fmt[m.end():]
            elif re.match(r'\w?[*/]', fmt):  # 再判断独立的乘除
                m = re.match(r'\w?[*/]\w?', fmt)
                a, value = expand(value, m.group())
                op = '\\times' if '*' in m.group() else '\\div'
                ls.append(f'{strfunc(a)} {op}')
                fmt = fmt[m.end():]
            elif fmt[0] in '{[(':
                pass
            else:
                raise NotImplementedError
    except TypeError:
        return ''
    ls.append(f'{strfunc(value)}')
    return ' '.join(ls)


def ensure_brack(s, force=False):
    """如果表达式s里
        有括号[]，加上花括号
        有括号()，加上方括号
    :param s: 表达式
    :param force:
        False: 如果只有乘除运算，不加括号
        True: 无论什么情况都增加高一级的括号
    """
    # 1、如果有加减符在里面，是强制要加括号的
    x = r'(' + grp_bracket(left='(') + '|' + grp_bracket(left='[') + '|' + grp_bracket(left='{') + ')'
    t = re.sub(x, '', s)
    if '+' in t or '-' in t: force = True

    if '{' in s:
        return s
    elif '[' in s:
        if force: return '\\left\\{' + s + '\\right\\}'
    elif '(' in s:
        if force: return '\\left[' + s + '\\right]'
    else:
        if force: return '\\left(' + s + '\\right)'


def suffix_addsub(text, ans, r):
    """已有一套式子text的答案是ans，在末尾随机加减一个分数
    有r的概率获得新的式子text和答案ans
    有1-r的概率不进行改变
    """
    if random.random() < r:
        r1 = random.random()
        if r1 < 0.5:  # 再加上一个数，前后加随机
            c, t = ans.expand_sub()
            if not t: return None, None
            s1, s2 = shuffle([text, t.latex2()])
            return f'{s1} + {s2}', c
        elif r1 < 0.75 or '[' in text:  # 减去一个数
            c, t = ans.expand_add()
            return f'{text} - {t.latex2()}', c
        else:  # 被一个数减
            t, c = ans.expand_sub()
            if not t: return None, None
            return f'{t.latex2()} - {ensure_brack(text)}', c
    else:
        return text, ans


def simfrac(d):
    """生成一个与d相乘很好运算的分数"""
    t = expand_mul(d.n)
    ls = [1, d.n, d.n*2, d.n*3]
    p = [2, 1, 1, 1]
    if t:
        ls.extend(t)
        p.extend([5, 5])
    m = choice(ls, p)

    t = expand_mul(d.m)
    n = max(t) if t else d.m * randint(1, 100//d.m)
    return Frac(int(m), int(n)).reduction()


def get接近整十整数():
    r = random.random()
    if r < 0.25: b = -2
    elif r < 0.5: b = -1
    elif r < 0.75: b = 1
    else: b = 2

    if random.random() < 0.2:  # 10
        a = 10
    else:  # 整百
        a = randint(1, 9) * 100

    return a, b


def get接近整十小数():
    a, b = get接近整十整数()
    r = random.random()
    if a == 10:
        if r < 0.5:
            a = 1
            b *= Decimal('0.1')
    else:
        if r < 0.5:
            a *= Decimal('0.1')
            b *= Decimal('0.1')
        else:
            a *= Decimal('0.01')
            b *= Decimal('0.01')

    return a, b


def gen_value():
    """获得一个整数或小数或分数，不过本质上还是将其转为分数类Fraction来返回"""
    r = random.random()
    if r < 0.4:
        return Frac(randint(2, 99), 1)
    elif r < 0.8:
        t = Fraction(get_decimal() + randint(0, 4))
        return Frac(t.numerator, t.denominator)
    else:
        return Frac()


def gen_value2():
    """190813周二15:17，按照昨天方康老师给的意见重新调整"""
    r = random.random()
    if r < 0.4:
        return Frac(randint(2, 99), 1)
    elif r < 0.8:
        t = Fraction(get_decimal() + randint(0, 4))
        return Frac(t.numerator, t.denominator)
    else:
        return Frac()


def 获得适合转百分比的小数():
    a = randint(0, 99)
    b = choice([0, random.choice(range(10, 100, 10)), randint(11, 99)], [2, 5, 3])
    a = Decimal(a * 100 + b) / Decimal(10000)
    a += choice([0, 1, 2, 3, 4], [10, 5, 3, 2, 1])
    return a


def latexeval(s0, mode=None):
    r"""
    :param s0:
    :return:

    >>> latexeval('3+4')
    '7'
    >>> latexeval('$6*7+4=$')
    '46'
    >>> latexeval(r'$20 \div 6 = $')
    '$3 \\cdots\\cdots 2$'
    >>> latexeval(r'$50\div9=$')
    '$5 \\cdots\\cdots 5$'
    >>> latexeval(r'$32\div5 = $')
    '6.4'
    >>> latexeval(r'$36\div9\div9 = $')
    '$0 \\cdots\\cdots 4$'
    >>> latexeval(r'$97\div8= $')
    '12.125'
    >>> latexeval(r'$0.48 \div 0.4 = $')
    '1.2'
    >>> latexeval(r'$2\dfrac{4}{7} \times 1\dfrac{2}{7} = $', mode='dfrac')
    '$\\dfrac{162}{49}$'

    基本开发思路：
        1、将latex式子使用convert()转成Python的Fraction有理数数值运算
        2、用eval计算出结果值Fraction
        3、对结果Fraction调用mystr转为latex
        4、方括号替换为圆括号
        TODO 后续如果还有括号、积分等需求可以扩展
    """
    from fractions import Fraction
    from decimal import Decimal

    s = str(s0).replace('$', '')
    s = s.replace('=', '')
    s = s.replace(r'\times', '*').replace(r'\div', '/').replace(':', '/')
    s = s.replace('[', '(').replace(']', ')')
    s = re.sub('[ ]+', r'', s)  # 去除所有的空格

    dfrac_digits = r'(?:{\d+}{\d+}|\d\d|{\d+}\d|\d{\d+})'

    def convert(m):
        """小学都是有理数计算，将所有数值转为Fraction类型"""
        s0 = m.group()
        if re.match(r'\d', s0) and 'frac' in s0:
            t = s0[s0.find('frac')+4:]
            if re.match(r'{\d+}{\d+}', t):
                m = re.match(r'(\d+)\\d?frac{(\d+)}{(\d+)}', s0)
            elif re.match(r'\d\d', t):
                m = re.match(r'(\d+)\\d?frac(\d)(\d)', s0)
            elif re.match(r'{\d+}\d', t):
                m = re.match(r'(\d+)\\d?frac{(\d+)}(\d)', s0)
            elif re.match(r'\d{\d+}', t):
                m = re.match(r'(\d+)\\d?frac(\d){(\d+)}', s0)
            else:
                dprint(s0)
                raise ValueError
            a, b, c = map(int, m.groups())
            s = f'Fraction({a * c + b}, {c})'
        elif 'frac' in s0:  # 用分数初始化
            t = s0[s0.find('frac') + 4:]
            if re.match(r'{\d+}{\d+}', t):
                m = re.match(r'{(\d+)}{(\d+)}', t)
            elif re.match(r'\d\d', t):
                m = re.match(r'(\d)(\d)', t)
            elif re.match(r'{\d+}\d', t):
                m = re.match(r'{(\d+)}(\d)', t)
            elif re.match(r'\d{\d+}', t):
                m = re.match(r'(\d){(\d+)}', t)
            else:
                dprint(s0)
                raise ValueError
            s = f'Fraction({m.group(1)}, {m.group(2)})'
        elif '.' in s0:  # 用小数初始化
            s = f'Fraction("{s0}")'
        else:  # 用整数初始化
            s = f'Fraction({s0}, 1)'
        return s

    s = re.sub(r'(\d+\\d?frac' + dfrac_digits + r'|\\d?frac' + dfrac_digits + '|\d+\.\d+|\d+)', convert, s)

    t = eval(s, {'Fraction': Fraction, 'Decimal': Decimal})

    def mystr(a, mode=None):
        """
        :param a:
        :param mode:
            None：分数转为小数显示
            dfrac：分数用\dfrac分数显示
        :return:
        """
        if isinstance(a, Fraction):
            if mode == 'dfrac':
                if a.numerator == 0:
                    return '0'
                elif a.denominator == 1:
                    return str(a.numerator)
                else:
                    return f'$\\dfrac{{{a.numerator}}}{{{a.denominator}}}$'

            d = Decimal(a.numerator) / Decimal(a.denominator)
            return str(d)
        else:
            return str(a)

    return mystr(t, mode)


def generate_equation(pattern, ch2types, mulname='\\times', divname='\\div'):
    r"""
    :param pattern: 模式描述简式
        模式可以用±表示随机加减、※随机乘除、或o随机加减乘除
    :param ch2types: 对模式串中进行替换
    :param mulname: 乘法在明文中的展示，默认是latex效果的\times，也可以改成“×”
    :param divname: 除法在明文中的展示，默认是latex效果的\div，也可以改成“÷”
    :return:

    >> print(generate_equation('f*[(f+f)+f]', {'f': Float}))
    ('0.04\\times[(3.27+8.44)+4.12]', '0.6332')
    """
    # 1、展开模式串
    s0 = pattern
    while '±' in s0: s0 = s0.replace('±', random.choice('+-'), 1)  # ±表示随机加减
    while '※' in s0: s0 = s0.replace('※', random.choice('*/'), 1)  # ※表示随机乘除
    while 'o' in s0: s0 = s0.replace('o', random.choice('+-*/'), 1)  # o表示随机加减乘除

    # 2、计算式子值
    names = dict()  # 命名空间
    for v in ch2types.values():
        names[v.__name__] = v

    ans = None
    while not ans:
        s = s0
        for k, v in ch2types.items():
            while re.search(r'(?<![A-Za-z])' + k + r'(?![A-Za-z])', s):  # TODO 这里应该用正则来分析，而不是文本判断
                s = re.sub(r'(?<![A-Za-z])' + k + r'(?![A-Za-z])', v().defstr(), s, count=1)

        try:
            t = s.replace('[', '(').replace('{', '(').replace(']', ')').replace('}', ')')
            ans = eval(t, names)
        except TypeError:  # 生成失败，不满足约束规则
            continue
        except AttributeError:
            continue
        except Exception as e:  # 遇到特殊情况，直接报错
            raise e

    # 3、生成成功，返回表达式和对应结果
    # 这里匹配并不是非常严谨，要确保都是传入类，首字母大小，且初始化的时候参数里不带原括号
    ls = re.split(r'([A-Z]\w+\(.+?\))', s)
    ls = map(lambda x: eval(f'str({x})', names) if re.match(r'[A-Z]\w+\(.+?\)', x) else x, ls)
    equation = ''.join(ls).replace('*', mulname).replace('/', divname)
    return equation, str(ans)


def demo_generate_equation():
    t = generate_equation('(F + F) / F', {'F': Frac})
    print(t)


def get_decimal3():
    """随机生成一个小数
    :return:
    """
    baseval = choice([randint(1, 9), randint(11, 99), randint(101, 999)], [0.1, 0.7, 0.2])
    if baseval % 10 == 0: baseval += randint(1, 9)  # 末尾不为0
    v = Decimal(baseval)
    r = random.random()
    if v < 10: v /= 10 if r < 0.9 else 100  # 90%除以10，10%除以100
    elif v < 100: v /= 10 if r < 0.8 else 100  # 80%除以10，20%除以100
    else:  v /= 10 if r < 0.1 else 100    # 10%除以10，90%除以100
    if len_decimal(v) > 1 and random.random() < 0.8: v = (v * 10).normalize()
    return v


class Float:
    """本章节通用小数类"""
    def __init__(self, a=None, b=None):
        if a is None and b is None:
            # 最多带两位小数
            f = Fraction(get_decimal2())
        else:
            f = Fraction(a, b)
        self.m, self.n = f.numerator, f.denominator

    def defstr(self):
        """得到定义当前值的表达式"""
        return f'{type(self).__name__}({self.m}, {self.n})'

    def __add__(self, other):
        """重载加法，返回None可以判定为无效除操作"""
        f = Fraction(self.m, self.n) + Fraction(other.m, other.n)
        m, n = f.numerator, f.denominator
        t = Float(m, n)
        if not_simple_decimal3(t): return None
        return t

    def __sub__(self, other):
        """重载减法，返回None可以判定为无效除操作"""
        f = Fraction(self.m, self.n) - Fraction(other.m, other.n)
        m, n = f.numerator, f.denominator
        if m < 0: return  # 加法中非负约束
        t = Float(m, n)
        if not_simple_decimal3(t): return None
        return t

    def __mul__(self, other):
        """重载除法，返回None可以判定为无效除操作"""
        if not_simple_decimal2(other): return None
        f = Fraction(self.m, self.n) * Fraction(other.m, other.n)
        m, n = f.numerator, f.denominator
        t = Float(m, n)
        if not_simple_decimal3(t): return None
        return t

    def __truediv__(self, other):
        """重载除法，返回None可以判定为无效除操作"""
        if other.m == 0: return
        if not_simple_decimal2(other): return None
        f = Fraction(self.m, self.n) / Fraction(other.m, other.n)
        m, n = f.numerator, f.denominator
        t = Float(m, n)
        if not_simple_decimal3(t): return None
        return t

    def __str__(self):
        """明文展示的规则"""
        return str(Decimal(self.m) / Decimal(self.n))


def not_simple_decimal2(x):
    """一个数中的数字长度超过3"""
    if isinstance(x, Fraction):
        x = Decimal(x.numerator) / Decimal(x.denominator)
    elif isinstance(x, Frac):
        x = Decimal(x.m) / Decimal(x.n)
    return len(str(x).replace('.', '').lstrip('0')) > 2


def not_simple_decimal3(x):
    """一个数中的数字长度超过3"""
    if isinstance(x, Fraction):
        x = Decimal(x.numerator) / Decimal(x.denominator)
    elif isinstance(x, Frac):
        x = Decimal(x.m) / Decimal(x.n)
    return len(str(x).replace('.', '').lstrip('0')) > 3


if __name__ == '__main__':
    timer = Timer(__file__, start_now=True)

    Frac.__str__ = Frac.latex4
    print(generate_equation('F*F*F', {'F': Frac}))

    # 重复题判断(os.path.abspath('.'))

    timer.stop_and_report()
