#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

from pyxllib.prog.newbie import round_int


class StrDecorator:
    """将函数的返回值字符串化，仅调用朴素的str字符串化

    装饰器开发可参考： https://mp.weixin.qq.com/s/Om98PpncG52Ba1ZQ8NIjLA
    """

    def __init__(self, func):
        self.func = func  # 使用self.func可以索引回原始函数名称
        self.last_raw_res = None  # last raw result，上一次执行函数的原始结果

    def __call__(self, *args, **kwargs):
        self.last_raw_res = self.func(*args, **kwargs)
        return str(self.last_raw_res)


class PrintDecorator:
    """将函数返回结果直接输出"""

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        s = self.func(*args, **kwargs)
        print(s)
        return s  # 输出后仍然会返回原函数运行值


def binary_cut_str(s, fmt='0'):
    """180801坤泽：“二分”切割字符串
    :param s: 要截取的全字符串
    :param fmt: 截取格式，本来是想只支持0、1的，后来想想支持23456789也行
        0：左边一半
        1：右边的1/2
        2：右边的1/3
        3：右边的1/4
        ...
        9：右边的1/10
    :return: 截取后的字符串

    >>> binary_cut_str('1234', '0')
    '12'
    >>> binary_cut_str('1234', '1')
    '34'
    >>> binary_cut_str('1234', '10')
    '3'
    >>> binary_cut_str('123456789', '20')
    '7'
    >>> binary_cut_str('123456789', '210')  # 向下取整，'21'获得了9，然后'0'取到空字符串
    ''
    """
    for t in fmt:
        t = int(t)
        n = len(s) // (1 + max(1, t))
        if t == 0:
            s = s[:n]
        else:
            s = s[(len(s) - n):]
    return s


def digits2roman(d):
    """
    >>> digits2roman(2)
    'Ⅱ'
    >>> digits2roman(12)
    'Ⅻ'
    """
    rmn = '~ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ'  # roman数字number的缩写

    d = int(d)  # 确保是整数类型
    if d <= 12:
        return rmn[d]
    else:
        raise NotImplementedError


def roman2digits(d):
    """
    >>> roman2digits('Ⅱ')
    2
    >>> roman2digits('Ⅻ')
    12
    """
    rmn = '~ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ'
    if d in rmn:
        return rmn.index(d)
    else:
        raise NotImplemented


def digits2circlednumber(d):
    d = int(d)
    if 0 < d <= 20:
        return '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'[d - 1]
    else:
        raise NotImplemented


def circlednumber2digits(d):
    t = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'
    if d in t:
        return t.index(d) + 1
    else:
        raise NotImplemented


def endswith(s, tags):
    """除了模拟str.endswith方法，输入的tag也可以是可迭代对象

    >>> endswith('a.dvi', ('.log', '.aux', '.dvi', 'busy'))
    True
    """
    if isinstance(tags, str):
        return s.endswith(tags)
    elif isinstance(tags, (list, tuple)):
        for t in tags:
            if s.endswith(t):
                return True
    else:
        raise TypeError
    return False


def xldictstr(d, key_value_delimit='=', item_delimit=' '):
    """将一个字典转成字符串"""
    res = []
    for k, v in d.items():
        res.append(str(k) + key_value_delimit + str(v).replace('\n', r'\n'))
    res = item_delimit.join(res)
    return res


def findnth(haystack, needle, n):
    """https://stackoverflow.com/questions/1883980/find-the-nth-occurrence-of-substring-in-a-string"""
    if n < 0:
        n += haystack.count(needle)
    if n < 0:
        return -1

    parts = haystack.split(needle, n + 1)
    if len(parts) <= n + 1:
        return -1
    return len(haystack) - len(parts[-1]) - len(needle)


def refine_digits_set(digits):
    """美化连续数字的输出效果

    >>> refine_digits_set([210, 207, 207, 208, 211, 212])
    '207,208,210-212'
    """
    arr = sorted(list(set(digits)))  # 去重
    n = len(arr)
    res = ''
    i = 0
    while i < n:
        j = i + 2
        if j < n and arr[i] + 2 == arr[j]:
            while j < n and arr[j] - arr[i] == j - i:
                j += 1
            j = j if j < n else n - 1
            res += str(arr[i]) + '-' + str(arr[j]) + ','
            i = j + 1
        else:
            res += str(arr[i]) + ','
            i += 1
    return res[:-1]  # -1是去掉最后一个','


def del_tail_newline(s):
    """删除末尾的换行"""
    if len(s) > 1 and s[-1] == '\n':
        s = s[:-1]
    return s


def bracket_match(s, idx):
    """括号匹配位置
    这里以{、}为例，注意也要适用于'[]', '()'
    >>> bracket_match('{123}', 0)
    4
    >>> bracket_match('0{23{5}}89', 1)
    7
    >>> bracket_match('0{23{5}}89', 7)
    1
    >>> bracket_match('0{23{5}78', 1) is None
    True
    >>> bracket_match('0{23{5}78', 20) is None
    True
    >>> bracket_match('0[2[4]{7}]01', 9)
    1
    >>> bracket_match('0{[34{6}89}', -4)
    5
    """
    key = '{[(<>)]}'
    try:
        if idx < 0:
            idx += len(s)
        ch1 = s[idx]
        idx1 = key.index(ch1)
    except ValueError:  # 找不到ch1
        return None
    except IndexError:  # 下标越界，表示没有匹配到右括号
        return None
    idx2 = len(key) - idx1 - 1
    ch2 = key[idx2]
    step = 1 if idx2 > idx1 else -1
    cnt = 1
    i = idx + step
    if i < 0:
        i += len(s)
    while 0 <= i < len(s):
        if s[i] == ch1:
            cnt += 1
        elif s[i] == ch2:
            cnt -= 1
        if cnt == 0:
            return i
        i += step
    return None


def bracket_match2(s, idx):
    r"""与“bracket_match”相比，会考虑"\{"转义字符的影响

    >>> bracket_match2('a{b{}b}c', 1)
    6
    >>> bracket_match2('a{b{\}b}c}d', 1)
    9
    """
    key = '{[(<>)]}'
    try:
        if idx < 0:
            idx += len(s)
        ch1 = s[idx]
        idx1 = key.index(ch1)
    except ValueError:  # 找不到ch1
        return None
    except IndexError:  # 下标越界，表示没有匹配到右括号
        return None
    idx2 = len(key) - idx1 - 1
    ch2 = key[idx2]
    step = 1 if idx2 > idx1 else -1
    cnt = 1
    i = idx + step
    if i < 0:
        i += len(s)
    while 0 <= i < len(s):
        if i and s[i - 1] == '\\':
            pass
        elif s[i] == ch1:
            cnt += 1
        elif s[i] == ch2:
            cnt -= 1
        if cnt == 0:
            return i
        i += step
    return None


def latexstrip(s):
    """latex版的strip"""
    return s.strip('\t\n ~')


def add_quote(s):
    return f'"{s}"'


def fold_dict(d, m=5):
    """ 将字典折叠为更紧凑的排版格式

    :param d: 一个字典对象
    :param m: 按照每行放m个元素重排
    :return: 重排后的字典内容
    """
    vals = [f"'{k}': {v}" for k, v in d.items()]
    line = [', '.join(vals[i:i + 5]) for i in range(0, len(vals), m)]
    return '{' + ',\n'.join(line) + '}'


def format_big_decimal(value):
    """ 较大的十进制数值的美化展示

    :param float|int value: 数值
    :return: 整数值 + 单位
        单位说明
            kilo- =  1e3   = K-
            mega- =  1e6   = M-
            giga- =  1e9   = G-
            tera- =  1e12  = T-
            peta- =  1e15  = P-
            exa- =   1e18  = E-
            zetta- = 1e21  = Z-
            yotta- = 1e24  = Y-
    """
    x, i, unit = int(value), 0, 'KMGTPEZY'
    while x > 1000:
        i += 1
        x = round_int(x / 1000)
    return f'{x}{unit[i - 1]}'
