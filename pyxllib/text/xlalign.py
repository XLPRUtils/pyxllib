#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/01


"""很久以前搞得一个，按照中文域宽对齐相关操作
也是 pyxllib v0.0.xx 最底层基础的__str.py 文件
"""

import copy
import re
import textwrap

from pyxllib.prog.basic import len_in_dim2, GrowingList


def strwidth(s):
    """ string width

    中英字符串实际宽度
    >>> strwidth('ab')
    2
    >>> strwidth('a⑪中⑩')
    7

    ⑩等字符的宽度还是跟字体有关的，不过在大部分地方好像都是域宽2，目前算法问题不大
    """
    try:
        res = len(s.encode('gbk'))
    except UnicodeEncodeError:
        count = len(s)
        for x in s:
            if ord(x) > 127:
                count += 1
        res = count
    return res


def strwidth_proc(s, fmt='r', chinese_char_width=1.8):
    """ 此函数可以用于每个汉字域宽是w=1.8等奇怪的情况

    为了让字符串域宽为一个整数，需要补充中文空格，会对原始字符串进行修改。
    故返回值有2个，第1个是修正后的字符串s，第2个是实际宽度w。

    :param s: 一个字符串
    :param fmt: 目标对齐格式
    :param chinese_char_width: 每个汉字字符宽度
    :return: (s, w)
        s: 修正后的字符串值s
        w: 修正后字符串的实际宽度

    >>> strwidth_proc('哈哈a', chinese_char_width=1.8)
    ('　　　哈哈a', 10)
    """
    # 1 计算一些参数值
    s = str(s)  # 确保是字符串类型
    l1 = len(s)
    l2 = strwidth(s)
    y = l2 - l1  # 中文字符数
    x = l1 - y  # 英文字符数
    ch = chr(12288)  # 中文空格
    w = x + y * chinese_char_width  # 当前字符串宽度
    # 2 计算需要补充t个中文空格
    error = 0.05  # 允许误差范围
    t = 0  # 需要补充中文字符数
    while error < w % 1 < 1 - error:  # 小数部分超过误差
        t += 1
        w += chinese_char_width
    # 3 补充中文字符
    if t:
        if fmt == 'r':
            s = ch * t + s
        elif fmt == 'l':
            s = s + ch * t
        else:
            s = ch * (t - t // 2) + s + ch * (t // 2)
    return s, int(w)


def realign(text, least_blank=4, tab2blank=4, support_chinese=False, sep=None):
    r""" 一列文本的对齐
    :param text: 一段文本
        支持每行列数不同
    :param least_blank: 每列最少间距空格数
    :param tab2blank:
    :param support_chinese: 支持中文域宽计算
    :param sep: 每列分隔符，默认为least_blank个空格
    :return: 对齐美化的一段文本

    >>> realign('  Aget      keep      hold         show\nmaking    selling    giving    collecting')
    'Aget      keep       hold      show\nmaking    selling    giving    collecting'
    """
    # 1 预处理
    s = text.replace('\t', ' ' * tab2blank)
    s = re.sub(' {' + str(least_blank) + ',}', r'\t', s)  # 统一用\t作为分隔符
    lenfunc = strwidth if support_chinese else len
    if sep is None: sep = ' ' * least_blank

    # 2 计算出每一列的最大宽度
    lines = s.splitlines()
    n = len(lines)
    max_width = GrowingList()  # 因为不知道有多少列，用自增长的list来存储每一列的最大宽度
    for i, line in enumerate(lines):
        line = line.strip().split('\t')
        m = len(line)
        for j in range(m): max_width[j] = max(max_width[j] or 0, lenfunc(line[j]))
        lines[i] = line
    if len(max_width) == 1: return '\n'.join(map(lambda x: x[0], lines))

    # 3 重组内容
    for i, line in enumerate(lines):
        for j in range(len(line) - 1): line[j] += ' ' * (max_width[j] - lenfunc(line[j]))  # 注意最后一列就不用加空格了
        lines[i] = sep.join(line)
    return '\n'.join(lines)


def listalign(ls, fmt='r', *, width=None, fillchar=' ', prefix='', suffix='', chinese_char_width=2):
    """文档： https://blog.csdn.net/code4101/article/details/80985218（不过文档有些过时了）

    listalign列表对齐
    py3中str的len是计算字符数量，例如len('ab') --> 2， len('a中b') --> 3。
    但在对齐等操作中，是需要将每个汉字当成宽度2来处理，计算字符串实际宽度的。
    所以我们需要开发一个strwidth函数，效果： strwidth('ab') --> 2，strwidth('a中b') --> 4。

    :param ls:
        要处理的列表，会对所有元素调用str处理，确保全部转为string类型
            且会将换行符转为\n显示
    :param fmt: （format）
        l: left，左对齐
        c: center，居中
        r: right，右对齐
        多个字符: 扩展fmt长度跟ls一样，每一个元素单独设置对齐格式。如果fmt长度小于ls，则扩展的格式按照fmt[-1]设置
    :param width:
        None或者设置值小于最长字符串: 不设域宽，直接按照最长的字符串为准
    :param fillchar: 填充字符
    :param prefix: 添加前缀
    :param suffix: 添加后缀
    :param chinese_char_width: 每个汉字字符宽度

    :return:
        对齐后的数组ls，每个元素会转为str类型

    >>> listalign(['a', '哈哈', 'ccd'])
    ['   a', '哈哈', ' ccd']
    >>> listalign(['a', '哈哈', 'ccd'], chinese_char_width=1.8)
    ['        a', '　　　哈哈', '      ccd']
    """
    # 1 处理fmt数组
    if len(fmt) == 1:
        fmt = [fmt] * len(ls)
    elif len(fmt) < len(ls):
        fmt = list(fmt) + [fmt[-1]] * (len(ls) - len(fmt))

    # 2 算出需要域宽
    if chinese_char_width == 2:
        strs = [str(x).replace('\n', r'\n') for x in ls]  # 存储转成字符串的元素
        lens = [strwidth(x) for x in strs]  # 存储每个元素的实际域宽
    else:
        strs = []  # 存储转成字符串的元素
        lens = []  # 存储每个元素的实际域宽
        for i, t in enumerate(ls):
            t, n = strwidth_proc(t, fmt[i], chinese_char_width)
            strs.append(t)
            lens.append(n)
    w = max(lens)
    if width and isinstance(width, int) and width > w:
        w = width

    # 3 对齐操作
    for i, s in enumerate(strs):
        if fmt[i] == 'r':
            strs[i] = fillchar * (w - lens[i]) + strs[i]
        elif fmt[i] == 'l':
            strs[i] = strs[i] + fillchar * (w - lens[i])
        elif fmt[i] == 'c':
            t = w - lens[i]
            strs[i] = fillchar * (t - t // 2) + strs[i] + fillchar * (t // 2)
        strs[i] = prefix + strs[i] + suffix
    return strs


def arr_hangclear(arr, depth=None):
    """ 清除连续相同值，简化表格内容
    >> arr_hangclear(arr, depth=2)
    原表格：
        A  B  D
        A  B  E
        A  C  E
        A  C  E
    新表格：
        A  B  D
              E
           C  E
              E

    :param arr: 二维数组
    :param depth: 处理列上限
        例如depth=1，则只处理第一层
        depth=None，则处理所有列

    >>> arr_hangclear([[1, 2, 4], [1, 2, 5], [1, 3, 5], [1, 3, 5]])
    [[1, 2, 4], ['', '', 5], ['', 3, 5], ['', '', 5]]
    >>> arr_hangclear([[1, 2, 4], [1, 2, 5], [2, 2, 5], [1, 2, 5]])
    [[1, 2, 4], ['', '', 5], [2, 2, 5], [1, 2, 5]]
    """
    m = depth or len_in_dim2(arr) - 1
    a = copy.deepcopy(arr)

    # 算法原理：从下到上，从右到左判断与上一行重叠了几列数据
    for i in range(len(arr) - 1, 0, -1):
        for j in range(m):
            if a[i][j] == a[i - 1][j]:
                a[i][j] = ''
            else:
                break
    return a


def arr2table(arr, rowmerge=False):
    """ 数组转html表格代码

    :param arr:  需要处理的数组
    :param rowmerge: 行单元格合并
    :return: html文本格式的<table>

    这个arr2table是用来画合并单元格的
    >> browser(arr2table([['A', 1, 'a'], ['', 2, 'b'], ['B', 3, 'c'], ['', '', 'd'], ['', 5, 'e']], True), 'a.html')
    效果图：http://i1.fuimg.com/582188/c452f40b5a072f8d.png
    """
    n = len(arr)
    m = len_in_dim2(arr)
    res = ['<table border="1"><tbody>']
    for i, line in enumerate(arr):
        res.append('<tr>')
        for j, ele in enumerate(line):
            if rowmerge:
                if ele != '':
                    cnt = 1
                    while i + cnt < n and arr[i + cnt][j] == '':
                        for k in range(j - 1, -1, -1):
                            if arr[i + cnt][k] != '':
                                break
                        else:
                            cnt += 1
                            continue
                        break
                    if cnt > 1:
                        res.append(f'<td rowspan="{cnt}">{ele}</td>')
                    else:
                        res.append(f'<td>{ele}</td>')
                elif j == m - 1:
                    res.append(f'<td>{ele}</td>')
            else:
                res.append(f'<td>{ele}</td>')
        res.append('</tr>')
    res.append('</tbody></table>')
    return ''.join(res)


def east_asian_len(s, ambiguous_width=None):
    import pandas.io.formats.format as fmt
    return fmt.EastAsianTextAdjustment().len(s)


def east_asian_shorten(s, width=50, placeholder='...'):
    """考虑中文情况下的域宽截断

    :param s: 要处理的字符串
    :param width: 宽度上限，仅能达到width-1的宽度
    :param placeholder: 如果做了截断，末尾补足字符

    # width比placeholder还小
    >>> east_asian_shorten('a', 2)
    'a'
    >>> east_asian_shorten('a啊b'*4, 3)
    '..'
    >>> east_asian_shorten('a啊b'*4, 4)
    '...'

    >>> east_asian_shorten('a啊b'*4, 5, '...')
    'a...'
    >>> east_asian_shorten('a啊b'*4, 11)
    'a啊ba啊...'
    >>> east_asian_shorten('a啊b'*4, 16, '...')
    'a啊ba啊ba啊b...'
    >>> east_asian_shorten('a啊b'*4, 18, '...')
    'a啊ba啊ba啊ba啊b'
    """
    # 一、如果字符串本身不到width设限，返回原值
    s = textwrap.shorten(s, width * 3, placeholder='')  # 用textwrap的折行功能，尽量不删除文本
    n = east_asian_len(s)
    if n < width: return s

    # 二、如果输入的width比placeholder还短
    width -= 1
    m = east_asian_len(placeholder)
    if width <= m:
        return placeholder[:width]

    # 三、需要添加 placeholder
    # 1 计算长度
    width -= m

    # 2 截取s
    try:
        s = s.encode('gbk')[:width].decode('gbk', errors='ignore')
    except UnicodeEncodeError:
        i, count = 0, m
        while i < n and count <= width:
            if ord(s[i]) > 127:
                count += 2
            else:
                count += 1
            i += 1
        s = s[:i]

    return s + placeholder
