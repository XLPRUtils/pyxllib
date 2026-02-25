#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 10:51

import re


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


def digits2chinese(n):
    """TODO：目前处理范围有限，还需要再扩展
    """
    s = '十一二三四五六七八九'
    if n == 0:
        return '零'
    elif n <= 10:
        return s[n % 10]
    elif n < 20:
        return '十' + s[n % 10]
    elif n < 100:
        return s[n // 10] + s[n % 10]
    else:
        raise NotImplementedError


def chinese2digits(chinese_str):
    """把汉字变为阿拉伯数字
    https://blog.csdn.net/leon_wzm/article/details/78963082
    """

    def inner(m):
        t = m.group()
        if t is None or t.strip() == '':
            raise ValueError(f'input error for {chinese_str}')
        t = t.strip()
        t = t.replace('百十', '百一十')
        common_used_numerals = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
                                '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
                                '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
        total = 0
        r = 1  # right，右边一位的值
        for i in range(len(t) - 1, -1, -1):  # 从右往左一位一位读取
            val = common_used_numerals.get(t[i])  # 使用get不存在会返回None
            if val is None:
                # dprint(chinese_str)
                return chinese_str
                # raise ValueError(f't[i]={t[i]} can not be accepted.')
            if val >= 10 and i == 0:  # 最左位是“十百千万亿”这样的单位数词
                if val > r:  # 一般是“十三”这类会进入这个if分支
                    r = val
                    total += val
                else:
                    r *= val
            elif val >= 10:
                if val > r:  # 跳了单位数词（正常情况都会跳），例如 一万一百零三
                    r = val
                else:  # 单位数词叠加情况，例如 一千亿
                    r *= val
            else:  # 不是单位数词的数词，如果上一步是单位数词，增加一个单位量
                total += r * val
        return str(total)

    return re.sub(r'[零一二两三四五六七八九十百千万亿]+', inner, chinese_str)


def int2myalphaenum(n):
    """
    :param n: 0~52的数字
    """
    if 0 <= n <= 52:
        return '_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[n]
    else:
        print('警告：不在处理范围内的数值', n)
        raise ValueError


def digit2weektag(d):
    """ 输入数字1~7，转为“周一~周日”

    >>> digit2weektag(1)
    '周一'
    >>> digit2weektag('7')
    '周日'
    """
    d = int(d)
    if 1 <= d <= 7:
        return '周' + '一二三四五六日'[d - 1]
    else:
        raise ValueError


def fullwidth2halfwidth(ustring):
    """ 把字符串全角转半角

    python3环境下的全角与半角转换代码和测试_大数据挖掘SparkExpert的博客-CSDN博客:
    https://blog.csdn.net/sparkexpert/article/details/82749207

    >>> fullwidth2halfwidth("你好ｐｙｔｈｏｎａｂｄａｌｄｕｉｚｘｃｖｂｎｍ")
    '你好pythonabdalduizxcvbnm'
    """
    ss = []
    for s in ustring:
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            ss.append(chr(inside_code))
    return ''.join(ss)


def fullwidth2halfwidth2(ustring):
    """ 不处理标点符号的版本

    >>> fullwidth2halfwidth2("你好ｐｙｔｈｏｎａｂｄａ，ｌｄｕｉｚｘｃｖｂｎｍ")
    '你好pythonabda，lduizxcvbnm'
    """
    ss = []
    for s in ustring:
        for uchar in s:
            if uchar in '：；！（），？＂．':
                ss.append(uchar)
            else:
                inside_code = ord(uchar)
                if inside_code == 12288:  # 全角空格直接转换
                    inside_code = 32
                elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
                    inside_code -= 65248
                ss.append(chr(inside_code))
    return ''.join(ss)


def halfwidth2fullwidth(ustring):
    """ 把字符串全角转半角

    >>> halfwidth2fullwidth("你好ｐｙｔｈｏｎａｂｄａｌｄｕｉｚｘｃｖｂｎｍ")
    '你好ｐｙｔｈｏｎａｂｄａｌｄｕｉｚｘｃｖｂｎｍ'
    """
    ss = []
    for s in ustring:
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 32:  # 全角空格直接转换
                inside_code = 12288
            elif 33 <= inside_code <= 126:  # 全角字符（除空格）根据关系转化
                inside_code += 65248
            ss.append(chr(inside_code))
    return ''.join(ss)
