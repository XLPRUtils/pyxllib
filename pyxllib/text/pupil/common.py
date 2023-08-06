#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 22:56

"""
文本处理、常用正则匹配模式

下面大量的函数前缀含义：
grp，generate regular pattern，生成正则模式字符串
grr，generate regular replace，生成正则替换目标格式
"""

import base64
import bisect
import collections
import io
import logging
import os
import re
import sys

from pyxllib.prog.newbie import RunOnlyOnce
from pyxllib.prog.pupil import dprint
from pyxllib.text.newbie import circlednumber2digits, digits2circlednumber, roman2digits, digits2roman


def shorten(s, width=200, placeholder='...'):
    """
    :param width: 这个长度是上限，即使用placeholder时的字符串总长度也在这个范围内

    >>> shorten('aaa', 10)
    'aaa'
    >>> shorten('hell world! 0123456789 0123456789', 11)
    'hell wor...'
    >>> shorten("Hello  world!", width=12)
    'Hello world!'
    >>> shorten("Hello  world!", width=11)
    'Hello wo...'
    >>> shorten('0123456789 0123456789', 2, 'xyz')  # 自己写的shorten
    'xy'

    注意textwrap.shorten的缩略只针对空格隔开的单词有效，我这里的功能与其不太一样
    >>> textwrap.shorten('0123456789 0123456789', 11)  # 全部字符都被折叠了
    '[...]'
    >>> shorten('0123456789 0123456789', 11)  # 自己写的shorten
    '01234567...'
    """
    s = re.sub(r'\s+', ' ', str(s))
    n, m = len(s), len(placeholder)
    if n > width:
        s = s[:max(width - m, 0)] + placeholder
    return s[:width]  # 加了placeholder在特殊情况下也会超，再做个截断最保险

    # return textwrap.shorten(str(s), width)


def strfind(fullstr, objstr, *, start=None, times=0, overlap=False):
    r""" 进行强大功能扩展的的字符串查找函数

    TODO 性能有待优化

    :param fullstr: 原始完整字符串
    >>> strfind('aabbaabb', 'bb')  # 函数基本用法
    2

    :param objstr: 需要查找的目标字符串，可以是一个list或tuple
    TODO 有空看下AC自动机，看这里是否可以优化提速，或者找现成的库接口
    >>> strfind('bbaaaabb', 'bb') # 查找第1次出现的位置
    0
    >>> strfind('aabbaabb', 'bb', times=1) # 查找第2次出现的位置
    6
    >>> strfind('aabbaabb', 'cc') # 不存在时返回-1
    -1
    >>> strfind('aabbaabb', ['aa', 'bb'], times=2)
    4

    :param start: 起始查找位置。默认值为0，当times<0时start的默认值为-1。
    >>> strfind('aabbaabb', 'bb', start=2) # 恰好在起始位置
    2
    >>> strfind('aabbaabb', 'bb', start=3)
    6
    >>> strfind('aabbaabb', ['aa', 'bb'], start=5)
    6

    :param times: 定位第几次出现的位置，默认值为0，即从前往后第1次出现的位置。
        如果是负数，则反向查找，并返回的是目标字符串的起始位置。
    >>> strfind('aabbaabb', 'aa', times=-1)
    4
    >>> strfind('aabbaabb', 'aa', start=5, times=-1)
    4
    >>> strfind('aabbaabb', 'aa', start=3, times=-1)
    0
    >>> strfind('aabbaabb', 'bb', start=7, times=-1)
    6

    :param overlap: 重叠情况是否重复计数
    >>> strfind('aaaa', 'aa', times=1)  # 默认不计算重叠部分
    2
    >>> strfind('aaaa', 'aa', times=1, overlap=True)
    1

    >>> strfind(r'\item=\item+', (r'\item', r'\test'), start=1)
    6
    """

    def nonnegative_min_value(*arr):
        """计算出最小非负整数，如果没有非负数，则返回-1"""
        arr = tuple(filter(lambda x: x >= 0, arr))
        return min(arr) if arr else -1

    def nonnegative_max_value(*arr):
        """计算出最大非负整数，如果没有非负数，则返回-1"""
        arr = tuple(filter(lambda x: x >= 0, arr))
        return max(arr) if arr else -1

    # 1 根据times不同，start的初始默认值设置方式也不同
    if times < 0 and start is None:
        start = len(fullstr) - 1  # 反向查找start设到末尾字符-1
    if start is None:
        start = 0  # 正向查找start设为0
    p = -1  # 记录答案位置，默认找不到

    # 2 单串匹配
    if isinstance(objstr, str):  # 单串匹配
        offset = 1 if overlap else len(objstr)  # overlap影响每次偏移量

        # A、正向查找
        if times >= 0:
            p = start - offset
            for _ in range(times + 1):
                p = fullstr.find(objstr, p + offset)
                if p == -1:
                    return -1

        # B、反向查找
        else:
            p = start + offset + 1
            for _ in range(-times):
                p = fullstr.rfind(objstr, 0, p - offset)
                if p == -1:
                    return -1

    # 3 多模式匹配（递归调用，依赖单串匹配功能）
    else:
        # A、正向查找
        if times >= 0:
            p = start - 1
            for _ in range(times + 1):
                # 把每个目标串都找一遍下一次出现的位置，取最近的一个
                #   因为只找第一次出现的位置，所以overlap参数传不传都没有影响
                # TODO 需要进行性能对比分析，有必要的话后续可以改AC自动机实现多模式匹配
                ls = tuple(map(lambda x: strfind(fullstr, x, start=p + 1, overlap=overlap), objstr))
                p = nonnegative_min_value(*ls)
                if p == -1:
                    return -1

        # B、反向查找
        else:
            p = start + 1
            for _ in range(-times):  # 需要循环处理的次数
                # 使用map对每个要查找的目标调用strfind
                ls = tuple(map(lambda x: strfind(fullstr, x, start=p - 1, times=-1, overlap=overlap), objstr))
                p = nonnegative_max_value(*ls)
                if p == -1:
                    return -1

    return p


def findspan(src, sub, start=0, end=None):
    """ str.find的封装

    :param sub:
        str，普通的字符串查找
        re.Pattern，正则模式的查找
    :return: (start, end)
        找不到的时候返回 (-1, -1)
        否则返回区间的左开右闭位置
    """
    if end is None:
        end = len(src)

    if isinstance(sub, str):
        pos = src.find(sub, start, end)
    elif isinstance(sub, re.Pattern):
        pattern = sub
        m = pattern.search(src[start:end])
        if m:
            pos = m.start() + start
            sub = m.group()
        else:
            pos = -1
    else:
        raise TypeError

    if pos == -1:
        return -1, -1
    else:
        return pos, pos + len(sub)


def substr_count(src, sub, overlape=False):
    """ 判断字符串src中符合pattern的字串有几个 """
    if overlape:
        raise NotImplementedError
    else:
        if isinstance(sub, str):
            cnt = src.count(sub)
        elif isinstance(sub, re.Pattern):
            cnt = len(sub.findall(src))
        else:
            raise TypeError

    return cnt


class Stdout:
    """重定向标准输出流，切换print标准输出位置

    使用with语法调用
    """

    def __init__(self, path=None, mode='w'):
        """
        :param path: 可选参数
            如果是一个合法的文件名，在__exit__时，会将结果写入文件
            如果不合法不报错，只是没有功能效果
        :param mode: 写入模式
            'w': 默认模式，直接覆盖写入
            'a': 追加写入
        """
        self.origin_stdout = sys.stdout
        self._path = path
        self._mode = mode
        self.strout = io.StringIO()
        self.result = None

    def __enter__(self):
        sys.stdout = self.strout
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.origin_stdout
        self.result = str(self)

        # 如果输入的是一个合法的文件名，则将中间结果写入
        if not self._path:
            return

        try:
            with open(self._path, self._mode) as f:
                f.write(self.result)
        except TypeError as e:
            logging.exception(e)
        except FileNotFoundError as e:
            logging.exception(e)

        self.strout.close()

    def __str__(self):
        """在这个期间获得的文本内容"""
        if self.result:
            return self.result
        else:
            return self.strout.getvalue()


def int2myalphaenum(n):
    """
    :param n: 0~52的数字
    """
    if 0 <= n <= 52:
        return '_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[n]
    else:
        print('警告：不在处理范围内的数值', n)
        raise ValueError


def ensure_gbk(s):
    """检查一个字符串的所有内容是否能正常转为gbk，
    如果不能则ignore掉不能转换的部分"""
    try:
        s.encode('gbk')
    except UnicodeEncodeError:
        origin_s = s
        s = s.encode('gbk', errors='ignore').decode('gbk')
        print('警告：字符串存在无法转为gbk的字符', origin_s, s)
    return s


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


class ContentPartSpliter:
    """ 文本内容分块处理 """

    @classmethod
    def multi_blank_lines(cls, content, leastlines=2):
        """ 用多个空行隔开的情况

        :param leastlines: 最少2个空行隔开，为新的一块内容
        """
        fmt = r'\n{' + str(leastlines) + ',}'
        parts = [x.strip() for x in re.split(fmt, content)]
        parts = list(filter(bool, parts))  # 删除空行
        return parts


class ContentLine(object):
    """ 用行数的特性分析一段文本 """

    def __init__(self, content):
        """用一段文本初始化"""
        self.content = content  # 原始文本
        self.linepos = list()  # linepos[i-1] = v：第i行终止位置（\n）所在下标为v
        for i in range(len(self.content)):
            if self.content[i] == '\n':
                self.linepos.append(i)
        self.linepos.append(len(self.content))
        self.lines = self.content.splitlines()  # 每一行的文本内容

    def line_start_pos(self, line):
        """第line行的其实pos位置"""
        pass

    def lines_num(self):
        """返回总行数"""
        return self.content.count('\n')

    def match_lines(self, pattern):
        """返回符合正则规则的行号

        180515扩展： pattern也能输入一个函数
        """
        # 1 定义函数句柄
        if not callable(pattern):
            def f(s):
                return re.search(pattern, s)
        else:
            f = pattern
        # 2 循环判断
        res = list()
        for i, line in enumerate(self.lines):
            if f(line):
                res.append(i)
        return res

    def in_line(self, ob):
        """输入关键词ob，返回行号"""

        if hasattr(ob, 'span'):
            return self.in_line(ob.span()[0])
        elif isinstance(ob, int):
            "如果给入一个下标值，如23，计算第23个字符处于原文中第几行"
            return bisect.bisect_right(self.linepos, ob - 1) + 1
        elif isinstance(ob, str):
            "输入一段文本，判断该文中有哪些行与该行内容相同"
            res = list()
            for i, line in enumerate(self.lines):
                if line == ob:
                    res.append(i + 1)
            return res
        elif isinstance(ob, (list, tuple, collections.Iterable)):
            return list(map(self.in_line, ob))
        else:
            raise ValueError(f'类型错误 {type(ob)}')

    def regular_search(self, re_str):
        """同InLine，但是支持正则搜索"""
        return self.in_line(re.finditer(re_str, self.content))

    def lines_content(self, lines) -> str:
        """返回lines集合中数字所对行号的所有内容

        注意输入的lines起始编号是1
        """
        lines = sorted(set(lines))  # 去重
        res = map(lambda n: '{:6} {}'.format(n, self.lines[n - 1]), lines)
        return '\n'.join(res)

    def __str__(self):
        return self.content


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


def briefstr(s):
    """对文本内容进行一些修改，从而简化其内容，提取关键信息
    一般用于字符串近似对比
    """
    # 1 删除所有空白字符
    # debuglib.dprint(debuglib.typename(s))
    s = re.sub(r'\s+', '', s)
    # 2 转小写字符
    s = s.casefold()
    return s


@RunOnlyOnce
def grp_bracket(depth=0, left='{', right=None, inner=False):
    r"""括号匹配，默认花括号匹配，也可以改为圆括号、方括号匹配。

    效果类似于“{.*?}”，
    但是左右花括号是确保匹配的，有可选参数可以提升支持的嵌套层级，
    数字越大匹配嵌套能力越强，但是速度性能会一定程度降低。
    例如“grp_bracket(5)”。

    :param depth: 括号递归深度
    :param left: 左边字符：(、[、{
    :param right: 右边字符
    :param inner: 默认只是返回匹配的正则表达式，不编组
        如果设置inner=True，则会对括号内的内容编组
        该功能用来代替原来的BRACE5等机制

    :return:

    先了解一下正则常识：
    >>> re.sub(r'[^\[\]]', r'', r'a[b]a[]') # 删除非方括号
    '[][]'
    >>> re.sub(r'[^\(\)]', r'', r'a(b)a()') # 删除非圆括号
    '()()'
    >>> re.sub(r'[^()]', r'', r'a(b)a()') # 不用\也可以
    '()()'

    该函数使用效果：
    >>> re.sub(grp_bracket(5), r'', r'x{aaa{b{d}b}ccc{d{{}e}ff}gg}y')
    'xy'
    >>> re.sub(grp_bracket(5, '(', ')'), r'', r'x(aaa(b(d)b)ccc(d(()e)ff)gg)y')
    'xy'
    >>> re.sub(grp_bracket(5, '[', ']'), r'', r'x[aaa[b[d]b]ccc[d[[]e]ff]gg]y')
    'xy'
    """
    # 用a, b简化引用名称
    a, b = left, right
    b = b or {'(': ')', '[': ']', '{': '}'}[a]
    # 特殊符号需要转义
    if a in '([':
        a = '\\' + a
    if b in ')]':
        b = '\\' + b
    c = f'[^{a}{b}]'
    # 建立匹配素材
    pattern_0 = f'{a}{c}*{b}'
    pat_left = f'{a}(?:{c}|'
    pat_right = f')*{b}'

    # 生成匹配规则的函数
    def gen(pattern, depth=0):
        while depth:
            pattern = pat_left + pattern + pat_right
            depth -= 1
        return pattern

    s = gen(pattern_0, depth=depth)

    # inner
    if inner:
        return f'{a}({s[len(a):len(s) - len(b)]}){b}'
    else:
        return s


def grp_chinese_char():
    return r'[\u4e00-\u9fa5，。；？（）【】、①-⑨]'


def grr_check(m):
    """用来检查匹配情况"""
    s0 = m.group()
    pass  # 还没想好什么样的功能是和写到re.sub里面的repl
    return s0


def printoneline(s):
    """将输出控制在单行，适应终端大小"""
    try:
        columns = os.get_terminal_size().columns - 3  # 获取终端的窗口宽度
    except OSError:  # 如果没和终端相连，会抛出异常
        # 这应该就是在PyCharm，直接来个大值吧
        columns = 500
    s = shorten(s, columns)
    print(s)


def count_word(s, *patterns):
    """ 统计一串文本中，各种规律串出现的次数

    :param s: 文本内容
    :param patterns: （正则规则）
        匹配的多个目标模式list
        按优先级一个一个往后处理，被处理掉的部分会用\x00代替
    :return: Counter.most_common() 对象
    """
    s = str(s)

    if not patterns:  # 不写参数的时候，默认统计所有单个字符
        return collections.Counter(list(s)).most_common()

    ls = []
    for t in patterns:
        ls += re.findall(t, s)
        s = re.sub(t, '\x00', s)
        # s = re.sub(r'\x00+', '\x00', s)  # 将连续的特殊删除设为1，减短字符串长度，还未试验这段代码精确度与效率
    ct = collections.Counter(ls)

    ls = ct.most_common()
    for i in range(len(ls)):
        ls[i] = (ls[i][1], repr(ls[i][0])[1:-1])
    return ls


class Base85Coder:
    """base85编码、解码器

    对明文，加密/编码/encode 后已经是乱了看不懂，但是对这个结果还要二次转义
    对乱码，解密/解码/decode 时顺序要反正来，先处理二次转义，再处理base85

    使用示例：
    key = 'xV~>Y|@muL<UK$*agCQp=t4c0R_y`Z2;q%s?o8S9(3D5W^-NA&}6v){Twj7MzGePJEfik1bBhn!d#I+HlXFOr'
    coder = Base85Coder(key)
    b = coder.encode('陈坤泽 abc')
    dprint(b)  # b<str>=d@7;B}ww?}zfGP#;1
    s = coder.decode(b)
    dprint(s)  # s<str>=陈坤泽 abc
    """
    DEFAULT_KEY = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~'
    CHARS_SET = set(DEFAULT_KEY)

    def __init__(self, key=None):
        """key，允许设置密钥，必须是"""
        # 1 分析key是否合法
        if key:
            if len(key) != 85 or set(key) != Base85Coder.CHARS_SET:
                dprint(key)  # 输入key无效
                key = None
        self.key = key

        # 2 制作转换表 trantab
        if key:
            self.encode_trantab = str.maketrans(Base85Coder.DEFAULT_KEY, key)
            self.decode_trantab = str.maketrans(key, Base85Coder.DEFAULT_KEY)
        else:
            self.encode_trantab = self.decode_trantab = None

    def encode(self, s):
        """将字符串转字节"""
        b = base64.b85encode(s.encode('utf8'))
        b = str(b)[2:-1]
        if self.encode_trantab:
            b = b.translate(self.encode_trantab)
        return b

    def decode(self, b):
        if self.decode_trantab:
            b = b.translate(self.decode_trantab)
        b = b.encode('ascii')
        s = base64.b85decode(b).decode('utf8')
        return s


def check_text_row_column(s):
    """对一段文本s，用换行符分割行，用至少4个空格或\t分割列，分析数据的行、列数
    :return:
        (n, m)，每列的列数相等，则会返回n、m>=0的tuple
        (m1, m2, ...)，如果有列数不相等，则会返回每行的列数组成的tuple
            每个元素用负值代表不匹配
    """
    # 拆开每行的列
    if not s: return (0, 0)
    lines = [re.sub(r'( {4,}|\t)+', r'\t', line.strip()).split('\t') for line in s.splitlines()]
    cols = [len(line) for line in lines]  # 计算每行的列数
    if min(cols) == max(cols):
        return len(lines), cols[0]
    else:
        return [-col for col in cols]


class ListingFormat:
    r"""列表格式化工具

    >>> li = ListingFormat('（1）')
    >>> li
    （1）
    >>> li.next()
    >>> li
    （2）

    >>> li = ListingFormat(('一、选择题', '二、填空题', '三、解答题'))
    >>> li
    一、选择题
    >>> li.next()
    >>> li
    二、填空题
    """
    formats = {'[零一二三四五六七八九十]+': (chinese2digits, digits2chinese),
               r'\d+': (int, str),
               '[A-Z]': (lambda x: ord(x) - ord('A') + 1, lambda x: chr(ord('A') + x - 1)),
               '[a-z]': (lambda x: ord(x) - ord('a') + 1, lambda x: chr(ord('a') + x - 1)),
               '[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]': (circlednumber2digits, digits2circlednumber),
               '[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]': (roman2digits, digits2roman)}

    def __init__(self, s='1'):
        """
        :param s: 列表的格式，含数值和装饰
            数值形式，目前有六种形式：一 1 A a ① Ⅰ
                起始值可以不是1，例如写'三'、'D'等
                装饰的格式，常见的有：'({})'  '（{}）'  '{}、'  '{}.' '{}. '
            list或tuple，按顺序取用，用完后不再设置前缀
        >> ListingFormat('一', '{}、')

        TODO 目前只考虑值较小的情况，如果值太大，有些情况会出bug、报错
        """
        if isinstance(s, str):
            for k, funcs in ListingFormat.formats.items():
                if re.search(k, s):
                    self.form = re.sub(k, '{}', s)
                    self.value = int(funcs[0](re.search(k, s).group()))
                    self.func = funcs[1]
                    break
            else:
                raise ValueError('列表初始化格式不对 s=' + str(s))
        elif isinstance(s, (list, tuple)):
            self.form = s
            self.value = 0
            self.func = None
        else:
            raise ValueError('列表初始化格式不对 s=' + str(s))

    def reset(self, start=1):
        """重置初始值"""
        self.value = start

    def next(self):
        self.value += 1

    def __repr__(self):
        if self.func:
            return self.form.format(self.func(self.value))
        else:
            return self.form[self.value]


class StrDiffType:
    typename = {
        0: '完全相同',
        1: '忽略case后，相同',
        2: '忽略blank后，相同',
        3: '忽略case+blank后，相同',
        4: 'dt是gt局部信息（精度ok，召回不行）',
        5: '忽略case后，dt是gt局部信息',
        6: '忽略blank后，dt是gt局部信息',
        7: '忽略case+blank后，dt是gt局部信息',
        8: 'gt是dt局部信息（召回ok，精度不行）',
        9: '忽略case后，gt是dt局部信息',
        10: '忽略blank后，gt是dt局部信息',
        11: '忽略case+blank后，gt是dt局部信息',
        16: '其他情况 （可以根据实验情况，后续继续细分类别）'
    }

    @classmethod
    def main_difftype(cls, dt, gt):
        if not dt or not gt:
            return 16
        elif dt == gt:
            return 0
        elif dt in gt:
            return 4
        elif gt in dt:
            return 8
        else:
            return 16

    @classmethod
    def difftype(cls, dt, gt):
        """ 判断两段字符串dt,gt的差异所属类别
        """
        if not dt or not gt: return 16

        t = cls.main_difftype(dt, gt)
        if t < 16:
            return t

        t = cls.main_difftype(dt.lower(), gt.lower()) + 1
        if t < 16:
            return t

        dt2, gt2 = re.sub(r'\s+', '', dt), re.sub(r'\s+', '', gt)
        t = cls.main_difftype(dt2, gt2) + 2
        if t < 16:
            return t

        dt3, gt3 = dt2.lower(), gt2.lower()
        t = cls.main_difftype(dt3, gt3) + 3
        if t < 16:
            return t
        else:
            return 16


class BookContents:
    """ 书本目录类 """

    def __init__(self):
        self.contents = []  # 目录条目按顺序保存在list中

    def add(self, level, title, page=None):
        """
        Args:
            level:
            title:
            page: 不一定要放整数的页数，也可以放其他一些比例之类的数值

        Returns:

        """
        self.contents.append([level, title, page])

    def format_numbers(self, number='normal', *, indent='', start_level=1, jump=False):
        """ 每级目录的编号

        :param number: 编号格式，目前有默认方式，以后有需要可以扩展其他模式
        :param start_level: 开始展示的层级（高层级也会展示，只是不带编号和缩进）
            可以设为负数，表示自动推算，比如-1
        :param jump: 支持跳级，比如2级"3"，跳到4级本来是"3.0.1"，但开启该参数则会优化为"3.1"
        :return: list，跟contents等长，表示每个标题的编号，可能为空''
        """
        # 1
        if start_level == -1:
            # 自动推算合适的开始编号
            # -1模式，表示第一个不只一项的level
            levels = [x[0] for x in self.contents]
            levels_cnt = collections.Counter(levels)
            for i in range(min(levels), max(levels) + 1):
                if levels_cnt[i] > 1:
                    start_level = i
                    break

        # 2
        ls = []
        ct = collections.defaultdict(int)
        for x in self.contents:
            # print(x)
            level = x[0]
            sign = indent * (level - start_level)

            # 处理计数器
            ct[level] += 1
            for k, v in ct.items():
                if k > level:
                    ct[k] = 0

            # 当前编号
            if number == 'normal':
                numbers = [ct[i] for i in range(start_level, level + 1)]
                if jump:  # 过滤0
                    numbers = [x for x in numbers if x]
                sign += '.'.join(map(str, numbers))
            else:
                pass

            ls.append(sign)

        return ls

    def format_str(self, indent='\t', *, number='normal', page=False, start_level=1, jump=False):
        """ 转文本展示

        :param indent: 每级展示的缩进量
        :param page: 是否展示页码
        """
        numbers = self.format_numbers(number, indent=indent, start_level=start_level, jump=jump)

        # 2
        ls = []
        for num, x in zip(numbers, self.contents):
            level, title, page_ = x

            # 标题
            if level < start_level:
                sign = title
            else:
                sign = ' '.join([num, x[1]])

            # 加后缀
            if page:
                sign += f'，{page_}'

            ls.append(sign)

        return '\n'.join(ls)


def continuous_zero(s):
    """ 返回一个字符串中连续0的位置

    :param s: 一个字符串

    做html转latex表格中，合并单元格的处理要用到这个函数计算cline

    >>> continuous_zero('0100')  # 从0开始编号，左闭右开区间
    [(0, 1), (2, 4)]
    """
    return [m.span() for m in re.finditer(r'0+', s)]
