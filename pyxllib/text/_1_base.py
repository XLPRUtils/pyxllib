#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2018/12/27


"""
文本处理、常用正则匹配模式

下面大量的函数前缀含义：
grp，generate regular pattern，生成正则模式字符串
grr，generate regular replace，生成正则替换目标格式
"""

import base64
import bisect
import collections

from pyxllib.debug import *

import ahocorasick

____section_0_import = """
try ... except不影响效率的
主要是导入特殊包，好像是比较耗费时间，这里要占用掉0.1秒多时间
"""

# 这个需要C++14编译器 https://download.microsoft.com/download/5/f/7/5f7acaeb-8363-451f-9425-68a90f98b238/visualcppbuildtools_full.exe
# 在需要的时候安装，防止只是想用pyxllib很简单的功能，但是在pip install阶段处理过于麻烦
try:
    # MatchSimString计算编辑距离需要
    import Levenshtein
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'python-Levenshtein'])
    import Levenshtein

# import textract     # ensure_content读取word文档需要

try:  # 拼写检查库，即词汇库
    from spellchecker import SpellChecker
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'pyspellchecker'])
    from spellchecker import SpellChecker

____section_1_text = """
一些文本处理函数和类
"""


class ContentLine(object):
    """用行数的特性分析一段文本"""

    def __init__(self, content):
        """用一段文本初始化"""
        self.content = ensure_content(content)  # 原始文本
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
            dprint(typename(ob))  # 类型错误
            raise ValueError

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
        common_used_numerals = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                                '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
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


def gettag_name(tagstr):
    """
    >>> gettag_name('%<topic type=danxuan description=单选题>')
    'topic'
    >>> gettag_name('</topic>')
    'topic'
    """
    m = re.search(r'</?([a-zA-Z_]+)', tagstr)
    if m:
        return m.group(1)
    else:
        return None


def settag_name(tagstr, *, new_name=None, switch=None):
    """设置标签名称，或者将标签类型设为close类型

    >>> settag_name('%<topic type=danxuan description=单选题>', new_name='mdzz')
    '%<mdzz type=danxuan description=单选题>'
    >>> settag_name('<topic type=danxuan description=单选题>', switch=False)
    '</topic>'
    """
    if new_name:  # 是否设置新名称
        tagstr = re.sub(r'(</?)([a-zA-Z_]+)', lambda m: m.group(1) + new_name, tagstr)

    if switch is not None:  # 是否设置标签开关
        if switch:  # 将标签改为开
            tagstr = tagstr.replace('</', '<')
        else:  # 将标签改为关
            name = gettag_name(tagstr)
            res = f'</{name}>'  # 会删除所有attr属性
            tagstr = '%' + res if '%<' in tagstr else res

    return tagstr


def gettag_attr(tagstr, attrname):
    r"""tagstr是一个标签字符串，attrname是要索引的名字
    返回属性值，如果不存在该属性则返回None

    >>> gettag_attr('%<topic type=danxuan description=单选题> 123\n<a b=c></a>', 'type')
    'danxuan'
    >>> gettag_attr('%<topic type="dan xu an" description=单选题>', 'type')
    'dan xu an'
    >>> gettag_attr("%<topic type='dan xu an' description=单选题>", 'type')
    'dan xu an'
    >>> gettag_attr('%<topic type=dan xu an description=单选题>', 'description')
    '单选题'
    >>> gettag_attr('%<topic type=dan xu an description=单选题>', 'type')
    'dan'
    >>> gettag_attr('%<topic type=danxuan description=单选题 >', 'description')
    '单选题'
    >>> gettag_attr('%<topic type=danxuan description=单选题 >', 'description123') is None
    True
    """
    import bs4
    soup = BeautifulSoup(tagstr, 'lxml')
    try:
        for tag in soup.p.contents:
            if isinstance(tag, bs4.Tag):
                return tag.get(attrname, None)
    except AttributeError:
        dprint(tagstr)
    return None


def settag_attr(tagstr, attrname, target_value):
    r"""tagstr是一个标签字符串，attrname是要索引的名字
    重设该属性的值，设置成功则返回新的tagstr；否则返回原始值

    close类型不能用这个命令，用了的话不进行任何处理，直接返回

    >>> settag_attr('%<topic type=danxuan> 123\n<a></a>', 'type', 'tiankong')
    '%<topic type="tiankong"> 123\n<a></a>'
    >>> settag_attr('%<topic>', 'type', 'tiankong')
    '%<topic type="tiankong">'
    >>> settag_attr('</topic>', 'type', 'tiankong')
    '</topic>'
    >>> settag_attr('<seq value="1">', 'value', '练习1.2')
    '<seq value="练习1.2">'
    >>> settag_attr('<seq type=123 value=1>', 'type', '')  # 删除attr操作
    '<seq value=1>'
    >>> settag_attr('<seq type=123 value=1>', 'value', '')  # 删除attr操作
    '<seq type=123>'
    >>> settag_attr('<seq type=123 value=1>', 'haha', '')  # 删除attr操作
    '<seq type=123 value=1>'
    """
    # 如果是close类型是不处理的
    if tagstr.startswith('</'): return tagstr

    # 预处理targetValue的值，删除空白
    target_value = re.sub(r'\s', '', target_value)
    r = re.compile(r'(<|\s)(' + attrname + r'=)(.+?)(\s+\w+=|\s*>)')
    gs = r.search(tagstr)
    if target_value:
        if not gs:  # 如果未找到则添加attr与value
            n = tagstr.find('>')
            return tagstr[:n] + ' ' + attrname + '="' + target_value + '"' + tagstr[n:]
        else:  # 如果找到则更改value
            # TODO: 目前的替换值是直接放到正则式里了，这样会有很大的风险，后续看看能不能优化这个处理算法
            return r.sub(r'\1\g<2>"' + target_value + r'"\4', tagstr)
    else:
        if gs:
            return r.sub(r'\4', tagstr)
        else:
            return tagstr


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


def brieftexstr(s):
    """对比两段tex文本
    """
    # 1 删除百分注
    s = re.sub(r'%' + grp_bracket(2, '<', '>'), r'', s)
    # 2 删除所有空白字符
    # debuglib.dprint(debuglib.typename(s))
    s = re.sub(r'\s+', '', s)
    # 3 转小写字符
    s = s.casefold()
    return s


class MatchSimString:
    """匹配近似字符串

    mss = MatchSimString()

    # 1 添加候选对象
    mss.append_candidate('福州+厦门2018初数暑假讲义-请录入-快乐学习\初一福厦培优-测试用')
    mss.append_candidate('2018_快乐数学_六年级_秋季_第01讲_圆柱与圆锥_教案（教师版）')
    mss.append_candidate('删除所有标签中间多余的空白')

    # 2 需要匹配的对象1
    s = '奕本初一福周厦门培油'

    idx, sim = mss.match(s)
    print('匹配目标：', mss[idx])  # 匹配目标： 福州+厦门2018初数暑假讲义-请录入-快乐学习\初一福厦培优-测试用
    print('相似度：', sim)         # 相似度： 0.22

    # 3 需要匹配的对象2
    s = '圆柱与【圆锥】_教案空白版'

    idx, sim = mss.match(s)
    print('匹配目标：', mss[idx])  # 2018_快乐数学_六年级_秋季_第01讲_圆柱与圆锥_教案（教师版）
    print('相似度：', sim)         # 相似度： 0.375

    如果append_candidate有传递2个扩展信息参数，可以索引获取：
    mss.ext_value[idx]
    """

    def __init__(self, method=briefstr):
        self.preproc = method
        self.origin_str = list()  # 原始字符串内容
        self.key_str = list()  # 对原始字符串进行处理后的字符
        self.ext_value = list()  # 扩展存储一些信息

    def __getitem__(self, item):
        return self.origin_str[item]

    def __delitem__(self, item):
        del self.origin_str[item]
        del self.key_str[item]
        del self.ext_value[item]

    def __len__(self):
        return len(self.key_str)

    def append_candidate(self, k, v=None):
        self.origin_str.append(k)
        if callable(self.preproc):
            k = self.preproc(k)
        self.key_str.append(k)
        self.ext_value.append(v)

    def match(self, s):
        """跟候选字符串进行匹配，返回最佳匹配结果
        """
        idx, sim = -1, 0
        for i in range(len(self)):
            k, v = self.key_str[i], self.ext_value[i]
            sim_ = Levenshtein.ratio(k, s)
            if sim_ > sim:
                sim = sim_
                idx = i
            i += 1
        return idx, sim

    def match_test(self, s, count=-1, showstr=lambda x: x[:50]):
        """输入一个字符串s，和候选项做近似匹配

        :param s: 需要进行匹配的字符串s
        :param count: 只输出部分匹配结果
            -1：输出所有匹配结果
            0 < count < 1：例如0.4，则只输出匹配度最高的40%结果
            整数：输出匹配度最高的count个结果
        :param showstr: 字符串显示效果
        """
        # 1 计算编辑距离，存储结果到res
        res = []
        n = len(self)
        for i in range(n):
            k, v = self.key_str[i], self.ext_value[i]
            sim = Levenshtein.ratio(k, s)
            res.append([i, v, sim, showstr(k)])  # 输出的时候从0开始编号
            i += 1

        # 2 排序、节选结果
        res = sorted(res, key=lambda x: -x[2])
        if 0 < count < 1:
            n = max(1, int(n * count))
        elif isinstance(count, int) and count > 0:
            n = min(count, n)
        res = res[:n]

        # 3 输出
        df = pd.DataFrame.from_records(res, columns=('序号', '标签', '编辑距离', '内容'))
        s = dataframe_str(df)
        s = s.replace('\u2022', '')  # texstudio无法显示会报错的字符
        print(s)


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


def mydictstr(d, key_value_delimit='=', item_delimit=' '):
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


def printoneline(s):
    """将输出控制在单行，适应终端大小"""
    try:
        columns = os.get_terminal_size().columns - 3  # 获取终端的窗口宽度
    except OSError:  # 如果没和终端相连，会抛出异常
        # 这应该就是在PyCharm，直接来个大值吧
        columns = 500
    s = shorten(s, columns)
    print(s)


def del_tail_newline(s):
    """删除末尾的换行"""
    if len(s) > 1 and s[-1] == '\n':
        s = s[:-1]
    return s


____section_2_regular = """
跟正则相关的一些文本处理函数和类
"""


def grp_bracket(depth=0, left='{', right=None):
    r"""括号匹配，默认花括号匹配，也可以改为圆括号、方括号匹配。

    效果类似于“{.*?}”，
    但是左右花括号是确保匹配的，有可选参数可以提升支持的嵌套层级，
    数字越大匹配嵌套能力越强，但是速度性能会一定程度降低。
    例如“grp_bracket(5)”。

    :param depth: 括号递归深度
    :param left: 左边字符：(、[、{
    :param right: 右边字符
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
    if b is None:
        if a == '(':
            b = ')'
        elif a == '[':
            b = ']'
        elif a == '{':
            b = '}'
        else:
            raise NotImplementedError
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
    return s


# 定义常用的几种格式，并且只匹配抓取花括号里面的值，不要花括号本身
SQUARE3 = r'\\[(' + grp_bracket(3, '[')[3:-3] + r')\\]'
BRACE1 = '{(' + grp_bracket(1)[1:-1] + ')}'
BRACE2 = '{(' + grp_bracket(2)[1:-1] + ')}'
BRACE3 = '{(' + grp_bracket(3)[1:-1] + ')}'
BRACE4 = '{(' + grp_bracket(4)[1:-1] + ')}'
BRACE5 = '{(' + grp_bracket(5)[1:-1] + ')}'
"""使用示例
>> m = re.search(r'\\multicolumn' + BRACE3*3, r'\multicolumn{2}{|c|}{$2^{12}$个数}')
>> m.groups()
('2', '|c|', '$2^{12}$个数')
"""


def grp_figure(cnt_groups=0, parpic=False):
    """生成跟图片匹配相关的表达式

    D:\2017LaTeX\D招培试卷\高中地理，用过  \captionfig{3-3.eps}{图~3}
    奕本从2018秋季教材开始使用多种图片格式

    191224周二18:20 更新：匹配到的图片名不带花括号
    """
    if cnt_groups == 0:  # 不分组
        s = r'\\(?:includegraphics|figt|figc|figr|fig).*?' + grp_bracket(3)  # 注意第1组fig要放最后面
    elif cnt_groups == 1:  # 只分1组，那么只对图片括号内的内容分组
        s = r'\\(?:includegraphics|figt|figc|figr|fig).*?' + BRACE3
    elif cnt_groups == 2:  # 只分2组，那么只对插图命令和图片分组
        s = r'\\(includegraphics|figt|figc|figr|fig).*?' + BRACE3
    elif cnt_groups == 3:
        s = r'\\(includegraphics|figt|figc|figr|fig)(.*?)' + BRACE3
    else:
        s = None

    if s and parpic:
        s = r'{?\\parpic(?:\[.\])?{' + s + r'}*'

    return s


def grp_topic(*, type_value=None):
    """定位topic

    :param type_value: 设置题目类型（TODO: 功能尚未开发）
    """
    s = r'%<topic.*?%</topic>'  # 注意外部使用的re要开flags=re.DOTALL
    return s


def grp_chinese_char():
    return r'[\u4e00-\u9fa5，。；？（）【】、①-⑨]'


def grr_check(m):
    """用来检查匹配情况"""
    s0 = m.group()
    pass  # 还没想好什么样的功能是和写到re.sub里面的repl
    return s0


def regularcheck(pattern, string, flags=0):
    arr = []
    cl = ContentLine(string)
    for i, m in enumerate(re.finditer(pattern, string, flags)):
        ss = map(lambda x: textwrap.shorten(x, 200), m.groups())
        arr.append([i + 1, cl.in_line(m.start(0)), *ss])
    tablehead = ['行号'] + list(map(lambda x: f'第{x}组', range(len_in_dim2(arr) - 2)))
    df = pd.DataFrame.from_records(arr, columns=tablehead)
    res = f'正则模式：{pattern}，匹配结果：\n' + dataframe_str(df)
    return res


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


____section_3_ensure_content = """
从任意类型文件读取文本数据的功能
"""


def readtext(filename, encoding=None):
    """读取普通的文本文件
    会根据tex、py文件情况指定默认编码
    """
    try:
        with open(filename, 'rb') as f:  # 以二进制读取文件，注意二进制没有\r\n参数
            bstr = f.read()
    except FileNotFoundError:
        return None

    if not encoding:
        encoding = get_encoding(bstr)
    s = bstr.decode(encoding=encoding, errors='ignore')
    if '\r' in s:  # 注意这个问题跟gb2312和gbk是独立的，用gbk编码也要做这个处理
        s = s.replace('\r\n', '\n')  # 如果用\r\n作为换行符会有一些意外不好处理
    return s


def ensure_content(ob=None, encoding=None):
    """
    :param ob:
        未输入：从控制台获取文本
        存在的文件名：读取文件的内容返回
            tex、py、
            docx、doc
            pdf
        有read可调用成员方法：返回f.read()
        其他字符串：返回原值
    :param encoding: 强制指定编码
    """
    # TODO: 如果输入的是一个文件指针，也能调用f.read()返回所有内容
    # TODO: 增加鲁棒性判断，如果输入的不是字符串类型也要有出错判断
    if ob is None:
        return sys.stdin.read()  # 注意输入是按 Ctrl + D 结束
    elif File(ob):  # 如果存在这样的文件，那就读取文件内容（bug点：如果输入是目录名会PermissionError）
        if ob.endswith('.docx'):  # 这里还要再扩展pdf、doc文件的读取
            try:
                import textract
            except ModuleNotFoundError:
                dprint()  # 缺少textract模块，安装详见： https://blog.csdn.net/code4101/article/details/79328636
                raise ModuleNotFoundError
            text = textract.process(ob)
            return text.decode('utf8', errors='ignore')
        elif ob.endswith('.doc'):
            raise NotImplementedError
        elif ob.endswith('.pdf'):
            raise NotImplementedError
        else:  # 按照普通的文本文件读取内容
            return readtext(ob, encoding)
    else:  # 判断不了的情况，也认为是字符串
        return ob


def file_lastlines(fn, n):
    """获得一个文件最后的几行内容
    参考资料: https://stackoverflow.com/questions/136168/get-last-n-lines-of-a-file-with-python-similar-to-tail

    >> s = FileLastLine('book.log', 1)
    'Output written on book.dvi (2 pages, 7812 bytes).'
    """
    f = ensure_content(fn)
    assert n >= 0
    pos, lines = n + 1, []
    while len(lines) <= n:
        try:
            f.seek(-pos, 2)
        except IOError:
            f.seek(0)
            break
        finally:
            lines = list(f)
        pos *= 2
    f.close()
    return ''.join(lines[-n:])


____section_4_spell_check = """
拼写检查
190923周一21:54，源自 完形填空ocr 识别项目
"""


class MySpellChecker(SpellChecker):
    def __init__(self, language="en", local_dictionary=None, distance=2, tokenizer=None, case_sensitive=False,
                 df=None):
        from collections import defaultdict, Counter

        # 1 原初始化功能
        super(MySpellChecker, self).__init__(language=language, local_dictionary=local_dictionary,
                                             distance=distance, tokenizer=tokenizer,
                                             case_sensitive=case_sensitive)

        # 2 自己要增加一个分析用的字典
        self.checkdict = defaultdict(Counter)
        for k, v in self.word_frequency._dictionary.items():
            self.checkdict[k][k] = v

        # 3 如果输入了一个df对象要进行更新
        if df: self.update_by_dataframe(df)

    def update_by_dataframe(self, df, weight_times=1):
        """
        :param df: 这里的df有要求，是DataFrame对象，并且含有这些属性列：old、new、count
        :param weight_times: 对要加的count乘以一个倍率
        :return:
        """
        # 1 是否要处理大小写
        #   如果不区分大小写，需要对df先做预处理，全部转小写
        #   而大小写不敏感的时候，self.word_frequency._dictionary在init时已经转小写，不用操心
        if not self._case_sensitive:
            df.loc[:, 'old'] = df.loc[:, 'old'].str.lower()
            df.loc[:, 'new'] = df.loc[:, 'new'].str.lower()

        # 2 df对self.word_frequency._dictionary、self.check的影响
        d = self.word_frequency._dictionary
        for index, row in df.iterrows():
            old, new, count = row['old'].decode(), row['new'].decode(), row['count'] * weight_times
            d[old] += count if old == new else -count
            # if row['id']==300: dprint(old, new, count)
            self.checkdict[old][new] += count

        # 3 去除d中负值的key
        self.word_frequency.remove_words([k for k in d.keys() if d[k] <= 0])

    def _ensure_term(self, term):
        if term not in self.checkdict:
            d = {k: self.word_frequency._dictionary[k] for k in self.candidates(term)}
            self.checkdict[term] = d

    def correction(self, term):
        # 1 本来就是正确的
        w = term if self._case_sensitive else term.lower()
        if w in self.word_frequency._dictionary: return term

        # 2 如果是错的，且是没有记录的错误情况，则做一次候选项运算
        self._ensure_term(w)

        # 3 返回权重最大的结果
        res = max(self.checkdict[w], key=self.checkdict[w].get)
        val = self.checkdict[w].get(res)
        if val <= 0: res = '^' + res  # 是一个错误单词，但是没有推荐修改结果，就打一个^标记
        return res

    def correction_detail(self, term):
        """更加详细，给出所有候选项的纠正

        >> a.correction_detail('d')
        [('d', 9131), ('do', 1), ('old', 1)]
        """
        w = term if self._case_sensitive else term.lower()
        self._ensure_term(w)
        ls = [(k, v) for k, v in self.checkdict[w].items()]
        ls = sorted(ls, key=lambda x: x[1], reverse=True)
        return ls


def demo_myspellchecker():
    # 类的初始化大概要0.4秒
    a = MySpellChecker()

    # sql的加载更新大概要1秒
    # hsql = HistudySQL('ckz', 'tr_develop')
    # df = hsql.query('SELECT * FROM spell_check')
    # a.update_by_dataframe(df)

    # dprint(a.correction_detail('d'))
    # dprint(a.correction_detail('wrod'))  # wrod有很多种可能性，但word权重是最大的
    # dprint(a.correction_detail('ckzckzckzckzckzckz'))  # wrod有很多种可能性，但word权重是最大的
    # dprint(a.correction('ckzckzckzckzckzckz'))  # wrod有很多种可能性，但word权重是最大的
    dprint(a.correction_detail('ike'))
    dprint(a.correction_detail('dean'))
    dprint(a.correction_detail('stud'))
    dprint(a.correction_detail('U'))


____section_temp = """
临时添加的新功能
"""


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


def demo_spellchecker():
    """演示如何使用spellchecker库
    官方介绍文档 pyspellchecker · PyPI: https://pypi.org/project/pyspellchecker/
    190909周一15:58，from 陈坤泽
    """
    # 0 安装库和导入库
    #   spellchecker模块主要有两个类，SpellChecker和WordFrequency
    #       WordFrequency是一个词频类
    #       一般导入SpellChecker就行了：from spellchecker import SpellChecker
    try:  # 拼写检查库，即词汇库
        from spellchecker import SpellChecker
    except ModuleNotFoundError:
        subprocess.run(['pip3', 'install', 'pyspellchecker'])
        from spellchecker import SpellChecker

    # 1 创建对象
    # 可以设置语言、大小写敏感、拼写检查的最大距离
    #   默认'en'英语，大小写不敏感
    spell = SpellChecker()
    # 如果是英语，SpellChecker会自动加载语言包site-packages\spellchecker\resources\en.json.gz，大概12万个词汇，包括词频权重
    d = spell.word_frequency  # 这里的d是WordFrequency对象，其底层用了Counter类进行数据存储
    dprint(d.unique_words, d.total_words)  # 词汇数，权重总和

    # 2 修改词频表 spell.word_frequency
    dprint(d['ckz'])  # 不存在的词汇直接输出0
    d.add('ckz')  # 可以添加ckz词汇的一次词频
    d.load_words(['ckz', 'ckz', 'lyb'])  # 可以批量添加词汇
    dprint(d['ckz'], d['lyb'])  # d['ckz']=3  d['lyb']=1
    d.load_words(['ckz'] * 100 + ['lyb'] * 500)  # 可以用这种技巧进行大权重的添加
    dprint(d['ckz'], d['lyb'])  # d['ckz']=103  d['lyb']=501

    # 同理，去除也有remove和remove_words两种方法
    d.remove('ckz')
    # d.remove_words(['ckz', 'lyb'])  # 不过注意不能删除已经不存在的key（'ckz'），否则会报KeyError
    dprint(d['ckz'], d['lyb'])  # d['ckz']=0  d['lyb']=501
    # remove是完全去除单词，如果只是要减权重可以访问底层的_dictionary对象操作
    d._dictionary['lyb'] -= 100  # 当然不太建议直接访问下划线开头的成员变量~~
    dprint(d['lyb'])  # ['lyb']=401

    # 还可以按阈值删除词频不超过设置阈值的词汇
    d.remove_by_threshold(5)

    # 3 spell的基本功能
    # （1）用unknown可以找到可能拼写错误的单词，再用correction可以获得最佳修改意见
    misspelled = spell.unknown(['something', 'is', 'hapenning', 'here'])
    dprint(misspelled)  # misspelled<set>={'hapenning'}

    for word in misspelled:
        # Get the one `most likely` answer
        dprint(spell.correction(word))  # <str>='happening'
        # Get a list of `likely` options
        dprint(spell.candidates(word))  # <set>={'henning', 'happening', 'penning'}

    # 注意默认的spell不区分大小写，如果词库存储了100次'ckz'
    #   此时判断任意大小写形式组合的'CKZ'都是返回原值
    #   例如 spell.correction('ckZ') => 'ckZ'

    # （2）可以通过修改spell.word_frequency影响correction的计算结果
    dprint(d['henning'], d['happening'], d['penning'])
    # d['henning']<int>=53    d['happening']<int>=4538    d['penning']<int>=23
    d._dictionary['henning'] += 10000
    dprint(spell.correction('hapenning'))  # <str>='henning'

    # （3）词汇在整个字典里占的权重
    dprint(spell.word_probability('henning'))  # <float>=0.0001040741914298211


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


def latexstrip(s):
    """latex版的strip"""
    return s.strip('\t\n ~')


____section_ac = """
AC自动机
"""


def make_automaton(words):
    """ 根据输入的一串words模式，生成一个AC自动机 """
    a = ahocorasick.Automaton()
    for index, word in enumerate(words):
        a.add_word(word, (index, word))
    a.make_automaton()
    return a


def count_words(content, word, scope=2, exclude=None):
    # 1 统计所有词汇出现次数
    c = Counter()
    c += Counter(re.findall(f'.{{,{scope}}}{word}.{{,{scope}}}', content))
    # 2 排除掉不处理的词 （注意因为这里每句话都已经是被筛选过的，所以处理比较简单，并不需要复杂到用区间集处理）
    if exclude:
        new_c = Counter()
        a = make_automaton(exclude)  # 创建AC自动机
        for k in c.keys():
            if not next(a.iter(k), None):
                # 如果k没匹配到需要排除的词汇，则拷贝到新的计数器
                new_c[k] = c[k]
        c = new_c
    return c
