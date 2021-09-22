#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/02 20:16

"""
xml等网页结构方面的处理
"""
import collections
from collections import Counter, defaultdict
import re
import textwrap
import os

import requests
import pandas as pd
import bs4
from bs4 import BeautifulSoup
from humanfriendly import format_size

from pyxllib.debug.pupil import dprint
from pyxllib.prog.newbie import round_int
from pyxllib.prog.pupil import EnchantBase
from pyxllib.text.newbie import xldictstr
from pyxllib.text.pupil import listalign, int2myalphaenum, shorten, ensure_gbk, RunOnlyOnce, BookContents, strwidth, \
    grp_chinese_char
from pyxllib.file.specialist import File, Dir, get_etag

____section_1_dfs_base = """
一个通用的递归功能
"""


def dfs_base(node, *,
             child_generator=None, select_depth=None, linenum=False,
             mystr=None, msghead=True, lsstr=None, show_node_type=False, prefix='    '):
    """ 输入一个节点node，以及该节点当前depth

    :param prefix: 缩进格式，默认用4个空格
    :param node: 节点
    :param child_generator: 子节点生成函数
        函数支持输入一个节点参数
        返回一个子节点列表
    :param select_depth: 要显示的深度
        单个数字：获得指定层
        Sequences： 两个整数，取出这个闭区间内的层级内容
    :param mystr: 自定义单个节点字符串方式
        标准是输入2个参数 mystr(node, depth)，返回字符串化的结果，记得前缀缩进也要自己控制的！
        也可以只输入一个参数 mystr(node)：
            这种情况会自动按照每层4个空格进行缩进
    :param lsstr: 自定义整个列表的字符串化方法，在mystr的基础上调控更加灵活，但要写的代码也更多
    :param linenum：节点从1开始编号
        行号后面，默认会跟一个类似Excel列名的字母，表示层级深度
    :param msghead: 第1行输出一些统计信息
    :param show_node_type:
    :return 返回一个遍历清单ls
        ls的每个元素是一个列表
            第1个值是depth
            第2个值是节点ref

    Requires
        textwrap：用到shorten
        align.listalign：生成列编号时对齐
    """

    # 1 子节点生成器，与配置
    def bs4_child_generator(node):
        try:
            return node.children
        except AttributeError:
            return []

    # 配置子节点生成器
    if not child_generator:
        child_generator = bs4_child_generator

    # 2 dfs实际实现代码，获得节点清单
    def inner(node, depth=0):
        """dfs实际实现代码
        TODO：把depth过滤写进inner不生成？！ 不过目前还是按照生成整棵树处理，能统计到一些信息。
        """
        ls = [[node, depth]]
        for t in child_generator(node):
            ls += inner(t, depth + 1)
        return ls

    ls = inner(node)
    total_node = len(ls)
    total_depth = max(map(lambda x: x[1], ls))
    head = f'总节点数：1~{total_node}，总深度：0~{total_depth}'

    # 4 过滤与重新整理ls（select_depth）
    logo = True
    cnt = 0
    tree_num = 0
    if isinstance(select_depth, int):

        for i in range(total_node):
            if ls[i][1] == select_depth:
                ls[i][1] = 0
                cnt += 1
                logo = True
            elif ls[i][1] < select_depth and logo:  # 遇到第1个父节点添加一个空行
                ls[i] = ''
                tree_num += 1
                logo = False
            else:  # 删除该节点，不做任何显示
                ls[i] = None
        head += f'；挑选出的节点数：{cnt}，所选深度：{select_depth}，树数量：{tree_num}'

    elif hasattr(select_depth, '__getitem__'):
        for i in range(total_node):
            if select_depth[0] <= ls[i][1] <= select_depth[1]:
                ls[i][1] -= select_depth[0]
                cnt += 1
                logo = True
            elif ls[i][1] < select_depth[0] and logo:  # 遇到第1个父节点添加一个空行
                ls[i] = ''
                tree_num += 1
                logo = False
            else:  # 删除该节点，不做任何显示
                ls[i] = None
        head += f'；挑选出的节点数：{cnt}，所选深度：{select_depth[0]}~{select_depth[1]}，树数量：{tree_num}'
    """注意此时ls[i]的状态，有3种类型
        (node, depth)：tuple类型，第0个元素是node对象，第1个元素是该元素所处层级
        None：已删除元素，但为了后续编号方便，没有真正的移出，而是用None作为标记
        ''：已删除元素，但这里涉及父节点的删除，建议此处留一个空行
    """

    # 5 格式处理
    def default_mystr(node, depth):
        s1 = prefix * depth
        s2 = typename(node) + '，' if show_node_type else ''
        s3 = textwrap.shorten(str(node), 200)
        return s1 + s2 + s3

    def default_lsstr(ls):
        nonlocal mystr
        if not mystr:
            mystr = default_mystr
        else:
            try:  # 测试两个参数情况下是否可以正常运行
                mystr('', 0)
            except TypeError:
                # 如果不能正常运行，则进行封装从而支持2个参数
                func = mystr

                def str_plus(node, depth):  # 注意这里函数名要换一个新的func
                    return prefix * depth + func(node)

                mystr = str_plus

        line_num = listalign(range(1, total_node + 1))
        res = []
        for i in range(total_node):
            if ls[i] is not None:
                if isinstance(ls[i], str):  # 已经指定该行要显示什么
                    res.append(ls[i])
                else:
                    if linenum:  # 增加了一个能显示层级的int2excel_col_name
                        res.append(line_num[i] + int2myalphaenum(ls[i][1]) + ' ' + mystr(ls[i][0], ls[i][1]))
                    else:
                        res.append(mystr(ls[i][0], ls[i][1]))

        s = '\n'.join(res)
        return s

    if not lsstr:
        lsstr = default_lsstr

    s = lsstr(ls)

    # 是否要添加信息头
    if msghead:
        s = head + '\n' + s

    return s


def treetable(childreds, parents, arg3=None, nodename_colname=None):
    """输入childres子结点id列表，和parents父结点id列表
    两个列表长度必须相等
    文档：http://note.youdao.com/noteshare?id=126200f45d301fcb4364d06a0cae8376

    有两种调用形式
    >> treetable(childreds, parents)  --> DataFrame  （新建df）
    >> treetable(df, child_colname, parent_colname)  --> DataFrame （修改后的df）

    返回一个二维列表
        新的childreds （末尾可能回加虚结点）
        新的parents
        函数会计算每一行childred对应的树排序后的排序编号order
        以及每个节点深度depth

    >> ls1 = [6, 2, 4, 5, 3], ls2 = [7, 1, 2, 2, 1], treetable(ls1, ls2)
          child_id   parent_id   depth     tree_order    tree_struct
        5        7     root        1           1         = = 7
        0        6        7        2           2         = = = = 6
        6        1     root        1           3         = = 1
        1        2        1        2           4         = = = = 2
        2        4        2        3           5         = = = = = = 4
        3        5        2        3           6         = = = = = = 5
        4        3        1        2           7         = = = = 3
    """
    # 0 参数预处理
    if isinstance(childreds, pd.DataFrame):
        df = childreds
        child_colname = parents
        parent_colname = arg3
        if not arg3: raise TypeError
        childreds = df[child_colname].tolist()
        parents = df[parent_colname].tolist()
    else:
        df = None

    # 1 建立root根节点，确保除了root其他结点都存在记录
    lefts = set(parents) - set(childreds)  # parents列中没有在childreds出现的结点
    cs, ps = list(childreds), list(parents)

    if len(lefts) == 0:
        # b_left为空一定有环，b_left不为空也不一定是正常的树
        raise ValueError('有环，不是树结构')
    elif len(lefts) == 1:  # 只有一个未出现的结点，那么它既是根节点
        root = list(lefts)[0]
    else:  # 多个父结点没有记录，则对这些父结点统一加一个root父结点
        root = 'root'
        allnode = set(parents) | set(childreds)  # 所有结点集合
        while root in allnode: root += '-'  # 一直在末尾加'-'，直到这个结点是输入里未出现的
        # 添加结点
        lefts = list(lefts)
        lefts.sort(key=lambda x: parents.index(x))
        for t in lefts:
            cs.append(t)
            ps.append(root)

    n = len(cs)
    depth, tree_order, len_childs = [-1] * n, [-1] * n, [0] * n

    # 2 构造父结点-孩子结点的字典dd
    dd = defaultdict(list)
    for i in range(n): dd[ps[i]] += [i]

    # 3 dfs
    cnt = 1

    def dfs(node, d):
        """找node的所有子结点"""
        nonlocal cnt
        for i in dd.get(node, []):
            tree_order[i], depth[i], len_childs[i] = cnt, d, len(dd[cs[i]])
            cnt += 1
            dfs(cs[i], d + 1)

    dfs(root, 1)

    # 4 输出格式
    tree_struct = list(map(lambda i: f"{'_ _ ' * depth[i]}{cs[i]}" + (f'[{len_childs[i]}]' if len_childs[i] else ''),
                           range(n)))

    if df is None:
        ls = list(zip(cs, ps, depth, tree_order, len_childs, tree_struct))
        df = pd.DataFrame.from_records(ls, columns=('child_id', 'parent_id',
                                                    'depth', 'tree_order', 'len_childs', 'tree_struct'))
    else:
        k = len(df)
        df = df.append(pd.DataFrame({child_colname: cs[k:], parent_colname: ps[k:]}), sort=False, ignore_index=True)
        if nodename_colname:
            tree_struct = list(
                map(lambda i: f"{'_ _ ' * depth[i]}{cs[i]} {df.iloc[i][nodename_colname]}"
                              + (f'[{len_childs[i]}]' if len_childs[i] else ''), range(n)))
        df['depth'], df['tree_order'], df['len_childs'], df['tree_struct'] = depth, tree_order, len_childs, tree_struct
    df.sort_values('tree_order', inplace=True)  # 注意有时候可能不能排序，要维持输入时候的顺序
    return df


def treetable_flatten(df, *, reverse=False, childid_colname='id', parentid_colname='parent_id', format_colname=None):
    """获得知识树横向展开表：列为depth-3, depth-2, depth-1，表示倒数第3级、倒数第2级、倒数第1级
    :param df: DataFrame数据
    :param reverse:
        False，正常地罗列depth1、depth2、depth3...等结点信息
        True，反向列举所属层级，即显示倒数第1层parent1，然后是倒数第2层parent2...
    :param childid_colname: 孩子结点列
    :param parentid_colname: 父结点列
    :param format_colname: 显示的数值
        None，默认采用 childid_colname 的值
        str，某一列的名称，采用那一列的值（可以实现设置好格式）
    :return:
    """
    # 1 构造辅助数组
    if format_colname is None: format_colname = parentid_colname
    parentid = dict()  # parentid[k] = v， 存储结点k对应的父结点v
    nodeval = dict()  # nodeval[k] = v，  存储结点k需要显示的数值情况
    if len(df[df.index.duplicated()]):
        dprint(len(set(df.index)), len(df.index))  # 有重复index
        raise ValueError

    for idx, row in df.iterrows():
        parentid[row[childid_colname]] = row[parentid_colname]
        nodeval[row[childid_colname]] = str(row[format_colname])

    # 2 每个结点往上遍历出所有父结点
    parents = []
    for idx, row in df.iterrows():
        ps = [nodeval[row[childid_colname]]]  # 包含结点自身的所有父结点名称
        p = row[parentid_colname]
        while p in parentid:
            ps.append(nodeval[p])
            p = parentid[p]
        parents.append(ps)
    num_depth = max(map(len, parents), default=0)

    # 3 这里可以灵活调整最终要显示的格式效果
    df['parents'] = parents
    if reverse:
        for j in range(num_depth, 0, -1): df[f'depth-{j}'] = ''
        for idx, row in df.iterrows():
            for j in range(1, len(row.parents) + 1):
                df.loc[idx, f'depth-{j}'] = row.parents[j - 1]
    else:
        for j in range(num_depth): df[f'depth{j}'] = ''
        for idx, row in df.iterrows():
            for j in range(len(row.parents)):
                df.loc[idx, f'depth{j}'] = row.parents[-j - 1]
    df.drop('parents', axis=1, inplace=True)
    return df


____section_2_xml = """
xml相关的一些功能函数
"""


def readurl(url):
    """从url读取文本"""
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    s = soup.get_text()
    return s


____section_3_xmlparser = """
"""


class EnchantBs4Tag(EnchantBase):
    @classmethod
    @RunOnlyOnce
    def enchant(cls):
        """ 把xlcv的功能嵌入cv2中

        不太推荐使用该类，可以使用CvImg类更好地解决问题。
        """
        names = cls.check_enchant_names([bs4.Tag])
        propertys = {'tag_name'}
        cls._enchant(bs4.Tag, propertys, mode='staticmethod2property')
        cls._enchant(bs4.Tag, names - propertys, mode='staticmethod2objectmethod')

    @staticmethod
    def tag_name(self):
        """输入一个bs4的Tag或NavigableString，
        返回tag.name或者'NavigableString'
        """
        if self.name:
            return self.name
        elif isinstance(self, bs4.NavigableString):
            return 'NavigableString'
        else:
            dprint(self)  # 获取结点t名称失败
            return None

    @staticmethod
    def subtag_names(self):
        """ 列出结点的所有直接子结点（花括号后面跟的数字是连续出现次数）
        例如body的： p{137}，tbl，p{94}，tbl，p{1640}，sectPr
        """

        def counter(m):
            s1 = m.group(1)
            n = (m.end(0) - m.start(0)) // len(s1)
            s = s1[:-1] + '{' + str(n) + '}'
            if m.string[m.end(0) - 1] == '，':
                s += '，'
            return s

        if self.name and self.contents:
            s = '，'.join([x.tag_name for x in self.contents]) + '，'
            s = re.sub(r'([^，]+，)(\1)+', counter, s)
        else:
            s = ''
        if s and s[-1] == '，':
            s = s[:-1]
        return s

    @staticmethod
    def treestruct_raw(self, **kwargs):
        """ 查看树形结构的raw版本
        各参数含义详见dfs_base
        """
        # 1 先用dfs获得基本结果
        s = dfs_base(self, **kwargs)
        return s

    @staticmethod
    def treestruct_brief(self, linenum=True, prefix='- ', **kwargs):
        """ 查看树形结构的简洁版
        """

        def mystr(node):
            # if isinstance(node, (bs4.ProcessingInstruction, code4101py.stdlib.bs4.ProcessingInstruction)):
            if isinstance(node, bs4.ProcessingInstruction):
                s = 'ProcessingInstruction，' + str(node)
            # elif isinstance(node, (bs4.Tag, code4101py.stdlib.bs4.Tag)):
            elif isinstance(node, bs4.Tag):
                s = node.name + '，' + xldictstr(node.attrs, item_delimit='，')
            # elif isinstance(node, (bs4.NavigableString, code4101py.stdlib.bs4.NavigableString)):
            elif isinstance(node, bs4.NavigableString):
                # s = 'NavigableString'
                s = shorten(str(node), 200)
                if not s.strip():
                    s = '<??>'
            else:
                s = '遇到特殊类型，' + str(node)
            return s

        s = dfs_base(self, mystr=mystr, prefix=prefix, linenum=linenum, **kwargs)
        return s

    @staticmethod
    def treestruct_stat(self):
        """生成一个两个二维表的统计数据
            ls1, ls2 = treestruct_stat()
                ls1： 结点规律表
                ls2： 属性规律表
        count_tagname、check_tag的功能基本都可以被这个函数代替
        """

        def text(t):
            """ 考虑到结果一般都是存储到excel，所以会把无法存成gbk的字符串删掉
            另外控制了每个元素的长度上限
            """
            s = ensure_gbk(t)
            s = s[:100]
            return s

        def depth(t):
            """结点t的深度"""
            return len(tuple(t.parents))

        t = self.contents[0]
        # ls1 = [['element序号', '层级', '结构', '父结点', '当前结点', '属性值/字符串值', '直接子结点结构']]
        # ls2 = [['序号', 'element序号', '当前结点', '属性名', '属性值']]  #
        ls1 = []  # 这个重点是分析结点规律
        ls2 = []  # 这个重点是分析属性规律
        i = 1
        while t:
            # 1 结点规律表
            d = depth(t)
            line = [i, d, '_' * d + str(d), t.parent.tag_name, t.tag_name,
                    text(xldictstr(t.attrs) if t.name else t),  # 结点存属性，字符串存值
                    t.subtag_names()]
            ls1.append(line)
            # 2 属性规律表
            if t.name:
                k = len(ls2)
                for attr, value in t.attrs.items():
                    ls2.append([k, i, t.tag_name, attr, value])
                    k += 1
            # 下个结点
            t = t.next_element
            i += 1
        df1 = pd.DataFrame.from_records(ls1, columns=['element序号', '层级', '结构', '父结点', '当前结点', '属性值/字符串值', '直接子结点结构'])
        df2 = pd.DataFrame.from_records(ls2, columns=['序号', 'element序号', '当前结点', '属性名', '属性值'])
        return df1, df2

    @staticmethod
    def count_tagname(self):
        """统计每个标签出现的次数：
             1                    w:rpr  650
             2                 w:rfonts  650
             3                   w:szcs  618
             4                      w:r  565
             5                     None  532
             6                      w:t  531
        """
        ct = collections.Counter()

        def inner(node):
            try:
                ct[node.name] += 1
                for t in node.children:
                    inner(t)
            except AttributeError:
                pass

        inner(self)
        return ct.most_common()

    @staticmethod
    def check_tag(self, tagname=None):
        """ 统计每个标签在不同层级出现的次数：

        :param tagname:
            None：统计全文出现的各种标签在不同层级出现次数
            't'等值： tagname参数允许只检查特殊标签情况，此时会将所有tagname设为第0级

        TODO 检查一个标签内部是否有同名标签？
        """
        d = defaultdict()

        def add(name, depth):
            if name not in d:
                d[name] = defaultdict(int)
            d[name][depth] += 1

        def inner(node, depth):
            if isinstance(node, bs4.ProcessingInstruction):
                add('ProcessingInstruction', depth)
            elif isinstance(node, bs4.Tag):
                if node.name == tagname and depth:
                    dprint(node, depth)  # tagname里有同名子标签
                add(node.name, depth)
                for t in node.children:
                    inner(t, depth + 1)
            elif isinstance(node, bs4.NavigableString):
                add('NavigableString', depth)
            else:
                add('其他特殊结点', depth)

        # 1 统计结点在每一层出现的次数
        if tagname:
            for t in self.find_all(tagname):
                inner(t, 0)
        else:
            inner(self, 0)

        # 2 总出现次数和？

        return d

    @staticmethod
    def check_namespace(self):
        """检查名称空间问题，会同时检查标签名和属性名：
            1  cNvPr  pic:cNvPr(579)，wps:cNvPr(52)，wpg:cNvPr(15)
            2   spPr                   pic:spPr(579)，wps:spPr(52)
        """
        # 1 获得所有名称
        #    因为是采用node的原始xml文本，所以能保证会取得带有名称空间的文本内容
        ct0 = Counter(re.findall(r'<([a-zA-Z:]+)', str(self)))
        ct = defaultdict(str)
        s = set()
        for key, value in ct0.items():
            k = re.sub(r'.*:', '', key)
            if k in ct:
                s.add(k)
                ct[k] += f'，{key}({value})'
            else:
                ct[k] = f'{key}({value})'

        # 2 对有重复和无重复的元素划分存储
        ls1 = []  # 有重复的存储到ls1
        ls2 = []  # 没有重复的正常结果存储到ls2，可以不显示
        for k, v in ct.items():
            if k in s:
                ls1.append([k, v])
            else:
                ls2.append([k, v])

        # 3 显示有重复的情况
        # browser(ls1, filename='检查名称空间问题')
        return ls1

    @staticmethod
    def get_catalogue(self, *args, size=False, start_level=-1, **kwargs):
        """ 找到所有的h生成文本版的目录

        :param bool|int size: 布尔或者乘因子，表示是否展示文本，以及乘以倍率，比如双语阅读时，size可以缩放一半

        *args, **kwargs 参考 BookContents.format_str

        注意这里算法跟css样式不太一样，避免这里能写代码，能做更细腻的操作
        """
        bc = BookContents()
        for h in self.find_all(re.compile(r'h\d')):
            if size:
                part_size = h.section_text_size(size, fmt=True)
                bc.add(int(h.name[1]), h.get_text().replace('\n', ' '), part_size)
            else:
                bc.add(int(h.name[1]), h.get_text().replace('\n', ' '))

        if 'page' not in kwargs:
            kwargs['page'] = size

        if bc.contents:
            return bc.format_str(*args, start_level=start_level, **kwargs)
        else:
            return ''

    @staticmethod
    def section_text_size(self, factor=1, fmt=False):
        """ 计算某节标题下的正文内容长度 """
        if not re.match(r'h\d+$', self.name):
            raise TypeError

        # 这应该是相对比较简便的计算每一节内容多长的算法~~
        part_size = 0
        for x in self.next_siblings:
            if x.name == self.name:
                break
            else:
                text = str(x) if isinstance(x, bs4.NavigableString) else x.get_text()
                part_size += strwidth(text)
        part_size = round_int(part_size * factor)

        if fmt:
            return format_size(part_size).replace(' ', '').replace('bytes', 'B')
        else:
            return part_size

    @staticmethod
    def head_add_size(self, factor=1):
        """ 标题增加每节内容大小标记

        :param factor: 乘因子，默认是1。但双语阅读等情况，内容会多拷贝一份，此时可以乘以0.5，显示正常原文的大小。
        """
        for h in self.find_all(re.compile(r'h\d')):
            part_size = h.section_text_size(factor, fmt=True)
            navi_str = list(h.strings)[-1].rstrip()
            navi_str.replace_with(str(navi_str) + '，' + part_size)

    @staticmethod
    def head_add_number(self, start_level=-1, jump=True):
        """ 标题增加每节编号
        """
        bc = BookContents()
        heads = list(self.find_all(re.compile(r'h\d')))
        for h in heads:
            bc.add(int(h.name[1]), h.get_text().replace('\n', ' '))

        if not bc.contents:
            return

        nums = bc.format_numbers(start_level=start_level, jump=jump)
        for i, h in enumerate(heads):
            navi_str = list(h.strings)[0]
            if nums[i]:
                nums[i] += '&nbsp;'
            navi_str.replace_with(nums[i] + str(navi_str))

    @staticmethod
    def xltext(self):
        """ 自己特用的文本化方法

        有些空格会丢掉，要用这句转回来
        """
        return self.prettify(formatter=lambda s: s.replace(u'\xa0', '&nbsp;'))


EnchantBs4Tag.enchant()

____section_temp = """
"""


def mathjax_html_head(s):
    """增加mathjax解析脚本"""
    head = r"""<!DOCTYPE html>
<html>
<head>
<head><meta http-equiv=Content-Type content="text/html;charset=utf-8"></head>
<script src="https://a.cdn.histudy.com/lib/config/mathjax_config-klxx.js?v=1.1"></script>
<script type="text/javascript" async src="https://a.cdn.histudy.com/lib/mathjax/2.7.1/MathJax/MathJax.js?config=TeX-AMS-MML_SVG">
MathJax.Hub.Config(MATHJAX_KLXX_CONFIG);
</script>
</head>
<body>"""
    tail = '</body></html>'
    return head + s + tail


class MakeHtmlNavigation:
    """ 给网页添加一个带有超链接跳转的导航栏 """

    @classmethod
    def from_url(cls, url, **kwargs):
        """ 自动下载url的内容，缓存到本地后，加上导航栏打开 """
        content = requests.get(url).content.decode('utf8')
        etag = get_etag(url)  # 直接算url的etag，不用很严谨
        return cls.from_content(content, etag, **kwargs)

    @classmethod
    def from_file(cls, file, **kwargs):
        """ 输入本地一个html文件的路径，加上导航栏打开 """
        file = File(file)
        content = file.read()
        # 输入文件的情况，生成的_content等html要在同目录
        return cls.from_content(content, os.path.splitext(str(file))[0], **kwargs)

    @classmethod
    def from_content(cls, html_content, title='temphtml', *, encoding=None):
        """
        :param html_content: 原始网页的完整内容
        :param title: 页面标题，默认会先找head/title，如果没有，则取一个随机名称（TODO 未实装，目前固定名称）
        :param encoding: 保存的几个文件编码，默认是utf8，但windows平台有些特殊场合也可能要存储gbk

        算法基本原理：读取原网页，找出所有h标签，并增设a锚点
            另外生成一个导航html文件
            然后再生成一个主文件，让用户通过主文件来浏览页面

        # 读取csdn博客并展示目录 （不过因为这个存在跳级，效果不是那么好）
        >> file = 自动制作网页标题的导航栏(requests.get(r'https://blog.csdn.net/code4101/article/details/83009000').content.decode('utf8'))
        >> browser(str(file))
        http://i2.tiimg.com/582188/64f40d235705de69.png
        """
        from humanfriendly import format_size

        # 1 对原html，设置锚点，生成一个新的文件f2
        cnt = 0

        # 这个refs是可以用py算法生成的，目前是存储在github上引用
        refs = ['<html><head>',
                '<link rel=Stylesheet type="text/css" media=all href="https://code4101.github.io/css/navigation0.css">',
                '</head><body>']

        f2 = File(title + '_content', Dir.TEMP, suffix='.html')

        def func(m):
            nonlocal cnt
            cnt += 1
            name, content = m.group('name'), m.group('inner')
            content = BeautifulSoup(content, 'lxml').get_text()
            # 要写<h><a></a></h>，不能写<a><h></h></a>，否则css中设置的计数器重置不会起作用
            refs.append(f'<{name}><a href="{f2}#navigation{cnt}" target="showframe">{content}</a></{name}>')
            return f'<a name="navigation{cnt}"/>' + m.group()

        html_content = re.sub(r'<(?P<name>h\d+)(?:>|\s.*?>)(?P<body>\s*(?P<inner>.*?)\s*)</\1>',
                              func, html_content, flags=re.DOTALL)
        f2 = f2.write(html_content, encoding=encoding, if_exists='replace')

        # 2 f1除了导航栏，可以多附带一些有用的参考信息
        # 2.1 前文的refs已经存储了超链接的导航

        # 2.2 文本版的目录
        refs.append(f'<br/>【文本版的目录】')
        bs = BeautifulSoup(html_content, 'lxml')
        catalogue = bs.get_catalogue(indent='\t', start_level=-1, jump=True, size=True)
        refs.append(f'<pre>{catalogue}</pre>')

        # 2.3 文章总大小
        text = bs.get_text()
        n = strwidth(text)
        refs.append('<br/>【Total Bytes】' + format_size(n))

        # 2.4 文中使用的高频词
        # 英文可以直接按空格切开统计，区分大小写
        text2 = re.sub(grp_chinese_char(), '', text)  # 删除中文，先不做中文的功能~~
        text2 = re.sub(r'[,\.，。\(\)（）;；?？"]', ' ', text2)  # 标点符号按空格处理
        words = Counter(text2.split())
        msg = '\n'.join([(x[0] if x[1] == 1 else f'{x[0]}，{x[1]}') for x in words.most_common()])
        msg += f'<br/>共{len(words)}个词汇，用词数{sum(words.values())}。'
        refs.append(f'<br/>【词汇表】<pre>{msg}</pre>')

        # 2.5 收尾，写入f1
        refs.append('</body>\n</html>')
        f1 = File(title + '_catalogue', Dir.TEMP, suffix='.html').write('\n'.join(refs), encoding=encoding,
                                                                        if_exists='replace')

        # 3 生成主页 f0
        main_content = f"""<html>
        <frameset cols="20%,80%">
        	<frame src="{f1}">
        	<frame src="{f2}" name="showframe">
        </frameset></html>"""

        f0 = File(title + '_index', Dir.TEMP, suffix='.html').write(main_content, encoding=encoding,
                                                                    if_exists='replace')
        return f0
