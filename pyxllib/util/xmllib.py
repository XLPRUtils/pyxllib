#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xml等网页结构方面的处理
"""

__author__ = '陈坤泽'
__email__ = '877362867@qq.com'
__date__ = '2018/09/28 14:37'


from code4101py.util.textlib import *
from collections import defaultdict, Counter

try:
    import bs4
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'BeautifulSoup4'])
    import bs4
    from bs4 import BeautifulSoup


____section_1_dfs_base = """
一个通用的递归功能
"""


def dfs_base(node, *,
             child_generator=None, select_depth=None, linenum=False,
             mystr=None, msghead=True, lsstr=None, show_node_type=False, prefix='    '):
    """输入一个节点node，以及该节点当前depth
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
    # 1、子节点生成器，与配置
    def bs4_child_generator(node):
        try:
            return node.children
        except AttributeError:
            return []

    # 配置子节点生成器
    if not child_generator:
        child_generator = bs4_child_generator

    # 2、dfs实际实现代码，获得节点清单
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

    # 4、过滤与重新整理ls（select_depth）
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

    # 5、格式处理
    def default_mystr(node, depth):
        s1 = prefix * depth
        s2 = typename(node)+'，' if show_node_type else ''
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
    # 0、参数预处理
    if isinstance(childreds, pd.DataFrame):
        df = childreds
        child_colname = parents
        parent_colname = arg3
        if not arg3: raise TypeError
        childreds = df[child_colname].tolist()
        parents = df[parent_colname].tolist()
    else:
        df = None

    # 1、建立root根节点，确保除了root其他结点都存在记录
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
    depth, tree_order, len_childs = [-1]*n, [-1]*n, [0]*n

    # 2、构造父结点-孩子结点的字典dd
    dd = defaultdict(list)
    for i in range(n): dd[ps[i]] += [i]

    # 3、dfs
    cnt = 1
    def dfs(node, d):
        """找node的所有子结点"""
        nonlocal cnt
        for i in dd.get(node, []):
            tree_order[i], depth[i], len_childs[i] = cnt, d, len(dd[cs[i]])
            cnt += 1
            dfs(cs[i], d+1)
    dfs(root, 1)

    # 4、输出格式
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
    # 1、构造辅助数组
    if format_colname is None: format_colname = parentid_colname
    parentid = dict()  # parentid[k] = v， 存储结点k对应的父结点v
    nodeval = dict()   # nodeval[k] = v，  存储结点k需要显示的数值情况
    if len(df[df.index.duplicated()]):
        dprint(len(set(df.index)), len(df.index))  # 有重复index
        raise ValueError

    for idx, row in df.iterrows():
        parentid[row[childid_colname]] = row[parentid_colname]
        nodeval[row[childid_colname]] = str(row[format_colname])

    # 2、每个结点往上遍历出所有父结点
    parents = []
    for idx, row in df.iterrows():
        ps = [nodeval[row[childid_colname]]]  # 包含结点自身的所有父结点名称
        p = row[parentid_colname]
        while p in parentid:
            ps.append(nodeval[p])
            p = parentid[p]
        parents.append(ps)
    num_depth = max(map(len, parents), default=0)

    # 3、这里可以灵活调整最终要显示的格式效果
    df['parents'] = parents
    if reverse:
        for j in range(num_depth, 0, -1): df[f'depth-{j}'] = ''
        for idx, row in df.iterrows():
            for j in range(1, len(row.parents)+1):
                df.loc[idx, f'depth-{j}'] = row.parents[j-1]
    else:
        for j in range(num_depth): df[f'depth{j}'] = ''
        for idx, row in df.iterrows():
            for j in range(len(row.parents)):
                df.loc[idx, f'depth{j}'] = row.parents[-j-1]
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


def tag_name(t):
    """输入一个bs4的Tag或NavigableString，
    返回tag.name或者'NavigableString'
    """
    if t.name:
        return t.name
    elif isinstance(t, bs4.NavigableString):
        return 'NavigableString'
    else:
        dprint(t)  # 获取结点t名称失败
        return None


def subtag_names(t):
    """列出结点t的所有直接子结点（花括号后面跟的数字是连续出现次数）
    例如body的： p{137}，tbl，p{94}，tbl，p{1640}，sectPr
    """
    def counter(m):
        s1 = m.group(1)
        n = (m.end(0) - m.start(0)) // len(s1)
        s = s1[:-1] + '{' + str(n) + '}'
        if m.string[m.end(0)-1] == '，':
            s += '，'
        return s

    if t.name and t.contents:
        s = '，'.join(map(tag_name, t.contents)) + '，'
        s = re.sub(r'([^，]+，)(\1)+', counter, s)
    else:
        s = ''
    if s and s[-1] == '，':
        s = s[:-1]
    return s


class XmlParser:
    def __init__(self, node=None):
        """两种初始化方式
            提供node：用某个bs4的PageElement等对象初始化
            未提供node，一般是方便给MyBs4等类继承使用
        """
        if node:  # TODO：可以扩展，支持不同类型的初始化
            self._node = node

    def node(self):
        """获得xml结点的接口函数"""
        return self._node if getattr(self, '_node') else self

    def treestruct_raw(self, **kwargs):
        """查看树形结构的raw版本
        各参数含义详见dfs_base
        """
        # 1、先用dfs获得基本结果
        s = dfs_base(self.node(), **kwargs)
        return s

    def treestruct_brief(self, linenum=True, prefix='- ', **kwargs):
        """查看树形结构的简洁版
        """
        def mystr(node):
            # if isinstance(node, (bs4.ProcessingInstruction, code4101py.stdlib.bs4.ProcessingInstruction)):
            if isinstance(node, bs4.ProcessingInstruction):
                s = 'ProcessingInstruction，' + str(node)
            # elif isinstance(node, (bs4.Tag, code4101py.stdlib.bs4.Tag)):
            elif isinstance(node, bs4.Tag):
                s = node.name + '，' + mydictstr(node.attrs, item_delimit='，')
            # elif isinstance(node, (bs4.NavigableString, code4101py.stdlib.bs4.NavigableString)):
            elif isinstance(node, bs4.NavigableString):
                # s = 'NavigableString'
                s = shorten(str(node), 200)
                if not s.strip():
                    s = '<??>'
            else:
                s = '遇到特殊类型，' + str(node)
            return s

        s = dfs_base(self.node(), mystr=mystr, prefix=prefix, linenum=linenum, **kwargs)
        return s

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
            # 1、结点规律表
            d = depth(t)
            line = [i, d, '_'*d+str(d), tag_name(t.parent), tag_name(t),
                    text(mydictstr(t.attrs) if t.name else t),  # 结点存属性，字符串存值
                    subtag_names(t)]
            ls1.append(line)
            # 2、属性规律表
            if t.name:
                k = len(ls2)
                for attr, value in t.attrs.items():
                    ls2.append([k, i, tag_name(t), attr, value])
                    k += 1
            # 下个结点
            t = t.next_element
            i += 1
        df1 = pd.DataFrame.from_records(ls1, columns=['element序号', '层级', '结构', '父结点', '当前结点', '属性值/字符串值', '直接子结点结构'])
        df2 = pd.DataFrame.from_records(ls2, columns=['序号', 'element序号', '当前结点', '属性名', '属性值'])
        return df1, df2

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

        inner(self.node())
        return ct.most_common()

    def check_tag(self, tagname=None):
        """统计每个标签在不同层级出现的次数：

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
                    inner(t, depth+1)
            elif isinstance(node, bs4.NavigableString):
                add('NavigableString', depth)
            else:
                add('其他特殊结点', depth)

        # 1、统计结点在每一层出现的次数
        if tagname:
            for t in self.node().find_all(tagname):
                inner(t, 0)
        else:
            inner(self.node(), 0)

        # 2、总出现次数和？

        return d

    def check_namespace(self):
        """检查名称空间问题，会同时检查标签名和属性名：
            1  cNvPr  pic:cNvPr(579)，wps:cNvPr(52)，wpg:cNvPr(15)
            2   spPr                   pic:spPr(579)，wps:spPr(52)
        """
        # 1、获得所有名称
        #    因为是采用node的原始xml文本，所以能保证会取得带有名称空间的文本内容
        ct0 = Counter(re.findall(r'<([a-zA-Z:]+)', str(self.node())))
        ct = defaultdict(str)
        s = set()
        for key, value in ct0.items():
            k = re.sub(r'.*:', '', key)
            if k in ct:
                s.add(k)
                ct[k] += f'，{key}({value})'
            else:
                ct[k] = f'{key}({value})'

        # 2、对有重复和无重复的元素划分存储
        ls1 = []  # 有重复的存储到ls1
        ls2 = []  # 没有重复的正常结果存储到ls2，可以不显示
        for k, v in ct.items():
            if k in s:
                ls1.append([k, v])
            else:
                ls2.append([k, v])

        # 3、显示有重复的情况
        # chrome(ls1, filename='检查名称空间问题')
        return ls1


class MyBs4(BeautifulSoup, XmlParser):
    """xml、html 等数据通用处理算法，常用功能有：

    show_brief：显示xml结构
    count_tagname： 统计各个结点名称出现次数
    """
    def __init__(self, markup="", features='lxml', *args, **kwargs):
        # markup = Path(markup).read()
        # TODO: **kwargs我不知道怎么传进来啊，不过感觉也不删大雅没什么鸟用吧~~
        super().__init__(markup, features, *args, **kwargs)

    def insert_after(self, successor):
        pass

    def insert_before(self, successor):
        pass


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


def 自动制作网页标题的导航栏(html_content, title='temphtml'):
    """
    :param html_content: 原始网页的完整内容
    :param title: 页面标题，默认会先找head/title，如果没有，则取一个随机名称（TODO 未实装，目前固定名称'test'）

    算法基本原理：读取原网页，找出所有h标签，并增设a锚点
        另外生成一个导航html文件
        然后再生成一个主文件，让用户通过主文件来浏览页面

    # 读取csdn博客并展示目录 （不过因为这个存在跳级，效果不是那么好）
    >> file = 自动制作网页标题的导航栏(requests.get(r'https://blog.csdn.net/code4101/article/details/83009000').content.decode('utf8'))
    >> chrome(str(file))
    http://i2.tiimg.com/582188/64f40d235705de69.png
    """
    # 1、对原html，设置锚点，生成一个新的文件f2；生成导航目录文件f1。
    cnt = 0

    # TODO 目前不支持跳级的情况
    # 这个refs是可以用py算法生成的，目前是存储在github上引用
    refs = ['<html><head>',
            '<link rel=Stylesheet type="text/css" media=all href="https://code4101.github.io/css/navigation0.css">',
            '</head><body>']

    f2 = Path(title + '_内容', '.html', Path.TEMP)

    def func(m):
        nonlocal cnt
        cnt += 1
        name, content = m.group('name'), m.group('inner')
        content = BeautifulSoup(content, 'lxml').get_text()
        refs.append(f'<a href="{f2}#生成导航栏浏览网页{cnt}" target="showframe"><{name}>{content}</{name}></a>')
        return f'<a name="生成导航栏浏览网页{cnt}"/>' + m.group()

    html_content = re.sub(r'<(?P<name>h\d+)(?:>|\s.*?>)(?P<body>\s*(?P<inner>.*?)\s*)</\1>',
                          func, html_content, flags=re.DOTALL)

    refs.append('</body>\n</html>')

    f1 = Path(title + '_导航', '.html', Path.TEMP).write('\n'.join(refs))
    f2 = f2.write(html_content)

    # 2、生成首页 f0
    main_content = f"""<html>
<frameset cols="20%,80%">
	<frame src="{f1}">
	<frame src="{f2}" name="showframe">
</frameset></html>"""

    f0 = Path(title, '.html', Path.TEMP).write(main_content)
    return f0
