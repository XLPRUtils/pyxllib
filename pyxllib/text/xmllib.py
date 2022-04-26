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
import os

import requests
import pandas as pd
import bs4
from bs4 import BeautifulSoup
from humanfriendly import format_size

from pyxllib.algo.pupil import SearchBase
from pyxllib.debug.pupil import dprint
from pyxllib.debug.specialist import browser
from pyxllib.prog.newbie import round_int
from pyxllib.prog.pupil import EnchantBase, EnchantCvt, run_once
from pyxllib.text.newbie import xldictstr
from pyxllib.text.pupil import shorten, ensure_gbk, BookContents, strwidth, grp_chinese_char
from pyxllib.file.specialist import File, Dir, get_etag


class EnchantBs4Tag(EnchantBase):
    @classmethod
    @run_once
    def enchant(cls):
        """ 把xlcv的功能嵌入cv2中

        不太推荐使用该类，可以使用CvImg类更好地解决问题。
        """
        names = cls.check_enchant_names([bs4.element.Tag])
        propertys = {'tag_name'}
        cls._enchant(bs4.element.Tag, propertys, EnchantCvt.staticmethod2property)
        cls._enchant(bs4.element.Tag, names - propertys)

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
        sb = SearchBase(self)
        s = sb.fmt_nodes(**kwargs)
        return s

    @staticmethod
    def treestruct_brief(self, linenum=True, prefix='- ', **kwargs):
        """ 查看树形结构的简洁版
        """

        class Search(SearchBase):
            def fmt_node(self, node, depth, *, prefix=prefix, show_node_type=False):
                if isinstance(node, bs4.ProcessingInstruction):
                    s = 'ProcessingInstruction，' + str(node)
                elif isinstance(node, bs4.Tag):
                    s = node.name + '，' + xldictstr(node.attrs, item_delimit='，')
                elif isinstance(node, bs4.NavigableString):
                    s = shorten(str(node), 200)
                    if not s.strip():
                        s = '<??>'
                else:
                    s = '遇到特殊类型，' + str(node)
                return (prefix * depth) + s

        search = Search(self)
        res = search.fmt_nodes(linenum=linenum, **kwargs)
        return res

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
            navi_strs = list(h.strings)
            if navi_strs:
                navi_str = navi_strs[0]
                if nums[i]:
                    navi_str.replace_with(nums[i] + ' ' + str(navi_str))
            else:
                h.string = nums[i]

    @staticmethod
    def xltext(self):
        """ 自己特用的文本化方法

        有些空格会丢掉，要用这句转回来

        210924周五20:23，但后续实验又遭到了质疑，目前这功能虽然留着，但不建议使用
        """
        # return self.prettify(formatter=lambda s: s.replace(u'\xa0', '&nbsp;'))
        # \xa0好像是些特殊字符，删掉就行。。。  不对，也不是特殊字符~~
        # return self.prettify(formatter=lambda s: s.replace(u'\xa0', ''))
        # return self.prettify()
        return str(self)

    @staticmethod
    def browser(self):
        browser.html(self)

    @staticmethod
    @run_once('id,str')
    def get_nonempty_childrens(self, *args):
        """ 获得所有Tag类型的直接子结点 （偏定制，不是那么通用的接口）

        会同时检查NavigableString类型，且必须是空白字符串，比如空格、\n之类
        """

        def check(x):
            if isinstance(x, bs4.element.Tag):
                return True
            elif isinstance(x, bs4.element.Comment):
                return False
            elif isinstance(x, bs4.element.NavigableString):
                assert not x.strip(), f'非空字符串值：{x}'
                return False
            else:
                raise ValueError(f'未见类型 {x}')

        ls = list(filter(check, self.children))

        if len(args):
            return ls[args[0]].get_nonempty_childrens(*args[1:])
        else:
            return ls

    @staticmethod
    def get_nonempty_children(self, *args):
        if len(args):
            ls = self.get_nonempty_childrens(*args[:-1])
            return ls[args[-1]]
        else:
            return self


EnchantBs4Tag.enchant()


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


def html_bitran_template(htmlcontent):
    """ 双语翻译的html模板，html bilingual translation template

    一般是将word导出的html文件，转成方便谷歌翻译操作，进行双语对照的格式

    基本原理，是利用chrome识别class="notranslate"标记会跳过不翻译的特性
    对正文标签p拷贝两份，一份原文，一份带notranslate标记的内容
    这样在执行谷歌翻译后，就能出现双语对照的效果

    其实最好的办法，是能调用翻译API，直接给出双语成果的html
    但谷歌的googletrans连不上外网无法使用
    其他公司的翻译接口应该没问题，但我嫌其可能没有google好，以及不是重点，就先暂缓开发
    ---
    习惯来说，一般上面是英文，下面是中文，但是我又想使用中文标题~~
    """
    from pyxllib.text.nestenv import NestEnv

    # 0 将所有负margin-left变为0
    htmlcontent = re.sub(r'margin-left:-\d+(\.\d+)', 'margin-left:0', htmlcontent)

    # 1 区间定位分组
    ne = NestEnv(htmlcontent)
    ne2 = ne.xmltag('p')
    for name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'ol', 'li'):
        ne2 += ne.xmltag(name, symmetry=True)

    # 以下是针对python document复制到word的情况，不一定具有广泛泛用性
    # 目的是让代码块按块复制，而不是按行复制
    ne2 += ne.find2(re.compile("<div style=['\"]mso-element:para-border-div;.+?#AACC99"), '</div>')

    # 2 每个区间的处理规则
    def func(s):
        """ 找出p、h后，具体对每个tag要执行的操作

        分前后两波文本s1（原文），s2（翻译文）
        """

        # 1 s1 只要加 notranslate
        s1 = s
        bs = BeautifulSoup(s1, 'lxml')
        x = next(bs.body.children)
        cls_ = x.get('class', None)
        x['class'] = (cls_ + ['notranslate']) if cls_ else 'notranslate'
        s1 = x.prettify()

        # 2 s2 可能要做些骚操作
        s2 = s
        bs = BeautifulSoup(s2, 'lxml')
        x = next(bs.body.children)

        # 比如自定义翻译，这个无伤大雅的，如果搞不定，可以先注释掉，后面再说
        # if re.match(r'h\d+$', x.name):
        #     for y in x.descendants:
        #         if isinstance(y, NavigableString):
        #             y.replace_with(re.sub(r'Conclusion', '总结', str(y)))
        #         else:
        #             for z in y.strings:
        #                 z.replace_with(re.sub(r'Conclusion', '总结', str(z)))
        #     y.replace_with(re.sub(r'^Abstract$', '摘要', str(y)))
        # s2 = str(x)

        if re.match(r'h\d+$', x.name):
            x.name = 'p'  # 去掉标题格式，统一为段落格式
            s2 = x.prettify()
        elif x.name in ('div', 'pre'):
            # 实际使用体验，想了下，代码块还是不如保留原样最方便，不用拷贝翻译
            s2 = ''
        # 如果p没有文本字符串，也不拷贝
        if not x.get_text().strip():
            s2 = ''
        # if x.name == 'p' and x.get('style', None) and 'margin-left' in x['style']:
        #     x['style'] = re.sub(r'(margin-left:)\d+(\.\d+)?', r'\g<1>0', x['style'])

        return s1 + '\n' + s2

    res = ne2.replace(func)

    return res


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
    def from_content(cls, html_content, title='temphtml', *,
                     encoding=None, number=True, text_catalogue=True):
        """
        :param html_content: 原始网页的完整内容
        :param title: 页面标题，默认会先找head/title，如果没有，则取一个随机名称（TODO 未实装，目前固定名称）
        :param encoding: 保存的几个文件编码，默认是utf8，但windows平台有些特殊场合也可能要存储gbk
        :param number: 是否对每节启用自动编号的css

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
                '<link rel=Stylesheet type="text/css" media=all '
                f'href="https://code4101.github.io/css/navigation{int(number)}.css">',
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
        bs = BeautifulSoup(html_content, 'lxml')
        text = bs.get_text()
        if text_catalogue:
            # 目录
            refs.append(f'<br/>【文本版的目录】')
            catalogue = bs.get_catalogue(indent='\t', start_level=-1, jump=True, size=True)
            refs.append(f'<pre>{catalogue}</pre>')
            # 全文长度
            n = strwidth(text)
            refs.append('<br/>【Total Bytes】' + format_size(n))

        # 2.3 文中使用的高频词
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


class HtmlParser:
    """ 对树形结构、位置比较固定的html文档的一个解析框架 """

    def __init__(self, root):
        """ 输入根节点root """
        self.root = root

    @classmethod
    @run_once
    def get_parse_funcs(cls):
        res = []

        # 获取所有的方法名
        members = dir(cls)
        methods = filter(lambda m: callable(getattr(cls, m)), members)

        # 以parse、parse_0、parse_0_2等格式命名的函数，是解析树结构特定位置，这里自动执行解析
        for method in methods:
            if re.match(r'parse(_\d+)*$', method):
                # 智能获取对应下标的结构变量
                res.append(method)

        return res

    def run(self):
        for method in self.get_parse_funcs():
            # 智能获取对应下标的结构变量
            idxs = [int(v) for v in method[5:].split('_') if v]
            x = self.root.get_nonempty_children(*idxs)
            # 自动执行函数
            getattr(self, method)(x)
