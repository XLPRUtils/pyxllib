#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/06/28 21:40

import datetime
import logging
import re
import warnings
import os

import bs4
import pytz
from xml.etree import ElementTree
from tqdm import tqdm

# 过滤这类警告
warnings.filterwarnings("ignore", category=bs4.MarkupResemblesLocatorWarning, module='bs4')

import win32com.client

if win32com.client.gencache.is_readonly:
    win32com.client.gencache.is_readonly = False
    win32com.client.gencache.Rebuild()

from pyxllib.prog.newbie import SingletonForEveryClass
from pyxllib.prog.pupil import Timeout
from pyxllib.file.specialist import XlPath
from pyxllib.algo.treelib import Node

"""
参考了onepy的实现，做了重构。OnePy：Provides pythonic wrappers around OneNote COM interfaces
"""

namespace = "{http://schemas.microsoft.com/office/onenote/2013/onenote}"

# 缓存文件地址
CACHE_DIR = XlPath.tempdir() / 'OneNote/SearchCache'
os.makedirs(CACHE_DIR, exist_ok=True)


class ONProcess(metaclass=SingletonForEveryClass):
    """ onenote 底层win32的接口 """

    def __init__(self, timeout=30):
        """ onenote的win32接口方法是驼峰命名，这个ONProcess做了一层功能封装
        而且估计理论上对所有可以获得的接口都做了封装了

        :param timeout: 读取单个页面的时候，限制用时，单位：秒
            本来只想限制5秒，但发现会有一些页面特别长，需要多一些时间~
            再后来发现还有更慢的页面，半分钟的都有，就再改成30秒了
        """
        # TODO 这里需要针对不同的OneNote版本做自动化兼容，不要让用户填版本
        #   因为让用户填版本，会存在多个实例化对象，使用getxml会有各种问题
        # 目前是支持onenote2016的，但不知道其他版本onenote会怎样
        self.process = win32com.client.gencache.EnsureDispatch('OneNote.Application')
        self.namespace = "{http://schemas.microsoft.com/office/onenote/2013/onenote}"

        # 官方原版的实现，但我觉得可以去掉版本号
        # try:
        #     if version == 16:
        #         self.process = win32com.client.gencache.EnsureDispatch('OneNote.Application')
        #         self.namespace = "{http://schemas.microsoft.com/office/onenote/2013/onenote}"
        #     if version == 15:
        #         self.process = win32com.client.gencache.EnsureDispatch('OneNote.Application.15')
        #         self.namespace = "{http://schemas.microsoft.com/office/onenote/2013/onenote}"
        #     if version == 14:
        #         self.process = win32com.client.gencache.EnsureDispatch('OneNote.Application.14')
        #         self.namespace = "{http://schemas.microsoft.com/office/onenote/2010/onenote}"
        # except Exception as e:
        #     # pywintypes.com_error: (-2147221005, '无效的类字符串', None, None)
        #     # pywintypes.com_error: (-2147221005, '无效的类字符串', None, None)
        #     print(e)
        #     print("error starting onenote {}".format(version))

        global namespace
        namespace = self.namespace

        self.timeout_seconds = timeout

    def get_hierarchy(self, start_node_id="", hierarchy_scope=4):
        """
          HierarchyScope
          0 - Gets just the start node specified and no descendants.
          1 - Gets the immediate child nodes of the start node, and no descendants in higher or lower subsection groups.
          2 - Gets all notebooks below the start node, or root.
          3 - Gets all sections below the start node, including sections in section groups and subsection groups.
          4 - Gets all pages below the start node, including all pages in section groups and subsection groups.
        """
        return self.process.GetHierarchy(start_node_id, hierarchy_scope)

    def update_hierarchy(self, changes_xml_in):
        try:
            self.process.UpdateHierarchy(changes_xml_in)
        except Exception as e:
            print("Could not Update Hierarchy")

    def open_hierarchy(self, path, relative_to_object_id, object_id, create_file_type=0):
        """
          CreateFileType
          0 - Creates no new object.
          1 - Creates a notebook with the specified name at the specified location.
          2 - Creates a section group with the specified name at the specified location.
          3 - Creates a section with the specified name at the specified location.
        """
        try:
            return self.process.OpenHierarchy(path, relative_to_object_id, "", create_file_type)
        except Exception as e:
            print("Could not Open Hierarchy")

    def delete_hierarchy(self, object_id, excpect_last_modified=""):
        try:
            self.process.DeleteHierarchy(object_id, excpect_last_modified)
        except Exception as e:
            print("Could not Delete Hierarchy")

    def create_new_page(self, section_id, new_page_style=0):
        """
          NewPageStyle
          0 - Create a Page that has Default Page Style
          1 - Create a blank page with no title
          2 - Createa blank page that has no title
        """
        try:
            self.process.CreateNewPage(section_id, "", new_page_style)
        except Exception as e:
            print("Unable to create the page")

    def close_notebook(self, notebook_id):
        try:
            self.process.CloseNotebook(notebook_id)
        except Exception as e:
            print("Could not Close Notebook")

    def get_page_content(self, page_id, page_info=0):
        """
          PageInfo
          0 - Returns only file page content, without selection markup and binary data objects. This is the standard value to pass.
          1 - Returns page content with no selection markup, but with all binary data.
          2 - Returns page content with selection markup, but no binary data.
          3 - Returns page content with selection markup and all binary data.
        """
        with Timeout(self.timeout_seconds):
            return self.process.GetPageContent(page_id, "", page_info)

    def update_page_content(self, page_changes_xml_in, excpect_last_modified=0):
        try:
            self.process.UpdatePageContent(page_changes_xml_in, excpect_last_modified)
        except Exception as e:
            print("Could not Update Page Content")

    def get_binary_page_content(self, page_id, callback_id):
        try:
            return self.process.GetBinaryPageContent(page_id, callback_id)
        except Exception as e:
            print("Could not Get Binary Page Content")

    def delete_page_content(self, page_id, object_id, excpect_last_modified=0):
        try:
            self.process.DeletePageContent(page_id, object_id, excpect_last_modified)
        except Exception as e:
            print("Could not Delete Page Content")

    # Actions

    def navigate_to(self, object_id, new_window=False):
        try:
            self.process.NavigateTo(object_id, "", new_window)
        except Exception as e:
            print("Could not Navigate To")

    def publish(self, hierarchy_id, target_file_path, publish_format, clsid_of_exporter=""):
        """
         PublishFormat
          0 - Published page is in .one format.
          1 - Published page is in .onea format.
          2 - Published page is in .mht format.
          3 - Published page is in .pdf format.
          4 - Published page is in .xps format.
          5 - Published page is in .doc or .docx format.
          6 - Published page is in enhanced metafile (.emf) format.
        """
        try:
            self.process.Publish(hierarchy_id, target_file_path, publish_format, clsid_of_exporter)
        except Exception as e:
            print("Could not Publish")

    def open_package(self, path_package, path_dest):
        try:
            return self.process.OpenPackage(path_package, path_dest)
        except Exception as e:
            print("Could not Open Package")

    def get_hyperlink_to_object(self, hierarchy_id, target_file_path=""):
        try:
            return self.process.GetHyperlinkToObject(hierarchy_id, target_file_path)
        except Exception as e:
            print("Could not Get Hyperlink")

    def find_pages(self, start_node_id, search_string, display):
        try:
            return self.process.FindPages(start_node_id, search_string, "", False, display)
        except Exception as e:
            print("Could not Find Pages")

    def get_special_location(self, special_location=0):
        """
          SpecialLocation
          0 - Gets the path to the Backup Folders folder location.
          1 - Gets the path to the Unfiled Notes folder location.
          2 - Gets the path to the Default Notebook folder location.
        """
        try:
            return self.process.GetSpecialLocation(special_location)
        except Exception as e:
            print("Could not retreive special location")


class _CommonMethods:
    """ 笔记本、分区组、分区、页面 共有的一些成员方法 """

    @property
    def ancestors(self):
        """ 获得所有父结点 """
        parents = []
        p = self
        while getattr(p, 'parent', False):
            parents.append(p.parent)
            p = p.parent
        return reversed(parents)

    @property
    def abspath_name(self):
        names = [x.name for x in self.ancestors]
        names.append(self.name)
        return '/'.join(names)

    def search(self, pattern, child_depth=1, *, return_mode='text', padding_mode=0):
        """ 查找内容

        Page、Section、SectionGroup等实际所用的search方法

        :param pattern:
            text, 检索出现该关键词的node，并展示其相关的上下文内容
            func, 可以输入自定义函数 check_node(node)->bool，True表示符合检索条件的node
            re.compile，可以输入编译的正则模式，会使用re.search进行匹配
        :param int child_depth: 对于检索到的node，向下展开几层子结点
            -1，表示全部展开
            0，表示不展开子结点
            1，只展开直接子结点
            ...
        :param return_mode:
            text，文本展示
            html，网页富文本
        """
        # 1 按照规则检索内容
        if isinstance(pattern, str):
            def check_node(node):
                # 纯文本部分
                if pattern in node.name:
                    return True
                # 如果纯文本找不到，也会在富文本格式里尝试匹配
                html_text = getattr(node, '_html_content', '')
                if html_text and pattern in html_text:
                    return True
                return False

        elif isinstance(pattern, re.Pattern):
            def check_node(node):
                if pattern.search(node.name):
                    return True
                html_text = getattr(node, '_html_content', '')
                if html_text and pattern.search(html_text):
                    return True
                return False
        else:
            check_node = pattern

        root = self._search(print_mode=True)  # 显示进度条
        root.sign_node(check_node, flag_name='_flag', child_depth=child_depth)

        # 2 展示内容
        if return_mode == 'text':
            return root.render(filter_=lambda x: getattr(x, '_flag', 0))
        elif return_mode == 'html':
            return root.render_html('_html_content',
                                    filter_=lambda x: getattr(x, '_flag', 0),
                                    padding_mode=padding_mode)
        else:
            raise ValueError


class OneNote(ONProcess, _CommonMethods):
    """ OneNote软件，这是一个单例类

    注意，从ONProcess继承的OneNote也是单例类
    但从ONProcess、OneNote生成的是不同的两个对象
    """

    def __init__(self, timeout=30):
        """
        如果出现这个错误：This COM object can not automate the makepy process - please run makepy manually for this object
            可以照 https://github.com/varunsrin/one-py 文章末尾的方式操作
            把 HKEY_CLASSES_ROOT\TypeLib\{0EA692EE-BB50-4E3C-AEF0-356D91732725} 的 1.0 删掉
            （这个 KEY ID 值大家电脑上都是一样的）
        """
        # trick: 这里有跟单例类有关的一些问题，导致ONProcess需要提前初始化一次
        super().__init__(timeout)

        self.xml = self.get_hierarchy("", 4)
        self.object_tree = ElementTree.fromstring(self.xml)
        self.hierarchy = Hierarchy(self.object_tree)
        self._children = list(self.hierarchy)
        self.name = 'onenote'

    def get_page_content(self, page_id):
        page_content_xml = ElementTree.fromstring(super(OneNote, self).get_page_content(page_id))
        return PageContent(page_content_xml)

    def update_page_content(self, page_changes_xml_in):
        """
        :param page_changes_xml_in:
            xml，可以是原始的xml文本
                onenote.update_page_content(page.getxml().replace('曹一众', '曹二众'))
            soup, 可以传入一个bs4的soup对象

        这里设置pytz时间的东西我也看不懂，但大受震撼~~有点莫名其妙
        How to debug win32com call in python：
        https://stackoverflow.com/questions/34904094/how-to-debug-win32com-call-in-python/34979646#34979646
        """
        return super(OneNote, self).update_page_content(page_changes_xml_in,
                                                        pytz.utc.localize(datetime.datetime(1899, 12, 30)))

    def names(self):
        """ 所有笔记本的名称 """
        ls = list(map(lambda x: x.name, self.hierarchy))
        return ls

    def nicknames(self):
        """ 所有笔记本的昵称 """
        ls = list(map(lambda x: x.nickname, self.hierarchy))
        return ls

    def __getitem__(self, item):
        """ 通过编号或名称索引获得笔记本 """
        return self.hierarchy[item]

    def _search(self, *, print_mode=False):
        root = Node('Notebook：' + self.name)
        for x in tqdm(self._children, desc='解析笔记本中', smoothing=0, disable=not print_mode):
            cur_node = x._search()
            cur_node.parent = root

        return root


class Hierarchy:

    def __init__(self, xml=None):
        self.xml = xml
        self._children = []
        if xml is not None:
            self.__deserialize_from_xml(xml)

    def __deserialize_from_xml(self, xml):
        self._children = [Notebook(n) for n in xml]

    def __iter__(self):
        for c in self._children:
            yield c

    def __getitem__(self, item):
        """通过编号或名称索引子节点内容"""
        if isinstance(item, int):
            return self._children[item]
        elif isinstance(item, str):
            for nb in self:
                if nb.nickname == item:
                    return nb
        return None


class HierarchyNode:

    def __init__(self, parent=None):
        self.name = ""
        self.path = ""
        self.id = ""
        self.last_modified_time = ""
        self.synchronized = ""

    def deserialize_from_xml(self, xml):
        self.name = xml.get("name")
        self.path = xml.get("path")
        self.id = xml.get("ID")
        self.last_modified_time = xml.get("lastModifiedTime")


class Notebook(HierarchyNode, _CommonMethods):

    def __init__(self, xml=None):
        self.xml = xml
        super().__init__(self)
        self.nickname = ""
        self.color = ""
        self.is_currently_viewed = ""
        self.recycleBin = None
        self._children = []
        if xml is not None:
            self.__deserialize_from_xml(xml)

    def __deserialize_from_xml(self, xml):
        HierarchyNode.deserialize_from_xml(self, xml)
        self.nickname = xml.get("nickname")
        self.color = xml.get("color")
        self.is_currently_viewed = xml.get("isCurrentlyViewed")
        self.recycleBin = None
        for node in xml:
            if node.tag == namespace + "Section":
                self._children.append(Section(node, self))

            elif node.tag == namespace + "SectionGroup":
                if node.get("isRecycleBin"):
                    self.recycleBin = SectionGroup(node, self)
                else:
                    self._children.append(SectionGroup(node, self))

    def __iter__(self):
        for c in self._children:
            yield c

    def __str__(self):
        return self.name

    def __getitem__(self, item):
        """通过编号或名称索引子节点内容"""
        if isinstance(item, int):
            return self._children[item]
        elif isinstance(item, str):
            for nb in self:
                if nb.name == item:
                    return nb
        return None

    def _search(self, *, print_mode=False):
        root = Node('Notebook：' + self.name)
        for x in tqdm(self._children, desc='解析分区(组)中', smoothing=0, disable=not print_mode):
            cur_node = x._search()
            cur_node.parent = root

        return root


class SectionGroup(HierarchyNode, _CommonMethods):
    """ 分区组 """

    def __init__(self, xml=None, parent_node=None):
        self.xml = xml
        super().__init__(self)
        self.is_recycle_Bin = False
        self._children = []
        self.parent = parent_node
        if xml is not None:
            self.__deserialize_from_xml(xml)

    def __iter__(self):
        # ckz: 这个遍历的时候，就是OneNote里看到的从左到右的顺序：先所有分区，然后所有分区组
        #
        for c in self._children:
            yield c

    def __str__(self):
        return self.name

    def __deserialize_from_xml(self, xml):
        HierarchyNode.deserialize_from_xml(self, xml)
        self.is_recycle_Bin = xml.get("isRecycleBin")
        for node in xml:
            if node.tag == namespace + "SectionGroup":
                self._children.append(SectionGroup(node, self))
            if node.tag == namespace + "Section":
                self._children.append(Section(node, self))

    def __getitem__(self, item):
        """ 通过 编号 或 名称 索引子节点内容

        注意使用字符串引用的时候，可能会有重名的问题！
        """
        if isinstance(item, int):
            return self._children[item]
        elif isinstance(item, str):
            for nb in self:
                if nb.name == item:
                    return nb
        return None

    def get_page_num(self):
        return sum([x.get_page_num() for x in self._children])

    def _search(self, *, print_mode=False):
        root = Node('SectionGroup：' + self.name)
        for x in tqdm(self._children, desc='解析分区(组)中', smoothing=0, disable=not print_mode):
            cur_node = x._search()
            cur_node.parent = root
        return root


class Section(HierarchyNode, _CommonMethods):
    """ 分区 """

    def __init__(self, xml=None, parent_node=None):
        self.xml = xml
        super().__init__(self)
        self.color = ""
        self.read_only = False
        self.is_currently_viewed = False
        self._children = []
        self.parent = parent_node
        if xml is not None:
            self.__deserialize_from_xml(xml)

    def __iter__(self):
        for c in self._children:
            yield c

    def __str__(self):
        return self.name

    def __deserialize_from_xml(self, xml):
        HierarchyNode.deserialize_from_xml(self, xml)
        self.color = xml.get("color")
        try:
            self.read_only = xml.get("readOnly")
        except Exception as e:
            self.read_only = False
        try:
            self.is_currently_viewed = xml.get("isCurrentlyViewed")
        except Exception as e:
            self.is_currently_viewed = False

        self._children = [Page(node, self) for node in xml]

    def __getitem__(self, item):
        """通过编号或名称索引子节点内容"""
        if isinstance(item, int):
            return self._children[item]
        elif isinstance(item, str):
            for nb in self:
                if nb.name == item:
                    return nb
        return None

    def get_page_num(self):
        return len(self._children)

    def _search(self, *, print_mode=False):
        root = Node('Section：' + self.name)
        page_lv1, page_lv2 = root, root

        for x in tqdm(self._children, desc='解析页面中', smoothing=0, disable=not print_mode):
            # print(x.name)
            cur_page = x._search()
            if x.page_level == '1':
                cur_page.parent = root
                page_lv1 = cur_page
            elif x.page_level == '2':
                cur_page.parent = page_lv1
                page_lv2 = cur_page
            else:
                cur_page.parent = page_lv2

        return root


class Page(_CommonMethods):
    """ 页面 """

    def __init__(self, xml=None, parent_node=None):
        self.xml = xml
        self.name = ""
        self.id = ""
        self.date_time = ""
        self.last_modified_time = ""
        self.page_level = ""
        self.is_currently_viewed = ""
        self._children = []
        self.parent = parent_node
        if xml is not None:  # != None is required here, since this can return false
            self.__deserialize_from_xml(xml)

    def __iter__(self):
        for c in self._children:
            yield c

    def __str__(self):
        return self.name

        # Get / Set Meta

    @property
    def root(self):
        p = self
        while getattr(p, 'parent', False):
            p = p.parent
        return p

    def __deserialize_from_xml(self, xml):
        self.xml = xml
        self.name = xml.get("name")
        self.id = xml.get("ID")
        self.date_time = xml.get("dateTime")
        self.last_modified_time = xml.get("lastModifiedTime")
        self.page_level = xml.get("pageLevel")
        self.is_currently_viewed = xml.get("isCurrentlyViewed")
        self._children = [Meta(node) for node in xml]

    def getxml(self, page_info=0):
        """ 获得页面的xml内容 """
        try:
            res = super(OneNote, onenote).get_page_content(self.id, page_info)
        except TimeoutError as e:
            e.args = [e.args[0] + f'\n\t{self.abspath_name} 页面获取失败，请检查可能包含的office公式并删除。' \
                                  f'并且看下您的OneNote可能无响应了，请重启OneNote。']
            raise e

        if res is None:
            logging.warning(f'{self.abspath_name} 未成功提取页面内容')

        return res

    def browser_xml(self):
        from pyxllib.debug.specialist import browser, XlPath
        xml = self.getxml()
        browser(xml, file=XlPath.tempfile('.xml'))

    def parse_oe_tree(self, root=None):
        """ 获得本Page页面的树形结构，返回一个Node根节点

        因为层次结构有多种不同的表现形式
            层次结构A：Outline 文本框
            层次结构B：h1、h2、h3 等标题结构
            层次结构C：正文里的缩进层级
        所以实现这里parent的引用，算法会稍复杂。需要动态更新，实时返回新的parent。
            比如在遍历tag.contents的时候，因为[层次结构C]的原因，会出现 parent = dfs_parse_node(y, parent) 的较奇怪的写法
            在parse_oe中，parent的层次实现规则，也会较复杂，有些trick
        """
        from pyxllib.text.xmllib import BeautifulSoup

        xml = self.getxml()
        soup = BeautifulSoup(xml or '', 'xml')

        # self.browser_xml()  # 可以用这个查原始的xml内容

        def parse_oe(x, parent):
            # 1 获得3个主要属性
            def get_text(x):
                if y := x.find('T', recursive=False):
                    t1 = BeautifulSoup(y.text, 'lxml').text
                    t2 = y.text
                elif y := x.find('Table', recursive=False):
                    # 先Columns标记了一共m列，每列的宽度
                    # 然后每一行是一个Row，里面有m个Cell
                    t1 = '[Table]'
                    t2 = t1
                elif y := x.find('Image', recursive=False):
                    t1 = '[Image]'
                    t2 = t1
                else:
                    t1 = ''
                    t2 = ''
                return t1, t2

            style_name = style_defs.get(x.get('quickStyleIndex', ''), '')
            pure_text, html_text = get_text(x)  # 文本内容
            m = x.find('OEChildren', recursive=False)  # 文本性质子结点

            # 空数据跳过
            # if not pure_text and not m:
            #     return parent

            # 2 处理层次结构B
            if re.match(r'h\d$', style_name):  # 标题类
                while True:
                    parent_style_name = getattr(parent, '_style_name', '')
                    if re.match(r'h\d$', parent_style_name) and parent_style_name >= style_name:
                        # 如果父结点也是标题类型，且数值上不大于当前结点，则当前结点的实际父结点要往上层找
                        parent = parent.parent
                    else:
                        break
                # 标题类，会重置parent，本身作为一个中间结点
                cur_node = parent = Node(pure_text, parent, _style_name=style_name, _html_content=html_text)
            else:
                cur_node = Node(pure_text, parent, _html_content=html_text)

            # 3 表格、图片等特殊结构增设层级
            if pure_text.startswith('[Table]'):
                for z in x.find_all('T'):
                    Node(BeautifulSoup(z.text, 'lxml').text, cur_node, _html_content=z.text)
            elif pure_text.startswith('[Image]'):
                y = x.find('Image', recursive=False)
                for z in y.get('alt', '').splitlines():
                    Node(z, cur_node)

            # 4 处理层次结构C
            if m:
                for y in m.find_all('OE', recursive=False):
                    parse_oe(y, cur_node)

            return parent

        def dfs_parse_node(x, parent):
            if isinstance(x, bs4.element.Tag):
                if x.name == 'QuickStyleDef':
                    style_defs[x['index']] = x['name']
                elif x.name == 'Title':
                    parent.root.name = BeautifulSoup(x.T.text, 'lxml').text
                elif x.name == 'Outline':
                    # 处理层次结构A
                    nonlocal outline_cnt
                    if outline_cnt:
                        if outline_cnt == 1:
                            parent = Node(f'Outline{outline_cnt}', parent)
                        else:
                            parent = Node(f'Outline{outline_cnt}', parent.parent)
                        outline_cnt += 1
                    for y in x.contents:
                        parent = dfs_parse_node(y, parent)
                elif x.name == 'OE':
                    parent = parse_oe(x, parent)
                else:
                    for y in x.contents:
                        parent = dfs_parse_node(y, parent)
            return parent  # parent有可能会更新

        if root is None:
            root = Node('root')
        style_defs = {}
        # trick: outline_cnt不仅用来标记是否有多个Outline需要设中介结点。也记录了当前Outline的编号。
        outline_cnt = 1 if len(soup.find_all('Outline')) > 1 else 0
        dfs_parse_node(soup, root)
        return root

    def get_page_num(self):
        return 1

    def _search(self, *, print_mode=False):
        """ 先生成所有结点

        :param print_mode: 统一所有的_search定义范式，虽然在Page这里其实没用~

        """
        file = CACHE_DIR / (self.id + self.last_modified_time.replace(':', '') + '.pkl')

        if file.is_file() and False:
            root = file.read_pkl()
        else:
            root = self.parse_oe_tree()
            root.name = self.name

            if not root.is_leaf:
                # 删除旧时间点的缓存文件，存储新的缓存文件
                for f in CACHE_DIR.glob(f'{self.id}*.pkl'):
                    f.delete()
                file.write_pkl(root)

        return root


class Meta:

    def __init__(self, xml=None):
        self.xml = xml
        self.name = ""
        self.content = ""
        if xml is not None:
            self.__deserialize_from_xml(xml)

    def __str__(self):
        return self.name

    def __deserialize_from_xml(self, xml):
        self.name = xml.get("name")
        self.id = xml.get("content")

    def getxml(self):
        """获得页面的"""
        return super(OneNote, onenote).get_page_content(self.id)


class PageContent:

    def __init__(self, xml=None):
        self.xml = xml
        self.name = ""
        self.id = ""
        self.date_time = ""
        self.last_modified_time = ""
        self.page_level = ""
        self.lang = ""
        self.is_currently_viewed = ""
        self._children = []
        self.files = []
        if xml is not None:
            self.__deserialize_from_xml(xml)

    def __iter__(self):
        for c in self._children:
            yield c

    def __str__(self):
        return self.name

    def __deserialize_from_xml(self, xml):
        self.name = xml.get("name")
        self.id = xml.get("ID")
        self.date_time = xml.get("dateTime")
        self.last_modified_time = xml.get("lastModifiedTime")
        self.page_level = xml.get("pageLevel")
        self.lang = xml.get("lang")
        self.is_currently_viewed = xml.get("isCurrentlyViewed")
        for node in xml:
            if node.tag == namespace + "Outline":
                self._children.append(Outline(node))
            elif node.tag == namespace + "Ink":
                self.files.append(Ink(node))
            elif node.tag == namespace + "Image":
                self.files.append(Image(node))
            elif node.tag == namespace + "InsertedFile":
                self.files.append(InsertedFile(node))
            elif node.tag == namespace + "Title":
                self._children.append(Title(node))


class Title:

    def __init__(self, xml=None):
        self.xml = xml
        self.style = ""
        self.lang = ""
        self._children = []
        if xml is not None:
            self.__deserialize_from_xml(xml)

    def __str__(self):
        return "Page Title"

    def __iter__(self):
        for c in self._children:
            yield c

    def __deserialize_from_xml(self, xml):
        self.style = xml.get("style")
        self.lang = xml.get("lang")
        for node in xml:
            if node.tag == namespace + "OE":
                self._children.append(OE(node, self))


class Outline:

    def __init__(self, xml=None):
        self.xml = xml
        self.author = ""
        self.author_initials = ""
        self.last_modified_by = ""
        self.last_modified_by_initials = ""
        self.last_modified_time = ""
        self.id = ""
        self._children = []
        if xml is not None:
            self.__deserialize_from_xml(xml)

    def __iter__(self):
        for c in self._children:
            yield c

    def __str__(self):
        return "Outline"

    def __deserialize_from_xml(self, xml):
        self.author = xml.get("author")
        self.author_initials = xml.get("authorInitials")
        self.last_modified_by = xml.get("lastModifiedBy")
        self.last_modified_by_initials = xml.get("lastModifiedByInitials")
        self.last_modified_time = xml.get("lastModifiedTime")
        self.id = xml.get("objectID")
        append = self._children.append
        for node in xml:
            if node.tag == namespace + "OEChildren":
                for childNode in node:
                    if childNode.tag == namespace + "OE":
                        append(OE(childNode, self))


class Position:

    def __init__(self, xml=None, parent_node=None):
        self.xml = xml
        self.x = ""
        self.y = ""
        self.z = ""
        self.parent = parent_node
        if xml is not None:
            self.__deserialize_from_xml(xml)

    def __deserialize_from_xml(self, xml):
        self.x = xml.get("x")
        self.y = xml.get("y")
        self.z = xml.get("z")


class Size:

    def __init__(self, xml=None, parent_node=None):
        self.xml = xml
        self.width = ""
        self.height = ""
        self.parent = parent_node
        if xml is not None:
            self.__deserialize_from_xml(xml)

    def __deserialize_from_xml(self, xml):
        self.width = xml.get("width")
        self.height = xml.get("height")


class OE:

    def __init__(self, xml=None, parent_node=None):
        self.xml = xml
        self.creation_time = ""
        self.last_modified_time = ""
        self.last_modified_by = ""
        self.id = ""
        self.alignment = ""
        self.quick_style_index = ""
        self.style = ""
        self.text = ""
        self._children = []
        self.parent = parent_node
        self.files = []
        if xml is not None:
            self.__deserialize_from_xml(xml)

    def __iter__(self):
        for c in self._children:
            yield c

    def __str__(self):
        try:
            return self.text
        except AttributeError:
            return "Empty OE"

    def __deserialize_from_xml(self, xml):
        self.creation_time = xml.get("creationTime")
        self.last_modified_time = xml.get("lastModifiedTime")
        self.last_modified_by = xml.get("lastModifiedBy")
        self.id = xml.get("objectID")
        self.alignment = xml.get("alignment")
        self.quick_style_index = xml.get("quickStyleIndex")
        self.style = xml.get("style")

        for node in xml:
            if node.tag == namespace + "T":
                if node.text is not None:
                    self.text = node.text
                else:
                    self.text = "NO TEXT"

            elif node.tag == namespace + "OEChildren":
                for childNode in node:
                    if childNode.tag == namespace + "OE":
                        self._children.append(OE(childNode, self))

            elif node.tag == namespace + "Image":
                self.files.append(Image(node, self))

            elif node.tag == namespace + "InkWord":
                self.files.append(Ink(node, self))

            elif node.tag == namespace + "InsertedFile":
                self.files.append(InsertedFile(node, self))


class InsertedFile:

    # need to add position data to this class

    def __init__(self, xml=None, parent_node=None):
        self.xml = xml
        self.path_cache = ""
        self.path_source = ""
        self.preferred_name = ""
        self.last_modified_time = ""
        self.last_modified_by = ""
        self.id = ""
        self.parent = parent_node
        if xml is not None:
            self.__deserialize_from_xml(xml)

    def __iter__(self):
        yield None

    def __str__(self):
        try:
            return self.preferredName
        except AttributeError:
            return "Unnamed File"

    def __deserialize_from_xml(self, xml):
        self.path_cache = xml.get("pathCache")
        self.path_source = xml.get("pathSource")
        self.preferred_name = xml.get("preferredName")
        self.last_modified_time = xml.get("lastModifiedTime")
        self.last_modified_by = xml.get("lastModifiedBy")
        self.id = xml.get("objectID")


class Ink:

    # need to add position data to this class

    def __init__(self, xml=None, parent_node=None):
        self.xml = xml
        self.recognized_text = ""
        self.x = ""
        self.y = ""
        self.ink_origin_x = ""
        self.ink_origin_y = ""
        self.width = ""
        self.height = ""
        self.data = ""
        self.callback_id = ""
        self.parent = parent_node

        if xml is not None:
            self.__deserialize_from_xml(xml)

    def __iter__(self):
        yield None

    def __str__(self):
        try:
            return self.recognizedText
        except AttributeError:
            return "Unrecognized Ink"

    def __deserialize_from_xml(self, xml):
        self.recognized_text = xml.get("recognizedText")
        self.x = xml.get("x")
        self.y = xml.get("y")
        self.ink_origin_x = xml.get("inkOriginX")
        self.ink_origin_y = xml.get("inkOriginY")
        self.width = xml.get("width")
        self.height = xml.get("height")

        for node in xml:
            if node.tag == namespace + "CallbackID":
                self.callback_id = node.get("callbackID")
            elif node.tag == namespace + "Data":
                self.data = node.text


class Image:

    def __init__(self, xml=None, parent_node=None):
        self.xml = xml
        self.format = ""
        self.original_page_number = ""
        self.last_modified_time = ""
        self.id = ""
        self.callback_id = None
        self.data = ""
        self.parent = parent_node
        if xml is not None:
            self.__deserialize_from_xml(xml)

    def __iter__(self):
        yield None

    def __str__(self):
        return self.format + " Image"

    def __deserialize_from_xml(self, xml):
        self.format = xml.get("format")
        self.original_page_number = xml.get("originalPageNumber")
        self.last_modified_time = xml.get("lastModifiedTime")
        self.id = xml.get("objectID")
        for node in xml:
            if node.tag == namespace + "CallbackID":
                self.callback_id = node.get("callbackID")
            elif node.tag == namespace + "Data":
                if node.text is not None:
                    self.data = node.text


onenote = OneNote()
onenote2 = OneNote()
