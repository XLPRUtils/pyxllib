#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/06/28 21:40

import datetime
import logging
import re
import time
import warnings
import os
from threading import Thread
import copy
from functools import reduce
import urllib.parse

import bs4
import pytz
from xml.etree import ElementTree
from humanfriendly import format_size
from anytree.importer import DictImporter
from anytree.exporter import DictExporter

# 过滤这类警告
warnings.filterwarnings("ignore", category=bs4.MarkupResemblesLocatorWarning, module='bs4')

import win32com.client

if win32com.client.gencache.is_readonly:
    win32com.client.gencache.is_readonly = False
    win32com.client.gencache.Rebuild()

from pyxllib.prog.newbie import SingletonForEveryClass
from pyxllib.prog.pupil import Timeout
from pyxllib.prog.specialist import tqdm
from pyxllib.file.specialist import XlPath, get_etag
from pyxllib.algo.treelib import Node, XlNode
from pyxllib.text.xmllib import BeautifulSoup, XlBs4Tag

"""
参考了onepy的实现，做了重构。OnePy：Provides pythonic wrappers around OneNote COM interfaces
"""

namespace = "{http://schemas.microsoft.com/office/onenote/2013/onenote}"

# 还未绑定父结点的游离page node，用于进度条子线程
_free_page_nodes = []

# 缓存文件地址
CACHE_DIR = XlPath.tempdir() / 'OneNote/SearchCache'
os.makedirs(CACHE_DIR, exist_ok=True)

# 页面解析结果的缓存，用于解析加速
_page_parsed_cache = {}
_page_parsed_cache_file = CACHE_DIR / 'page_parsed_cache_file.pkl'
if _page_parsed_cache_file.is_file():
    _page_parsed_cache = _page_parsed_cache_file.read_pkl()

# 用来读取、保存序列化的node数据
importer = DictImporter()
exporter = DictExporter()


class ONProcess(metaclass=SingletonForEveryClass):
    """ onenote 底层win32的接口

    详细功能可以查官方文档：
        Application interface (OneNote) | Microsoft Docs:
        https://docs.microsoft.com/en-us/office/client-developer/onenote/application-interface-onenote
    """

    def __init__(self, timeout=30):
        """ onenote的win32接口方法是驼峰命名，这个ONProcess做了一层功能封装
        而且估计理论上对所有可以获得的接口都做了封装了

        :param timeout: 读取单个页面的时候，限制用时，单位：秒
            本来只想限制5秒，但发现会有一些页面特别长，需要多一些时间~
            再后来发现还有更慢的页面，半分钟的都有，就再改成30秒了
        """
        # TODO 这里需要针对不同的OneNote版本做自动化兼容，不要让用户填版本
        #   因为让用户填版本，会存在多个实例化对象，使用get_xml会有各种问题
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

    def navigate_to(self, page_id, object_id='', new_window=False):
        try:
            self.process.NavigateTo(page_id, object_id, new_window)
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

    def get_hyperlink_to_object(self, page_id, object_id=""):
        """

        :param str page_id:
            The OneNote ID for the notebook, section group, section, or page for which you want a hyperlink.
        :param str object_id: The OneNote ID for the object within the page for which you want a hyperlink.
        """
        try:
            return self.process.GetHyperlinkToObject(page_id, object_id)
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

    def init_node(self):
        node = Node(self.name, _category=type(self).__name__)
        # if type(self).__name__ != 'Page':
        node._html_content = f'<a href="onenote/linkid?id={self.id}" target="_blank">{self.name}</a>'
        return node

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

    def get_page_num(self):
        return sum([x.get_page_num() for x in self._children])

    def get_search_tree(self, *, print_mode=True, use_node_cache=True, reparse=False):
        """ 获得检索树的根节点

        :param print_mode: 输出所包含页面解析进度
        :param use_node_cache: 检查xml是否有更新
            默认可以不检查，如果一个widget有_node结点可以直接使用
        :param reparse: 使用旧的解析过xml的持久化文件数据
            这个好处是速度快，坏处是如果解析代码功能有更新，这样提取到的是错误的旧的解析数据
            这个缓存文件的处理规则，为了一些细节优化，也稍复杂~~
        """
        global _page_parsed_cache

        # 1 进度条工具，开一个子线程，每秒监控页面解析进度
        def timer_progress():
            """ 为了实现这个进度条，还有点技术含量呢，用了子线程和一些trick """
            global _free_page_nodes

            def dfs(x):
                cnt = 0

                _category = getattr(x, '_category', '')
                if _category == 'Page':
                    cnt += 1
                elif _category == '':
                    return cnt

                for y in x.children:
                    cnt += dfs(y)
                return cnt

            total = self.get_page_num()
            _tqdm = tqdm(desc='OneNote解析页面', disable=False, total=total)
            while not stop_flag:
                num = dfs(self._node)  # 只读取，不修改self._node，而且进度条稍微有些错没关系，所以不加锁
                _free_page_nodes = list(filter(lambda x: x.root != self._node, _free_page_nodes))
                num += len(_free_page_nodes)
                _tqdm.n = num
                _tqdm.refresh()

                time.sleep(1)

            # 最后统计一轮
            num = dfs(self._node)
            _free_page_nodes = []
            _tqdm.total = _tqdm.n = num
            _tqdm.refresh()

            # print(f'一共{total}个页面，实际解析出{num}个页面')
            # 实际解析成功的页面数
            return num

        # 2 主线程解析页面的过程，子线程每秒钟展示一次进度情况
        if reparse and type(self).__name__ == 'OneNote':  # 如果是OneNote层面reparse，直接重置整个缓存文件
            _page_parsed_cache = {}
        cache_num = len(_page_parsed_cache)

        if print_mode:
            stop_flag = False
            timer_thread = Thread(target=timer_progress)
            timer_thread.start()
            root = self._search(use_node_cache=use_node_cache, reparse=reparse)
            stop_flag = True  # 使用进程里的共享变量，进行主线程和子线程之间的通信
            timer_thread.join()
        else:
            root = self._search(use_node_cache=use_node_cache, reparse=reparse)

        def save_cache():
            if cache_num != len(_page_parsed_cache) or reparse:
                # 如果前后数量不一致，表示有更新内容，重新写入一份缓存文件。或者明确使用了reparse了也要保存。
                _page_parsed_cache_file.write_pkl(_page_parsed_cache)

        savefile_thread = Thread(target=save_cache)
        savefile_thread.start()  # 开子线程去保存文件，和后面的拷贝副本并行处理

        # 3 拷贝副本，并等待保存文件的子线程也执行完
        root.parent = None
        savefile_thread.join()
        return root

    def search(self, pattern, child_depth=0, *,
               edits=None, reparse=False,
               print_mode=False, return_mode='text',
               padding_mode=0, dedent=1, href_mode=1):
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
        :param href_mode: 超链接的模式
            0，不设超链接。text模式下强制设为0。
            1，静态链接（需要调用onenote生成，要多花一点点时间）
            2，动态链接，在开启server的时候使用才有意义
        """
        # 1 按照规则检索内容
        if isinstance(pattern, str):  # 文本关键词检索
            def check_node(node):
                # 纯文本部分
                if pattern in node.name:
                    return True
                # 如果纯文本找不到，也会在富文本格式里尝试匹配
                html_text = getattr(node, '_html_content', '')
                if html_text and pattern in html_text:
                    return True
                return False

        elif isinstance(pattern, re.Pattern):  # 正则检索
            def check_node(node):
                if pattern.search(node.name):
                    return True
                html_text = getattr(node, '_html_content', '')
                if html_text and pattern.search(html_text):
                    return True
                return False
        else:  # 自定义检索
            check_node = pattern

        # 2 更新索引并获得解析树
        start_time = time.time()
        edits = edits or []
        edits = [list(name.split('/')) for name in edits]
        # 更新易变数据的检索树
        for path in edits:
            node = self(path)
            print(node.abspath_name)
            node.get_search_tree(print_mode=False, use_node_cache=False, reparse=reparse)
        root = self.get_search_tree(print_mode=print_mode, use_node_cache=True, reparse=reparse)
        elapsed1 = time.time() - start_time

        # 3 检索内容
        start_time = time.time()
        n = XlNode.sign_node(root, check_node, flag_name='_flag', child_depth=child_depth, reset_flag=True)
        elapsed2 = time.time() - start_time

        # 4 html情况下的渲染算法
        def node_to_html(x, depth):
            import html

            if depth < 0:
                return '<br/>'

            content = f'{getattr(x, "_html_content", html.escape(x.name))}'
            if not hasattr(x, '_category'):
                pass
            elif x._category == 'OE':
                if href_mode == 1:
                    url = onenote.get_hyperlink_to_object(x._page_id, x._object_id)
                    content += f'&nbsp;<a href="{url}">go</a>'
                elif href_mode == 2:
                    url = f"onenote/linkid?id={x._page_id}&object_id={x._object_id}"
                    content += f'&nbsp;<a href="{url}" target="_blank">go</a>'
            elif x._category == 'Page':
                color = ['#009900', '#00b300', '#00cc00'][x._page_level - 1]
                content = f'<font color="{color}">{x.name}</font>'
                if href_mode == 1:
                    url = onenote.get_hyperlink_to_object(x._page_id)
                    content = f'<a href="{url}">{content}</a>'
                elif href_mode == 2:
                    url = f"onenote/linkid?id={x._page_id}"
                    content = f'<a href="{url}" target="_blank">{content}</a>'

            content = content.replace('\n', ' ')

            if padding_mode == 1:
                div = f'<div>{"&nbsp;" * depth * 4}{content}</div>'
            else:
                div = f'<div style="padding-left:{depth * 2 + 1}em;text-indent:-1em">{content}</div>'

            return div

        # 5 展示内容
        texts = [f'更新数据：{elapsed1:.2f}秒，内容检索：{elapsed2:.2f}秒，匹配条目数：{n}']
        if return_mode == 'text':
            body = XlNode.render(root, filter_=lambda x: getattr(x, '_flag', 0), dedent=dedent)
            texts[0] += f'，内容大小：{format_size(len(body.encode()), binary=True)}\n'
            texts.append(body)
            return '\n'.join(texts)
        elif return_mode == 'html':
            body = XlNode.render_html(root, node_to_html, filter_=lambda x: getattr(x, '_flag', 0), dedent=dedent)
            texts[0] += f'，内容大小：{format_size(len(body.encode()), binary=True)}<br/>'
            texts.append(body)
            return '<br/>'.join(texts)
        else:
            raise ValueError

    def __call__(self, item=None):
        """ 通过路径形式定位 """
        if isinstance(item, str):
            if '/' in item:
                return reduce(lambda x, name: x[name], [self] + list(item.split('/')))
            else:
                return self[item]
        else:
            return self


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
        self.id = self._children[0].id  # 以第1个笔记本作为OneNote的id，方便一些功能统一设计
        self._node = self.init_node()

    def get_href(self):
        # OneNote软件本身没有跳转，这里默认设置跳转到第1个笔记本
        return self._children[0].get_href()

    def init_node(self):
        node = Node(self.name, _category='OneNote', _html_content=f'<font color="purple">OneNote</font>')
        return node

    def get_page_content(self, page_id):
        page_content_xml = ElementTree.fromstring(super(OneNote, self).get_page_content(page_id))
        return PageContent(page_content_xml)

    def update_page_content(self, page_changes_xml_in):
        """
        :param page_changes_xml_in:
            xml，可以是原始的xml文本
                onenote.update_page_content(page.get_xml().replace('曹一众', '曹二众'))
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

    def _search(self, *, use_node_cache=True, reparse=False):
        if not self._node.is_leaf and use_node_cache:
            return self._node
        else:
            self._node = self.init_node()

        for x in self._children:
            cur_node = x._search(use_node_cache=use_node_cache, reparse=reparse)
            cur_node.parent = self._node

        return self._node


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
        self.path = xml.get("path")  # page没有这个属性，但也不会报错的
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

        self._node = self.init_node()

    def get_href(self):
        return 'onenote:' + self.path[:-1]

    def init_node(self):
        node = Node(self.name, _category='Notebook',
                    _html_content=f'<a href="{self.get_href()}"><font color="red">《{self.name}》</font></a>')
        return node

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

    def _search(self, *, use_node_cache=True, reparse=False):
        if not self._node.is_leaf and use_node_cache:
            return self._node
        else:
            self._node = self.init_node()

        for x in self._children:
            cur_node = x._search(use_node_cache=use_node_cache, reparse=reparse)
            cur_node.parent = self._node

        return self._node


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

        self._node = self.init_node()

    def get_href(self):
        return 'onenote:' + self.path[:-1]

    def init_node(self):
        node = Node(self.name, _category='SectionGroup',
                    _html_content=f'<a href="{self.get_href()}"><font color="#e68a00">〖{self.name}〗</font></a>')
        return node

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

    def _search(self, *, use_node_cache=True, reparse=False):
        if not self._node.is_leaf and use_node_cache:
            return self._node
        else:
            self._node = self.init_node()

        for x in self._children:
            cur_node = x._search(use_node_cache=use_node_cache, reparse=reparse)
            cur_node.parent = self._node

        # 这里多线程效率几乎没差，就不开了
        # def run_unit(x):
        #     cur_node = x._search(reset=reset)
        #     cur_node.parent = self._node
        # mtqdm(run_unit, self._children, max_workers=2, disable=True)

        return self._node


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

        self._node = self.init_node()

    def get_href(self):
        return 'onenote:' + self.path

    def init_node(self):
        node = Node(self.name, _category='Section',
                    _html_content=f'<a href="{self.get_href()}"><font color="#b38600">〈{self.name}〉</font></a>')
        return node

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

    def _search(self, *, use_node_cache=True, reparse=False):
        if not self._node.is_leaf and use_node_cache:
            return self._node
        else:
            self._node = self.init_node()

        page_lv1, page_lv2 = self._node, self._node

        for x in self._children:
            # print(x.name)
            cur_page = x._search(use_node_cache=use_node_cache, reparse=reparse)
            if x.page_level == '1':
                cur_page.parent = self._node
                page_lv2 = page_lv1 = cur_page
            elif x.page_level == '2':
                cur_page.parent = page_lv1
                page_lv2 = cur_page
            else:
                cur_page.parent = page_lv2

        return self._node


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

        self._node = self.init_node()  # 供全文检索的树形结点

    def get_href(self, strict=False):
        """ 按照OneNote的链接规则，如果同一个分区下，有重名Page，只会查找到第1个
        如果要精确指向页面，需要使用get_hyperlink_to_object方法
        """
        if strict:
            return onenote.get_hyperlink_to_object(self.id)
        else:
            return self.parent.get_href() + f'#{self.name}'

    def init_node(self):
        node = Node(self.name, _category='Page', _page_id=self.id, _page_level=int(self.page_level))
        return node

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

    def get_xml(self, page_info=0):
        """ 获得页面的xml内容 """
        # 1 有缓存的文件直接读取
        prefix = f'{self.id}_{page_info}_'
        file = CACHE_DIR / (prefix + self.last_modified_time.replace(':', '') + f'.xml')

        if file.is_file():
            return file.read_text()

        # 2 否则没有缓存，或者文件不是最新，则使用onenote的接口获得文件内容
        try:
            res = super(OneNote, onenote).get_page_content(self.id, page_info)
        except TimeoutError as e:
            e.args = [e.args[0] + f'\n\t{self.abspath_name} 页面获取失败，请检查可能包含的office公式并删除。' \
                                  f'并且看下您的OneNote可能无响应了，请重启OneNote。']
            raise e

        if res is None:
            logging.warning(f'{self.abspath_name} 未成功提取页面内容')
        else:
            # 删除旧时间点的缓存文件，存储新的缓存文件
            for f in CACHE_DIR.glob(f'{prefix}*.xml'):
                f.delete()
            file.write_text(res)

        return res

    def browser_xml(self, page_info=0):
        from pyxllib.debug.specialist import browser, XlPath
        xml = self.get_xml(page_info)
        browser(xml, file=XlPath.tempfile('.xml'))

    def parse_xml(self, root=None, *, page_info=0, reparse=False):
        """ 获得本Page页面的树形结构，返回一个Node根节点

        :param reparse: 默认直接使用缓存里的解析结果，如果设置了reparse则强制重新解析

        因为层次结构有多种不同的表现形式
            层次结构A：Outline 文本框
            层次结构B：h1、h2、h3 等标题结构
            层次结构C：正文里的缩进层级
        所以实现这里parent的引用，算法会稍复杂。需要动态更新，实时返回新的parent。
            比如在遍历tag.contents的时候，因为[层次结构C]的原因，会出现 parent = dfs_parse_node(y, parent) 的较奇怪的写法
            在parse_oe中，parent的层次实现规则，也会较复杂，有些trick
        """

        # 0 函数

        def _parse_xml(root):
            soup = BeautifulSoup(xml or '', 'xml')
            # self.browser_xml()  # 可以用这个查原始的xml内容

            style_defs = {}

            # trick: outline_cnt不仅用来标记是否有多个Outline需要设中介结点。也记录了当前Outline的编号。
            outline_cnt = 1 if len(soup.find_all('Outline')) > 1 else 0

            cur_node: XlBs4Tag = soup
            parent = root
            while cur_node:
                x = cur_node
                if isinstance(x, bs4.element.Tag):
                    # if分支后注释的数字，是实际逻辑结构上先后遇到的顺序，但为了效率，按照出现频率重排序了
                    if x.name == 'OE':  # 3
                        parent = OETag.parse2tree(x, parent, style_defs)
                        cur_node = cur_node.next_preorder_node(False)
                        continue
                    elif x.name == 'Outline':  # 2
                        # 处理层次结构A
                        if outline_cnt:
                            if outline_cnt == 1:
                                parent = Node(f'Outline{outline_cnt}', parent)
                            else:
                                pp = XlNode.find_parent(parent, re.compile('^Outline'))
                                parent = Node(f'Outline{outline_cnt}', pp.parent if pp else parent)
                            outline_cnt += 1
                    elif x.name == 'QuickStyleDef':  # 1
                        style_defs[x['index']] = x['name']
                        cur_node = cur_node.next_preorder_node(False)
                        continue

                cur_node = XlBs4Tag.next_preorder_node(cur_node)

            return root

        # 1 在一次程序执行中，相同的xml内容解析出的树也是一样的，可以做个缓存
        xml = self.get_xml(page_info=page_info)
        etag = get_etag(xml)

        if root is None:
            root = Node('root')

        if etag in _page_parsed_cache and not reparse:
            root.children = importer.import_(_page_parsed_cache[etag]).children
            return root

        # 2 否则进入正常解析流程
        root = _parse_xml(root)
        _page_parsed_cache[etag] = exporter.export(root)

        return root

    def get_page_num(self):
        return 1

    def _search(self, *, use_node_cache=True, reparse=False):
        """ 先生成所有结点
        """
        if not self._node.is_leaf and use_node_cache:
            # 首先要有孩子结点，不是叶子结点，才表示可能解析过的node，此时开启use_node_cache的话，则不重复解析
            return self._node
        else:
            self._node = self.init_node()

        self.parse_xml(self._node, reparse=reparse)
        _free_page_nodes.append(self._node)
        return self._node


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

    def get_xml(self):
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


class OETag(bs4.element.Tag):

    def get_text2(self):
        """ 这是给bs4.Tag准备的功能接口 """
        if y := self.find('T', recursive=False):
            t1 = BeautifulSoup(y.text, 'lxml').text
            t2 = y.text
        elif y := self.find('Table', recursive=False):
            # 先Columns标记了一共m列，每列的宽度
            # 然后每一行是一个Row，里面有m个Cell
            t1 = '[Table]'
            t2 = t1
        elif y := self.find('Image', recursive=False):
            t1 = '[Image]'
            t2 = t1
        else:
            t1 = ''
            t2 = ''
        return t1, t2

    def parse2tree(self, parent, style_defs):
        """ 从Tag结点，解析出 anytree 格式的结点树

        :param Node parent: anytree的node父结点
            会将当前Tag解析的内容，转存，挂到parent.children下
        :param style_defs: 前文解析到的样式表
        """

        # 1 获得3个主要属性
        style_name = style_defs.get(self.get('quickStyleIndex', ''), '')
        pure_text, html_text = OETag.get_text2(self)  # 文本内容
        m = self.find('OEChildren', recursive=False)  # 文本性质子结点

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

        setattr(cur_node, '_category', 'OE')
        setattr(cur_node, '_page_id', XlNode.find_parent(self, 'Page')['ID'])  # noqa find_parent适用于bs4.element.Tag
        setattr(cur_node, '_object_id', self['objectID'])

        # 3 表格、图片等特殊结构增设层级
        if pure_text.startswith('[Table]'):
            for z in self.find_all('T'):
                Node(BeautifulSoup(z.text, 'lxml').text, cur_node, _html_content=z.text)
        elif pure_text.startswith('[Image]'):
            y = self.find('Image', recursive=False)
            for z in y.get('alt', '').splitlines():
                Node(z, cur_node)

        # 4 处理层次结构C
        if m:
            for y in m.find_all('OE', recursive=False):
                OETag.parse2tree(y, cur_node, style_defs)

        return parent


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


def start_server(root=None, edits=None, *, port=80, reparse=False):
    """ 在本地开启一个onenote搜索服务

    :param str root: 要检索的onenote根目录，未设置时默认初始化所有OneNote笔记
        所有的路径一律用反斜杠/隔开
        注意这样写是会降低灵活性的
            一方面本来是可以使用数字直接引用下标的，这种模式下出现的数字会直接判定为是页面名称
            另一方面路径中本来就可能存在/
        但这些情况是小概率事件，一般遇到有问题的子目录，改到大目录里定位就好
            实在不行，读者可以自己复制这个函数自行扩展
    :param str|list[str] edits: 每次检索前要强制更新检索树的目录
        str, 用英文逗号隔开多个条目
        list[str]里的str，统一只要写在root下的相对路径

    >> search_server('核心', ['2022ch4'])
    >> search_server('共享/陈坤泽', ['杂项', 'CF', '吃土乡/大家的幻想乡'])
    """
    from flask import Flask, request

    # 1 初始化检索数据
    parent = onenote(root)
    parent.get_search_tree(print_mode=True, reparse=reparse)

    if isinstance(edits, str):
        # 如果输入是字符串，可能使用命令行启动的，需要转义为list
        edits = edits.split(',')  # 英文逗号隔开多个参数

    # 2 开服务接口
    app = Flask(__name__)

    @app.route('/search/onenote', methods=['GET'])
    def search_onenote():
        def get_args(key, default=None):
            return request.args.get(key, default)

        pattern = get_args('pattern')
        if pattern:
            # 解析功能细节
            res = parent.search(pattern, edits=edits,
                                child_depth=int(get_args('child_depth', 0)),
                                return_mode=get_args('return_mode', 'html'),
                                padding_mode=int(get_args('padding_mode', 0)),
                                print_mode=get_args('print_mode', True),
                                href_mode=int(get_args('href_mode', 2)),  # 默认使用动态链接
                                dedent=int(get_args('dedent', 1)))
        else:
            ref_url = 'http://localhost/search/onenote?pattern=test'
            return f'请输入检索内容，例如 <a href={ref_url}>{ref_url}</a>'
        return res

    @app.route('/search/onenote/linkid', methods=['GET'])
    def linkid():
        """ 通过id进行目标跳转 """
        page_id = request.args.get('id', None)
        object_id = request.args.get('object_id', '')
        onenote.navigate_to(page_id, object_id)
        # 返回一个直接自关闭的页面内容
        return '<script type="text/javascript">window.close();</script>'

    # 子线程无法调用win32com生成的onenote资源，如果要使用linkid功能，只能留一个主线程处理
    app.run(host='0.0.0.0', port=port, threaded=False)


if __name__ == '__main__':
    import fire

    fire.Fire()
