#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/01/01

import os
import html
import re
import time
from enum import Enum
import pprint
import json

import requests
import urllib.parse
import pandas as pd

from fastcore.basics import GetAttr
from loguru import logger

from pyxllib.prog.newbie import SingletonForEveryInitArgs
from pyxllib.prog.pupil import run_once
from pyxllib.text.pupil import UrlQueryBuilder, shorten
from pyxllib.text.nestenv import NestEnv
from pyxllib.text.xmllib import BeautifulSoup, XlBs4Tag
from pyxllib.file.specialist import refinepath, XlPath
from pyxllib.cv.xlcvlib import xlcv


def __1_语雀主api():
    pass


def update_yuque_doc_by_dp(doc_url):
    """ 语雀更新文档的api，并不会触发pc上客户端对应内容的更新
    除非打开浏览器编辑更新一下，这里可以用dp模拟自动进行这一操作

    注意新建文档添加到目录的api是可以立即在客户端生效的，只是更新不行
    """
    from DrissionPage import Chromium, ChromiumOptions
    from pyxllib.ext.drissionlib import dp_check_quit

    # 1 打开浏览器：注意必须事先手动登录过账号
    co = ChromiumOptions()
    # co.headless()  # 无头模式效果不一定稳，一般还是不能开
    # co.set_argument('--window-size', '100,100')  # 修改窗口尺寸也不行
    # 就只有最正常的模式才能马上触发更新
    browser = Chromium(co)

    # 2 确认url
    if not doc_url.startswith('https://www.yuque.com/'):
        doc_url = 'https://www.yuque.com/' + doc_url
    tab = browser.new_tab(f'{doc_url}/edit')

    # 3 操作更新文档
    tab('t:button@@data-ne-type=clearFormat')  # 通过查找一个元素确定文档已经加载成功
    time.sleep(1)
    tab.actions.type(' ')
    time.sleep(1)

    tab.actions.key_down('BACKSPACE')
    time.sleep(2)
    tab.actions.key_up('BACKSPACE')

    time.sleep(5)
    tab('t:button@@text():更新').click(by_js=True)
    time.sleep(20)
    tab.close()

    # 4 退出
    dp_check_quit()


class Yuque(metaclass=SingletonForEveryInitArgs):
    """
    https://www.yuque.com/yuque/developer/openapi
    语雀请求限制：每小时最多 5000 次请求，每秒最多 100 次请求
    """

    def __init__(self, token=None, user_id=None):
        self.base_url = "https://www.yuque.com/api/v2"
        self.headers = {
            "X-Auth-Token": token or os.getenv('YUQUE_TOKEN'),
            "Content-Type": "application/json"
        }
        self._user_id = os.getenv('YUQUE_USER_ID') or user_id

    def get_user(self):
        """ 获取用户信息

        # 获得的内容
         {'data':
            {'_serializer': 'v2.user',  # 数据序列化版本
              # 用户头像URL
              'avatar_url': 'https://cdn.nlark.com/yuque/0/2020/.../d5f8391e-2fdd-4f1b-8ea5-299be8fceecd.png',
              'books_count': 12,  # 知识库数量
              'created_at': '2018-11-16T07:26:27.000Z',  # 账户创建时间
              'description': '',  # 用户描述
              'followers_count': 54,  # 跟随者数量
              'following_count': 6,  # 关注者数量
              'id': 123456,  # 用户唯一标识
              'login': 'code4101',  # 用户登录名
              'name': '代号4101',  # 户昵称或姓名
              'public': 1,  # 用户公开状态，1为公开
              'public_books_count': 2,  # 公开的知识库数量
              'type': 'User',  # 数据类型，这里为'User'
              'updated_at': '2023-12-31T02:43:16.000Z',  # 信息最后更新时间
              'work_id': ''}}
        """
        url = f"{self.base_url}/user"
        resp = requests.get(url, headers=self.headers)
        return resp.json()

    @property
    def user_id(self):
        """ 很多接口需要用到用户ID，这里缓存一下 """
        if self._user_id is None:
            self._user_id = self.get_user()['data']['id']
        return self._user_id

    def __1_知识库操作(self):
        pass

    @run_once('id,str')  # todo 应该有更好的缓存机制，目前这样的实现，需要重启程序才会刷新
    def get_repos(self, return_mode=0):
        """ 获取某个用户的知识库列表

        :param int|str return_mode: 返回模式
            0（默认），返回原始json结构
            df，df结构
            nickname2id，获取知识库 "namespace和昵称"到ID的映射
        """
        if return_mode == 0:
            url = f"{self.base_url}/users/{self.user_id}/repos"
            resp = requests.get(url, headers=self.headers)
            return resp.json()
        elif return_mode == 'df':
            data = self.get_repos()
            columns = ['id', 'name', 'items_count', 'namespace']

            ls = []
            for d in data['data']:
                ls.append([d[col] for col in columns])

            df = pd.DataFrame(ls, columns=columns)
            return df
        elif return_mode == 'nickname2id':  # namespace、name到id的映射（注意这里不考虑）
            data = self.get_repos()
            names2id = {d['name']: d['id'] for d in data['data']}
            # 例如："日志"知识库的namespace是journal，然后 journal -> 24363220
            namespace2id = {d['namespace'].split('/')[-1]: d['id'] for d in data['data']}
            names2id.update(namespace2id)
            return names2id
        else:
            raise ValueError(f'不支持的return_mode={return_mode}')

    def get_repo_id(self, repo_id):
        """ repo_id支持输入"昵称"来获得实际id
        """
        if isinstance(repo_id, str) and not re.match(r'\d+$', repo_id):
            repo_id = self.get_repos('nickname2id')[repo_id]
        return repo_id

    def get_repo_docs(self, repo_id, *, offset=0, limit=100, return_mode=0):
        """ 获取知识库的文档列表

        :param repo_id: 知识库的ID或Namespace（如"日志"是我改成的"journal"）
        :param int offset: 偏移多少篇文章后再取
        :param int limit: 展示多少篇文章，默认100篇
        :param int|str return_mode: 返回模式
            私人文档：https://www.yuque.com/code4101/journal/ztvg5qh5m3ga7gh7?inner=ubc5753c5
            0（默认），返回原始json结构
            -1（df），df结构
        :return: 文档列表
            底层接口获得的数据默认是按照创建时间排序的，但是我这里会重新按照更新时间重排序
        """
        repo_id = self.get_repo_id(repo_id)
        if return_mode == 0:
            uqb = UrlQueryBuilder()
            uqb.add_param('offset', offset)
            uqb.add_param('limit', limit)
            url = uqb.build_url(f"{self.base_url}/repos/{repo_id}/docs")
            logger.info(url)
            d = requests.get(url, headers=self.headers).json()
            return d
        elif return_mode in (-1, 'df'):
            data = self.get_repo_docs(repo_id, offset=offset, limit=limit)
            # 按照updated_at降序
            data['data'].sort(key=lambda x: x['updated_at'], reverse=True)
            columns = ['id', 'title', 'word_count', 'description', 'updated_at']

            ls = []
            for d in data['data']:
                ls.append([d.get(col) for col in columns])

            # ls.sort(key=lambda x: x[0])  # id一般就是创建顺序
            df = pd.DataFrame(ls, columns=columns)
            # df['updated_at']把'2024-08-07T06:13:10.000Z'转成datetime，并改到utf8时区
            df['updated_at'] = pd.to_datetime(df['updated_at']).dt.tz_convert('Asia/Shanghai')
            # 不显示时区
            df['updated_at'] = df['updated_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
            return df
        else:
            raise ValueError(f'不支持的return_mode={return_mode}')

    def __2_目录操作(self):
        pass

    def get_repo_toc(self, repo_id):
        """ 获取知识库目录 """
        repo_id = self.get_repo_id(repo_id)
        url = f"{self.base_url}/repos/{repo_id}/toc"
        resp = requests.get(url, headers=self.headers)
        d = resp.json()
        # logger.info(d)
        return d['data']

    def repo_toc_move(self, repo_id,
                      cur_doc=None, dst_doc=None,
                      *,
                      insert_ahead=False,
                      to_child=False):
        """ 知识库/目录/移动 模式

        :param dst|dict cur_doc: 移动哪个当前节点
            注意把cur移动到dst的时候，cur的子节点默认都会跟着移动

            这里输入的类型，一般是根据get_repo_toc，获得字典类型
            但如果输入是str，则默认使用的是url模式，需要这个函数里再主动做一次get_repo_toc
        :param dst|dict dst_doc: 移动到哪个目标节点
        :param insert_ahead: 默认插入在目标后面
            如果要插入到前面，可以启动这个参数
        :param to_child: 作为目标节点的子节点插入
        :return: 好像是返回新的整个目录
        """
        repo_id = self.get_repo_id(f'{repo_id}')

        # 输入为字符串时，默认使用的url进行定位。url只要包括'/'末尾最后的文档id即可，前缀可以省略
        if isinstance(cur_doc, str) or isinstance(dst_doc, str):
            toc = self.get_repo_toc(repo_id)  # 知识库的整个目录
            if isinstance(cur_doc, str):
                cur_doc = next((d for d in toc if d['url'] == cur_doc.split('/')[-1]))
            if isinstance(dst_doc, str):
                dst_doc = next((d for d in toc if d['url'] == dst_doc.split('/')[-1]))

        url = f"{self.base_url}/repos/{repo_id}/toc"
        cfg = {
            'node_uuid': cur_doc['uuid'],
            'target_uuid': dst_doc['uuid'],
            'action': 'prependNode' if insert_ahead else 'appendNode',
            'action_mode': 'child' if to_child else 'sibling',
        }
        resp = requests.put(url, json=cfg, headers=self.headers)
        return resp.json()

    def __3_文档操作(self):
        pass

    def ____1_获取文档(self):
        pass

    def _get_doc(self, repo_id, doc_id):
        """ 获取单篇文档的详细信息

        :param repo_id: 知识库的ID或Namespace
        :param doc_id: 文档的ID
        :return: 文档的详细信息
        """
        repo_id = self.get_repo_id(repo_id)
        url = f"{self.base_url}/repos/{repo_id}/docs/{doc_id}"
        resp = requests.get(url, headers=self.headers)
        return resp.json()

    def get_doc(self, doc_url, return_mode='md'):
        """ 从文档的URL中获取文档的详细信息

        :param doc_url: 文档的URL
            可以只输入最后知识库、文档部分的url标记
        :param return_mode: 返回模式，
            json, 为原始json结构
            md, 返回文档的主体md内容
            title_and_md, 返回文档的标题和md内容
        :return: 文档的详细信息
        """
        repo_slug, doc_slug = doc_url.split('/')[-2:]
        data = self._get_doc(repo_slug, doc_slug)['data']

        if return_mode == 'json':
            return data
        elif return_mode == 'md':
            return data['body']
        elif return_mode == 'title_and_md':
            return data["title"], data["body"]

    def export_markdown(self, url, output_dir=None, post_mode=1):
        """ 导出md格式文件

        :param str|list[str] url: 文档的URL
            可以导出单篇文档，也可以打包批量导出多篇文档的md文件
        :param output_dir: 导出目录
            单篇的文件名是按照文章标题自动生成的
            多篇的可以自己指定具体文件名
        :param post_mode: 后处理模式
            0，不做处理
            1，做适当的精简
        """
        # 1 获得内容
        data = self.get_doc(url, return_mode='json')
        body = data['body']
        if post_mode == 0:
            pass
        elif post_mode == 1:
            body = re.sub(r'<a\sname=".*?"></a>\n', '', body)

        # 2 写入文件
        if output_dir is not None:
            title2 = refinepath(data['title'])
            f = XlPath(output_dir) / f'{title2}.md'
            f.write_text(body)

        return body

    def ____2_新建文档(self):
        pass

    def _to_doc_data(self, doc_data, md_cvt=True):
        """ 将非规范文档内容统一转为标准字典格式

        :param str|dict doc_data: 文本内容（md，html，lake）或字典表达的数据
            可以直接传入要更新的新的（md | html | lake）内容，会自动转为 {'body': content}
                注意无论原始是body_html、body_lake，都是要上传到body字段的

            其他具体参数功能：
                slug可以调整url路径名
                title调整标题
                public参数调整公开性，0:私密, 1:公开, 2:企业内公开
                format设置导入的内容格式，markdown:Markdown 格式, html:HTML 标准格式, lake:语雀 Lake 格式

        :param bool md_cvt: 是否需要转换md格式
            默认的md文档格式直接放回语雀，是会丢失换行的，需要对代码块外的内容，执行\n替换
        """
        # 1 字符串转字典
        if isinstance(doc_data, str):
            doc_data = {'body': doc_data}

        # 2 判断文本内容格式
        if 'format' not in doc_data:
            m = re.match(r'<!doctype\s(\w+?)>', doc_data['body'], flags=re.IGNORECASE)
            if m:
                doc_data['format'] = m.group(1).lower()

        # 3 如果是md格式还要特殊处理
        if doc_data.get('format', 'markdown') == 'markdown' and md_cvt:
            ne = NestEnv(doc_data['body']).search(r'^```[^\n]*\n(.+?)\n^```',
                                                  flags=re.MULTILINE | re.DOTALL).invert()
            doc_data['body'] = ne.replace('\n', '\n\n')

        return doc_data

    def create_doc(self, repo_id, doc_data,
                   *,
                   dst_doc=None, insert_ahead=False, to_child=False,  # 设置文档所在位置
                   ):
        """ 创建单篇文档，并放到知识库下某指定为止

        示例用法：
        yuque.create_doc('周刊摘录',
                    {'title': '标题', 'body': '内容', 'slug': 'custom_slug/url'},
                    dst_doc='目标文档的slug/url', to_child=True)
        """
        # 1 创建文档
        repo_id = self.get_repo_id(repo_id)
        url = f"{self.base_url}/repos/{repo_id}/docs"

        doc_data = self._to_doc_data(doc_data)
        out_data = requests.post(url, json=doc_data, headers=self.headers).json()['data']
        doc_id = out_data['id']

        # 2 将文档添加到目录中
        url2 = f"{self.base_url}/repos/{repo_id}/toc"
        in_data2 = {
            "action": "prependNode",  # 默认添加到知识库目录最顶上的位置
            "action_mode": "child",
            "doc_id": doc_id,
        }
        # 返回的是知识库新的目录
        toc = requests.put(url2, json=in_data2, headers=self.headers).json()['data']

        # 3 如果有设置目录具体位置的需求
        if dst_doc:
            self.repo_toc_move(repo_id, toc[0], dst_doc, insert_ahead=insert_ahead, to_child=to_child)

        # 即使有dst_doc移动了目录位置，但这个新建文档本来就不带位置信息的，所以不用根据dst_doc再重新获得
        return out_data

    def ____3_更新文档(self):
        pass

    def _update_doc(self, repo_id, doc_id, doc_data):
        """ 更新单篇文档的详细信息

        :param repo_id: 知识库的ID或Namespace
        :param doc_id: 文档的ID
        :param dict doc_data: 包含文档更新内容的字典
        :return: 更新后的文档的详细信息
        """
        repo_id = self.get_repo_id(repo_id)
        url = f"{self.base_url}/repos/{repo_id}/docs/{doc_id}"
        resp = requests.put(url, json=doc_data, headers=self.headers)
        return resp.json()

    def update_doc(self, doc_url, doc_data, *, return_mode='json', use_dp=False):
        """ 从文档的URL中更新文档的详细信息

        :param doc_url: 文档的URL
        :param str|json doc_data: 包含文档更新内容的字典，详见_to_doc_data接口
        :param use_dp: 语雀的这个更新接口，虽然网页端可以实时刷新，但在PC端的软件并不会实时加载渲染。
            所以有需要的话，要开启这个参数，使用爬虫暴力更新下文档内容。
            使用此模式的时候，doc_url还需要至少有用户名的url地址，即类似'用户id/知识库id/文档id'
        :param str return_mode: 返回的是更新后文档的内容，不过好像有bug，这里返回的body存储的并不是md格式
            'md', 返回更新后文档的主体md内容
            'json', 为原始json结构

            不建议拿这个返回值，完全可以另外再重新取返回值，就是正常的md格式了
        :return: 更新后的文档的详细信息
        """
        # 1 基础配置
        repo_slug, doc_slug = doc_url.split('/')[-2:]
        doc_data = self._to_doc_data(doc_data)

        # 2 提交更新文档
        data = self._update_doc(repo_slug, doc_slug, doc_data)['data']

        # 3 使用爬虫在浏览器模拟编辑，触发客户端更新通知
        if use_dp:
            update_yuque_doc_by_dp(doc_url)

        # 4 拿到返回值
        if return_mode == 'md':
            return data['body']
        elif return_mode == 'json':
            return data

    def ____4_删除文档(self):
        pass

    def _delete_doc(self, repo_id, doc_id):
        """ 删除文档

        这个是真删除，不是从目录中移除的意思哦。
        虽然可以短期内从回收站找回来。

        :param repo_id: 知识库的ID或Namespace
        :param doc_id: 文档的ID
        """
        repo_id = self.get_repo_id(repo_id)
        url = f"{self.base_url}/repos/{repo_id}/docs/{doc_id}"
        resp = requests.delete(url, headers=self.headers)
        return resp.json()

    def delete_doc(self, doc_url):
        repo_slug, doc_slug = doc_url.split('/')[-2:]
        print(self._delete_doc(repo_slug, doc_slug))

    def __4_内容操作(self):
        pass

    def read_tables_from_doc(self, url, header=0):
        """ 从文档中读取表格

        :param url: 文档的URL
        :return: 表格列表
        """
        res = self.get_doc(url, return_mode='json')
        tables = pd.read_html(res['body_html'], header=header)
        return tables


def __2_语雀lake格式结构化解析基础工具():
    pass


# 语雀代码块支持的语言类型
class LakeCodeModes(Enum):
    PLAIN = 'plain'
    ABAP = 'abap'
    AGDA = 'agda'
    ARKTS = 'arkts'
    ASM = 'z80'
    BASH = 'bash'
    BASIC = 'basic'
    C = 'c'
    CSHARP = 'csharp'
    CPP = 'cpp'
    CLOJURE = 'clojure'
    CMAKE = 'cmake'
    CSS = 'css'
    CYPHER = 'cypher'
    DART = 'dart'
    DIFF = 'diff'
    DOCKFILE = 'dockerfile'
    ERLANG = 'erlang'
    FSHARP = 'fsharp'
    FORTRAN = 'fortran'
    GIT = 'git'
    GLSL = 'glsl'
    GO = 'go'
    GRAPHQL = 'graphql'
    GROOVY = 'groovy'
    HASKELL = 'haskell'
    HTML = 'html'
    HTTP = 'http'
    JAVA = 'java'
    JAVASCRIPT = 'javascript'
    JSON = 'json'
    JSX = 'jsx'
    JULIA = 'julia'
    KATEX = 'katex'
    KOTLIN = 'kotlin'
    LATEX = 'latex'
    LESS = 'less'
    LISP = 'lisp'
    LUA = 'lua'
    MAKEFILE = 'makefile'
    MARKDOWN = 'markdown'
    MATLAB = 'matlab'
    NGINX = 'nginx'
    OBJECTIVEC = 'objectivec'
    OCAML = 'ocaml'
    PASCAL = 'pascal'
    PERL = 'perl'
    PHP = 'php'
    PLSQL = 'plsql'
    POWERSHELL = 'powershell'
    PROPERTIES = 'properties'
    PROTOBUF = 'protobuf'
    PYTHON = 'python'
    R = 'r'
    RUBY = 'ruby'
    RUST = 'rust'
    SASS = 'sass'
    SCALA = 'scala'
    SCHEME = 'scheme'
    SHELL = 'shell'
    SOLIDITY = 'solidity'
    SQL = 'sql'
    STEX = 'stex'
    SWIFT = 'swift'
    SYSTEMVERILOG = 'systemverilog'
    TCL = 'tcl'
    TOML = 'toml'
    TSX = 'tsx'
    TYPESCRIPT = 'typescript'
    VBNET = 'vbnet'
    VELOCITY = 'velocity'
    VERILOG = 'verilog'
    VUE = 'vue'
    XML = 'xml'
    YAML = 'yaml'

    def __str__(self):
        return self.value


def encode_block_value(value_dict):
    """ 把当前value_dict的字典值转为html标签结构 """
    json_str = json.dumps(value_dict)
    url_encoded = urllib.parse.quote(json_str)
    final_str = "data:" + url_encoded
    return final_str


def decode_block_value(encoded_str):
    """ 解析value字段的值 """
    encoded_data = encoded_str[5:]
    decoded_str = urllib.parse.unquote(encoded_data)
    return json.loads(decoded_str)


class LakeBlockTypes(Enum):
    HEADING = 'heading'  # 标题
    P = 'p'  # 普通段落
    OL = 'ol'  # 有序列表
    IMAGE = 'image'  # 图片
    CODEBLOCK = 'codeblock'  # 代码块
    SUMMARY = 'summary'  # 折叠块标题
    COLLAPSE = 'collapse'  # 折叠块
    UNKNOWN = 'unknown'  # 未知类型
    STR = 'str'  # 文本。这个语雀本身没这个类型，是用bs解析才引出的。

    def __str__(self):
        return self.value


def check_block_type(tag):
    """ 这个分类会根据对语雀文档结构了解的逐渐深入和逐步细化 """
    match tag.tag_name:
        case 'p' if tag.find('card'):
            return LakeBlockTypes.IMAGE
        case 'p':
            return LakeBlockTypes.P
        case 'card':
            return LakeBlockTypes.CODEBLOCK
        case 'ol':
            return LakeBlockTypes.OL
        case s if re.match(r'h\d+$', s):
            return LakeBlockTypes.HEADING
        case 'details':
            return LakeBlockTypes.COLLAPSE
        case 'summary':
            return LakeBlockTypes.SUMMARY
        case 'NavigableString':
            return LakeBlockTypes.STR
        case _:
            raise LakeBlockTypes.UNKNOWN


def parse_blocks(childrens):
    """ 获得最原始的段落数组 """
    blocks = []
    for c in childrens:
        match check_block_type(c):
            case LakeBlockTypes.P:
                c = LakeP(c)
            case LakeBlockTypes.HEADING:
                c = LakeHeading(c)
            case LakeBlockTypes.IMAGE:
                c = LakeImage(c)
            case LakeBlockTypes.CODEBLOCK:
                c = LakeCodeBlock(c)
            case LakeBlockTypes.COLLAPSE:
                c = LakeCollapse(c)
            case _:
                c = LakeBlock(c)
        blocks.append(c)
    return blocks


def print_blocks(blocks, indent=0):
    """ 检查文档基本内容 """

    def myprint(t):
        print(indent * '\t' + t)

    for i, b in enumerate(blocks):
        match b.type:
            case LakeBlockTypes.STR:
                myprint(f'{i}、{b.type} {shorten(b.text, 200)}')
            case LakeBlockTypes.CODEBLOCK | LakeBlockTypes.IMAGE:  # 使用 | 匹配多个类型
                myprint(f'{i}、{b.type} {shorten(b.prettify(), 100)}\n')
                myprint(shorten(pprint.pformat(b.value_dict), 400))
            case LakeBlockTypes.COLLAPSE:
                myprint(f'{i}、{b.type} {shorten(b.prettify(), 100)}')
                print_blocks(b.get_blocks(), indent=indent + 1)
            case _:  # 默认情况
                myprint(f'{i}、{b.type} {shorten(b.prettify(), 200)}')
        print()
    return blocks


def __3_语雀结构化模组():
    pass


class LakeBlock(GetAttr, XlBs4Tag):
    """ 语雀文档中的基本块类型 """
    _default = 'tag'

    def __init__(self, tag):  # noqa
        self.tag = tag
        self.type = check_block_type(tag)

    def is_foldable(self):
        # 默认是不可折叠块
        return False

    def is_empty_line(self):
        # 默认不是空行
        return False

    def __part功能(self):
        pass

    def _get_part_number(self, text):
        m = re.match(r'^(\d+|\+)、', text)
        if m:
            t = m.group(1)
            return t if t != '+' else True

    def get_part_number(self):
        """ 【私人】我的日记普遍用 "1、" "2、" "+、" 的模式来区分内容分块
        该函数用来判断当前块是否是这种分块的开始

        :return: 找到的话会返回匹配的数字，'+'会返回True，如果没有返回None
            目前我的笔记编号一般不存在从0开始编号，也从来没用过负值。但有需要的话这里是可以扩展的。
        """
        # 默认return None
        return

    @classmethod
    def _set_part_number(cls, tag, number):
        """ 常规xml模式的设置编号 """
        # 1 如果没有span，在最前面加上span
        if not tag.find('span'):
            childrens = list(tag.children)
            if childrens:
                childrens[0].insert_html_before('<span></span>')
            else:
                tag.append_html('<span></span>')

        # 2 定位第一个span
        span = tag.span

        # 3 string删掉原有编号
        span.string = re.sub(r'^(\d+|\+)、', '', span.text)

        # 4 string加上新的编号前缀
        if number is not None:
            span.string = f'{number}、{span.text}'

        return span.string

    def set_part_number(self, number):
        """ 【私人】更新当前块的序号

        如果当前part没有编号，会增设编号标记
        如果number设为None，会删除编号标记
        """
        # 这个功能很特别，还是每类节点里单独实现更合理
        raise NotImplementedError

    def is_part_end(self):
        """ 【私人】判断当前块是否是分块的结束 """
        return False


class LakeHeading(LakeBlock):
    def __init__(self, tag):  # noqa
        super().__init__(tag)
        self.type = LakeBlockTypes.HEADING

    def is_part_end(self):
        return True


class LakeP(LakeBlock):
    """ 语雀代码块 """

    def __init__(self, tag):  # noqa
        super().__init__(tag)
        self.type = LakeBlockTypes.P

    def set_part_number(self, number):
        return self._set_part_number(self, number)

    def is_empty_line(self):
        return self.text.strip() == ''

    def is_part_end(self):
        # 出现新的编号块内容的时候，就是上一个part结束的时候
        return not self.get_part_number()

    def get_part_number(self):
        return self._get_part_number(self.text)


class LakeImage(LakeBlock):
    """ 语雀文档中的图片类型

    这个内容结构一般是 <p><card type="inline" name="image" value="data:..."></card></p>
    其中value是可以解析出一个字典，有详细数据信息的
    """

    def __init__(self, tag):  # noqa
        super().__init__(tag)
        self.type = LakeBlockTypes.IMAGE
        self._value_dict = None

    @property
    def value_dict(self):
        if self._value_dict is None:
            self._value_dict = decode_block_value(self.tag.card.attrs['value'])
        return self._value_dict

    def update_value_dict(self):
        """ 把当前value_dict的字典值更新回html标签 """
        self.tag.card.attrs['value'] = encode_block_value(self.value_dict)

    def __初始化(self):
        pass

    @classmethod
    def _init_from_src(cls, src):
        """ 传入一个图片url或base64的值，转为语雀图片节点 """
        soup = BeautifulSoup('<p><card type="inline" name="image" value=""></card></p>', 'lxml')
        soup.card.attrs['value'] = encode_block_value({'src': src})
        return cls(soup)

    @classmethod
    def from_url(cls, url):
        """ 通过一个url图片位置来初始化一张图片 """
        return cls._init_from_src(url)

    @classmethod
    def _reduce_img(cls, img, limit_size, suffix):
        # 1 初步读取压缩
        im = xlcv.read(img)
        if suffix is None:
            suffix = '.png' if min(xlcv.imsize(im)) > 50 else '.jpg'
        im = xlcv.reduce_filesize(im, limit_size, suffix)

        # 2 如果是过小的.jpg图片（最短边小余50像素），需要改用.png保存
        if suffix == '.jpg' and min(xlcv.imsize(im)) < 50:
            suffix = '.png'

        # 3
        return im, suffix

    @classmethod
    def from_local_image(cls, img, *, limit_size=0.6 * 1024 * 1024, suffix=None):
        """ 传入一个本地图片。本地图片必须转换为base64格式

        :param limit_size: 整个文档有1MB的限制，所以单张图片一般最大也只能给0.5MB的尺寸
        """
        im, suffix = cls._reduce_img(img, limit_size, suffix)
        buffer = xlcv.to_buffer(im, suffix, b64encode=True).decode('utf-8')
        return cls._init_from_src(f'data:image/{suffix[1:]};base64,{buffer}')


class XlLakeImage(LakeImage):
    @classmethod
    def from_local_image(cls, img, limit_size=1 * 1024 * 1024, suffix=None):
        """ 私人定制用功能
        由于语雀限制不能上传太大的base64图片数据，所以使用url的形式中转

        重载掉原本的from_local_image接口，这里需要把图片中转成我本地/download服务的文件接口

        :param limit_size: 哪怕通过url提供，中转的图片理论上也不应该搞得太大，最好要压缩到一定大小内
        :param suffix: 图片扩展名，使用None的时候会有一定自动判定机制
            图片尺寸较小就用png，否则用jpg
        """
        from pyxllib.file.specialist import GetEtag
        from pyxllib.prog.xlenv import xlhome_path, get_xl_hostname

        # 1 读取、压缩、保存图片
        im, suffix = cls._reduce_img(img, limit_size, suffix)
        etag = GetEtag.from_bytes(xlcv.to_buffer(im, suffix))
        file = xlhome_path(f'data/m2405network/download/images/{etag}{suffix}')
        xlcv.write(im, file)

        # 2 提供图片的url
        hn = get_xl_hostname()
        return cls.from_url(f'{os.getenv("MAIN_WEBSITE")}/{hn}/download/images/{file.name}')


class LakeCodeBlock(LakeBlock):
    """ 语雀代码块 """

    def __init__(self, tag):  # noqa
        super().__init__(tag)
        self.type = LakeBlockTypes.CODEBLOCK
        self._value_dict = None

    @property
    def value_dict(self):
        if self._value_dict is None:
            self._value_dict = decode_block_value(self.tag.attrs['value'])
        return self._value_dict

    def update_value_dict(self):
        """ 把当前value_dict的字典值更新回html标签 """
        self.tag.attrs['value'] = encode_block_value(self.value_dict)

    def is_foldable(self):
        return True

    def fold(self):
        """ 折叠代码块 """
        self.tag.value_dict['collapsed'] = True

    def unfold(self):
        """ 展开代码块 """
        self.tag.value_dict['collapsed'] = False

    def __part功能(self):
        pass

    def set_part_number(self, number):
        title = self.value_dict['name']
        title = re.sub(r'^(\d+|\+)、', '', title)
        if number is not None:
            title = f'{number}、{title}'
        self.value_dict['name'] = title
        self.update_value_dict()
        return title

    def is_part_end(self):
        return not self.get_part_number()

    def get_part_number(self):
        return self._get_part_number(self.value_dict.get('name'))


class LakeCollapse(LakeBlock):
    """ 语雀折叠块 """

    def __init__(self, tag):  # noqa
        super().__init__(tag)

    @classmethod
    def create(cls, summary='', blocks=None, *, open=True):
        """ 创建一个折叠块 """
        # 1 details
        summary = html.escape(summary)
        summary = f'<summary><span>{summary}</span></summary>'
        details = f'<details class="lake-collapse" open="{str(open).lower()}">{summary}</details>'

        # 2 tag
        details = BeautifulSoup(details, 'lxml').details
        for b in blocks:
            details.append_html(b.tag.prettify())
        return cls(details)

    def get_blocks(self):
        """ 获得最原始的段落数组 """
        return parse_blocks(self.tag.children)

    def print_blocks(self, indent=1):
        """ 检查文档基本内容 """
        blocks = self.get_blocks()
        print_blocks(blocks, indent=indent)

    def add_block(self, node):
        """ 在折叠块末尾添加一个新节点内容

        :param node: 要添加的节点内容，或者html文本
        """
        self.tag.append_html(node)

    def is_foldable(self):
        return True

    def fold(self):
        """ 折叠代码块 """
        self.tag.attrs['open'] = 'false'

    def unfold(self):
        """ 展开代码块 """
        self.tag.attrs['open'] = 'true'

    def __part功能(self):
        pass

    def set_part_number(self, number):
        return self._set_part_number(self.summary, number)

    def is_part_end(self):
        return not self.get_part_number()

    def get_part_number(self):
        return self._get_part_number(self.summary.text)


# GetAttr似乎必须放在前面，这样找不到的属性似乎是会优先使用GetAttr机制的，但后者又可以为IDE提供提示
class LakeDoc(GetAttr, XlBs4Tag):
    """ 语雀文档类型 """
    _default = 'soup'

    def __init__(self, soup):  # noqa，这个类初始化就是跟父类不同的
        # 原始完整的html文档内容
        self.soup: XlBs4Tag = soup
        self.type = 'doc'

    def __文档导入导出(self):
        pass

    @classmethod
    def from_url(cls, url, *, yuque=None):
        """ 输入语雀笔记的url
        """
        yuque = yuque or Yuque()
        data = yuque.get_doc(url, return_mode='json')
        doc = LakeDoc.from_html(data['body_lake'])
        return doc

    @classmethod
    def from_html(cls, lake_html_str='<body></body>'):
        """

        :param lake_html_str: 至少要有一个<body></body>结构。
            不过好在bs本身有很多兼容处理，基本只要输入任意正常内容，就会自动补上body结构的，我不需要手动做太多特判处理
        :return:
        """
        if not lake_html_str.startswith('<!doctype lake>'):
            lake_html_str = '<!doctype lake>' + lake_html_str
        soup = BeautifulSoup(lake_html_str, 'lxml')
        return cls(soup)

    def to_lake_str(self):
        """ 转换成语雀html格式的字符串 """
        content = self.soup.prettify().replace('\n', '')
        content = re.sub('^<!DOCTYPE lake>', '<!doctype lake>', content)
        content = re.sub(r'\s{2,}', '', content)
        return content

    def to_url(self, url, *, yuque=None, use_dp=False):
        """ 把文章内容更新到指定url位置
        """
        yuque = yuque or Yuque()
        yuque.update_doc(url, self.to_lake_str(), use_dp=use_dp)

    def __其他功能(self):
        pass

    def get_blocks(self):
        """ 获得最原始的段落数组 """
        return parse_blocks(self.soup.body.children)

    def print_blocks(self):
        """ 检查文档基本内容 """
        blocks = self.get_blocks()
        return print_blocks(blocks)

    def delete_lake_id(self):
        """ 删除文档中所有语雀标签的id标记 """
        for tag in self.soup.find_all(True):
            for name in ['data-lake-id', 'id']:
                if name in tag.attrs:
                    del tag[name]

    def add_block(self, node):
        """ 在正文末尾添加一个新节点内容

        :param node: 要添加的节点内容，或者html文本
        """
        self.soup.body.append_html(node)

    def fold_blocks(self):
        """ 把文档中的可折叠块全部折叠 """
        for b in self.get_blocks():
            if b.is_foldable():
                b.fold()

    def remove_empty_lines_between_collapses(self):
        """ 删除文档中可折叠块之间的全部空行 """
        blocks = self.get_blocks()
        collapse_indices = []
        for idx, b in enumerate(blocks):
            if b.is_foldable():
                collapse_indices.append(idx)

        to_remove = []

        # 检查每对相邻的可折叠块之间的块
        for i in range(len(collapse_indices) - 1):
            prev_idx = collapse_indices[i]
            current_idx = collapse_indices[i + 1]
            start = prev_idx + 1
            end = current_idx - 1

            if start > end:
                continue

            # 检查中间所有块是否均为空行
            all_empty = True
            for j in range(start, end + 1):
                block = blocks[j]
                if not block.is_empty_line():
                    all_empty = False
                    break
            if all_empty:
                to_remove.extend(range(start, end + 1))

        # 按逆序删除，避免索引变化
        for index in sorted(to_remove, reverse=True):
            blocks[index].tag.decompose()

    def remove_empty_lines_at_start(self):
        """ 删除文档开头的空行，最多保留一个 """
        blocks = self.get_blocks()
        if not blocks:
            return

        # 找到第一个非空行的索引
        first_non_empty = None
        for idx, b in enumerate(blocks):
            if not b.is_empty_line():
                first_non_empty = idx
                break

        # 如果没有非空行，直接返回
        if first_non_empty is None:
            return

        # 删除第一个非空行之前的所有空行
        for i in range(first_non_empty):
            if blocks[i].is_empty_line():
                blocks[i].tag.decompose()

    def __part系列功能(self):
        """ 偏个人向笔记风格的定制功能 """
        pass

    def reset_part_numbers(self):
        """ 重置文档中所有的序号 """
        blocks = self.get_blocks()
        cnt = 0
        for b in blocks:
            if b.type == LakeBlockTypes.HEADING:
                cnt = 0
            elif b.get_part_number() is not None:
                cnt += 1
                b.set_part_number(cnt)

    @classmethod
    def get_part_blocks(cls, blocks, start_idx):
        """ 获得指定part的全部blocks """
        # 1 先找到硬结尾
        part_blocks = [blocks[start_idx]]
        for b in blocks[start_idx + 1:]:
            if b.type == LakeBlockTypes.HEADING:
                break
            elif b.get_part_number() is not None:
                break
            part_blocks.append(b)

        # 2 再去掉软结尾，即末尾是空白的行都去掉
        while part_blocks:
            b = part_blocks[-1]
            if b.is_empty_line():
                part_blocks.pop()
            else:
                break

        return part_blocks

    def part_to_collapse(self):
        """ 把文档中的part内容转为折叠块 """
        blocks = self.get_blocks()

        i = 0
        while i < len(blocks):
            b = blocks[i]
            if b.get_part_number() is None:
                pass  # 没有编号的跳过，不是part的起始位置
            elif b.type == LakeBlockTypes.P:  # 只对段落类型的编号进行处理
                part_blocks = self.get_part_blocks(blocks, i)
                summary = part_blocks[0].text
                collapse = LakeCollapse.create(summary, part_blocks, open=False)
                # 以下.tag都不能省略
                b.tag.insert_html_before(collapse.tag)
                for b2 in part_blocks:
                    b2.tag.decompose()
                i += len(part_blocks) - 1
            i += 1


if __name__ == '__main__':
    pass
