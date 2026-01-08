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
from urllib.parse import urljoin

from pyxllib.prog.lazyimport import lazy_import

try:
    from loguru import logger
except ModuleNotFoundError:
    logger = lazy_import('from loguru import logger')

try:
    from fastcore.basics import GetAttr
except ModuleNotFoundError:
    GetAttr = lazy_import('from fastcore.basics import GetAttr')

try:
    import requests
except ModuleNotFoundError:
    requests = lazy_import('requests')

# urllib是Python标准库，不需要转换
import urllib.parse

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import('pandas')

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

    def __init__(self, *, token=None, user_id=None):
        self.base_url = "https://www.yuque.com/api/v2/"  # 注意末尾的/一定要输入
        self.token = token or os.getenv('YUQUE_TOKEN')
        assert self.token, "Yuque Token is required."

        # 缓存
        self._user = None
        self._user_id = user_id or os.getenv('YUQUE_USER_ID')
        self._groups = None

    def __1_基础操作(self):
        pass

    def _request(self, method, endpoint, **kwargs):
        """ 内部通用请求方法：
        1. 自动拼接 URL
        2. 自动处理 headers
        3. 自动解包 response['data']
        4. 统一错误处理
        """
        # 去掉 endpoint 开头的 /，防止 urljoin 覆盖 base_url 的 path
        endpoint = endpoint.lstrip('/')
        url = urljoin(self.base_url, endpoint)

        headers = {
            "X-Auth-Token": self.token,
            "Content-Type": "application/json",
            "User-Agent": "Python-Yuque-Client/1.0",
        }

        try:
            response = requests.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()  # 检查 4xx, 5xx 错误

            # 语雀 API 成功时通常返回 { "data": { ... } }
            # 这里直接把 'data' 剥离出来，让外部调用更爽
            res_json = response.json()
            if isinstance(res_json, dict) and 'data' in res_json:
                return res_json['data']
            return res_json

        except requests.exceptions.HTTPError as e:
            # 这里可以加更详细的日志或自定义异常
            print(f"Request Error: {e}")
            raise

    def _paginate(self, method, **kwargs):
        """ 通用翻页生成器 (核心逻辑)

        :param method: 要调用的获取单页数据的方法 (如 self.get_repos)
        :param kwargs: 传递给 method 的参数 (如 user_id=123, public=1)
        """
        offset = 0
        limit = kwargs.get('limit', 100)  # 默认100，如果在kwargs里指定了就用指定的

        while True:
            # 动态更新 offset
            kwargs['offset'] = offset

            # 调用传入的方法
            items = method(**kwargs)

            if not items:
                break

            # 逐个 yield 返回，这就变成了生成器
            yield from items

            # 如果取到的数量少于 limit，说明是最后一页了
            if len(items) < limit:
                break

            offset += limit

    def _parse_url_parts(self, text):
        """ [纯逻辑] 从 URL/Path 中提取 (Namespace, Slug)

        修复版逻辑：优先提取 user/repo/doc 三层结构
        """
        if not isinstance(text, str): return None, None

        # 1. 预处理：如果是 URL，先提取 path 部分
        if text.startswith('http'):
            # 简单去除 query 和 fragment
            clean = text.split('?')[0].split('#')[0]
            # 移除协议头
            clean = clean.replace('https://', '').replace('http://', '')
            # 移除域名 (找到第一个/)
            if '/' in clean:
                clean = clean[clean.find('/') + 1:]
        else:
            clean = text.split('?')[0].split('#')[0]

        clean = clean.strip('/')
        parts = clean.split('/')

        # 2. 提取逻辑
        if len(parts) >= 3:
            # 命中标准结构: namespace_part1 / namespace_part2 / doc_slug
            # 例如: code4101 / journal / architecture-design
            # 返回: ("code4101/journal", "architecture-design")
            return f"{parts[-3]}/{parts[-2]}", parts[-1]

        elif len(parts) == 2:
            # 命中短结构: repo_slug / doc_slug
            # 注意：这种结构在 resolve_repo_id 时可能会因为缺少用户名而产生歧义
            # 但作为 fallback 依然保留
            return parts[0], parts[1]

        return None, None

    def _is_valid_id(self, val):
        """ [纯逻辑] 判断是否为合法的数字 ID (int 或 纯数字字符串) """
        if isinstance(val, int):
            return True
        return isinstance(val, str) and val.isdigit()

    def _is_valid_uuid(self, val):
        """ [纯逻辑] 判断是否为合法的 UUID (TOC Node) """
        # 语雀 UUID 通常较长且无特殊符号
        return isinstance(val, str) and len(val) > 15 and '/' not in val

    def _is_namespace(self, val):
        """ [纯逻辑] 判断是否为 Namespace (User/Slug) """
        return isinstance(val, str) and '/' in val and not val.startswith('http')

    def __2_用户与搜索(self):
        pass

    @property
    def user(self):
        """ 获取用户信息

        {'_serializer': 'v2.user',  # 数据序列化版本
          # 用户头像URL
          'avatar_url': 'https://cdn.nlark.com/yuque/0/2020/.../d5f8391e-2fdd-4f1b-8ea5-299be8fceecd.png',
          'books_count': 12,  # 知识库数量
          'description': '',  # 用户描述
          'followers_count': 54,  # 跟随者数量
          'following_count': 6,  # 关注者数量
          'id': 123456,  # 用户唯一标识
          'login': 'code4101',  # 用户登录名
          'name': '代号4101',  # 户昵称或姓名
          'public': 1,  # 用户公开状态，1为公开
          'public_books_count': 2,  # 公开的知识库数量
          'type': 'User',  # 数据类型，这里为'User'
          'created_at': '2018-11-16T07:26:27.000Z',  # 账户创建时间
          'updated_at': '2023-12-31T02:43:16.000Z',  # 信息最后更新时间
          'work_id': ''}
        """
        if self._user is None:
            self._user = self._request('GET', 'user')
        return self._user

    @property
    def user_id(self):
        if self._user_id is None:
            self._user_id = self.user['id']
        return self._user_id

    def get_groups(self, user_id=None, role=None, offset=0):
        """ 获取用户加入的团队/组织列表

        :param user_id: 用户登录名或数字ID。不填则默认为当前登录用户(self.user_id)
        :param role: 筛选角色。0:管理员, 1:普通成员。不填则返回所有。
        :param offset: 偏移量，默认 0（每页固定返回100条）
        :return: list[dict] 团队列表
            [{'id': 1694077,
              'type': 'Group',
              'login': 'xlpr',
              'name': '厦门理工模式识别与图像理解重点实验室',
              'avatar_url': 'https://cdn.nlark.com/yuque/0/2020/png/209178/1593912283515-avatar/a8d20cb3-0c25-41c9-9d1b-72ead95b0a7f.png',
              'books_count': 7,
              'public_books_count': 2,
              'members_count': 37,
              'public': 1,
              'description': '',
              'created_at': '2020-07-05T01:24:46.000Z',
              'updated_at': '2025-12-11T03:25:55.000Z',
              '_serializer': 'v2.group'}
             ]
        """
        target_id = user_id or self.user_id
        params = {"offset": offset}
        if role is not None: params["role"] = role
        return self._request("GET", f"users/{target_id}/groups", params=params)

    def get_all_groups(self, user_id=None, role=None):
        """ 获取所有团队 (生成器) """
        return self._paginate(self.get_groups, user_id=user_id, role=role)

    @property
    def groups(self):
        if self._groups is None:
            self._groups = self.get_groups()
        return self._groups

    @property
    def group_id(self):
        # 默认获得第1组的id
        return self.groups[0]['id']

    @run_once('id,str')
    def find_group_by_name(self, name, user_id=None):
        """ (辅助) 根据名称查找团队
        注意：这里只在用户加入的团队列表中查找
        """
        # 使用 get_all_groups 自动翻页查找
        for group in self.get_all_groups(user_id=user_id):
            if group['name'] == name:
                return group
        return None

    def resolve_group_id(self, group_ident):
        """ 解析团队标识 -> 返回 ID (int) 或 Login (str)

        支持输入:
        1. 123456 (int/str) -> 直接返回 ID
        2. "xlpr" (Login)   -> 直接返回 Login (API支持)
        3. "实验室" (Name)  -> 触发查找 -> 返回 ID
        """
        # 1. Direct ID
        if self._is_valid_id(group_ident):
            return int(group_ident)

        s_ident = str(group_ident).strip()

        # 2. 判断是否为 Login (英文/数字/横杠/点)
        # 如果包含非 ASCII 字符 (如中文)，则肯定不是 Login，必须走查找
        is_login = all(ord(c) < 128 for c in s_ident)

        if is_login:
            # 即使看起来像 Login，也有可能是英文的 Name。
            # 但语雀 API 对 Login 很宽容，通常直接传 Login 即可。
            # 这里做一个简单策略：如果是纯英文，优先视为 Login 直接返回，
            # 除非你极其严格，否则不建议这就去发请求验证，浪费性能。
            return s_ident

        # 3. Lookup Name (中文名称必走这里)
        if group := self.find_group_by_name(s_ident):
            return group['id']

        # 4. 没找到
        raise ValueError(f"Group not found: {group_ident}")

    def get_group_members(self, group_id=None, role=None):
        """ 获取指定团队的成员列表

        :param group_id: 团队的 ID
        :param role: 筛选角色。0:管理员, 1:成员, 2:只读成员。不填则返回所有。
        :return: list[dict] 成员列表
            [{'id': 999999, 'group_id': 1234567, 'user_id': 123456, 'role': 0,
              'created_at': '2020-07-05T01:24:46.000Z',
              'updated_at': '2025-09-25T08:25:31.000Z',
              'user': '格式见self.user'}, ...]
        """
        params = {}
        group_id = group_id or self.group_id
        if role is not None: params["role"] = role
        return self._request("GET", f"groups/{group_id}/users", params=params)

    def update_group_member(self, group_id, user_id, role):
        """ 更新或添加团队成员角色

        :param group_id: 团队 ID
        :param user_id: 目标用户的 login 或 ID
        :param role: 目标角色。0:管理员, 1:成员, 2:只读成员
        :return: dict 操作结果
        """
        endpoint = f"groups/{group_id}/users/{user_id}"
        payload = {"role": role}
        return self._request("PUT", endpoint, json=payload)

    def remove_group_member(self, group_id, user_id):
        """ 移除团队成员

        :param group_id: 团队 ID
        :param user_id: 目标用户的 login 或 ID
        :return: dict 操作结果
        """
        endpoint = f"groups/{group_id}/users/{user_id}"
        return self._request("DELETE", endpoint)

    def search(self, q, target_type="doc", scope=None, related=True):
        """ 全局或指定范围搜索

        :param q: (必填) 搜索关键词
        :param target_type: (必填) 搜索类型。'doc' (文档) 或 'repo' (知识库)。默认为 'doc'
        :param scope: 搜索范围路径。
            - 留空: 搜索所有范围
            - 示例: 'my-group' (搜团队内), 'my-group/wiki' (搜知识库内)
        :param bool related: 是否仅搜索与我相关的。默认 True
        :return: list[dict] 搜索结果列表
            [{'id': 181252454, 'type': 'doc', 'title': '语雀api',
              'summary': 'test 当要导出多篇文章的时候，每篇文章开头都列一个标准的基础描述？...',
              'url': '/code4101/st/otxh38ur5czslgyb', 'info': '代号4101 / 商汤科技',
              'target': {'id': 181252454, 'type': 'Doc', 'slug': 'otxh38ur5czslgyb', 'title': '语雀api',
                         'description': '官方文档：https://www.yuque.com/yuque/developer/openapi官方文档...',
                         'cover': 'https://cdn.nlark.com/yuque/0/2024/png/209178/1723014323667-46a51bbe-b39d-4261-9d7c-988366bd3753.png',
                         'user_id': 209178, 'book_id': 33323197, 'last_editor_id': 209178, 'public': 0, 'status': 1,
                         'likes_count': 0, 'read_count': 6, 'hits': 6, 'comments_count': 0, 'word_count': 165,
                         'created_at': '2024-08-07T05:52:09.000Z', 'updated_at': '2024-08-07T10:48:40.000Z',
                         'content_updated_at': '2024-08-07T10:48:40.000Z', 'published_at': '2024-08-07T10:48:40.000Z',
                         'first_published_at': '2024-08-07T05:53:23.194Z',
                         'book': {'id': 33323197, 'type': 'Book', 'slug': 'st', 'name': '商汤科技', 'user_id': 209178,
                                  'description': '', 'public': 0,
                                  'content_updated_at': '2025-12-10T13:39:51.000Z',
                                  'created_at': '2022-10-21T07:09:49.000Z',
                                  'updated_at': '2025-12-10T13:39:51.000Z',
                                  'user': {'...': '前面字段参考self.user', 'organization_id': 0, '_serializer': 'v2.user'},
                                  'namespace': 'code4101/st',
                                  '_serializer': 'v2.book'},
                         '_serializer': 'v2.doc'},
              '_serializer': 'v2.search_result'}, ...]
        """
        assert target_type in ["doc", "repo"], "target_type must be 'doc' or 'repo'"
        params = {
            "q": q,
            "type": target_type,
            "related": "true" if related else "false"
        }
        if scope:
            params["scope"] = scope
        return self._request("GET", "search", params=params)

    def __3_知识库管理(self):
        pass

    @run_once('id,str')
    def get_repos(self, user_id=None, group_id=None, offset=0, limit=100):
        """ 获取知识库列表

        逻辑说明：
        1. 如果指定了 group_id，则获取该团队下的知识库。
        2. 如果没指定 group_id，但指定了 user_id，则获取该用户下的知识库。
        3. 如果两者都没指定，默认获取当前登录用户(self.user_id)的知识库。

        :param user_id: 用户 ID 或 login
        :param group_id: 团队 ID 或 login
        :param offset: 偏移量（默认0）
        :param limit: 每页数量（默认100，最大100）
        :return: list[dict]
            示例：个人知识库
            [{'id': 24363220, 'type': 'Book', 'slug': 'journal', 'name': '日志', 'user_id': 209178,
              'description': '',
              'creator_id': 209178, 'public': 0, 'items_count': 703, 'likes_count': 0, 'watches_count': 1,
              'content_updated_at': '2025-12-10T19:00:59.943Z', 'created_at': '2022-01-03T02:24:26.000Z',
              'updated_at': '2025-12-10T19:01:00.000Z',
              'user': {'...': '前置字段见self.user', 'organization_id': 0, '_serializer': 'v2.user'},
              'namespace': 'code4101/journal', '_serializer': 'v2.book'}, ...]

            示例：团队知识库
            [{'id': 1621213, 'type': 'Book', 'slug': 'pyxllib', 'name': 'pyxllib', 'user_id': 1694077,
              'description': '公开的通用功能、工具库',
              'creator_id': 209178, 'public': 1, 'items_count': 157, 'likes_count': 0, 'watches_count': 5,
              'content_updated_at': '2025-12-11T03:35:24.980Z', 'created_at': '2020-08-15T07:59:13.000Z',
              'updated_at': '2025-12-11T03:35:25.000Z',
              'user': {'...': ..., 'organization_id': 0, '_serializer': 'v2.user'},
              'namespace': 'xlpr/pyxllib', '_serializer': 'v2.book'}, ...]
        """
        params = {"offset": offset, "limit": limit}

        # 优先判断是否获取团队知识库
        if group_id:
            endpoint = f"groups/{group_id}/repos"
        else:
            target_user = user_id or self.user_id
            endpoint = f"users/{target_user}/repos"

        return self._request("GET", endpoint, params=params)

    def get_all_repos(self, user_id=None, group_id=None):
        """ 获取所有知识库 (生成器) """
        return self._paginate(self.get_repos, user_id=user_id, group_id=group_id)

    def create_repo(self, name, slug, group_id=None, description="", public=0, type="Book"):
        """ 创建知识库

        :param name: (必填) 知识库名称
        :param slug: (必填) 知识库路径，需唯一
        :param group_id: 如果要创建在团队下，请填入团队ID；否则默认创建在当前用户下
        :param description: 简介
        :param public: 0:私密, 1:公开, 2:企业/团队内公开 (默认0)
        :param type: 'Book'(文档库) 或 'Design'(画板库), 默认 Book
            注：该参数原语雀api文档未标明，有待测试验证
        :return: dict (新创建的知识库详情)
        """
        payload = {
            "name": name,
            "slug": slug,
            "description": description,
            "public": public,
            "type": type
        }

        if group_id:
            endpoint = f"groups/{group_id}/repos"
        else:
            endpoint = f"users/{self.user_id}/repos"

        return self._request("POST", endpoint, json=payload)

    def get_repo(self, book_id_or_namespace):
        """ 获取知识库详情

        :param book_id_or_namespace:
            可以传数字 ID (推荐): 123456
            也可以传路径字符串: 'user_login/book_slug' 或 'group_login/book_slug'
        :return: dict 知识库详情
            {'id': 24363220, 'type': 'Book', 'slug': 'journal', 'name': '日志', 'user_id': 209178, 'description': '',
             'toc_yml': '- type: META\n  count: 700\n  display_level: 1\n  tail_type: UPDATED_AT\n  base_version_id: 643303947\n  published: true\n  max_level: 5\n  last_updated_at: 2025-12-08T10:14:24.857Z\n  version_id: 643304020\n- type: DOC\n  title: 卷一 开辟鸿蒙~2008.7\n  uuid: s0Qe3nbBpQbp4nVw\n  url: bdvp1k\n  prev_uuid: \'\'\n  sibling_uuid: IoWlb7U8TUyihUYA\n  child_uuid: sXaktev2EegzA1GA\n  parent_uuid: \'\'\n  doc_id: 89991304\n  level: 0\n  id: 89991304\n  open_window: 0\n  visible: 1\n- type: DOC\n  title: 初中\n ...',
             'creator_id': 209178, 'public': 0, 'items_count': 703, 'likes_count': 0, 'watches_count': 1,
             'content_updated_at': '2025-12-10T19:00:59.943Z', 'created_at': '2022-01-03T02:24:26.000Z',
             'updated_at': '2025-12-10T19:01:00.000Z',
             'user': {'...': ..., 'organization_id': 0, '_serializer': 'v2.user'},
             'namespace': 'code4101/journal',
             '_serializer': 'v2.book_detail'}
        """
        # 如果传入的是路径 (包含 /)，API处理路径参数通常不需要 /repos/ 前缀的特殊处理
        # 但语雀文档指出：GET /repos/{book_id} 或 /repos/{namespace}
        return self._request("GET", f"repos/{str(book_id_or_namespace)}")

    def update_repo(self, book_id, name=None, slug=None, public=None, description=None, toc_markdown=None):
        """ 更新知识库信息

        注意：此接口功能强大，甚至可以重置目录结构。

        :param book_id: 知识库 ID (必须是数字 ID，或者 namespace)
        :param name: 新名称
        :param slug: 新路径
        :param public: 公开性
        :param description: 简介
        :param toc_markdown: (高级) 传入一段 Markdown 列表文本，直接重置整个知识库的目录结构！
               例如: "- [Page A](slug-a)\n  - [Page B](slug-b)"
        :return: dict 更新后的详情
        """
        payload = {}
        if name is not None: payload['name'] = name
        if slug is not None: payload['slug'] = slug
        if public is not None: payload['public'] = public
        if description is not None: payload['description'] = description
        if toc_markdown is not None: payload['toc'] = toc_markdown

        if not payload:
            print("Warning: update_repo called with no fields to update.")
            return {}

        return self._request("PUT", f"repos/{book_id}", json=payload)

    def delete_repo(self, book_id):
        """ 删除知识库 (慎用)

        :param book_id: 知识库 ID
        """
        return self._request("DELETE", f"repos/{book_id}")

    @run_once('id,str')
    def find_repo_by_name(self, name, group_id=None):
        """ 根据名称查找知识库 ID (自动全量搜索) """
        # 这里使用 get_all_repos，它是一个生成器
        # 循环时它会在后台自动一页页请求，找到就停止，不浪费资源
        for repo in self.get_all_repos(group_id=group_id):
            if repo['name'] == name:
                return repo
        return None

    @run_once('id,str')
    def resolve_repo_id(self, repo_identity):
        """ 将 知识库标识 解析为 API 可用的 ID 或 Namespace

        支持输入:
        1. 123456 (int/str) -> 直接返回 (ID)
        2. "user/slug"      -> 直接返回 (Namespace)
        3. "项目周报"       -> 触发查找 -> 返回 ID
        """
        # 1. 如果是数字 ID，直接返回
        if self._is_valid_id(repo_identity):
            return int(repo_identity)

        # 2. 如果是 Namespace (user/slug)，API 原生支持，直接返回
        # 这是一个重要的优化：避免了 API 调用
        if self._is_namespace(repo_identity):
            return repo_identity

        # 3. 剩下的一律视为“名称”，需要查找
        # 优先查个人，再查团队 (依赖 find_repo_by_name 的缓存)
        if repo := self.find_repo_by_name(repo_identity):
            return repo['id']

        # 4. (可选) 遍历群组查找，视需求开启
        for group in self.groups:
            if repo := self.find_repo_by_name(repo_identity, group['id']):
                return repo['id']

        raise ValueError(f"Repository not found: {repo_identity}")

    def __4_文档操作(self):
        pass

    def ____1_获取文档(self):
        pass

    def resolve_doc_ref(self, repo_identity, doc_identity=None):
        """ 解析文档上下文，返回 (RepoID, DocSlug/ID)

        注意不要把函数名改成resolve_doc_id，这个函数返回值是元组，改名id会引起误解
        这里是ref是引用位置的含义

        支持两种调用方式:
        1. resolve_doc_ref(repo, doc_slug)
        2. resolve_doc_ref(full_doc_url) -> 此时 doc_identity 为 None
        """
        repo_target = repo_identity
        doc_target = doc_identity

        # 场景 A: 传入了完整 URL (如 "https://.../repo/doc")
        if doc_target is None and isinstance(repo_target, str) and '/' in repo_target:
            # 尝试拆解 URL
            extracted_repo, extracted_doc = self._parse_url_parts(repo_target)
            if extracted_repo and extracted_doc:
                repo_target = extracted_repo
                doc_target = extracted_doc

        # 1. 确保获取到了 Repo ID
        real_repo_id = self.resolve_repo_id(repo_target)

        # 2. 确保获取到了 Doc 标识
        assert doc_target, "Could not resolve document identity."

        # 注意：API 中 GET /docs/{id_or_slug} 同时支持 ID 和 Slug，
        # 所以这里不需要像 Repo 那样强制转 ID，直接返回即可。
        return real_repo_id, doc_target

    def get_docs(self, repo_identity, offset=0, limit=100):
        """ 获取单页文档列表

        [{'id': 248359251, 'type': 'Doc', 'slug': 'aak9uvncacb9rzi7',
          'title': '大类拆分与架构设计技术分析：以 Python SDK 封装为例',
          'description': '版本：1.0背景：...',
          'cover': '', 'user_id': 209178, 'book_id': 1621213, 'last_editor_id': 209178, 'public': 1, 'status': 1,
          'likes_count': 0, 'read_count': 0, 'comments_count': 0, 'word_count': 1702,
          'created_at': '2025-12-11T03:25:54.000Z', 'updated_at': '2025-12-11T03:35:26.000Z',
          'content_updated_at': '2025-12-11T03:35:25.000Z', 'published_at': '2025-12-11T03:35:25.000Z',
          'first_published_at': '2025-12-11T03:26:55.735Z',
          'user': {'...': ..., 'organization_id': 0, '_serializer': 'v2.user'},
          'last_editor': {'id': 209178, 'type': 'User', 'login': 'code4101', 'name': '陈坤泽(代号4101)',
                          'avatar_url': 'https://cdn.nlark.com/yuque/0/2020/png/209178/1590664181441-avatar/d5f8391e-2fdd-4f1b-8ea5-299be8fceecd.png',
                          'followers_count': 56, 'following_count': 6, 'public': 1, 'description': '',
                          'created_at': '2018-11-16T07:26:27.000Z', 'updated_at': '2025-12-11T04:11:12.000Z',
                          'work_id': '', 'organization_id': 0, '_serializer': 'v2.user'}, 'hits': 0,
          '_serializer': 'v2.doc'}, ...]
        """
        # 1. 智能解析 ID
        real_id = self.resolve_repo_id(repo_identity)
        assert real_id, f"Repository not found: {repo_identity}"

        # 2. 请求
        endpoint = f"repos/{real_id}/docs"
        params = {"offset": offset, "limit": limit}
        return self._request("GET", endpoint, params=params)

    def get_all_docs(self, repo_identity):
        """ 获取该知识库下所有文档 (生成器) """
        return self._paginate(self.get_docs, repo_identity=repo_identity)

    def get_doc(self, repo_ident, doc_ident=None):
        r_id, d_slug = self.resolve_doc_ref(repo_ident, doc_ident)
        return self._request("GET", f"repos/{r_id}/docs/{d_slug}")

    def get_doc_content(self, url_or_keys, return_mode='md'):
        """ 获取文档内容的高级封装

        :param url_or_keys: 文档 URL 字符串，或者 (repo, doc) 元组
        :param return_mode:
            - 'json': 返回完整 API 响应字典
            - 'md': 仅返回 Markdown 正文
            - 'title_and_md': 返回 (title, md_body) 元组
        """
        # 1. 获取数据
        if isinstance(url_or_keys, (list, tuple)):
            data = self.get_doc(url_or_keys[0], url_or_keys[1])
        else:
            data = self.get_doc(url_or_keys)

        # 2. 根据模式返回
        if return_mode == 'json':
            return data
        elif return_mode == 'md':
            return data.get('body', '')
        elif return_mode == 'title_and_md':
            return data.get('title', 'Untitled'), data.get('body', '')
        else:
            raise ValueError(f"Unknown return_mode: {return_mode}")

    def export_markdown(self, url, output_dir=None, post_mode=1):
        """ 导出 Markdown 文件

        :param url: 文档 URL
        :param output_dir: 导出目录 (str 或 Path 对象)
        :param post_mode: 1=清洗锚点标签, 0=原文
        :return: body (str)
        """
        # 1. 获取内容
        title, body = self.get_doc_content(url, return_mode='title_and_md')

        # 2. 后处理 (Post Processing)
        if post_mode == 1:
            # 移除语雀特有的空锚点 <a name="ABCD"></a>
            body = re.sub(r'<a\s+name=".*?"></a>\n?', '', body)
            # 也可以在这里移除 <br /> 等其他非标准 md 标记

        # 3. 写入文件
        if output_dir:
            out_path = XlPath(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # 处理文件名中的非法字符
            safe_title = refinepath(title)
            file_path = out_path / f"{safe_title}.md"

            # 使用 utf-8 写入
            file_path.write_text(body, encoding='utf-8')
            # print(f"Exported: {file_path}")

        return body

    def ____2_新建文档(self):
        pass

    def _prepare_doc_payload(self, doc_data, md_auto_fix=True):
        """ 预处理文档数据 (内部 Helper)

        :param str|dict doc_data: 文本内容（md，html，lake）或字典表达的数据
            可以直接传入要更新的新的（md | html | lake）内容，会自动转为 {'body': content}
                注意无论原始是body_html、body_lake，都是要上传到body字段的

            其他具体参数功能：
                slug可以调整url路径名
                title调整标题
                public参数调整公开性，0:私密, 1:公开, 2:企业内公开
                format设置导入的内容格式，markdown:Markdown 格式, html:HTML 标准格式, lake:语雀 Lake 格式
        :param md_auto_fix: 是否自动修复 Markdown 换行问题 (语雀老编辑器特性)
            默认的md文档格式直接放回语雀，是会丢失换行的，需要对代码块外的内容，执行\n替换
        :return: dict 准备好发送给 API 的 payload
        """
        # 1. 统一转为字典
        if isinstance(doc_data, str):
            payload = {'body': doc_data}
        else:
            payload = doc_data.copy()

        # 2. 只有当存在 body 时，才尝试自动推断格式
        #    修复 Bug: 避免只更新属性(如 slug)时，错误地重置了 format 和 body
        if 'body' in payload and 'format' not in payload:
            body = payload.get('body', '')
            if isinstance(body, str) and '<!doctype lake>' in body.lower():
                payload['format'] = 'lake'
            elif isinstance(body, str) and '<!doctype html>' in body.lower():
                payload['format'] = 'html'
            else:
                payload['format'] = 'markdown'

        # 3. Markdown 换行修复 (仅当格式确认为 markdown 且存在 body 时)
        if md_auto_fix and payload.get('format') == 'markdown' and 'body' in payload:
            body = payload.get('body', '')
            if body:
                ne = NestEnv(payload['body']).search(r'^```[^\n]*\n(.+?)\n^```',
                                                     flags=re.MULTILINE | re.DOTALL).invert()
                payload['body'] = ne.replace('\n', '\n\n')

        # 4. 删除默认标题设置
        #    修复 Bug: 避免 update 时覆盖原有标题。
        #    Create 接口如果缺标题，语雀 API 会自动处理为“无标题”，无需 Python 侧干预。
        # if 'title' not in payload:
        #     payload['title'] = '无标题'

        return payload

    def _to_doc_data(self, doc_data, md_cvt=True):
        """ 将非规范文档内容统一转为标准字典格式

        :param bool md_cvt: 是否需要转换md格式
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

    def create_doc(self, repo_identity, doc_data,
                   *,
                   dst_doc=None, insert_ahead=False, to_child=False,  # 位置控制
                   md_fix=True):  # 格式控制
        """ 创建文档并自动挂载到指定位置

        :param repo_identity: 知识库标识
        :param doc_data: 文档内容 (str 或 dict)
        :param dst_doc: 挂载的目标参照文档 (Slug/URL/ID)，不填则默认挂载到根目录
        :param insert_ahead: 插到目标前面
        :param to_child: 插到目标内部
        :param md_fix: 是否自动修复 Markdown 换行
        :return: dict 新建文档的详情

        yq.create_doc('周刊摘录',
            {'title': '标题', 'body': '内容', 'slug': 'custom_slug/url'},
            dst_doc='目标文档的slug/url', to_child=True)
        """
        # 1. 解析与预处理
        repo_id = self.resolve_repo_id(repo_identity)
        if not repo_id:
            raise ValueError(f"Repo not found: {repo_identity}")

        payload = self._prepare_doc_payload(doc_data, md_auto_fix=md_fix)

        # 2. 【API】创建文档实体 (此时文档处于游离状态)
        res_doc = self._request("POST", f"repos/{repo_id}/docs", json=payload)
        new_doc_id = res_doc['id']

        # 3. 解析目标位置 UUID
        # 只有当指定了 dst_doc 时，resolve_node_uuid 才会内部调用 get_toc 去查找
        # 否则 target_uuid 为空字符串，无需额外的 TOC 读取请求 -> 效率优化
        target_uuid = ""
        if dst_doc:
            target_uuid = self.resolve_node_uuid(repo_id, dst_doc)
            # 容错：如果用户指定了目标但没找到，target_uuid 为 ""，
            # 此时逻辑会自动降级为 "挂载到根目录"，保证文档至少能显示出来。
            if not target_uuid:
                print(f"[Warn] Target doc '{dst_doc}' not found. Appending to root.")

        # 4. 【API】挂载目录
        toc_payload = {
            "action": "prependNode" if insert_ahead else "appendNode",
            "action_mode": "child" if to_child else "sibling",
            "target_uuid": target_uuid,
            "type": "DOC",  # 明确类型为文档
            "doc_ids": [new_doc_id]  # 新建/挂载操作核心参数
        }

        self.update_toc(repo_id, toc_payload)

        return res_doc

    def ____3_更新文档(self):
        pass

    def _update_doc(self, repo_id, doc_id, doc_data):
        """ 更新单篇文档的详细信息

        :param repo_id: 知识库的ID或Namespace
        :param doc_id: 文档的ID
        :param dict doc_data: 包含文档更新内容的字典
        :return: 更新后的文档的详细信息
        """
        repo_id = self.resolve_repo_id(repo_id)
        url = f"{self.base_url}/repos/{repo_id}/docs/{doc_id}"
        resp = requests.put(url, json=doc_data, headers=self.headers)
        return resp.json()

    def update_doc(self, repo_identity, doc_data, doc_slug=None,
                   *,
                   md_fix=True,
                   use_dp=False,
                   return_mode='json'
                   ):
        """ 更新文档内容
        【安全增强版】自动回填未修改的字段，防止内容丢失。

        :param repo_identity: 文档 URL 或 知识库标识。
            - 推荐直接传入完整 URL (如 "https://yuque.com/user/repo/slug")
            - 也可以传 RepoID/Name (需配合 doc_slug)
            - 注意：开启 use_dp 时，此处最好传入 URL，否则传给爬虫的可能只是 ID
        :param doc_data: 更新内容 (字符串 或 字典)
        :param doc_slug: 文档 Slug/ID (如果 repo_identity 是 URL 则此项忽略)
        :param md_fix: 是否自动修复 Markdown 换行 (默认 True)
        :param use_dp: 语雀的这个更新接口，虽然网页端可以实时刷新，但在PC端的软件并不会实时加载渲染。
            所以有需要的话，要开启这个参数，使用爬虫暴力更新下文档内容。
            使用此模式的时候，doc_url还需要至少有用户名的url地址，即类似'用户id/知识库id/文档id'
        :param return_mode:
            - 'json': 返回 API 原始响应 (推荐)
            - 'md': 返回 body 字段 (注意：API 刚更新完返回的 body 有时可能是旧的或 HTML)

        修改笔记url的方法：  但是注意使用update_doc都会对原本的空格排版情况产生变化
        yq.update_doc('https://www.yuque.com/code4101/journal/w240415-', {'slug': 'w240415'})
        """
        # 1. 解析身份 (获取 API 所需的 ID 和 Slug)
        # 无论输入是 URL 还是 ID 组合，这里都能拿到准确的 repo_id 和 doc_slug
        repo_id, real_doc_slug = self.resolve_doc_ref(repo_identity, doc_slug)

        # 2. 【核心修改】防止内容丢失：如果没传 body，先去把原文档的 body 取回来填上
        if 'body' not in doc_data:
            current_doc = self.get_doc(repo_id, real_doc_slug)
            doc_data['body'] = current_doc.get('body', '')
            md_fix = False  # 这种情况强制关闭二次转换

        # 3. 预处理数据 (转字典 + 自动补全格式 + 修复MD换行)
        payload = self._prepare_doc_payload(doc_data, md_auto_fix=md_fix)

        # 4. 【API】提交更新
        endpoint = f"repos/{repo_id}/docs/{real_doc_slug}"
        res_data = self._request("PUT", endpoint, json=payload)

        # 5. (可选) 爬虫强制刷新
        # 直接复用你已有的 update_yuque_doc_by_dp 函数
        if use_dp:
            # 假设 repo_identity 此时就是 url。
            # 如果调用者传的是 ID，这里转成字符串传进去，具体由你的爬虫函数去处理或报错
            update_yuque_doc_by_dp(str(repo_identity))

        # 6. 返回值处理 (兼容旧代码习惯)
        if return_mode == 'md':
            return res_data.get('body', '')

        # 默认 return_mode == 'json'
        return res_data

    def ____4_删除文档(self):
        pass

    def delete_doc(self, repo_identity, doc_slug=None):
        """ 删除文档 (逻辑删除)

        注意：API 执行的是逻辑删除，文档会进入语雀的回收站，可以恢复。

        :param repo_identity: 文档 URL 或 知识库标识 (ID/Namespace)
        :param doc_slug: 文档 Slug/ID (如果 repo_identity 是 URL 则此项忽略)
        :return: dict (被删除文档的详情数据)
        """
        # 1. 统一解析身份
        # 无论传入的是 "https://..." 还是 (123, "slug")，都能拿到准确的 ID
        repo_id, real_doc_slug = self.resolve_doc_ref(repo_identity, doc_slug)

        # 2. 执行请求
        # DELETE /repos/{book_id}/docs/{id}
        endpoint = f"repos/{repo_id}/docs/{real_doc_slug}"

        # _request 会自动处理 headers, base_url 和 异常检查
        return self._request("DELETE", endpoint)

    def __5_目录管理(self):
        pass

    def get_toc(self, repo_identity):
        """ 获取知识库目录树 """
        real_id = self.resolve_repo_id(repo_identity)
        return self._request("GET", f"repos/{real_id}/toc")

    def resolve_node_uuid(self, repo_id, node_ident, toc_list=None):
        """ 解析目录节点 -> 返回 UUID (str)

        :param node_ident: UUID / Node对象 / 文档URL / 文档Slug / **节点标题**
        """
        if not node_ident: return ""

        # 1. Direct Object
        if isinstance(node_ident, dict): return node_ident.get('uuid', '')

        s_ident = str(node_ident).strip()

        # 2. Direct UUID
        if self._is_valid_uuid(s_ident): return s_ident

        # 3. Lookup in TOC
        # 尝试提取 Slug (为了支持传入文档 URL 来定位节点)
        # 注意：如果 s_ident 只是普通标题 (如 "TCP/IP")，_parse_url_parts 可能会误判，
        # 所以下面匹配时，我们会同时对比 'search_key' (解析后的) 和 's_ident' (原始输入)
        _, derived_slug = self._parse_url_parts(s_ident)
        search_key = derived_slug if derived_slug else s_ident

        if toc_list is None:
            toc_list = self.get_toc(repo_id)

        for node in toc_list:
            node_url = node.get('url')
            # doc_id 可能是 int, None. 统一转字符串，None 转为空串避免匹配到字符串"None"
            node_doc_id = str(node.get('doc_id') or '')
            node_title = node.get('title')

            # 匹配逻辑优先级：
            # 1. 精确匹配 Slug (URL)
            # 2. 精确匹配 DocID
            # 3. 精确匹配 标题 (原始输入) -> 解决 "TCP/IP" 被误切分的问题
            # 4. 精确匹配 标题 (解析后) -> 解决标题本身就是纯英文 Slug 的情况
            if (node_url == search_key) or \
                    (node_doc_id == search_key) or \
                    (node_title == s_ident) or \
                    (node_title == search_key):
                return node['uuid']

        return ""

    def update_toc(self, repo_identity, action_payload):
        """ (底层核心) 更新目录结构的通用接口

        :param action_payload: 构造好的动作参数 (dict)
        """
        repo_id = self.resolve_repo_id(repo_identity)
        endpoint = f"repos/{repo_id}/toc"
        # 这里的 PUT 返回的是新的目录结构列表
        return self._request("PUT", endpoint, json=action_payload)

    def move_node(self, repo_identity, node_identity, target_identity="",
                  *, insert_ahead=False, to_child=False):
        """ 移动目录节点 (原 repo_toc_move 的升级版)

        :param repo_identity: 知识库标识 (ID/名称/Namespace)
        :param node_identity: 要移动的节点 (UUID/文档URL/文档Slug/文档ID)
        :param target_identity: 目标参照节点 (同上，留空则代表根节点)
        :param insert_ahead: 插到目标前面 (True) 还是后面 (False, 默认)
        :param to_child: 作为目标子节点 (True) 还是同级节点 (False, 默认)
        :return: list[dict] 更新后的目录结构
        """
        # 1. 解析知识库 ID
        repo_id = self.resolve_repo_id(repo_identity)

        # 2. 预取 TOC (一次请求，供后续两次查找使用，效率最高)
        toc_list = self.get_toc(repo_id)

        # 3. 解析 UUID
        node_uuid = self.resolve_node_uuid(repo_id, node_identity, toc_list)
        target_uuid = self.resolve_node_uuid(repo_id, target_identity, toc_list)

        # 校验：源节点必须存在
        if not node_uuid:
            raise ValueError(f"Source node not found: {node_identity}")

        # 4. 构造动作参数
        # 语雀 API 逻辑：
        # - action: appendNode (后/内尾), prependNode (前/内头)
        # - action_mode: sibling (同级), child (子级)
        payload = {
            "action": "prependNode" if insert_ahead else "appendNode",
            "action_mode": "child" if to_child else "sibling",
            "target_uuid": target_uuid,
            "node_uuid": node_uuid  # 移动操作核心参数
        }

        return self.update_toc(repo_id, payload)

    def __6_数据统计(self):
        pass

    def get_statistics_summary(self, group_identity):
        """ 获取团队汇总统计数据 """
        # 解析：中文名 -> ID/Login
        real_group_id = self.resolve_group_id(group_identity)
        return self._request("GET", f"groups/{real_group_id}/statistics")

    def get_statistics_docs(self, group_identity, offset=0, limit=100, **kwargs):
        """ 获取文档统计列表 """
        real_group_id = self.resolve_group_id(group_identity)
        endpoint = f"groups/{real_group_id}/statistics/docs"
        params = {"offset": offset, "limit": limit}
        params.update(kwargs)
        return self._request("GET", endpoint, params=params)

    def get_all_statistics_docs(self, group_identity, **kwargs):
        """ 生成器：所有文档统计 """
        # 注意：这里传给 _paginate 的 kwargs 必须包含 group_identity
        # 但因为 _paginate 会把参数透传给 get_statistics_docs，
        # 所以这里不需要预先解析，让 get_statistics_docs 内部去解析即可。
        return self._paginate(self.get_statistics_docs, group_identity=group_identity, **kwargs)

    def get_statistics_books(self, group_identity, offset=0, limit=100, **kwargs):
        """ 获取知识库统计列表 """
        real_group_id = self.resolve_group_id(group_identity)
        endpoint = f"groups/{real_group_id}/statistics/books"
        params = {"offset": offset, "limit": limit}
        params.update(kwargs)
        return self._request("GET", endpoint, params=params)

    def get_all_statistics_books(self, group_identity, **kwargs):
        return self._paginate(self.get_statistics_books, group_identity=group_identity, **kwargs)

    def get_statistics_members(self, group_identity, offset=0, limit=100, **kwargs):
        """ 获取成员贡献统计列表 """
        real_group_id = self.resolve_group_id(group_identity)
        endpoint = f"groups/{real_group_id}/statistics/members"
        params = {"offset": offset, "limit": limit}
        params.update(kwargs)
        return self._request("GET", endpoint, params=params)

    def get_all_statistics_members(self, group_identity, **kwargs):
        return self._paginate(self.get_statistics_members, group_identity=group_identity, **kwargs)

    def __7_内容操作(self):
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


class LakeBlock(GetAttr):
    """ 语雀文档中的基本块类型 """
    _default = 'tag'

    def __init__(self, tag):  # noqa
        self.tag = tag
        self.type = check_block_type(tag)

    def fix_hints(self):
        class Hint(LakeBlock, XlBs4Tag): pass

        return typing.cast(Hint, self)

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


class LakeDoc(GetAttr):
    """ 语雀文档类型 """
    _default = 'soup'

    def __init__(self, soup):  # noqa，这个类初始化就是跟父类不同的
        # 原始完整的html文档内容
        self.soup: XlBs4Tag = soup
        self.type = 'doc'

    def fix_hints(self):
        class Hint(LakeDoc, XlBs4Tag): pass

        return typing.cast(Hint, self)

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
    from xlproject.code4101 import *

    # 1. 初始化
    yq = Yuque()
    yq.update_doc('https://www.yuque.com/code4101/journal/w240415-', {'slug': 'w240415'})

    # yq.export_markdown(doc_url, XlPath.desktop())

#     doc_info = yq.get_doc(doc_url)
#
#     current_title = doc_info['title']
#     current_slug = doc_info['slug']
#     # 获取 Markdown 正文，截取前 2000 个字符供 AI 参考，避免太长
#     content_preview = doc_info.get('body', '')[:2000]
#
#     # 4. 生成 Prompt
#     prompt = f"""我有一篇语雀文档，信息如下：
# 【标题】：{current_title}
# 【当前 Slug (URL路径)】：{current_slug}
# 【文档内容】：
# {content_preview}
# ...
#
# 请帮我分析当前的 Slug 是否合适。
# 如果不合适（例如它是随机字符、中文拼音、或者与内容不符），请基于内容生成一个更好的英文 Slug（短横线命名法，如 architecture-design）。
# 你可以输出多个备选方案，并分析解释，还可以跟我在后续进一步推敲探讨。
# """
#
#     print("-" * 30)
#     print("请复制以下内容发送给 AI：")
#     print("-" * 30)
#     print(prompt)
#     print("-" * 30)
