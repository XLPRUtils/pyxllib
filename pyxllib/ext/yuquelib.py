#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/01/01


import requests

from pyxllib.xl import *
from pyxllib.algo.stat import *
from pprint import pprint


class Yuque:
    def __init__(self, token, user_id=None):
        self.base_url = "https://www.yuque.com/api/v2"
        self.headers = {
            "X-Auth-Token": token,
            "Content-Type": "application/json"
        }
        self.user_id = user_id

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
        response = requests.get(url, headers=self.headers)
        return response.json()

    def get_user_id(self):
        if self.user_id is None:
            self.user_id = self.get_user()['data']['id']
        return self.user_id

    @run_once('id,str')
    def get_repos(self):
        """ 获取某个用户的知识库列表

        """
        url = f"{self.base_url}/users/{self.user_id}/repos"
        response = requests.get(url, headers=self.headers)
        return response.json()

    @run_once('id,str')
    def get_repos_name2id(self):
        """ 获取知识库名字到ID的映射
        """
        data = self.get_repos()
        return {d['namespace'].split('/')[-1]: d['id'] for d in data['data']}

    def get_repos2(self):
        """ 获取df结构的知识库描述
        """
        data = self.get_repos()
        columns = ['id', 'name', 'items_count', 'namespace']

        ls = []
        for d in data['data']:
            ls.append([d[col] for col in columns])

        df = pd.DataFrame(ls, columns=columns)
        return df

    def get_repo_docs(self, repo_id):
        """ 获取知识库的文档列表

        :param repo_id: 知识库的ID或Namespace
        :return: 文档列表，只能获得最近的100篇文档
        """
        url = f"{self.base_url}/repos/{repo_id}/docs"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def get_repo_docs2(self, repo_id):
        """ 获取df结构的文档描述
        """
        data = self.get_repo_docs(repo_id)
        columns = ['id', 'title', 'word_counts', 'description']

        ls = []
        for d in data['data']:
            ls.append([d[col] for col in columns])

        ls.sort(key=lambda x: x[0])  # id一般就是创建顺序
        df = pd.DataFrame(ls, columns=columns)
        return df

    def get_doc(self, repo_id, doc_id):
        """ 获取单篇文档的详细信息

        :param repo_id: 知识库的ID或Namespace
        :param doc_id: 文档的ID
        :return: 文档的详细信息
        """
        url = f"{self.base_url}/repos/{repo_id}/docs/{doc_id}"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def get_doc_from_url(self, url, return_mode=1):
        """ 从文档的URL中获取文档的详细信息

        :param url: 文档的URL
        :param return_mode: 返回模式，
            0为原始json结构，
            1为返回文档内容
        :return: 文档的详细信息
        """
        repo_slug, doc_slug = url.split('/')[-2:]
        name2id = self.get_repos_name2id()
        if repo_slug not in name2id:
            raise ValueError(f'知识库"{repo_slug}"不存在')
        repo_id = name2id[repo_slug]
        res = self.get_doc(repo_id, doc_slug)

        if return_mode == 0:
            return res['data']
        elif return_mode == 1:
            return res['data']['body']

    def export_markdown(self, url, output_dir=None, post_mode=1):
        """ 导出md格式文件

        :param url: 文档的URL
        :param output_dir: 导出目录（文件名是按照文章标题自动生成的）
        :param post_mode: 后处理模式
            0，不做处理
            1，做适当的精简
        """
        # 1 获得内容
        data = self.get_doc_from_url(url, return_mode=0)
        body = data['body']
        if post_mode == 0:
            pass
        elif post_mode == 1:
            body = re.sub(r'<a\sname=".*?"></a>\n', '', body)

        # 2 写入文件
        if output_dir is not None:
            title2 = refinepath(data['title'].replace(': ', '：'))
            f = XlPath(output_dir) / f'{title2}.md'
            f.write_text(body)

        return body

    def read_tables_from_doc(self, url, header=0):
        """ 从文档中读取表格

        :param url: 文档的URL
        :return: 表格列表
        """
        res = self.get_doc_from_url(url, return_mode=0)
        tables = pd.read_html(res['body_html'], header=header)
        return tables
