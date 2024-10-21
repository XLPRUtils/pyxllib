#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/01/01


import requests

from pyxllib.xl import *
from pyxllib.algo.stat import *
from pprint import pprint

from pyxllib.text.nestenv import NestEnv


class Yuque:
    """
    https://www.yuque.com/yuque/developer/openapi
    语雀请求限制：每小时最多 5000 次请求，每秒最多 100 次请求
    """

    def __init__(self, token=None, user_id=None):
        self.base_url = "https://www.yuque.com/api/v2"
        self.headers = {
            "X-Auth-Token": token or os.environ.get('YUQUE_TOKEN'),
            "Content-Type": "application/json"
        }
        self._user_id = user_id

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
            nickname2id，获取知识库namespace或昵称到ID的映射
        """
        if return_mode == 0:
            url = f"{self.base_url}/users/{self.user_id}/repos"
            response = requests.get(url, headers=self.headers)
            return response.json()
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

    def get_repo_docs(self, repo_id, *, return_mode=0):
        """ 获取知识库的文档列表

        :param repo_id: 知识库的ID或Namespace（如"日志"是我改成的"journal"）
        :param int|str return_mode: 返回模式
            0（默认），返回原始json结构
            -1（df），df结构
        :return: 文档列表，只能获得最近的100篇文档
        """
        repo_id = self.get_repo_id(repo_id)
        if return_mode == 0:
            url = f"{self.base_url}/repos/{repo_id}/docs"
            response = requests.get(url, headers=self.headers)
            return response.json()
        elif return_mode in (-1, 'df'):
            data = self.get_repo_docs(repo_id)
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

    def __2_文档操作(self):
        pass

    def create_doc(self, repo_id, title, md_content):
        """ 创建单篇文档，并放到目录开头
        """
        # 1 创建文档
        repo_id = self.get_repo_id(repo_id)
        url = f"{self.base_url}/repos/{repo_id}/docs"
        in_data = {
            "title": title,
            "body": md_content,
        }
        resp = requests.post(url, json=in_data, headers=self.headers)
        out_data = resp.json()
        doc_id = out_data['data']['id']

        # 2 将文档添加到目录中
        url2 = f"{self.base_url}/repos/{repo_id}/toc"
        in_data2 = {
            "action": "prependNode",  # 默认添加到知识库目录最顶上的位置
            "action_mode": "child",
            "doc_id": doc_id,
        }
        resp2 = requests.put(url2, json=in_data2, headers=self.headers)
        out_data2 = resp2.json()

        return out_data2

    def get_doc(self, repo_id, doc_id):
        """ 获取单篇文档的详细信息

        :param repo_id: 知识库的ID或Namespace
        :param doc_id: 文档的ID
        :return: 文档的详细信息
        """
        repo_id = self.get_repo_id(repo_id)
        url = f"{self.base_url}/repos/{repo_id}/docs/{doc_id}"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def get_doc_from_url(self, url, return_mode='md'):
        """ 从文档的URL中获取文档的详细信息

        :param url: 文档的URL
        :param return_mode: 返回模式，
            json, 为原始json结构
            md, 返回文档的主体md内容
            title_and_md, 返回文档的标题和md内容
        :return: 文档的详细信息
        """
        repo_slug, doc_slug = url.split('/')[-2:]
        res = self.get_doc(repo_slug, doc_slug)

        if return_mode == 'json':
            return res['data']
        elif return_mode == 'md':
            return res['data']['body']
        elif return_mode == 'title_and_md':
            return res["data"]["title"], res["data"]["body"]

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
        data = self.get_doc_from_url(url, return_mode='json')
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

    def update_doc(self, repo_id, doc_id, doc_data):
        """ 更新单篇文档的详细信息

        :param repo_id: 知识库的ID或Namespace
        :param doc_id: 文档的ID
        :param doc_data: 包含文档更新内容的字典
        :return: 更新后的文档的详细信息
        """
        repo_id = self.get_repo_id(repo_id)
        url = f"{self.base_url}/repos/{repo_id}/docs/{doc_id}"
        response = requests.put(url, json=doc_data, headers=self.headers)
        return response.json()

    def update_doc_from_url(self, url, doc_data, *, md_cvt=True, return_mode='json'):
        """ 从文档的URL中更新文档的详细信息

        :param url: 文档的URL
        :param str|json doc_data: 包含文档更新内容的字典
            可以直接传入要更新的新的md内容
            注意无论原始是body_html、body_lake，都是要上传到body字段
        :param str return_mode: 返回的是更新后文档的内容，不过好像有bug，这里返回的body存储的并不是md格式
            'md', 返回更新后文档的主体md内容
            'json', 为原始json结构

            不建议拿这个返回值，完全可以另外再重新取返回值，就是正常的md格式了
        :param bool md_cvt: 是否需要转换md格式
            默认的md文档格式直接放回语雀，是会丢失换行的，需要对代码块外的内容，执行\n替换
        :return: 更新后的文档的详细信息
        """
        # 1 基础配置
        repo_slug, doc_slug = url.split('/')[-2:]
        if isinstance(doc_data, str):
            doc_data = {'body': doc_data}

        # 2 格式转换
        if md_cvt:
            ne = NestEnv(doc_data['body']).search(r'^```[^\n]*\n(.+?)\n^```',
                                                  flags=re.MULTILINE | re.DOTALL).invert()
            doc_data['body'] = ne.replace('\n', '\n\n')

        # 3 提交更新文档
        res = self.update_doc(repo_slug, doc_slug, doc_data)

        # 4 拿到返回值
        if return_mode == 'md':
            return res['data']
        elif return_mode == 'json':
            return res['data']['body']

    def __3_内容操作(self):
        pass

    def read_tables_from_doc(self, url, header=0):
        """ 从文档中读取表格

        :param url: 文档的URL
        :return: 表格列表
        """
        res = self.get_doc_from_url(url, return_mode=0)
        tables = pd.read_html(res['body_html'], header=header)
        return tables


if __name__ == '__main__':
    pass
