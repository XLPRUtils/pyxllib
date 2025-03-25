#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/01/01

import re
import time

import requests
import urllib.parse

from fastcore.basics import GetAttr
from pprint import pprint

from pyxllib.xl import *
from pyxllib.algo.stat import *
from pyxllib.prog.newbie import SingletonForEveryInitArgs
from pyxllib.text.pupil import UrlQueryBuilder
from pyxllib.text.nestenv import NestEnv
from pyxllib.text.xmllib import BeautifulSoup, XlBs4Tag
from pyxllib.cv.xlcvlib import xlcv


def update_yuque_doc_by_dp(doc_url):
    from DrissionPage import Chromium, ChromiumOptions
    from pyxllib.ext.drissionlib import dp_check_quit

    # 1 打开浏览器
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
    time.sleep(0.5)
    tab.actions.key_up('BACKSPACE')
    tab('t:button@@text():更新').click(by_js=True)
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


class LakeImage(GetAttr, XlBs4Tag):
    """ 语雀文档中的图片类型

    这个内容结构一般是 <p><card type="inline" name="image" value="data:..."></card></p>
    其中value是可以解析出一个字典，有详细数据信息的
    """
    _default = 'tag'

    def __init__(self, tag):  # noqa
        self.tag = tag
        self._value_dict = None

    def __初始化(self):
        pass

    @classmethod
    def _encode_value(cls, value_dict):
        """ 把当前value_dict的字典值转为html标签结构 """
        json_str = json.dumps(value_dict)
        url_encoded = urllib.parse.quote(json_str)
        final_str = "data:" + url_encoded
        return final_str

    @classmethod
    def _init_from_src(cls, src):
        """ 传入一个图片url或base64的值，转为语雀图片节点 """
        soup = BeautifulSoup('<p><card type="inline" name="image" value=""></card></p>', 'lxml')
        soup.card.attrs['value'] = cls._encode_value({'src': src})
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

    def to_url(self):
        """ 确认当前图片是以url的模式存储 """

    def to_base64(self):
        """ 确认当前图片是以base64的模式存储 """

    def __功能(self):
        pass

    @property
    def value_dict(self):
        if self._value_dict is None:
            encoded_str = self.tag.card.attrs['value']
            # 去掉 "data:" 前缀
            encoded_data = encoded_str.replace("data:", "")
            # URL 解码
            decoded_str = urllib.parse.unquote(encoded_data)
            self._value_dict = json.loads(decoded_str)
        return self._value_dict

    def update_value_dict(self):
        """ 把当前value_dict的字典值更新回html标签 """
        self.tag.card.attrs['value'] = self._encode_value(self.value_dict)


# GetAttr似乎必须放在前面，这样找不到的属性似乎是会优先使用GetAttr机制的，但后者又可以为IDE提供提示
class LakeDoc(GetAttr, XlBs4Tag):
    """ 语雀文档类型 """
    _default = 'soup'

    def __init__(self, soup):  # noqa，这个类初始化就是跟父类不同的
        # 原始完整的html文档内容
        self.soup: XlBs4Tag = soup

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

    def get_raw_paragraphs(self):
        """ 获得最原始的段落数组 """
        return list(self.soup.body.children)

    def print_raw_paragraphs(self):
        for i, c in enumerate(self.get_raw_paragraphs(), start=1):
            tp = self.check_type(c)
            if tp == 'image':
                # img = LakeImage(c)
                print(f'{i}、{c.tag_name} {shorten(c.prettify(), 200)}')
            elif tp == 'str':
                print(f'{i}、{c.tag_name} {shorten(c.text, 200)}')
            else:
                print(f'{i}、{c.tag_name} {shorten(c.prettify(), 200)}')
            print()

    def delete_lake_id(self):
        """ 删除文档中所有语雀标签的id标记 """
        for tag in self.soup.find_all(True):
            for name in ['data-lake-id', 'id']:
                if name in tag.attrs:
                    del tag[name]

    @classmethod
    def check_type(cls, tag):
        """ 这个分类会根据对语雀文档结构了解的逐渐深入和逐步细化 """
        tag_name = tag.tag_name
        if tag_name == 'p':
            if tag.find('card'):
                return 'image'
            else:
                return 'p'  # 就用p表示最普通的段落类型
        elif tag_name == 'card':
            return 'codeblock'
        elif tag_name == 'ol':
            return 'ol'
        elif re.match(r'h\d+$', tag_name):
            return 'heading'
        elif tag_name == 'details':
            return 'lake-collapse'
        elif tag_name == 'NavigableString':
            return 'str'
        else:
            raise TypeError('未识别类型')

    def body_add(self, node):
        """ 在正文末尾添加一个新节点内容

        :param node: 要添加的节点内容，或者html文本
        """
        self.soup.body.append_html(node)


if __name__ == '__main__':
    pass
