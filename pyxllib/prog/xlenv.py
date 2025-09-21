#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/10/30

import base64
import json
import os
import re
import socket
import subprocess

from pyxllib.prog.lazyimport import lazy_import

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import('pandas')

try:
    from envariable import setenv, unsetenv
except ModuleNotFoundError:
    setenv, unsetenv = lazy_import('from envariable import setenv, unsetenv')

try:
    from deprecated import deprecated
except ModuleNotFoundError:
    deprecated = lazy_import('from deprecated import deprecated')

from pyxllib.text.newbie import add_quote


class XlEnv:
    """ 环境变量数据解析类

    可以读取、存储json的字符串值，或者普通str
    有些敏感信息，可以再加一层base64加密存储

    环境变量也可以用来实现全局变量的信息传递，虽然不太建议这样做

    >> XlEnv.persist_set('TP10_ACCOUNT',
                           {'server': '172.16.250.250', 'port': 22, 'user': 'ckz', 'passwd': '123456'},
                           True)
    >> print(XlEnv.get('TP10_ACCOUNT'), True)  # 展示存储的账号信息
    eyJzZXJ2ZXIiOiAiMTcyLjE2LjE3MC4xMzQiLCAicG9ydCI6IDIyLCAidXNlciI6ICJjaGVua3VuemUiLCAicGFzc3dkIjogImNvZGV4bHByIn0=
    >> XlEnv.unset('TP10_ACCOUNT')
    """

    @classmethod
    def get(cls, name, *, decoding=False):
        """ 获取环境变量值

        :param name: 环境变量名
        :param decoding: 是否需要先进行base64解码
        :return:
            返回json解析后的数据
            或者普通的字符串值
        """
        value = os.getenv(name, None)
        if value is None:
            return value

        if decoding:
            value = base64.b64decode(value.encode())

        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            if isinstance(value, bytes):
                return value.decode()
            else:
                return value

    @classmethod
    def set(cls, name, value, *, encoding=False):
        """ 临时改变环境变量

        :param name: 环境变量名
        :param value: 要存储的值
        :param encoding: 是否将内容转成base64后，再存储环境变量
            防止一些密码信息，明文写出来太容易泄露
            不过这个策略也很容易被破解；只防君子，难防小人

            当然，谁看到这有闲情功夫的话，可以考虑做一套更复杂的加密系统
            并且encoding支持多种不同的解加密策略，这样单看环境变量值就很难破译了
        :return: str, 最终存储的字符串内容
        """
        # 1 打包
        if isinstance(value, str):
            value = add_quote(value)  # 不加引号下次也能json.loads，但对于"123"这种会变成int类型，而非原本的str类型
        else:
            value = json.dumps(value)

        # 2 编码
        if encoding:
            value = base64.b64encode(value.encode()).decode()

        # 3 存储到环境变量
        os.environ[name] = value

        return value

    @classmethod
    def persist_set(cls, name, value, encoding=False, *, cfgfile=None):
        """ python里默认是改不了系统变量的，需要使用一些特殊手段
        https://stackoverflow.com/questions/17657686/is-it-possible-to-set-an-environment-variable-from-python-permanently/17657905

        :param cfgfile: 在linux系统时，可以使用该参数
            默认是把环境变量写入 ~/.bashrc，可以考虑写到
            TODO 有这个设想，但很不好实现，不是很关键的功能，所以还未开发

        """
        # 写入环境变量这里是有点小麻烦的，要考虑unix和windows不同平台，以及怎么持久化存储的问题，这里直接调用一个三方库来解决
        from envariable import setenv

        value = cls.set(name, value, encoding)
        if value[0] == value[-1] == '"':
            value = '\\' + value + '\\'
        setenv(name, value)

        return value

    @classmethod
    def unset(cls, name):
        """ 删除环境变量 """
        from envariable import unsetenv
        unsetenv(name)

    @classmethod
    def get_df(cls, name, *, decoding=False):
        """ 将内容按照表格的形式读取成pandas的df """
        data = cls.get(name, decoding=decoding)
        if data:
            return pd.DataFrame(data[1:], columns=data[0])


def get_xl_homedir(host=None, *, reset=False):
    """ 获取用户工作目录 """
    from pyxllib.file.specialist import XlPath
    if not os.getenv('XL_HOMEDIR') or reset:
        os.environ['XL_HOMEDIR'] = XlHosts.find_homedir(host)
    return XlPath(os.getenv('XL_HOMEDIR'))


def get_xl_hostname(*, reset=False):
    """ 特殊定制版的获取主机名 """
    if not os.getenv('XL_HOSTNAME') or reset:
        # 1 获得基础名
        hostname = socket.getfqdn()
        # 2 初步预处理
        hostname = hostname.replace('-', '_')
        hostname = hostname.split('.')[0]
        # 3 如果环境变量有配置则进一步映射名称
        df = XlHosts.get_df('XL_MACHINES')
        # 获得的df有raw_name、name两个字段
        # 查找df['raw_name']中，如果有等于hostname的值，需要把当前hostname映射到df中对应的df['name']的值
        matched_rows = df[df['raw_name'] == hostname]
        if not matched_rows.empty: hostname = matched_rows['name'].iloc[0]
        os.environ['XL_HOSTNAME'] = hostname

    return os.getenv('XL_HOSTNAME')


class XlHosts:
    """ 设备、链接、服务等相关管理器 """

    def __1_一级工具(self):
        pass

    @classmethod
    def get_df(cls, env_name):
        """ 读取一个环境变量的表格数据 """
        # 1 读取初步数据
        df = XlEnv.get_df(env_name)
        assert df is not None, f'未配置环境变量{env_name}'

        # 2 第1列如果有逗号，要展开数据
        processed_rows = []
        for _, row in df.iterrows():
            col = df.columns[0]
            names = row[col]
            # 如果from字段包含逗号，拆分成多条记录
            if pd.notna(names) and ',' in str(names):
                names = str(names).split(',')
                for name in names:
                    new_row = row.copy()
                    new_row[col] = name.strip()
                    # 链接表，局域网清单展开的本机自联，ip统一改为127.0.0.1
                    if env_name == 'XL_LINKS' and name == row['to']:
                        new_row['ip'] = '127.0.0.1'
                    processed_rows.append(new_row)
            else:
                processed_rows.append(row)

        df = pd.DataFrame(processed_rows).reset_index(drop=True)

        # 3 有些特殊数据需要补充处理
        if env_name == 'XL_LINKS':
            df = df.drop_duplicates(subset=['from', 'to'], keep='first')

        return df

    def __2_二级工具(self):
        """ 解析表格 """
        pass

    @classmethod
    def find_host(cls, host):
        """获取主机的详细信息（工作目录、账号密码等）

        :param host: 主机名
        :return: 主机信息字典，如果不存在返回None
        """
        host = host or get_xl_hostname()
        hosts = cls.get_df('XL_HOSTS')
        matched = hosts.loc[hosts['host'] == host]
        # 返回第一条匹配的记录
        return matched.iloc[0].to_dict() if len(matched) else {}

    @classmethod
    def find_link(cls, to_host, *, from_host=None):
        """查找从from_host到to_host的链接配置

        :param to_host: 目标主机名
        :param from_host: 源主机名，如果为None则使用当前主机名
        :return: 匹配的链接记录，如果没有找到返回None
        """
        # 获取基础参数
        from_host = from_host or get_xl_hostname()
        links = cls.get_df('XL_LINKS')

        # 优先级1：精确匹配from_host到to_host的映射（包括本机自连）
        matched = links.loc[(links['from'] == from_host) & (links['to'] == to_host)]

        # 优先级2：通用映射（from为"*"表示任意主机）
        if len(matched) == 0:
            matched = links.loc[(links['from'] == '*') & (links['to'] == to_host)]

        return matched.iloc[0].to_dict() if len(matched) else {}

    @classmethod
    def find_service(cls, to_host, service_type):
        df = cls.get_df('XL_SERVICES')
        matched = df[(df['host'] == to_host) & (df['service'] == service_type)]
        return matched.iloc[0].to_dict() if len(matched) else {}

    def __3_三级工具(self):
        """ 最终用途接口 """
        pass

    @classmethod
    def find_homedir(cls, host=None):
        info = cls.find_host(host)
        # 注意这里不采用Path.home()，而是把目录直接展开，确保下游相关功能稳定、准确性更强
        #   因为windows里的~并不会直接判别为用户目录，是可以直接作为路径名称的
        return info['homedir'] if info else os.path.expanduser('~')

    @classmethod
    def find_locator(cls, to_host, service_type, *, from_host=None):
        """ 查找某个服务的端口或路径，对于web来说，则是对应的url
        """
        link = cls.find_link(to_host, from_host=from_host) if isinstance(to_host, str) else to_host
        assert service_type in link['service'], f"{to_host} 不存在 {service_type} 服务"
        return link['service'][service_type]

    @classmethod
    def find_passwd(cls, to_host, service_type, user_name):
        accounts = cls.find_service(to_host, service_type)['accounts']
        assert user_name in accounts, f"{to_host}/{service_type}，不存在 {user_name} 用户"
        return accounts[user_name]


def __xlhome系列():
    """ 跟文件相关的创建功能，默认根目录是homedir """


def xlhome_dir(dir, root=None):
    """ 创建、定位在home目录下的subdir目录 """
    from pyxllib.file.specialist import XlPath
    root = get_xl_homedir() if root is None else XlPath(root)
    d = root / dir
    d.mkdir(exist_ok=True, parents=True)
    return d


def xlhome_wkdir(dir, root=None):
    """ 在home路面下创建subdir目录，并切换到该目录作为工作目录 """
    d = xlhome_dir(dir, root)
    os.chdir(d)
    return d


def xlhome_path(file, root=None):
    """ 定位在home目录下的file文件
    如果不存在，会自动创建文件所属的所有父目录
    """
    from pyxllib.file.specialist import XlPath
    root = get_xl_homedir() if root is None else XlPath(root)
    f = root / file
    f.parent.mkdir(exist_ok=True, parents=True)
    return f


def open_network_file(host, name):
    """ 打开network相关文件
    """

    # 1 分类处理
    if name.endswith('.py'):
        file = xlhome_path(f'slns/xlproject/xlserver/{name}')
    elif name == 'nginx.conf':
        file = xlhome_path(f'data/m2405network/sync/{host}/{name}')
        if not file.is_file():
            file = xlhome_path(f'data/m2405network/sync/{host}/nginx-1.26.1/conf/{name}')
    else:
        file = xlhome_path(f'data/m2405network/sync/{host}/{name}')

    # 2 打开文件
    if get_xl_hostname() == 'codepc_aw':
        subprocess.run([os.getenv('CODEPC_AW_PYCHARM'), str(file)])
    else:
        os.startfile(str(file))


def __service系列():
    pass


def link_to_host_service(to_host, service='web'):
    """ 连接目标主机的网页、API服务

    根据主机映射关系构建目标主机的访问URL。

    :param str to_host: 目标主机名
    :return str|None: 构建的URL，如果无法构建则返回None
    """
    # 1 查找链接、web配置
    link = XlHosts.find_link(to_host)
    locator = XlHosts.find_locator(link, service)
    ip = link.get('ip', '')

    # 2 处理locator格式
    if isinstance(locator, int):
        locator = f':{locator}'
    elif isinstance(locator, str) and locator:
        if locator.isdigit():
            locator = f':{locator}'
        else:
            locator = f'/{locator}'
    else:
        locator = ''

    # 3 确定协议（有端口号、IP地址或localhost时用http，否则用https）
    is_ip = re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', ip)
    has_port = locator.startswith(':')
    is_localhost = ip == 'localhost'

    protocol = 'http://' if (is_ip or has_port or is_localhost) else 'https://'

    # 4 拼接结果
    return f'{protocol}{ip}{locator}'
