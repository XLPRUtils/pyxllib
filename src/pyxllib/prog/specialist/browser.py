#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 22:43

import builtins
import enum
import html
import inspect
import os
import subprocess
import sys
import datetime
import platform
import re
import types

from loguru import logger

from pyxllib.prog.lazyimport import lazy_import

try:
    from humanfriendly import format_size
except ModuleNotFoundError:
    format_size = lazy_import('from humanfriendly import format_size')

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import('pandas')

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    BeautifulSoup = lazy_import('from bs4 import BeautifulSoup', 'beautifulsoup4')

from pyxllib.prog.newbie import typename
from pyxllib.prog.pupil import func_input_message, is_url, is_file
from pyxllib.prog.specialist.common import (
    TypeConvert,
    NestedDict,
    KeyValuesCounter,
    dataframe_str,
)
from pyxllib.text.pupil import ensure_gbk, shorten
from pyxllib.file.specialist.dirlib import File, Dir, get_etag, XlPath


def __1_introspector_内省工具集():
    pass


def getasizeof(*objs, **opts):
    """ 获得所有类的大小，底层用pympler.asizeof实现

    :param objs: 要计算大小的对象
    :param opts: 传递给 asizeof 的其他参数
    :return int: 对象总大小，如果计算失败返回 -1
    """
    from pympler import asizeof

    try:
        res = asizeof.asizeof(*objs, **opts)
    except:
        res = -1
    return res


class Introspector:
    """ 负责分析对象结构，提取成员变量和方法，不处理任何展示逻辑 """

    def __init__(self, obj):
        """ 初始化内省器

        :param obj: 要分析的对象
        """
        self.obj = obj

    def get_memory_info(self):
        """ 获取内存消耗信息

        :return str: 内存信息描述字符串
        """
        size = sys.getsizeof(self.obj)
        recursive_size = getasizeof(self.obj)
        t = f'内存消耗：{format_size(size, binary=True)}'
        t += f"（递归子类总大小：{format_size(recursive_size, binary=True) if recursive_size != -1 else 'Unknown'}）"
        return t

    def get_mro_dataframe(self):
        """ 获取 MRO 继承链的 DataFrame

        :return pd.DataFrame: 包含继承链信息的表格
        """
        mro = inspect.getmro(type(self.obj))
        data = [[str(cls)] for cls in mro]
        df = pd.DataFrame(data, columns=['类继承层级'])
        return df

    def get_meta_info(self):
        """ 获取对象的元数据（继承关系、内存大小等）

        :return str: 元数据描述字符串
        """
        memory_info = self.get_memory_info()
        mro = inspect.getmro(type(self.obj))
        return f'==== 类继承关系：{mro}，{memory_info} ===='

    def get_html_meta_info(self):
        """ 获取对象的 HTML 格式元数据

        :return str: HTML 格式的元数据
        """
        return '<p>' + html.escape(self.get_meta_info()) + '</p>'

    def get_members(self):
        """ 获取成员列表，返回 (Fields_DataFrame, Methods_DataFrame)

        :return tuple: (df_fields, df_methods)
        """
        members = self._get_all_members()

        # 分离变量与方法
        fields_data = []
        methods_data = []

        for name, value in members:
            # 过滤掉内部特殊标记
            if name.endswith('________'):
                continue

            # 简单的判断：可调用的是方法，不可调用的是变量
            if callable(value):
                methods_data.append([name, str(value)])
            else:
                fields_data.append([name, self._format_field_value(value)])

        df_fields = pd.DataFrame(fields_data, columns=['成员变量', '描述'])
        df_methods = pd.DataFrame(methods_data, columns=['成员函数', '描述'])

        return df_fields, df_methods

    def _get_all_members(self):
        """ 提取所有成员

        :return list: 包含 (name, value) 元组的列表
        """
        results = []
        processed = set()

        # 1. 尝试 dir()
        names = dir(self.obj)

        # 2. 补漏 (DynamicClassAttribute 等)
        if hasattr(self.obj, '__bases__'):
            for base in self.obj.__bases__:
                for k, v in base.__dict__.items():
                    if isinstance(v, types.DynamicClassAttribute):
                        names.append(k)

        for key in names:
            try:
                value = getattr(self.obj, key)
            except Exception:
                # 某些属性可能在 getattr 时报错，尝试从类字典获取
                found = False
                for base in inspect.getmro(type(self.obj)):
                    if key in base.__dict__:
                        value = base.__dict__[key]
                        found = True
                        break
                if not found:
                    continue

            if key not in processed:
                results.append((key, value))
                processed.add(key)

        results.sort(key=lambda pair: pair[0])
        return results

    def _format_field_value(self, value):
        """ 处理单个变量值的格式化

        :param value: 变量值
        :return str: 格式化后的字符串
        """
        if isinstance(value, enum.IntFlag):
            return f'{typename(value)}，{int(value)}，{value}'
        try:
            return f'{typename(value)}，{value}'
        except:
            return '无法转换为str'


class ObjectFormatter:
    """ 负责将 Introspector 提供的数据渲染成不同格式 """

    def __init__(self, introspector, width=200):
        """ 初始化格式化器

        :param Introspector introspector: 内省器对象
        :param int width: 字符串显示的最大宽度
        """
        self.introspector = introspector
        self.width = width

    def to_text(self):
        """ 生成适合控制台输出的纯文本

        :return str: 纯文本报告
        """
        memory_info = self.introspector.get_memory_info()
        df_mro = self.introspector.get_mro_dataframe()
        df_fields, df_methods = self.introspector.get_members()

        # 截断过长的字符串
        for df in [df_fields, df_methods]:
            if not df.empty:
                df.iloc[:, 1] = df.iloc[:, 1].apply(
                    lambda x: shorten(x, width=self.width)
                )

        res = [f'==== {memory_info} ====']
        res.append('[类继承关系]')
        res.append(dataframe_str(df_mro))
        res.append('[成员变量]')
        res.append(dataframe_str(df_fields))
        res.append('[成员函数]')
        res.append(dataframe_str(df_methods))
        return '\n'.join(res)

    def to_html(self, title_name='Object'):
        """ 生成适合浏览器查看的 HTML

        :param str title_name: 报告标题
        :return str: HTML 格式的报告
        """
        memory_info = self.introspector.get_memory_info()
        df_mro = self.introspector.get_mro_dataframe()
        df_fields, df_methods = self.introspector.get_members()

        # HTML 内容构建
        html_parts = []

        # 1. Header
        html_parts.append(f'<h1>{title_name} 查看报告</h1>')
        html_parts.append(f'<p>{html.escape(memory_info)}</p>')

        # 2. Helper to style tables
        def _style_df(df, type_char, header_color='LightSkyBlue'):
            if df.empty:
                return '<p>No members found.</p>'

            # 转字符串防止 HTML 注入或编码错误
            df_str = df.map(str)
            # 截断（如果只有一列则不截断第二列，因为 MRO 只有一列）
            if df_str.shape[1] > 1:
                df_str.iloc[:, 1] = df_str.iloc[:, 1].apply(
                    lambda x: shorten(x, width=self.width)
                )

            df_str.index += 1
            html_content = df_str.to_html()

            # 使用 BS4 美化
            soup = BeautifulSoup(html_content, 'lxml')
            if soup.thead and soup.thead.tr:
                soup.thead.tr['bgcolor'] = header_color
                # 设置表头
                th_label = f'编号{type_char}{len(df)}'
                if soup.thead.tr.th:
                    soup.thead.tr.th.string = th_label
            return soup.prettify()

        # 3. Append Tables
        html_parts.append(_style_df(df_mro, 'C', header_color='Khaki'))
        html_parts.append('<br/>')
        html_parts.append(_style_df(df_fields, 'F', header_color='LightGreen'))
        html_parts.append('<br/>')
        html_parts.append(_style_df(df_methods, 'M', header_color='LightSkyBlue'))

        return '<br/>'.join(html_parts)


def inspect_object(obj, mode='str', width=200):
    """ 查看对象信息的通用入口函数

    :param obj: 要查看的对象
    :param str mode: 查看模式
        - auto: 自动选择，Windows 默认 browser，其他环境默认 console
        - console|text: 打印到控制台
        - str: 返回纯文本字符串
        - html_str: 返回 HTML 字符串
        - browser|html: 在浏览器中打开报告
    :param int width: 字符串截断宽度
    :return str: 报告内容
    """
    introspector = Introspector(obj)
    formatter = ObjectFormatter(introspector, width=width)

    if mode == 'auto':
        mode = 'browser' if sys.platform == 'win32' else 'console'

    if mode in ('console', 'text'):
        content = formatter.to_text()
        logger.info(content)
        return content
    elif mode == 'str':
        return formatter.to_text()
    elif mode == 'html_str':
        return formatter.to_html()
    elif mode in ('browser', 'html'):
        obj_name = type(obj).__name__
        content = formatter.to_html(title_name=obj_name)
        f = File(obj_name, Dir.TEMP, suffix='.html')
        f.write(ensure_gbk(content), if_exists='replace')
        browser(f.to_str())
        return content


def showdir(c, *, mode='auto', width=200):
    """ 兼容旧接口的 wrapper

    :param c: 对象
    :param str mode: 模式
    :param int width: 宽度
    :return str: 报告内容
    """
    return inspect_object(c, mode=mode, width=width)


def __2_browser基础组件():
    pass


def viewfiles(procname, *files, **kwargs):
    """ 调用procname相关的文件程序打开files

    :param str procname: 程序名
    :param files: 一个文件名参数清单，每一个都是文件路径，或者是字符串等可以用writefile转成文件的路径
    :param kwargs:
        - save: 如果True，则会按时间保存文件名；否则采用特定名称，每次运行就会把上次的覆盖掉
        - wait:
            - True：在同一个进程中执行子程序，即会等待bc退出后，再进入下一步
            - False：在新的进程中执行子程序
            - 默认 False (如果未提供)
        - filename: 控制写入的文件名
        - if_exists: 写入文件时的模式，默认 'error'
    :return: 运行耗时

    >>> ls = list(range(100))
    >>> viewfiles('notepad', ls, save=True)  # doctest: +SKIP
    """
    # 1 生成文件名
    ls = []  # 将最终所有绝对路径文件名存储到ls

    basename = ext = None
    if kwargs.get('filename'):
        basename, ext = os.path.splitext(kwargs['filename'])

    for i, t in enumerate(files):
        if File(t) or is_url(t):
            ls.append(str(t))
        else:
            bn = basename or ...
            ls.append(
                File(bn, Dir.TEMP, suffix=ext)
                .write(t, if_exists=kwargs.get('if_exists', 'error'))
                .to_str()
            )

    # 2 调用程序
    try:
        if kwargs.get('wait'):
            subprocess.run([procname, *ls])
        else:
            subprocess.Popen([procname, *ls])
    except FileNotFoundError:
        if procname in ('chrome', 'chrome.exe'):
            procname = 'explorer'  # 如果是谷歌浏览器找不到，尝试用系统默认浏览器
            viewfiles(procname, *files, **kwargs)
        else:
            raise FileNotFoundError(
                f'未找到程序：{procname}。请检查是否有安装及设置了环境变量。'
            )


class Explorer:
    """ 资源管理器类，负责调用系统程序打开文件 """

    def __init__(self, app='explorer', shell=False):
        """ 初始化 Explorer

        :param str app: 应用程序名称或路径，默认为 'explorer'
        :param bool shell: 是否通过 shell 执行命令，默认为 False
        """
        self.app = app
        self.shell = shell

    def __call__(self, *args, wait=True, **kwargs):
        """ 调用程序打开文件或执行命令

        :param args: 命令行参数
        :param bool wait: 是否等待程序运行结束再继续执行后续python命令，默认为 True
        :param kwargs: 扩展参数，参考 subprocess 接口
        :return: None
        """
        args = [self.app] + list(args)

        if 'shell' not in kwargs:
            kwargs.update({'shell': self.shell})
        if re.match(r'open\s', self.app):
            args = args[0] + ' ' + args[1]
            kwargs.update({'shell': True})
        try:
            if wait:
                subprocess.run(args, **kwargs)
            else:
                subprocess.Popen(args, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"Application/Command not found：{' '.join(args)}")


class Browser(Explorer):
    """ 使用浏览器查看数据文件

    标准库 webbrowser 也有一套类似的功能，那套主要用于url的查看，不支持文件
    而我这个主要就是把各种数据转成文件来查看
    """

    def __init__(self, app=None, shell=False):
        """ 初始化 Browser

        :param str|None app: 使用的浏览器程序，例如 'msedge', 'chrome'，也可以输入程序绝对路径
            - 默认值 None 会自动检测标准的 msedge、chrome 目录是否在环境变量，自动获取
            - 如果要用其他浏览器，或者不在标准目录，请务必要设置 app 参数值
            - 在找没有的情况下，默认使用 'explorer'
        :param bool shell: 是否通过 shell 执行
        """
        if app is None:
            if platform.system() == 'Windows':
                paths = os.environ['PATH']
                chrome_dir = r'Google\Chrome\Application'
                msedge_dir = r'Microsoft\Edge\Application'
                if chrome_dir in paths:
                    app = 'chrome'
                elif msedge_dir in paths:
                    app = 'msedge'
                else:  # 默认使用谷歌。之前试过explorer不行~~
                    app = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
            elif platform.system() == 'Linux':  # Linux系统（包括Ubuntu）
                app = 'google-chrome'
            else:
                app = 'open -a "Google Chrome"'
        super().__init__(app, shell)

    @classmethod
    def _write_to_temp(cls, content, file=None, suffix='.html'):
        """ 将内容写入临时文件，并支持 etag 重命名

        :param content: 文件内容
        :param file: 指定文件名
        :param str suffix: 文件后缀
        :return File: 生成的文件对象
        """
        if file is None:
            # 1 生成临时文件并写入
            f = File(..., Dir.TEMP, suffix=suffix).write(content)
            # 2 根据内容哈希重命名，实现避重
            f = f.rename(get_etag(str(f)) + f.suffix, if_exists='replace')
        else:
            f = File(file).write(content)
        return f

    @classmethod
    def to_browser_file(cls, arg, file=None, clsmsg=True, to_html_args=None):
        """ 将任意数值类型的arg转存到文件，转换风格会尽量适配浏览器的使用

        :param arg: 任意类型的一个数据
        :param file: 想要存储的文件名，没有输入的时候会默认生成到临时文件夹，文件名使用哈希值避重
        :param bool clsmsg: 显示开头一段类型继承关系、对象占用空间的信息
        :param dict to_html_args: df.to_html相关格式参数，写成字典的形式输入，常用的参数有如下
            - escape: 默认True，将内容转移明文显示；可以设为False，这样在df存储的链接等html语法会起作用
        :return File: 生成的文件对象

        说明：其实所谓的用更适合浏览器的方式查看，在我目前的算法版本里，就是尽可能把数据转成DataFrame表格
        """
        # 1 如果已经是文件、url，则不处理
        if is_file(arg) or is_url(arg) or isinstance(arg, File):
            return arg

        # 2 如果是其他类型，则根据类型获取内容和后缀
        arg_ = TypeConvert.try2df(arg)
        if isinstance(arg_, pd.DataFrame):  # DataFrame在网页上有更合适的显示效果
            content = Introspector(arg).get_html_meta_info() if clsmsg else ''
            content += arg_.to_html(**(to_html_args or {}))
            return cls._write_to_temp(content, file, suffix='.html')
        elif hasattr(arg, 'render'):
            # pyecharts 等表格对象，可以用render生成html表格显示
            try:
                name = arg.options['title'][0]['text']
            except (LookupError, TypeError):
                name = datetime.datetime.now().strftime('%H%M%S_%f')
            res_file = File(file or name, Dir.TEMP, suffix='.html')
            arg.render(path=str(res_file))
            return res_file
        else:  # 不在预设格式里的数据，转成普通的txt查看
            return cls._write_to_temp(arg, file, suffix='.txt')

    def html(self, arg, **kwargs):
        """ 将内容转为html展示

        :param arg: 要展示的内容
        :param kwargs: 传递给 __call__ 的其他参数，可以包含 'file' 参数指定文件名
        """
        file = kwargs.pop('file', None)
        res_file = self._write_to_temp(arg, file, suffix='.html')
        self.__call__(str(res_file), **kwargs)

    def url(self, *args, wait=True, **kwargs):
        """ 打开 URL

        :param args: URL 列表
        :param bool wait: 是否等待
        :param kwargs: 其他参数
        """
        super().__call__(*args, wait=wait, **kwargs)

    def __call__(
        self, arg, file=None, *, wait=True, clsmsg=True, to_html_args=None, **kwargs
    ):
        """ 该版本会把arg转存文件重设为文件名

        :param arg: 要查看的数据或文件路径
        :param file: 默认可以不输入，会按七牛的etag哈希值生成临时文件
            如果输入，则按照指定的名称生成文件
        :param bool wait: 是否等待
        :param bool clsmsg: 是否显示类信息
        :param dict to_html_args: 转 HTML 参数
        """
        res_file = self.to_browser_file(
            arg, file, clsmsg=clsmsg, to_html_args=to_html_args
        )
        super().__call__(str(res_file), wait=wait, **kwargs)


browser = Browser()


def __2_showdir():
    pass


# -----------------------------------------------------------------------------
# 1. Core Logic: 对象内省器 (只负责提取数据，不负责展示)
# -----------------------------------------------------------------------------


def __3_其他browser相关功能():
    pass


def browse_json(f):
    """ 可视化一个json文件结构

    :param f: json文件路径
    """
    data = File(f).read()
    # 使用NestedDict.to_html_table转成html的嵌套表格代码，存储到临时文件夹
    htmlfile = File('chrome_json.html', root=Dir.TEMP).write(
        NestedDict.to_html_table(data)
    )
    # 展示html文件内容
    browser(htmlfile)


def browse_jsons_kv(
    fd, files='**/*.json', encoding=None, max_items=10, max_value_length=100
):
    """ demo_keyvaluescounter，查看目录下json数据的键值对信息

    :param fd: 目录
    :param files: 匹配的文件格式
    :param encoding: 文件编码
    :param max_items: 项目显示上限，有些数据项目太多了，要精简下
            设为假值则不设上限
    :param max_value_length: 添加的值，进行截断，防止有些值太长
    :return: None
    """
    kvc = KeyValuesCounter()
    d = Dir(fd)
    for p in d.select_files(files):
        data = p.read(encoding=encoding, mode='.json')
        kvc.add(data, max_value_length=max_value_length)
    p = File('demo_keyvaluescounter.html', Dir.TEMP)
    p.write(kvc.to_html_table(max_items=max_items), if_exists='replace')
    browser(p.to_str())


def check_repeat_filenames(dir, key='stem', link=True):
    """ 检查目录下文件结构情况的功能函数

    https://www.yuque.com/xlpr/pyxllib/check_repeat_filenames

    :param dir: 目录Dir类型，也可以输入路径，如果没有files成员，则默认会获取所有子文件
    :param key: 以什么作为行分组的key名称，基本上都是用'stem'，偶尔可能用'name'
        遇到要忽略 -eps-to-pdf.pdf 这种后缀的，也可以自定义处理规则
        例如 key=lambda p: re.sub(r'-eps-to-pdf', '', p.stem).lower()
    :param bool link: 默认True会生成文件超链接
    :return pd.DataFrame: 一个df表格，行按照key的规则分组，列默认按suffix扩展名分组
    """
    # 1 智能解析dir参数
    if not isinstance(dir, Dir):
        dir = Dir(dir)
    if not dir.subs:
        dir = dir.select('**/*', type_='file')

    # 2 辅助函数，智能解析key参数
    if isinstance(key, str):
        def extract_key(p):
            return getattr(p, key).lower()
    elif callable(key):
        extract_key = key
    else:
        raise TypeError

    # 3 制作df表格数据
    columns = ['key', 'suffix', 'filename']
    li = []
    for f in dir.subs:
        p = File(f)
        li.append([extract_key(p), p.suffix.lower(), f])
    df = pd.DataFrame.from_records(li, columns=columns)

    # 4 分组
    def joinfile(files):
        if len(files):
            if link:
                return ', '.join(
                    [f"<a href='{dir / f}' target='_blank'>{f}</a>" for f in files]
                )
            else:
                return ', '.join(files)
        else:
            return ''

    groups = df.groupby(['key', 'suffix']).agg({'filename': joinfile})
    groups.reset_index(inplace=True)
    view_table = groups.pivot(index='key', columns='suffix', values='filename')
    view_table.fillna('', inplace=True)

    # 5 判断每个key的文件总数
    count_df = df.groupby('key').agg({'filename': 'count'})
    view_table = pd.concat([view_table, count_df], axis=1)
    view_table.rename({'filename': 'count'}, axis=1, inplace=True)

    browser(view_table, to_html_args={'escape': not link})
    return df


# 注册进builtins，可以在任意地方直接使用
setattr(builtins, 'browser', browser)
setattr(builtins, 'showdir', showdir)


if __name__ == '__main__':
    import fire

    fire.Fire()
