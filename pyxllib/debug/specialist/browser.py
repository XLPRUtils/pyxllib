#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 22:43


import enum
import html
import inspect
import os
import subprocess
import sys

import pandas as pd
from bs4 import BeautifulSoup

from pyxllib.debug.pupil import dprint, func_input_message
from pyxllib.debug.specialist.common import TypeConvert, NestedDict, KeyValuesCounter, dataframe_str
from pyxllib.file.specialist import File, Dir, get_etag
from pyxllib.prog.newbie import typename
from pyxllib.prog.pupil import is_url, is_file
from pyxllib.text.pupil import ensure_gbk
from pyxllib.debug.specialist.datetime import Datetime
from pyxllib.debug.specialist.tictoc import TicToc


def getasizeof(*objs, **opts):
    """获得所有类的大小，底层用pympler.asizeof实现"""
    from pympler import asizeof

    try:
        res = asizeof.asizeof(*objs, **opts)
    # except TypeError:  # sqlalchemy.exc.InvalidRequestError
    except:
        res = -1
    return res


def viewfiles(procname, *files, **kwargs):
    """ 调用procname相关的文件程序打开files

    :param procname: 程序名
    :param files: 一个文件名参数清单，每一个都是文件路径，或者是字符串等可以用writefile转成文件的路径
    :param kwargs:
        save: 如果True，则会按时间保存文件名；否则采用特定名称，每次运行就会把上次的覆盖掉
        wait: 是否等待当前进程结束后，再运行后续py代码
        filename: 控制写入的文件名
        TODO：根据不同软件，这里还可以扩展很多功能
    :param kwargs:
        wait:
            True：在同一个进程中执行子程序，即会等待bc退出后，再进入下一步
            False：在新的进程中执行子程序

    细节：注意bc跟其他程序有比较大不同，建议使用专用的bcompare函数
    目前已知可以扩展多文件的有：chrome、notepad++、texstudio

    >> ls = list(range(100))
    >> viewfiles('notepad++', ls, save=True)
    """
    # 1 生成文件名
    ls = []  # 将最终所有绝对路径文件名存储到ls
    save = kwargs.get('save')

    basename = ext = None
    if 'filename' in kwargs and kwargs['filename']:
        basename, ext = os.path.splitext(kwargs['filename'])

    for i, t in enumerate(files):
        if File(t) or is_url(t):
            ls.append(str(t))
        else:
            bn = basename or ...
            ls.append(File(bn, Dir.TEMP, suffix=ext).write(t, if_exists=kwargs.get('if_exists', 'error')).to_str())

    # 2 调用程序（并计算外部操作时间）
    tictoc = TicToc()
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
            raise FileNotFoundError(f'未找到程序：{procname}。请检查是否有安装及设置了环境变量。')
    return tictoc.tocvalue()


class Explorer:
    def __init__(self, app='explorer', shell=False):
        self.app = app
        self.shell = shell

    # def check_app(self, raise_error=False):
    #     """ 检查是否能找到对应的app
    #
    #     FIXME 不能提前检查，因为有些命令运行是会产生实际影响的，无法静默测试
    #         例如explorer是会打开资源管理器的
    #     """
    #     try:
    #         subprocess.run(self.app)
    #         return True
    #     except FileNotFoundError:
    #         if raise_error:
    #             raise FileNotFoundError(f'Application/Command not found：{self.app}')
    #         return False

    def __call__(self, *args, wait=True, **kwargs):
        """
        :param args: 命令行参数
        :param wait: 是否等待程序运行结束再继续执行后续python命令
        :param kwargs: 扩展参数，参考subprocess接口
        :return:

        TODO 获得返回值分析
        """
        args = [self.app] + list(args)
        if 'shell' not in kwargs:
            kwargs.update({'shell': self.shell})

        try:
            if wait:
                subprocess.run(args, **kwargs)
            else:
                subprocess.Popen(args, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f'Application/Command not found：{" ".join(args)}')


class Browser(Explorer):
    """ 使用浏览器查看数据文件

    标准库 webbrowser 也有一套类似的功能，那套主要用于url的查看，不支持文件
    而我这个主要就是把各种数据转成文件来查看
    """

    def __init__(self, app=None, shell=False):
        """
        :param app: 使用的浏览器程序，例如'msedge', 'chrome'，也可以输入程序绝对路径
            默认值None会自动检测标准的msedge、chrome目录是否在环境变量，自动获取
            如果要用其他浏览器，或者不在标准目录，请务必要设置app参数值
            在找没有的情况下，默认使用 'explorer'
        :param shell:
        """

        if app is None:
            # 智能判断环境变量，选择存在的浏览器，我的偏好 msedge > chrome
            paths = os.environ['PATH']
            msedge_dir = r'C:\Program Files (x86)\Microsoft\Edge\Application'
            chrome_dir = r'C:\Program Files\Google\Chrome\Application'
            if msedge_dir in paths:
                app = 'msedge'
            elif chrome_dir in paths:
                app = 'chrome'
            else:
                app = 'explorer'
        super().__init__(app, shell)

    @classmethod
    def to_brower_file(cls, arg, file=None, clsmsg=True, to_html_args=None):
        """ 将任意数值类型的arg转存到文件，转换风格会尽量适配浏览器的使用

        :param arg: 任意类型的一个数据
        :param file: 想要存储的文件名，没有输入的时候会默认生成到临时文件夹，文件名使用哈希值避重
        :param clsmsg: 显示开头一段类型继承关系、对象占用空间的信息
        :param to_html_args: df.to_html相关格式参数，写成字典的形式输入，常用的参数有如下
            escape, 默认True，将内容转移明文显示；可以设为False，这样在df存储的链接等html语法会起作用

        说明：其实所谓的用更适合浏览器的方式查看，在我目前的算法版本里，就是尽可能把数据转成DataFrame表格
        """
        # 1 如果已经是文件、url，则不处理
        if is_file(arg) or is_url(arg) or isinstance(arg, File):
            return arg

        # 2 如果是其他类型，则先转成文件，再打开
        arg_ = TypeConvert.try2df(arg)
        if isinstance(arg_, pd.DataFrame):  # DataFrame在网页上有更合适的显示效果
            if clsmsg:
                t = f'==== 类继承关系：{inspect.getmro(type(arg))}，' \
                    + f'内存消耗：{sys.getsizeof(arg)}（递归子类总大小：{getasizeof(arg)}）Byte ===='
                content = '<p>' + html.escape(t) + '</p>'
            else:
                content = ''
            # TODO 把标题栏改成蓝色~~
            content += arg.to_html(**(to_html_args or {}))
            if file is None:
                file = File(..., Dir.TEMP, suffix='.html').write(content)
                file = file.rename(get_etag(str(file)) + '.html', if_exists='delete')
            else:
                file = File(file).write(content)
        elif getattr(arg, 'render', None):  # pyecharts 等表格对象，可以用render生成html表格显示
            try:
                name = arg.options['title'][0]['text']
            except (LookupError, TypeError):
                name = Datetime().strftime('%H%M%S_%f')
            if file is None:
                file = File(name, Dir.TEMP, suffix='.html').to_str()
            arg.render(path=str(file))
        else:  # 不在预设格式里的数据，转成普通的txt查看
            if file is None:
                file = File(..., Dir.TEMP, suffix='.txt').write(arg)
                file = file.rename(get_etag(str(file)) + file.suffix, if_exists='delete')
            else:
                file = File(file).write(arg)
        return file

    def __call__(self, arg, file=None, *, wait=True, clsmsg=True, to_html_args=None,
                 **kwargs):  # NOQA Browser的操作跟标准接口略有差异
        """ 该版本会把arg转存文件重设为文件名

        :param file: 默认可以不输入，会按七牛的etag哈希值生成临时文件
            如果输入，则按照指定的名称生成文件
        """
        file = str(self.to_brower_file(arg, file, clsmsg=clsmsg, to_html_args=to_html_args))
        super().__call__(str(file), wait=wait, **kwargs)


browser = Browser()


def browser_json(f):
    """ 可视化一个json文件结构 """
    data = File(f).read()
    # 使用NestedDict.to_html_table转成html的嵌套表格代码，存储到临时文件夹
    htmlfile = File(r'chrome_json.html', root=Dir.TEMP).write(NestedDict.to_html_table(data))
    # 展示html文件内容
    browser(htmlfile)


def browser_jsons_kv(fd, files='**/*.json', encoding=None, max_items=10, max_value_length=100):
    """ demo_keyvaluescounter，查看目录下json数据的键值对信息

    :param fd: 目录
    :param files: 匹配的文件格式
    :param encoding: 文件编码
    :param max_items: 项目显示上限，有些数据项目太多了，要精简下
            设为假值则不设上限
    :param max_value_length: 添加的值，进行截断，防止有些值太长
    :return:
    """
    kvc = KeyValuesCounter()
    d = Dir(fd)
    for p in d.select(files).subfiles():
        # print(p)
        data = p.read(encoding=encoding, mode='.json')
        kvc.add(data, max_value_length=max_value_length)
    p = File(r'demo_keyvaluescounter.html', Dir.TEMP)
    p.write(kvc.to_html_table(max_items=max_items), if_exists='delete')
    browser(p.to_str())


def check_repeat_filenames(dir, key='stem', link=True):
    """ 检查目录下文件结构情况的功能函数

    https://www.yuque.com/xlpr/pyxllib/check_repeat_filenames

    :param dir: 目录Dir类型，也可以输入路径，如果没有files成员，则默认会获取所有子文件
    :param key: 以什么作为行分组的key名称，基本上都是用'stem'，偶尔可能用'name'
        遇到要忽略 -eps-to-pdf.pdf 这种后缀的，也可以自定义处理规则
        例如 key=lambda p: re.sub(r'-eps-to-pdf', '', p.stem).lower()
    :param link: 默认True会生成文件超链接
    :return: 一个df表格，行按照key的规则分组，列默认按suffix扩展名分组
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
                return ', '.join([f'<a href="{dir / f}" target="_blank">{f}</a>' for f in files])
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


def getmembers(object, predicate=None):
    """自己重写改动的 inspect.getmembers

    TODO 这个实现好复杂，对于成员，直接用dir不就好了？
    """
    from inspect import isclass, getmro
    import types

    if isclass(object):
        mro = (object,) + getmro(object)
    else:
        mro = ()
    results = []
    processed = set()
    names = dir(object)
    # :dd any DynamicClassAttributes to the list of names if object is a class;
    # this may result in duplicate entries if, for example, a virtual
    # attribute with the same name as a DynamicClassAttribute exists
    try:
        for base in object.__bases__:
            for k, v in base.__dict__.items():
                if isinstance(v, types.DynamicClassAttribute):
                    names.append(k)
    except AttributeError:
        pass
    for key in names:
        # First try to get the value via getattr.  Some descriptors don't
        # like calling their __get__ (see bug #1785), so fall back to
        # looking in the __dict__.
        try:
            value = getattr(object, key)
            # handle the duplicate key
            if key in processed:
                raise AttributeError
        # except AttributeError:
        except:  # 加了这种异常获取，190919周四15:14，sqlalchemy.exc.InvalidRequestError
            dprint(key)  # 抓不到对应的这个属性
            for base in mro:
                if key in base.__dict__:
                    value = base.__dict__[key]
                    break
            else:
                # could be a (currently) missing slot member, or a buggy
                # __dir__; discard and move on
                continue

        if not predicate or predicate(value):
            results.append((key, value))
        processed.add(key)
    results.sort(key=lambda pair: pair[0])
    return results


def showdir(c, *, to_html=None, printf=True):
    """查看类信息
    会罗列出类c的所有成员方法、成员变量，并生成一个html文

    查阅一个对象的成员变量及成员方法
    为了兼容linux输出df时也能对齐，有几个中文域宽处理相关的函数

    :param c: 要处理的对象
    :param to_html:
        win32上默认True，用chrome、explorer打开
        linux上默认False，直接输出到控制台
    :param printf:
        默认是True，会输出到浏览器或控制条
        设为False则不输出
    """
    # 1 输出类表头
    res = []
    object_name = func_input_message(2)['argnames'][0]
    if to_html is None:
        to_html = sys.platform == 'win32'
    newline = '<br/>' if to_html else '\n'

    t = f'==== 对象名称：{object_name}，类继承关系：{inspect.getmro(type(c))}，' \
        + f'内存消耗：{sys.getsizeof(c)}（递归子类总大小：{getasizeof(c)}）Byte ===='

    if to_html:
        res.append('<p>')
        t = html.escape(t) + '</p>'
    res.append(t + newline)

    # 2 html的样式精调
    def df2str(df):
        if to_html:
            df = df.applymap(str)  # 不转成文本经常有些特殊函数会报错
            df.index += 1  # 编号从1开始
            # pd.options.display.max_colwidth = -1  # 如果临时需要显示完整内容
            t = df.to_html()
            table = BeautifulSoup(t, 'lxml')
            table.thead.tr['bgcolor'] = 'LightSkyBlue'  # 设置表头颜色
            ch = 'A' if '成员变量' in table.tr.contents[3].string else 'F'
            table.thead.tr.th.string = f'编号{ch}{len(df)}'
            t = table.prettify()
        else:
            # 直接转文本，遇到中文是会对不齐的，但是showdir主要用途本来就是在浏览器看的，这里就不做调整了
            t = dataframe_str(df)
        return t

    # 3 添加成员变量和成员函数
    # 成员变量
    members = getmembers(c)
    methods = filter(lambda m: not callable(getattr(c, m[0])), members)
    ls = []
    for ele in methods:
        k, v = ele
        if k.endswith(r'________'):  # 这个名称的变量是我代码里的特殊标记，不显示
            continue
        attr = getattr(c, k)
        if isinstance(attr, enum.IntFlag):  # 对re.RegexFlag等枚举类输出整数值
            v = typename(attr) + '，' + str(int(attr)) + '，' + str(v)
        else:
            try:
                text = str(v)
            except:
                text = '取不到str值'
            v = typename(attr) + '，' + text
        ls.append([k, v])
    df = pd.DataFrame.from_records(ls, columns=['成员变量', '描述'])
    res.append(df2str(df) + newline)

    # 成员函数
    methods = filter(lambda m: callable(getattr(c, m[0])), members)
    df = pd.DataFrame.from_records(methods, columns=['成员函数', '描述'])
    res.append(df2str(df) + newline)
    res = newline.join(res)

    # 4 使用chrome.exe浏览或输出到控制台
    #   这里底层可以封装一个chrome函数来调用，但是这个chrome需要依赖太多功能，故这里暂时手动简单调用
    if to_html:
        filename = File(object_name, Dir.TEMP, suffix='.html'). \
            write(ensure_gbk(res), if_exists='delete').to_str()
        browser(filename)
    else:  # linux环境直接输出表格
        print(res)

    return res


def render_echart(ob, name, show=False):
    """ 渲染显示echart图表

    https://www.yuque.com/xlpr/pyxllib/render_echart

    :param ob: 一个echart图表对象
    :param name: 存储的文件名
    :param show: 是否要立即在浏览器显示
    :return: 存储的文件路径
    """
    # 如果没有设置页面标题，则默认采用文件名作为标题
    if not ob.page_title or ob.page_title == 'Awesome-pyecharts':
        ob.page_title = name
    f = ob.render(str(File(f'{name}.html', Dir.TEMP)))
    if show: browser(f)
    return f
