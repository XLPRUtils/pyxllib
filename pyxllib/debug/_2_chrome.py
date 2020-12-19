#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 22:43


import html

from pyxllib.debug._1_typelib import *


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
    """调用procname相关的文件程序打开files
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


def chrome(arg_, **kwargs):
    r"""使用谷歌浏览器查看变量、文件等内容，详细用法见底层函数 viewfiles

    :param arg_: 支持输入多种类型
        文件、url，会用浏览器直接打开
        dict，会先转df
    :param kwargs: 如果数据可以转为df，在使用to_html接口的时候可以设置相关格式参数，常用的有
        escape=True，是否解析html内容，默认True不解析

    >> chrome(r'C:\Users\kzche\Desktop\b.xml')  # 使用chrome查看文件内容
    >> chrome('aabb')  # 使用chrome查看一个字符串值
    >> chrome([123, 456])  # 使用chrome查看一个变量值

    这个函数可以浏览文本、list、dict、DataFrame表格数据、图片、html等各种文件的超级工具
    """
    # 1 如果是文件、url，则直接打开
    if is_file(arg_) or is_url(arg_):
        viewfiles('chrome.exe', arg_)
        return

    # 2 如果是其他类型，则先转成文件，再打开
    arg = try2df(arg_)
    if isinstance(arg, pd.DataFrame):  # DataFrame在网页上有更合适的显示效果
        t = f'==== 类继承关系：{inspect.getmro(type(arg_))}，' \
            + f'内存消耗：{sys.getsizeof(arg_)}（递归子类总大小：{getasizeof(arg_)}）Byte ===='
        t = '<p>' + html.escape(t) + '</p>'
        content = arg.to_html(**kwargs)
        f = File(..., Dir.TEMP, suffix='.html').write(t + content)
        f = f.rename(get_etag(str(f)) + '.html', if_exists='delete')
        viewfiles('chrome.exe', str(f))
    elif getattr(arg, 'render', None):  # pyecharts 等表格对象，可以用render生成html表格显示
        try:
            name = arg.options['title'][0]['text']
        except (LookupError, TypeError):
            name = Datetime().strftime('%H%M%S_%f')
        filename = File(name, Dir.TEMP, suffix='.html').to_str()
        arg.render(path=filename)
        viewfiles('chrome.exe', filename)
    else:
        f = File(..., Dir.TEMP, suffix='.txt').write(arg)
        f = f.rename(get_etag(str(f)) + f.suffix, if_exists='delete')
        viewfiles('chrome.exe', str(f))


def view_jsons_kv(fd, files='**/*.json', encoding=None, max_items=10, max_value_length=100):
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
    chrome(p.to_str())


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

    chrome(view_table, escape=not link)
    return df
