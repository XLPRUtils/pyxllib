#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/30 22:43


import os
import subprocess


from pyxllib.extend.pathlib_ import Path


def viewfile(procname, file, **kwargs):
    """调用procname相关的文件程序打开file
    :param procname: 程序名
    :param file: 一个文件名参数清单，每一个都是文件路径，或者是字符串等可以用writefile转成文件的路径
    :param kwargs:
        wait: 是否等待当前进程结束后，再运行后续py代码
            True：在同一个进程中执行子程序，即会等待bc退出后，再进入下一步
            False：在新的进程中执行子程序
        filename: 控制写入的文件名
        TODO 根据不同软件，这里还可以扩展很多功能

    细节：注意bc跟其他程序有比较大不同，建议使用专用的bcompare函数
    目前已知可以扩展多文件的有：chrome、notepad++、texstudio

    >> ls = list(range(100))
    >> viewfiles('notepad++', ls)
    """
    # 1、生成文件名
    basename = ext = None
    if 'filename' in kwargs and kwargs['filename']:
        basename, ext = os.path.splitext(kwargs['filename'])

    for i, t in enumerate(files):
        if Path(t).is_file() or isurl(t):
            ls.append(t)
        else:
            # 如果要保存，则设为None，程序会自动按时间戳存储，否则设为特定名称的文件，下次运行就会把上次的覆盖了
            bn = None if save else f'file{i + 1}'
            if basename:
                bn = basename
            ls.append(writefile(t, bn, suffix=ext, root=Path.TEMP))

    # 2、调用程序（并计算外部操作时间）
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


def chrome(file, filename=None, ndim=None, **kwargs):
    r"""使用谷歌浏览器查看内容，详细用法见底层函数viewfiles
    >> chrome(r'C:\Users\kzche\Desktop\b.xml')  # 使用chrome查看文件内容
    >> chrome('aabb')  # 使用chrome查看一个字符串值
    >> chrome([123, 456])  # 使用chrome查看一个变量值

    这个函数可以浏览文本、list、dict、DataFrame表格数据、图片、html等各种文件的超级工具
    """
    if Path(file).is_file() or isurl(file):
        viewfiles('chrome.exe', file)
        return

    t = f'==== 类继承关系：{inspect.getmro(type(file))}，' \
        + f'内存消耗：{sys.getsizeof(file)}（递归子类总大小：{getasizeof(file)}）Byte ===='
    t = '<p>' + html.escape(t) + '</p>'

    if isinstance(file, dict):
        file = dict2list(file, nsort=True)
        file = pd.DataFrame.from_records(file, columns=('key', 'value'), **kwargs)

    if isinstance(file, (list, tuple)) and ndim != 1:
        try:  # 能转DataFrame处理的转DateFrame
            file = pd.DataFrame.from_records(list(file), **kwargs)
        except TypeError:  # TypeError: 'int' object is not iterable
            pass
    elif ndim == 1:
        file = pd.Series(file)
        file = pd.DataFrame(file, **kwargs)

    if isinstance(file, pd.Series):
        file = pd.DataFrame(file)

    if isinstance(file, pd.DataFrame):  # DataFrame在网页上有更合适的显示效果
        df = file
        if not filename: filename = 'a.html'
        if not filename.endswith('.html'): filename += '.html'
        viewfiles('chrome.exe', t + df.to_html(**kwargs), filename=filename)
    elif getattr(file, 'render', None):  # pyecharts等表格对象，可以用render生成html表格显示
        if not filename:
            try:
                filename = file.options['title'][0]['text'] + '.html'
            except (LookupError, TypeError):
                filename = 'render.html'
        file = file.render(path=os.path.join(tempfile.gettempdir(), filename))
        viewfiles('chrome.exe', file)
    else:
        viewfiles('chrome.exe', file, filename=filename)
