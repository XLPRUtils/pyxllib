#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30


import collections
import filecmp
import os
import pathlib
import random
import re
import shutil
import tempfile

import humanfriendly

# 大小写不敏感字典
from pyxllib.prog.newbie import first_nonnone
from pyxllib.algo.pupil import natural_sort
from pyxllib.text.pupil import strfind
from pyxllib.file.specialist import get_etag, PathBase, File, XlPath


def __1_Dir类():
    """
    支持文件或文件夹的对比复制删除等操作的函数：filescmp、filesdel、filescopy
    """


class Dir(PathBase):
    r"""类似NestEnv思想的文件夹处理类

    这里的测试可以全程自己造一个
    """
    __slots__ = ('_path', 'subs', '_origin_wkdir')

    # 零、常用的目录类
    TEMP = pathlib.Path(tempfile.gettempdir())
    if os.getenv('Desktop', None):  # 如果修改了win10默认的桌面路径，需要在环境变量添加一个正确的Desktop路径值
        DESKTOP = os.environ['Desktop']
    else:
        DESKTOP = os.path.join(str(pathlib.Path.home()), 'Desktop')  # 这个不一定准，桌面是有可能被移到D盘等的
    DESKTOP = pathlib.Path(DESKTOP)

    # 添加 HOME 目录？ 方便linux操作？

    # 一、基本目录类功能

    def __init__(self, path=None, root=None, *, subs=None, check=True):
        """根目录、工作目录

        >> Dir()  # 以当前文件夹作为root
        >> Dir(r'C:/pycode/code4101py')  # 指定目录

        :param path: 注意哪怕path传入的是Dir，也只会设置目录，不会取其paths成员值
        :param subs: 该目录下，选中的子文件（夹）
        """

        self._path = None
        self.subs = subs or []  # 初始默认没有选中任何文件（夹）

        # 1 快速初始化
        if root is None:
            if isinstance(path, Dir):
                self._path = path._path
                # 注意用Dir A 初始化 Dir B，并不会把A的subs传递给B
                return
            elif isinstance(path, pathlib.Path):
                self._path = path

        # 2 普通初始化
        if self._path is None:
            self._path = self.abspath(path, root)

        # 3 检查
        if check:
            if not self._path:
                raise ValueError(f'无效路径 {self._path}')
            elif self._path.is_file():
                raise ValueError(f'不能用文件初始化一个Dir对象 {self._path}')

    @classmethod
    def safe_init(cls, path, root=None, *, subs=None):
        """ 如果失败不raise，而是返回None的初始化方式 """
        try:
            d = Dir(path, root, subs=subs)
            d._path.is_file()  # 有些问题上一步不一定测的出来，要再补一个测试
            return d
        except (ValueError, TypeError, OSError, PermissionError):
            # ValueError：文件名过长，代表输入很可能是一段文本，根本不是路径
            # TypeError：不是str等正常的参数
            # OSError：非法路径名，例如有 *? 等
            # PermissionError: linux上访问无权限、不存在的路径
            return None

    @property
    def size(self) -> int:
        """ 计算目录的大小，会递归目录计算总大小

        https://stackoverflow.com/questions/1392413/calculating-a-directory-size-using-python

        >> Dir('D:/slns/pyxllib').size  # 这个算的就是真实大小，不是占用空间
        2939384
        """
        if self:
            total_size = 0
            for dirpath, dirnames, Pathnames in os.walk(str(self)):
                for f in Pathnames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
        else:  # 不存在的对象
            total_size = 0
        return total_size

    @property
    def psize(self) -> str:
        """ 美化显示的文件大小 """
        return humanfriendly.format_size(self.size, binary=True)

    def __truediv__(self, key) -> pathlib.Path:
        r""" 路径拼接功能

        >>> Dir('C:/a') / 'b.txt'
        WindowsPath('C:/a/b.txt')
        """
        return self._path / str(key)

    def with_dirname(self, value):
        return Dir(self.name, value)

    def absdst(self, dst):
        """ 在copy、move等中，给了个"模糊"的目标位置dst，智能推导出实际file、dir绝对路径
        """
        dst_ = self.abspath(dst)
        if isinstance(dst, str) and dst[-1] in ('\\', '/'):
            dst_ = Dir(self.name, dst_)
        else:
            dst_ = Dir(dst_)
        return dst_

    def ensure_dir(self):
        r""" 确保目录存在
        """
        if not self:
            os.makedirs(str(self))

    def copy(self, dst, if_exists=None):
        return self.process(dst, shutil.copytree, if_exists)

    def rename(self, dst, if_exists=None):
        r""" 重命名
        """
        return self.move(Dir(dst, self.parent), if_exists)

    def delete(self):
        r""" 删除自身文件
        """
        if self:
            try:
                shutil.rmtree(str(self))
            except OSError:
                # OSError: Cannot call rmtree on a symbolic link
                # TODO 本来不应该try except，而是先用os.path.islink判断的，但是这个好像有bug，判断不出来~~
                os.unlink(str(self))

    # 二、目录类专有功能

    def sample(self, n=None, frac=None):
        """
        :param n: 在 paths 中抽取n个文件
        :param frac: 按比例抽取文件
        :return: 新的Dir文件选取状态
        """
        n = n or int(frac * len(self.subs))
        paths = random.sample(self.subs, n)
        return Dir(self._path, subs=paths)

    def subpaths(self):
        """ 返回所有subs的绝对路径 """
        return [self._path / p for p in self.subs]

    def subfiles(self):
        """ 返回所有subs的File对象 （过滤掉文件夹对象） """
        return list(map(File, filter(lambda p: not p.is_dir(), self.subpaths())))

    def subdirs(self):
        """ 返回所有subs的File对象 （过滤掉文件对象） """
        return list(map(Dir, filter(lambda p: not p.is_file(), self.subpaths())))

    def select(self, patter, nsort=True, type_=None,
               ignore_backup=False, ignore_special=False,
               min_size=None, max_size=None,
               min_ctime=None, max_ctime=None, min_mtime=None, max_mtime=None,
               **kwargs):
        r""" 增加选中文件，从filesmatch衍生而来，参数含义见 filesfilter

        :param bool nsort: 是否使用自然排序，关闭可以加速
        :param str type_:
            None，所有文件
            'file'，只匹配文件
            'dir', 只匹配目录
        :param bool ignore_backup: 如果设为False，会过滤掉自定义的备份文件格式，不获取备份类文件
        :param bool ignore_special: 自动过滤掉 '.git'、'$RECYCLE.BIN' 目录下文件
        :param int min_size: 文件大小过滤，单位Byte
        :param int max_size: ~
        :param str min_ctime: 创建时间的过滤，格式'2019-09-01'或'2019-09-01 00:00'
        :param str max_ctime: ~
        :param str min_mtime: 修改时间的过滤
        :param str max_mtime: ~
        :param kwargs: see filesfilter
        :seealso: filesfilter

        注意select和exclude的增减操作是不断叠加的，而不是每次重置！
        如果需要重置，应该重新定义一个Folder类

        >> Dir('C:/pycode/code4101py').select('*.pyw').select('ckz.py')
        C:/pycode/code4101py: ['ol批量修改文本.pyw', 'ckz.py']
        >> Dir('C:/pycode/code4101py').select('**/*.pyw').select('ckz.py')
        C:/pycode/code4101py: ['ol批量修改文本.pyw', 'chenkz/批量修改文本.pyw', 'winr/bc.pyw', 'winr/reg/FileBackup.pyw', 'ckz.py']

        >> Dir('C:/pycode/code4101py').select('*.py', min_size=200*1024)  # 200kb以上的文件
        C:/pycode/code4101py: ['liangyb.py']

        >> Dir(r'C:/pycode/code4101py').select('*.py', min_mtime=datetime.date(2020, 3, 1))  # 修改时间在3月1日以上的
        """
        subs = filesmatch(patter, root=str(self), type_=type_,
                          ignore_backup=ignore_backup, ignore_special=ignore_special,
                          min_size=min_size, max_size=max_size,
                          min_ctime=min_ctime, max_ctime=max_ctime, min_mtime=min_mtime, max_mtime=max_mtime,
                          **kwargs)
        subs = self.subs + subs
        if nsort: subs = natural_sort(subs)
        return Dir(self._path, subs=subs)

    def select_files(self, patter, nsort=True,
                     ignore_backup=False, ignore_special=False,
                     min_size=None, max_size=None,
                     min_ctime=None, max_ctime=None, min_mtime=None, max_mtime=None):
        """ TODO 这系列的功能可以优化加速，在没有复杂规则的情况下，可以尽量用源生的py检索方式实现 """
        subs = filesmatch(patter, root=str(self), type_='file',
                          ignore_backup=ignore_backup, ignore_special=ignore_special,
                          min_size=min_size, max_size=max_size,
                          min_ctime=min_ctime, max_ctime=max_ctime,
                          min_mtime=min_mtime, max_mtime=max_mtime)
        if nsort:
            subs = natural_sort(subs)
        for x in subs:
            yield File(self._path / x, check=False)

    def select_dirs(self, patter, nsort=True,
                    ignore_backup=False, ignore_special=False,
                    min_size=None, max_size=None,
                    min_ctime=None, max_ctime=None, min_mtime=None, max_mtime=None):
        subs = filesmatch(patter, root=str(self), type_='dir',
                          ignore_backup=ignore_backup, ignore_special=ignore_special,
                          min_size=min_size, max_size=max_size,
                          min_ctime=min_ctime, max_ctime=max_ctime,
                          min_mtime=min_mtime, max_mtime=max_mtime)
        if nsort:
            subs = natural_sort(subs)
        for x in subs:
            yield Dir(self._path / x, check=False)

    def select_paths(self, patter, nsort=True,
                     ignore_backup=False, ignore_special=False,
                     min_size=None, max_size=None,
                     min_ctime=None, max_ctime=None, min_mtime=None, max_mtime=None):
        subs = filesmatch(patter, root=str(self),
                          ignore_backup=ignore_backup, ignore_special=ignore_special,
                          min_size=min_size, max_size=max_size,
                          min_ctime=min_ctime, max_ctime=max_ctime,
                          min_mtime=min_mtime, max_mtime=max_mtime)
        if nsort:
            subs = natural_sort(subs)
        for x in subs:
            yield self._path / x

    def procpaths(self, func, start=None, end=None, ref_dir=None, pinterval=None, max_workers=1, interrupt=True):
        """ 对选中的文件迭代处理

        :param func: 对每个文件进行处理的自定义接口函数
            参数 p: 输入参数 Path 对象
            return: 可以没有返回值
                TODO 以后可以返回字典结构，用不同的key表示不同的功能，可以控制些高级功能
        :param ref_dir: 使用该参数时，则每次会给func传递两个路径参数
            第一个是原始的file，第二个是ref_dir目录下对应路径的file

        TODO 增设可以bfs还是dfs的功能？


        将目录 test 的所有文件拷贝到 test2 目录 示例代码：

        def func(p1, p2):
            File(p1).copy(p2)

        Dir('test').select('**/*', type_='file').procfiles(func, ref_dir='test2')

        """
        from pyxllib.prog.specialist import Iterate

        if ref_dir:
            ref_dir = Dir(ref_dir)
            paths1 = self.subpaths()
            paths2 = [(ref_dir / self.subs[i]) for i in range(len(self.subs))]

            def wrap_func(data):
                func(*data)

            data = zip(paths1, paths2)

        else:
            data = self.subpaths()
            wrap_func = func

        Iterate(data).run(wrap_func, start=start, end=end, pinterval=pinterval,
                          max_workers=max_workers, interrupt=interrupt)

    def select_invert(self, patter='**/*', nsort=True, **kwargs):
        """ 反选，在"全集"中，选中当前状态下没有被选中的那些文件

        这里设置的选择模式，是指全集的选择范围
        """
        subs = Dir(self).select(patter, nsort, **kwargs).subs
        cur_subs = set(self.subs)
        new_subs = []
        for s in subs:
            if s not in cur_subs:
                new_subs.append(s)
        return Dir(self._path, subs=new_subs)

    def exclude(self, patter, **kwargs):
        """ 去掉部分选中文件

        d1 = Dir('test').select('**/*.eps')
        d2 = d1.exclude('subdir/*.eps')
        d3 = d2.select_invert(type_='file')
        print(d1.files)  # ['AA20pH-c1=1-1.eps', 'AA20pH-c1=1-2.eps', 'subdir/AA20pH-c1=1-2 - 副本.eps']
        print(d2.files)  # ['AA20pH-c1=1-1.eps', 'AA20pH-c1=1-2.eps']
        print(d3.files)  # ['subdir/AA20pH-c1=1-2 - 副本.eps']
        """
        subs = set(filesmatch(patter, root=str(self), **kwargs))
        new_subs = []
        for s in self.subs:
            if s not in subs:
                new_subs.append(s)
        return Dir(self._path, subs=new_subs)

    def describe(self):
        """ 输出目录的一些基本统计信息
        """
        msg = []
        dir_state = self.select('*')
        files = dir_state.subfiles()
        suffixs = collections.Counter([f.suffix for f in files]).most_common()
        dir_size = self.size
        msg.append(f'size: {dir_size} ≈ {humanfriendly.format_size(dir_size, binary=True)}')
        msg.append(f'files: {len(files)}, {suffixs}')
        msg.append(f'dirs: {len(dir_state.subdirs())}')
        res = '\n'.join(msg)
        print(res)

    def __enter__(self):
        """ 使用with模式可以进行工作目录切换

        注意！注意！注意！
        切换工作目录和多线程混合使用会有意想不到的坑，要慎重！
        """
        self._origin_wkdir = os.getcwd()
        os.chdir(str(self))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self._origin_wkdir)


def __2_filesxxx():
    """
    本来Path、File是能同时处理文件、目录的
    改版后，files底层因为有用到File，现在却不能支持目录的操作了
    可能会有些bug，尽量不要用这些旧功能，或者尽早移除
    """


def filescmp(f1, f2, shallow=True):
    """只有两个存在且是同类型的文件或文件夹，内容相同才会返回True，否则均返回False
    :param f1: 待比较的第1个文件（文件夹）
    :param f2: 待比较的第2个文件（文件夹）
    :param shallow: 默认True，即是利用os.stat()返回的基本信息进行比较
        例如其中的文件大小，但修改时间等是不影响差异判断的
        如果设为False，则会打开比较具体内容，速度会慢一点
    """
    if os.path.isfile(f1) and os.path.isfile(f2):
        cmp = filecmp.cmp(f1, f2, shallow)
    elif os.path.isdir(f1) and os.path.isdir(f2):
        # 文件夹只确保直接子目录下的清单名称，不比较具体每个文件内容是否相同，和子目录相同
        t = filecmp.dircmp(f1, f2, shallow)
        cmp = False
        try:
            if not t.left_only and not t.right_only:
                cmp = True
        except TypeError:
            pass
    else:  # 有不存在的文件
        cmp = False
    return cmp


def filesfilter(files, *, root=os.curdir, type_=None,
                ignore_backup=False, ignore_special=False,
                min_size=None, max_size=None,
                min_ctime=None, max_ctime=None,
                min_mtime=None, max_mtime=None):
    """
    :param files: 类list对象
    :param type_:
        None，所有文件
        'file'，只匹配文件
        'dir', 只匹配目录
    :param ignore_backup: 如果设为False，会过滤掉自定义的备份文件格式，不获取备份类文件
    :param ignore_special: 自动过滤掉 '.git'、'$RECYCLE.BIN' 目录下文件
    :param min_size: 文件大小过滤，单位Byte
    :param max_size: ~
    :param min_ctime: 创建时间的过滤，格式'2019-09-01'或'2019-09-01 00:00'
    :param max_ctime: ~
    :param min_mtime: 修改时间的过滤
    :param max_mtime: ~
    :return:
    """
    from datetime import datetime

    def judge(f):
        if root: f = os.path.join(root, f)
        if type_ == 'file' and not os.path.isfile(f):
            return False
        elif type_ == 'dir' and not os.path.isdir(f):
            return False

        # 尽量避免调用 os.stat，判断是否有自定义大小、时间规则，没有可以跳过这部分
        check_arg = first_nonnone([min_size, max_size, min_ctime, max_ctime, min_mtime, max_mtime])
        if check_arg is not None:
            msg = os.stat(f)
            if first_nonnone([min_size, max_size]) is not None:
                size = File(f).size
                if min_size is not None and size < min_size: return False
                if max_size is not None and size > max_size: return False

            if min_ctime or max_ctime:
                file_ctime = datetime.fromtimestamp(msg.st_ctime)
                if min_ctime and file_ctime < min_ctime: return False
                if max_ctime and file_ctime > max_ctime: return False

            if min_mtime or max_mtime:
                file_mtime = datetime.fromtimestamp(msg.st_mtime)
                if min_mtime and file_mtime < min_mtime: return False
                if max_mtime and file_mtime > max_mtime: return False

        if ignore_special:
            parts = File(f).parts
            if '.git' in parts or '$RECYCLE.BIN' in parts:
                return False

        if ignore_backup and File(f).backup_time:
            return False

        return True

    root = os.path.abspath(root)
    return list(filter(judge, files))


def filesmatch(patter, *, root=os.curdir, **kwargs) -> list:
    r"""
    :param patter:
        str，
            不含*、?、<、>，普通筛选规则
            含*、?、<、>，支持Path.glob的通配符模式，使用**可以表示任意子目录
                glob其实支持[0-9]这种用法，但是[、]在文件名中是合法的，
                    为了明确要使用glob模式，我这里改成<>模式
                **/*，是不会匹配到根目录的
        re.Patter，正则筛选规则（这种方法会比较慢，但是很灵活）  或者其他有match成员函数的类也可以
            会获得当前工作目录下的所有文件相对路径，组成list
            对list的所有元素使用re.match进行匹配
        list、tuple、set对象
            对每一个元素，递归调用filesmatch
    其他参数都是文件筛选功能，详见filesfilter中介绍
    :return: 匹配到的所有存在的文件、文件夹，返回“相对路径”

    TODO patter大小写问题？会导致匹配缺失的bug吗？

    >> os.chdir('F:/work/filesmatch')  # 工作目录

    1、普通匹配
    >> filesmatch('a')  # 匹配当前目录下的文件a，或者目录a
    ['a']
    >> filesmatch('b/a/')
    ['b\\a']
    >> filesmatch('b/..\\a/')
    ['a']
    >> filesmatch('c')  # 不存在c则返回 []
    []

    2、通配符模式
    >> filesmatch('work/*.png')  # 支持通配符
    []
    >> filesmatch('*.png')  # 支持通配符
    ['1.png', '1[.png', 'logo.png']
    >> filesmatch('**/*.png')  # 包含所有子目录下的png图片
    ['1.png', '1[.png', 'logo.png', 'a\\2.png']
    >> filesmatch('?.png')
    ['1.png']
    >> filesmatch('[0-9]/<0-9>.txt')  # 用<0-9>表示[0-9]模式
    ['[0-9]\\3.txt']

    3、正则模式
    >> filesmatch(re.compile(r'\d\[\.png$'))
    ['1[.png']

    4、其他高级用法
    >> filesmatch('**/*', type_='dir', max_size=0)  # 筛选空目录
    ['b', '[0-9]']
    >> filesmatch('**/*', type_='file', max_size=0)  # 筛选空文件
    ['b/a', '[0-9]/3.txt']
    """
    from pathlib import Path
    root = os.path.abspath(root)

    # 0 规则匹配
    # patter = str(patter)  # 200916周三14:59，这样会处理不了正则，要关掉
    glob_chars_pos = strfind(patter, ('*', '?', '<', '>')) if isinstance(patter, str) else -1

    # 1 普通文本匹配  （没有通配符，单文件查找）
    if isinstance(patter, str) and glob_chars_pos == -1:
        path = Path(os.path.join(root, patter))
        if path:  # 文件存在
            p = str(path.resolve())
            if p.startswith(root): p = p[len(root) + 1:]
            res = [p]
        else:  # 文件不存在
            res = []
    # 2 glob通配符匹配
    elif isinstance(patter, str) and glob_chars_pos != -1:
        patter = patter.replace('\\', '/')
        t = patter[:glob_chars_pos].rfind('/')
        # 计算出这批文件实际所在的目录dirname
        if t == -1:  # 模式里没有套子文件夹
            dirname, basename = root, patter
        else:  # 模式里有套子文件夹
            dirname, basename = os.path.abspath(os.path.join(root, patter[:t])), patter[t + 1:]
        basename = basename.replace('<', '[').replace('>', ']')
        files = map(str, Path(dirname).glob(basename))

        n = len(root) + 1
        res = [(x[n:] if x.startswith(root) else x) for x in files]
    # 3 正则匹配 （只要有match成员函数就行，不一定非要正则对象）
    elif hasattr(patter, 'match'):
        files = filesmatch('**/*', root=root)
        res = list(filter(lambda x: patter.match(x), files))
    # 4 list等迭代对象
    elif isinstance(patter, (list, tuple, set)):
        res = []
        for p in patter: res += filesmatch(p, root=root)
    # 5 可调用对象
    elif callable(patter):
        from pyxllib.file.specialist import XlPath
        res = [f.relpath(root).as_posix() for f in XlPath(root).rglob('*') if patter(f)]
    else:
        raise TypeError

    # 2 filetype的筛选
    res = filesfilter(res, root=root, **kwargs)

    return [x.replace('\\', '/') for x in res]


def filesdel(path, **kwargs):
    """删除文件或文件夹
    支持filesfilter的筛选规则
    """
    for f in filesmatch(path, **kwargs):
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f)
        # TODO 确保删除后再执行后续代码 但是一直觉得这样写很别扭
        while os.path.exists(f): pass


def _files_copy_move_base(src, dst, filefunc, dirfunc,
                          *, if_exists=None, treeroot=None, **kwargs):
    # 1 辅助函数
    def proc_onefile(f, dst):
        # dprint(f, dst)
        # 1 解析dst参数：对文件或目录不同情况做预处理
        #   （输入的时候dst_可以只是目标的父目录，要推算出实际要存储的目标名）
        if os.path.isfile(f):
            if os.path.isdir(dst) or dst[-1] in ('/', '\\'):
                dst = os.path.join(dst, os.path.basename(f))
            func = filefunc
        else:
            if dst[0] in ('/', '\\'):
                dst = os.path.join(dst, os.path.basename(f))
            func = dirfunc

        # 2 根据目标是否已存在和if_exists分类处理
        File(dst).ensure_parent()
        # 目前存在，且不是把文件移向文件夹的操作
        if os.path.exists(dst):
            # 根据if_exists参数情况分类处理
            if if_exists is None:  # 智能判断
                if not filescmp(f, dst):  # 如果内容不同则backup
                    File(dst).backup(move=True)
                    func(f, dst)
                elif os.path.abspath(f).lower() == os.path.abspath(dst).lower():
                    # 如果内容相同，再判断其是否实际是一个文件，则调用重命名功能
                    os.rename(f, dst)
            elif if_exists == 'backup':
                File(dst).backup(move=True)
                func(f, dst)
            elif if_exists == 'replace':
                filesdel(dst)
                func(f, dst)
            elif if_exists == 'ignore':
                pass  # 跳过，不处理
            else:
                raise ValueError
        else:
            func(f, dst)  # TODO 这里有bug \2020LaTeX\C春季教材\初数\初一上\Word+外包商原稿

    # 2 主体代码
    files = filesmatch(src, **kwargs)

    if len(files) == 1:
        proc_onefile(files[0], dst)
    elif len(files) > 1:  # 多文件模式拆解为单文件模式操作
        # 如果设置了 treeroot，这里要预处理下
        if treeroot:
            treeroot = filesmatch(treeroot)[0]
            if treeroot[-1] not in ('/', '\\'):
                treeroot += '/'
        n = len(treeroot) if treeroot else 0
        if treeroot: treeroot = treeroot.replace('\\', '/')

        # 迭代操作
        for f in files:
            dst_ = dst
            if treeroot and f.startswith(treeroot):
                dst_ = os.path.join(dst, f[n:])
            proc_onefile(f, dst_)


def filescopy(src, dst, *, if_exists=None, treeroot=None, **kwargs):
    r"""会自动添加不存在的目录的拷贝

    :param src: 要处理的目标
        'a'，复制文件a，或者整个文件夹a
        'a/*.txt'，复制文件夹下所有的txt文件
        更多匹配模式详见 filesmatch
    :param dst: 移到目标位置
        'a',
            如果a是已存在的目录，效果同'a/'
            如果是已存在的文件，且src只有一个要复制的文件，也是合法的。否则报错
                错误类型包括，把一个目录复制到已存在的文件
                把多个文件复制到已存在的文件
            如果a不存在，则
                src只是一个待复制的文件时是合法的
        'a/'，（可以省略写具体值，只写父级目录）将src匹配到的所有文件，放到目标a目录下
    :param if_exists: backup和replace含智能处理，如果内容相同则直接ignore
        'ignore'，跳过
        'backup'（默认），备份
            注意多文件操作时，来源不同的文件夹可能有同名文件
        'replace'，强制替换
    :param treeroot: 输入一个目录名开启该功能选项 （此模式下dst末尾强制要有一个'/'）
        对src中匹配到的所有文件，都会去掉treeroot的父目录前缀
            然后将剩下文件的所有相对路径结构，拷贝到dst目录下
        示例：将a目录下所有png图片原结构拷贝到b目录下
            filescopy('a/**/*.png', 'b/', if_exists='replace', treeroot='a')
        友情提示：treeroot要跟src使用同样的相对或绝对路径值，否则可能出现意外错误

        >> filescopy('filesmatch/**/*.png', 'filesmatch+/', treeroot='filesmatch')
        filesmatch： 1.png，a/2.png  -> filesmatch+：1.png，a/2.png

        >> filescopy('filesmatch/**/*.png', 'filesmatch+/')
        filesmatch： 1.png，a/2.png  -> filesmatch+：1.png，2.png

    TODO filescopy和filesmove还是有瑕疵和效率问题的，有空要继续优化
    """
    return _files_copy_move_base(src, dst, shutil.copy2, shutil.copytree,
                                 if_exists=if_exists, treeroot=treeroot, **kwargs)


def filesmove(src, dst, *, if_exists=None, treeroot=None, **kwargs):
    r"""与filescopy高度相同，见filescopy文档

    >> filesmove('a.xslx', 'A.xlsx', if_exists='replace')  # 等价于 os.rename('a.xlsx', 'A.xlsx')
    """
    return _files_copy_move_base(src, dst, shutil.move, shutil.move,
                                 if_exists=if_exists, treeroot=treeroot, **kwargs)


def refinepath(s, reserve=''):
    """
    :param reserve: 保留的字符，例如输入'*?'，会保留这两个字符作为通配符
    """
    if not s: return s
    # 1 去掉路径中的不可见字符，注意这里第1个参数里有一个不可见字符！别乱动这里的代码！
    s = s.replace(chr(8234), '')
    chars = set(r'\/:*?"<>|') - set(reserve)
    for ch in chars:  # windows路径中不能包含的字符
        s = s.replace(ch, '')

    # 2 去除目录、文件名前后的空格
    s = re.sub(r'\s+([/\\])', r'\1', s)
    s = re.sub(r'([/\\])\s+', r'\1', s)

    return s


def writefile(ob, path='', *, encoding='utf8', if_exists='backup', suffix=None, root=None, etag=None) -> str:
    """往文件path写入ob内容
    :param ob: 写入的内容
        如果要写txt文本文件且ob不是文本对象，只会进行简单的字符串化
    :param path: 写入的文件名，使用空字符串时，会使用etag值
    :param encoding: 强制写入的编码
    :param if_exists: 如果文件已存在，要进行的操作
    :param suffix: 文件扩展名
        以'.'为开头，设置“候补扩展名”，即只在fn没有指明扩展名时，会采用
    :param root: 相对位置
    :return: 返回写入的文件名，这个主要是在写临时文件时有用
    """
    if etag is None: etag = (not path)
    if path == '': path = ...
    f = File(path, root, suffix=suffix).write(ob, encoding=encoding, if_exists=if_exists)
    if etag:
        f = f.rename(get_etag(str(f)))
    return str(f)


def merge_dir(src, dst, if_exists='skip'):
    """ 将src目录下的数据拷贝到dst目录
    """

    def func(p1, p2):
        p1.copy(p2, if_exists=if_exists)

    # 只拷文件和空目录，不然逻辑会乱
    Dir(src).select('**/*', type_='dir', max_size=0).select('**/*', type_='file').procpaths(func, ref_dir=dst)


def extract_files(src, dst, pattern, if_exists='replace'):
    """ 提取满足pattern模式的文件
    """
    d1, d2 = Dir(src), Dir(dst)
    files = d1.select(pattern).subs
    for f in files:
        p1, p2 = File(d1 / f), File(d2 / f)
        p1.copy(p2, if_exists=if_exists)


def file_or_dir_size(path):
    if os.path.isfile(path):
        return File(path).size
    elif os.path.isdir(path):
        return Dir(path).size
    else:
        return 0


def reduce_dir_depth(srcdir, unwrap=999):
    """ 精简冗余嵌套的目录

    比如a目录下只有一个文件：a/b/1.txt，
    那么可以精简为a/1.txt，不需要多嵌套一个b目录

    :param srcdir: 要处理的目录
    :param unwrap: 打算解开的层数，未设置则会尽可能多解开
    """
    import tempfile
    root = p = XlPath(srcdir)
    depth = 0

    ps = list(p.glob('*'))
    while len(ps) == 1 and ps[0].is_dir() and depth < unwrap:
        depth += 1
        p = ps[0]
        ps = list(p.glob('*'))

    if depth:
        # 注意这里技巧，为了避免多层目录里会有相对同名的目录，导致出现不可预料的bug
        # 算法原理是把要搬家的那层目录里的文件先移到临时文件，然后把原目录树结构删除后，再报临时文件的文件移回来
        tmpdir = tempfile.mktemp()
        shutil.move(str(p), str(tmpdir))
        if depth > 1:
            shutil.rmtree(next(root.glob('*')))

        for pp in XlPath(tmpdir).glob('*'):
            shutil.move(str(pp), str(root))
        shutil.rmtree(tmpdir)
