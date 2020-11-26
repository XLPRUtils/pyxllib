#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/30 20:37


from urllib.parse import urlparse
import io
import json
import os
import pathlib
import pickle
import re
import shutil
import subprocess
import tempfile

import chardet
import qiniu
import requests
import yaml

from pyxllib.basic._1_strlib import struct_unpack
from pyxllib.basic._2_timelib import Datetime

____judge = """
"""


def is_url(arg):
    """输入是一个字符串，且值是一个合法的url"""
    if not isinstance(arg, str): return False
    try:
        result = urlparse(arg)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def is_url_connect(url, timeout=5):
    try:
        _ = requests.head(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        pass
    return False


def is_file(arg, exists=True):
    """相较于标准库的os.path.isfile，对各种其他错误类型也会判False

    :param exists: arg不仅需要是一个合法的文件名，还要求其实际存在
        设为False，则只判断文件名合法性，不要求其一定要存在
    """
    if not isinstance(arg, str): return False
    if not exists:
        raise NotImplementedError
    return os.path.isfile(arg)


____qiniu = """
"""


def get_etag(arg):
    """七牛原有etag功能基础上做封装
    :param arg: 支持bytes二进制、文件、url地址
    """
    if isinstance(arg, bytes):  # 二进制数据
        return qiniu.utils.etag_stream(io.BytesIO(arg))
    elif is_file(arg):  # 输入是一个文件
        return qiniu.etag(arg)
    elif is_url(arg):  # 输入是一个网页上的数据源
        return get_etag(requests.get(arg).content)
    elif isinstance(arg, str):  # 明文字符串转二进制
        return get_etag(arg.encode('utf8'))
    else:
        raise TypeError('不识别的数据类型')


def is_etag(s):
    """字母、数字和-、_共64种字符构成的长度28的字符串"""
    return re.match(r'[a-zA-Z0-9\-_]{28}$', s)


def test_etag():
    print(get_etag(r'\chematom{+8}{2}{8}{}'))
    # Fjnu-ZXyDxrqLoZmNJ2Kj8FcZGR-

    print(get_etag(__file__))
    # 每次代码改了这段输出都是不一样的


def test_etag2():
    """ 字符串值和写到文件判断的etag，是一样的
    """
    s = 'code4101'
    print(get_etag(s))
    # FkAD2McB6ugxTiniE8ebhlNHdHh9

    f = Path('1.tex', root=Path.TEMP).write(s, if_exists='replace').fullpath
    print(get_etag(f))
    # FkAD2McB6ugxTiniE8ebhlNHdHh9


____chardet = """
"""


def get_encoding(bstr, maxn=1024):
    """ 输入二进制字符串或文本文件，返回字符编码

    https://www.yuque.com/xlpr/pyxllib/get_encoding

    :param bstr: 二进制字符串、文本文件
    :param maxn: 分析字节数上限，越小速度越快，但也会降低精准度
    :return: utf8, utf-8-sig, gbk, utf16
    """
    # 1 读取编码
    if isinstance(bstr, bytes):  # 如果输入是一个二进制字符串流则直接识别
        encoding = chardet.detect(bstr[:maxn])['encoding']  # 截断一下，不然太长了，太影响速度
    elif is_file(bstr):  # 如果是文件，则按二进制打开
        # 如果输入是一个文件名则进行读取
        if bstr.endswith('.pdf'):
            print('二进制文件，不应该进行编码分析，暂且默认返回utf8', bstr)
            return 'utf8'
        with open(bstr, 'rb') as f:  # 以二进制读取文件，注意二进制没有\r\n的值
            bstr = f.read()
        encoding = chardet.detect(bstr[:maxn])['encoding']
    else:  # 其他类型不支持
        return 'utf8'
    # 检测结果存储在encoding

    # 2 智能适应优化，最终应该只能是gbk、utf8两种结果中的一种
    if encoding in ('ascii', 'utf-8', 'ISO-8859-1'):
        # 对ascii类编码，理解成是utf-8编码；ISO-8859-1跟ASCII差不多
        encoding = 'utf8'
    elif encoding in ('GBK', 'GB2312'):
        encoding = 'gbk'
    elif encoding == 'UTF-16':
        encoding = 'utf16'
    elif encoding == 'UTF-8-SIG':
        # 保留原值的一些正常识别结果
        encoding = 'utf-8-sig'
    elif bstr.strip():  # 如果不在预期结果内，且bstr非空，则用常见的几种编码尝试
        # dprint(encoding)
        type_ = ('utf8', 'gbk', 'utf16')

        def try_encoding(bstr, encoding):
            try:
                bstr.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                return None

        for t in type_:
            encoding = try_encoding(bstr, t)
            if encoding: break
    else:
        encoding = 'utf8'

    return encoding


____path = """
"""


class Path:
    r""" 通用文件、路径处理类，也可以处理目录，可以把目录对象理解为特殊类型的文件

    document: https://www.yuque.com/xlpr/python/pyxllib.debug.path

    大部分基础功能是从pathlib.Path衍生过来的
        但开发中该类不能直接从Path继承，会有很多问题

    TODO 这里的doctest过于针对自己的电脑了，应该改成更具适用性测试代码
    """
    __slots__ = ('_path', 'assume_dir')

    # 零、常用的目录类
    TEMP = tempfile.gettempdir()
    if os.environ.get('Desktop', None):  # 如果修改了win10默认的桌面路径，需要在环境变量添加一个正确的Desktop路径值
        DESKTOP = os.environ['Desktop']
    else:
        DESKTOP = os.path.join(str(pathlib.Path.home()), 'Desktop')  # 这个不一定准，桌面是有可能被移到D盘等的

    # 一、基础功能

    def __init__(self, path=None, suffix=None, root=None):
        r""" 初始化参数含义详见 abspath 函数解释

        TODO 这个初始化也有点过于灵活了，需要降低灵活性，增加使用清晰度
            或者看下性能速度，可以考虑加速
            特别是用Path作为参数拷贝的情况，应该可以加速减少分析、拷贝

        >>> Path('D:/pycode/code4101py')
        Path('D:/pycode/code4101py')
        >>> Path(Path('D:/pycode/code4101py'))
        Path('D:/pycode/code4101py')

        >> Path()  # 不输入参数的时候，默认为当前工作目录
        Path('D:/pycode/code4101py')

        >> Path('a.txt', root=Path.TEMP)
        Path('D:/Temp/a.txt')
        >>> Path('F:/work/CreatorTemp')
        Path('F:/work/CreatorTemp')

        注意！如果使用了符号链接（软链接），则路径是会解析转向实际位置的！例如
        >> Path('D:/pycode/code4101py')
        Path('D:/slns/pycode/code4101py')
        """
        strpath = str(path)
        self._path = None
        self.assume_dir = False  # 假设是一个目录
        if len(strpath) and strpath[-1] in r'\/':
            self.assume_dir = True

        try:
            self._path = pathlib.Path(self.abspath(path, suffix, root)).resolve()
            # 有些问题上一步不一定测的出来，要再补一个测试
            # self._path.is_file()
        except (ValueError, TypeError, OSError):
            # ValueError：文件名过长，代表输入很可能是一段文本，根本不是路径
            # TypeError：不是str等正常的参数
            # OSError：非法路径名，例如有 *? 等
            self._path = None

    @staticmethod
    def abspath(path=None, suffix=None, root=None) -> str:
        r""" 根据各种不同的组合参数信息，推导出具体的路径位置

        (即把人常识易理解的指定思维，转成计算机能清晰明白的具体指向对象)
        :param path: 主要的参数，如果后面的参数有矛盾，以path为最高参考标准
            ''，可以输入空字符串，效果跟None是不一样的，意思是显示地指明要生成一个随机名称的文件名
        :param suffix:
            以 '.' 指明的扩展名，会强制替换
            否则，只作为参考扩展名，只在原有path没有指明的时候才添加
        :param root: 未输入的时候，则为当前工作目录

        >> Path.abspath()
        'C:\\pycode\\code4101py'

        >> Path.abspath('a')
        'C:\\pycode\\code4101py\\a'

        >> Path.abspath('F:/work', '.txt')  # F:/work是个实际存在的目录
        'F:/work\\tmpahqm6nod.txt'

        >> Path.abspath('F:/work/a.txt', 'py')  # 参考后缀不修改
        'F:/work/a.txt'
        >> Path.abspath('F:/work/a.txt', '.py')  # 强制后缀会修改
        'F:/work/a.py'

        >> Path.abspath(suffix='.tex', root=Path.TEMP)
        'F:\\work\\CreatorTemp\\tmp5vo2lpqd.tex'

        # F:/work/a.txt不存在，而且看起来像文件名，但是末尾再用/则显式指明这其实是一个目录
        #   又用.py指明要添加一个随机名称的py文件
        >> Path.abspath('F:/work/a.txt/', '.py')
        'F:/work/a.txt/tmpg_q7a7ft.py'

        >> Path.abspath('work/a.txt/', '.py', Path.TEMP)  # 在临时文件夹下的work/a.txt目录新建一个随机名称的py文件
        'F:\\work\\CreatorTemp\\work/a.txt/tmp2jn5cqkc.py'

        >> Path.abspath('C:/a.txt/')  # 会保留最后的斜杠特殊标记
        'C:/a.txt/'
        >> Path.abspath('C:/a.txt\\')  # 会保留最后的斜杠特殊标记
        'C:/a.txt\\'
        """
        # 1 判断参考目录
        if root:
            root = str(root)
        else:
            root = os.getcwd()

        # 2 判断主体文件名 path
        if str(path) == '':
            path = tempfile.mktemp(dir=root)
        elif not path:
            path = root
        else:
            path = os.path.join(root, str(path))

        # 3 补充suffix
        if suffix:
            # 如果原来的path只是一个目录，则要新建一个文件
            # if os.path.isdir(path) or path[-1] in ('\\', '/'):
            if path[-1] in ('\\', '/'):
                path = tempfile.mktemp(dir=path)
            # 判断后缀
            li = os.path.splitext(path)
            if (not li[1]) or suffix[0] == '.':
                ext = suffix if suffix[0] == '.' else ('.' + suffix)
                path = li[0] + ext

        return path

    def __bool__(self):
        return bool(self._path)

    def exists(self):
        r""" 判断文件是否存在

        重置WindowsPath的bool逻辑，返回值变成存在True，不存在为False

        >>> Path('D:/slns').exists()
        True
        >>> Path('D:/pycode/code4101').exists()
        False
        """
        return self._path and self._path.exists()

    def __repr__(self):
        return self._path.__repr__()

    def __str__(self):
        return str(self._path).replace('\\', '/') + ('/' if self.assume_dir else '')

    def to_str(self):
        return self.__str__()

    def __eq__(self, other):
        """ pathlib.Path内置了windows和linux的区别
            在windows是不区分大小写的，在linux则区分大小写
        """
        if not isinstance(other, Path):
            raise TypeError
        return self._path == other._path

    def __truediv__(self, key):
        r""" 路径拼接功能

        >>> Path('C:/a') / 'b.txt'
        Path('C:/a/b.txt')
        """
        return Path(self._path / str(key))

    def resolve(self):
        return self._path.resolve()

    def glob(self, pattern):
        return self._path.glob(pattern)

    def match(self, path_pattern):
        return self._path.match(path_pattern)

    # 二、获取、修改路径中部分值的功能

    @property
    def fullpath(self) -> str:
        return str(self._path)

    @property
    def drive(self) -> str:
        return self._path.drive

    @drive.setter
    def drive(self, value):
        """修改磁盘位置"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        r"""
        >>> Path('D:/pycode/a.txt').name
        'a.txt'
        >>> Path('D:/pycode/code4101py').name
        'code4101py'
        >>> Path('D:/pycode/.gitignore').name
        '.gitignore'
        """
        return self._path.name

    @name.setter
    def name(self, value):
        raise NotImplementedError

    @property
    def parent(self):
        r"""
        >>> Path('D:/pycode/code4101py').parent
        Path('D:/pycode')
        """
        return Path(self._path.parent) if self._path else None

    @property
    def dirname(self) -> str:
        r"""
        >>> Path('D:/pycode/code4101py').dirname
        'D:/pycode'
        >>> Path(r'D:\toweb\a').dirname
        'D:/toweb'
        """
        return str(self.parent)

    def with_dirname(self, value):
        return Path(self.name, root=value)

    @property
    def stem(self) -> str:
        r"""
        >>> Path('D:/pycode/code4101py/ckz.py').stem
        'ckz'
        >>> Path('D:/pycode/.gitignore').stem  # os.path.splitext也是这种算法
        '.gitignore'
        >>> Path('D:/pycode/.123.45.6').stem
        '.123.45'
        """
        return self._path.stem

    def with_stem(self, stem):
        """
        注意不能用@stem.setter来做
            如果setter要rename，那if_exists参数怎么控制？
            如果setter不要rename，那和用with_stem实现有什么区别？
        """
        return Path(stem, self.suffix, self.dirname)

    @property
    def parts(self) -> tuple:
        r"""
        >>> Path('D:/pycode/code4101py').parts
        ('D:\\', 'pycode', 'code4101py')
        """
        return self._path.parts

    @property
    def suffix(self) -> str:
        r"""
        >>> Path('D:/pycode/code4101py/ckz.py').suffix
        '.py'
        >>> Path('D:/pycode/code4101py').suffix
        ''
        >>> Path('D:/pycode/code4101py/ckz.').suffix
        ''
        >>> Path('D:/pycode/code4101py/ckz.123.456').suffix
        '.456'
        >>> Path('D:/pycode/code4101py/ckz.123..456').suffix
        '.456'
        >>> Path('D:/pycode/.gitignore').suffix
        ''
        """
        return self._path.suffix if self._path else ''

    def with_suffix(self, suffix):
        r""" 指向同目录下后缀为suffix的文件

        >>> Path('a.txt').with_suffix('.py').fullpath.split('\\')[-1]  # 强制替换
        'a.py'
        >>> Path('a.txt').with_suffix('py').fullpath.split('\\')[-1]  # 参考替换
        'a.txt'
        >>> Path('a.txt').with_suffix('').fullpath.split('\\')[-1]  # 删除
        'a'
        """
        if suffix and (suffix[0] == '.' or not self.suffix):
            if suffix[0] != '.': suffix = '.' + suffix
            return Path(self._path.with_suffix(suffix))
        elif not suffix:
            # suffix 为假值则删除扩展名
            return Path(self.stem, '', self.dirname)
        return self

    def joinpath(self, *args):
        return Path(self._path.joinpath(*args))

    @property
    def backup_time(self):
        r""" 返回文件的备份时间戳，如果并不是备份文件，则返回空字符串

        备份文件都遵循特定的命名规范
        如果是文件，是：'chePre 171020-153959.tex'
        如果是目录，是：'figs 171020-153959'
        通过后缀分析，可以判断这是不是一个备份文件

        >>> Path('chePre 171020-153959.tex').backup_time
        '171020-153959'
        >>> Path('figs 171020-153959').backup_time
        '171020-153959'
        >>> Path('figs 171020').backup_time
        ''
        """
        name = self.stem
        if len(name) < 14:
            return ''
        g = re.match(r'(\d{6}-\d{6})', name[-13:])
        return g.group(1) if g else ''

    # 三、获取文件相关属性值功能

    def is_dir(self):
        return self._path and self._path.is_dir()

    def is_file(self):
        return self._path and self._path.is_file()

    @property
    def encoding(self):
        """ 文件的编码

        非文件、不存在时返回 None
        """
        if self.is_file():
            return get_encoding(self.fullpath)
        return None

    @property
    def size(self) -> int:
        """ 计算文件、目录的大小，对于目录，会递归目录计算总大小

        https://stackoverflow.com/questions/1392413/calculating-a-directory-size-using-python

        >> Path('D:/slns/pyxllib').size  # 这个算的就是真实大小，不是占用空间
        2939384
        """
        path = str(self._path)
        if self._path.is_file():
            total_size = os.path.getsize(path)
        elif self._path.is_dir():
            total_size = 0
            for dirpath, dirnames, Pathnames in os.walk(path):
                for f in Pathnames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
        else:  # 不存在的对象
            total_size = 0
        return total_size

    @property
    def mtime(self) -> Datetime:
        r""" 文件的最近修改时间

        >> Path(r"C:\pycode\code4101py").mtime
        2020-03-10 17:32:37
        """
        return Datetime(os.stat(self.fullpath).st_mtime)

    @property
    def ctime(self) -> Datetime:
        r""" 文件的创建时间

        >> Path(r"C:\pycode\code4101py").ctime
        2018-05-25 10:46:37
        """
        # 注意：st_ctime是平台相关的值，在windows是创建时间，但在Unix是metadate最近修改时间
        return Datetime(os.stat(self.fullpath).st_ctime)

    # 四、文件操作功能

    def abs_dstpath(self, dst=None, suffix=None, root=None):
        r""" 参照当前Path的父目录，来确定dst的具体路径

        >>> f = Path('C:/Windows/System32/cmd.exe')
        >>> f.abs_dstpath('chen.py')
        Path('C:/Windows/System32/chen.py')
        >>> f.abs_dstpath('E:/')  # 原始文件必须存在，否则因为无法判断实际类型，目标路径可能会错
        Path('E:/cmd.exe')
        >>> f.abs_dstpath('D:/aabbccdd.txt')
        Path('D:/aabbccdd.txt')
        >>> f.abs_dstpath('D:/aabbccdd.txt/')  # 并不存在aabbccdd.txt这样的对象，但末尾有个/表明这是个目录
        Path('D:/aabbccdd.txt/cmd.exe')
        """
        if not root: root = self.dirname
        dst = Path(dst, suffix, root)
        # print(dst, dst.assume_dir)

        if self.is_dir() or (self.assume_dir and not self.is_file()):
            if dst.is_file():
                raise ValueError(f'{dst}是已存在的文件类型，不能对{self}目录执行指定操作')
            elif dst.assume_dir:  # 明确要把目录放到另一个目录下
                dst = dst / self.name
        else:  # 否则self是文件，或者不存在，均视为文件处理
            if dst.is_dir() or dst.assume_dir:
                dst = dst / self.name
            # 否则dst是文件，或者不存在的路径，均视为文件类型处理

        return dst

    def relative_path(self, ref_dir) -> str:
        r""" 当前路径，相对于ref_dir的路径位置

        >>> Path('C:/a/b/c.txt').relative_path('C:/a/')
        'b/c.txt'
        >>> Path('C:/a/b\\c.txt').relative_path('C:\\a/')
        'b/c.txt'
        """
        if not isinstance(ref_dir, Path):
            ref_dir = Path(ref_dir)
        # s1, s2 = str(self), str(ref_dir)
        # if s1.startswith(s2):
        #     s1 = s1[len(s2):]
        return os.path.relpath(str(self), str(ref_dir))

    def preprocess(self, if_exists='error', exclude=None):
        """ 这个功能要结合参数一起理解，不是简单的理解成“预处理”
        这个实际上是在做copy等操作前，如果目标文件已存在，需要预先删除等的预处理
        并返回判断，是否需要执行下一步操作

        有时候情况比较复杂，process无法满足需求时，可以用preprocess这个底层函数协助

        :param if_exists:
            'error': （默认）如果要替换的目标文件已经存在，则报错
            'overwrite', 'replace': 替换 （提前把已存在的目标文件删除）
            'skip', 'ignore': 忽略、不处理  （不用执行后续的功能）
            'backup': 备份后写入  （对原文件先做一个备份）
        :param exclude: 排除掉不分析的目录，用于有些重命名等自身可能操作自身的情况
            如果self是exclude这个路径，默认直接need_run=True
        """
        # 1 如果src和dst是同一个文件，因为重命名等特殊功能，可以直接执行，不用管提前存在目标文件的问题
        if exclude and self == Path(exclude):
            return True

        # 2
        need_run = True
        if self.exists():
            if if_exists == 'error':
                raise FileExistsError(f'目标文件已存在： {self}')
            elif if_exists in ('replace', 'overwrite'):  # None的话相当于replace，但是不会事先delete，可能会报错
                self.delete()
            elif if_exists in ('ignore', 'skip'):
                need_run = False
            elif if_exists == 'backup':
                self.backup(if_exists='backup')
                self.delete()
        return need_run

    def process(self, dst, func, if_exists='error'):
        r""" copy或move的本质底层实现

        文件复制等操作中src、dst不同组合下的效果：https://www.yuque.com/xlpr/pyxllib/mgwe19

        :param dst: 目标路径对象，注意如果使用相对路径，是相对于self的路径！
        :param func: 传入arg1和arg2参数，可以自定义
            默认分别是self和dst的fullpath
        :return : 返回dst
        """
        # 1 判断目标是有已存在，进行不同的指定规则处理
        dst0 = self.abs_dstpath(dst)

        # 2 执行特定功能
        if dst0.preprocess(if_exists, self):
            # 此时dst已是具体路径，哪怕是"目录"也可以按照"文件"对象理解，避免目录会重复生成，多层嵌套
            #   本来是 a --> C:/target/a ，避免 C:/target/a/a 的bug
            dst0.ensure_dir('file')

            if dst0 == self:  # 重命名自身的操作比较特别，需要打补丁
                func(self.fullpath, os.path.join(dst0.dirname, pathlib.Path(str(dst)).name))
                dst0 = self.abs_dstpath(dst)
            else:
                func(self.fullpath, dst0)

        return dst0

    def ensure_dir(self, pathtype=None):
        r""" 确保path中指定的dir都存在

        :param pathtype: 如果self.path的对象不存在，根据self.path的类型不同，有不同的处理方案
            'dir'：则包括自身也会创建一个空目录
            'file'： 只会创建到dirname所在目录，并不会创建自身的Path
            'None'：通过名称智能判断，如果能读取到suffix，则代表是Path类型，否则是dir类型
        :return:

        >> Path(r'D:\toweb\a\b.txt').ensure_dir()  # 只会创建toweb、a目录

        # a和一个叫b.txt的目录都会创建
        # 当然，如果b.txt是一个已经存在的文件对象，则该函数不会进行操作
        >> Path(r'D:\toweb\a\b.txt').ensure_dir(pathtype='dir')
        """
        if not self.exists():
            if not pathtype:
                pathtype = 'file' if (not self.assume_dir and self.suffix) else 'dir'
            dirname = self.fullpath if pathtype == 'dir' else self.dirname
            if not os.path.exists(dirname):
                os.makedirs(dirname)

    def copy(self, dst, if_exists='error'):
        """ 复制文件

        """
        if self.is_dir():
            return self.process(dst, shutil.copytree, if_exists)
        elif self.is_file():
            return self.process(dst, shutil.copy2, if_exists)

    def move(self, dst, if_exists='error'):
        """ 移动文件

        """
        if self.exists():
            return self.process(dst, shutil.move, if_exists)

    def rename(self, dst, if_exists='error'):
        r""" 文件重命名，或者也可以理解成文件移动

        :param dst: 如果使用相对目录，则是该类本身所在的目录作为工作环境
        :param if_exists:
            'error': 如果要替换的目标文件已经存在，则报错
            'replace': 替换
            'ignore': 忽略、不处理
            'backup': 备份后替换
        :return:
        """
        # rename是move的一种特殊情况
        return self.move(dst, if_exists)

    def delete(self):
        r""" 删除自身文件

        """
        if self.is_file():
            os.remove(self.fullpath)
        elif self.is_dir():
            shutil.rmtree(self.fullpath)
        # TODO 确保删除后再执行后续代码 但是一直觉得这样写很别扭
        while self.exists(): pass

    def backup(self, tail=None, if_exists='replace', move=False):
        r""" 对文件末尾添加时间戳备份，也可以使用自定义标记tail

        :param tail: 自定义添加后缀
            tail为None时，默认添加特定格式的时间戳
        :param if_exists: 备份的目标文件名存在时的处理方案
        :param move:
            是否删除原始文件

        # TODO：有个小bug，如果在不同时间实际都是相同一个文件，也会被不断反复备份
        #    如果想解决这个，就要读取目录下最近的备份文件对比内容了
        """
        # 1 判断自身文件是否存在
        if not self.exists():
            return None

        # 2 计算出新名称
        if not tail:
            tail = self.mtime.strftime(' %y%m%d-%H%M%S')  # 时间戳
        name, ext = os.path.splitext(self.fullpath)
        dst = name + tail + ext

        # 3 备份就是特殊的copy操作
        if move:
            return self.move(dst, if_exists)
        else:
            return self.copy(dst, if_exists)

    # 五、其他综合性功能

    def read(self, *, encoding=None, mode=None):
        """ 读取文件

        :param encoding: 文件编码
            默认None，则在需要使用encoding参数的场合，会使用self.encoding自动判断编码
        :param mode: 读取模式（例如 '.json'），默认从扩展名识别，也可以强制指定
            'b': 特殊标记，表示按二进制读取文件内容
        :return:
        """
        if self.is_file():  # 如果存在这样的文件，那就读取文件内容
            # 获得文件扩展名，并统一转成小写
            name, suffix = self.fullpath, self.suffix
            if not mode: mode = suffix
            mode = mode.lower()
            if mode == 'bytes':
                with open(name, 'rb') as f:
                    return f.read()
            elif mode == '.pkl':  # pickle库
                with open(name, 'rb') as f:
                    return pickle.load(f)
            elif mode == '.json':
                # 先读成字符串，再解析，会比rb鲁棒性更强，能自动过滤掉开头可能非正文特殊标记的字节
                if not encoding: encoding = self.encoding
                with open(name, 'r', encoding=encoding) as f:
                    return json.loads(f.read())
            elif mode == '.yaml':
                with open(name, 'r', encoding=encoding) as f:
                    return yaml.safe_load(f.read())
            elif mode in ('.jpg', '.jpeg', '.png', '.bmp', 'b'):
                # 二进制读取
                with open(name, 'rb') as fp:
                    return fp.read()
            else:
                with open(name, 'rb') as f:
                    bstr = f.read()
                if not encoding:
                    encoding = self.encoding
                    if not encoding:
                        raise ValueError(f'{self} 自动识别编码失败，请手动指定文件编码')
                s = bstr.decode(encoding=encoding, errors='ignore')
                if '\r' in s: s = s.replace('\r\n', '\n')  # 如果用\r\n作为换行符会有一些意外不好处理
                return s
        else:  # 非文件对象
            raise FileNotFoundError(f'{self} 文件不存在，无法读取。')

    def write(self, ob, *, encoding='utf8', if_exists='error', etag=False, mode=None):
        """ 保存为文件

        :param ob: 写入的内容
            如果要写txt文本文件且ob不是文本对象，只会进行简单的字符串化
        :param encoding: 强制写入的编码
        :param if_exists: 如果文件已存在，要进行的操作
        :param etag: 创建的文件，是否需要再进一步重命名为etag名称
        :param mode: 写入模式（例如 '.json'），默认从扩展名识别，也可以强制指定
        :return: 返回写入的文件名，这个主要是在写临时文件时有用
        """

        # 1 核心功能：将ob写入文件path
        if self.preprocess(if_exists):
            self.ensure_dir(pathtype='file')
            name, suffix = self.fullpath, self.suffix
            if not mode: mode = suffix
            mode = mode.lower()
            if mode == '.pkl':
                with open(name, 'wb') as f:
                    pickle.dump(ob, f)
            elif mode == '.json':
                with open(name, 'w', encoding=encoding) as f:
                    json.dump(ob, f, ensure_ascii=False, indent=2)
            elif mode == '.yaml':
                with open(name, 'w', encoding=encoding) as f:
                    yaml.dump(ob, f)
            elif isinstance(ob, bytes):
                with open(name, 'wb') as f:
                    f.write(ob)
            else:  # 其他类型认为是文本类型
                with open(name, 'w', errors='ignore', encoding=encoding) as f:
                    f.write(str(ob))

        # 2 如果使用了etag命名机制
        if etag:
            return self.rename(get_etag(self.fullpath) + self.suffix, if_exists='ignore')
        else:
            return self

    def explorer(self, proc='explorer'):
        """ 使用windows的explorer命令打开文件

        还有个类似的万能打开命令 start

        :param proc: 可以自定义要执行的主程序
        """
        subprocess.run([proc, self.fullpath])


def demo_path():
    """Path类的综合测试"""
    # 切换工作目录到临时文件夹
    os.chdir(Path.TEMP)

    p = Path('demo_path', root=Path.TEMP)
    p.delete()  # 如果存在先删除
    p.ensure_dir()  # 然后再创建一个空目录

    print(Path('demo_path', '.py'))
    # F:\work\CreatorTemp\demo_path.py

    print(Path('demo_path/', '.py'))
    # F:\work\CreatorTemp\demo_path\tmp65m8mc0b.py

    # 空字符串区别于None，会随机生成一个文件名
    print(Path('', root=Path.TEMP))
    # F:\work\CreatorTemp\tmpwp4g1692

    # 可以在随机名称基础上，再指定文件扩展名
    print(Path('', '.txt', root=Path.TEMP))
    # F:\work\CreatorTemp\tmpimusjtu1.txt


def demo_path_rename():
    # 初始化一个空的测试目录
    f = Path('temp/', root=Path.TEMP)
    f.delete()
    f.ensure_dir()

    # 建一个空文件
    f1 = f / 'a.txt'
    # Path('F:/work/CreatorTemp/temp/a.txt')
    with open(f1.fullpath, 'wb') as p: pass  # 写一个空文件

    f1.rename('A.tXt')  # 重命名
    # Path('F:/work/CreatorTemp/temp/a.txt')

    f1.rename('figs/b')  # 放到一个新的子目录里，并再次重命名
    # Path('F:/work/CreatorTemp/temp/figs/b')

    # TODO 把目录下的1 2 3 4 5重命名为5 4 3 2 1时要怎么搞？


class XlBytesIO(io.BytesIO):
    """ 自定义的字节流类，封装了struct_unpack操作

    https://www.yuque.com/xlpr/pyxllib/xlbytesio

    """

    def __init__(self, init_bytes):
        if isinstance(init_bytes, (Path, str)):
            # with open的作用：可以用.read循序读入，而不是我的Path.read一口气读入。
            #   这在只需要进行局部数据分析，f本身又非常大的时候很有用。
            #   但是我这里操作不太方便等原因，还是先全部读入BytesIO流了
            init_bytes = Path(init_bytes).read(mode='b')
        super().__init__(init_bytes)

    def unpack(self, fmt):
        return struct_unpack(self, fmt)

    def readtext(self, char_num, encoding='gbk', errors='ignore', code_length=2):
        """ 读取二进制流，将其解析未文本内容

        :param char_num: 字符数
        :param encoding: 所用编码，一般是gbk，因为如果是utf8是字符是变长的，不好操作
        :param errors: decode出现错误时的处理方式
        :param code_length: 每个字符占的长度
        :return: 文本内容
        """
        return self.read(code_length * char_num).decode(encoding, errors)


if __name__ == '__main__':
    demo_path_rename()
