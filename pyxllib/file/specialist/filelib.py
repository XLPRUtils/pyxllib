#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 20:37

from pyxllib.prog.pupil import check_install_package

check_install_package('filetype')

from typing import Callable, Any
import io
import json
import os
import pathlib
import pickle
import re
import shutil
import subprocess
import tempfile
import ujson
from collections import defaultdict, Counter
import math
from itertools import islice
import datetime

# import chardet
import charset_normalizer
import qiniu
import requests
import yaml
import humanfriendly
from more_itertools import chunked
import filetype
from tqdm import tqdm

from pyxllib.prog.newbie import round_int, human_readable_size
from pyxllib.prog.pupil import is_url, is_file, DictTool
from pyxllib.algo.pupil import Groups
from pyxllib.file.pupil import struct_unpack, gen_file_filter


def __1_judge():
    pass


def is_url_connect(url, timeout=5):
    try:
        _ = requests.head(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        pass
    return False


def __2_qiniu():
    pass


def get_etag(arg):
    """ 七牛原有etag功能基础上做封装

    :param arg: 支持bytes二进制、文件、url地址

    只跟文件内容有关，跟文件创建、修改日期没关系
    如果读取文件后再处理etag，要尤其小心 '\r\n' 的问题！
    文件里如果是\r\n，我的File.read会变成\n，所以按文件取etag和read的内容算etag会不一样。
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
    """ 字母、数字和-、_共64种字符构成的长度28的字符串 """
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

    f = File('1.tex', tempfile.gettempdir()).write(s, if_exists='replace').to_str()
    print(get_etag(f))
    # FkAD2McB6ugxTiniE8ebhlNHdHh9


def __3_chardet():
    pass


def get_encoding(data,
                 cp_isolation=('utf_8', 'gbk', 'gb18030', 'utf_16'),
                 preemptive_behaviour=True,
                 explain=False):
    """ 从字节串中检测编码类型

    :param bytes data: 要检测的字节串
    :param List[str] cp_isolation: 指定要检测的字符集编码类型列表，只检测该列表中指定的编码类型，而不是所有可能的编码类型，默认为 None
        注意charset_normalizer是无法识别utf-8-sig的情况的。但要获得解析后的文本内容，其实也不用通过get_encoding来中转。
    :param bool preemptive_behaviour: 指定预处理行为，如果设置为 True，将在检测之前对数据进行预处理，例如去除 BOM、转换大小写等操作，默认为 True
    :param bool explain: 指定是否打印出检测过程的详细信息，如果设置为 True，将打印出每个 chunk 的检测结果和置信度，默认为 False
    :return str: 检测到的编码类型，返回字符串表示
    """
    result = charset_normalizer.from_bytes(data,
                                           cp_isolation=cp_isolation,
                                           preemptive_behaviour=preemptive_behaviour,
                                           explain=explain)
    best_match = result.best()
    if best_match:
        return best_match.encoding
    else:  # 注意，这个实现是有可能会找不到编码的，此时默认返回None
        return


def __4_file():
    """
    路径、文件、目录相关操作功能

    主要是为了提供readfile、wrritefile函数
    与普通的读写文件相比，有以下优点：
    1、智能识别pkl等特殊格式文件的处理
    2、智能处理编码
    3、目录不存在自动创建
    4、自动备份旧文件，而不是强制覆盖写入

    其他相关文件处理组件：isfile、get_encoding、ensure_folders
    """


class PathBase:
    """ File和Dir共有的操作逻辑功能 """
    __slots__ = ('_path',)

    @classmethod
    def abspath(cls, path=None, root=None, *, suffix=None, resolve=True) -> pathlib.Path:
        r""" 根据各种不同的组合参数信息，推导出具体的路径位置

        :param path: 主要的参数，如果后面的参数有矛盾，以path为最高参考标准
            ...，可以输入Ellipsis对象，效果跟None是不一样的，意思是显式地指明要生成一个随机名称的文件
            三个参数全空时，则返回当前目录
        :param suffix:
            以 '.' 指明的扩展名，会强制替换
            否则，只作为参考扩展名，只在原有path没有指明的时候才添加
        :param root: 未输入的时候，则为当前工作目录

        >>> os.chdir("C:/Users")

        # 未输入任何参数，则返回当前工作目录
        >>> PathBase.abspath()
        WindowsPath('C:/Users')

        # 基本的指定功能
        >>> PathBase.abspath('a')
        WindowsPath('C:/Users/a')

        # 额外指定父目录
        >>> PathBase.abspath('a/b', 'D:')
        WindowsPath('D:/a/b')

        # 设置扩展名
        >>> PathBase.abspath('a', suffix='.txt')
        WindowsPath('C:/Users/a.txt')

        # 扩展名的高级用法
        >>> PathBase.abspath('F:/work/a.txt', suffix='py')  # 参考后缀不修改
        WindowsPath('F:/work/a.txt')
        >>> PathBase.abspath('F:/work/a.txt', suffix='.py')  # 强制后缀会修改
        WindowsPath('F:/work/a.py')
        >>> PathBase.abspath('F:/work/a.txt', suffix='')  # 删除后缀
        WindowsPath('F:/work/a')

        # 在临时目录下，新建一个.tex的随机名称文件
        >> PathBase.abspath(..., tempfile.gettempdir(), suffix='.tex')
        WindowsPath('D:/Temp/tmp_sey0yeg.tex')
        """
        # 1 判断参考目录
        if root is None:
            root = os.getcwd()
        else:
            root = str(root)

        # 2 判断主体文件名 path
        if path is None:
            return pathlib.Path(root)
        elif path is Ellipsis:
            path = tempfile.mktemp(dir=root)
        else:
            path = os.path.join(root, str(path))

        # 3 补充suffix
        if suffix is not None:
            # 判断后缀
            li = os.path.splitext(path)
            if suffix == '' or suffix[0] == '.':
                path = li[0] + suffix
            elif li[1] == '':
                path = li[0] + '.' + suffix

        if resolve:
            return pathlib.Path(path).resolve()
        else:
            return pathlib.Path(path)

    def __bool__(self):
        r""" 判断文件、文件夹是否存在

        重置WindowsPath的bool逻辑，返回值变成存在True，不存在为False

        >>> bool(File('C:/Windows/System32/cmd.exe'))
        True
        >>> bool(File('C:/Windows/System32/cmdcmd.exe'))
        False
        """
        return self._path.exists()

    def __repr__(self):
        s = self._path.__repr__()
        if s.startswith('WindowsPath'):
            s = self.__class__.__name__ + s[11:]
        elif s.startswith('PosixPath'):
            s = self.__class__.__name__ + s[9:]
        return s

    def __str__(self):
        return str(self._path).replace('\\', '/')

    def to_str(self):
        return self.__str__()

    def resolve(self):
        return self._path.resolve()

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
        >>> File('D:/pycode/a.txt').name
        'a.txt'
        >>> File('D:/pycode/code4101py').name
        'code4101py'
        >>> File('D:/pycode/.gitignore').name
        '.gitignore'
        """
        return self._path.name

    @name.setter
    def name(self, value):
        raise NotImplementedError

    @property
    def parent(self):
        r"""
        >>> File('D:/pycode/code4101py').parent
        WindowsPath('D:/pycode')
        """
        return self._path.parent

    @property
    def dirname(self) -> str:
        r"""
        >>> File('D:/pycode/code4101py').dirname
        'D:\\pycode'
        >>> File(r'D:\toweb\a').dirname
        'D:\\toweb'
        """
        return str(self.parent)

    @property
    def parts(self) -> tuple:
        r"""
        >>> File('D:/pycode/code4101py').parts
        ('D:\\', 'pycode', 'code4101py')
        """
        return self._path.parts

    @property
    def backup_time(self):
        r""" 返回文件的备份时间戳，如果并不是备份文件，则返回空字符串

        备份文件都遵循特定的命名规范
        如果是文件，是：'chePre 171020-153959.tex'
        如果是目录，是：'figs 171020-153959'
        通过后缀分析，可以判断这是不是一个备份文件

        >>> File('chePre 171020-153959.tex').backup_time
        '171020-153959'
        >>> File('figs 171020-153959').backup_time
        '171020-153959'
        >>> File('figs 171020').backup_time
        ''
        """
        name = self.stem
        if len(name) < 14:
            return ''
        g = re.match(r'(\d{6}-\d{6})', name[-13:])
        return g.group(1) if g else ''

    @property
    def mtime(self):
        r""" 文件的最近修改时间

        >> PathBase(r"C:\pycode\code4101py").mtime
        datetime.datetime(2021, 9, 6, 17, 28, 18, 960713)
        """
        from datetime import datetime
        return datetime.fromtimestamp(os.stat(str(self)).st_mtime)

    @property
    def ctime(self):
        r""" 文件的创建时间

        >> Path(r"C:\pycode\code4101py").ctime
        2018-05-25 10:46:37
        """
        from datetime import datetime
        # 注意：st_ctime是平台相关的值，在windows是创建时间，但在Unix是metadate最近修改时间
        return datetime.fromtimestamp(os.stat(str(self)).st_ctime)

    def relpath(self, ref_dir) -> str:
        r""" 当前路径，相对于ref_dir的路径位置

        >>> File('C:/a/b/c.txt').relpath('C:/a/')
        'b/c.txt'
        >>> File('C:/a/b\\c.txt').relpath('C:\\a/')
        'b/c.txt'

        >> File('C:/a/b/c.txt').relpath('D:/')  # ValueError
        """
        return os.path.relpath(str(self), str(ref_dir)).replace('\\', '/')

    def exist_preprcs(self, if_exists=None):
        """ 这个实际上是在做copy等操作前，如果目标文件已存在，需要预先删除等的预处理
        并返回判断，是否需要执行下一步操作

        有时候情况比较复杂，process无法满足需求时，可以用exist_preprcs这个底层函数协助

        :param if_exists:
            None: 不做任何处理，直接运行，依赖于功能本身是否有覆盖写入机制
            'error': 如果要替换的目标文件已经存在，则报错
            'replace': 把存在的文件先删除
                本来是叫'delete'更准确的，但是考虑用户理解，
                    一般都是用在文件替换场合，叫成'delete'会非常怪异，带来不必要的困扰、误解
                所以还是决定叫'replace'
            'skip': 不执行后续功能
            'backup': 先做备份  （对原文件先做一个备份）
        """
        need_run = True
        if self.exists():
            if if_exists is None:
                return need_run
            elif if_exists == 'error':
                raise FileExistsError(f'目标文件已存在： {self}')
            elif if_exists == 'replace':
                self.delete()
            elif if_exists == 'skip':
                need_run = False
            elif if_exists == 'backup':
                self.backup(move=True)
            else:
                raise ValueError(f'{if_exists}')
        return need_run

    def copy(self, *args, **kwargs):
        raise NotImplementedError

    def delete(self):
        raise NotImplementedError

    def absdst(self, dst):
        raise NotImplementedError

    def backup(self, tail=None, if_exists='replace', move=False):
        r""" 对文件末尾添加时间戳备份，也可以使用自定义标记tail

        :param tail: 自定义添加后缀
            tail为None时，默认添加特定格式的时间戳
        :param if_exists: 备份的目标文件名存在时的处理方案
            这个概率非常小，真遇到，先把已存在的删掉，重新写入一个是可以接受的
        :param move: 是否删除原始文件

        # TODO：有个小bug，如果在不同时间实际都是相同一个文件，也会被不断反复备份
        #    如果想解决这个，就要读取目录下最近的备份文件对比内容了
        """
        # 1 判断自身文件是否存在
        if not self:
            return None

        # 2 计算出新名称
        if not tail:
            tail = self.mtime.strftime(' %y%m%d-%H%M%S')  # 时间戳
        name, ext = os.path.splitext(str(self))
        dst = name + tail + ext

        # 3 备份就是特殊的copy操作
        if move:
            return self.move(dst, if_exists)
        else:
            return self.copy(dst, if_exists)

    def move(self, dst, if_exists=None):
        """ 移动文件
        """
        if self:
            return self.process(dst, shutil.move, if_exists)

    def process(self, dst, func, if_exists=None):
        r""" copy或move的本质底层实现

        :param dst: 目标路径对象
        :param func: 传入arg1和arg2参数，可以自定义
            默认分别是self和dst的字符串
        :return : 返回dst
        """
        # 1 判断目标是否已存在，进行不同的指定规则处理
        dst_ = self.absdst(dst)

        # 2 执行特定功能
        if self == dst_:
            # 如果文件是自身的话，并不算exists，可以直接run，不用exist_preprcs
            dst_.ensure_parent()
            func(str(self), str(dst_))
            dst_._path = dst_._path.resolve()  # 但是需要重新解析一次dst_._path，避免可能有重命名等大小写变化
        elif dst_.exist_preprcs(if_exists):
            dst_.ensure_parent()
            func(str(self), str(dst_))

        return dst_

    def explorer(self, proc='explorer'):
        """ 使用windows的explorer命令打开文件

        还有个类似的万能打开命令 start

        :param proc: 可以自定义要执行的主程序
        """
        subprocess.run([proc, str(self)])

    def ensure_parent(self):
        r""" 确保父目录存在
        """
        p = self.parent
        if not p.exists():
            os.makedirs(str(p))

    def __getattr__(self, item):
        return getattr(self._path, item)

    def start(self):
        os.startfile(str(self))


class File(PathBase):
    r""" 通用文件处理类，大部分基础功能是从pathlib.Path衍生过来的

    document: https://www.yuque.com/xlpr/python/pyxllib.debug.path
    """
    __slots__ = ('_path',)

    # 一、基础功能

    def __init__(self, path, root=None, *, suffix=None, check=True):
        r""" 初始化参数含义详见 PathBase.abspath 函数解释

        :param path: 只传入一个File、pathlib.Path对象，可以提高初始化速度，不会进行多余的解析判断
            如果要安全保守一点，可以传入str类型的path
        :param root: 父目录
        :param check: 在某些特殊场合、内部开发中，可以确保传参一定不会出错，在上游已经有了严谨的检查
            此时可以设置check=False，提高初始化速度

        注意！如果使用了符号链接（软链接），则路径是会解析转向实际位置的！例如
        >> File('D:/pycode/code4101py')
        File('D:/slns/pycode/code4101py')
        """
        self._path = None

        # 1 快速初始化
        if root is None and suffix is None:
            if isinstance(path, File):
                self._path = path._path
                return  # 直接完成初始化过程
            elif isinstance(path, pathlib.Path):
                self._path = path

        # 2 普通初始化
        if self._path is None:
            self._path = self.abspath(path, root, suffix=suffix)

        # 3 检查
        if check:
            if not self._path:
                raise ValueError(f'无效路径 {self._path}')
            elif self._path.is_dir():
                raise ValueError(f'不能用目录初始化一个File对象 {self._path}')

    @classmethod
    def safe_init(cls, path, root=None, *, suffix=None):
        """ 如果失败不raise，而是返回None的初始化方式 """
        try:
            f = File(path, root, suffix=suffix)
            f._path.is_file()  # 有些问题上一步不一定测的出来，要再补一个测试
            return f
        except (ValueError, TypeError, OSError, PermissionError):
            # ValueError：文件名过长，代表输入很可能是一段文本，根本不是路径
            # TypeError：不是str等正常的参数
            # OSError：非法路径名，例如有 *? 等
            # PermissionError: linux上访问无权限、不存在的路径
            return None

    # 二、获取、修改路径中部分值的功能

    def with_dirname(self, value):
        return File(self.name, value)

    @property
    def stem(self) -> str:
        r"""
        >>> File('D:/pycode/code4101py/ckz.py').stem
        'ckz'
        >>> File('D:/pycode/.gitignore').stem  # os.path.splitext也是这种算法
        '.gitignore'
        >>> File('D:/pycode/.123.45.6').stem
        '.123.45'
        """
        return self._path.stem

    def with_stem(self, stem):
        """
        注意不能用@stem.setter来做
            如果setter要rename，那if_exists参数怎么控制？
            如果setter不要rename，那和用with_stem实现有什么区别？
        """
        return File(stem, self.parent, suffix=self.suffix)

    def with_name(self, name):
        return File(name, self.parent)

    @property
    def suffix(self) -> str:
        r"""
        >>> File('D:/pycode/code4101py/ckz.py').suffix
        '.py'
        >>> File('D:/pycode/code4101py').suffix
        ''
        >>> File('D:/pycode/code4101py/ckz.').suffix
        ''
        >>> File('D:/pycode/code4101py/ckz.123.456').suffix
        '.456'
        >>> File('D:/pycode/code4101py/ckz.123..456').suffix
        '.456'
        >>> File('D:/pycode/.gitignore').suffix
        ''
        """
        return self._path.suffix

    def with_suffix(self, suffix):
        r""" 指向同目录下后缀为suffix的文件

        >>> File('a.txt').with_suffix('.py').name  # 强制替换
        'a.py'
        >>> File('a.txt').with_suffix('py').name  # 参考替换
        'a.txt'
        >>> File('a.txt').with_suffix('').name  # 删除
        'a'
        """
        return File(self.abspath(self._path, suffix=suffix))

    # 三、获取文件相关属性值功能

    # @property
    # def encoding(self):
    #     """ 文件的编码
    #
    #     非文件、不存在时返回 None
    #     """
    #     if self:
    #         return get_encoding(str(self))

    @property
    def size(self) -> int:
        """ 计算文件大小
        """
        if self.exists():
            total_size = os.path.getsize(str(self))
        else:
            total_size = 0
        return total_size

    def size2(self) -> int:
        """ 220102周日17:23

        size有点bug，临时写个函数接口
        这个bug有点莫名其妙，搞不定
        """
        total_size = os.path.getsize(str(self))
        return total_size

    # 四、文件操作功能

    def absdst(self, dst):
        """ 在copy、move等中，给了个"模糊"的目标位置dst，智能推导出实际file、dir绝对路径
        """
        from pyxllib.file.specialist.dirlib import Dir
        dst_ = self.abspath(dst)
        if isinstance(dst, Dir) or (isinstance(dst, str) and dst[-1] in ('\\', '/')) or dst_.is_dir():
            dst_ = File(self.name, dst_)
        else:
            dst_ = File(dst_)
        return dst_

    def copy(self, dst, if_exists=None):
        """ 复制文件
        """
        return self.process(dst, shutil.copy2, if_exists)

    def rename(self, dst, if_exists=None):
        r""" 文件重命名，或者也可以理解成文件移动
        该接口和move的核心区别：move的dst是相对工作目录，而rename则是相对self.parent路径
        """
        # rename是move的一种特殊情况
        return self.move(File(dst, self.parent), if_exists)

    def delete(self):
        r""" 删除自身文件
        """
        if self.is_file():
            os.remove(str(self))

    # 五、其他综合性功能

    def read(self, *, encoding=None, mode=None):
        """ 读取文件

        :param encoding: 文件编码
            默认None，则在需要使用encoding参数的场合，会使用self.encoding自动判断编码
        :param mode: 读取模式（例如 '.json'），默认从扩展名识别，也可以强制指定
            'b': 特殊标记，表示按二进制读取文件内容
        :return:
        """
        if self:  # 如果存在这样的文件，那就读取文件内容
            # 获得文件扩展名，并统一转成小写
            name, suffix = str(self), self.suffix
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
                with open(name, 'rb') as f:
                    bstr = f.read()
                    if not encoding: encoding = get_encoding(bstr)
                try:
                    return ujson.loads(bstr.decode(encoding=encoding))
                except ValueError:  # ujson会有些不太标准的情况处理不了
                    return json.loads(bstr.decode(encoding=encoding))
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
                    encoding = get_encoding(bstr)
                    if not encoding:
                        raise ValueError(f'{self} 自动识别编码失败，请手动指定文件编码')
                s = bstr.decode(encoding=encoding, errors='ignore')
                if '\r' in s: s = s.replace('\r\n', '\n')  # 如果用\r\n作为换行符会有一些意外不好处理
                return s
        else:  # 非文件对象
            raise FileNotFoundError(f'{self} 文件不存在，无法读取。')

    def write(self, ob, *, encoding='utf8', if_exists=None, mode=None, **kwargs):
        """ 保存为文件

        :param ob: 写入的内容
            如果要写txt文本文件且ob不是文本对象，只会进行简单的字符串化
        :param encoding: 强制写入的编码
            如果原文件存在且有编码，则使用原文件的编码
            如果没有，则默认使用utf8
            当然，其实有些格式是用不到编码信息的~~例如pkl文件
        :param if_exists: 如果文件已存在，要进行的操作
        :param mode: 写入模式（例如 '.json'），默认从扩展名识别，也可以强制指定
        :param kwargs:
            写入json格式的时候
                ensure_ascii: json.dump默认是True，但是我这里默认值改成了False
                    改成False可以支持在json直接显示中文明文
                indent: json.dump是None，我这里默认值遵循json.dump
                    我原来是2，让文件结构更清晰、更加易读
        :return: 返回写入的文件名，这个主要是在写临时文件时有用
        """

        # # 将ob写入文件path
        # def get_enc():
        #     # 编码在需要的时候才获取分析，减少不必要的运算开销
        #     # 所以封装一个函数接口，需要的时候再计算
        #     if encoding is None:
        #         # return self.encoding or 'utf8'
        #     return encoding

        if self.exist_preprcs(if_exists):
            self.ensure_parent()
            name, suffix = str(self), self.suffix
            if not mode: mode = suffix
            mode = mode.lower()
            if mode == '.pkl':
                with open(name, 'wb') as f:
                    pickle.dump(ob, f)
            elif mode == '.json':
                with open(name, 'w', encoding=encoding) as f:
                    DictTool.ior(kwargs, {'ensure_ascii': False})
                    json.dump(ob, f, **kwargs)
            elif mode == '.yaml':
                with open(name, 'w', encoding=encoding) as f:
                    yaml.dump(ob, f)
            elif isinstance(ob, bytes):
                with open(name, 'wb') as f:
                    f.write(ob)
            else:  # 其他类型认为是文本类型
                with open(name, 'w', errors='ignore', encoding=encoding) as f:
                    f.write(str(ob))

        return self

    def unpack(self, dst_dir=None):
        """ 解压缩

        :param dst_dir: 如果没有输入，默认会套一层压缩文件原名的目录里

        TODO 本来是想，如果没有传dst_dir，则类似Bandizip的自动解压机制，解压到当前文件夹
            但是不太好实现
        """
        p = str(self)
        if not dst_dir:
            dst_dir = os.path.splitext(p)[0]
        shutil.unpack_archive(p, str(dst_dir))

    def __eq__(self, other):
        return str(self) == str(other)

    def exists(self):
        return self._path.exists()


def make_filter():
    """ 从filesmatch """


class XlPath(type(pathlib.Path())):

    @classmethod
    def desktop(cls):
        if os.environ.get('Desktop', None):  # 如果修改了win10默认的桌面路径，需要在环境变量添加一个正确的Desktop路径值
            desktop = os.environ['Desktop']
        else:
            desktop = os.path.join(pathlib.Path.home(), 'Desktop')  # 这个不一定准，桌面是有可能被移到D盘等的
        return cls(desktop)

    @classmethod
    def userdir(cls):
        from os.path import expanduser
        return cls(expanduser("~"))

    @classmethod
    def tempdir(cls):
        return cls(tempfile.gettempdir())

    @classmethod
    def create_tempdir_path(cls, dir=None):
        if dir is None:
            dir = tempfile.gettempdir()
        dst = cls(tempfile.mktemp(dir=dir))
        return dst

    @classmethod
    def create_tempfile_path(cls, suffix="", dir=None):
        if dir is None:
            dir = tempfile.gettempdir()
        return cls(tempfile.mktemp(suffix=suffix, dir=dir))

    tempfile = create_tempfile_path

    @classmethod
    def init(cls, path, root=None, *, suffix=None):
        """ 仿照原来File的初始化接口形式 """
        p = XlPath(path)
        if root:
            p = XlPath(root) / p

        if suffix:
            if suffix[0] == '.':  # 使用.时强制修改后缀
                p = p.with_suffix(suffix)
            elif not p.suffix:
                p = p.with_suffix('.' + suffix)

        return p

    @classmethod
    def safe_init(cls, arg_in):
        """ 输入任意类型的_in，会用比较安全的机制，判断其是否为一个有效的路径格式并初始化
        初始化失败则返回None
        """
        try:
            p = XlPath(str(arg_in))
            p.is_file()  # 有些问题上一步不一定测的出来，要再补一个测试。具体存不存在是不是文件并不重要，而是使用这个能检查出问题。
            return p
        except (ValueError, TypeError, OSError, PermissionError):
            # ValueError：文件名过长，代表输入很可能是一段文本，根本不是路径
            # TypeError：不是str等正常的参数
            # OSError：非法路径名，例如有 *? 等
            # PermissionError: linux上访问无权限、不存在的路径
            return None

    def start(self, *args, **kwargs):
        """ 使用关联的程序打开p，类似于双击的效果

        这就像在 Windows 资源管理器中双击文件、文件夹
        """
        os.startfile(self, *args, **kwargs)

    def exists_type(self):
        """
        不存在返回0，文件返回1，目录返回-1
        这样任意两个文件类型编号相乘，负数就代表不匹配
        """
        if self.is_file():
            return 1
        elif self.is_dir():
            return -1
        else:
            return 0

    def mtime(self):
        # windows会带小数，linux使用%Ts只有整数部分。
        # 这里不用四舍五入，取整数部分就是对应的。
        return int(os.stat(self).st_mtime)

    def size(self, *, human_readable=False):
        """ 获取文件/目录的大小 """
        if self.is_file():
            sz = os.path.getsize(self)
        elif self.is_dir():
            sz = sum([os.path.getsize(p) for p in self.rglob('*') if p.is_file()])
        else:
            sz = 0

        if human_readable:
            return humanfriendly.format_size(sz, binary=True)
        else:
            return sz

    def sub_rel_paths(self, mtime=False):
        """ 返回self目录下所有含递归的文件、目录，存储相对路径，as_posix
        当带有mtime参数时，会返回字典，并附带返回mtime的值

        主要用于 scp 同步目录数据时，对比目录下文件情况
        """

        if mtime:
            res = {}
            for p in self.glob('**/*'):
                res[p.relative_to(self).as_posix()] = p.mtime()
        else:
            res = set()
            for p in self.glob('**/*'):
                res.add(p.relative_to(self).as_posix())
        return res

    def relpath(self, ref_dir) -> str:
        r""" 当前路径，相对于ref_dir的路径位置

        >>> File('C:/a/b/c.txt').relpath('C:/a/')
        'b/c.txt'
        >>> File('C:/a/b\\c.txt').relpath('C:\\a/')
        'b/c.txt'

        >> File('C:/a/b/c.txt').relpath('D:/')  # ValueError
        """
        return XlPath(os.path.relpath(self, str(ref_dir)))

    def as_windows_path(self):
        """ 返回windows风格的路径，即使用\\分隔符 """
        return self.as_posix().replace('/', '\\')

    def __contains__(self, item):
        """ 判断item的路径是否是在self里的，不考虑item是目录、文件，还是不存在文件的路径
        以前缀判断为准，有需要的话请自行展开resolve、expanduser再判断
        """
        if not self.is_file():
            # 根据操作系统判断路径分隔符
            separator = '\\' if os.name == 'nt' else '/'

            # 获取路径对象的字符串形式
            abs_path_str = str(self) + separator
            item_str = str(XlPath(item))

            # 判断路径字符串是否包含相对路径字符串
            return item_str.startswith(abs_path_str) or abs_path_str == item_str

    def get_total_lines(self, encoding='utf-8', skip_blank=False):
        """ 统计文件的行数（注意会统计空行，所以在某些场合可能与预期理解的条目数不太一致）

        :param str encoding: 文件编码，默认为'utf-8'
        :param bool skip_blank: 是否跳过空白行，默认为True
        :return: 文件的行数
        """
        line_count = 0
        with open(self, 'r', encoding=encoding) as file:
            for line in file:
                if skip_blank and not line.strip():  # 跳过空白行
                    continue
                line_count += 1
        return line_count

    def yield_line(self, start=0, end=None, step=1, batch_size=None, encoding='utf-8'):
        """ 返回指定区间的文件行

        :param int start: 起始行，默认为0
        :param int end: 结束行，默认为None（读取到文件末尾）
        :param int step: 步长，默认为1
        :param int batch_size: 每批返回的行数，如果为None，则逐行返回
        """
        total_lines = None  # 使用局部变量缓存总行数
        # 处理负索引
        if start < 0 or (end is not None and end < 0):
            total_lines = total_lines or self.get_total_lines()
            if start < 0:
                start = total_lines + start
            if end is not None and end < 0:
                end = total_lines + end

        with open(self, 'r', encoding=encoding) as file:
            iterator = islice(file, start, end, step)
            while True:
                batch = list(islice(iterator, batch_size))
                if not batch:
                    break
                batch = [line.rstrip('\n') for line in batch]  # 删除每行末尾的换行符
                if batch_size is None:
                    yield from batch
                else:
                    yield batch

    def split_to_dir(self, lines_per_file, dst_dir=None, encoding='utf-8',
                     filename_template="_{index}{suffix}"):
        """ 将文件按行拆分到多个子文件中

        :param int lines_per_file: 打算拆分的每个新文件的行数
        :param str dst_dir: 目标目录，未输入的时候，输出到同stem名的目录下
        :param str filename_template: 文件名模板，可以包含 {stem}, {index} 和 {suffix} 占位符
        :return list: 拆分的文件路径列表
            拆分后文件名类似如下： 01.jsonl, 02.jsonl, ...
        """
        # 1 检查输入参数
        if dst_dir is None:
            # 如果未提供目标目录，则拆分的文件保存到当前工作目录
            dst_dir = self.parent / f"{self.stem}"
        else:
            # 如果提供了目标目录，将拆分的文件保存到目标目录
            dst_dir = XlPath(dst_dir)

        if dst_dir.is_dir():
            raise FileExistsError(f"目标目录已存在，若确定要重置目录，请先删除目录：{dst_dir}")

        dst_dir.mkdir(parents=True, exist_ok=True)

        # 2 拆分文件
        split_files = []  # 用于保存拆分的文件路径
        outfile = None
        filename_format = "{:04d}"
        outfile_index = 0
        line_counter = 0
        suffix = self.suffix

        with open(self, 'r', encoding=encoding) as f:
            for line in f:
                if line_counter % lines_per_file == 0:
                    if outfile is not None:
                        outfile.close()
                    outfile_path = dst_dir / f"{self.stem}_{filename_format.format(outfile_index)}{suffix}"
                    outfile = open(outfile_path, 'w', encoding='utf-8')
                    split_files.append(outfile_path)  # 先占位，后面再填充
                    outfile_index += 1
                outfile.write(line)
                line_counter += 1

        if outfile is not None:
            outfile.close()

        # 3 重新设置文件名的对齐宽度
        new_filename_format = "{:0" + str(len(str(len(split_files)))) + "d}"
        for i, old_file in enumerate(split_files):
            new_name = dst_dir / filename_template.format(stem=self.stem,
                                                          index=new_filename_format.format(i),
                                                          suffix=suffix)
            os.rename(old_file, new_name)
            split_files[i] = new_name

        # 返回拆分的文件路径列表
        return split_files

    def merge_from_files(self, files,
                         ignore_empty_lines_between_files=False,
                         encoding='utf-8'):
        """ 将多个文件合并到一个文件中

        :param list files: 要合并的文件列表
        :param bool ignore_empty_lines_between_files: 是否忽略文件间的空行
        :param str encoding: 文件编码，默认为'utf-8'
        :return XlPath: 合并后的文件路径
        """
        # 合并文件
        prev_line_end_with_newline = True  # 记录上一次text的最后一个字符是否为'\n'
        with open(self, 'w', encoding=encoding) as outfile:
            for i, file in enumerate(files):
                file = XlPath(file)
                text = file.read_text(encoding=encoding)
                if ignore_empty_lines_between_files:
                    text = text.rstrip('\n')
                if i > 0 and not prev_line_end_with_newline and text != '':
                    outfile.write('\n')
                outfile.write(text)
                prev_line_end_with_newline = text.endswith('\n')

    def merge_from_dir(self, src_dir, filename_template="_{index}{suffix}", encoding='utf-8'):
        """ 将目录中的多个文件合并到一个文件中

        :param str src_dir: 要合并的文件所在的目录
        :param str filename_template: 文件名模板，可以包含 {stem}, {index} 和 {suffix} 占位符
        :param str encoding: 文件编码，默认为'utf-8'
        :return XlPath: 合并后的文件路径
        """
        src_dir = XlPath(src_dir)
        stem = src_dir.name

        pattern = filename_template.format(stem=stem, index="(\d+)", suffix=".*")
        files = [file for file in src_dir.iterdir() if re.match(pattern, file.name)]  # 获取目录中符合模式的文件

        self.merge_from_files(files, ignore_empty_lines_between_files=True, encoding=encoding)

    def __1_read_write(self):
        """ 参考标准库的
        read_bytes、read_text
        write_bytes、write_text
        """
        pass

    def read_text(self, encoding='utf8', errors='strict', return_mode: bool = False):
        """
        :param encoding: 效率拷贝，默认是设成utf8，但也可以设成None变成自动识别编码
        """
        if not encoding:
            result = charset_normalizer.from_path(self, cp_isolation=('utf_8', 'gbk', 'utf_16'))
            best_match = result.best()
            s = str(best_match)
            encoding = best_match.encoding
        else:
            with open(self, 'r', encoding=encoding) as f:
                s = f.read()

        # 如果用\r\n作为换行符会有一些意外不好处理
        if '\r' in s:
            s = s.replace('\r\n', '\n')

        if return_mode:
            return s, encoding
        else:
            return s

    def readlines_batch(self, batch_size, *, encoding='utf8'):
        """ 将文本行打包，每次返回一个批次多行数据

        python的io.IOBase.readlines有个hint参数，不是预期的读取几行的功能，所以这里重点是扩展了一个readlines的功能

        :param batch_size: 默认每次获取一行内容，可以设参数，每次返回多行内容
            如果遍历每次只获取一行，一般不用这个接口，直接对open得到的文件句柄f操作就行了

        :return: list[str]
            注意返回的每行str，末尾都带'\n'
            但最后一行视情况可能有\n，可能没有\n

        注，开发指南：不然扩展支持batch_size=-1获取所有数据
        """
        f = open(self, 'r', encoding=encoding)
        return chunked(f, batch_size)

    def write_text(self, data, encoding='utf8', errors=None, newline=None):
        with open(self, 'w', encoding=encoding, errors=errors, newline=newline) as f:
            return f.write(data)

    def write_text_unix(self, data, encoding='utf8', errors=None, newline='\n'):
        with open(self, 'w', encoding=encoding, errors=errors, newline=newline) as f:
            return f.write(data)

    def read_pkl(self):
        with open(self, 'rb') as f:
            return pickle.load(f)

    def write_pkl(self, data):
        with open(self, 'wb') as f:
            pickle.dump(data, f)

    def read_json(self, encoding='utf8', *, errors='strict', return_mode: bool = False):
        """

        Args:
            encoding: 可以主动指定编码，否则默认会自动识别编码
            return_mode: 默认只返回读取的数据
                开启后，得到更丰富的返回信息: data, encoding
                    该功能常用在需要自动识别编码，重写回文件时使用相同的编码格式
        Returns:

        """
        s, encoding = self.read_text(encoding=encoding, errors=errors, return_mode=True)
        try:
            data = ujson.loads(s)
        except ValueError:  # ujson会有些不太标准的情况处理不了
            data = json.loads(s)

        if return_mode:
            return data, encoding
        else:
            return data

    def write_json(self, data, encoding='utf8', **kwargs):
        with open(self, 'w', encoding=encoding) as f:
            DictTool.ior(kwargs, {'ensure_ascii': False})
            json.dump(data, f, **kwargs)

    def read_jsonl(self, encoding='utf8', max_items=None, *,
                   errors='strict', return_mode: bool = False):
        """ 从文件中读取JSONL格式的数据

        :param str encoding: 文件编码格式，默认为utf8
        :param str errors: 读取文件时的错误处理方式，默认为strict
        :param bool return_mode: 是否返回文件编码格式，默认为False
        :param int max_items: 限制读取的条目数，默认为None，表示读取所有条目
        :return: 返回读取到的数据列表，如果return_mode为True，则同时返回文件编码格式

        >> read_jsonl('data.jsonl', max_items=10)  # 读取前10条数据
        """
        s, encoding = self.read_text(encoding=encoding, errors=errors, return_mode=True)

        data = []
        # todo 这一步可能不够严谨，不同的操作系统文件格式不同。但使用splitlines也不太好，在数据含有NEL等特殊字符时会多换行。
        for line in s.split('\n'):
            if line:
                try:  # 注意，这里可能会有数据读取失败
                    data.append(json.loads(line))
                except json.decoder.JSONDecodeError:
                    pass
            # 如果达到了限制的条目数，就停止读取
            if max_items is not None and len(data) >= max_items:
                break

        if return_mode:
            return data, encoding
        else:
            return data

    def write_jsonl(self, list_data, ensure_ascii=False, default=None):
        """ 由于这种格式主要是跟商汤这边对接，就尽量跟它们的格式进行兼容 """
        content = '\n'.join([json.dumps(x, ensure_ascii=ensure_ascii, default=default) for x in list_data])
        self.write_text_unix(content + '\n')

    def read_csv(self, encoding='utf8', *, errors='strict', return_mode: bool = False,
                 delimiter=',', quotechar='"', **kwargs):
        """
        :return:
            data，n行m列的list
        """
        import csv

        s, encoding = self.read_text(encoding=encoding, errors=errors, return_mode=True)
        data = list(csv.reader(s.splitlines(), delimiter=delimiter, quotechar=quotechar, **kwargs))

        if return_mode:
            return data, encoding
        else:
            return data

    def read_yaml(self, encoding='utf8', *, errors='strict', rich_return=False):
        s, encoding = self.read_text(encoding=encoding, errors=errors, return_mode=True)
        data = yaml.safe_load(s)

        if rich_return:
            return data, encoding
        else:
            return data

    def write_yaml(self, data, encoding='utf8', *, sort_keys=False, **kwargs):
        with open(self, 'w', encoding=encoding) as f:
            yaml.dump(data, f, sort_keys=sort_keys, **kwargs)

    def read_auto(self, *args, **kwargs):
        """ 根据文件后缀自动识别读取函数 """
        if self.is_file():  # 如果存在这样的文件，那就读取文件内容
            # 获得文件扩展名，并统一转成小写
            mode = self.suffix.lower()[1:]
            read_func = getattr(self, 'read_' + mode, None)
            if read_func:
                return read_func(*args, **kwargs)
            elif mode in ('jpg', 'png'):  # 常见的一些二进制数据
                return self.read_bytes()
            else:
                return self.read_text(*args, **kwargs)
        else:  # 非文件对象
            raise FileNotFoundError(f'{self} 文件不存在，无法读取。')

    def write_auto(self, data, *args, if_exists=None, **kwargs):
        """ 根据文件后缀自动识别写入函数 """
        mode = self.suffix.lower()[1:]
        write_func = getattr(self, 'write_' + mode, None)
        if self.exist_preprcs(if_exists):
            if write_func:
                return write_func(data, *args, **kwargs)
            else:
                return self.write_text(str(data), *args, **kwargs)
        return self

    def __2_glob(self):
        """ 类型判断、glob系列 """
        pass

    def ____1_常用glob(self):
        pass

    def glob(self, pattern):
        # TODO 加正则匹配模式
        if isinstance(pattern, str):
            return super(XlPath, self).glob(pattern)
        elif isinstance(pattern, (list, tuple)):
            from itertools import chain
            ls = [super(XlPath, self).glob(x) for x in pattern]
            # 去重
            exists = set()
            files = []
            for f in chain(*ls):
                k = f.as_posix()
                if k not in exists:
                    exists.add(k)
                    files.append(f)
            return files

    def glob_files(self, pattern='*'):
        for f in self.glob(pattern):
            if f.is_file():
                yield f

    def rglob_files(self, pattern='*'):
        for f in self.rglob(pattern):
            if f.is_file():
                yield f

    def glob_dirs(self, pattern='*'):
        for f in self.glob(pattern):
            if f.is_dir():
                yield f

    def rglob_dirs(self, pattern='*'):
        for f in self.rglob(pattern):
            if f.is_dir():
                yield f

    def glob_stems(self):
        """ 按照文件的stem分组读取

        :return: 返回格式类似 {'stem1': [suffix1, suffix2, ...], 'stem2': [suffix1, suffix2, ...], ...}
        """
        from collections import defaultdict
        d = defaultdict(set)
        for f in self.glob_files():
            d[f.stem].add(f.suffix)
        return d

    def glob_suffixs(self):
        """ 判断目录下有哪些扩展名的文件 """
        suffixs = set()
        for f in self.glob_files():
            suffixs.add(f.suffix)
        return suffixs

    def ____2_定制glob(self):
        pass

    def is_image(self):
        return self.is_file() and filetype.is_image(self)

    def _glob_images(self, glob_func, pattern):
        """ 在满足glob规则基础上，后缀还必须是合法的图片格式后缀
        """
        suffixs = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'gif'}
        for f in glob_func(pattern):
            if f.is_file() and f.suffix[1:].lower() in suffixs:
                yield f

    def glob_images(self, pattern='*'):
        """ 按照文件后缀，获取所有图片文件 """
        return self._glob_images(self.glob, pattern)

    def rglob_images(self, pattern='*'):
        return self._glob_images(self.rglob, pattern)

    def ____3_xglob(self):
        """ xglob系列，都是按照文件的实际内容，进行类型检索遍历的 """

    def xglob_images(self, pattern='*'):
        """ 按照文件实际内容，获取所有图片文件 """
        for f in self.glob(pattern):
            if f.is_file() and filetype.is_image(f):
                yield f

    def xglob_videos(self, pattern='*'):
        """ 按照文件实际内容，获取所有图片文件 """
        for f in self.glob(pattern):
            if f.is_file() and filetype.is_video(f):
                yield f

    def xglob_archives(self, pattern='*'):
        """ 找出所有压缩文件

        只找通用意义上的压缩包，不找docx等这种形式的文件
        """
        for f in self.glob(pattern):
            if f.is_file() and filetype.is_archive(f):
                if f.suffix in {'.pdf', '.docx', '.xlsx', '.pptx'}:
                    continue
                yield f

    def __3_文件基础操作(self):
        """ 复制、删除、移动等操作 """
        pass

    def exist_preprcs(self, if_exists=None):
        """ 从旧版File机制复制过来的函数

        这个实际上是在做copy等操作前，如果目标文件已存在，需要预先删除等的预处理
        并返回判断，是否需要执行下一步操作

        有时候情况比较复杂，process无法满足需求时，可以用exist_preprcs这个底层函数协助

        :param if_exists:
            None: 不做任何处理，直接运行，依赖于功能本身是否有覆盖写入机制
            'error': 如果要替换的目标文件已经存在，则报错
            'replace': 把存在的文件先删除
                本来是叫'delete'更准确的，但是考虑用户理解，
                    一般都是用在文件替换场合，叫成'delete'会非常怪异，带来不必要的困扰、误解
                所以还是决定叫'replace'
            'skip': 不执行后续功能
            'backup': 先做备份  （对原文件先做一个备份）
        """
        need_run = True
        if self.exists():
            if if_exists is None:
                return need_run
            elif if_exists == 'error':
                raise FileExistsError(f'目标文件已存在： {self}')
            elif if_exists == 'replace':
                self.delete()
            elif if_exists == 'skip':
                need_run = False
            elif if_exists == 'backup':
                self.backup(move=True)
            else:
                raise ValueError(f'{if_exists}')
        return need_run

    def copy(self, dst, if_exists=None):
        """ 用于一般的文件、目录拷贝

        不返回结果文件，是因为 if_exists 的逻辑比较特殊，比如skip的时候，这里不好实现返回的是最后的目标文件
        """
        if not self.exists():
            return

        dst = XlPath(dst)
        if dst.exist_preprcs(if_exists):
            if self.is_file():
                shutil.copy2(self, dst)
            else:
                shutil.copytree(self, dst)

    def move(self, dst, if_exists=None):
        return self.rename2(dst, if_exists)

    def rename2(self, dst, if_exists=None):
        """ 相比原版的rename，搞了更多骚操作，但性能也会略微下降，所以重写一个功能名 """
        if not self.exists():
            return self

        dst = XlPath(dst)
        if self == dst:
            # 同一个文件，可能是调整了大小写名称
            if self.as_posix() != dst.as_posix():
                tmp = self.tempfile(dir=self.parent)  # self不一定是file，也可能是dir，但这个名称通用
                self.rename(tmp)
                self.delete()
                tmp.rename(dst)
        elif dst.exist_preprcs(if_exists):
            self.rename(dst)
        return dst

    def delete(self):
        if self.is_file():
            os.remove(self)
        elif self.is_dir():
            shutil.rmtree(self)

    def backup(self, tail=None, if_exists='replace', move=False):
        r""" 对文件末尾添加时间戳备份，也可以使用自定义标记tail

        :param tail: 自定义添加后缀
            tail为None时，默认添加特定格式的时间戳
        :param if_exists: 备份的目标文件名存在时的处理方案
            这个概率非常小，真遇到，先把已存在的删掉，重新写入一个是可以接受的
        :param move: 是否删除原始文件

        # TODO：有个小bug，如果在不同时间实际都是相同一个文件，也会被不断反复备份
        #    如果想解决这个，就要读取目录下最近的备份文件对比内容了
        """
        from datetime import datetime

        # 1 判断自身文件是否存在
        if not self:
            return None

        # 2 计算出新名称
        if not tail:
            tail = datetime.fromtimestamp(self.mtime()).strftime(' %y%m%d-%H%M%S')  # 时间戳
        name, ext = os.path.splitext(str(self))
        dst = name + tail + ext

        # 3 备份就是特殊的copy操作
        if move:
            return self.move(dst, if_exists)
        else:
            return self.copy(dst, if_exists)

    def __4_重复文件相关功能(self):
        """ 检查目录里的各种文件情况 """

    def glob_repeat_files(self, pattern='*', *, sort_mode='count', print_mode=False,
                          files=None, hash_func=None):
        """ 返回重复的文件组

        :param files: 直接指定候选文件清单，此时pattern默认失效
        :param hash_func: hash规则，默认使用etag规则
        :param sort_mode:
            count: 按照重复的文件数量从多到少排序
            size: 按照空间总占用量从大到小排序
        :return: [(etag, files, per_file_size), ...]
        """
        # 0 文件清单和hash方法
        if files is None:
            files = list(self.glob_files(pattern))

        if hash_func is None:
            def hash_func(f):
                return get_etag(str(f))

        # 1 获取所有etag，这一步比较费时
        hash2files = defaultdict(list)

        for f in tqdm(files, desc='get etags', disable=not print_mode):
            etag = hash_func(f)
            hash2files[etag].append(f)

        # 2 转格式，排序
        hash2files = [(k, vs, vs[0].size()) for k, vs in hash2files.items() if len(vs) > 1]
        if sort_mode == 'count':
            hash2files.sort(key=lambda x: (-len(x[1]), -len(x[1]) * x[2]))
        elif sort_mode == 'size':
            hash2files.sort(key=lambda x: (-len(x[1]) * x[2], -len(x[1])))

        # 3 返回每一组数据
        return hash2files

    def delete_repeat_files(self, pattern='*', *, sort_mode='count', print_mode=True, debug=False,
                            files=None, hash_func=None):
        """
        :param debug:
            True，只是输出检查清单，不做操作
            False, 保留第1个文件，删除其他文件
            TODO，添加其他删除模式
        :param print_mode:
            0，不输出
            'str'，普通文本输出（即返回的msg）
            'html'，TODO 支持html富文本显示，带超链接
        """
        from humanfriendly import format_size

        def printf(*args, **kwargs):
            if print_mode == 'html':
                raise NotImplementedError
            elif print_mode:
                print(*args, **kwargs)
            msg.append(' '.join(args))

        fmtsize = lambda x: format_size(x, binary=True)

        msg = []
        files = self.glob_repeat_files(pattern, sort_mode=sort_mode, print_mode=print_mode,
                                       files=files, hash_func=hash_func)
        for i, (etag, files, _size) in enumerate(files, start=1):
            n = len(files)
            printf(f'{i}、{etag}\t{fmtsize(_size)} × {n} ≈ {fmtsize(_size * n)}')

            for j, f in enumerate(files, start=1):
                if debug:
                    printf(f'\t{f.relpath(self)}')
                else:
                    if j == 1:
                        printf(f'\t{f.relpath(self)}')
                    else:
                        f.delete()
                        printf(f'\t{f.relpath(self)}\tdelete')
            if print_mode:
                printf()

        return msg

    def check_repeat_files(self, pattern='**/*', **kwargs):
        if 'debug' not in kwargs:
            kwargs['debug'] = True
        return self.delete_repeat_files(pattern, **kwargs)

    def check_repeat_name_files(self, pattern='**/*', **kwargs):
        if 'hash_func' not in kwargs:
            kwargs['hash_func'] = lambda p: p.name.lower()
        return self.check_repeat_files(pattern, **kwargs)

    def __5_文件后缀相关功能(self):
        """ 检查目录里的各种文件情况 """

    def refine_files_suffix(self, pattern='*', *, print_mode=True, if_exists=None, debug=False):
        """ 优化文件的后缀名 """
        j = 1
        for i, f1 in enumerate(self.glob_files(pattern), start=1):
            suffix1 = f1.suffix

            suffix2 = suffix1.lower()
            if suffix2 == '.jpeg':
                suffix2 = '.jpg'

            if suffix1 != suffix2:
                f2 = f1.with_suffix(suffix2)

                if print_mode:
                    print(f'{i}/{j} {f1} -> {suffix2}')

                if not debug:
                    f1.rename2(f2, if_exists=if_exists)
                j += 1

    def _check_faker_suffix(self, file_list):
        """ 检查文件扩展名是否匹配实际内容类型，并迭代文件列表进行处理。

        :param file_list: 文件列表
        :return: 迭代器，产生文件路径和相应的信息
        """
        for file_path in file_list:
            t = filetype.guess(file_path)
            if not t:
                continue

            ext = '.' + t.extension
            ext0 = file_path.suffix

            if ext0 in ('.docx', '.xlsx', '.pptx'):
                ext0 = '.zip'
            elif ext0 in ('.JPG', '.jpeg'):
                ext0 = '.jpg'

            if ext != ext0:
                yield file_path, ext

    def xglob_faker_suffix_files(self, pattern='*'):
        """ 检查文件扩展名是不是跟实际内容类型不匹配，有问题

        注：推荐先运行 refine_files_suffix，本函数是大小写敏感，并且不会区分jpeg和jpg
        """
        return self._check_faker_suffix(self.glob_files(pattern))

    def xglob_faker_suffix_images(self, pattern='*'):
        """ 只检查原本就是图片名称标记的文件的相关数据正误
        """
        return self._check_faker_suffix(self.glob_images(pattern))

    def rename_faker_suffix_files(self, pattern='*', *, print_mode=True, if_exists=None, debug=False):
        for i, (f1, suffix2) in enumerate(self.xglob_faker_suffix_files(pattern), start=1):
            if print_mode:
                print(f'{i}、{f1} -> {suffix2}')

            if not debug:
                f2 = f1.with_suffix(suffix2)
                f1.rename2(f2, if_exists=if_exists)

    def __6_文件夹分析诊断(self):
        pass

    def check_size(self, return_mode='str'):
        import pandas as pd

        msg = []
        file_sizes = {}  # 缓存文件大小，避免重复计算
        suffix_counts = Counter()
        suffix_sizes = defaultdict(int)

        dir_count, file_count = 0, 0
        for root, dirs, files in os.walk(self):
            dir_count += len(dirs)
            file_count += len(files)
            for file in files:
                file_size = os.path.getsize(os.path.join(root, file))
                file_sizes[(root, file)] = file_size

                _, suffix = os.path.splitext(file)
                suffix_counts[suffix] += 1
                suffix_sizes[suffix] += file_size

        sz = human_readable_size(sum(file_sizes.values()))
        # 这里的目录指"子目录"数，不包含self
        msg.append(f'一、目录数：{dir_count}，文件数：{file_count}，总大小：{sz}')

        data = []
        for suffix, count in suffix_counts.most_common():
            size = suffix_sizes[suffix]
            data.append([suffix, count, size])
        data.sort(key=lambda x: (-x[2], -x[1], x[0]))  # 先按文件数，再按文件名排序
        df = pd.DataFrame(data, columns=['suffix', 'count', 'size'])
        df['size'] = [human_readable_size(x) for x in df['size']]
        df.reset_index(inplace=True)
        df['index'] += 1
        msg.append('\n二、各后缀文件数')
        msg.append(df.to_string(index=False))

        if return_mode == 'str':
            return '\n'.join(msg)
        elif return_mode == 'list':
            return msg
        else:
            return msg

    def check_summary(self, print_mode=True, return_mode=False, **kwargs):
        if self.is_dir():
            res = self._check_dir_summary(print_mode, **kwargs)
        elif self.is_file():
            res = self._check_file_summary(print_mode, **kwargs)
        else:
            res = '文件不存在'
            print(res)

        if return_mode:
            return res

    def _check_file_summary(self, print_mode=True, **kwargs):
        """ 对文件进行通用的状态检查

        :param bool print_mode: 是否将统计信息打印到控制台
        :return dict: 文件的统计信息
        """
        file_summary = {}

        # 文件大小
        file_summary['文件大小'] = self.size(human_readable=True)

        # 文件行数
        file_summary['文件行数'] = self.get_total_lines()

        # 文件修改时间
        mod_time_str = datetime.datetime.fromtimestamp(self.mtime()).strftime('%Y-%m-%d %H:%M:%S')
        file_summary['修改时间'] = mod_time_str

        # 如果print_mode为True，则将统计信息打印到控制台
        if print_mode:
            for key, value in file_summary.items():
                print(f"{key}: {value}")

        return file_summary

    def _check_dir_summary(self, print_mode=True, hash_func=None, run_mode=99):
        """ 对文件夹情况进行通用的状态检查

        :param hash_func: 可以传入自定义的hash函数，用于第四块的重复文件运算
            其实默认的get_etag就没啥问题，只是有时候为了性能考虑，可能会传入一个支持，提前有缓存知道etag的函数
        :param int run_mode: 只运行编号内的功能
        """
        if not self.is_dir():
            return ''

        def printf(s):
            if print_mode:
                print(s)
            msg.append(s)

        # 一 目录大小，二 各后缀文件大小
        msg = []
        if run_mode >= 1:  # 1和2目前是绑定一起运行的
            printf('【' + self.as_posix() + '】目录检查')
            printf('\n'.join(self.check_size('list')))

        # 三 重名文件
        if run_mode >= 3:
            printf('\n三、重名文件（忽略大小写，跨目录检查name重复情况）')
            printf('\n'.join(self.check_repeat_name_files(print_mode=False)))

        # 四 重复文件
        if run_mode >= 4:
            printf('\n四、重复文件（etag相同）')
            printf('\n'.join(self.check_repeat_files(print_mode=False, hash_func=hash_func)))

        # 五 错误扩展名
        if run_mode >= 5:
            printf('\n五、错误扩展名')
            for i, (f1, suffix2) in enumerate(self.xglob_faker_suffix_files('**/*'), start=1):
                printf(f'{i}、{f1.relpath(self)} -> {suffix2}')

        # 六 文件配对
        if run_mode >= 6:
            printf('\n六、文件配对（检查每个目录里stem名称是否配对，列出文件组成不单一的目录结构，请重点检查落单未配对的情况）')
            prompt = False
            for root, dirs, files in os.walk(self):
                suffix_counts = defaultdict(list)
                for file in files:
                    stem, suffix = os.path.splitext(file)
                    suffix_counts[stem].append(suffix)
                suffix_counts = {k: tuple(sorted(v)) for k, v in suffix_counts.items()}
                suffix_counts2 = {v: k for k, v in suffix_counts.items()}  # 反向存储，如果有重复v会进行覆盖
                ct = Counter(suffix_counts.values())
                if len(ct.keys()) > 1:
                    printf(root)
                    for k, v in ct.most_common():
                        tag = f'\t{k}: {v}'
                        if v == 1:
                            tag += f'，{suffix_counts2[k]}'
                        if len(k) > 1 and not prompt:
                            tag += f'\t标记注解：有{v}组stem相同文件，配套有{k}这些后缀。其他标记同理。'
                            prompt = True
                        printf(tag)

        return '\n'.join(msg)

    def __7_目录复合操作(self):
        """ 比较高级的一些目录操作功能 """

    def delete_empty_subdir(self, recursive=True, topdown=False):
        """ 删除指定目录下的所有空目录。

        :param recursive: 是否递归删除所有子目录。
        :param topdown: 是否从顶部向下遍历目录结构。
            默认False，要先删除内部目录，再删除外部目录。
        """

        def _delete_empty_dirs(dir_path):
            for dir_name in os.listdir(dir_path):
                dir_fullpath = os.path.join(dir_path, dir_name)
                if os.path.isdir(dir_fullpath):
                    # 递归删除子目录中的空目录
                    _delete_empty_dirs(dir_fullpath)
                    if not os.listdir(dir_fullpath):
                        # 删除空目录
                        os.rmdir(dir_fullpath)

        if not recursive:
            for dirname in os.listdir(self):
                dir_fullpath = os.path.join(self, dirname)
                if os.path.isdir(dir_fullpath) and not os.listdir(dir_fullpath):
                    os.rmdir(dir_fullpath)
        else:
            for dirpath, dirnames, filenames in os.walk(self, topdown=topdown):
                for dirname in dirnames:
                    dir_fullpath = os.path.join(dirpath, dirname)
                    if not os.listdir(dir_fullpath):
                        os.rmdir(dir_fullpath)
                # 如果不递归删除子目录，则直接跳过
                if not recursive:
                    break

    def flatten_directory(self, *, clear_empty_subdir=True):
        """ 将子目录的文件全部取出来，放到外面的目录里

        :param clear_empty_subdir: 移除文件后，删除空子目录
        """
        # 1 检查是否有重名文件，如果有重名文件则终止操作
        if msg := self.check_repeat_name_files(print_mode=False):
            print('\n'.join(msg))
            raise ValueError('有重名文件，终止操作')

        # 2 操作
        self._flatten_directory_recursive(self, clear_empty_subdir)

    def _flatten_directory_recursive(self, current_dir, clear_empty_subdir):
        for name in os.listdir(current_dir):
            path = os.path.join(current_dir, name)

            if os.path.isdir(path):
                # If it's a directory, recursively flatten it
                self._flatten_directory_recursive(path, clear_empty_subdir)
                if clear_empty_subdir:
                    shutil.rmtree(path)
            elif os.path.isfile(path):
                # If it's a file, move it to the top-level directory
                destination_path = os.path.join(self, name)
                shutil.move(path, destination_path)

    def _nest_directory_core(self, file_names, min_files_per_batch=None, groupby=None):
        """ 核心方法：将文件列表按照指定规则进行分组

        :param file_names: 文件列表
        :param min_files_per_batch: 每个批次最少包含的文件数，默认为 None，即不限制最少文件数
        :param groupby: 分组函数，用于指定按照哪个属性进行分组，默认为 None
        :return: 分组结果列表
        """
        from pyxllib.algo.pupil import Groups, natural_sort

        if groupby is None:
            groupby = lambda p: p.stem.lower()
        file_groups = Groups.groupby(file_names, groupby).data

        if min_files_per_batch is None:
            min_files_per_batch = 1

        result_groups, current_group = [], []
        for stem in natural_sort(file_groups.keys()):
            current_group += file_groups[stem]
            if len(current_group) >= min_files_per_batch:
                result_groups.append(current_group)
                current_group = []
        if current_group:
            result_groups.append(current_group)

        return result_groups

    def nest_directory(self, min_files_per_batch=None, groupby=None, batch_name=None, bias=0, tail_limit=None):
        """ 将直接子文件按照一定规则拆分成多个batch子目录
        注意这个功能和flatten_directory是对称的，所以函数名也是对称的

        :param min_files_per_batch: 每个batch最少含有的文件数
            None，相当于int=1的效果
            int, 如果输入一个整数，则按照这个数量约束分成多个batch
        :param groupby: 默认会把stem.lower()相同的强制归到一组
            def groupby(p: XlPath) -> 分组用的key，相同key会归到同一组
            这个不仅用于分组，返回的字符串，也会作为字典序排序的依据，如果想用自然序，记得加natural_sort_key进行转换
        :param batch_name: 设置batch的名称，默认 'batch{}'
        :param bias: 希望用不到这个参数，只有中途出bug，需要继续处理的时候，用来自动增加编号
        :param tail_limit: 限制数量少于多少的batch，合并到上一个batch中
        """
        from pyxllib.algo.pupil import Groups

        # 1 按stem分组，确定分组数
        file_names = list(self.glob_files('*'))
        if groupby is None:
            groupby = lambda p: p.stem.lower()
        file_groups = Groups.groupby(file_names, groupby).data

        if min_files_per_batch is None:
            min_files_per_batch = 1

        # 2 将文件组合并为多个分组，每组至少包含 min_files_per_batch 个文件
        result_groups, current_group = [], []
        for stem in sorted(file_groups.keys()):  # 注意这里需要对取到的key按照自然序排序
            current_group += file_groups[stem]
            if len(current_group) >= min_files_per_batch:
                result_groups.append(current_group)
                current_group = []
        if current_group:
            if tail_limit is None:
                tail_limit = min_files_per_batch // 10

            if len(current_group) < tail_limit and result_groups:
                result_groups[-1] += current_group
            else:
                result_groups.append(current_group)

        # 3 整理实际的文件
        if batch_name is None:
            group_num = len(result_groups)
            width = len(str(group_num))
            batch_name = f'batch{{:0{width}}}'
        for i, group in enumerate(result_groups, start=1):
            d = self / batch_name.format(i + bias)
            d.mkdir(exist_ok=True)
            for f in group:
                f.move(d / f.name)

    def select_file(self, pos_filter=None, *, neg_filter=None):
        from pyxllib.file.specialist.dirlib import Dir

        d = Dir(self)
        if pos_filter is None:
            # 基于filesmatch的底层来实现，速度会比较慢一些，但功能丰富，不用重复造轮子
            d = d.select('**/*', type_='file')
        else:
            d = d.select(pos_filter, type_='file')
        if neg_filter is not None:
            d = d.exclude(neg_filter)

        files = [(self / f) for f in d.subs]
        return files

    def copy_file_filter(self, dst, pos_filter=None, *, neg_filter=None, if_exists=None):
        """ 只能用于目录，在复制文件的时候，进行一定的筛选，而不是完全拷贝

        :param dst: 目标目录
        :param pos_filter: 对文件的筛选规则
            其实常用的就'*.json'这种通配符
            支持自定义函数，def pos_filter(p: XlPath)
            参数详细用法参考 filesmatch，注意这里筛选器只对file启用，不会对dir启用，否则复制逻辑会非常乱
        :param neg_filter:

        注意！这个功能还是有点特殊的，不建议和XlPath.copy的接口做合并。

        正向、反向两个过滤器可以组合使用，逻辑上是先用正向获取全部文件，然后扣除掉反向。
        正向默认全选，反向默认不扣除。

        注意：这样过滤后，空目录不会被拷贝。
            如果有拷贝目录的需求，请自己另外写逻辑实现。
            如果只是拷贝空目录结构的需求，可以使用 copy_dir_structure

        >> p = XlPath('build')
        >> p.copy_filter('build2', '*.toc')  # 复制直接子目录下的所有toc文件
        >> p.copy_filter('build2', '**/*.toc')  # 复制所有toc文件
        >> p.copy_filter('build2', lambda p: p.suffix == '.toc')  # 复制所有toc文件
        """
        files = self.select_file(pos_filter, neg_filter=neg_filter)

        dst = XlPath(dst)
        for f in files:
            dst2 = dst / f
            dst2.parent.mkdir(exist_ok=True)
            (self / f).copy(dst2, if_exists=if_exists)

    def copy_dir_structure(self, dst):
        """ 只复制目录结构，不复制文件内容 """
        dst = XlPath(dst)
        for root, dirs, _ in os.walk(self):
            # 构造目标路径
            dst_dir = dst / XlPath(root).relative_to(self)
            # 创建目录
            dst_dir.mkdir(parents=True, exist_ok=True)

    # 无法选定文件
    def _move_selectable(self, dst_dir):
        """ 目录功能，将目录下可选中的文件移动到目标目录

        1、要理解这个看似有点奇怪的功能，需要理解，在数据处理中，可能会拿到超长文件名的文件，
            这种在windows平台虽然手动可以操作，但在代码中，会glob不到，强制指定也会说文件不存在
        2、为了解决这类文件问题，一般需要对其进行某种规则的重命名。因为linux里似乎不会限制文件名长度，所以要把这些特殊文件打包到linux里处理。
        3、因为这些文件本来就无法被选中，所以只能反向操作，将目录下的可选中文件移动到目标目录。
        """
        for p in self.glob('*'):
            if p.exists():
                p.move(dst_dir / p.name)

    def move_unselectable(self, dst_dir):
        """ 见_move_selectable，因为无法对这些特殊文件进行移动
        所以这里只是对_move_selectable的封装，中间通过文件重命名，来伪造移动了无法选中文件的操作效果
        """
        tempdir = self.create_tempdir_path(dir=self.parent)
        tempdir.mkdir(exist_ok=True)
        self._move_selectable(tempdir)
        self.rename2(dst_dir)
        tempdir.move(self)


def demo_file():
    """ File类的综合测试"""
    temp = tempfile.gettempdir()

    # 切换工作目录到临时文件夹
    os.chdir(temp)

    p = File('demo_path', temp)
    p.delete()  # 如果存在先删除
    p.ensure_parent()  # 然后再创建一个空目录

    print(File('demo_path', suffix='.py'))
    # F:\work\CreatorTemp\demo_path.py

    print(File('demo_path/', suffix='.py'))
    # F:\work\CreatorTemp\demo_path\tmp65m8mc0b.py

    # ...区别于None，会随机生成一个文件名
    print(File(..., temp))
    # F:\work\CreatorTemp\tmpwp4g1692

    # 可以在随机名称基础上，再指定文件扩展名
    print(File('', temp, suffix='.txt'))
    # F:\work\CreatorTemp\tmpimusjtu1.txt


def demo_file_rename():
    # 建一个空文件
    f1 = File('temp/a.txt', tempfile.gettempdir())
    # Path('F:/work/CreatorTemp/temp/a.txt')
    with open(str(f1), 'wb') as p: pass  # 写一个空文件

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
        if isinstance(init_bytes, (File, str)):
            # with open的作用：可以用.read循序读入，而不是我的Path.read一口气读入。
            #   这在只需要进行局部数据分析，f本身又非常大的时候很有用。
            #   但是我这里操作不太方便等原因，还是先全部读入BytesIO流了
            init_bytes = File(init_bytes).read(mode='b')
        super().__init__(init_bytes)

    def unpack(self, fmt):
        return struct_unpack(self, fmt)

    def readtext(self, char_num, encoding='gbk', errors='ignore', code_length=2):
        """ 读取二进制流，将其解析为文本内容

        :param char_num: 字符数
        :param encoding: 所用编码，一般是gbk，因为如果是utf8是字符是变长的，不好操作
        :param errors: decode出现错误时的处理方式
        :param code_length: 每个字符占的长度
        :return: 文本内容
        """
        return self.read(code_length * char_num).decode(encoding, errors)


class PathGroups(Groups):
    """ 按stem文件名（不含后缀）分组的相关功能 """

    @classmethod
    def groupby(cls, files, key=lambda x: os.path.splitext(XlPath(x).as_posix())[0], ykey=lambda y: y.suffix[1:]):
        """
        :param files: 用Dir.select选中的文件、目录清单
        :param key: D:/home/datasets/textGroup/SROIE2019+/data/task3_testcrop/images/X00016469670
        :param ykey: ['jpg', 'json', 'txt']
        :return: dict
            1, task3_testcrop/images/X00016469670：['jpg', 'json', 'txt']
            2, task3_testcrop/images/X00016469671：['jpg', 'json', 'txt']
            3, task3_testcrop/images/X51005200931：['jpg', 'json', 'txt']
        """
        return super().groupby(files, key, ykey)

    def select_group(self, judge):
        """ 对于某一组，只要该组有满足judge的元素则保留该组 """
        data = {k: v for k, v in self.data.items() if judge(k, v)}
        return type(self)(data)

    def select_group_which_hassuffix(self, pattern, flags=re.IGNORECASE):
        def judge(k, values):
            for v in values:
                m = re.match(pattern, v, flags=flags)
                if m and len(m.group()) == len(v):
                    # 不仅要match满足，还需要整串匹配，比如jpg就必须是jpg，不能是jpga
                    return True
            return False

        return self.select_group(judge)

    def select_group_which_hasimage(self, pattern=r'jpe?g|png|bmp', flags=re.IGNORECASE):
        """ 只保留含有图片格式的分组数据 """
        return self.select_group_which_hassuffix(pattern, flags)

    def find_files(self, name, *, count=-1):
        """ 找指定后缀的文件

        :param name: 支持 '1.jpg', 'a/1.jpg' 等格式
        :param count: 返回匹配数量上限，-1表示返回所有匹配项
        :return: 找到第一个匹配项后返回
            找的有就返回XlPath对象
            找没有就返回None

        注意这个功能是大小写敏感的，如果出现大小写不匹配
            要么改文件名本身，要么改name的格式

        TODO 如果有大量的检索，这样每次遍历会很慢，可能要考虑构建后缀树来处理
        """
        ls = []
        stem, ext = os.path.splitext(name)
        stem = '/' + stem.replace('\\', '/')
        ext = ext[1:]
        for k, v in self.data.items():
            if k.endswith(stem) and ext in v:
                ls.append(XlPath.init(k, suffix=ext))
                if len(ls) >= count:
                    break
        return ls


def cache_file(file, make_data_func: Callable[[], Any] = None, *, reset=False, **kwargs):
    """ 局部函数功能结果缓存

    输入的文件file如果存在则直接读取内容；
    否则用make_data_func生成，并且备份一个file文件

    :param file: 需要缓存的文件路径
    :param make_data_func: 如果文件不存在，则需要生成一份，要提供数据生成函数
        cache_file可以当装饰器用，此时不用显式指定该参数
    :param reset: 如果file是否已存在，都用make_data_func强制重置一遍
    :param kwargs: 可以传递read、write支持的扩展参数
    :return: 从缓存文件直接读取到的数据
    """

    def decorator(func):
        def wrapper(*args2, **kwargs2):
            f = XlPath.init(file, XlPath.tempdir())
            if f.exists() and not reset:  # 文件存在，直接读取返回
                data = f.read_auto(**kwargs)
            else:  # 文件不存在则要生成一份数据
                data = func(*args2, **kwargs2)
                f.write_auto(data, **kwargs)
            return data

        return wrapper

    return decorator(make_data_func)() if make_data_func else decorator


class UsedRecords:
    """存储用户的使用记录到一个文件"""

    def __init__(self, filename, default_value=None, *, use_temp_root=False, limit_num=30):
        """ 记录存储文件

        :param filename: 文件路径与名称
        :param default_value:
        :param use_temp_root: 使用临时文件夹作为根目录
        :param limit_num: 限制条目上限
        """
        from os.path import join, dirname, basename, exists
        from pyxllib.text.specialist import ensure_content

        # 1 文件名处理
        if use_temp_root:
            dirname = join(os.getenv('TEMP'), 'code4101py_config')
            basename = basename(filename)
            fullname = join(dirname, basename)
        else:
            dirname = dirname(filename)
            basename = basename(filename)
            fullname = filename

        # 2 读取值
        if exists(fullname):
            ls = ensure_content(fullname).splitlines()
        else:
            ls = list(default_value)

        # 3 存储到类
        self.dirname = dirname
        self.basename = basename
        self.fullname = fullname
        self.ls = ls
        self.limit_num = limit_num

    def save(self):
        """保存记录文件"""
        File(self.dirname + '/').ensure_parent()
        File(self.fullname).write('\n'.join(self.ls), if_exists='replace')

    def add(self, s):
        """新增一个使用方法
        如果s在self.ls里，则把方法前置到第一条
        否则在第一条添加新方法

        如果总条数超过30要进行删减
        """
        if s in self.ls:
            del self.ls[self.ls.index(s)]

        self.ls = [s] + list(self.ls)

        if len(self.ls) > self.limit_num:
            self.ls = self.ls[:self.limit_num]

    def __str__(self):
        res = list()
        res.append(self.fullname)
        for t in self.ls:
            res.append(t)
        return '\n'.join(res)


def __5_filedfs():
    """
    对目录的遍历查看目录结构
    """


def file_generator(f):
    """普通文件迭代生成器
    :param f: 搜索目录
    """
    if os.path.isdir(f):
        try:
            dirpath, dirnames, filenames = myoswalk(f).__next__()
        except StopIteration:
            return []

        ls = filenames + dirnames
        ls = map(lambda x: dirpath + '/' + x, ls)
        return ls
    else:
        return []


def pyfile_generator(f):
    """py文件迭代生成器
    :param f: 搜索目录
    """
    if os.path.isdir(f):
        try:
            dirpath, dirnames, filenames = myoswalk(f).__next__()
        except StopIteration:
            return []
        filenames = list(filter(lambda x: x.endswith('.py'), filenames))
        ls = filenames + dirnames
        ls = map(lambda x: dirpath + '/' + x, ls)
        return ls
    else:
        return []


def texfile_generator(f):
    """tex  文件迭代生成器
    :param f: 搜索目录
    """
    if os.path.isdir(f):
        try:
            dirpath, dirnames, filenames = myoswalk(f).__next__()
        except StopIteration:
            return []

        filenames = list(filter(lambda x: x.endswith('.tex'), filenames))
        ls = filenames + dirnames
        ls = map(lambda x: dirpath + '/' + x, ls)
        return ls
    else:
        return []


def file_str(f):
    """
    :param f: 输入完整路径的文件夹或文件名
    :return: 返回简化的名称
        a/b     ==> <b>
        a/b.txt ==> b.txt
    """
    name = os.path.basename(f)
    if os.path.isdir(f):
        s = '<' + name + '>'
    else:
        s = name
    return s


def filedfs(root,
            child_generator=file_generator, select_depth=None, linenum=True,
            mystr=file_str, msghead=True, lsstr=None, show_node_type=False, prefix='\t'):
    """对文件结构的递归遍历
    注意这里的子节点生成器有对非常多特殊情况进行过滤，并不是通用的文件夹查看工具
    """
    from pyxllib.text.specialist import dfs_base

    if isinstance(child_generator, str):
        if child_generator == '.py':
            child_generator = pyfile_generator
        elif child_generator == '.tex':
            child_generator = texfile_generator
        else:
            raise ValueError

    return dfs_base(root, child_generator=child_generator, select_depth=select_depth, linenum=linenum,
                    mystr=mystr, msghead=msghead, lsstr=lsstr, show_node_type=show_node_type, prefix=prefix)


def genfilename(fd='.'):
    """生成一个fd目录下的文件名
    注意只是文件名，并未实际产生文件，输入目录是为了防止生成重名文件（以basename为标准的无重名）

    格式为：180827周一195802，如果出现重名，前面的6位记为数值d1，是年份+月份+日期的标签
        后面的6位记为数值d2，类似小时+分钟+秒的标签，但是在出现重名时，
        d2会一直自加1直到没有重名文件，所以秒上是可能会出现“99”之类的值的。
    """
    from datetime import datetime
    # 1 获取前段标签
    dt = datetime.now()
    weektag = '一二三四五六日'
    s1 = dt.strftime('%y%m%d') + f'周{weektag[dt.weekday()]}'  # '180827周一'

    # 2 获取后端数值标签
    d2 = int(datetime.now().strftime('%H%M%S'))

    # 3 获取目录下文件，并迭代确保生成一个不重名文件
    ls = os.listdir(fd)
    files = set(map(lambda x: os.path.basename(os.path.splitext(x)[0]), ls))  # 收集basename

    while s1 + str(d2) in files:
        d2 += 1

    return s1 + str(d2)


def myoswalk(root, filter_rule=None, recur=True):
    """
    :param root: 根目录
    :param filter_rule:
        字符串
            以点.开头的，统一认为是进行后缀格式识别
        其他字符串类型会认为是一个正则规则，只要相对root的全名能search到规则即认为匹配
            可以将中文问号用于匹配任意汉字
        也可以输入自定义函数： 输入参数是相对root目录下的文件全名
    :param recur: 是否进行子文件夹递归
    :return:
    """
    if isinstance(filter_rule, str):
        filter_rule = gen_file_filter(filter_rule)

    # prefix_len = len(root)  # 计算出前缀长度
    for dirpath, dirnames, filenames in os.walk(root):
        # relative_root = dirpath[prefix_len+1:]  # 我想返回相对路径，但是好像不太规范会对很多东西造成麻烦
        #  过滤掉特殊目录
        for t in ('.git', '$RECYCLE.BIN', '__pycache__', 'temp', 'Old', 'old'):
            try:
                del dirnames[dirnames.index(t)]
            except ValueError:
                pass
        # 去掉备份文件
        dirnames = list(filter(lambda x: not File(x).backup_time and '-冲突-' not in x, dirnames))
        filenames = list(filter(lambda x: not File(x).backup_time and '-冲突-' not in x, filenames))

        # 调用特殊过滤规则
        if filter_rule:
            dirnames = list(filter(lambda x: filter_rule(f'{dirpath}\\{x}'), dirnames))
            filenames = list(filter(lambda x: filter_rule(f'{dirpath}\\{x}'), filenames))

        # 如果该文件夹下已经没有文件，不返回该目录
        if not (filenames or dirnames):
            continue

        # 返回生成结果
        yield dirpath, dirnames, filenames

        if not recur:  # 不进行递归
            break


def mygetfiles(root, filter_rule=None, recur=True):
    """对myoswalk进一步封装，返回所有匹配的文件
    会递归查找所有子文件

    可以这样遍历一个目录下的所有文件：
    for f in mygetfiles(r'C:\pycode\code4101py', r'.py'):
        print(f)
    这个函数已经自动过滤掉备份文件了
    筛选规则除了“.+后缀”，还可以写正则匹配

    参数含义详见myoswalk
    """
    for root, _, files in myoswalk(root, filter_rule, recur):
        for f in files:
            yield root + '\\' + f


def __6_high():
    """ 一些高级的路径功能 """


class DirsFileFinder:
    """ 多目录里的文件检索类 """

    def __init__(self, *dirs):
        """ 支持按优先级输入多个目录dirs，会对这些目录里的文件进行统一检索 """
        self.names = defaultdict(list)
        self.stems = defaultdict(list)

        for d in dirs:
            self.add_dir(d)

    def add_dir(self, p):
        """ 添加备用检索目录
        当前面的目录找不到匹配项的时候，会使用备用目录的文件
        备用目录可以一直添加，有多个，优先级逐渐降低
        """
        files = list(XlPath(p).rglob_files())
        for f in files:
            self.names[f.name].append(f)
            self.stems[f.stem].append(f)

    def find_name(self, name):
        """ 返回第一个匹配的结果 """
        names = self.find_names(name)
        if names:
            return names[0]

    def find_names(self, name):
        """ 返回所有匹配的结果 """
        return self.names[name]

    def find_stem(self, stem):
        stems = self.find_stems(stem)
        if stems:
            return stems[0]

    def find_stems(self, stem):
        return self.stems[stem]


class TwinDirs:
    def __init__(self, src_dir, dst_dir):
        """ 一对'孪生'目录，一般是有一个src_dir，还有个同结构的dst_dir。
        但dst_dir往往并不存在，是准备从src_dir处理过来的。
        """
        self.src_dir = XlPath(src_dir)
        self.dst_dir = XlPath(dst_dir)
        self.src_dir_finder = None

    def reset_dst_dir(self):
        """ 重置目标目录 """
        self.dst_dir.delete()
        self.dst_dir.mkdir()

    def copy_file(self, src_file, if_exists=None):
        """ 从src复制一个文件到dst里

        :param XlPath src_file: 原文件位置（其实输入目录类型也是可以的，内部实现逻辑一致的）
        """
        src_file = XlPath(src_file)
        _src_file = src_file.relpath(self.src_dir)
        dst_file = self.dst_dir / _src_file
        dst_dir = dst_file.parent
        dst_dir.mkdir(exist_ok=True, parents=True)  # 确保目录结构存在
        src_file.copy(dst_file, if_exists=if_exists)
        return dst_file

    def copy_ext_file(self, ext_file, if_exists=None):
        """ 和copy_file区别，这里输入的ext_file是来自其他目录的文件
        但是要在src_file里找到位置，按照结构复制到dst_dir
        """
        # 1 找到原始文件位置
        ext_file = XlPath(ext_file)
        if self.src_dir_finder is None:
            self.src_dir_finder = DirsFileFinder(self.src_dir)

        src_files = self.src_dir_finder.find_names(ext_file.name)
        assert len(src_files) < 2, '出现多种匹配可能性，请检查目录'

        if len(src_files) == 0:
            src_files = self.src_dir_finder.find_stems(ext_file.stem)
            assert len(src_files), '没有找到可匹配的文件'

        src_file = src_files[0]

        # 2 复制文件
        _src_file = src_file.relpath(self.src_dir)
        dst_file = self.dst_dir / _src_file
        dst_dir = dst_file.parent
        dst_dir.mkdir(exist_ok=True, parents=True)  # 确保目录结构存在
        ext_file.copy(dst_file, if_exists=if_exists)
        print(ext_file, '->', dst_file)
        return dst_file

    def copy_dir_structure(self):
        """ 复制目录结构 """
        self.src_dir.copy_dir_structure(self.dst_dir)
