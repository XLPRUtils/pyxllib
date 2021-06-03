#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 20:37


import io
import json
import ujson
import os
import pathlib
import pickle
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Any

import chardet
import qiniu
import requests
import yaml

from pyxllib.algo.group import Groups
from pyxllib.file.basic import struct_unpack
from pyxllib.time.datetime import Datetime
from pyxllib.prog.basic import is_url, is_file

____judge = """
"""


def is_url_connect(url, timeout=5):
    try:
        _ = requests.head(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        pass
    return False


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

    f = File('1.tex', tempfile.gettempdir()).write(s, if_exists='delete').to_str()
    print(get_etag(f))
    # FkAD2McB6ugxTiniE8ebhlNHdHh9


____chardet = """
"""


def get_encoding_old(bstr):
    """ 输入二进制字符串或文本文件，返回字符编码

    https://www.yuque.com/xlpr/pyxllib/get_encoding

    :param bstr: 二进制字符串、文本文件
    :return: utf8, utf-8-sig, gbk, utf16
    """

    # 0 检查工具
    def inner_detect(bdata):
        """ https://mp.weixin.qq.com/s/gSXNf3K_JWydhej8KpyhMg """
        from chardet.universaldetector import UniversalDetector
        detector = UniversalDetector()
        for part in bdata.split():
            detector.feed(part)
            if detector.done:
                break
        detector.close()
        return detector.result['encoding']

    # 1 读取编码
    if isinstance(bstr, bytes):  # 如果输入是一个二进制字符串流则直接识别
        # encoding = chardet.detect(bstr[:maxn])['encoding']  # 截断一下，不然太长了，太影响速度
        encoding = inner_detect(bstr)  # 截断一下，不然太长了，太影响速度
    elif is_file(bstr):  # 如果是文件，则按二进制打开
        # 如果输入是一个文件名则进行读取
        if bstr.endswith('.pdf'):
            print('二进制文件，不应该进行编码分析，暂且默认返回utf8', bstr)
            return 'utf8'
        with open(bstr, 'rb') as f:  # 以二进制读取文件，注意二进制没有\r\n的值
            bstr = f.read()
        # encoding = chardet.detect(bstr[:maxn])['encoding']
        encoding = inner_detect(bstr)
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
        # print('encoding=', encoding)
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


def get_encoding(bstr, *, maxn=100):
    # 1 从第一个大于127的字节开始判断
    for start_idx in range(len(bstr)):
        if bstr[start_idx] > 127:
            break
    else:  # 没有>127的字节
        return 'utf8'

    # 只要截取部分子节就能大概分析出了
    enc = chardet.detect(bstr[start_idx:start_idx + maxn])['encoding']

    # 2 转换为常见编码名
    if enc in ('utf-8', 'ascii', 'ISO-8859-1', 'Windows-1252', 'Windows-1254', 'ISO-8859-9'):
        return 'utf8'
    elif enc in ('UTF-8-SIG',):
        return 'utf-8-sig'
    elif enc in ('GB2312', 'GBK'):
        return 'gbk'
    elif enc in ('UTF-16',):
        return 'utf16'
    else:
        raise ValueError(f"{enc}: Can't get file encoding")


____file = """
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
    def abspath(cls, path=None, root=None, *, suffix=None) -> pathlib.Path:
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

        return pathlib.Path(path).resolve()

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

    def glob(self, pattern):
        return self._path.glob(pattern)

    def match(self, path_pattern):
        return self._path.match(path_pattern)

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
    def mtime(self) -> Datetime:
        r""" 文件的最近修改时间

        >> Path(r"C:\pycode\code4101py").mtime
        2020-03-10 17:32:37
        """
        return Datetime(os.stat(str(self)).st_mtime)

    @property
    def ctime(self) -> Datetime:
        r""" 文件的创建时间

        >> Path(r"C:\pycode\code4101py").ctime
        2018-05-25 10:46:37
        """
        # 注意：st_ctime是平台相关的值，在windows是创建时间，但在Unix是metadate最近修改时间
        return Datetime(os.stat(str(self)).st_ctime)

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

        有时候情况比较复杂，process无法满足需求时，可以用preprocess这个底层函数协助

        :param if_exists:
            None: 不做任何处理，直接运行，依赖于功能本身是否有覆盖写入机制
            'error': 如果要替换的目标文件已经存在，则报错
            'delete': 把存在的文件先删除
            'skip': 不执行后续功能
            'backup': 先做备份  （对原文件先做一个备份）
        """
        need_run = True
        if self:
            if if_exists is None:
                return need_run
            elif if_exists == 'error':
                raise FileExistsError(f'目标文件已存在： {self}')
            elif if_exists == 'delete':
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

    def backup(self, tail=None, if_exists='delete', move=False):
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


class File(PathBase):
    r""" 通用文件处理类，大部分基础功能是从pathlib.Path衍生过来的

    document: https://www.yuque.com/xlpr/python/pyxllib.debug.path
    """
    __slots__ = ('_path',)

    # 一、基础功能

    def __init__(self, path, root=None, *, suffix=None):
        r""" 初始化参数含义详见 PathBase.abspath 函数解释

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
        if self:
            total_size = os.path.getsize(str(self))
        else:
            total_size = 0
        return total_size

    # 四、文件操作功能

    def absdst(self, dst):
        """ 在copy、move等中，给了个"模糊"的目标位置dst，智能推导出实际file、dir绝对路径
        """
        dst_ = self.abspath(dst)
        if (isinstance(dst, str) and dst[-1] in ('\\', '/')) or dst_.is_dir():
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
                    if 'ensure_ascii' not in kwargs:
                        kwargs['ensure_ascii'] = False
                    ujson.dump(ob, f, **kwargs)
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
    """ 按文件名（不含后缀）分组的相关功能 """

    @classmethod
    def groupby(cls, files, key=lambda x: os.path.splitext(str(x))[0], ykey=lambda y: y.suffix[1:]):
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
            找的有就返回File对象
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
                ls.append(File(k, suffix=ext))
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
            f = File(file)
            if f and not reset:  # 文件存在，直接读取返回
                data = f.read(**kwargs)
            else:  # 文件不存在则要生成一份数据
                data = func(*args2, **kwargs2)
                f.write(data, **kwargs)
            return data

        return wrapper

    return decorator(make_data_func)() if make_data_func else decorator


if __name__ == '__main__':
    print(__file__)
