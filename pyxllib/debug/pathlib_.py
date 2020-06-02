#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/30 20:37


import json
import os
import pathlib
import pickle
import re
import shutil
import subprocess
import tempfile


from pyxllib.debug.arrow_ import Datetime
from pyxllib.debug.chardet_ import get_encoding


class Path:
    r"""通用文件、路径处理类，也可以处理目录，可以把目录对象理解为特殊类型的文件
    大部分基础功能是从pathlib.Path衍生过来的
        但开发中该类不能直接从Path继承，会有很多问题

    1、Path参考自pathlib.Path，很多接口也是直接引用过来的。
    2、标准库的pathlib.Path已经带有大部分路径处理功能了，我主要是做了各种属性获取：
        size大小，mtime最近修改时间等等。
    3、以及文件方面操作的功能扩展：
        copy复制，move移动，rename重命名，delete删除，backup备份等等
    4、目录可以理解成是一个特殊的文件，Path类也支持对目录的处理。
        这也是我没把这个类叫做File的原因，叫File并不准确，其实该类涵盖了各种路径、文件、目录处理的功能。

    Path的初始化是非常特别的，除了一般要给第一个核心参数path，
        还有suffix和root两个参数，可以非常灵活地去指明自己想要哪个文件。
        其底层是abspath函数，里面有各种示例。copy和move等也是非常智能地去判断你所指定的目标位置的，
        能避免很多繁琐、误操作。
    因为pathlib.Path的扩展名叫suffix而不是ext，所以我把writefile的扩展名接口也改成suffix，代码的命名要上下文统一。
    Path.rename有一段比较综合的演示代码
        例如其中的rename，
            你如果用os.rename功能，表明A.tXt，它默认放置的是当前工作目录，而并不是a.txt原来所在的f目录。
            但我们重命名、移动、复制等一般是相对于原来所在的父目录操作的而并不是当前工作目录。
            而且，os.rename是不支持跨磁盘操作的，我的rename则本质上是调用了move的实现，
            rename可以理解成是move的一种特例，所以我的rename不仅能自动创建事先不存在的目录层级，还能跨磁盘操作。

    后续会基于Path，扩展多文件操作类Folder，图片ImageFile、EpsFile等类，集成更多功能。

    TODO 这里的doctest过于针对自己的电脑了，应该改成更具适用性测试代码
    """
    __slots__ = ('_path',)

    # 零、常用的目录类
    TEMP = tempfile.gettempdir()
    if os.environ.get('Desktop', None):  # 如果修改了win10默认的桌面路径，需要在环境变量添加一个正确的Desktop路径值
        DESKTOP = os.environ['Desktop']
    else:
        DESKTOP = os.path.join(str(pathlib.Path.home()), 'Desktop')  # 这个不一定准，桌面是有可能被移到D盘等的

    # 一、基础功能

    def __init__(self, path=None, suffix=None, root=None):
        r"""初始化参数含义详见 abspath 函数解释
        TODO 这个初始化也有点过于灵活了，需要降低灵活性，增加使用清晰度

        >>> Path('D:/pycode/code4101py')
        Path('D:/pycode/code4101py')
        >>> Path(Path('D:/pycode/code4101py'))
        Path('D:/pycode/code4101py')

        >> Path()  # 不输入参数的时候，默认为当前工作目录
        Path('D:/pycode/code4101py')

        >>> Path('a.txt', root=Path.TEMP)
        Path('F:/work/CreatorTemp/a.txt')
        >>> Path('F:/work/CreatorTemp')
        Path('F:/work/CreatorTemp')

        注意！如果使用了符号链接（软链接），则路径是会解析转向实际位置的！例如
        >> Path('D:/pycode/code4101py')
        Path('D:/slns/pycode/code4101py')
        """
        path = str(path)
        self._path = None

        try:
            self._path = pathlib.Path(self.abspath(path, suffix, root)).resolve()
            # 有些问题上一步不一定测的出来，要再补一个测试
            self._path.is_file()
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

        >>> Path.abspath('F:/work/a.txt', 'py')  # 参考后缀不修改
        'F:/work/a.txt'
        >>> Path.abspath('F:/work/a.txt', '.py')  # 强制后缀会修改
        'F:/work/a.py'

        >> Path.abspath(suffix='.tex', root=Path.TEMP)
        'F:\\work\\CreatorTemp\\tmp5vo2lpqd.tex'

        # F:/work/a.txt不存在，而且看起来像文件名，但是末尾再用/则显式指明这其实是一个目录
        #   又用.py指明要添加一个随机名称的py文件
        >> Path.abspath('F:/work/a.txt/', '.py')
        'F:/work/a.txt/tmpg_q7a7ft.py'

        >> Path.abspath('work/a.txt/', '.py', Path.TEMP)  # 在临时文件夹下的work/a.txt目录新建一个随机名称的py文件
        'F:\\work\\CreatorTemp\\work/a.txt/tmp2jn5cqkc.py'

        >>> Path.abspath('C:/a.txt/')  # 会保留最后的斜杠特殊标记
        'C:/a.txt/'
        >>> Path.abspath('C:/a.txt\\')  # 会保留最后的斜杠特殊标记
        'C:/a.txt\\'
        """
        # 1、判断参考目录
        if not root: root = os.getcwd()

        # 2、判断主体文件名 path
        if str(path) == '':
            path = tempfile.mktemp(dir=root)
        elif not path:
            path = root
        else:
            path = os.path.join(root, str(path))

        # 3、补充suffix
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
        r"""判断文件是否存在
        重置WindowsPath的bool逻辑，返回值变成存在True，不存在为False

        >>> Path('D:/slns').exists()
        True
        >>> Path('D:/pycode/code4101').exists()
        False
        """
        return self._path and self._path.exists()

    def __repr__(self):
        return 'Path' + self._path.__repr__()[11:]

    def __str__(self):
        return str(self._path)

    def __eq__(self, other):
        if not isinstance(other, Path):
            raise TypeError
        return self._path == other._path

    def __truediv__(self, key):
        r"""路径拼接功能
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
        'D:\\pycode'
        >>> Path(r'D:\toweb\a').dirname
        'D:\\toweb'
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
        """with_suffix和suffix.setter区别是，前者是生成一个新指向的类，后者是重命名

        >>> Path('a.txt').with_suffix('.py')  # 强制替换
        Path('D:/slns/pyxllib/pyxllib/debug/a.py')
        >>> Path('a.txt').with_suffix('py')  # 参考替换
        Path('D:/slns/pyxllib/pyxllib/debug/a.txt')
        >>> Path('a.txt').with_suffix('')  # 删除
        Path('D:/slns/pyxllib/pyxllib/debug/a')
        """
        if suffix and (suffix[0] == '.' or not self.suffix):
            if suffix[0] != '.': suffix = '.' + suffix
            return Path(self._path.with_suffix(suffix))
        elif not suffix:
            # suffix 为假值则删除扩展名
            return Path(self.stem, '', self.dirname)
        return self

    @property
    def backup_time(self):
        r"""返回文件的备份时间戳，如果并不是备份文件，则返回空字符串
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
    def size(self) -> int:
        """计算文件、目录的大小，对于目录，会递归目录计算总大小
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

    def abs_dstpath(self, dst=None, suffix=None, root=None) -> str:
        r""" 参照当前Path的父目录，来确定dst的具体路径
        >>> f = Path('C:/Windows/System32/cmd.exe')
        >>> f.abs_dstpath('chen.py')
        'C:\\Windows\\System32\\chen.py'
        >>> f.abs_dstpath('E:/')  # 原始文件必须存在，否则因为无法判断实际类型，目标路径可能会错
        'E:/cmd.exe'
        >>> f.abs_dstpath('D:/aabbccdd.txt/')  # 并不存在aabbccdd.txt这样的对象，但末尾有个/表明这是个目录
        'D:/aabbccdd.txt/cmd.exe'
        """
        if not root: root = self.dirname
        dst = Path.abspath(dst, suffix, root)

        # 原始是一个文件，但这里目标只写了一个目录，则按照原名推导到目标文件名
        if self.is_file() and (os.path.isdir(dst) or dst[-1] in ('\\', '/')):
            dst = os.path.join(dst, self.name)

        return dst

    def process(self, dst, func, if_exists='error', arg1=None, arg2=None):
        r"""copy或move的本质底层实现
        :param if_exists:
            'error': （默认）如果要替换的目标文件已经存在，则报错
            'replace': 替换
            'ignore': 忽略、不处理
            'backup': 备份后写入
        :param func: 传入arg1和arg2参数，可以自定义
            默认分别是self和dst的fullpath
        """
        dst = Path(self.abs_dstpath(dst))
        need_run = True

        if dst.exists():
            if dst == self and arg1 is None and arg2 is None:
                # 同一个文件，估计只是修改大小写名称，不做任何特殊处理，准备直接跑函数
                # 200601周一19:23：要补arg1、arg2的判断，不然Path.write会出错
                pass
            elif if_exists == 'error':
                raise FileExistsError(f'目标文件已存在： {self} — {func.__name__} —> {dst}')
            elif if_exists == 'replace':  # None的话相当于replace，但是不会事先delete，可能会报错
                dst.delete()
            elif if_exists == 'ignore':
                need_run = False
            elif if_exists == 'backup':
                dst.backup(if_exists='backup')
                dst.delete()

        if need_run:
            dst.ensure_dir(pathtype='dir' if self.is_dir() else 'Path')
            if arg1 is None: arg1 = self.fullpath
            if arg2 is None: arg2 = dst.fullpath
            func(arg1, arg2)
            return dst
        else:
            return self

    def ensure_dir(self, pathtype=None):
        r"""确保path中指定的dir都存在
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
                pathtype = 'file' if self.suffix else 'dir'
            dirname = self.fullpath if pathtype == 'dir' else self.dirname
            if not os.path.exists(dirname):
                os.makedirs(dirname)

    def copy(self, dst, if_exists='error'):
        """复制文件"""
        if self.is_dir():
            return self.process(dst, shutil.copytree, if_exists)
        elif self.is_file():
            return self.process(dst, shutil.copy2, if_exists)

    def move(self, dst, if_exists='error'):
        """移动文件"""
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
        r"""对文件末尾添加时间戳备份，也可以使用自定义标记tail

        :param tail: 自定义添加后缀
            tail为None时，默认添加特定格式的时间戳
        :param if_exists: 备份的目标文件名存在时的处理方案
        :param move:
            是否删除原始文件

        # TODO：有个小bug，如果在不同时间实际都是相同一个文件，也会被不断反复备份
        #    如果想解决这个，就要读取目录下最近的备份文件对比内容了
        """
        # 1、判断自身文件是否存在
        if not self.exists():
            return None

        # 2、计算出新名称
        if not tail:
            tail = self.mtime.strftime(' %y%m%d-%H%M%S')  # 时间戳
        name, ext = os.path.splitext(self.fullpath)
        dst = name + tail + ext

        # 3、备份就是特殊的copy操作
        if move:
            return self.move(dst, if_exists)
        else:
            return self.copy(dst, if_exists)

    # 五、其他综合性功能

    def read(self, *, encoding=None):
        if self.is_file():  # 如果存在这样的文件，那就读取文件内容
            # 获得文件扩展名，并统一转成小写
            name, suffix = self.fullpath, self.suffix.lower()
            if suffix == '.pkl':  # pickle库
                with open(name, 'rb') as f:
                    return pickle.load(f)
            elif suffix == '.json':
                import json
                with open(name, 'rb') as f:
                    return json.loads(f.read())
            elif suffix in ('.jpg', '.jpeg', '.png', '.bmp'):
                with open(name, 'rb') as fp:
                    return fp.read()
            else:
                with open(name, 'rb') as f:
                    bstr = f.read()
                if not encoding: encoding = get_encoding(bstr)
                s = bstr.decode(encoding=encoding, errors='ignore')
                if '\r' in s: s = s.replace('\r\n', '\n')  # 如果用\r\n作为换行符会有一些意外不好处理
                return s
        else:  # 非文件对象
            raise ValueError('文件不存在，无法读取。')

    def write(self, ob, *, encoding='utf8', if_exists='error', etag=False):
        """
        :param ob: 写入的内容
            如果要写txt文本文件且ob不是文本对象，只会进行简单的字符串化
        :param encoding: 强制写入的编码
        :param if_exists: 如果文件已存在，要进行的操作
        :param etag: 创建的文件，是否需要再进一步重命名为etag名称
        :return: 返回写入的文件名，这个主要是在写临时文件时有用
        """

        # 1、核心写入功能
        def data2file(ob, path):
            """将ob写入文件path，如果path已存在，也会被直接覆盖"""
            path.ensure_dir(pathtype='file')
            name, suffix = path.fullpath, path.suffix
            if suffix == '.pkl':
                with open(name, 'wb') as f:
                    pickle.dump(ob, f)
            elif suffix == '.json':
                with open(name, 'w') as f:
                    json.dump(ob, f)
            elif isinstance(ob, bytes):
                with open(name, 'wb') as f:
                    f.write(ob)
            else:  # 其他类型认为是文本类型
                with open(name, 'w', errors='ignore', encoding=encoding) as f:
                    f.write(str(ob))

        # 2、推导出目标文件的完整名，并判断是否存在，进行不同处理
        self.process(self, data2file, if_exists, arg1=ob, arg2=self)
        if etag:
            from pyxllib.debug.qiniu_ import get_etag
            # TODO etag如果出现重复文件，是可以ignore的，但为了rename函数能成功返回目标Path，还是先执行replace吧
            return self.rename(get_etag(self.fullpath) + self.suffix, if_exists='replace')
        else:
            return self

    def explorer(self, proc='explorer'):
        """使用windows的explorer命令打开文件
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


if __name__ == '__main__':
    demo_path_rename()
