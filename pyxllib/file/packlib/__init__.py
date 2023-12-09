#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/02/15 09:51

import os
import shutil
from tarfile import TarFile
# 这个包里的ZipFile是拷贝后修改过的，不影响标准库zipfile里的功能
import pyxllib.file.packlib.zipfile as zipfile
from pyxllib.file.packlib.zipfile import ZipFile

from pyxllib.file.specialist import XlPath, reduce_dir_depth
from pyxllib.prog.pupil import inject_members

""" 问题：py官方的ZipFile解压带中文的文件会乱码

解决办法及操作流程
1、定期将官方的zipfile.py文件更新到pyxllib.file.packlib.zipfile
2、将代码中两处cp437都替换为gbk
"""


def unpack_zipfile(filename, extract_dir):
    """ 为了修复zipfile底层的中文解压乱码问题，修改了shutil._UNPACK_FORMATS的底层功能
    """
    from shutil import ReadError

    zip = zipfile.ZipFile(filename)
    if not zipfile.is_zipfile(filename):
        raise ReadError("%s is not a zip file" % filename)
    try:
        for info in zip.infolist():
            name = info.filename

            # don't extract absolute paths or ones with .. in them
            if name.startswith('/') or '..' in name:
                continue

            target = os.path.join(extract_dir, *name.split('/'))
            if not target:
                continue

            os.makedirs(XlPath(target).parent, exist_ok=True)
            if not name.endswith('/'):
                # file
                data = zip.read(info.filename)
                f = open(target, 'wb')
                try:
                    f.write(data)
                finally:
                    f.close()
                    del data
    finally:
        zip.close()


# 解决unzip中文乱码问题
shutil._UNPACK_FORMATS['zip'] = (['.zip'], unpack_zipfile, [], "ZIP file")


def _parse_depth(names):
    """ 判断有几层目录都只有独立的一个目录

    :param names: 一组相对路径名
    :return: 如果直接目录下文件已经不唯一，返回0

    >>> _parse_depth(['a'])
    1
    >>> _parse_depth(['a', 'a/1'])
    2
    >>> _parse_depth(['a', 'a/b', 'a/b/1', 'a/b/2'])
    2
    >>> _parse_depth(['a', 'a/b', 'a/b/1', 'a/c/3'])
    1
    """
    parents = []
    for name in names:
        p = XlPath(name)
        ps = [x.as_posix() for x in p.parents]
        parents.append(ps[-2::-1] + [p.as_posix()])
    i = 0
    while len({x[i] for x in parents if len(x) > i}) == 1:
        i += 1
    return i


def _unpack_base(packfile, namelist, format=None, extract_dir=None, wrap=0):
    """ 解压

    :param packfile: 压缩包本身的文件路径名称，在wrap=1时可能会用到，用来添加一个同名目录
    :param namelist: 压缩包里所有文件的相对路径
    :param format: 压缩包类型，比如zip、tar，可以不输入，此时直接采用shutil.unpack_archive里的判断机制
    :param extract_dir: 解压目标目录，可以不输入，会自动判断
        自动解压规则：压缩包里目录不唯一，则以压缩包本身的名称创建一个目录存放
            否则以压缩包里的目录名为准
    :param wrap: 解压时倾向的处理机制
        0，默认，不进行任何处理
        1，如果压缩包里一级目录下不止一个对象，使用压缩包本身的名称再包一层
        -1，如果压缩包里一级目录下只有一个目录，则丢弃这个目录，只取里面的内容出来
            同理，可能这一个目录里的文件，还是只有一个目录，此时支持设置-2、-3等解开上限
            也就-1比较常用，-2、-3等很少见了
        TODO 还有一种解压机制，只取里面的部分文件，有空可以继续调研扩展下
    :return: 无论怎样，最后都是放到一个目录里，返回改目录路径
    """

    # 1 默认解压目录是self所在目录，注意这点跟shutil.unpack_archive的当前工作目录不同
    p0 = XlPath(packfile)
    if extract_dir is None:
        extract_dir = p0.parent

    depth = _parse_depth(namelist)

    # 2 是否要加目录层级，或者去掉目录层级
    if wrap == 1 and depth == 0:
        # 压缩包里直接目录下有多个文件，且指定了wrap，则加一层目录
        shutil.unpack_archive(packfile, extract_dir / p0.stem, format)
    elif wrap < 0 < depth:
        # 压缩包有depth层冗余目录，并且指定wrap要解开这几层目录
        # 先正常解压（每种压缩包格式内部处理机制都有点不太一样，所以先解压，然后用目录文件功能处理更合适）
        shutil.unpack_archive(packfile, extract_dir, format)
        reduce_dir_depth(extract_dir, unwrap=-wrap)
    else:
        # 正常解压
        shutil.unpack_archive(packfile, extract_dir, format)


class XlZipFile(ZipFile):

    def infolist2(self, prefix=None, zipinfo=True):
        """>> self.infolist2()  # getinfo的多文件版本
             1           <ZipInfo filename='[Content_Types].xml' compress_type=deflate file_size=1495 compress_size=383>
             2                    <ZipInfo filename='_rels/.rels' compress_type=deflate file_size=590 compress_size=243>
            ......
            20            <ZipInfo filename='word/fontTable.xml' compress_type=deflate file_size=1590 compress_size=521>
            21               <ZipInfo filename='docProps/app.xml' compress_type=deflate file_size=720 compress_size=384>

        :param prefix:
            可以筛选文件的前缀，例如“word/”可以筛选出word目录下的
        :param zipinfo:
            返回的list每个元素是zipinfo数据类型
        """
        ls = self.infolist()
        if prefix:
            ls = list(filter(lambda t: t.filename.startswith(prefix), ls))
        if not zipinfo:
            ls = list(map(lambda x: x.filename, ls))
        return ls

    def unpack(self, extract_dir=None, format='zip', wrap=0):
        _unpack_base(self.filename, self.namelist(), format, extract_dir, wrap)


class XlTarFile(TarFile):

    def unpack(self, extract_dir=None, format='tar', wrap=0):
        _unpack_base(self.name, self.getnames(), format, extract_dir, wrap)


inject_members(XlZipFile, ZipFile)
inject_members(XlTarFile, TarFile)


def unpack_archive(filename, extract_dir=None, format=None, *, wrap=1):
    """ 对shutil.unpack_archive的扩展，增加了一个wrap的接口功能 """
    if format is None:
        format = shutil._find_unpack_format(str(filename).lower())

    if format.endswith('zip'):
        XlZipFile(filename).unpack(extract_dir, format, wrap)
    elif format.endswith('tar'):
        XlTarFile(filename).unpack(extract_dir, format, wrap)
    else:
        # 其他还没扩展的格式，不支持wrap功能，但仍然可以使用shutil标准的接口解压
        shutil.unpack_archive(filename, extract_dir, format)


def compress_to_zip(source_path, target_zip_path=None, wrap=None,
                    ignore_func=None):
    """ 压缩指定的文件或文件夹为ZIP格式。

    :param str source_path: 要压缩的文件或文件夹路径
    :param str target_zip_path: 目标ZIP文件路径（可选）
    :param str wrap: 在ZIP文件内部创建的目录名，所有内容将被放在这个目录下
    :param func ignore_func: 忽略文件的函数，输入目录或文件的路径，返回True表示不取用
    """
    # 根据输入路径生成默认的目标ZIP文件路径
    if target_zip_path is None:
        # 获取输入路径的基本名称（文件名或文件夹名）
        base_name = os.path.basename(source_path)
        # 为基本名称添加'.zip'后缀
        target_zip_path = os.path.join(os.path.dirname(source_path), f"{base_name}.zip")

    # 创建一个ZipFile对象来写入压缩文件
    with XlZipFile(target_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 如果是文件夹，则遍历文件夹中的所有文件和子文件夹
        if os.path.isdir(source_path):
            for root, _, files in os.walk(source_path):
                if ignore_func and ignore_func(root):
                    continue

                for file in files:
                    if ignore_func and ignore_func(file):
                        continue

                    file_path = os.path.join(root, file)
                    # 计算文件在ZIP中的相对路径
                    arcname = os.path.relpath(file_path, source_path)
                    if wrap:
                        arcname = os.path.join(wrap, arcname)
                    zipf.write(file_path, arcname)
        # 如果是文件，则直接将文件添加到ZIP中
        elif os.path.isfile(source_path):
            arcname = os.path.basename(source_path)
            if wrap:
                arcname = os.path.join(wrap, arcname)
            zipf.write(source_path, arcname)

    return target_zip_path


def smart_compress_zip(root, paths, check_func=None):
    """ 智能压缩，比compress_to_zip功能会更强些

    :param root: 要有个根目录，才知道paths里添加到压缩包的时候存储到怎样的相对目录下
    :param paths: 所有待添加到压缩包的文件或目录
    :param check_func: 规则判断函数，输入文件或目录的路径
        return
            0，明确不要
            1，要使用，但要递归内部文件精细检查
            2，全部都要
    """

    if check_func is None:
        def check_func(path):
            p = XlPath(path)

            if set(p.parts) & {'__pycache__', '.git', '.idea', 'dist', 'build', '.pytest_cache'}:
                return 0

            if p.parts[-1].endswith('.egg-info'):
                return 0

            return 1

    def add_dir(dir):
        for subroot, _, names in os.walk(dir):
            for name in names:
                file = XlPath(subroot) / name
                zipf.write(file, file.relpath(root))

    def add_path(path):
        tag = check_func(path)

        if path.is_file():
            if tag:
                zipf.write(path, path.relpath(root))
        elif path.is_dir():
            if tag == 2:
                add_dir(path)
            elif tag == 1:
                for subroot, _, names in os.walk(path):
                    tag2 = check_func(subroot)
                    if tag2 == 2:
                        add_dir(subroot)
                    elif tag2 == 1:
                        for name in names:
                            file = XlPath(subroot) / name
                            if check_func(file):
                                zipf.write(file, file.relpath(root))

    # 工作目录会自动切换的，这里取工作目录名即可
    if not isinstance(paths, list):
        paths = [paths]

    root = XlPath(root)
    num = len(paths)
    if num > 1:
        zipf = XlZipFile(root.name + '.zip', 'w', zipfile.ZIP_DEFLATED)
    elif num == 1:
        zipf = XlZipFile(XlPath(paths[0]).name + '.zip', 'w', zipfile.ZIP_DEFLATED)
    else:
        return

    for subroot in paths:
        add_path(XlPath(subroot))

    zipf.close()
