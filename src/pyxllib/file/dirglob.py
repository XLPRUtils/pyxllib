#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/03 21:17

import os
import re
import shutil
from pyxllib.file.xlpath import XlPath


def recreate_folders(*dsts):
    """重建一个空目录"""
    for dst in dsts:
        try:
            # 删除一个目录（含内容），设置ignore_errors可以忽略目录不存在时的错误
            shutil.rmtree(dst, ignore_errors=True)
            os.makedirs(dst)  # 重新新建一个目录，注意可能存在层级关系，所以要用makedirs
        except TypeError:
            pass


def checkpathfile(name):
    r"""判断环境变量path下是否有name这个文件，有则返回绝对路径，无则返回None
    常用的有：BCompare.exe、Chrome.exe、mogrify.exe、xelatex.exe

    >> checkpathfile('xelatex.exe')
    'C:\\CTEX\\MiKTeX\\miktex\\bin\\xelatex.exe'
    >> checkpathfile('abcd.exe')
    """
    for path in os.getenv('path').split(';'):
        fn = os.path.join(path, name)
        if os.path.exists(fn):
            return fn
    return None


def hasext(f, *exts):
    """判断文件f是否是exts扩展名中的一种，如果不是返回False，否则返回对应的值

    所有文件名统一按照小写处理
    """
    ext = os.path.splitext(f)[1].lower()
    exts = tuple(map(lambda x: x.lower(), exts))
    if ext in exts:
        return ext
    else:
        return False


def isdir(fn):
    """判断输入的是不是合法的路径格式，且存在确实是一个文件夹"""
    try:
        return os.path.isdir(fn)
    except ValueError:  # 出现文件名过长的问题
        return False
    except TypeError:  # 输入不是字符串类型
        return False


def gen_file_filter(s):
    """生成一个文件名过滤函数"""
    if s[0] == '.':
        return lambda x: x.endswith(s)
    else:
        s = s.replace('？', r'[\u4e00-\u9fa5]')  # 中文问号可以匹配任意中文字符
        return lambda x: re.search(s, x)


def getfiles(root, filter_rule=None):
    r""" 对os.walk进一步封装，返回所有匹配的文件

    可以这样遍历一个目录下的所有文件：
    for f in getfiles(r'C:\pycode\code4101py', r'.py'):
        print(f)
    筛选规则除了“.+后缀”，还可以写正则匹配
    """
    if isinstance(filter_rule, str):
        filter_rule = gen_file_filter(filter_rule)

    for root, _, files in os.walk(root, filter_rule):
        for f in files:
            if filter_rule and not filter_rule(f):
                continue
            yield root + '\\' + f


def tex_content_filefilter(f):
    """只获取正文类tex文件"""
    if f.endswith('.tex') and 'Conf' not in f and 'settings' not in f:
        return True
    else:
        return False


def tex_conf_filefilter(f):
    """只获取配置类tex文件"""
    if f.endswith('.tex') and ('Conf' in f or 'settings' in f):
        return True
    else:
        return False


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
