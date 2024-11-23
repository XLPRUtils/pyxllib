#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/09 11:02

import logging
import os
import shutil
from typing import Callable, List, Optional
from urllib import request

import requests
from bs4 import BeautifulSoup

from pyxllib.file.specialist import File, Dir, refinepath


def download_file(url, fn=None, *, encoding=None, if_exists=None, ext=None, temp=False):
    r""" 类似writefile，只是源数据是从url里下载

    :param url: 数据下载链接
    :param fn: 保存位置，会从url智能提取文件名
    :param if_exists: 详见 File.write 参数解释
    :para temp: 将文件写到临时文件夹
    :return:

    >> download_file(image_url)  # 保存在当前工作目录下
    D:/home/chenkunze/slns/xlproject/xlsln/ckz2024/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg
    >> download_file(image_url, fn=r"D:/home/chenkunze/slns/xlproject/xlsln/ckz2024/a.png")  # 指定路径
    D:/home/chenkunze/slns/xlproject/xlsln/ckz2024/a.png
    >> download_file(image_url, fn=r"D:/home/chenkunze/slns/xlproject/xlsln/ckz2024")  # 暂不支持目录
    ValueError: 不能用目录初始化一个File对象 D:\home\chenkunze\slns\xlproject\xlsln\ckz2023
    """
    if not fn: fn = refinepath(url.split('/')[-1])[-80:]  # 这里故意截断文件名最长80个字符
    root = Dir.TEMP if temp else None
    fn = File(fn, root, suffix=ext).write(requests.get(url).content,
                                          encoding=encoding, if_exists=if_exists, etag=(not fn))
    return fn.to_str()


def read_from_ubuntu(url):
    """从paste.ubuntu.com获取数据"""
    if isinstance(url, int):  # 允许输入一个数字ID来获取网页内容
        url = 'https://paste.ubuntu.com/' + str(url) + '/'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    content = soup.find_all(name='div', attrs={'class': 'paste'})[2]
    return content.get_text()


def download(
        url: str, dir: str, *, filename: Optional[str] = None, progress: bool = True
) -> str:
    """ 取自fvcore：Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Download a file from a given URL to a directory. If file exists, will not
        overwrite the existing file.

    Args:
        url (str):
        dir (str): the directory to download the file
        filename (str or None): the basename to save the file.
            Will use the name in the URL if not given.
        progress (bool): whether to use tqdm to draw a progress bar.

    Returns:
        str: the path to the downloaded file or the existing one.
    """
    os.makedirs(dir, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1]
        assert len(filename), "Cannot obtain filename from url {}".format(url)
    fpath = os.path.join(dir, filename)
    logger = logging.getLogger(__name__)

    if os.path.isfile(fpath):
        logger.info("File {} exists! Skipping download.".format(filename))
        return fpath

    tmp = fpath + ".tmp"  # download to a tmp file first, to be more atomic.
    try:
        logger.info("Downloading from {} ...".format(url))
        if progress:
            import tqdm

            def hook(t: tqdm.tqdm) -> Callable[[int, int, Optional[int]], None]:
                last_b: List[int] = [0]

                def inner(b: int, bsize: int, tsize: Optional[int] = None) -> None:
                    if tsize is not None:
                        t.total = tsize
                    t.update((b - last_b[0]) * bsize)  # type: ignore
                    last_b[0] = b

                return inner

            with tqdm.tqdm(  # type: ignore
                    unit="B", unit_scale=True, miniters=1, desc=filename, leave=True
            ) as t:
                tmp, _ = request.urlretrieve(url, filename=tmp, reporthook=hook(t))

        else:
            tmp, _ = request.urlretrieve(url, filename=tmp)
        statinfo = os.stat(tmp)
        size = statinfo.st_size
        if size == 0:
            raise IOError("Downloaded an empty file from {}!".format(url))
        # download to tmp first and move to fpath, to make this function more
        # atomic.
        shutil.move(tmp, fpath)
    except IOError:
        logger.error("Failed to download {}".format(url))
        raise
    finally:
        try:
            os.unlink(tmp)
        except IOError:
            pass

    logger.info("Successfully downloaded " + fpath + ". " + str(size) + " bytes.")
    return fpath


def ensure_localfile(localfile, from_url, *, if_exists=None, progress=True):
    """ 判断本地文件 localfile 是否存在，如果不存在，自动从指定的 from_url 下载下来

    TODO 增加 md5校验、自动解压 等功能

    :param if_exists: 参数含义见 file.exist_preprcs
        使用 'replace' 可以强制下载重置文件
    :param progress: 是否显示下载进度

    >> ensure_localfile('ufo.csv', r'https://gitee.com/code4101/TestData/raw/master/ufo.csv')
    """
    path, file = str(localfile), File(localfile)

    if file.exist_preprcs(if_exists):
        dirname, name = os.path.split(path)
        download(from_url, dirname, filename=name, progress=progress)
    return localfile


def ensure_localdir(localdir, from_url, *, if_exists=None, progress=True, wrap=0):
    """ 判断本地目录 localdir 是否存在，如果不存在，自动从指定的 from_url 下载下来

    :param from_url: 相比 ensure_localfile，这个链接一般是一个压缩包，下载到本地后要解压到目标目录
    :param wrap:
        wrap=1，如果压缩包里的文件有多个，可以多加一层目录包装起来
        wrap<0，有的从url下载的压缩包，还会自带一层目录，为了避免冗余，要去掉
    """
    from pyxllib.file.packlib import unpack_archive
    d = Dir(localdir)
    if not d.is_dir():
        if d.exist_preprcs(if_exists):
            pack_file = download(from_url, d.parent, progress=progress)
            unpack_archive(pack_file, localdir, wrap=wrap)
            os.remove(pack_file)

    return localdir


def get_font_file(name):
    """ 获得指定名称的字体文件

    :param name: 记得要写后缀，例如 "simfang.ttf"
        simfang.ttf，仿宋
        msyh.ttf，微软雅黑
    """
    from pyxllib.file.specialist import ensure_localfile, XlPath

    # 0 当前目录有，则优先返回当前目录的文件
    p = XlPath(name)
    if p.is_file():
        return p

    # 1 windows直接找系统的字体目录
    font_file = XlPath(f'C:/Windows/Fonts/{name}')
    if font_file.is_file():
        return font_file

    # 2 否则下载到.xlpr/fonts
    # 注意不能下载到C:/Windows/Fonts，会遇到权限问题，报错
    font_file = XlPath.userdir() / f'.xlpr/fonts/{name}'
    # 去github上paddleocr项目下载
    # from_url = f'https://raw.githubusercontent.com/code4101/data1/main/fonts/{name}'
    from_url = f'https://xmutpriu.com/download/fonts/{name}'
    try:
        ensure_localfile(font_file, from_url)
    except TimeoutError as e:
        raise TimeoutError(f'{font_file} 下载失败：{from_url}')

    return font_file
