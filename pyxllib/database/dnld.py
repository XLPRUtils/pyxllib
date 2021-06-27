#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/25 17:02

"""
文件下载相关功能库
"""

import logging
import os
import shutil
from typing import Callable, List, Optional
from urllib import request

# import subprocess


# 这个系列功能，默认存储到 os.getenv("FVCORE_CACHE", "~/.torch/iopath_cache")
# try:
#     import iopath
# except ModuleNotFoundError:
#     subprocess.run(['pip3', 'install', 'iopath'])
#
# from iopath.common.file_io import HTTPURLHandler
# from iopath.common.file_io import PathManager as PathManagerBase
#
# PathManager = PathManagerBase()
# PathManager.register_handler(HTTPURLHandler())


from pyxllib.file.specialist import File


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


def ensure_localfile(localfile, from_url, *, if_exists=None):
    """ 判断本地文件 localfile 是否存在，如果不存在，自动从指定的 from_url 下载下来

    TODO 增加 md5校验、自动解压 等功能

    :param if_exists: 参数含义见 file.exist_preprcs
        使用 'replace' 可以强制下载重置文件

    >> ensure_localfile(File('ufo.csv'), r'https://gitee.com/code4101/TestData/raw/master/ufo.csv')
    """
    path, file = str(localfile), File(localfile)

    if file.exist_preprcs(if_exists):
        dirname, name = os.path.split(path)
        download(from_url, dirname, filename=name)
    return localfile
