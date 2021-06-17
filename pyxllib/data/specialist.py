#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 11:16

import logging
import os
import shutil
from typing import Callable, List, Optional
from urllib import request

import numpy as np

from pyxllib.file.pupil import struct_unpack
from pyxllib.file.specialist import XlBytesIO


def read_from_dgrl(dgrl):
    """ 解析中科院的DGRL格式数据

    Database Home: http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html
    CASIA 在线和离线中文手写数据库的一些数据读取功能

    参考代码：https://blog.csdn.net/DaGongJiGuoMaLu09/article/details/107050519
        有做了大量简化、工程封装

    TODO 可以考虑做一个返回类似labelme格式的接口，会更通用
        因为有时候会需要取整张原图
        而且如果有整个原图，那么每个文本行用shape形状标记即可，不需要取出子图

    :param dgrl: dgrl 格式的文件，或者对应的二进制数据流
    :return: [(img0, label0), (img1, label1), ...]
    """
    # 输入参数可以是bytes，也可以是文件
    f = XlBytesIO(dgrl)
    # 表头尺寸
    header_size = f.unpack('I')
    # 表头剩下内容，提取 code_length
    header = f.read(header_size - 4)
    code_length = struct_unpack(header[-4:-2], 'H')  # 每个字符存储的字节数，一般都是用gbk编码，2个字节
    # 读取图像尺寸信息，文本行数量
    height, width, line_num = f.unpack('I' * 3)

    # 读取每一行的信息
    res = []
    for k in range(line_num):
        # 读取该行的字符数量
        char_num = f.unpack('I')
        label = f.readtext(char_num, code_length=code_length)
        label = label.replace('\x00', '')  # 去掉不可见字符 \x00，这一步不加的话后面保存的内容会出现看不见的问题

        # 读取该行的位置和尺寸
        y, x, h, w = f.unpack('I' * 4)

        # 读取该行的图片
        bitmap = f.unpack('B' * (h * w))
        bitmap = np.array(bitmap).reshape(h, w)

        res.append((bitmap, label))

    return res


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
