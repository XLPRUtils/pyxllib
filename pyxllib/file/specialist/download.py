#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/09 11:02

import requests
from bs4 import BeautifulSoup

from pyxllib.file.specialist.dirlib import File, Dir


def download_file(url, fn=None, *, encoding=None, if_exists=None, ext=None, temp=False):
    """ 类似writefile，只是源数据是从url里下载

    :param url: 数据下载链接
    :param fn: 保存位置，会从url智能提取文件名
    :param if_exists: 详见 File.write 参数解释
    :para temp: 将文件写到临时文件夹
    :return:
    """
    if not fn: fn = url.split('/')[-1]
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
