#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/30 20:39


import subprocess


import requests


try:
    import qiniu
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'qiniu'])
    import qiniu


def get_etag(arg):
    """七牛原有etag功能基础上做封装
    :param arg: 支持bytes二进制、文件、url地址
    """
    from io import BytesIO
    from pyxllib.debug.judge import is_url, is_file

    if isinstance(arg, bytes):  # 二进制数据
        return qiniu.utils.etag_stream(BytesIO(arg))
    elif is_file(arg):  # 输入是一个文件
        return qiniu.etag(arg)
    elif is_url(arg):  # 输入是一个网页上的数据源
        return get_etag(requests.get(arg).content)
    elif isinstance(arg, str):  # 明文字符串转二进制
        return get_etag(arg.encode('utf8'))
    else:
        raise TypeError('不识别的数据类型')


def test_etag():
    print(get_etag(r'\chematom{+8}{2}{8}{}'))
    # Fjnu-ZXyDxrqLoZmNJ2Kj8FcZGR-

    print(get_etag(__file__))
    # 每次代码改了这段输出都是不一样的


if __name__ == '__main__':
    test_etag()
