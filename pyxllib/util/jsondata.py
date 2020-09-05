#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/08/15 00:59

from pyxllib.basic import get_encoding, Path

____labelme_json = """
"""


def is_labelme_json_data(data):
    """ 是labelme的标注格式
    :param data: dict
    :return: True or False
    """
    has_keys = set('version flags shapes imagePath imageData imageHeight imageWidth'.split())
    return not (has_keys - data.keys())


def reduce_labelme_jsonfile(jsonpath):
    p = Path(jsonpath)
    data = p.read(mode='.json')
    if is_labelme_json_data(data) and data['imageData']:
        data['imageData'] = None
        p.write(data, encoding=p.encoding, if_exists='replace')
