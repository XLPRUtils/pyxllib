#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/08/15 00:59

from pyxllib.basic import get_encoding, Path


def is_labelme_json_data(data):
    """ 是labelme的标注格式
    :param data: dict
    :return: True or False
    """
    return set(data.keys()) == set('version flags shapes imagePath imageData imageHeight imageWidth'.split())


def reduce_labelme_jsonfile(jsonpath, encoding=None):
    p = Path(jsonpath)
    if not encoding:
        encoding = get_encoding(p.fullpath)
    data = p.read(encoding=encoding, mode='.json')
    if is_labelme_json_data(data) and data['imageData']:
        data['imageData'] = None
        p.write(data, if_exists='replace')