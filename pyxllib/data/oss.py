#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/06/15

"""
oss2 · PyPI: https://pypi.org/project/oss2/
"""

from pyxllib.prog.lazyimport import lazy_import

try:
    import oss2
except ModuleNotFoundError:
    oss2 = lazy_import('oss2', 'oss2')

from pyxllib.file.specialist import File


class OssBucket:
    def __init__(self, bucket_name, endpoint, access_key_id, access_key_secret):
        self.bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)

    def upload(self, key, localfile, if_exists='replace', force=False):
        """ 如果云端已存在，默认会进行覆盖

        :param key: 上传后存储的文件名
        :param localfile: 本地文件
        :param if_exists:
            replace, 如果oss上已存在也替换掉
            ignore, 如果oss上已经存在则忽略
        :param force: 是否在导入因意外失败后，重复上传，直到成功为止
            False 时会报错终止程序
        :return: 返回云端是否存在该文件（已存在或者上传后存在，都是True）
        """
        done = False
        while not done:
            try:
                e = self.check_exists(key)
                if e and if_exists == 'ignore': return True
                oss2.resumable_upload(self.bucket, key, localfile)
                done = True
            except Exception as e:
                print('如果一直弹出这一条，则要检查oss账号是否设置正确')
                if not force:
                    raise e
        return True

    def check_exists(self, key):
        """ 检查一个文件在oss是否存在

        :param key:
        :return: 存在返回 GetObjectResult 对象，不存在返回False
        """
        try:
            return self.bucket.get_object(key)
        except oss2.exceptions.NoSuchKey:
            return False

    def download(self, key, localfile):
        File(localfile).ensure_parent()
        if self.check_exists(key):
            return self.bucket.get_object_to_file(key, localfile)
        else:
            return None

    def ObjectIterator(self, **kwargs):
        """ 遍历某个目录下的所有文件（含子目录里的文件）

        >> print(len(list(oss.ObjectIterator(prefix='histudy/tr/teacher/lateximage/'))))
        """
        return oss2.ObjectIterator(self.bucket, **kwargs)
