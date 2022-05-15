#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/05/14 10:02

import base64
import datetime
import io
import json

import PIL.Image

from pyxllib.data.sqlite import Connection
from pyxllib.file.specialist import get_etag, XlPath


class XlprDb(Connection):
    """ xlpr统一集中管理的一个数据库 """

    def __init__(self, dbfile=None, *args, **kwargs):
        if dbfile is None:
            dbfile = XlPath.userdir() / '.xlpr/xlpr.db'

        super().__init__(dbfile, *args, **kwargs)

    def init_jpgimages_table(self):
        """ 存储图片数据的表

        其实把图片存储到sql中很不规范，最好是用OSS
            但是相关框架太麻烦，还要花我钱、流量费
            我就暴力存储本地了
        而存储本地为了省空间，我又把所有图片格式转成jpg，这样整体能少消耗些空间

        etag: 来自  get_etag(xlcv.to_buffer(im))

        + phash？
        """
        if self.has_table('jpgimages'):
            return
        cols = ['etag text', 'filesize integer', 'height integer', 'width integer', 'base64_content text']
        cols = ', '.join(cols)
        self.execute(f'CREATE TABLE jpgimages ({cols}, PRIMARY KEY (etag))')
        self.commit()

    def init_aipocr_table(self):
        """ 存储百度识别接口结果

        1、已处理过的图，可以不用重复调用接口
        2、某些图可以手动修改标注，以后再调用接口能获得更好的效果
        """
        if self.has_table('aipocr'):
            return
        cols = ['mode text', 'im_etag text', 'options text', 'result text', 'update_time text']
        cols = ', '.join(cols)
        self.execute(f'CREATE TABLE aipocr ({cols}, PRIMARY KEY (mode, im_etag, options))')
        self.commit()

    def get_aipocr_result(self, mode, im_etag, options):
        """

        加上日期限制？比如一年前的不获取，强制让其重调接口更新？
            思考：不推荐这么干，可以从数据入手，一年前的，手动清理备份到旧数据
        """
        x = self.execute('SELECT result FROM aipocr WHERE mode=? AND im_etag=? AND options=?',
                         (mode, im_etag, json.dumps(options, ensure_ascii=False))).fetchone()
        if x:
            return json.loads(x[0])

    def insert_aipocr_record(self, mode, im_etag, options, result, *, if_exists='IGNORE', **kwargs):
        """ 往数据库记录当前操作内容

        :param replace: 如果已存在，是否强制替换掉
        :return:
            True, 更新成功
            False, 未更新
        """
        kw = {'mode': mode,
              'im_etag': im_etag,
              'options': options,
              'result': result,
              'update_time': datetime.datetime.today().today().strftime('%Y-%m-%d %H:%M:%S')}
        kw.update(kwargs)
        self.insert('aipocr', kw, if_exists=if_exists)
        self.commit()

    def insert_jpgimage_from_buffer(self, buffer, *, etag=None, **kwargs):
        """
        为了运算效率考虑，除了etag需要用于去重，是必填字段
        其他字段默认先不填充计算
        """
        # 1 已经有的不重复存储
        if etag is None:
            etag = get_etag(buffer)

        res = self.execute('SELECT etag FROM jpgimages WHERE etag=?', (etag,)).fetchone()
        if res:
            return

        # 2 没有的图，做个备份
        kwargs['etag'] = etag
        kwargs['base64_content'] = base64.b64encode(buffer)
        kwargs['filesize'] = len(buffer)  # 这个计算成本比较低

        # 本来图片尺寸等想惰性计算，额外写功能再更新的，但发现这个操作似乎挺快的，基本不占性能消耗~~
        im = PIL.Image.open(io.BytesIO(buffer))
        kwargs['height'] = im.height
        kwargs['width'] = im.width

        self.insert('jpgimages', kwargs)
        self.commit()
