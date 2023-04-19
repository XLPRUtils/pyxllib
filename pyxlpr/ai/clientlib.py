#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 21:14

"""
百度人工智能API，常用URL
使用文档：https://cloud.baidu.com/doc/OCR/s/Ek3h7xypm
调用次数：https://console.bce.baidu.com/ai/?_=1653139065257#/ai/ocr/overview/index


"""
import os
import base64
import json
import pprint
import statistics
import time

import cv2
import numpy as np
import requests

from pyxllib.prog.newbie import round_int
from pyxllib.prog.pupil import check_install_package, is_url
from pyxllib.prog.xlosenv import XlOsEnv
from pyxllib.algo.geo import xywh2ltrb, rect_bounds
from pyxllib.file.specialist import XlPath, get_etag
from pyxllib.cv.expert import xlcv


def __1_转类labelme标注():
    """ 将百度api获得的各种业务结果，转成类似labelme的标注格式

    统一成一种风格，易分析的结构
    """


def loc2points(loc, ratio=1):
    """ 百度的location格式转为 {'points': [[a, b], [c, d]], 'shape_type: 'rectangle'} """
    # 目前关注到的都是矩形，不知道有没可能有其他格式
    ltrb = xywh2ltrb([loc['left'], loc['top'], loc['width'], loc['height']])
    if ratio != 1:
        ltrb = [x * ratio for x in ltrb]
    l, t, r, b = round_int(ltrb, ndim=1)
    return {'points': [[l, t], [r, b]],
            'shape_type': 'rectangle'}


def loc2points2(loc, ratio=1):
    """ 百度的location格式转为 {'points': [[a, b], [c, d]], 'shape_type: 'rectangle'} """
    # 目前关注到的都是矩形，不知道有没可能有其他格式
    ltrb = xywh2ltrb([loc['left'], loc['top'], loc['right'] - loc['left'], loc['bottom'] - loc['top']])
    if ratio != 1:
        ltrb = [x * ratio for x in ltrb]
    l, t, r, b = round_int(ltrb, ndim=1)
    return {'points': [[l, t], [r, b]],
            'shape_type': 'rectangle'}


def polygon2rect(pts):
    l, t, r, b = rect_bounds(pts)
    return [[l, t], [r, b]]


def zoom_point(pt, ratio):
    """ 返回的还是{x: 123, y: 456}的字典结构 """
    if ratio != 1:
        return {k: round_int(v * ratio) for k, v in pt.items()}
    return pt


def zoom_point2(pt, ratio):
    """ 输入x,y的字典坐标，返回的还是[123, 456]的list结构 """
    pt = [pt['x'], pt['y']]
    if ratio != 1:
        return [round_int(v * ratio) for v in pt]
    return pt


def zoom_labelme(d, ratio):
    """ 对labelme的标注进行缩放 """
    ratio2 = 1 / ratio
    for sp in d['shapes']:
        sp['points'] = [[round_int(p[0] * ratio2), round_int(p[1] * ratio2)] for p in sp['points']]
    d['imageHeight'] = round_int(d['imageHeight'] * ratio2)
    d['imageWidth'] = round_int(d['imageWidth'] * ratio2)
    return d


def labelmelike_extend_args(core_func):
    """ 扩展 main_func 函数，支持一些通用上下游切面功能

    支持 main_keys，主要数据字段名称
    支持 remove_keys，批量移除一些不需要的参数值
    """

    def wrapper(data, ratio, main_key, remove_keys=None, *, clear_empty_shape=False):
        """
        :param clear_empty_shape: 在考虑要不要把清除空shape，四边形转矩形等功能写到这里，暂时还未写对应功能
        """
        # 1 转主要的数据结构
        if main_key in data:
            data['shapes'] = core_func(data[main_key], ratio)

        # 2 删除不需要的键值
        if remove_keys is None:
            _remove_keys = set()
        elif not isinstance(remove_keys, (list, tuple, set)):
            _remove_keys = [remove_keys]
        else:
            _remove_keys = remove_keys
        for k in ({main_key} | set(_remove_keys)):
            if k in data:
                del data[k]

        return data

    return wrapper


class ToLabelmeLike:
    """ 注意配合装饰器的使用
    这里每个函数，只要实现输入核心识别结果清单，返回shapes的代码即可。上下游一些切面操作是统一的。
    """

    @staticmethod
    @labelmelike_extend_args
    def list_word(ls, ratio):
        shapes = []
        for w in ls:
            shape = {}
            if 'location' in w:
                shape.update(loc2points(w['location'], ratio))
            if 'words' in w:
                # shape['label'] = json.dumps({'text': x['words']}, ensure_ascii=False)
                shape['label'] = {'text': w['words']}  # 不转字符串
            shapes.append(shape)
        return shapes

    @staticmethod
    @labelmelike_extend_args
    def list_word2(ls, ratio):
        shapes = []
        for w in ls:
            shape = {}
            shape.update(loc2points(w['words']['words_location'], ratio))
            shape['label'] = {'text': w['words']['word'], 'category': w['words_type']}
            shapes.append(shape)
        return shapes

    @staticmethod
    @labelmelike_extend_args
    def dict_word(d, ratio):
        shapes = []
        for k, w in d.items():
            shape = {'label': {'category': k}}
            if 'location' in w:
                shape.update(loc2points(w['location'], ratio))
            if 'words' in w:
                shape['label']['text'] = w['words']
            shapes.append(shape)
        return shapes

    @staticmethod
    @labelmelike_extend_args
    def dict_str(d, ratio):
        return [{'label': {'text': v, 'category': k}} for k, v in d.items()]

    @staticmethod
    @labelmelike_extend_args
    def dict_strs(d, ratio):
        shapes = []
        for k, texts in d.items():
            shapes.append({'label': {'category': k, 'text': ','.join(texts)}})
        return shapes


def __2_定制不同输出格式():
    pass


class XlAiClient:
    """
    封装该类
        目的1：合并输入文件和url的识别
        目的2：带透明底的png百度api识别不了，要先转成RGB格式
    """
    META_OPTS_NAME = {'record_files',  # 是否记录传入的图片等数据
                      'record_xlapi',  # 是否记录api处理结果
                      'use_exists_xlapi',  # 是否使用已有的api结果
                      'record_xlserver',  # 是否记录调用纪录
                      }

    def __init__(self, auto_login=True, check=True):
        # 1 默认值
        self.db = None

        self._aipocr = None

        self._mathpix_header = None

        self._priu_header = None
        self._priu_host = None

        # 2 如果环境变量预存了账号信息，自动加载使用
        accounts = XlOsEnv.get('XlAiAccounts', decoding=True)
        if accounts and auto_login:
            if 'aipocr' in accounts:
                self.login_aipocr(**accounts['aipocr'])
            if 'mathpix' in accounts:
                self.login_mathpix(**accounts['mathpix'])
            if 'priu' in accounts:
                self.login_priu(**accounts['priu'], check=check)

    def __A1_登录账号(self):
        pass

    def set_database(self, db):
        """ 是否关联数据库，查找已运行过的结果，或存储运行结果

        from pyxllib.data.pglib import XlprDb
        db = XlprDb.connect()
        self.setup_database(db)
        """
        self.db = db

    def login_aipocr(self, app_id, api_key, secret_key):
        """
        注：带透明底的png百度api识别不了，要先转成RGB格式
        """
        check_install_package('aip', 'baidu-aip')
        import aip

        self._aipocr = aip.AipOcr(str(app_id), api_key, secret_key)

    def login_mathpix(self, app_id, app_key):
        self._mathpix_header = {'Content-type': 'application/json'}
        self._mathpix_header.update({'app_id': app_id, 'app_key': app_key})

    def login_priu(self, token, host=None, *, check=True):
        """ 福建省模式识别与图像理解重点实验室

        :param str|list[str] host:
            str, 主机IP，比如'118.195.202.82'
            list, 也可以输入一个列表，会从左到右依次尝试链接，使用第一个能成功链接的ip，常用语优先选用局域网，找不到再使用公网接口
        """
        self._priu_header = {'Content-type': 'application/json'}
        self._priu_header.update({'Token': token})

        if host is None:
            # 优先尝试局域网链接，如果失败则尝试公网链接。公网默认用https，如果需要效率可以自己输入http的host。
            hosts = ['172.16.170.136', 'https://xmutpriu.com']
        elif isinstance(host, str):
            hosts = [host]
        elif isinstance(host, (list, tuple)):
            hosts = host
        else:
            raise TypeError

        # 确保都有http或https前缀
        for i, host in enumerate(hosts):
            if 'http' not in host:
                hosts[i] = f'http://{host}'

        if check:
            connent = False
            for host in hosts:
                try:
                    if '欢迎来到 厦门理工' in requests.get(f'{host}/test_page', timeout=5).text:
                        self._priu_host = host
                        connent = True
                        break
                except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
                    continue

            if not connent:
                raise requests.exceptions.ConnectionError('PRIU接口登录失败')
            return connent
        else:
            self._priu_host = hosts[0]  # 不检查的场景，一般都是局域网内使用
            return None

    def __A2_调整图片和关联数据库(self):
        pass

    @classmethod
    def adjust_image(cls, in_, flags=1, *, b64decode=True, to_buffer=True, b64encode=False,
                     min_length=15, max_length=4096,
                     limit_b64buffer_size=4 * 1024 ** 2):
        """ 这里是使用API接口，比较通用的一套图片处理操作

        :param in_: 可以是本地文件，也可以是图片url地址，也可以是Image对象
            注意这个函数，输入是url，也会获取重置图片数据上传
            如果为了效率明确只传url，可以用aip.AipOcr原生的相关url函数
        :param b64decode: 如果输入是bytes类型，是否要用b64解码，默认需要
        :return: 返回图片文件二进制值的buffer, 缩放系数(小余1是缩小，大于1是放大)
        """
        # 1 取不同来源的数据
        # 下面应该是比较通用的一套操作，如果有特殊接口，可以另外处理，不一定要通过该接口处理图片
        if isinstance(in_, bytes):
            im = xlcv.read_from_buffer(in_, flags, b64decode=b64decode)
        elif is_url(in_):
            im = xlcv.read_from_url(in_, flags)
        else:
            im = xlcv.read(in_, flags)
        origin_height = im.shape[0]

        if im.dtype == 'uint16':
            im = cv2.convertScaleAbs(im)

        # 2 图片尺寸不符合要求，要缩放
        if min_length or max_length:
            im = xlcv.adjust_shape(im, min_length, max_length)

        # 3 图片文件不能过大，要调整
        if limit_b64buffer_size:
            # b64后大小固定会变4/3，所以留给原文件的大小是要缩水，只有0.75；再以防万一总不能卡得刚刚好，所以设为0.74
            im = xlcv.reduce_filesize(im, limit_b64buffer_size * 0.74)
        current_height = im.shape[0]

        if to_buffer:
            im = xlcv.to_buffer(im, '.jpg', b64encode=b64encode)
        ratio = current_height / origin_height
        return im, ratio

    def _divide_options_with_meta(self, options, meta_opts):
        """ 会改变原始输入的两个字典值，但是新返回的才是真正有效的字典 """
        for name in self.META_OPTS_NAME:
            if name in options:
                meta_opts[name] = options[name]
                del options[name]

        options = options or {}
        options = {k: options[k] for k in sorted(options.keys())}  # 对参数进行排序，方便记录到数据库后标记唯一

        return options, meta_opts

    def run_aipocr_with_db(self, func, buffer, options=None, **meta_opts):
        """ 这一套主要是适配百度视觉api接口的情况，基本都是输入一个buffer和一个options
        如果自己模型有特殊接口范式，可以额外定制，比如 run_textapi_with_db 等，可以拷贝这个函数后做定制微调

        :param func: 被封装的带执行的api函数
        :param buffer: 图片二进制数据
        :param options: 执行api功能的配套采纳数
        :param meta_opts:
            record_files: 是否保存数据文件
                False, 0，表示不存储
                True，表示存储，但默认只存储4MB以内的文件
                int，可以写一个整数，表示只存储多大Byte范围内的文件
            record_xlapi: 新结果是否记录到数据库
            use_exists_xlapi: 如果数据库里已有记录，是否直接复用
                注意看函数实现，但这个机制生效已经获得res的时候，变不会显式执行record_files
                这是出于效率的考虑，一般情况下，这种应该是已经存储过文件了
                如果要怕以前执行过api但没存储files，可以关闭use_exists_xlapi，就可以显式重新执行保存一次
            mode_name: 可以指定存入数据库的功能名，默认用func的名称
        """
        # 1 参数处理
        options, meta_opts = self._divide_options_with_meta(options, meta_opts)
        if self.db is None:  # 没有数据库的时候直接执行函数返回结果就好
            return func(buffer, options)

        # 以下元参数默认值，根据不同的api接口需求，可以设置不同的默认值
        record_files = meta_opts.get('record_files', True)  # 是否保存数据文件
        if record_files is True:
            record_files = 4 * 1024 ** 2
        record_xlapi = meta_opts.get('record_xlapi', True)
        use_exists_xlapi = meta_opts.get('use_exists_xlapi', True)
        mode_name = meta_opts.get('mode_name', func.__name__)

        image_etag = None  # 先不计算etag，只在需要的时候再计算
        res = None  # api结果

        # 2 核心函数
        if use_exists_xlapi:  # 定义如何获取已有的识别结果
            image_etag = image_etag or get_etag(buffer)  # trick: 使用这个机制确保etag从头到尾不会重复计算，需要的话只会计算一次
            res = self.db.get_xlapi_record(mode=mode_name, image=image_etag, **options)

        # 否则调用百度的接口识别
        # TODO 这里使用协程逻辑最合理但配置麻烦，需要func底层等做协程的适配支持
        #   使用多线程测试了并没有更快，也发现主要耗时是post，数据库不会花太多时间，就先不改动了
        #   等以后数据库大了，看运行是否会慢，可以再测试是否有必要弄协程
        if res is None or 'error_code' in res:
            tt = time.time()
            res = func(buffer, options)
            elapse_ms = round_int(1000 * (time.time() - tt))

            if len(buffer) < record_files:  # 定义数据的缓存方式
                image_etag = image_etag or get_etag(buffer)
                self.db.insert_row2files(buffer, etag=image_etag, name='.jpg')

            if record_xlapi:  # 定义api的执行结果的保存方式
                image_etag = image_etag or get_etag(buffer)
                input = {'mode': mode_name, 'image': image_etag}
                if options:
                    input.update(options)
                xlapi_id = self.db.insert_row2xlapi(input, res, elapse_ms, on_conflict='REPLACE')
                res['xlapi_id'] = xlapi_id

        # 3 收尾
        if 'log_id' in res:  # 有xlapi_id的标记，就不用百度原本的log_id了
            del res['log_id']

        return res

    def __B1_通用(self):
        pass

    def general(self, image, **options):
        """ 通用文字识别（标准含位置版）: https://cloud.baidu.com/doc/OCR/s/vk3h7y58v
        500次/天赠送 + 超出按量计费

        注意只要ratio!=1涉及到缩放的，偏移误差是会变大的~~
        """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.general, buffer, options)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def basicGeneral(self, image, **options):
        """ 通用文字识别（标准版）: https://cloud.baidu.com/doc/OCR/s/zk3h7xz52
        5万次/天赠送 + 超出按量计费
        """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.basicGeneral, buffer, options)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def accurate(self, image, **options):
        """ 通用文字识别（高精度含位置版）: https://cloud.baidu.com/doc/OCR/s/tk3h7y2aq """
        sz = 10 * 1024 ** 2
        buffer, ratio = self.adjust_image(image, max_length=8192, limit_b64buffer_size=sz, to_buffer=True)
        result_dict = self.run_aipocr_with_db(self._aipocr.accurate, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def basicAccurate(self, image, **options):
        """ 通用文字识别（高精度版）: https://cloud.baidu.com/doc/OCR/s/1k3h7y3db """
        sz = 10 * 1024 ** 2
        buffer, ratio = self.adjust_image(image, max_length=8192, limit_b64buffer_size=sz, to_buffer=True)
        result_dict = self.run_aipocr_with_db(self._aipocr.basicAccurate, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def webimageLoc(self, image, **options):
        """ 网络图片文字识别（含位置版）: https://cloud.baidu.com/doc/OCR/s/Nkaz574we """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.webimageLoc, buffer, options)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def webImage(self, image, **options):
        """ 网络图片文字识别: https://cloud.baidu.com/doc/OCR/s/Sk3h7xyad """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.webImage, buffer, options)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def handwriting(self, image, **options):
        """ 手写文字识别: https://cloud.baidu.com/doc/OCR/s/hk3h7y2qq """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.handwriting, buffer, options)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def numbers(self, image, **options):
        """ 数字识别: https://cloud.baidu.com/doc/OCR/s/Ok3h7y1vo """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.numbers, buffer, options)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def qrcode(self, image, **options):
        """ 二维码识别: https://cloud.baidu.com/doc/OCR/s/qk3h7y5o7 """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.qrcode, buffer, options)

        ls = result_dict['codes_result']
        shapes = []
        for x in ls:
            # assert len(x['text']) == 1, '取到的文本list长度大于1'
            shapes.append({'label': {'text': ','.join(x['text']), 'category': x['type']}})
        result_dict['shapes'] = shapes

        del result_dict['codes_result']
        del result_dict['codes_result_num']
        return result_dict

    def form(self, image, **options):
        """ 表格文字识别(同步接口): https://cloud.baidu.com/doc/OCR/s/ik3h7xyxf

        备注：
            0、有异步接口
            1、返回是四边形结构，实际值都是矩形，先转成矩形处理了
            2、因为矩形，这个不能处理倾斜的表格
            3、看起来支持多表格
        """
        from bisect import bisect_left
        from pyxllib.file.xlsxlib import XlWorkbook

        # 1 单表处理功能
        def zoom_coords(table, ratio):
            for k, xs in table.items():
                if k == 'vertexes_location':
                    table[k] = [zoom_point(pt, ratio) for pt in table[k]]
                else:
                    for x in table[k]:
                        x['vertexes_location'] = [zoom_point(pt, ratio) for pt in x['vertexes_location']]

        def parse_split_points(cells, cls='row', name='y'):
            # 计算行列数量，初始化空list
            n = max([x[cls] for x in cells]) + 2
            spans = [[] for i in range(n)]
            # 加入所有行列数据
            for x in cells:
                spans[x[cls]].append(x['vertexes_location'][0][name])
            spans[-1].append(table['vertexes_location'][2][name])
            # 计算平均行列位置
            spans = [statistics.mean(vs) for vs in spans if vs]
            # 合并临近的行列
            min_gap = 3  # 两行/列间距小余min_gap像素，合并为1行/列
            spans = [v for a, v in zip([0] + spans, spans) if v - a > min_gap]
            # 计算分割线（以中间为准）
            spans = [round_int((a + b) / 2) for a, b in zip(spans, spans[1:])]
            return spans

        def location2rowcol(loc, rowspan, colspan):
            p1, p2 = loc[0], loc[2]
            pos = {
                'row': bisect_left(rowspan, p1['y']) + 1,  # 我的数据改成从1开始编号了
                'column': bisect_left(colspan, p1['x']) + 1,
                'end_row': bisect_left(rowspan, p2['y']),
                'end_column': bisect_left(colspan, p2['x']),
            }
            return pos

        def parse_table(table):
            # 1 计算行、列分界线
            rowspan = parse_split_points(table['body'], 'row', 'y')
            colspan = parse_split_points(table['body'], 'column', 'x')

            # 2 shapes
            shapes = []
            # 这个表格暂时转labelme格式，但泛用来说，其实不应该转labelme，它有其特殊性
            # location虽然给的是四边形，但看目前数据其实就是矩形
            for k, xs in table.items():
                if k == 'vertexes_location':
                    sp = {'label': {'text': '', 'category': 'table'},
                          'points': polygon2rect([(p['x'], p['y']) for p in table['vertexes_location']]),
                          'shape_type': 'rectangle'}
                    shapes.append(sp)
                else:
                    for x in xs:
                        label = {'text': x['words'], 'probability': x['probability'], 'category': k}
                        label.update(location2rowcol(x['vertexes_location'], rowspan, colspan))
                        sp = {'label': label,
                              'points': polygon2rect([(p['x'], p['y']) for p in x['vertexes_location']]),
                              'shape_type': 'rectangle'}
                        shapes.append(sp)

            # 3 tables
            # 3.1 表格主体内容
            wb = XlWorkbook()
            ws = wb.active
            for sp in shapes:
                x = sp['label']
                if x['category'] != 'body':
                    continue
                # ws.cell(x['row'], x['column'], x['text'])
                cel = ws.cell(x['row'], x['column']).mcell()
                cel.value = (cel.value or '') + x['text']
                if x['end_row'] - x['row'] + x['end_column'] - x['column'] > 0:
                    ws.merge_cells(start_row=x['row'], start_column=x['column'],
                                   end_row=x['end_row'], end_column=x['end_column'])
            # wb.save('/home/chenkunze/data/aipocr_test/a.xlsx')  # debug: 保存中间结果查看

            # 3.2 其他内容整体性拼接
            htmltable = ['<div>']
            header = '<br/>'.join([x['words'] for x in table['header']])
            footer = '<br/>'.join([x['words'] for x in table['footer']])
            if header:
                htmltable.append(f'<p>{header}</p>')
            htmltable.append(ws.to_html())
            if footer:
                htmltable.append(f'<p>{footer}</p>')
            htmltable.append('</div>')

            return shapes, '\n'.join(htmltable)

        # 2 主体解析功能

        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.form, buffer, options)
        shapes = []
        htmltables = []
        for table in result_dict['forms_result']:
            if ratio != 1:
                zoom_coords(table, 1 / ratio)
            _shapes, htmltable = parse_table(table)
            shapes += _shapes
            htmltables.append(htmltable)

        # 3 收尾
        result_dict['shapes'] = shapes
        result_dict['htmltables'] = htmltables
        del result_dict['forms_result']
        del result_dict['forms_result_num']
        return result_dict

    def doc_analysis_office(self, image, **options):
        """ 办公文档识别: https://cloud.baidu.com/doc/OCR/s/ykg9c09ji """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.doc_analysis_office, buffer, options)
        result_dict = ToLabelmeLike.list_word2(result_dict, 1 / ratio, 'results', 'results_num')
        return result_dict

    def seal(self, image, **options):
        """ 印章识别: https://cloud.baidu.com/doc/OCR/s/Mk3h7y47a """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.seal, buffer, options)

        shapes = []
        for x in result_dict['result']:
            shape = {'label': {}}
            shape.update(loc2points(x['location'], 1 / ratio))
            shape['label']['text'] = x['major']['words']
            shape['label']['minor'] = ','.join(x['minor'])
            shape['label']['category'] = x['type']
            shapes.append(shape)

        result_dict['shapes'] = shapes
        del result_dict['result']
        del result_dict['result_num']
        return result_dict

    def __B2_卡证(self):
        pass

    def idcard(self, image, **options):
        """ 身份证识别: https://cloud.baidu.com/doc/OCR/s/rk3h7xzck """

        def idcard_front(image, options=None):
            return self._aipocr.idcard(image,
                                       options.get('id_card_side', 'front'),  # 默认识别带照片一面
                                       options)

        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(idcard_front, buffer, options)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result')
        return result_dict

    def idcard_back(self, image, **options):
        def func(image, options=None):
            return self._aipocr.idcard(image,
                                       options.get('id_card_side', 'back'),  # 识别国徽一面
                                       options)

        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(func, buffer, options, mode_name='idcard_back')
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result')
        return result_dict

    def bankcard(self, image, **options):
        """ 银行卡识别: https://cloud.baidu.com/doc/OCR/s/ak3h7xxg3 """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.bankcard, buffer, options)
        result_dict = ToLabelmeLike.dict_str(result_dict, 1 / ratio, 'result', 'words_result_num')
        return result_dict

    def businessLicense(self, image, **options):
        """ 营业执照识别: https://cloud.baidu.com/doc/OCR/s/sk3h7y3zs """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.businessLicense, buffer, options)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def businessCard(self, image, **options):
        """ 名片识别: https://cloud.baidu.com/doc/OCR/s/5k3h7xyi2 """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.businessCard, buffer, options)
        result_dict = ToLabelmeLike.dict_strs(result_dict, 1 / ratio, 'words_result')
        return result_dict

    def passport(self, image, **options):
        """ 护照识别: https://cloud.baidu.com/doc/OCR/s/Wk3h7y1gi """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.passport, buffer, options)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def HKMacauExitentrypermit(self, image, **options):
        """ 港澳通行证识别: https://cloud.baidu.com/doc/OCR/s/4k3h7y0ly """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.HKMacauExitentrypermit, buffer, options)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def taiwanExitentrypermit(self, image, **options):
        """ 台湾通行证识别: https://cloud.baidu.com/doc/OCR/s/kk3h7y2yc """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.taiwanExitentrypermit, buffer, options)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def householdRegister(self, image, **options):
        """ 户口本识别: https://cloud.baidu.com/doc/OCR/s/ak3h7xzk7 """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.householdRegister, buffer, options)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def birthCertificate(self, image, **options):
        """ 出生医学证明识别: https://cloud.baidu.com/doc/OCR/s/mk3h7y1o6 """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.birthCertificate, buffer, options)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def __B3_交通(self):
        pass

    def vehicleLicense(self, image, **options):
        """ 行驶证识别: https://cloud.baidu.com/doc/OCR/s/yk3h7y3ks """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.vehicleLicense, buffer, options)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def drivingLicense(self, image, **options):
        """ 驾驶证识别: https://cloud.baidu.com/doc/OCR/s/Vk3h7xzz7 """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.drivingLicense, buffer, options)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def licensePlate(self, image, **options):
        """ 车牌识别: https://cloud.baidu.com/doc/OCR/s/ck3h7y191 """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.licensePlate, buffer, options)

        if 'words_result' in result_dict:
            d = result_dict['words_result']
        else:
            d = {'number': '', 'color': '', 'probability': [0], 'vertexes_location': []}

        # 近似key_text，但有点不太一样
        # 已测试过 车牌 只能识别一个
        label = {'text': d['number'],
                 'color': d['color']}
        shape = {'label': label,
                 'score': sum(d['probability']) / len(d['probability']),
                 'points': [zoom_point2(pt, 1 / ratio) for pt in d['vertexes_location']],
                 'shape_type': 'polygon'}
        result_dict['shapes'] = [shape]

        if 'words_result' in result_dict:
            del result_dict['words_result']
        return result_dict

    def vinCode(self, image, **options):
        """ VIN码识别: https://cloud.baidu.com/doc/OCR/s/zk3h7y51e """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.vinCode, buffer, options)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def vehicleInvoice(self, image, **options):
        """ 机动车销售发票识别: https://cloud.baidu.com/doc/OCR/s/vk3h7y4tx """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.vehicleInvoice, buffer, options)
        result_dict = ToLabelmeLike.dict_str(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def vehicleCertificate(self, image, **options):
        """ 车辆合格证识别: https://cloud.baidu.com/doc/OCR/s/yk3h7y3sc """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.vehicleCertificate, buffer, options)
        result_dict = ToLabelmeLike.dict_str(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def raw_mixed_multi_vehicle(self, image, **options):
        """ 车辆证照混贴识别: https://cloud.baidu.com/doc/OCR/s/Kksfsbngb """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.mixed_multi_vehicle, buffer, options)

        if ratio != 1:
            for x in result_dict['words_result']:
                x['location'] = {k: round_int(v / ratio) for k, v in x['location'].items()}

        return result_dict

    def mixed_multi_vehicle(self, image, **options):
        result_dict = self.raw_mixed_multi_vehicle(image, options)

        shapes = []
        for x in result_dict['words_result']:
            shape = {'label': {}}
            shape.update(loc2points(x['location']))
            shape['label']['text'] = json.dumps({w['word_name']: w['word'] for w in x['license_info']},
                                                ensure_ascii=False)
            shape['label']['category'] = x['card_type']
            shape['label']['score'] = x['probability']
            shapes.append(shape)

        result_dict['shapes'] = shapes
        del result_dict['words_result']
        del result_dict['words_result_num']
        return result_dict

    def vehicle_registration_certificate(self, image, **options):
        """ 机动车登记证书识别: https://cloud.baidu.com/doc/OCR/s/qknzs5zzo """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.vehicle_registration_certificate, buffer, options)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def weightNote(self, image, **options):
        """ 磅单识别: https://cloud.baidu.com/doc/OCR/s/Uksfp9far """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.weightNote, buffer, options)

        shapes = []
        for x in result_dict['words_result']:
            for k, vs in x.items():
                shape = {'category': k,
                         'label': ''.join([v['word'] for v in vs])}
                shapes.append(shape)

        result_dict['shapes'] = shapes
        del result_dict['words_result']
        del result_dict['words_result_num']
        return result_dict

    def __B4_财务(self):
        pass

    def multipleInvoice(self, image, **options):
        """ 智能财务票据识别: https://cloud.baidu.com/doc/OCR/s/7ktb8md0j """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.multipleInvoice, buffer, options)

        shapes = []
        for x in result_dict['words_result']:
            shape = {'category': x['type'], 'label': {'text': ''}}
            shape.update(loc2points(x, 1 / ratio))
            shapes.append(shape)

        result_dict['shapes'] = shapes
        del result_dict['words_result']
        del result_dict['words_result_num']
        return result_dict

    def quotaInvoice(self, image, **options):
        """ 定额发票识别: https://cloud.baidu.com/doc/OCR/s/lk3h7y4ev """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.quotaInvoice, buffer, options)
        result_dict = ToLabelmeLike.dict_str(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def invoice(self, image, **options):
        """ 通用机打发票识别: https://cloud.baidu.com/doc/OCR/s/Pk3h7y06q """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.invoice, buffer, options)

        shapes = []
        for k, v in result_dict['words_result'].items():
            shape = {'label': {'category': k}}
            if isinstance(v, list):
                shape['label']['text'] = '\n'.join([w['word'] for w in v])
            else:
                shape['label']['text'] = v
                shapes.append(shape)

        result_dict['shapes'] = shapes
        del result_dict['words_result']
        del result_dict['words_result_num']
        return result_dict

    def trainTicket(self, image, **options):
        """ 火车票识别: https://cloud.baidu.com/doc/OCR/s/Ok3h7y35u """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.trainTicket, buffer, options)
        result_dict = ToLabelmeLike.dict_str(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def taxiReceipt(self, image, **options):
        """ 出租车票识别: https://cloud.baidu.com/doc/OCR/s/Zk3h7xxnn """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.taxiReceipt, buffer, options)
        result_dict = ToLabelmeLike.dict_str(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def airTicket(self, image, **options):
        """ 飞机行程单识别: https://cloud.baidu.com/doc/OCR/s/Qk3h7xzro """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.airTicket, buffer, options)
        result_dict = ToLabelmeLike.dict_str(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def onlineTaxiItinerary(self, image, **options):
        """ 网约车行程单识别: https://cloud.baidu.com/doc/OCR/s/Bkocoyu9n """

        def func(image, options=None):
            return self._aipocr.onlineTaxiItinerary(image)

        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(func, buffer, options, mode_name='onlineTaxiItinerary')

        shapes = []
        for k, v in result_dict['words_result'].items():
            if k == 'items':
                shape = {'label': {'category': 'item'}}
                shape['label']['text'] = json.dumps(v, ensure_ascii=False)
            else:
                shape = {'label': {'category': k}}
                shape['label']['text'] = v
            shapes.append(shape)

        result_dict['shapes'] = shapes
        del result_dict['words_result']
        del result_dict['words_result_num']
        return result_dict

    def receipt(self, image, **options):
        """ receipt: https://cloud.baidu.com/doc/OCR/s/6k3h7y11b """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.receipt, buffer, options)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def __B5_医疗(self):
        pass

    def raw_medicalInvoice(self, image, **options):
        """ 医疗发票识别: https://cloud.baidu.com/doc/OCR/s/yke30j1hq """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.medicalInvoice, buffer, options)
        return result_dict

    def raw_medicalDetail(self, image, **options):
        """ 医疗费用明细识别: https://cloud.baidu.com/doc/OCR/s/Bknjnwlyj """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.medicalDetail, buffer, options)
        return result_dict

    def raw_insuranceDocuments(self, image, **options):
        """ 保险单识别: https://cloud.baidu.com/doc/OCR/s/Wk3h7y0eb """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.insuranceDocuments, buffer, options)
        return result_dict

    def __B6_教育(self):
        pass

    def docAnalysis(self, image, **options):
        """ 试卷分析与识别: https://cloud.baidu.com/doc/OCR/s/jk9m7mj1l
        总量1000次
        """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.docAnalysis, buffer, options)

        shapes = []
        for x in result_dict['results']:
            shape = loc2points(x['words']['words_location'], 1 / ratio)
            # 有line_probability字段，但实际并没有返回置信度~
            shape['label'] = {'category': x['words_type'], 'text': x['words']['word']}
            shapes.append(shape)

        result_dict['shapes'] = shapes
        del result_dict['results']
        del result_dict['results_num']
        return result_dict

    def formula(self, image, **options):
        """ 公式识别: https://cloud.baidu.com/doc/OCR/s/Ok3h7xxva
        总共1000次
        """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.formula, buffer, options)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def __B7_其他(self):
        pass

    def meter(self, image, **options):
        """ 仪器仪表盘读数识别: https://cloud.baidu.com/doc/OCR/s/Jkafike0v """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.meter, buffer, options)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def raw_facade(self, image, **options):
        """ 门脸文字识别: https://cloud.baidu.com/doc/OCR/s/wk5hw3cvo """
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(self._aipocr.facade, buffer, options)
        return result_dict

    def facade(self, image, **options):
        result_dict = self.raw_facade(image, options)

        shapes = []
        for x in result_dict['words_result']:
            shape = {'label': {'text': x['words'], 'score': x['score']}}
            shapes.append(shape)

        result_dict['shapes'] = shapes
        del result_dict['words_result']
        del result_dict['words_result_num']
        return result_dict

    def __C_其他三方api接口(self):
        pass

    def mathpix_latex(self, image, **options):
        """ 调用mathpix识别单张图片的公式

        【return】
        {'auto_rotate_confidence': 0.0003554748584377876,
         'auto_rotate_degrees': 0,
         'detection_list': ['is_printed'],
         'detection_map': {'contains_chart': 0,
                           'contains_diagram': 0,
                           'contains_graph': 0,
                           'contains_table': 0,
                           'is_blank': 0,
                           'is_inverted': 0,
                           'is_not_math': 0,
                           'is_printed': 0.9996553659439087},
         'error': '',
         'latex': '\\left. \\begin{array} { l } { \\alpha / / \\beta } \\\\ { \\gamma '
                  '\\cap \\alpha = a } \\\\ { \\gamma \\cap \\beta = b } \\end{array} '
                  '\\right\\} \\Rightarrow a / / b',
         'latex_confidence': 0.9824940343387425,
         'latex_confidence_rate': 0.9994673295454546,
         'latex_list': [],
         'position': {'height': 160, 'top_left_x': 0, 'top_left_y': 0, 'width': 266},
         'request_id': '2022_05_28_762803ab687ff5ba1d80g'}
        """

        def func(buffer, options):
            image_uri = f'data:image/jpg;base64,' + base64.b64encode(buffer).decode()
            for i in range(10):  # 尝试10次，10次都失败就无语了~
                try:
                    r = requests.post('https://api.mathpix.com/v3/latex',
                                      data=json.dumps({'src': image_uri}),
                                      headers=self._mathpix_header)
                    return json.loads(r.text)
                except requests.exceptions.ConnectionError:
                    pass
            else:
                return {'latex': '', 'error': 'requests.exceptions.ConnectionError'}

        # mathpix的接口没有说限制图片大小，但我还是按照百度的规范处理下更好
        buffer, ratio = self.adjust_image(image)
        result_dict = self.run_aipocr_with_db(func, buffer, options, mode_name='mathpix_latex')
        if 'position' in result_dict:
            result_dict['position'] = zoom_point(result_dict['position'], ratio)
        return result_dict

    def __D_福建省模式识别与图像理解重点实验室(self):
        pass

    def _priu_read_image(self, image, min_length=None, max_length=None,
                         limit_b64buffer_size=None, b64encode=None, **kwargs):
        # 默认不进行尺寸、文件大小压缩，这样会由平台上负责进行ratio缩放计算
        buffer, ratio = self.adjust_image(image,
                                          min_length=min_length, max_length=max_length,
                                          limit_b64buffer_size=limit_b64buffer_size,
                                          b64encode=True, **kwargs)
        assert ratio == 1, f'本地不做缩放，由服务器进行缩放处理'
        return buffer.decode()

    def priu_api(self, mode, image=None, texts=None, options=None, **meta_opts):
        """ 借助服务器来调用该类中含有的其他函数接口

        使用该接口的时候，因为服务器一般会有图片等的备份，所以本接口默认不对图片进行备份
        另外因为可能会调用自定义的模型功能，自定义的模型可能迭代较快，不适合在数据库缓存结果，所以也不记录json结果

        :param mode: 使用的api接口名称
        :param image: image、texts、options都是比较常用的几个参数键值，所以显式地写出来
        :param options: 有时候为了严谨，api相关的参数可能要单独打包
            但其实这里的参数混入meta_opts也没事，底层会有divide拆分函数，能区分开不同的参数类型
        :param meta_opts: 也可以自己额外扩展一些键值，兼容一些特殊的输入范式的api接口
            use_exists_xlapi: 是否复用已有的识别结果，避免重复运算，节约api成本
            record_files: 是否保存图片等数据
            record_xlapi: 是否保存api执行结果
            record_xlserver: 是否保存本次调用纪录
            根据不同的api，这些参数有不同的默认值，但一般都是True
        """
        # 1 统一输入参数的范式
        data = {}
        if image is not None:
            data['image'] = self._priu_read_image(image, b64encode=True)
        if texts is not None:
            data['texts'] = texts
        if options:  # 支持对算法的参数打包，但其实也支持从kwargs直接零散输入
            data['options'] = options
        if meta_opts:
            data.update(meta_opts)

        # TODO 增加超时timeout=seconds参数？
        r = requests.post(f'{self._priu_host}/api/{mode}', json=data, headers=self._priu_header)
        if r.status_code == 200:
            res = json.loads(r.text)
        else:  # TODO 正常状态码不只200，可能还有重定向等某些不一定是错误的状态
            raise requests.exceptions.ConnectionError(r.text)

        # 2 统一返回值的范式，默认都是dict。 有些特殊格式表示是图片，这里会自动做后处理解析。
        if isinstance(res, dict) and len(res) == 1 and 'imageData' in res:
            # 只有一张图片情况的数据，直接返回图片
            return xlcv.read_from_buffer(res['imageData'], b64decode=True)
        else:
            return res

    def common_ocr(self, image, **options):
        """ 通用文字识别 """
        options.update({'image': self._priu_read_image(image, b64encode=True)})
        r = requests.post(f'{self._priu_host}/api/common_ocr', headers=self._priu_header,
                          json=options)
        res = json.loads(r.text)
        return res

    def ocr2texts(self, image, mode='common_ocr', **options):
        """ 通用的识别一张图的所有文本

        >> xlapi.ocr2texts(im)
        ['OCR文字识别软件', '如何将图片转换成Word', '文章来源：OCR文字识别软件', 'OCR']
        """
        texts = []  # 因图片太小等各种原因，没有识别到结果，默认就设空值
        try:
            d = self.priu_api(mode, image, options=options)
            if 'shapes' in d:
                texts = [sp['label']['text'] for sp in d['shapes']]
        except requests.exceptions.ConnectionError:
            pass
        return texts

    def rec_singleline(self, image, mode='common_ocr', **options):
        """ 通用的识别一张图的所有文本，并拼接到一起 """
        texts = self.ocr2texts(image, mode, **options)
        return ' '.join(texts)

    def hesuan_layout(self, image):
        """ 核酸版面分析 """
        r = requests.post(f'{self._priu_host}/api/hesuan_layout', headers=self._priu_header,
                          json={'image': self._priu_read_image(image)})
        return json.loads(r.text)

    def lexical_analysis(self, texts, options=None, return_mode=None):
        """ 词法分析，有分词的效果

        >> lexical_analysis(["今天是个好日子", "天气预报说今天要下雨"])
        [['今天', '是', '个', '好日子'], ['天气预报', '说', '今天', '要', '下雨']]
        """
        # TODO 可以增加接口参数，配置返回值类型。其他api同理。
        data = {'texts': texts}
        if options:
            data['options'] = options
        r = requests.post(f'{self._priu_host}/api/lac', json=data, headers=self._priu_header)
        res = json.loads(r.text)

        if return_mode == 'raw':
            pass
        else:
            res = [x['word'] for x in res]
        return res

    def sentiment_classify(self, texts, options=None):
        """ 情感分析

        返回的值越大表示句子的情感越积极
        >> sentiment_classify(['今天是个好日子', '天气预报说今天要下雨'])
        [0.4555, 0.2513]
        """
        data = {'texts': texts}
        if options:
            data['options'] = options
        r = requests.post(f'{self._priu_host}/api/senta_bilstm', json=data, headers=self._priu_header)
        return json.loads(r.text)

    def humanseg(self, image):
        """ 人像抠图

        TODO 还没测过特大、特小图会不会有问题~
        """
        im, _ = self.adjust_image(image, 1, to_buffer=False)  # 不能用xlcv.read，因为也可能作为服务端接口，输入buffer格式
        mask = self.priu_api('deeplabv3p_xception65_humanseg', im)
        new_im = np.concatenate([im, np.expand_dims(mask, axis=2)], axis=2)  # 变成BGRA格式图片
        return new_im

    def det_face(self, image):
        """ 人脸识别 """
        lmdict = self.priu_api('ultra_light_fast_generic_face_detector_1mb_640', image)
        return lmdict

    def super_resolution(self, image, times=1):
        """ 超分辨率（把图片变高清）

        :param times: 放大多少次，每次放大2倍（长宽个放大2倍）
        """
        for _ in range(times):
            image = self.priu_api('falsr_c', image)
        return image

    def rec_speech(self, audio_file):
        """ 语音识别 """
        if os.path.isfile(audio_file):
            audio = base64.b64encode(XlPath(audio_file).read_bytes()).decode()
        else:
            raise NotImplementedError
        text = self.priu_api('u2_conformer_wenetspeech', audio=audio)[0]
        return text


def demo_aipocr():
    import pprint
    import re

    from pyxlpr.data.labelme import LabelmeDict

    xlapi = XlAiClient()
    # xlapi.setup_database()
    xlapi._priu_host = 'http://localhost:5003'

    mode = 'general'

    _dir = XlPath("/home/chenkunze/data/aipocr_test")
    fmode = re.sub(r'^raw_', r'', mode)
    fmode = {'basicAccurate': 'accurate',
             'basicGeneral': 'general',
             'webimageLoc': 'webImage',
             'idcard_back': 'idcard',
             'vat_invoice_verification': 'vatInvoice',
             'mathpix_latex': 'formula',
             }.get(fmode, fmode)
    files = _dir.glob_images(f'*/{fmode}/**/*')

    for f in list(files):
        # 1 处理指定图片
        # if f.stem != '824ef2b9a5f05422199107721d299f30':
        #     continue

        # 2 检查字典
        print(f.as_posix())
        d = getattr(xlapi, mode)(f.as_posix())

        # browser.html(d['htmltables'][0])
        print()
        pprint.pprint(d)
        print('- ' * 20)

        # 3 前置的错误图可以删除；有shapes的可以转labelme；非labelme格式上面print后直接退出
        if 'error_code' in d:
            f.delete()
        elif d.get('shapes', 0):
            # tolabelme
            lmdata = LabelmeDict.gen_data(f)
            lmdata['shapes'] = d['shapes']
            for sp in lmdata['shapes']:
                sp['label'] = json.dumps(sp['label'], ensure_ascii=False)
            f.with_suffix('.json').write_json(lmdata)
            break
        else:
            break
