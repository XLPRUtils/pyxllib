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
import pprint

import base64
import json
import statistics
import requests

from pyxllib.prog.newbie import round_int
from pyxllib.prog.pupil import check_install_package
from pyxllib.prog.specialist import XlOsEnv
from pyxllib.debug.specialist import TicToc
from pyxllib.algo.geo import xywh2ltrb, rect_bounds
from pyxllib.cv.expert import xlcv
from pyxllib.data.xlprdb import XlprDb
from pyxllib.file.specialist import get_etag, XlPath


def __1_转类labelme标注():
    """ 将百度api获得的各种业务结果，转成类似labelme的标注格式

    统一成一种风格，易分析的结构
    """


def loc2points(loc, ratio=1):
    """ 百度的location格式转为 {'points': [[a, b], [c, d]], 'shape_type: 'rectangle'} """
    # 目前关注到的都是矩形，不知道有没可能有其他格式
    l, t, r, b = xywh2ltrb([loc['left'], loc['top'], loc['width'], loc['height']])
    if ratio != 1:
        l, t, r, b = [round_int(x * ratio) for x in [l, t, r, b]]
    return {'points': [[l, t], [r, b]],
            'shape_type': 'rectangle'}


def polygon2rect(pts):
    l, t, r, b = rect_bounds(pts)
    return [[l, t], [r, b]]


def zoom_point(pt, ratio):
    if ratio != 1:
        return {k: round_int(v * ratio) for k, v in pt.items()}


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

    def __init__(self):
        # 1 默认值
        self.db = None
        self._aipocr = None
        self._mathpix_header = None
        self._priu_header = None
        self._priu_host = None

        # 2 如果环境变量预存了账号信息，自动加载使用
        accounts = XlOsEnv.get('XlAiAccounts', decoding=True)
        if accounts:
            if 'aipocr' in accounts:
                self.setup_aipocr(**accounts['aipocr'])
            if 'mathpix' in accounts:
                self.setup_mathpix(**accounts['mathpix'])
            if 'priu' in accounts:
                self.setup_priu(**accounts['priu'])

    def setup_database(self, dbfile=XlPath.userdir() / '.xlpr/xlpr.db',
                       check_same_thread=False):
        """ 是否将运行结果存储到数据库中

        :param check_same_thread: 如果可能有多个线程同时操作数据库，特别是用flask做前端时，需要关闭
        """
        self.db = XlprDb(dbfile, check_same_thread=check_same_thread)
        self.db.init_jpgimages_table()
        self.db.init_aipocr_table()

    def setup_aipocr(self, app_id, api_key, secret_key):
        """
        注：带透明底的png百度api识别不了，要先转成RGB格式
        """
        check_install_package('aip', 'baidu-aip')
        import aip

        self._aipocr = aip.AipOcr(str(app_id), api_key, secret_key)

    def setup_mathpix(self, app_id, app_key):
        self._mathpix_header = {'Content-type': 'application/json'}
        self._mathpix_header.update({'app_id': app_id, 'app_key': app_key})

    def setup_priu(self, token, host='localhost:5003'):
        """ 福建省模式识别与图像理解重点实验室 """
        self._priu_header = {'Content-type': 'application/json'}
        self._priu_header.update({'Token': token})
        self._priu_host = host

    def run_with_db(self, func, buffer, options=None, *,
                    mode_name=None,
                    use_exists_record=True,
                    update_record=True,
                    save_buffer_threshold_size=0):
        """ 配合database数据库的情况下，调用API功能

        :param mode_name: 可以指定存入数据库的功能名，默认用func的名称
        :param use_exists_record: 如果数据库里已有记录，是否直接复用
        :param update_record: 新结果是否记录到数据库
        :param save_buffer_threshold_size: buffer小余多少，才存储进数据库

        """

        # 1 预处理，参数标准化
        if options is None:
            options = {}
        options = {k: options[k] for k in sorted(options.keys())}  # 对参数进行排序，方便去重

        # 2 调百度识别接口
        if self.db and (use_exists_record or update_record):  # 有开数据库，并且复用和更新至少有项开启了
            if mode_name is None:
                mode_name = func.__name__
            # 如果数据库里有处理过的记录，直接引用
            im_etag = get_etag(buffer)
            if use_exists_record:
                res = self.db.get_aipocr_result(mode_name, im_etag, options)
            else:
                res = None

            # 否则调用百度的接口识别
            # TODO 这里使用协程逻辑最合理但配置麻烦，需要func底层等做协程的适配支持
            #   使用多线程测试了并没有更快，也发现主要耗时是post，数据库不会花太多时间，就先不改动了
            #   等以后数据库大了，看运行是否会慢，可以再测试是否有必要弄协程
            if res is None or 'error_code' in res:
                res = func(buffer, options)
                if len(buffer) < save_buffer_threshold_size:
                    self.db.insert_jpgimage_from_buffer(buffer, etag=im_etag)
                if update_record:
                    self.db.insert_aipocr_record(mode_name, im_etag, options, res, if_exists='REPLACE')
        else:
            res = func(buffer, options)

        return res

    def __B1_通用(self):
        pass

    def general(self, im, options=None):
        """ 通用文字识别（标准含位置版）: https://cloud.baidu.com/doc/OCR/s/vk3h7y58v

        注意只要ratio!=1涉及到缩放的，偏移误差是会变大的~~
        """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.general, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def basicGeneral(self, im, options=None):
        """ 通用文字识别（标准版）: https://cloud.baidu.com/doc/OCR/s/zk3h7xz52
        """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.basicGeneral, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def accurate(self, im, options=None):
        """ 通用文字识别（高精度含位置版）: https://cloud.baidu.com/doc/OCR/s/tk3h7y2aq """
        sz = 10 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=8192, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.accurate, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def basicAccurate(self, im, options=None):
        """ 通用文字识别（高精度版）: https://cloud.baidu.com/doc/OCR/s/1k3h7y3db """
        sz = 10 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=8192, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.basicAccurate, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def webimageLoc(self, im, options=None):
        """ 网络图片文字识别（含位置版）: https://cloud.baidu.com/doc/OCR/s/Nkaz574we """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.webimageLoc, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def webImage(self, im, options=None):
        """ 网络图片文字识别: https://cloud.baidu.com/doc/OCR/s/Sk3h7xyad """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.webImage, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def handwriting(self, im, options=None):
        """ 手写文字识别: https://cloud.baidu.com/doc/OCR/s/hk3h7y2qq """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.handwriting, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def numbers(self, im, options=None):
        """ 数字识别: https://cloud.baidu.com/doc/OCR/s/Ok3h7y1vo """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.numbers, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def qrcode(self, im, options=None):
        """ 二维码识别: https://cloud.baidu.com/doc/OCR/s/qk3h7y5o7 """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.qrcode, buffer, options, save_buffer_threshold_size=sz)

        ls = result_dict['codes_result']
        shapes = []
        for x in ls:
            # assert len(x['text']) == 1, '取到的文本list长度大于1'
            shapes.append({'label': {'text': ','.join(x['text']), 'category': x['type']}})
        result_dict['shapes'] = shapes

        del result_dict['codes_result']
        del result_dict['codes_result_num']
        return result_dict

    def form(self, im, options=None):
        """ 表格文字识别(同步接口): https://cloud.baidu.com/doc/OCR/s/ik3h7xyxf

        备注：
            0、有异步接口
            1、返回是四边形结构，实际值都是矩形，先转成矩形处理了
            2、因为矩形，这个不能处理倾斜的表格
            3、看起来支持多表格
        """
        from bisect import bisect_left
        from pyxllib.file.xlsxlib import Workbook

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
            wb = Workbook()
            ws = wb.active
            for sp in shapes:
                x = sp['label']
                if x['category'] != 'body':
                    continue
                ws.cell(x['row'], x['column'], x['text'])
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
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.form, buffer, options, save_buffer_threshold_size=sz)
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

    def doc_analysis_office(self, im, options=None):
        """ 办公文档识别: https://cloud.baidu.com/doc/OCR/s/ykg9c09ji """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.doc_analysis_office, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word2(result_dict, 1 / ratio, 'results', 'results_num')
        return result_dict

    def seal(self, im, options=None):
        """ 印章识别: https://cloud.baidu.com/doc/OCR/s/Mk3h7y47a """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.seal, buffer, options, save_buffer_threshold_size=sz)

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

    def idcard(self, im, options=None):
        """ 身份证识别: https://cloud.baidu.com/doc/OCR/s/rk3h7xzck """

        def idcard_front(image, options=None):
            return self._aipocr.idcard(image,
                                       options.get('id_card_side', 'front'),  # 默认识别带照片一面
                                       options)

        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(idcard_front, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result')
        return result_dict

    def idcard_back(self, im, options=None):
        def func(image, options=None):
            return self._aipocr.idcard(image,
                                       options.get('id_card_side', 'back'),  # 识别国徽一面
                                       options)

        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(func, buffer, options, save_buffer_threshold_size=sz, mode_name='idcard_back')
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result')
        return result_dict

    def bankcard(self, im, options=None):
        """ 银行卡识别: https://cloud.baidu.com/doc/OCR/s/ak3h7xxg3 """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.bankcard, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_str(result_dict, 1 / ratio, 'result', 'words_result_num')
        return result_dict

    def businessLicense(self, im, options=None):
        """ 营业执照识别: https://cloud.baidu.com/doc/OCR/s/sk3h7y3zs """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.businessLicense, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def businessCard(self, im, options=None):
        """ 名片识别: https://cloud.baidu.com/doc/OCR/s/5k3h7xyi2 """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.businessCard, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_strs(result_dict, 1 / ratio, 'words_result')
        return result_dict

    def passport(self, im, options=None):
        """ 护照识别: https://cloud.baidu.com/doc/OCR/s/Wk3h7y1gi """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.passport, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def HKMacauExitentrypermit(self, im, options=None):
        """ 港澳通行证识别: https://cloud.baidu.com/doc/OCR/s/4k3h7y0ly """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.HKMacauExitentrypermit, buffer, options,
                                       save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def taiwanExitentrypermit(self, im, options=None):
        """ 台湾通行证识别: https://cloud.baidu.com/doc/OCR/s/kk3h7y2yc """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.taiwanExitentrypermit, buffer, options,
                                       save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def householdRegister(self, im, options=None):
        """ 户口本识别: https://cloud.baidu.com/doc/OCR/s/ak3h7xzk7 """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.householdRegister, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def birthCertificate(self, im, options=None):
        """ 出生医学证明识别: https://cloud.baidu.com/doc/OCR/s/mk3h7y1o6 """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.birthCertificate, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def __B3_交通(self):
        pass

    def vehicleLicense(self, im, options=None):
        """ 行驶证识别: https://cloud.baidu.com/doc/OCR/s/yk3h7y3ks """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.vehicleLicense, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def drivingLicense(self, im, options=None):
        """ 驾驶证识别: https://cloud.baidu.com/doc/OCR/s/Vk3h7xzz7 """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.drivingLicense, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def licensePlate(self, im, options=None):
        """ 车牌识别: https://cloud.baidu.com/doc/OCR/s/ck3h7y191 """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.licensePlate, buffer, options, save_buffer_threshold_size=sz)

        d = result_dict['words_result']

        # 近似key_text，但有点不太一样
        # 已测试过 车牌 只能识别一个
        label = {'text': d['number'],
                 'color': d['color']}
        shape = {'label': label,
                 'score': sum(d['probability']) / len(d['probability']),
                 'points': [zoom_point(pt, 1 / ratio) for pt in d['vertexes_location']],
                 'shape_type': 'polygon'}
        result_dict['shapes'] = [shape]

        del result_dict['words_result']
        return result_dict

    def vinCode(self, im, options=None):
        """ VIN码识别: https://cloud.baidu.com/doc/OCR/s/zk3h7y51e """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.vinCode, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def vehicleInvoice(self, im, options=None):
        """ 机动车销售发票识别: https://cloud.baidu.com/doc/OCR/s/vk3h7y4tx """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.vehicleInvoice, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_str(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def vehicleCertificate(self, im, options=None):
        """ 车辆合格证识别: https://cloud.baidu.com/doc/OCR/s/yk3h7y3sc """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.vehicleCertificate, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_str(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def raw_mixed_multi_vehicle(self, im, options=None):
        """ 车辆证照混贴识别: https://cloud.baidu.com/doc/OCR/s/Kksfsbngb """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.mixed_multi_vehicle, buffer, options,
                                       save_buffer_threshold_size=sz)

        if ratio != 1:
            for x in result_dict['words_result']:
                x['location'] = {k: round_int(v / ratio) for k, v in x['location'].items()}

        return result_dict

    def mixed_multi_vehicle(self, im, options=None):
        result_dict = self.raw_mixed_multi_vehicle(im, options)

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

    def vehicle_registration_certificate(self, im, options=None):
        """ 机动车登记证书识别: https://cloud.baidu.com/doc/OCR/s/qknzs5zzo """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.vehicle_registration_certificate, buffer, options,
                                       save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def weightNote(self, im, options=None):
        """ 磅单识别: https://cloud.baidu.com/doc/OCR/s/Uksfp9far """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.weightNote, buffer, options, save_buffer_threshold_size=sz)

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

    def multipleInvoice(self, im, options=None):
        """ 智能财务票据识别: https://cloud.baidu.com/doc/OCR/s/7ktb8md0j """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.multipleInvoice, buffer, options, save_buffer_threshold_size=sz)

        shapes = []
        for x in result_dict['words_result']:
            shape = {'category': x['type'], 'label': {'text': ''}}
            shape.update(loc2points(x, 1 / ratio))
            shapes.append(shape)

        result_dict['shapes'] = shapes
        del result_dict['words_result']
        del result_dict['words_result_num']
        return result_dict

    def quotaInvoice(self, im, options=None):
        """ 定额发票识别: https://cloud.baidu.com/doc/OCR/s/lk3h7y4ev """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.quotaInvoice, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_str(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def invoice(self, im, options=None):
        """ 通用机打发票识别: https://cloud.baidu.com/doc/OCR/s/Pk3h7y06q """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.invoice, buffer, options, save_buffer_threshold_size=sz)

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

    def trainTicket(self, im, options=None):
        """ 火车票识别: https://cloud.baidu.com/doc/OCR/s/Ok3h7y35u """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.trainTicket, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_str(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def taxiReceipt(self, im, options=None):
        """ 出租车票识别: https://cloud.baidu.com/doc/OCR/s/Zk3h7xxnn """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.taxiReceipt, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_str(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def airTicket(self, im, options=None):
        """ 飞机行程单识别: https://cloud.baidu.com/doc/OCR/s/Qk3h7xzro """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.airTicket, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.dict_str(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def onlineTaxiItinerary(self, im, options=None):
        """ 网约车行程单识别: https://cloud.baidu.com/doc/OCR/s/Bkocoyu9n """

        def func(image, options=None):
            return self._aipocr.onlineTaxiItinerary(image)

        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(func, buffer, options,
                                       save_buffer_threshold_size=sz, mode_name='onlineTaxiItinerary')

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

    def receipt(self, im, options=None):
        """ receipt: https://cloud.baidu.com/doc/OCR/s/6k3h7y11b """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.receipt, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def __B5_医疗(self):
        pass

    def raw_medicalInvoice(self, im, options=None):
        """ 医疗发票识别: https://cloud.baidu.com/doc/OCR/s/yke30j1hq """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.medicalInvoice, buffer, options, save_buffer_threshold_size=sz)
        return result_dict

    def raw_medicalDetail(self, im, options=None):
        """ 医疗费用明细识别: https://cloud.baidu.com/doc/OCR/s/Bknjnwlyj """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.medicalDetail, buffer, options, save_buffer_threshold_size=sz)
        return result_dict

    def raw_insuranceDocuments(self, im, options=None):
        """ 保险单识别: https://cloud.baidu.com/doc/OCR/s/Wk3h7y0eb """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.insuranceDocuments, buffer, options, save_buffer_threshold_size=sz)
        return result_dict

    def __B6_教育(self):
        pass

    def docAnalysis(self, im, options=None):
        """ 试卷分析与识别: https://cloud.baidu.com/doc/OCR/s/jk9m7mj1l """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.docAnalysis, buffer, options, save_buffer_threshold_size=sz)

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

    def formula(self, im, options=None):
        """ 公式识别: https://cloud.baidu.com/doc/OCR/s/Ok3h7xxva """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.formula, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def __B7_其他(self):
        pass

    def meter(self, im, options=None):
        """ 仪器仪表盘读数识别: https://cloud.baidu.com/doc/OCR/s/Jkafike0v """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.meter, buffer, options, save_buffer_threshold_size=sz)
        result_dict = ToLabelmeLike.list_word(result_dict, 1 / ratio, 'words_result', 'words_result_num')
        return result_dict

    def raw_facade(self, im, options=None):
        """ 门脸文字识别: https://cloud.baidu.com/doc/OCR/s/wk5hw3cvo """
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(self._aipocr.facade, buffer, options, save_buffer_threshold_size=sz)
        return result_dict

    def facade(self, im, options=None):
        result_dict = self.raw_facade(im, options)

        shapes = []
        for x in result_dict['words_result']:
            shape = {'label': {'text': x['words'], 'score': x['score']}}
            shapes.append(shape)

        result_dict['shapes'] = shapes
        del result_dict['words_result']
        del result_dict['words_result_num']
        return result_dict

    def __C_其他三方库接口(self):
        pass

    def mathpix_latex(self, im, options=None):
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

        def func(buffer, options=None):
            image_uri = f'data:image/jpg;base64,' + base64.b64encode(buffer).decode()
            r = requests.post('https://api.mathpix.com/v3/latex',
                              data=json.dumps({'src': image_uri}),
                              headers=self._mathpix_header)
            return json.loads(r.text)

        # mathpix的接口没有说限制图片大小，但我还是按照百度的规范处理下更好
        sz = 4 * 1024 ** 2
        buffer, ratio = xlcv.to_buffer2(im, min_length=15, max_length=4096, limit_b64buffer_size=sz)
        result_dict = self.run_with_db(func, buffer, options, save_buffer_threshold_size=sz, mode_name='mathpix_latex')
        result_dict['position'] = zoom_point(result_dict['position'], ratio)
        return result_dict

    def __D_福建省模式识别与图像理解重点实验室(self):
        pass

    def _priu_read_image(self, im):
        # 不进行尺寸、文件大小压缩，这样会由平台上负责进行ratio缩放计算
        buffer, ratio = xlcv.to_buffer2(im, b64encode=True)
        return buffer.decode()

    def priu_api(self, mode, im, options=None):
        """ 借助服务器来调用该类中含有的其他函数接口

        使用该接口的时候，因为服务器一般会有图片等的备份，所以本接口默认不对图片进行备份
        另外因为可能会调用自定义的模型功能，自定义的模型可能迭代较快，不适合在数据库缓存结果，所以也不记录json结果
        """
        data = {'mode': mode, 'image': self._priu_read_image(im)}
        if options:
            data['options'] = options
        r = requests.post(f'http://{self._priu_host}/api/priu_api', json.dumps(data), headers=self._priu_header)
        return json.loads(r.text)

    def common_ocr(self, im):
        data = {'image': self._priu_read_image(im)}
        r = requests.post(f'http://{self._priu_host}/api/common_ocr', json.dumps(data), headers=self._priu_header)
        return json.loads(r.text)

    def hesuan_layout(self, im):
        data = {'image': self._priu_read_image(im)}
        r = requests.post(f'http://{self._priu_host}/api/hesuan_layout', json.dumps(data), headers=self._priu_header)
        return json.loads(r.text)

    def rec_singleline(self, im, mode='general'):
        """ 通用的识别一张图的所有文本，并拼接到一起 """
        text = ''  # 因图片太小等各种原因，没有识别到结果，默认就设空值
        try:
            d = self.priu_api(mode, im)
            if 'shapes' in d:
                text = ' '.join([sp['label']['text'] for sp in d['shapes']])
        except requests.exceptions.ConnectionError:
            pass
        return text


def demo_aipocr():
    from tqdm import tqdm
    import pprint
    import re

    from pyxlpr.data.labelme import LabelmeDict
    from pyxllib.debug.specialist import browser

    xlapi = XlAiClient()
    # xlapi.setup_database()
    xlapi._priu_host = 'localhost:5003'

    mode = 'general'

    _dir = XlPath("/home/chenkunze/data/aipocr_test")
    fmode = re.sub(r'^raw_', r'', mode)
    fmode = {'basicAccurate': 'accurate',
             'basicGeneral': 'general',
             'webimageLoc': 'webImage',
             'idcard_back': 'idcard',
             'vat_invoice_verification': 'vatInvoice',
             'form2htmltables': 'form',
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


def demo_priu():
    xlapi = XlAiClient()
    # xlapi._priu_host = 'localhost:5003'

    # d = xlapi.common_ocr('/home/chenkunze/data/aipocr_test/01通用/general/1.jpg')
    # d = xlapi.hesuan_layout('/home/chenkunze/data/hesuan/data/test.jpg')
    # d = xlapi.priu_api('general', '/home/chenkunze/data/aipocr_test/01通用/general/1.jpg')
    d = xlapi.rec_singleline('/home/chenkunze/data/aipocr_test/01通用/general/1.jpg')

    pprint.pprint(d)


if __name__ == '__main__':
    with TicToc(__name__):
        # demo_aipocr()
        demo_priu()
