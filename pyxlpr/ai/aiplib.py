#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 21:14

"""
百度人工智能API接口

【开发层级框架】
BaiduServer，百度官方服务器
    |
    v
aip.AipOcr（百度官方给的py接口）
    |
    v
_AipOcrClient（统一输入接口）
    |
    v
AipOcr
1、+xldb 数据库
2、定制不同的输出格式

【常用URL】
使用文档：https://cloud.baidu.com/doc/OCR/s/Ek3h7xypm
调用次数：https://console.bce.baidu.com/ai/?_=1653139065257#/ai/ocr/overview/index
"""
import pprint

from pyxllib.prog.pupil import check_install_package

check_install_package('aip', 'baidu-aip')

import aip
import base64
import cv2
import json
import datetime

from pyxllib.prog.pupil import is_url
from pyxllib.prog.specialist import XlOsEnv
from pyxllib.debug.specialist import TicToc
from pyxllib.algo.geo import xywh2ltrb, rect_bounds
from pyxllib.cv.expert import xlcv
from pyxllib.data.xlprdb import XlprDb
from pyxllib.file.specialist import get_etag, XlPath


def __1_统一底层接口():
    pass


class _AipOcrClient(aip.AipOcr):
    """ 有些标准库的接口格式我不太喜欢，做些修改封装

    该类可以视为对底层一些算法接口的重新整理
    """

    def idcard(self, image, options=None):
        """ 身份证识别，统一接口范式
        """
        return super(_AipOcrClient, self).idcard(image,
                                                 options.get('id_card_side', 'front'),  # 默认识别带照片一面
                                                 options)

    def idcard_back(self, image, options=None):
        return super(_AipOcrClient, self).idcard(image,
                                                 options.get('id_card_side', 'back'),  # 识别国徽一面
                                                 options)

    def onlineTaxiItinerary(self, image, options=None):
        """ 网约车行程单识别
        """
        return super(_AipOcrClient, self).onlineTaxiItinerary(image)

    def vat_invoice_verification(self, image, options=None):
        """
            增值税发票验真
        """
        options = options or {}

        data = {}
        data['image'] = base64.b64encode(image).decode()

        data.update(options)

        return self._request("https://aip.baidubce.com/rest/2.0/ocr/v1/vat_invoice_verification", data)


def __2_转类labelme标注():
    """ 将百度api获得的各种业务结果，转成类似labelme的标注格式

    统一成一种风格，易分析的结构
    """


def loc2points(loc):
    """ 百度的location格式转为 {'points': [[a, b], [c, d]], 'shape_type: 'rectangle'} """
    # 目前关注到的都是矩形，不知道有没可能有其他格式
    l, t, r, b = xywh2ltrb([loc['left'], loc['top'], loc['width'], loc['height']])
    return {'points': [[l, t], [r, b]],
            'shape_type': 'rectangle'}


def polygon2rect(pts):
    l, t, r, b = rect_bounds(pts)
    return [[l, t], [r, b]]


def labelmelike_extend_args(core_func):
    """ 扩展 main_func 函数，支持一些通用上下游切面功能

    支持 main_keys，主要数据字段名称
    支持 remove_keys，批量移除一些不需要的参数值
    """

    def wrapper(data, main_keys, remove_keys=None, *, clear_empty_shape=False):
        """
        :param clear_empty_shape: 在考虑要不要把清除空shape，四边形转矩形等功能写到这里，暂时还未写对应功能
        """
        # 1 转主要的数据结构
        _main_keys = main_keys if isinstance(main_keys, (list, tuple, set)) else [main_keys]
        for k in _main_keys:
            if k in data:
                data['shapes'] = core_func(data[k])
                break

        # 2 删除不需要的键值
        if remove_keys is None:
            _remove_keys = set()
        elif not isinstance(remove_keys, (list, tuple, set)):
            _remove_keys = [remove_keys]
        else:
            _remove_keys = remove_keys
        for k in (set(_main_keys) | set(_remove_keys)):
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
    def list_word(ls):
        shapes = []
        for w in ls:
            shape = {}
            if 'location' in w:
                shape.update(loc2points(w['location']))
            if 'words' in w:
                # shape['label'] = json.dumps({'text': x['words']}, ensure_ascii=False)
                shape['label'] = {'text': w['words']}  # 不转字符串
            shapes.append(shape)
        return shapes

    @staticmethod
    @labelmelike_extend_args
    def list_word2(ls):
        shapes = []
        for w in ls:
            shape = {}
            shape.update(loc2points(w['words']['words_location']))
            shape['label'] = {'text': w['words']['word'], 'category': w['words_type']}
            shapes.append(shape)
        return shapes

    @staticmethod
    @labelmelike_extend_args
    def dict_word(d):
        shapes = []
        for k, w in d.items():
            shape = {'label': {'category': k}}
            if 'location' in w:
                shape.update(loc2points(w['location']))
            if 'words' in w:
                shape['label']['text'] = w['words']
            shapes.append(shape)
        return shapes

    @staticmethod
    @labelmelike_extend_args
    def dict_str(d):
        return [{'label': {'text': v, 'category': k}} for k, v in d.items()]

    @staticmethod
    @labelmelike_extend_args
    def dict_strs(d):
        shapes = []
        for k, texts in d.items():
            shapes.append({'label': {'category': k, 'text': ','.join(texts)}})
        return shapes


def __3_定制不同输出格式():
    pass


class AipOcr(_AipOcrClient):
    """
    封装该类
        目的1：合并输入文件和url的识别
        目的2：带透明底的png百度api识别不了，要先转成RGB格式
    """

    def __init__(self, user=None, app_id=None, api_key=None, secret_key=None, *,
                 rgba2rgb=True,
                 db=False):
        """ 账号信息可以配置在环境变量 AipOcrAccount 中

        :param db: 是否启用db数据库，存储图片、识别结果

        用法一：传统方式
            accounts = {'user': 'user1', 'app_id': 1, 'api_key': '2', 'secret_key': '3'}
            XlOsEnv.persist_set('AipOcrAccount', accounts, True)
            aipocr1 = AipOcr(**XlOsEnv.get('AipOcrAccount', decoding=True))
            aipocr2 = AipOcr('user1')  # 可以不用自己解环境变量，指定user即可
        用法二：多账号方式（推荐）
            accounts = [{'user': 'user1', 'app_id': 1, 'api_key': '2', 'secret_key': '3'},
                        {'user': 'user2', 'app_id': 4, 'api_key': '5', 'secret_key': '6'}]
            XlOsEnv.persist_set('AipOcrAccount', accounts, True)
            aipocr = AipOcr('user1')

        :param rgba2rgb: 百度有些接口传带透明通道的图会有问题，所以默认会把rgba转rgb
        """
        # 1 基本的初始化
        if not (app_id and api_key and secret_key):
            # 变成账号清单
            accounts = XlOsEnv.get('AipOcrAccount', decoding=True)
            if isinstance(accounts, dict):
                accounts = [accounts]
            if user:
                accounts = [account for account in accounts if account['user'] == user]
            if app_id:
                accounts = [account for account in accounts if account['app_id'] == app_id]
            if len(accounts):
                d = accounts[0]
            else:
                raise ValueError('没有指定账号的数据')
            app_id, api_key, secret_key = str(d['app_id']), d['api_key'], d['secret_key']
        super().__init__(app_id, api_key, secret_key)
        self.rgba2rgb = rgba2rgb

        # 2 是否带位置信息的接口转换
        self.open_det = {'basicGeneral': 'general',
                         'basicAccurate': 'accurate',
                         'webImage': 'webimageLoc'}
        self.close_det = {v: k for k, v in self.open_det.items()}

        # 3 数据库存储已识别过的结果
        if db:
            self.db = XlprDb(check_same_thread=False)  # 数据库
            self.db.init_jpgimages_table()
            self.db.init_aipocr_table()
        else:
            self.db = None

    @classmethod
    def get_img_buffer(cls, in_, *, rgba2rgb=False):
        """ 获取in_代表的图片的二进制数据

        :param in_: 可以是本地文件，也可以是图片url地址，也可以是Image对象
            注意这个函数，输入是url，也会获取重置图片数据上传
            如果为了效率明确只传url，可以用aip.AipOcr原生的相关url函数
        :return: 返回图片文件二进制值的buffer
        """
        if isinstance(in_, bytes):
            return in_

        # 1 取不同来源的数据
        if is_url(in_):
            im = xlcv.read_from_url(in_)
        else:
            im = xlcv.read(in_)

        # 2 特定的格式转换
        if rgba2rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

        # 3 图片尺寸过大，要调小
        # base64编码后小于4M，分辨率不高于4096x4096，请重新上传图片
        h, w, c = im.shape
        r = max(h, w) / 4000
        if r > 1:
            im = xlcv.resize2(im, (int(h / r), int(w / r)))
        # r = min(h, w) / 15
        # if r < 1:
        #     im = xlcv.resize2(im, (int(h / r), int(w / r)))

        # 图片要限制在6M，才能确保base64后一般不超过10M
        # im = xlcv.reduce_filesize(im, 6 * 1024 * 1024)

        buffer = xlcv.to_buffer(im)
        # print(len(buffer))
        return buffer

    def _ocr(self, im, options=None, mode='general'):
        """ 文字识别：这里封装了百度的多个功能

        :param im: 可以是图片路径，也可以是网页上的url，也可以是Image对象
        :param options: 可选参数，详见：https://cloud.baidu.com/doc/OCR/s/pjwvxzmtc，常见的有：
            language_type
                默认为CHN_ENG，中英文
                ENG，英文
                auto_detect，自动检测
            detect_direction，是否检测图片朝向
                True
                False
        :param mode:
            general，普通版本
            accurate，高精度版
            webImage，网络图片

        :return: 返回识别出的dict字典

        >> AipOcr().text('0.png')
        >> AipOcr().text(r'http://ksrc2.gaosiedu.com//...', {'language_type': 'ENG'})

        返回值示例：
            {'log_id': 1425721657750823872,
             'words_result': [{'words': '项目编号'},
                              {'words': '北京市通州区科技计划'},
                              {'words': '项目实施方案'},
                              {'words': '项目名称'},
                              {'words': '所属领域'},
                              {'words': '项目承担单位'},
                              {'words': '区科委主管科室'},
                              {'words': '起止年限'},
                              {'words': '年月至年月'},
                              {'words': '北京市通州区科学技术委员会制'},
                              {'words': 'O年月'}],
             'words_result_num': 11}

        开det检测的效果：
            {'log_id': 1525024072980321431,
             'words_result': [{'location': {'height': 57,
                                            'left': 43,
                                            'top': 1,
                                            'width': 190},
                                'words': 'D066l令'},
                                ...
        """
        # 1 预处理，参数标准化
        buffer = self.get_img_buffer(im, rgba2rgb=self.rgba2rgb)
        if options is None:
            options = {}
        options = {k: options[k] for k in sorted(options.keys())}  # 对参数进行排序，方便去重

        # 2 调百度识别接口
        if self.db:
            # 如果数据库里有处理过的记录，直接引用
            im_etag = get_etag(buffer)
            res = self.db.get_aipocr_result(mode, im_etag, options)
            # 否则调用百度的接口识别
            if res is None or 'error_code' in res:
                res = getattr(super(AipOcr, self), mode)(buffer, options)
                self.db.insert_aipocr_record(mode, im_etag, options, res, if_exists='REPLACE')
                self.db.insert_jpgimage_from_buffer(buffer, etag=im_etag)
        else:
            res = getattr(super(AipOcr, self), mode)(buffer, options)

        return res

    def ocr_url(self, url, options=None, mode='general', *, det=False):
        # TODO 实际没有url的接口，可以把数据下载到本地，转成buffer，调用buffer的接口
        mode = self.open_det.get(mode, mode) if det else self.close_det.get(mode, mode)
        res = getattr(self, mode + 'Url')(url, options)
        return self.to_labelme_like(mode, res)

    def ocr_pdf(self, pdf, options=None, mode='general', *, det=False):
        mode = self.open_det.get(mode, mode) if det else self.close_det.get(mode, mode)
        res = getattr(self, mode + 'Pdf')(pdf, options)
        return self.to_labelme_like(mode, res)

    def __B1_通用(self):
        pass

    def general(self, im, options=None):
        """
        TODO 返回值示例：...
        """
        result_dict = self._ocr(im, options, 'general')
        result_dict = ToLabelmeLike.list_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def basicGeneral(self, im, options=None):
        result_dict = self._ocr(im, options, 'basicGeneral')
        result_dict = ToLabelmeLike.list_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def accurate(self, im, options=None):
        result_dict = self._ocr(im, options, 'accurate')
        result_dict = ToLabelmeLike.list_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def basicAccurate(self, im, options=None):
        result_dict = self._ocr(im, options, 'basicAccurate')
        result_dict = ToLabelmeLike.list_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def webimageLoc(self, im, options=None):
        result_dict = self._ocr(im, options, 'webimageLoc')
        result_dict = ToLabelmeLike.list_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def webImage(self, im, options=None):
        result_dict = self._ocr(im, options, 'webImage')
        result_dict = ToLabelmeLike.list_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def handwriting(self, im, options=None):
        result_dict = self._ocr(im, options, 'handwriting')
        result_dict = ToLabelmeLike.list_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def numbers(self, im, options=None):
        result_dict = self._ocr(im, options, 'numbers')
        result_dict = ToLabelmeLike.list_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def qrcode(self, im, options=None):
        result_dict = self._ocr(im, options, 'qrcode')

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
        result_dict = self._ocr(im, options, 'form')

        # 这个表格暂时转labelme格式，但泛用来说，其实不应该转labelme，它有其特殊性
        shapes = []
        for table in result_dict['forms_result']:
            # location虽然给的是四边形，但看目前数据其实就是矩形
            for k, xs in table.items():
                if k == 'vertexes_location':
                    sp = {'label': {'text': '', 'category': 'table'},
                          'points': polygon2rect([(p['x'], p['y']) for p in table['vertexes_location']]),
                          'shape_type': 'rectangle'}
                    shapes.append(sp)
                else:
                    for x in xs:
                        sp = {'label': {'text': x['words'], 'row': x['row'], 'column': x['column'],
                                        'probability': x['probability'], 'category': k},
                              'points': polygon2rect([(p['x'], p['y']) for p in x['vertexes_location']]),
                              'shape_type': 'rectangle'}
                        shapes.append(sp)

        result_dict['shapes'] = shapes
        del result_dict['forms_result']
        del result_dict['forms_result_num']
        return result_dict

    def doc_analysis_office(self, im, options=None):
        result_dict = self._ocr(im, options, 'doc_analysis_office')
        result_dict = ToLabelmeLike.list_word2(result_dict, 'results', 'results_num')
        return result_dict

    def seal(self, im, options=None):
        result_dict = self._ocr(im, options, 'seal')

        shapes = []
        for x in result_dict['result']:
            shape = {'label': {}}
            shape.update(loc2points(x['location']))
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
        result_dict = self._ocr(im, options, 'idcard')
        result_dict = ToLabelmeLike.dict_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def idcard_back(self, im, options=None):
        result_dict = self._ocr(im, options, 'idcard_back')
        result_dict = ToLabelmeLike.dict_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def bankcard(self, im, options=None):
        result_dict = self._ocr(im, options, 'bankcard')
        result_dict = ToLabelmeLike.dict_str(result_dict, ['result', 'words_result'], 'words_result_num')
        return result_dict

    def businessLicense(self, im, options=None):
        result_dict = self._ocr(im, options, 'businessLicense')
        result_dict = ToLabelmeLike.dict_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def businessCard(self, im, options=None):
        result_dict = self._ocr(im, options, 'businessCard')
        result_dict = ToLabelmeLike.dict_strs(result_dict, 'words_result')
        return result_dict

    def passport(self, im, options=None):
        result_dict = self._ocr(im, options, 'passport')
        result_dict = ToLabelmeLike.dict_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def HKMacauExitentrypermit(self, im, options=None):
        result_dict = self._ocr(im, options, 'HKMacauExitentrypermit')
        result_dict = ToLabelmeLike.dict_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def taiwanExitentrypermit(self, im, options=None):
        result_dict = self._ocr(im, options, 'taiwanExitentrypermit')
        result_dict = ToLabelmeLike.dict_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def householdRegister(self, im, options=None):
        result_dict = self._ocr(im, options, 'householdRegister')
        result_dict = ToLabelmeLike.dict_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def birthCertificate(self, im, options=None):
        result_dict = self._ocr(im, options, 'birthCertificate')
        result_dict = ToLabelmeLike.dict_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def __B3_交通(self):
        pass

    def vehicleLicense(self, im, options=None):
        result_dict = self._ocr(im, options, 'vehicleLicense')
        result_dict = ToLabelmeLike.dict_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def drivingLicense(self, im, options=None):
        result_dict = self._ocr(im, options, 'drivingLicense')
        result_dict = ToLabelmeLike.dict_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def licensePlate(self, im, options=None):
        result_dict = self._ocr(im, options, 'licensePlate')

        d = result_dict['words_result']

        # 近似key_text，但有点不太一样
        # 已测试过 车牌 只能识别一个
        label = {'text': d['number'],
                 'color': d['color']}
        shape = {'label': label,
                 'score': sum(d['probability']) / len(d['probability']),
                 'points': [(p['x'], p['y']) for p in d['vertexes_location']],
                 'shape_type': 'polygon'}
        result_dict['shapes'] = [shape]

        del result_dict['words_result']
        return result_dict

    def vinCode(self, im, options=None):
        result_dict = self._ocr(im, options, 'vinCode')
        result_dict = ToLabelmeLike.list_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def vehicleInvoice(self, im, options=None):
        result_dict = self._ocr(im, options, 'vehicleInvoice')
        result_dict = ToLabelmeLike.dict_str(result_dict, ['result', 'words_result'], 'words_result_num')
        return result_dict

    def vehicleCertificate(self, im, options=None):
        result_dict = self._ocr(im, options, 'vehicleCertificate')
        result_dict = ToLabelmeLike.dict_str(result_dict, ['result', 'words_result'], 'words_result_num')
        return result_dict

    def raw_mixed_multi_vehicle(self, im, options=None):
        result_dict = self._ocr(im, options, 'mixed_multi_vehicle')
        return result_dict

    def mixed_multi_vehicle(self, im, options=None):
        result_dict = self._ocr(im, options, 'mixed_multi_vehicle')

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
        result_dict = self._ocr(im, options, 'vehicle_registration_certificate')
        result_dict = ToLabelmeLike.dict_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def weightNote(self, im, options=None):
        result_dict = self._ocr(im, options, 'weightNote')

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
        result_dict = self._ocr(im, options, 'multipleInvoice')

        shapes = []
        for x in result_dict['words_result']:
            shape = {'category': x['type'], 'label': {'text': ''}}
            shape.update(loc2points(x))
            shapes.append(shape)

        result_dict['shapes'] = shapes
        del result_dict['words_result']
        del result_dict['words_result_num']
        return result_dict

    def vat_invoice_verification(self, im, options=None):
        result_dict = self._ocr(im, options, 'vat_invoice_verification')
        return result_dict

    def quotaInvoice(self, im, options=None):
        result_dict = self._ocr(im, options, 'quotaInvoice')
        result_dict = ToLabelmeLike.dict_str(result_dict, ['result', 'words_result'], 'words_result_num')
        return result_dict

    def invoice(self, im, options=None):
        result_dict = self._ocr(im, options, 'invoice')

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
        result_dict = self._ocr(im, options, 'trainTicket')
        result_dict = ToLabelmeLike.dict_str(result_dict, ['result', 'words_result'], 'words_result_num')
        return result_dict

    def taxiReceipt(self, im, options=None):
        result_dict = self._ocr(im, options, 'taxiReceipt')
        result_dict = ToLabelmeLike.dict_str(result_dict, ['result', 'words_result'], 'words_result_num')
        return result_dict

    def airTicket(self, im, options=None):
        result_dict = self._ocr(im, options, 'airTicket')
        result_dict = ToLabelmeLike.dict_str(result_dict, ['result', 'words_result'], 'words_result_num')
        return result_dict

    def onlineTaxiItinerary(self, im, options=None):
        result_dict = self._ocr(im, options, 'onlineTaxiItinerary')

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
        result_dict = self._ocr(im, options, 'receipt')
        result_dict = ToLabelmeLike.list_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def __B5_医疗(self):
        pass

    def raw_medicalInvoice(self, im, options=None):
        result_dict = self._ocr(im, options, 'medicalInvoice')
        return result_dict

    def raw_medicalDetail(self, im, options=None):
        result_dict = self._ocr(im, options, 'medicalDetail')
        return result_dict

    def raw_insuranceDocuments(self, im, options=None):
        result_dict = self._ocr(im, options, 'insuranceDocuments')
        return result_dict

    def __B6_教育(self):
        pass

    def docAnalysis(self, im, options=None):
        result_dict = self._ocr(im, options, 'docAnalysis')

        shapes = []
        for x in result_dict['results']:
            shape = loc2points(x['words']['words_location'])
            # 有line_probability字段，但实际并没有返回置信度~
            shape['label'] = {'category': x['words_type'], 'text': x['words']['word']}
            shapes.append(shape)
        result_dict['shapes'] = shapes

        del result_dict['results']
        del result_dict['results_num']
        return result_dict

    def formula(self, im, options=None):
        result_dict = self._ocr(im, options, 'formula')
        result_dict = ToLabelmeLike.list_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def __B7_其他(self):
        pass

    def meter(self, im, options=None):
        result_dict = self._ocr(im, options, 'meter')
        result_dict = ToLabelmeLike.list_word(result_dict, 'words_result', 'words_result_num')
        return result_dict

    def raw_lottery(self, im, options=None):
        result_dict = self._ocr(im, options, 'lottery')
        return result_dict

    def raw_facade(self, im, options=None):
        result_dict = self._ocr(im, options, 'facade')
        return result_dict

    def facade(self, im, options=None):
        result_dict = self._ocr(im, options, 'facade')

        shapes = []
        for x in result_dict['words_result']:
            shape = {'label': {'text': x['words'], 'score': x['score']}}
            shapes.append(shape)

        result_dict['shapes'] = shapes
        del result_dict['words_result']
        del result_dict['words_result_num']
        return result_dict


def demo_aipocr():
    from tqdm import tqdm
    import pprint
    import re

    from pyxlpr.data.labelme import LabelmeDict

    aipocr = AipOcr('ckz', db=True)

    mode = 'general'

    dir_ = XlPath("/home/chenkunze/data/aipocr_test")
    fmode = re.sub(r'^raw_', r'', mode)
    fmode = {'basicAccurate': 'accurate',
             'basicGeneral': 'general',
             'webimageLoc': 'webImage',
             'idcard_back': 'idcard',
             'medicalDetail': 'medicalInvoice',
             'vat_invoice_verification': 'vatInvoice',
             }.get(fmode, fmode)
    files = dir_.glob_images(f'*/{fmode}/**/*')

    for f in list(files):
        # 1 处理指定图片
        # if f.stem != '824ef2b9a5f05422199107721d299f30':
        #     continue

        # 2 检查字典
        print(f)
        d = getattr(aipocr, mode)(f.as_posix())
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


if __name__ == '__main__':
    with TicToc(__name__):
        demo_aipocr()
