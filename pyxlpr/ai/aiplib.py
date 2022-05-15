#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 21:14

"""
百度人工智能API接口
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
from pyxllib.algo.geo import xywh2ltrb
from pyxllib.cv.expert import xlcv
from pyxllib.data.xlprdb import XlprDb
from pyxllib.file.specialist import get_etag, XlPath


class AipOcr(aip.AipOcr):
    """
    封装该类
        目的1：合并输入文件和url的识别
        目的2：带透明底的png百度api识别不了，要先转成RGB格式
    """

    __doc_analysis_office = "https://aip.baidubce.com/rest/2.0/ocr/v1/doc_analysis_office"

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
            self.db = XlprDb()  # 数据库
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

    @classmethod
    def to_labelme_like(cls, mode, data):
        """ 将百度api获得的各种业务结果，转成类似labelme的标注格式

        统一成一种风格，易分析的结构
        """

        # 1 一级辅助工具
        def loc2points(loc):
            """ 百度的location格式转为 {'points': [[a, b], [c, d]], 'shape_type: 'rectangle'} """
            # 目前关注到的都是矩形，不知道有没可能有其他格式
            l, t, r, b = xywh2ltrb([loc['left'], loc['top'], loc['width'], loc['height']])
            return {'points': [[l, t], [r, b]],
                    'shape_type': 'rectangle'}

        def extend_args(func):
            """ 扩展 func 函数，支持一些通用上下游切面功能

            支持批量的 main_keys，主要识别条目数据可能存储在多种不同的名称
            支持 remove_keys，批量移除一些不需要的参数值
            """

            def wrapper(main_keys, remove_keys=None, *, clear_empty_shape=False):
                # 1 转主要的数据结构
                main_keys = main_keys if isinstance(main_keys, (list, tuple, set)) else [main_keys]
                for k in main_keys:
                    if k in data:
                        data['shapes'] = func(data[k])
                        break

                # 2 删除不需要的键值
                if remove_keys is None:
                    remove_keys = set()
                for k in (set(main_keys) | set(remove_keys)):
                    if k in data:
                        del data[k]

            return wrapper

        # 2 二级辅助工具
        # 这里每个函数，只要实现输入核心识别结果清单，返回shapes的代码即可。上下游一些切面操作是统一的。
        @extend_args
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

        @extend_args
        def list_texts(ls):
            shapes = []
            for x in ls:
                # assert len(x['text']) == 1, '取到的文本list长度大于1'
                shapes.append({'label': {'text': ','.join(x['text']), 'class': x['type']}})
            return shapes

        @extend_args
        def dict_word(d):
            shapes = []
            for k, w in d.items():
                shape = {'label': {'class': k}}
                if 'location' in w:
                    shape.update(loc2points(w['location']))
                if 'words' in w:
                    shape['label']['text'] = w['words']
                shapes.append(shape)
            return shapes

        @extend_args
        def dict_str(d):
            return [{'label': {'text': v, 'class': k}} for k, v in d.items()]

        @extend_args
        def dict_strs(d):
            shapes = []
            for k, texts in d.items():
                shapes.append({'label': {'class': k, 'text': ','.join(texts)}})
            return shapes

        def key_words2shapes(keys):
            return [{'label': {'text': v['words'], 'class': k}} for k, v in keys.items()]

        # 2 主体功能
        if mode in {'basicGeneral', 'general', 'basicAccurate', 'accurate', 'webImage', 'webimageLoc',
                    'handwriting', 'numbers', 'vinCode', 'receipt', 'formula', 'meter'}:
            list_word(['words_result'], ['words_result_num'])
        elif mode in {'qrcode'}:
            list_texts('codes_result', 'codes_result_num')
        elif mode in {'idcard', 'businessLicense', 'passport',
                      'HKMacauExitentrypermit', 'taiwanExitentrypermit', 'householdRegister', 'birthCertificate',
                      'vehicleLicense', 'drivingLicense'}:
            # TODO key要排序？
            dict_word('words_result', ['words_result_num'])
        elif mode in {'bankcard', 'quotaInvoice', 'vehicleInvoice', 'vehicleCertificate',
                      'trainTicket', 'taxiReceipt', 'airTicket'}:
            dict_str(['result', 'words_result'], ['words_result_num'])
        elif mode in {'businessCard'}:
            dict_strs('words_result')
        elif mode == 'licensePlate':
            # 近似key_text，但有点不太一样
            # 已测试过 车牌 只能识别一个
            x = data['words_result']
            label = {'text': x['number'],
                     'color': x['color']}
            shape = {'label': label,
                     'score': sum(x['probability']) / len(x['probability']),
                     'points': [(p['x'], p['y']) for p in x['vertexes_location']],
                     'shape_type': 'polygon'}
            data['shapes'] = [shape]
            del data['words_result']
        elif mode == 'docAnalysis':
            shapes = []
            for x in data['results']:
                shape = loc2points(x['words']['words_location'])
                # 有line_probability字段，但实际并没有返回置信度~
                shape['label'] = {'class': x['words_type'], 'text': x['words']['word']}
                shapes.append(shape)
            data['shapes'] = shapes
            del data['results']
            del data['results_num']

        return data

    def ocr(self, im, options=None, mode='general'):
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

            数字、表格、二维码、办公文档 等就不封装了，因为这些都没有位置版的功能，没必要封装进这个接口，直接调用对应函数最简单。
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

        修正后的格式：类label格式
        {'log_id': 1525042208153510459,
         'shapes': [{'label': '{"text": "D066l令"}', 'points': [[43, 1], [233, 58]]},
                    {'label': '{"text": "HD 2"}', 'points': [[49, 36], [92, 54]]},
                    {'label': '{"text": "⊙N米988110：32"}', 'points': [[571, 14], [834, 45]]},
                    ...]
         'words_result_num': 48}
        """
        # 1 预处理
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
                res = getattr(self, mode)(buffer, options)
                self.db.insert_aipocr_record(mode, im_etag, options, res, if_exists='REPLACE')
                self.db.insert_jpgimage_from_buffer(buffer, etag=im_etag)
        else:
            res = getattr(self, mode)(buffer, options)

        # TODO 给百度的图，可能会进行一定的缩放，返回的结果如果有points需要批量放大

        pprint.pprint(res)

        return self.to_labelme_like(mode, res)

    def ocr_url(self, url, options=None, mode='general', *, det=False):
        # TODO 实际没有url的接口，可以把数据下载到本地，转成buffer，调用buffer的接口
        mode = self.open_det.get(mode, mode) if det else self.close_det.get(mode, mode)
        res = getattr(self, mode + 'Url')(url, options)
        return self.to_labelme_like(res)

    def ocr_pdf(self, pdf, options=None, mode='general', *, det=False):
        mode = self.open_det.get(mode, mode) if det else self.close_det.get(mode, mode)
        res = getattr(self, mode + 'Pdf')(pdf, options)
        return self.to_labelme_like(res)

    # def numbers(self, im, options=None):
    #     im = self.get_img_buffer(im)
    #     res = super(AipOcr, self).numbers(im, options)
    #     return self.to_labelme_like(res)

    def idcard(self, image, options=None):
        """ 身份证识别，统一接口范式
        """
        return super(AipOcr, self).idcard(image,
                                          options.get('id_card_side', 'front'),  # 默认识别带照片一篇
                                          options)

    def onlineTaxiItinerary(self, image, options=None):
        """ 网约车行程单识别
        """
        return super(AipOcr, self).onlineTaxiItinerary(image)

    # def doc_analysis(self, im, options=None):
    #     """ 办公文档识别
    #
    #     不过好像用不了
    #         {'error_code': 6, 'error_msg': 'No permission to access data'}
    #     """
    #     raise ValueError('改接口暂时用不了，但执行仍会占用免费额度')
    #
    #     options = options or {}
    #
    #     data = {}
    #     data['image'] = base64.b64encode(im).decode()
    #     data.update(options)
    #
    #     return self._request(self.__doc_analysis_office, data)


def demo_aipocr():
    from tqdm import tqdm
    import pprint

    mode = 'facade'

    aipocr = AipOcr('ckz', db=True)
    dir_ = XlPath("/home/chenkunze/data/aipocr_test")

    files = dir_.glob_images(f'*/{mode}/**/*')
    for f in list(files):
        # if f.stem != '73ccf4826f1371980a3daa79dca887e9':
        #     continue

        print(f)
        d = aipocr.ocr(f.as_posix(), mode=mode)
        print()
        pprint.pprint(d)
        print('- ' * 20)

        if 'error_code' in d:
            f.delete()
        elif d.get('shapes', 1):
            break


if __name__ == '__main__':
    with TicToc(__name__):
        demo_aipocr()
        # print(File('aipocraccount.pkl').read())
