#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/05/11 15:09

"""
厦门理工模式识别与图像理解重点实验室 API服务接口
"""

import base64
import requests
import json

import cv2
from pyxllib.cv.xlcvlib import xlcv


class XlServer:
    def __init__(self, host='172.16.170.134', port=5003, token='token'):
        """ 这里默认设了局域网的host，但是token依然是错的，需要拿到正确的token才能运行 """
        self.host = f'http://{host}:{port}'
        self.headers = {'Content-type': 'application/json', 'Token': token}

    def _read_image(self, path):
        # TODO 可以在__init__加一些图片默认的处理操作
        flag, buffer = cv2.imencode('.jpg', xlcv.read(path))
        buffer = base64.b64encode(bytes(buffer)).decode()
        return buffer

    def common_ocr(self, im):
        data = {'image': self._read_image(im)}
        r = requests.post(f'{self.host}/api/common_ocr', json.dumps(data), headers=self.headers)
        return json.loads(r.text)

    def hesuan_layout(self, im):
        data = {'image': self._read_image(im)}
        r = requests.post(f'{self.host}/api/hesuan_layout', json.dumps(data), headers=self.headers)
        return json.loads(r.text)

    def aipocr(self, im, mode='general'):
        data = {'image': self._read_image(im), 'mode': mode}
        r = requests.post(f'{self.host}/api/aipocr', json.dumps(data), headers=self.headers)
        return json.loads(r.text)

    def rec_singleline(self, im):
        """ 通用的识别一张图的所有文本，并拼接到一起 """
        data = {'image': self._read_image(im), 'mode': 'basicGeneral'}
        try:
            r = requests.post(f'{self.host}/api/aipocr', json.dumps(data), headers=self.headers)
            d = json.loads(r.text)
            if 'shapes' in d:
                text = ' '.join([sp['label']['text'] for sp in d['shapes']])
            else:  # 因图片太小等各种原因，没有识别到结果
                text = ''
        except requests.exceptions.ConnectionError:
            text = ''
        return text
