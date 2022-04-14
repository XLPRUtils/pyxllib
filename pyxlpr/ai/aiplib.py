#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/05/30 21:14

"""
百度人工智能API接口
"""

from pyxllib.prog.pupil import check_install_package

check_install_package('aip', 'baidu-aip')

import aip
import base64
import cv2

from pyxllib.prog.pupil import is_url
from pyxllib.prog.specialist import XlOsEnv
from pyxllib.debug.specialist import TicToc
from pyxllib.cv.expert import xlcv


class AipOcr(aip.AipOcr):
    """
    封装该类
        目的1：合并输入文件和url的识别
        目的2：带透明底的png百度api识别不了，要先转成RGB格式
    """

    __doc_analysis_office = "https://aip.baidubce.com/rest/2.0/ocr/v1/doc_analysis_office"

    def __init__(self, user=None, app_id=None, api_key=None, secret_key=None, *,
                 rgba2rgb=True):
        """ 账号信息可以配置在环境变量 AipOcrAccount 中

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

    @classmethod
    def get_img_buffer(cls, in_, *, rgba2rgb=False):
        """ 获取in_代表的图片的二进制数据

        :param in_: 可以是本地文件，也可以是图片url地址，也可以是Image对象
            注意这个函数，输入是url，也会获取重置图片数据上传
            如果为了效率明确只传url，可以用aip.AipOcr原生的相关url函数
        :return: 返回图片文件二进制值的buffer
        """

        # 1 取不同来源的数据
        if is_url(in_):
            im = xlcv.read_from_url(in_)
        else:
            im = xlcv.read(in_)

        # 2 特定的格式转换
        if rgba2rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        buffer = xlcv.to_buffer(im)
        return buffer

    def text(self, im, options=None, *, position=False, imtype='normal'):
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
        :param position: 是否返回坐标信息
        :param imtype:
            normal，普通版本
            accurate，高精度版
            webimage，网络图片

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
        """
        buffer = self.get_img_buffer(im, rgba2rgb=self.rgba2rgb)
        if imtype == 'normal':
            if position:
                return self.general(buffer, options)
            else:
                return self.basicGeneral(buffer, options)
        elif imtype == 'accurate':
            if position:
                return self.accurate(buffer, options)
            else:
                return self.basicAccurate(buffer, options)
        elif imtype == 'webimage':
            if position:  # 这个实际并没有提供位置版的接口，就用普通版代替了
                return self.general(buffer, options)
            else:
                return self.webImage(buffer, options)
        else:
            raise ValueError(f'{imtype}')

    def numbers(self, im, options=None):
        im = self.get_img_buffer(im)
        return super().numbers(im, options)

    def doc_analysis(self, im, options=None):
        """ 办公文档识别

        不过好像用不了
            {'error_code': 6, 'error_msg': 'No permission to access data'}
        """
        raise ValueError('改接口暂时用不了，但执行仍会占用免费额度')

        options = options or {}

        data = {}
        data['image'] = base64.b64encode(im).decode()
        data.update(options)

        return self._request(self.__doc_analysis_office, data)


def demo_aipocr():
    import pprint

    aipocr = AipOcr('ckz')
    d = aipocr.text(r"D:\home\datasets\textGroup\temp210730\北京市通州区科技计划项目实施方案\北京市通州区科技计划项目实施方案_01.png",
                    {'language_type': 'CHN_ENG'})
    pprint.pprint(d)
    # d = {'log_id': 8013455108426397566, 'words_result_num': 64,
    # 'words_result': [{'words': '1 . 4 . cre ated'}, {'words': 'B . shook'},
    # {'words': 'C . entered'}, {'words': 'D'}, {'words': 'C ,'}, {'words': 'D , until'},
    # {'words': '3 . A . break up'}, {'words': 'B . hold up'}, {'words': 'C . keep up'},
    # {'words': 'D . show up'}, {'words': '4 . A . whispered'}, {'words': 'B , fought'},
    # {'words': 'C . talked'}, {'words': 'D'}, {'words': '5 . A . throughout'}, {'words': 'D . after'},
    # {'words': '6 . A . where'}, {'words': 'B . although'}, {'words': 'C , whle'}, {'words': 'D . that'},
    # {'words': '7 . A . visitor'}, {'words': 'B . relative'}, {'words': 'C . nei gabor'},
    # {'words': 'D . stranger'}, {'words': 'B , interest'}, {'words': 'D . anger'}, {'words': 'B . differenc'},
    # {'words': 'C . point'}, {'words': '10 . A . forgot'}, {'words': 'B . supported'},
    # {'words': 'C , resi sted'}, {'words': 'D , valued'}, {'words': '11 . A . serious'},
    # {'words': 'B , nice'}, {'words': 'C , bad'}, {'words': 'D . generous'}, {'words': '12 . A . Gradually'},
    # {'words': 'B . Imm ediately C . Usuall'}, {'words': 'D , Real ar'}, {'words': '13 . 4 . mind'},
    # {'words': 'B , trouble'}, {'words': 'D , , order'}, {'words': 'C , lost'}, {'words': 'D , saved'},
    # {'words': '15 . A . experi'}, {'words': 'B . inform ation C . impression D . advice'},
    # {'words': '16 . A . However'}, {'words': 'B . Besides'}, {'words': 'C . Eventually D . Occasionally'},
    # {'words': '17 . A . wrong'}, {'words': 'B . confident'}, {'words': 'C . gulty'},
    # {'words': '18 . A . rem arned'}, {'words': 'B . retumed'}, {'words': 'C . changed'},
    # {'words': 'D , left'}, {'words': '19 . A . method'}, {'words': 'B , truth'}, {'words': 'C , skill'},
    # {'words': 'D , word'}, {'words': '20 . A . exist'}, {'words': 'B , remain'}, {'words': 'C , continue'},
    # {'words': 'D . happen'}]}


if __name__ == '__main__':
    with TicToc(__name__):
        demo_aipocr()
        # print(File('aipocraccount.pkl').read())
