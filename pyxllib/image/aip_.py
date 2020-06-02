#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Data   : 2020/05/30 21:14

"""
百度人工智能API接口
"""


import subprocess

import pandas as pd

try:
    import aip
except ModuleNotFoundError:
    subprocess.run(['pip3', 'install', 'baidu-aip'])
    import aip


from pyxllib.debug.pathlib_ import Path
from pyxllib.image.imlib import get_img_content


def create_account_df(file='aipocraccount.pkl'):
    """请在这里设置您个人的账户密码，并在运行完后，销毁明文信息"""
    df = pd.DataFrame.from_records([
        ['坤泽小号', '16936214', 'aaaaaa', '123456'],
        ['陈坤泽', '16913345', 'bbbbbb', '123456'],
        ['欧龙', '16933485', 'cccccc', '123456'],
        ['韩锦锦', '16933339', 'dddddd', '123456'],
    ], columns=['user', 'APP_ID', 'API_KEY', 'SECRET_KEY'])
    Path(file).write(df)


class AipOcr:
    """
    封装该类
        目的1：合并输入文件和url的识别
        目的2：带透明底的png百度api识别不了，要先转成RGB格式
    """
    client = None
    client_id = 0
    account_df = None

    @classmethod
    def init(cls, next_client=False, account_file_path=None):
        # 1、账号信息
        if cls.account_df is None:
            if not account_file_path:
                cls.account_df = (Path(__file__).parent / 'aipocraccount.pkl').read()

        # 2、初始化client
        if cls.client is None or next_client:
            t = cls.client_id + next_client
            if t > len(cls.account_df):
                raise ValueError('今天账号份额都用完啦！Open api daily request limit reached')
            row = cls.account_df.loc[t]
            AipOcr.client = aip.AipOcr(row.APP_ID, row.API_KEY, row.SECRET_KEY)
            AipOcr.client_id = t
        return AipOcr.client

    @classmethod
    def text(cls, in_, options=None):
        """ 调用baidu的普通文本识别
        这个函数你们随便调用，每天5万次用不完

        :param in_: 可以是图片路径，也可以是网页上的url，也可以是Image对象
        :param options: 可选参数
            详见：https://cloud.baidu.com/doc/OCR/s/pjwvxzmtc
        :return: 返回识别出的dict字典

        >> baidu_accurate_ocr('0.png')
        >> baidu_accurate_ocr(r'http://ksrc2.gaosiedu.com//...',
                                 {'language_type': 'ENG'})
        """
        client = cls.init()
        content = get_img_content(in_)
        return client.basicGeneral(content, options)

    @classmethod
    def accurate_text(cls, in_, options=None):
        """ 调用baidu的高精度文本识别

        :param in_: 可以是图片路径，也可以是url
        :param options: 可选参数
            详见：https://cloud.baidu.com/doc/OCR/s/pjwvxzmtc
        :return: 返回识别出的dict字典

        >> baidu_accurate_ocr('0.png')
        >> baidu_accurate_ocr(r'http://ksrc2.gaosiedu.com//...',
                                 {'language_type': 'ENG'})
        """
        client = cls.init()
        content = get_img_content(in_)
        # 会自动转base64
        while True:
            t = client.basicAccurate(content, options)
            # dprint(t)
            if t.get('error_code', None) == 17:
                client = AipOcr.init(next_client=True)
            elif t.get('error_code', None) == 18:
                # {'error_code': 18, 'error_msg': 'Open api qps request limit reached'}，继续尝试
                continue
            else:
                 break
        return t


def demo_aipocr():
    client = AipOcr()
    d = client.text("http://i1.fuimg.com/582188/7b0f9cb22c1770a0.png", {'language_type': 'ENG'})
    print(d)
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
    # create_account_df()
    demo_aipocr()
