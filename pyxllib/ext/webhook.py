#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2020/10/21 09:22


import time
import hmac
import hashlib
import base64
import urllib.parse

import requests


class WeixinRobot:
    """ 企业微信 机器人 """

    def __init__(self, url):
        self.url = url

    def push_text(self, s):
        msgtype = 'text'
        try:
            headers = {"Content-Type": "text/plain"}
            t = {"content": s} if isinstance(s, str) else s
            data = {"msgtype": msgtype, msgtype: t}  # msgtype: text、markdown
            requests.post(url=self.url, headers=headers, json=data)
        except requests.exceptions.ConnectionError:  # 没网发送失败的时候也不报错
            pass


class DingtalkRobot:
    """ 钉钉 自定义webhook机器人

    https://ding-doc.dingtalk.com/doc#/serverapi2/qf2nxq
    """

    def __init__(self, url, secret):
        self.url = url
        self.url += self.add_secret(secret)
        self.headers = {"Content-Type": "application/json"}

    @classmethod
    def add_secret(cls, secret):
        """ 钉钉机器人需要加签，确保安全性 """
        timestamp = str(round(time.time() * 1000))
        secret_enc = secret.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, secret)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))

        return f'&timestamp={timestamp}&sign={sign}'

    def push_data(self, data):
        try:
            requests.post(url=self.url, headers=self.headers, json=data)
        except requests.exceptions.ConnectionError:  # 没网发送失败的时候也不报错
            pass

    def push_text(self, content):
        msgtype = 'text'
        d = {}
        if content: d['content'] = content
        data = {"msgtype": msgtype, msgtype: d}
        self.push_data(data)

    def push_link(self, text='', title='', pic_url='', message_url=''):
        msgtype = 'link'
        d = {}
        if text: d['text'] = text
        if title: d['title'] = title
        if pic_url: d['picUrl'] = pic_url
        if message_url: d['messageUrl'] = message_url
        data = {"msgtype": msgtype, msgtype: d}
        self.push_data(data)

    def push_markdown(self, text='', title=''):
        msgtype = 'link'
        d = {}
        if text: d['text'] = text
        if title: d['title'] = title
        data = {"msgtype": msgtype, msgtype: d}
        self.push_data(data)

    def push_actioncard(self, text='', title='', siggle_url='', siggle_title='', btn_orientation='0'):
        raise NotImplementedError

    def push_feedcard(self):
        raise NotImplementedError
