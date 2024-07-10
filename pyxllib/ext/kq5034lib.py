# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/4

""" 惟海法师网课考勤工具

为了使用便捷明了，程序开发上
1、使用中文命名
2、牺牲了一定的工程、扩展灵活性，用了尽可能多的自动推导，而不是参数配置
"""
from pyxllib.prog.pupil import check_install_package

check_install_package('fire')  # 自动安装依赖包

from collections import Counter, defaultdict
from datetime import date, timedelta
import datetime
import math
import os
import re
import time
from io import StringIO
import csv

import fire
import pandas as pd
from tqdm import tqdm
import requests
# import requests_cache

from pyxllib.text.pupil import chinese2digits, grp_chinese_char
from pyxllib.file.xlsxlib import openpyxl
from pyxllib.file.specialist import XlPath, get_encoding
from pyxllib.prog.pupil import run_once
from pyxllib.prog.specialist import parse_datetime, browser, TicToc, dprint
from pyxllib.cv.rgbfmt import RgbFormatter
from pyxllib.data.sqlite import Connection
from pyxllib.ext.seleniumlib import XlChrome


class Xiaoetong:
    """ 写一个觉观的api
    """

    def __init__(self):
        self.token = ''

    def login(self, app_id, client_id, secret_key):
        """ 登录，获取token
        """
        # 启用缓存
        # requests_cache.install_cache('access_token_cache', expire_after=None)  # 设置缓存过期时间xx（单位：秒）
        # 接口地址
        url = "https://api.xiaoe-tech.com/token"
        params = {
            "app_id": app_id,
            "client_id": client_id,
            "secret_key": secret_key,
            "grant_type": "client_credential"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            result = response.json()
            if result['code'] == 0:
                # access_token 是小鹅通开放api的全局唯一接口调用凭据，店铺调用各接口时都需使用 access_token ,开发者需要进行妥善保管；
                self.token = result['data']['access_token']
            else:
                raise Exception("Error getting access token: {}".format(result['msg']))
        else:
            raise Exception("HTTP request failed with status code {}".format(response.status_code))
        # return result['data']['access_token']
    
    def get_alive_user_list(self, resource_id, page_size=100):
        """ 获取直播间用户
        """
        # 1 获取总页数
        url = "https://api.xiaoe-tech.com/xe.alive.user.list/1.0.0"  # 接口地址【路径：API列表 -> 直播管理 -> 获取直播间用户列表】
        data_1 = {
            "access_token": self.token,
            "resource_id": resource_id,
            "page": 1,
            "page_size": page_size
        }
        response_1 = requests.post(url, data=data_1)
        result_1 = response_1.json()
        page = math.ceil(result_1['data']['total'] / page_size)  # 页数

        # 2 获取直播间用户数据
        lst = result_1['data']['list']
        for i in range(1, page):  # 为什么从1开始，因为第一页的数据上面已经获取到了，这里没必要从新获取一次
            data = {
                "access_token": self.token,
                "resource_id": resource_id,
                "page": i + 1,
                "page_size": page_size
            }
            response = requests.post(url, data=data)
            result = response.json()
            data_1 = result['data']['list']
            lst += data_1
            # lst.extend(data_1)
        return lst

    def get_elock_actor(self, activity_id, page_size=100):
        """ 获取打卡参与用户
        """
        # 获取总页数
        url = "https://api.xiaoe-tech.com/xe.elock.actor/1.0.0"  # 接口地址【路径：API列表 -> 打卡管理 -> 获取打卡参与用户】
        data_1 = {
            "access_token": self.token,
            "activity_id": activity_id,
            "page_index": 1,
            "page_size": page_size
        }
        response_1 = requests.post(url, data=data_1)
        result_1 = response_1.json()
        page = math.ceil(result_1['data']['count'] / page_size)  # 页数
        # 获取打卡用户数据
        lst = result_1['data']['list']
        for i in range(1, page):  # 为什么从1开始，因为第一页的数据上面已经获取到了，这里没必要从新获取一次
            data = {
                "access_token": self.token,
                "activity_id": activity_id,
                "page_index": i + 1,
                "page_size": page_size
            }
            response = requests.post(url, data=data)
            result = response.json()
            data_1 = result['data']['list']
            lst += data_1
            # lst.extend(data_1)
        return lst


class 网课考勤:
    def __init__(self, today=None, *, 表格路径=None):
        self.返款标题 = ''
        self.表格路径 = 表格路径 or r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUlF1UnRackJ2Vm5U'  # 生成日报用
        self.开课日期 = '2022-01-08'
        self.视频返款 = [20, 15, 10, 5, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4天/第5天，完成观看的依次返款额。
        self.打卡返款 = {5: 100, 10: 150, 15: 200}  # 打卡满5/10/15次的返款额
        self.课程链接 = []
        self._init(today)

        self.driver = None  # 浏览器脚本

    def __1_基础数据处理(self):
        pass

    def _init(self, today=None):
        """ 可以手动指定today，适合某些场景下的调试 """
        self.课次数 = len(self.课程链接) - 1  # 课次一般都是固定21课，不用改
        self.达标时长 = 30  # 单位:分钟。在线听课的时长达到此标准以上，才计算为完成一节课的学习
        # 一般是3天，因为回放第1、2、3天都会数额，第4天就没数额了，所以第4天返款
        self.回放返款天数 = sum((x > 0) for x in self.视频返款[1:])

        self.root = XlPath(self.表格路径).parent
        self.wb = openpyxl.load_workbook(self.表格路径)
        self.ws = self.wb['考勤表']
        if today:
            dt = parse_datetime(today)
            self.today = date(dt.year, dt.month, dt.day)
        else:
            self.today = date.today()
        self.开课日期 = date.fromisoformat(self.开课日期)
        # self.觉观禅课 = (self.开课日期.day == 1)
        self.觉观禅课 = '觉观' in self.返款标题
        self.当天课次 = (self.today - self.开课日期).days + 1  # 这是用于逻辑运算的，可能超过实际课次数
        self.当天课次2 = min(self.当天课次, self.课次数)
        self.结束课次 = self.当天课次 - len(self.视频返款) + 1  # 这是用于逻辑运算的，可能有负值
        self.结束课次2 = max(0, self.结束课次)
        try:
            self.用户列表 = pd.read_csv(self.get_file('数据表/用户列表导出*.csv'), dtype={16: str})
        except:
            self.用户列表 = None
        # 用户列表 = 用户列表[用户列表['账号状态'] == '正常']

        self.考勤表出现次数 = Counter()
        for f in self.root.glob('数据表/**/*直播观看详情*.csv'):
            df = pd.read_csv(f, skiprows=1)
            self.考勤表出现次数 += Counter([x.strip() for x in df['用户ID']])
        for f in self.root.glob('数据表/**/*直播用户列表*.csv'):
            df = pd.read_csv(f)
            self.考勤表出现次数 += Counter([x.strip() for x in df['用户ID']])

        self.异常data = {}

        if hasattr(self, '异常处理'):
            self.异常处理()

    def get_file(self, p):
        # 返回最后一个匹配的文件
        return list(self.root.glob(p))[-1]

    def 查找用户(self, 昵称='', 手机号='', *, debug=False):
        """
        :param 昵称: 昵称可以输入一个列表，会在目标昵称、真实姓名同时做检索
        :param 手机号: 手机也可以输入一个列表，因为有些人可能报名时手机号填错，可以增加一些匹配规则
        :return:
        """

        # 1 统一输入参数格式
        def tolist(x):
            if x and not isinstance(x, list):
                x = [x]
            x = [str(a) for a in x if a]
            return x

        def check_telphone(v, refs):
            if not isinstance(v, str):
                try:
                    v = int(v)
                except:
                    pass
                v = str(v)
            for a in refs:
                if a in v:
                    return True

        昵称 = tolist(昵称)
        手机号 = tolist(手机号)
        手机号 = [f'{x}' for x in 手机号 if (x and x != 'None')]

        # 2 查找所有可能匹配的项目
        ls = []
        for idx, x in self.用户列表.iterrows():
            logo = 0
            if '真实姓名' in x:
                x['姓名'] = x['真实姓名']

            if 昵称:
                if x['昵称'] in 昵称 or x['姓名'] in 昵称:
                    logo += 1
            if 手机号:
                if check_telphone(x.get('账户绑定手机号'), 手机号) or check_telphone(x.get('最近采集的手机号'), 手机号):
                    logo += 2
            if logo:
                ls.append([x, logo])

        ls.sort(key=lambda x: x[1], reverse=True)
        ls = [x[0] for x in ls]

        if debug:
            print('\n'.join(map(self.用户信息摘要, ls)))
            return

        return ls

    def 用户信息摘要(self, x):
        """ 输入series类型的一个条目，输出其摘要信息 """
        ls = [f'{k}={v}' for k, v in x.items() if (not isinstance(v, float) or not math.isnan(v))]
        if x['用户ID'] in self.考勤表出现次数:
            ls.append('考勤' + str(self.考勤表出现次数[x['用户ID']]) + '次')
        return ', '.join(ls)

    def 匹配用户ID(self, sheet_name='报名表'):

        def try2int(x):
            try:
                return int(x)
            except:
                if isinstance(x, float) and math.isnan(x):
                    return ''
                return x

        def todict(row):
            """ 将表格第i行的数据提取为字典格式
            """
            msg = {}
            for k, x in enumerate(['真实姓名', '微信昵称', '手机号', '错误手机号']):
                msg[x] = ws.cell2(row, x).value
            return msg

        ws = self.wb[sheet_name]
        for row in tqdm(list(ws.iterrows('真实姓名')), desc='匹配进度'):
            # todo 这里要考虑相似度整体权重的，而不是几个属性的优先级
            x = todict(row)
            x['手机号'] = str(x['手机号']).lstrip('`')
            x['错误手机号'] = str(x['错误手机号']).lstrip('`')

            待查手机号 = [x['手机号']]
            t = try2int(x['错误手机号'])
            if t:
                待查手机号.append(t)
            ls = self.查找用户([x['真实姓名'], x['微信昵称']], 待查手机号)
            摘要ls = list(map(self.用户信息摘要, ls))

            用户ID = ''
            if len(ls) == 1:
                # 只有一个关联的直接匹配上，否则填空
                用户ID = ls[0]['用户ID']
            else:
                # 只有一条有考勤记录时
                flags = [('考勤' in text) for text in 摘要ls]
                if sum(flags) == 1:
                    idx = flags.index(1)
                    用户ID = ls[idx]['用户ID']
            if 用户ID == '':
                flags = [('账号状态=正常' in text) for text in 摘要ls]
                if sum(flags) == 1:
                    idx = flags.index(1)
                    用户ID = ls[idx]['用户ID']
            # 如果用户ID还是空的，则用手机号能匹配上的那条
            if 用户ID == '':
                for i, text in enumerate(摘要ls):
                    if str(x['手机号']) in text:
                        用户ID = ls[i]['用户ID']
                        break
            ws.cell2(row, '用户ID', 用户ID)
            ws.cell2(row, '参考信息', '\n'.join(摘要ls))
            row += 1
        self.wb.save(self.表格路径)

    def 获取每日统计表(self):
        ls = []
        columns = ['课次', '用户ID', '观看日期', '在线时长(分钟)']
        for f in self.root.glob('数据表/*.csv'):
            if m := re.match(r'(\d{4}\-\d{2}\-\d{2}).+?课.*?(\d+).+?直播观看详情', f.stem):
                stat_day, 课次 = date.fromisoformat(m.group(1)), int(m.group(2))
                skiprows = 1
            elif m := re.search(r'届(?:念住|觉观).+?(\d+).+?直播用户列表.+?(\d{4}\-\d{2}\-\d{2})', f.stem):
                stat_day, 课次 = date.fromisoformat(m.group(2)), int(m.group(1))
                skiprows = 0
            elif m := re.search(r'第(\d+)堂.+?届(?:念住|觉观).+?直播用户列表.+?(\d{4}\-\d{2}\-\d{2})', f.stem):
                stat_day, 课次 = date.fromisoformat(m.group(2)), int(m.group(1))
                skiprows = 0
            elif m := re.search(r'本体音艺网课-(\d+).+?直播用户列表.+?(\d{4}\-\d{2}\-\d{2})', f.stem):
                stat_day, 课次 = date.fromisoformat(m.group(2)), int(m.group(1))
                skiprows = 0
            else:
                continue

            if stat_day > self.today:  # 超过self.today的数据不记录
                continue
            观看日期 = (stat_day - self.开课日期).days - 课次 + 1
            df = pd.read_csv(f, skiprows=skiprows)
            for idx, r in df.iterrows():
                k = '累计观看时长(秒)' if 观看日期 else '直播观看时长(秒)'
                ls.append([课次, r['用户ID'].strip(), 观看日期, int(int(r[k]) / 60)])

        # ext: 异常数据修正，trick
        for k, v in self.异常data.items():
            课次, 用户ID = k
            # 观看日期和在线时长可以乱填，主要是把这个用户课次记录下，后面遍历才会处理到
            ls.append([int(re.search(r'\d+', 课次).group()), 用户ID, 0, 0])

        df = pd.DataFrame.from_records(ls, columns=columns)
        return df

    def 获取考勤表中出现的所有用户名(self):
        df = self.获取每日统计表()
        print(*set(df['用户ID']), sep='\n')

    def 修订(self, 学号, 课次, 完成标记, *args):
        try:
            user_id = self.ws.cell2({'学号': 学号}, '用户ID').value
        except ValueError:
            return
        if isinstance(课次, int):
            课次 = f'第{课次:02}课'
        self.异常data[课次, user_id] = 完成标记

    def 更新统计表(self, df):
        # 1 辅助函数
        ws = self.ws
        id2row = {}
        cel = ws.findcel('用户ID').down()
        while cel.value:
            id2row[cel.value] = cel.row
            cel = cel.down()

        col = ws.findcol('第01课') - 1

        def 完成标记(items):
            """ 输入某个用户某个课次每天的累计观看时长 """
            x = {t['观看日期']: t['在线时长(分钟)'] for idx, t in items.iterrows()}
            for k in sorted(x.keys()):
                if x[k] >= self.达标时长:
                    return f'第{k}天回放' if k else '完成当堂学习'
            return f'不足{self.达标时长}分钟'

        def write(课次, 用户ID, value):
            # 优先使用修订的标记
            value = self.异常data.get((f'第{课次:02}课', 用户ID), value)
            cel = ws.cell2(id2row[用户ID], f'第{课次:02}课')
            if cel is None:
                return
            cel.value = value
            color = None
            if '完成当堂学习' in value:
                color = RgbFormatter.from_name('鲜绿色')
            elif '回放' in value:
                color = RgbFormatter.from_name('黄色')
                v1 = self.视频返款[0]
                idx = min(int(re.search(r'第(\d+)天', value).group(1)),
                          len(self.视频返款) - 1)
                v2 = self.视频返款[idx]

                if v2:
                    color = color.light((v1 - v2) / v2)  # 根据返款额度自动变浅
                else:  # 如果无返款额度
                    color = RgbFormatter.from_name('灰色')
            elif 课次 <= self.当天课次 - len(self.视频返款) + 1:
                cel.value = '未完成学习'
                color = RgbFormatter.from_name('红色')

            if color:
                cel.fill_color(color)

        # 2 遍历更新考勤情况数据
        for [课次, 用户ID], items in df.groupby(['课次', '用户ID']):
            if 用户ID not in id2row:
                continue
            write(课次, 用户ID, 完成标记(items))

        # 3 处理剩余用户
        for i in range(1, min(self.课次数 + 1, self.结束课次 + 1)):
            for k, v in id2row.items():
                t = ws.cell(v, col + i).value
                if not t or t in ('未开始学习', f'不足{self.达标时长}分钟'):
                    write(i, k, '未完成学习')
        for i in range(max(self.结束课次 + 1, 1), min(self.课次数 + 1, self.当天课次 + 1)):
            for k, v in id2row.items():
                if not ws.cell(v, col + i).value:
                    write(i, k, '未开始学习')

    def 生成今日返款文件(self, msg):
        """ 生成今日返款文件，及缺勤报告

        生成的csv文件共4列：订单号，反馈额，备注说明，重复校验码
        第4列可以设置一个64长度以内的标识来判重，防止重复退款
            因为每个学员每个课次理论上应该只有一次退款操作
            所以我使用"{订单号}_class{课次号}"来生成校验码

        注意，涉及给人看的数据，大部分课次编号都不会加前导0，
        涉及程序、文件名对齐的，则一般都会设前导0
        """
        ls = []  # 返款汇总
        name = []  # 生成的csv文件名

        ws = self.ws
        _col = ws.findcol('第01课') - 1

        # 1 初期要详细提示
        if self.当天课次 < 4:
            msg.append('第1次建议大家都查看下完整考勤表，记一下自己的学号，方便以后核对考勤数据。'
                       '以后群里统一以学号的方式汇报异常的考勤数据，未列出则都是"正常完成当堂学习"的学员。')
            msg.append('如果发现自己考勤数据不对，可以群里、私聊反馈给我修正。')

        # 2 要过期的课程
        if 0 < self.结束课次 + 1 <= self.课次数:
            msg.append(f'第{self.结束课次 + 1}课回放、打卡第{len(self.视频返款) - 1}天，还未完成学习的同学们请抓紧时间。')

        # 3 回放
        回放课次 = self.当天课次 - self.回放返款天数
        if 0 < 回放课次 <= self.课次数:
            title = f'第{回放课次}课回放'
            msg.append('【' + title + '名单】')
            if 回放课次 == 1:
                msg[-1] += f'（第1课仍有{len(self.视频返款) - self.回放返款天数 - 1}天可以回放）'

            d = {}
            for i in range(1, self.回放返款天数 + 1):
                d[f'第{i}天回放'] = []
            d['未完成学习'] = []

            col = _col + 回放课次
            for r in ws.iterrows('用户ID'):
                v = ws.cell(r, col).value
                if v and '回放' in v:
                    t = int(re.search(r'第(\d+)天', v).group(1))
                    if self.视频返款[t]:
                        订单号 = ws.cell2(r, '交易订单号').value
                        if 订单号:
                            cols = [订单号, self.视频返款[t],
                                    f'{self.返款标题}第{回放课次}课第{t}天完成回放',
                                    f'{订单号}_class{回放课次:02}']  # 防止重复返款的校验码
                            ls.append(','.join(map(str, cols)))
                    else:
                        v = '未完成学习'
                elif v in (f'不足{self.达标时长}分钟', '未开始学习'):
                    v = '未完成学习'
                if v and v != '完成当堂学习':
                    d[v].append(ws.cell2(r, '学号').value)

            is_empty = True
            for k, v in d.items():
                if v:
                    m = re.search(r'\d+', k)
                    t = int(m.group()) if m else 5
                    msg[-1] += f'\n{k}(返{self.视频返款[t]}元): ' + ','.join(map(str, v))
                    is_empty = False
            if is_empty:
                msg[-1] = f'第{回放课次}课因当堂满勤，无回放名单'
            else:
                name.append(f'第{回放课次}课回放')

        # 4 当堂
        if self.当天课次 <= self.课次数:
            title = f'第{self.当天课次}课当堂'
            msg.append('【' + title + '缺勤名单】')
            if self.当天课次 == 1:
                msg[-1] += '（注意只统计早晨直播情况，今日的回放数据明天会更新）'
            # d = {f'不足{self.达标时长}分钟': [], '未开始学习': []}
            d = []

            col = _col + self.当天课次
            for r in ws.iterrows('用户ID'):
                v = ws.cell(r, col).value
                if v == '完成当堂学习':
                    订单号 = ws.cell2(r, '交易订单号').value
                    if 订单号:
                        cols = [订单号, self.视频返款[0],
                                f'{self.返款标题}第{self.当天课次}课完成当堂学习',
                                f'{订单号}_class{self.当天课次:02}']  # 防止重复返款的校验码
                        ls.append(','.join(map(str, cols)))
                else:
                    d.append(ws.cell(r, 1).value)

            if d:
                msg[-1] += '\n' + ','.join(map(str, d))
            else:
                msg[-1] = title + '满勤！！！'
            name.append(f'第{self.当天课次}课当堂')

        # 5 第22课的处理
        if self.觉观禅课 and self.当天课次 == 23:
            name.append('第22课')
            for r in ws.iterrows('用户ID'):
                v = ws.cell2(r, '交易订单号').value
                if v:
                    cols = [v, self.视频返款[0],
                            f'{self.返款标题}第{self.当天课次 - 1}课',
                            f'{v}_class22']  # 防止重复返款的校验码
                    ls.append(','.join(map(str, cols)))

        # 6 返款文件
        if ls:
            ls = [x for x in ls if ('订单号' not in x and 'None' not in x)]
            (self.root / (f'第{self.当天课次:02}天 ' + '+'.join(name) + '返款.csv')).write_text('\n'.join(ls))

        # 7 提示信息
        if name:
            msg.append('+'.join(name) + '促学金已返款，同学们请查收。')
        if self.觉观禅课 and self.当天课次 == 22:
            msg.append('今晚第22课答疑不考勤，明天统一返款。')
        return msg

    def 计算打卡返款(self, 打卡次数):
        from bisect import bisect_right

        ks = [0] + list(self.打卡返款.keys())
        ks.sort()

        def find_le(a, x):
            'Find rightmost value less than or equal to x'
            i = bisect_right(a, x)
            if i:
                return a[i - 1]
            raise ValueError

        k = find_le(ks, 打卡次数)
        return self.打卡返款.get(k, 0)

    def 打卡统计(self, msg):
        """ 不知道平台有每个课程的打卡次数，自己用另外一个途径暴力搞的，220226周六15:59 现在应该没用了"""
        from openpyxl.styles import PatternFill

        # 1 至少两个目录才统计
        dirs = list((self.root / '数据表').glob_dirs('用户学习统计*'))
        if len(dirs) < 2:
            return

        # 2 读取数据
        def merge_data(d):
            """ 输入目录，汇总目录下所有xlsx文件的打卡数据

            导出的打卡数据文件尽量小些，不然这个操作速度有点慢。。。
            """
            data = {}
            for f in d.glob('*.xlsx'):
                if f.stem.startswith('~$'):
                    continue
                df = pd.read_excel(f, header=1)
                for idx, x in df.iterrows():
                    user_id = x['用户ID\t']
                    if user_id == user_id:  # nan数值的特性，这个条件不成立
                        value = x['提交打卡\t']
                        data[user_id.strip()] = int(value) if value == value else 0
                else:
                    continue

            return data

        d1, d2 = dirs[0], dirs[-1]
        data1 = merge_data(d1)
        data2 = merge_data(d2)

        # 3 d1数据标准化
        # 最理想的是d1恰好是开课前一天的统计数据，那么d2和d1直接相减即可
        # 如果d1提前了，无法判断具体打卡日期，所以还是直接相减，只能都算上
        # 如果d1延后了，那么要对d1的数据进行预处理，要减掉对应天数，但最低为0，不能为负数
        def parse_date(name):
            """ 输入文件名，获得其日期 """
            m = re.search(r'(\d{2})(\d{2})(\d{2})', name)
            year, month, day = m.groups()
            year = '20' + year
            return date(int(year), int(month), int(day))

        delta = (parse_date(d1.stem) - self.开课日期).days + 1
        if delta > 0:
            data1 = {k: max(0, v - delta) for k, v in data1.items()}

        # 4 计算新的打卡次数
        # 打卡数是有可能超过21次的
        data3 = {k: v - data1.get(k, 0) for k, v in data2.items()}

        # 5 将打卡次数写入表格
        ls = []  # 返款汇总
        ws = self.ws
        color0 = [(255, 0, 0), (255, 255, 128), (255, 255, 0), (0, 255, 0)]
        for i in ws.iterrows('用户ID'):
            c1 = ws.cell2(i, ['打卡返款', '打卡数'])
            v = c1.value = data3.get(ws.cell2(i, '用户ID').value, 0)
            v //= 5
            if v:
                money = self.打卡返款[min(v - 1, 2)]
                订单号 = ws.cell2(i, '交易订单号').value
                cols = [订单号, money, f'{self.返款标题}返学修日志促学金', f'{订单号}_journal']
                if cols[0]:
                    ls.append(','.join(map(str, cols)))
            else:
                money = 0

            c2 = ws.cell2(i, ['打卡返款', '返款'])
            c2.value = money

            color = RgbFormatter(*color0[min(3, v)])
            c1.fill = PatternFill(fgColor=color.hex[-6:], fill_type="solid")
            c2.fill = PatternFill(fgColor=color.hex[-6:], fill_type="solid")

        # 并计算当前的返款额
        desc = '/'.join(map(str, self.打卡返款))
        if self.当天课次 == 25:
            msg.append('已生成截止目前的打卡数据，同学们可以预先核对下，明天最后更新打卡数据后返款。'
                       f'打卡达到"5/10/15"次，依次返回"{desc}"元。'
                       '注：因技术原因，打卡数据无法精确计算，统计遵循宁可多算但无漏算的原则，所以部分同学打卡数会超过21次。')
        elif self.当天课次 == self.课次数 + len(self.视频返款) - 1:
            # 生成打卡返款
            msg.append('已完成学修日志（打卡）促学金的返款。'
                       f'打卡达到"5/10/15"次，依次返回"{desc}"元。')
            (self.root / '学修日志返款.csv').write_text('\n'.join(ls))

    def 打卡统计2(self, msg):
        from openpyxl.styles import PatternFill

        # 1 有打卡统计表才计算
        files = list((self.root / '数据表').glob_files('*打卡*.csv'))
        if len(files) < 1:
            return

        # 2 读取数据
        # df = pd.read_csv(files[-1])
        # data = {}
        # for idx, row in df.iterrows():
        #     data[row['用户id']] = row['打卡次数']

        df = None
        try:
            df = pd.read_excel(files[-1])  # 220804周四08:28，小鹅通更新了模板
        except ValueError:
            pass

        if df is None:
            try:
                df = pd.read_csv(files[-1])  # 221005周三09:19，小鹅通又双叒更新了
            except UnicodeDecodeError:
                pass

        if df is None:
            try:
                df = pd.read_csv(files[-1], encoding="ANSI")  # 240226周一11:21，
            except UnicodeDecodeError:
                pass

        if df is None:
            raise ValueError

        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df.columns = df.columns.map(lambda x: x.strip() if isinstance(x, str) else x)

        # data = Counter([x for x in df['用户id']])
        try:
            data = {row['用户id']: row['打卡次数'] for _, row in df.iterrows()}
        except KeyError:  # 230202周四19:49，另一种实际是xlsx格式，然后再转出csv的情况
            data = Counter([row['user_id'] for _, row in df.iterrows()])

        # 3 将打卡次数写入表格
        ls = []  # 返款汇总
        ws = self.ws
        最高返款额 = max(self.打卡返款.values())
        for i in ws.iterrows('用户ID'):
            c1 = ws.cell2(i, ['打卡返款', '打卡数'])
            打卡次数 = c1.value = data.get(ws.cell2(i, '用户ID').value, 0)
            返款额 = self.计算打卡返款(打卡次数)

            if 返款额 > 0:
                订单号 = ws.cell2(i, '交易订单号').value
                cols = [订单号, 返款额, f'{self.返款标题}返学修日志促学金', f'{订单号}_journal']
                ls.append(','.join(map(str, cols)))

            c2 = ws.cell2(i, ['打卡返款', '返款'])
            c2.value = 返款额

            if 返款额 <= 0:
                color = RgbFormatter(255, 0, 0)
            elif 返款额 < 最高返款额:
                color = RgbFormatter(255, 255, 0).light(1 - 返款额 / 最高返款额)
            elif 返款额 == 最高返款额:  # 最高返款额
                color = RgbFormatter(0, 255, 0)

            c1.fill = PatternFill(fgColor=color.hex[-6:], fill_type="solid")
            c2.fill = PatternFill(fgColor=color.hex[-6:], fill_type="solid")

        if ls:
            ls = [x for x in ls if ('无订单号' not in x and 'None' not in x)]

        # 4 生成通知，及返款文件
        desc1 = '/'.join(map(str, self.打卡返款.keys()))
        desc2 = '/'.join(map(str, self.打卡返款.values()))
        if self.结束课次 == self.课次数 - 1:
            msg.append('已生成截止目前的打卡数据，同学们可以预先核对下，明天最后更新打卡数据后返款。'
                       f'打卡达到"{desc1}"次，依次返回"{desc2}"元。')
        elif self.结束课次 == self.课次数:
            # 生成打卡返款
            msg.append('已完成学修日志（打卡）促学金的返款。'
                       f'打卡达到"{desc1}"次，依次返回"{desc2}"元。')
            (self.root / '学修日志返款.csv').write_text('\n'.join(ls))

    def write_返款总计(self):
        ws = self.ws
        name2idx = {k: i for i, k in enumerate(['完成当堂学习', '第1天回放', '第2天回放', '第3天回放', '第4天回放'])}
        for i in ws.iterrows('用户ID'):
            total = 0
            for k in range(1, 22):
                cel = ws.cell2(i, f'第{k:02}课')
                if not cel:
                    continue
                v = cel.value
                if v in name2idx:
                    idx = name2idx[v]
                    if idx and k > self.当天课次 - self.回放返款天数:  # 还没返款的回放金额
                        continue
                    total += self.视频返款[idx]
            v2 = ws.cell2(i, ['打卡返款', '返款']).value
            if isinstance(v2, int):
                total += v2
            if self.觉观禅课 and self.当天课次 > 22:
                total += self.视频返款[0]
            ws.cell2(i, ['已返款', '总计'], total)

    def 考勤日报(self, debug=False, journal=False):
        msg = []  # 报告内容

        # 1 日常返款
        df = self.获取每日统计表()
        self.更新统计表(df)
        self.生成今日返款文件(msg)

        # 2 打卡返款
        if journal or self.结束课次 == self.课次数:
            self.打卡统计2(msg)
        self.write_返款总计()

        # 3 结课
        if self.当天课次 == (self.课次数 + len(self.视频返款) - 1):
            msg.append('考勤返款工作已全部完成，若对自己促学金还有疑问的近日可以再跟我反馈。')

        # 4 输出日报
        if len(msg) > 1:
            msg = '\n'.join([f'{i}、{x}' for i, x in enumerate(msg, start=1)])  # 编号
        else:
            msg = msg[0]
        if self.在线表格:
            msg = f'【完整考勤表】{self.在线表格}\n' + msg
        print(msg)

        # 5 展示或保存表格内容
        if debug:
            browser.html(self.ws.to_html())
        else:
            # self.wb.save(self.表格路径)
            self.wb.save(self.root / (XlPath(self.表格路径).stem + '.xlsx'))

    def cmd(self):
        """ 初始化类后不关闭，继续轮询执行功能 """
        cmd = input('>')
        while cmd != 'exit':
            fire.Fire(self, cmd)
            cmd = input('>')

    def __call__(self, *args, **kwargs):
        print('程序测试通过，可正常使用')

    def 表格内容对齐(self, ws_name1, ws_name2, col_name):
        wb = self.wb
        wb.merge_sheets_by_keycol([wb[ws_name1], wb[ws_name2]], col_name)

        # 保存。一般不用重新再加载self.wb、self.ws，因为这里需求本来基本都是要分段重新执行程序的。
        wb.save(self.表格路径)

    def 匹配交易单号(self, value):
        """ 更加智能的一条龙匹配操作

        :param value: 需要输入待检索的金额值
        """
        # 1 读取账单，账单文件可能不唯一，可能有重复可以自动去重
        files = XlPath('数据表').rglob_files('*基本账户*.csv')
        df_list = []
        for f in files:
            df = pd.read_csv(f)
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
        # 按照"资金流水单号"去重
        df = df.drop_duplicates(subset=['资金流水单号'], keep='first')
        df = df[df['收支类型'] == '`收入']
        df.reset_index(drop=True, inplace=True)

        # 2 因为情况比较特殊，这里不调用通用的对齐功能，而是定制化写过
        ws = self.wb['报名表']

        data = ws.iterrows('真实姓名', to_dict=['交易单号'])
        last_row = -1
        for i, row in data:
            last_row = i
            # 在 df['微信支付业务单号'] 找是否有 row['交易单号']
            items = df[df['微信支付业务单号'] == row['交易单号']]
            if items.empty:
                continue

            # 如果匹配，理论上只有一条
            item = items.iloc[0]
            ws.cell2(i, '交易订单号').value = item['业务凭证号'][1:]
            ws.cell2(i, '订单金额').value = item['收支金额(元)']

            # 在df中去掉所有items
            df = df.drop(items.index)

        # 3 匹配完后，还有目标金额的数据要列出来
        items = df[df['收支金额(元)'] == f'`{value:.2f}']
        for idx, row in items.iterrows():
            last_row += 1
            ws.cell2(last_row, '交易单号').value = row['微信支付业务单号']
            ws.cell2(last_row, '交易订单号').value = row['业务凭证号'][1:]
            ws.cell2(last_row, '订单金额').value = row['收支金额(元)']

        # 4 保存
        self.wb.save(self.表格路径)

    def __2_自动浏览网页(self):
        pass

    def ensure_driver(self):
        if not hasattr(self, 'driver'):
            self.driver = None
        if not self.driver:
            self.driver = XlChrome()
        return self.driver

    def 登录小鹅通(self, name, passwd):
        # 登录小鹅通
        driver = self.ensure_driver()
        driver.get('https://admin.xiaoe-tech.com/t/login#/acount')
        driver.locate('//*[@id="common_template_mounted_el_container"]'
                      '/div/div[1]/div[3]/div/div[4]/div/div[1]/div[1]/div/div[2]/input').send_keys(name)
        driver.locate('//*[@id="common_template_mounted_el_container"]'
                      '/div/div[1]/div[3]/div/div[4]/div/div[1]/div[2]/div/div/input').send_keys(passwd)
        driver.click('//*[@id="common_template_mounted_el_container"]/div/div[1]/div[3]/div/div[4]/div/div[2]')

        # 然后自己手动操作验证码
        # 以及选择"店铺"

    def 下载课次考勤数据(self, 起始课=None, 终止课=None, 文件名前缀=''):
        if 起始课 is None:
            起始课 = max(1, self.结束课次)
        if 终止课 is None:
            终止课 = min(self.当天课次, self.课次数)

        # 1 遍历课程下载表格
        driver = self.driver
        for i in range(起始课, 终止课 + 1):
            print(f'第{i}课', self.课程链接[i])
            driver.get('https://admin.xiaoe-tech.com/t/data_center/index')  # 必须要找个过渡页，不然不会更新课程链接
            driver.get(self.课程链接[i])
            # 不能写'第{i}课'，会有叫'第{i}堂'等其他情况
            driver.locate_text('//*[@id="app"]/div/div/div[1]/div[2]/div[1]/div[2]', f'{i}')  # 检查页面

            driver.click('//*[@id="tab-studentTab"]/span')  # 直播间用户
            driver.click('//*[@id="pane-studentTab"]/div/div[2]/div[2]/form/div[2]/button[2]/span/span')  # 导出列表
            driver.click('//*[@id="data-export-container"]/div/div[2]/div/div[2]/div[2]/button[2]/span/span')  # 导出
            time.sleep(1)

        # 2 下载表格
        # 2.1 等待下载文件生成完毕
        while True:
            driver.get('https://admin.xiaoe-tech.com/t/basic-platform/downloadCenter')
            if '任务撤回' in driver.locate('//*[@id="downloadsCenter"]/div[4]/div/div[4]/div[2]/table/tbody/tr[1]/'
                                           'td[9]/div/div/span').text:
                time.sleep(1)
            else:
                time.sleep(2)  # 不能下载太快，还是要稍微等一会
                break

        # 2.2 下载表格
        files = []
        for i in range(1, 终止课 - 起始课 + 2):
            driver.click(f'//*[@id="downloadsCenter"]/div[4]/div/div[4]/div[2]/table/tbody/tr[{i}]/'
                         f'td[9]/div/div/span[1]')
            file = driver.locate(f'//*[@id="downloadsCenter"]/div[4]/div/div[3]/table/tbody/tr[{i}]/'
                                 f'td[1]/div/div/div').text + '.csv'  # 下载后还会再多一个csv后缀
            files.append(file)
            time.sleep(1)

        # 2.3 移动表格
        time.sleep(1)
        # 这样应该也不是很泛用，有的人浏览器下载可能会放到各种自定义的其他地方
        download_path = XlPath(os.path.join(os.environ['USERPROFILE'], 'Downloads'))
        for file in files:
            src_file = (download_path / file)
            while True:
                if src_file.is_file():
                    src_file.move(self.root / '数据表' / (文件名前缀 + file), if_exists='replace')
                    break
                else:
                    time.sleep(1)

    def 批量退款(self):
        self.driver.get('https://pay.weixin.qq.com/index.php/xphp/cbatchrefund/batch_refund#/pages/index/index')


class 网课考勤2(网课考勤):
    def login_xe(self):
        self.xe = Xiaoetong()  # 实例化
        self.xe.login(self.app_id,
                      self.client_id,
                      self.secret_key)  # 获取了token
        
        return self.xe.token
    # 依据课程链接，获取资源id（与课次）
    def 获取课次与资源id(self):
        课程链接 = self.课程链接[1:]
        ls_resource_id = [""]
        for item in 课程链接:  # 课次
            resource_id = re.search(r"detail\?id=(.+?)\&", item)  # 资源id
            ls_resource_id.append(resource_id.group(1))
        return ls_resource_id

    # 获取直播间用户数据
    def 获取直播间用户数据(self, resource_id, path):
        if path.is_file():
            return
            # 2）获取直播间用户数据：
        lst = self.xe.get_alive_user_list(resource_id)
        fieldnames = ['用户ID', '用户昵称', '备注名', '状态', '直播间停留时长(秒)', '直播间停留时长',
                      '累计观看时长(秒)', '累计观看时长', '直播观看时长(秒)', '直播观看时长', '回放观看时长(秒)',
                      '回放观看时长', '评论次数', '直播间成交金额']
        p = path
        with open(p, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for x in lst:
                record = {
                    '用户ID': x['user_id'],
                    '用户昵称': x['wx_nickname'],
                    '备注名': None,
                    '状态': "其它关联权益",
                    '直播间停留时长(秒)': x['his_online_time'],
                    '直播间停留时长': str(timedelta(seconds=x['his_online_time'])),
                    '累计观看时长(秒)': x['his_learn_time'],
                    '累计观看时长': str(timedelta(seconds=x['his_learn_time'])),
                    '直播观看时长(秒)': x['his_learning_time'],
                    '直播观看时长': str(timedelta(seconds=x['his_learning_time'])),
                    '回放观看时长(秒)': x['his_learned_time'],
                    '回放观看时长': str(timedelta(seconds=x['his_learned_time'])),
                    '评论次数': x['comment_num'],
                    '直播间成交金额': x['user_total_price']
                }
                writer.writerow(record)

    def 获取课次列表(self):
        return list(range(max(self.结束课次2, 1), self.当天课次2 + 1))

    def 下载课程(self):
        prfx = self.prfx
        lt = self.获取课次与资源id()
        for i in tqdm(self.获取课次列表()):
            resource_id = lt[i]
            formatted_date = self.today  # datetime.datetime.now().strftime("%Y-%m-%d")
            path = prfx.format(x=i, y=formatted_date)
            self.获取直播间用户数据(resource_id, self.root / "数据表" / path)

    # 20240206 新增【针对打卡部分
    def 获取打卡id(self):
        """ 依据打卡链接，获取activity_id(打卡id)
        """
        打卡链接 = self.打卡链接[1:]
        ls_activity_id = []
        for item in 打卡链接:  # 课次
            activity_id = re.search(r"\?activity_id=(.+?)\&", item)
            ls_activity_id.append(activity_id.group(1))
        return ls_activity_id

    def 获取打卡参与用户(self, activity_id, path):
        # 1）如果路径中已经有了，就跳过
        if path.is_file():
            return
            # 2）获取打卡用户数据：
        lst = self.xe.get_elock_actor(activity_id)
        fieldnames = ['用户id', '用户昵称', '打卡昵称', '打卡分组', '姓名', '电话', '最近采集号码', '城市', '微信号',
                      '打卡天数', '打卡次数', '被点赞数', '被评论数', '被点评数', '被精选数', '参与时间']
        p = path
        with open(p, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for x in lst:
                record = {
                    '用户id': x['user_id'],
                    '用户昵称': x['wx_nickname'],
                    '打卡昵称': x['clock_nickname'],
                    '打卡分组': None,
                    '姓名': x['wx_nickname'],
                    '电话': x['phone'],
                    '最近采集号码': None,
                    '城市': x['wx_city'],
                    '微信号': None,
                    '打卡天数': x['clock_days'],
                    '打卡次数': x['clock_days'],
                    '被点赞数': x['zan_count'],
                    '被评论数': x['comment_count'],
                    '被点评数': x['review_count'],
                    '被精选数': 0,
                    '参与时间': x['created_at']
                }
                writer.writerow(record)

    def 下载打卡数据(self):
        prfx = "{x}-" + f"《{self.返款标题}技术公益网课【中心教室】-日历打卡学员数据.csv"
        lt = self.获取打卡id()
        for i in tqdm(range(1)):
            activity_id = lt[i]
            formatted_date = self.today  # datetime.datetime.now().strftime("%Y-%m-%d")
            path = prfx.format(x=formatted_date)
            self.获取打卡参与用户(activity_id, self.root / "数据表" / path)


class OldKqDb(Connection):
    """ 五一身心行修考勤工具 """

    def __init__(self, dbfile='5034.db', wbpath='考勤.xlsx', *args, **kwargs):
        super().__init__(dbfile, *args, **kwargs)
        p = XlPath(wbpath)
        self.wb = openpyxl.load_workbook(p)
        self.outwb = p.with_name(p.stem + '+' + p.suffix)
        # 开营第几天
        self.days = (datetime.datetime.today() - datetime.datetime(2022, 4, 29)).days

    def update_小鹅通数据(self):
        # 1 观看记录
        def add_one_csv(path, 课次名=None):
            """ 添加一个csv文件的数据 """
            # 1 确定表格存在
            if not self.has_table('观看记录'):
                cols = ['课次名 text', '用户ID text', '用户昵称 text',
                        '直播间停留秒数 integet', '累计观看秒数 integer', '直播观看秒数 integer',
                        '回放观看秒数 integer',
                        '首次进入时间 text', '最近进入时间 text', '记录时间点 text']
                cols = ', '.join(cols)
                self.execute(f'CREATE TABLE 观看记录 ({cols}, PRIMARY KEY (课次名, 用户ID, 直播间停留秒数))')

            # 2 解析csv的内容到sql中
            if not 课次名:
                课次名 = re.search(r'《(.+?)》', str(path)).group(1)
                if 课次名 == '测试链接2':
                    课次名 = '2022五一线上觉观营-0429开营'

            time_tag = datetime.datetime.fromtimestamp(os.stat(path).st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            df = pd.read_csv(path, skiprows=1)
            for idx, row in df.iterrows():
                d = {'课次名': 课次名,
                     '用户ID': row['用户ID'].strip(),
                     '用户昵称': row['用户昵称'],
                     '直播间停留秒数': row['直播间停留时长(秒)'],
                     '累计观看秒数': row['累计观看时长(秒)'],
                     '直播观看秒数': row['直播观看时长(秒)'],
                     '回放观看秒数': row['回放观看时长(秒)'],
                     '首次进入时间': row['首次进入时间'].strip(),
                     '最近进入时间': row['最近进入时间'].strip(),
                     '记录时间点': time_tag,
                     }
                self.insert_row('观看记录', d)
            self.commit()

        for f in XlPath('数据表').glob('*直播观看详情*.csv'):
            add_one_csv(f)

        # 2 打卡记录
        self.execute('DROP TABLE IF EXISTS 打卡记录')
        # 时间点最新的数据
        f = list(XlPath('数据表').glob('*打卡日记*.csv'))[-1]
        df = pd.read_excel(f)
        df.columns = df.columns.str.replace('\t', '')  # 删除列名里的\t
        df = df.replace('\t', '', regex=True)  # 删除值里的\t
        df.to_sql('打卡记录', con=self)

    def update_用户列表(self, path):
        """ 更新用户列表数据 """
        # 1 删除旧表，创建新表
        self.execute('DROP TABLE IF EXISTS 用户列表')
        # 需要预取一下数据，知道所有字段
        df = pd.read_csv(path)
        cols = {k: 'text' for k in df.columns}
        cols['年龄'] = 'integer'
        desc = ','.join([f'{k} {v}' for k, v in cols.items()])
        self.execute(f'CREATE TABLE 用户列表 ({desc}, PRIMARY KEY (用户ID))')
        self.commit()

        # 2 插入每一行数据
        for idx, row in df.iterrows():
            for k in ['账户绑定手机号', '最近采集手机号']:
                # 有的是写成公式的'=手机号'
                if isinstance(row[k], str) and '=' in row[k]:
                    row[k] = re.search(r'\d+', row[k]).group()
            self.insert_row('用户列表', row.to_dict())
        self.commit()

    def 用户信息摘要(self, x):
        """ 输入series类型的一个条目，输出其摘要信息 """
        ls = [f'{k}={v}' for k, v in x.items() if (not isinstance(v, float) or not math.isnan(v))]
        v = x["用户ID"].strip()
        # if v == 'u_615026b5b5db9_0uLTQndbsC':
        #     print(v)
        n = self.execute(f'SELECT COUNT(*) FROM 观看记录 WHERE 用户ID="{v}"').fetchone()[0]
        if n:
            ls.append(f'考勤{n}次')
        return ', '.join(ls)

    def 查找用户(self, 昵称='', 手机号='', *, debug=False):
        """
        :param 昵称: 昵称可以输入一个列表，会在目标昵称、真实姓名同时做检索
        :param 手机号: 手机也可以输入一个列表，因为有些人可能报名时手机号填错，可以增加一些匹配规则
        :return:
        """

        # 1 统一输入参数格式
        def tolist(x):
            if x and not isinstance(x, list):
                x = [x]
            x = [str(a) for a in x if a]
            return x

        def check_telphone(v, refs):
            if not isinstance(v, str):
                try:
                    v = int(v)
                except:
                    pass
                v = str(v)
            for a in refs:
                if a in v:
                    return True

        昵称 = tolist(昵称)
        手机号 = tolist(手机号)
        手机号 = [f'{x}' for x in 手机号]

        # 2 查找所有可能匹配的项目
        ls = []
        for x in self.exec_dict('SELECT * FROM 用户列表'):
            logo = False
            if 昵称:
                if x['昵称'] in 昵称 or x['真实姓名'] in 昵称:
                    logo = True
            if 手机号:
                if check_telphone(x['账户绑定手机号'], 手机号) or check_telphone(x['最近采集手机号'], 手机号):
                    logo = True
            if logo:
                ls.append(x)

        if debug:
            print('\n'.join(map(self.用户信息摘要, ls)))
            return

        return ls

    def 匹配用户ID(self, wbpath, sheet='报名表'):
        p = XlPath(wbpath)
        wb = openpyxl.load_workbook(wbpath)
        ws = wb[sheet]

        def try2int(x):
            try:
                return int(x)
            except:
                if isinstance(x, float) and math.isnan(x):
                    return ''
                return x

        def todict(row):
            """ 将表格第i行的数据提取为字典格式
            """
            msg = {}
            for k, x in enumerate(['姓名', '微信昵称', '手机', '微信']):
                msg[x] = ws.cell2(row, x).value
            return msg

        for row in tqdm(list(ws.iterrows('姓名')), desc='匹配进度'):
            x = todict(row)
            待查手机号 = [x['手机']]
            t = try2int(x['微信'])
            if t:
                待查手机号.append(t)
            ls = self.查找用户([x['姓名'], x['微信昵称']], 待查手机号)
            # 只有一个关联的直接匹配上，否则填空
            ws.cell2(row, '用户ID', ls[0]['用户ID'] if len(ls) == 1 else '')
            ws.cell2(row, '参考信息', '\n'.join(map(self.用户信息摘要, ls)))
            row += 1

            # if len(ls) > 1 and ls[1]['用户ID'] == 'u_615026b5b5db9_0uLTQndbsC':
            #     break

        wb.save(str(p.with_name(p.stem + '+' + p.suffix)))

    def get_user_观看打卡记录(self, user_id):
        """ 获得一个用户的所有观看记录情况 """

        # 1 观看记录
        classes = [
            ['2022五一线上觉观营-0429开营', '2022-04-29 19:00', '2022-04-29 20:58', 0],
            ['4.30第一堂', '2022-04-30 05:20', '2022-04-30 06:27', 420],
            ['4.30第二堂', '2022-04-30 08:00', '2022-04-30 10:40', 2400],
            # 张国莉师兄说以时间表为准，所以我适当放宽。以15:30结束为准，结束前3分钟退出不算早退。
            ['4.30第三堂', '2022-04-30 13:30', '2022-04-30 15:33', 180],
            ['4.30第四堂', '2022-04-30 19:30', '2022-04-30 21:03', 180],
            ['5.1第一堂', '2022-05-01 05:20', '2022-05-01 06:17', 0],
            ['5.1第二堂', '2022-05-01 08:00', '2022-05-01 10:13', 13 * 60],
            ['5.1第三堂', '2022-05-01 13:30', '2022-05-01 15:11', 0],
            ['5.1第四堂', '2022-05-01 19:30', '2022-05-01 21:00', 0],
            ['5.2第一堂', '2022-05-02 05:20', '2022-05-02 06:03', 0],
            ['5.2第二堂', '2022-05-02 08:00', '2022-05-02 10:07', 7 * 60],
            ['5.2第三堂', '2022-05-02 13:30', '2022-05-02 15:28', 0],
            ['5.2第四堂', '2022-05-02 19:30', '2022-05-02 21:02', 2 * 60],
            ['5.3第一堂', '2022-05-03 05:20', '2022-05-03 06:11', 0],
            ['5.3第二堂', '2022-05-03 08:00', '2022-05-03 10:12', 12 * 60],
            ['5.3第三堂', '2022-05-03 13:30', '2022-05-03 15:43', 13 * 60],
            ['5.3第四堂', '2022-05-03 19:30', '2022-05-03 20:57', 0],
            ['5.4第一堂', '2022-05-04 05:20', '2022-05-04 06:22', 2 * 60],
            ['5.4第二堂', '2022-05-04 08:00', '2022-05-04 09:39', 0],
            ['5.4第三堂', '2022-05-04 13:30', '2022-05-04 15:43', 13 * 60],
            ['5.4第四堂', '2022-05-04 19:30', '2022-05-04 21:02', 2 * 60],
            ['5.5第一堂', '2022-05-05 05:20', '2022-05-05 06:14', 0],
            ['5.5第二堂', '2022-05-05 08:00', '2022-05-05 10:05', 5 * 60],
            ['5.5第三堂', '2022-05-05 13:30', '2022-05-05 15:21', 0],
            ['5.5第四堂', '2022-05-05 19:30', '2022-05-05 21:02', 2 * 60],
            ['5.6第一堂', '2022-05-06 05:20', '2022-05-06 06:23', 3 * 60],
            ['5.6第二堂', '2022-05-06 08:00', '2022-05-06 10:15', 15 * 60],
            ['5.6第三堂', '2022-05-06 13:30', '2022-05-06 15:26', 0],
        ]

        def 课次message(name):
            for k, item in enumerate(classes):
                if item[0] == name:
                    da = datetime.datetime.fromisoformat(item[1])
                    db = datetime.datetime.fromisoformat(item[2])
                    return k, da, db, item[3]

        items = self.exec_dict(f'SELECT * FROM 观看记录 WHERE 用户ID="{user_id}"')
        ls = ['缺勤'] * len(classes) + [''] * (36 - len(classes))
        for x in items:
            k, da, db, bias = 课次message(x['课次名'])

            # d1: 提早进入直播间的，顺推至标准开课时间
            d1 = datetime.datetime.fromisoformat(x['首次进入时间'])
            if d1 < da:
                if d1 <= datetime.datetime(2022, 5, 4) and user_id != 'u_612d861be8fde_BfDyBtACTv':
                    # 5月4日之前的，缺勤做了大放水~~ 提前登录的，可以换好多正课时长~~ 但是李娟的单独调整
                    x['直播观看秒数'] += (da - d1).seconds
                d1 = da

            # d2: 进入时间，加上直播间观察时间作为退出时间
            d2 = d1 + datetime.timedelta(seconds=x['直播观看秒数'])
            # 结束时间不超过db
            if d2 > db:
                d2 = db

            desc = ''
            if max((d1 - da).seconds, 0) + max((db - d2).seconds, 0) >= (1800 + bias):
                desc = ',缺勤'
            else:
                if (d1 - da).seconds > 59:
                    desc = ',迟到'

            # 直播结束才进入的
            if d1 >= db:
                ls[k] = '缺勤'
            else:
                ls[k] = d1.strftime('%H:%M') + '/' + d2.strftime('%H:%M') + desc
        观看记录 = ls

        if user_id == 'u_61ad24c13ec15_qBIqwSvejE':  # 1组占萍
            观看记录[0] = '19:09/20:58,迟到'

        # 2 打卡记录
        def brief_time(x, ref):
            """ x相对ref的简化时间显示 """
            t = int((x - ref).seconds / 60)
            tag = f'{t // 60:02}:{t % 60:02}'
            if t < 0:
                tag = '-' + tag
            return tag

        打卡记录 = [''] * 9
        start_day = datetime.datetime(2022, 4, 29)
        for k, tag in enumerate(['立下学修目标', '第一天', '第二天', '第三天', '第四天', '第五天', '第六天', '第七天']):
            x = self.execute(f'SELECT 打卡时间 FROM 打卡记录 WHERE user_id="{user_id}" AND 所属主题="{tag}"').fetchall()
            x = [a[0] for a in x]  # 只有一列数据
            if not x:
                打卡记录[k] = '未打卡'
            else:
                x = sorted(x)  # 排序后分别处理
                d1 = start_day + datetime.timedelta(days=k)
                d2 = start_day + datetime.timedelta(days=k, hours=19)
                打卡记录[k] = ','.join([brief_time(datetime.datetime.fromisoformat(a), d1) for a in x])
                if datetime.datetime.fromisoformat(x[0]) > d2:
                    打卡记录[k] += ',迟打'

        # 3 观看和打卡数据合并
        用户记录 = [观看记录[0], 打卡记录[0]]
        for i in range(1, 9):
            b = i * 4 + 1
            用户记录 += 观看记录[b - 4:b] + [打卡记录[i]]

        return 用户记录

    def update_wb(self):
        """ 更新工作薄考勤内容 """
        ws = self.wb['考勤表']
        # 1 考勤数据
        for i in ws.iterrows('用户ID'):
            user_id = ws.cell2(i, '用户ID').value
            msg = self.get_user_观看打卡记录(user_id)
            for j, v in enumerate(msg, start=10):
                if v:
                    c = ws.cell(i, j)
                    v0 = c.value
                    if v0:  # 如果有特殊标记，需要修正处理
                        m = re.search(grp_chinese_char(), v)  # 判断原有v中是否有中文
                        if m:  # 如果v有中文，需要替换为v0的标记
                            v = re.sub(grp_chinese_char() + '+', v0, v)
                        elif v:  # 如果v没有中文，直接在加上特殊标记后缀
                            v += f',{v0}'
                        else:  # 如果v也没有标记，保留v0原始标记内容
                            v = v0
                    c.value = v
                    if '上班' in v or '工作' in v:
                        color = '鲜红'
                    elif '请假' in v:
                        color = '灰色'
                    elif '迟到' in v or '迟打' in v:
                        color = '鲜黄'
                    elif '缺勤' in v or '未打卡' in v:
                        color = '鲜红'
                    else:
                        color = '鲜绿色'
                    c.fill_color(color)

        # 2 剩余促学金
        for i in ws.iterrows('用户ID'):
            money = 1000
            累计缺勤天数 = 0
            for t in range(1, min(8, (datetime.datetime.today() - datetime.datetime(2022, 4, 28)).days)):  # 第几天
                logo = False
                # 每天上课情况
                迟到_num, 缺勤_num, 请假_num = 0, 0, 0
                for jj in range(1, 5):
                    j = t * 5 + 6 + jj
                    v = ws.cell(i, j).value
                    if not v:
                        v = ''
                    if '迟到' in v:
                        迟到_num += 1
                    elif '缺勤' in v or '上班' in v or '工作' in v:
                        缺勤_num += 1
                    elif '请假' in v:
                        请假_num += 1
                if 迟到_num + 缺勤_num >= 3:
                    money -= 240
                    logo = True
                else:
                    money -= 迟到_num * 25
                    money -= 缺勤_num * 60
                    # money -= 请假_num * 20

                if 缺勤_num:
                    累计缺勤天数 += 1
                    if 累计缺勤天数 >= 3:
                        money -= 1000
                else:
                    累计缺勤天数 = 0

                # 每天打卡情况
                v = ws.cell(i, t * 5 + 11).value
                if v and not logo:
                    if '迟打' in v:
                        money -= 30
                    elif '未打卡' in v:
                        money -= 60
            ws.cell2(i, ['促学金', '剩余']).value = max(0, money)

        # 3 给整行加上分割线
        from openpyxl.styles import Border, Side, Alignment
        for i in ws.iterrows('用户ID'):
            v = ws.cell(i, 1).value
            if isinstance(v, str) and '组' in v:
                side = Side(border_style='medium', color='000000')
                border = Border(top=side)
                for j in range(1, 50):
                    ws.cell(i, j).border = border
                    if j > 9:
                        ws.cell(i, j).alignment = Alignment(horizontal='left', vertical='center')

        # 4 各组综合情况
        # 4.1 各组打卡情况
        打卡天数 = 6
        ls = []
        title = ''  # 组名
        columns = ['分组', '姓名', '日期', '打卡状态']
        for i in ws.iterrows('用户ID'):
            v = ws.cell(i, 1).value
            if isinstance(v, str) and '组' in v:
                title = f'第{int(chinese2digits(v[1:-1])):02}组'
            for j in range(打卡天数):
                打卡状态 = ws.cell(i, 16 + j * 5).value
                ls.append([title, ws.cell2(i, '姓名').value, j, 打卡状态])
        df = pd.DataFrame.from_records(ls, columns=columns)

        ls = []
        columns = ['分组', '总人数', '每天平均补打人数及比例', '每天平均未打人数及未打比例',
                   '4月30日未打', '5月1日未打', '5月2日未打', '5月3日未打', '5月4日未打', '5月5日未打']
        for title, items in df.groupby('分组'):
            n = len(items)
            a = sum(items['打卡状态'].str.contains('迟打'))
            b = sum(items['打卡状态'].str.contains('未打卡'))
            record = [title, n // 打卡天数, f'{a / 打卡天数:.1f}，{a / n:.0%}', f'{b / 打卡天数:.1f}，{b / n:.0%}']
            for j in range(打卡天数):
                msg = []
                补打, 未打 = 0, 0
                # 迟打
                # xs = items[(items['日期'] == j) & (items['打卡状态'].str.contains('迟打'))]['姓名']
                # if len(xs):
                #     补打 += len(xs)
                #     msg.append(f'补打{len(xs)}人')
                # 未打
                xs = items[(items['日期'] == j) & (items['打卡状态'].str.contains('未打卡'))]['姓名']
                if len(xs):
                    未打 += len(xs)
                    msg.append(f'{len(xs)}人：{"，".join(xs)}')
                record.append('，'.join(msg))
            ls.append(record)

        df = pd.DataFrame.from_records(ls, columns=columns)
        # browser(df)

        # 4.2 每天打卡情况
        self.wb.save(self.outwb)


class 课次数据:
    """ 读取课次数据 """

    def __init__(self):
        # 将各种不同来源的数据，按用户ID分组存储起来
        self.用户观看数据 = defaultdict(list)  # 每个用户的观看数据情况
        self.start_day = None  # 记录所有加载文件中，最早的时间戳，大概率就是这个课次的开课时间

    def add_files(self, path, pattern):
        files = XlPath(path).select_file(pattern)
        for f in files:
            self.add_考勤数据(f)

    def add_考勤数据(self, file):
        """ 推荐使用这个综合接口，代替专用接口，这个接口能实现一些综合性的操作
        保存的用户数据，存储三元数值（观看时长，进度）
            150，表示观看时间满30分钟
            0~100，表示有进度数据，0%~100%
        """
        # 0 有普通考勤表，和从圈子下载的表格两种
        enc = get_encoding(file.read_bytes()) or 'gbk'
        df = pd.read_csv(file, encoding=enc, encoding_errors='ignore')

        if '参与状态' in df.columns:
            self.add_圈子进度表(file)
        else:
            self.add_小鹅通考勤表(file)

        dates = re.findall(r'(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}', file.stem)
        if dates:
            file_day = datetime.datetime.strptime(dates[-1], "%Y-%m-%d").date()
            if self.start_day is None or self.start_day > file_day:
                self.start_day = file_day

    def add_小鹅通考勤表(self, file):
        file = XlPath(file)

        df = pd.read_csv(file, encoding='utf8')
        assert '直播观看时长(秒)' in df.columns, f'好像下载到空数据表 {file}'

        for idx, x in df.iterrows():
            data = {'文件名': XlPath(file).stem}
            data['直播分钟'] = int(x['直播观看时长(秒)']) // 60
            data['回放分钟'] = int(x['回放观看时长(秒)']) // 60
            self.用户观看数据[x['用户ID']].append(data)

    def add_圈子进度表(self, file):
        """ 圈子的考勤数据比较特别，需要独立的一个体系来存储
        大部分会有播放进度，
        """
        file = XlPath(file)
        # 圈子数据目前是gbk编码，这样会有些问题。但可能哪天平台就改成utf8编码了，所以这里提前做好兼容。
        enc = get_encoding(file.read_bytes()) or 'gbk'
        df = pd.read_csv(file, encoding=enc, encoding_errors='ignore')

        for idx, x in df.iterrows():
            data = {'文件名': file.stem}

            if '播放进度' in x:  # 一般会有详细的播放进度
                data['百分比进度'] = int(re.search(r'\d+', x['播放进度']).group())
            else:  # 如果没有，也可以通过参与状态知道是否完成
                data['百分比进度'] = 100 if x['参与状态'] == '已完成' else 0
            # 虽然"已完成"，但显示进度可能不是'100%'，要补一个修正
            if x['参与状态'] == '已完成':
                data['百分比进度'] = 100

            self.用户观看数据[x['用户ID']].append(data)

    def 禅宗考勤结果(self, user_id):
        """ 禅宗考勤是比较简单的，不用考虑回放情况，只要在所有数据中，判断出有完成就行
        当然，细节考虑，也还是可以详细区分不同情况。

        :return:
            str，一般是返回字符串，表示结论
            dict，但也可以返回一个字典，表示特殊的格式设置，这种是用于单元格颜色设置
        """
        直播分钟, 回放分钟, 百分比进度 = 0, 0, 0

        if user_id not in self.用户观看数据:
            return {'value': f'未开始', 'color': '白色'}

        # 1 遍历所有文件数据，获得所有最大值
        data = self.用户观看数据[user_id]  # 这个类的功能框架，已经按照user_id进行了数据分组统计
        for x in data:
            # 前两条是兼容以前的通用的直播课程数据
            直播分钟 = max(直播分钟, x.get('直播分钟', 0))
            回放分钟 = max(回放分钟, x.get('回放分钟', 0))
            # 这一条是兼容禅宗特有的百分比进度
            百分比进度 = max(百分比进度, x.get('百分比进度', 0))

        # 2 归纳出本课次考勤结论
        yellow = RgbFormatter.from_name('鲜黄')
        if 百分比进度 == 100 or 回放分钟 >= 30:
            return {'value': '已完成', 'color': '鲜绿色'}
        elif 百分比进度:
            return {'value': f'进度{百分比进度}%', 'color': yellow.light((100 - 百分比进度) / 100)}
        elif 回放分钟:
            return {'value': f'观看{回放分钟}分钟', 'color': yellow.light((100 - 回放分钟) / 30)}
        else:
            return {'value': f'未开始', 'color': '白色'}

    def 小鹅通考勤结果(self, user_id, 需求分钟=30):
        """ 一般是把一个课次的多天数据添加到一起用，用这个接口获得该课次的观看状态
        比如是当堂，还是第1天回放等

        :param user_id: 要获得的用户信息
        :param 需求分钟: 判断完成的依据分钟数
        """
        # 1 检查文件日期
        if user_id not in self.用户观看数据:
            return '未开始学习'

        data = self.用户观看数据[user_id]

        # 理论上这里文件名就应该是排好序的
        filenames = [x['文件名'] for x in data]

        for i, filename in enumerate(filenames):
            # 后面一串是时分秒，并不提取，但是用来限定格式匹配，避免匹配错
            dates = re.findall(r'(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}', filename)
            if not dates:  # 找不到时间戳的不处理
                continue

            file_date_obj = datetime.datetime.strptime(dates[0], "%Y-%m-%d").date()
            delta_day = (file_date_obj - self.start_day).days

            # 2 按顺序提取时间
            if delta_day == 0 and data[i]['直播分钟'] >= 需求分钟:
                return '完成当堂学习'
            elif delta_day > 0 and data[i]['直播分钟'] + data[i]['回放分钟'] >= 需求分钟:
                return f'第{delta_day}天回放'

        if data[-1]['直播分钟'] + data[-1]['回放分钟']:
            return f'不足{需求分钟}分钟'
        else:
            return '未开始学习'

    def 小鹅通考勤结果2(self, user_id, 返款梯度, 要求在线分钟=30):
        """ 输入视频返款的梯度，这个函数相比`小鹅通考勤结果`，还会返回单元格颜色格式，对应的返款额 """
        # 1 判断课程回放是不是结束了
        text = self.小鹅通考勤结果(user_id, 要求在线分钟)
        if text == '未开始学习' or '不足' in text:  # 判断该课次是否已经结束了
            delta_day = datetime.timedelta(days=len(返款梯度))
            if (delta_day + self.start_day) <= datetime.date.today():
                text = '未完成学习'

        if text == '完成当堂学习':
            return text, '鲜绿色', 返款梯度[0]
        elif m := re.match(r'第(\d+)天回放', text):
            t = int(m.group(1))
            money = 返款梯度[t]
            if money:
                color = RgbFormatter.from_name('黄色')
                color = color.light((返款梯度[0] - money) / money)  # 根据返款额度自动变浅
            else:
                color = '灰色'
            return text, color, money
        elif text == '未开始学习' or '不足' in text:
            return text, '白色', 0
        elif text == '未完成学习':
            return text, '红色', 0
        else:
            raise ValueError


def get_driver(_driver_store=[None]):  # trick
    """ 考勤这边固定一个driver来使用 """
    if _driver_store[0] is None:
        _driver_store[0] = XlChrome()
    if not _driver_store[0]:  # 如果驱动没了，重新启动
        _driver_store[0] = XlChrome()
    return _driver_store[0]


def 登录小鹅通(name, passwd):
    # 登录小鹅通
    driver = get_driver()
    driver.get('https://admin.xiaoe-tech.com/t/login#/acount')
    driver.locate('//*[@id="common_template_mounted_el_container"]'
                  '/div/div[1]/div[3]/div/div[4]/div/div[1]/div[1]/div/div[2]/input').send_keys(name)
    driver.locate('//*[@id="common_template_mounted_el_container"]'
                  '/div/div[1]/div[3]/div/div[4]/div/div[1]/div[2]/div/div/input').send_keys(passwd)
    driver.click('//*[@id="common_template_mounted_el_container"]/div/div[1]/div[3]/div/div[4]/div/div[2]')

    # 然后自己手动操作验证码
    # 以及选择"店铺"


def 下载课次考勤数据(课程链接, 检查文本=''):
    # 1 遍历课程下载表格
    driver = get_driver()
    driver.get('https://admin.xiaoe-tech.com/t/data_center/index')  # 必须要找个过渡页，不然不会更新课程链接
    driver.get(课程链接)
    # 不能写'第{i}课'，会有叫'第{i}堂'等其他情况
    if 检查文本:  # 出现指定的文本才操作下一步
        driver.locate_text('//*[@id="app"]/div/div/div[1]/div[2]/div[1]/div[2]', 检查文本)
    else:  # 默认等待3秒
        time.sleep(3)

    driver.click('//*[@id="tab-studentTab"]/span')  # 直播间用户
    driver.click('//*[@id="pane-studentTab"]/div/div[2]/div[2]/form/div[2]/button[2]/span/span')  # 导出列表
    driver.click('//*[@id="data-export-container"]/div/div[2]/div/div[2]/div[2]/button[2]/span/span')  # 导出


def 聚合读取考勤数据(data_dir, name, judge_minute=30):
    """
    1、在目录"data_folder"下，找前缀包含name的所有文件
    2、并将其按日期排序，整理考勤数据，判断当堂、第一天完成等
    3、课次完成的标记，是judge_minute要满足特定分钟数
    """
    # 理论上获得的第一份就是当堂数据，第二份则是回放数据
    files = list(XlPath(data_dir).rglob_files(f'*{name}*'))
    data = 课次数据()
    for f in files:
        data.add_考勤数据(f)

    for user_id in data.用户观看数据:
        print(data.小鹅通考勤结果2(user_id, [100, 90, 80, 70, 0, 0]))
