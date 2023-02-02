#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2023/01/04 13:08

""" 修仙模拟器游戏脚本 """
import datetime
import os
import json
import random
import re
import time
from queue import PriorityQueue
from datetime import timedelta
import copy

import numpy as np
import win32com.client

from pyxllib.xl import XlPath, utc_now, utc_timestamp, xlwait, strfind, format_exception
from pyxllib.xlcv import xlcv, xlpil
from pyxllib.data.sqlite import Connection
from pyxllib.ext.autogui import pyautogui, list_windows, NamedLocate
from pyxlpr.ai.clientlib import XlAiClient


class 修仙模拟器助手:
    def __init__(self, token=''):
        # 1 定位软件窗口位置
        self.xywh = None
        self.im = None
        self.更新窗口位置()

        # 2 打开数据库、api、语音等辅助功能
        self.db = Connection('xxmnq.db')

        self.xlapi = XlAiClient()
        self.xlapi.login_priu(token, 'http://xmutpriu.com')

        self.speaker = win32com.client.Dispatch('SAPI.SpVoice')

        # 3 loclabel标签定位，获取账号名
        self.loc = NamedLocate('label', region=self.xywh)
        self.当前账号 = ''
        self.更新游戏画面()

        # 4 待办
        self.accounts = []
        self.cur_account_id = 0
        self.todolist = []  # 第1个值写时间戳，第2个值写账号编号，第3个值可以随便写个文本注释

    def __1_基础ocr功能(self):
        pass

    def 识别文本(self, arg):
        if isinstance(arg, str):
            arg = self.loc.rect2img(arg)
        text = self.xlapi.rec_singleline(arg)
        return text

    def 识别次数(self, arg):
        text = self.识别文本(arg)
        m = re.search(r'\d+', text) or 0
        if m:
            m = int(m.group())
        return m

    def 识别时间(self, arg):
        t = utc_now()
        text = self.识别文本(arg)
        if '次' in text:
            text = text[text.find('次')+1:]

        # 一般会有个需要等待的时间值在里面，计算等待时长。如果没有，则返回当前时间值
        _match = re.search(r'(\d{2}).(\d{2}).(\d{2})', text)
        if _match:  # 找到时间格式则计算延迟时间点
            nums = [int(x) for x in _match.groups()]
            nums[1] = min(nums[1], 59)
            nums[2] = min(nums[2], 59)
            dt = timedelta(hours=nums[0], minutes=nums[1], seconds=nums[2])
            return t + dt
        elif not text:  # 空值返回当前时间
            return t
        else:  # 否则返回-1异常
            return -1


    def __2_基础初始化功能(self):
        pass

    def 更新窗口位置(self):
        self.xywh = list_windows('AirDroid Cast')[-1][1]
        assert self.xywh[2] == 754
        assert self.xywh[3] == 972

    def 更新游戏画面(self):
        self.im = xlpil.to_cv2_image(pyautogui.screenshot(region=self.xywh))
        self.当前账号 = self.识别文本('主菜单/昵称').replace('仙义', '仙乄')

    def 记录账号状态(self):
        self.更新游戏画面()
        d = {k: self.识别文本(f'主菜单/{k}') for k in ['战力', '银币', '灵石', '通关']}
        self.write_log('状态', d)

    def 保存labelme标注(self, dst_im_file):
        self.更新游戏画面()

        dst = XlPath(dst_im_file)
        xlcv.write(self.im, dst)

        # data = self.xlapi.common_ocr(self.im)
        # data['imagePath'] = str(dst)
        # for sp in data['shapes']:
        #     sp['label'] = json.dumps(sp['label'], ensure_ascii=False)
        # data['shapes'] = []
        # XlPath(dst.with_suffix('.json')).write_json(data)

    def 语音播放(self, text):
        self.speaker.Speak(text)

    def __3_基础操作(self):
        pass

    def click(self, pos):
        """ 随机点击矩形中的任意位置 """
        self.loc.click(pos, random_bias=3)

    def _登录账户(self, 用户名, 密码):
        # 输入账号
        src_pause = pyautogui.PAUSE
        pyautogui.PAUSE = 0.2
        self.click('账号登录/用户名末尾')
        time.sleep(2)
        for i in range(20):
            pyautogui.hotkey('backspace')
            pyautogui.hotkey('delete')
        time.sleep(2)
        pyautogui.typewrite(用户名)
        time.sleep(2)

        # 输入密码
        self.click('账号登录/密码末尾')
        time.sleep(2)
        for i in range(20):
            pyautogui.hotkey('backspace')
            pyautogui.hotkey('delete')
        time.sleep(2)
        pyautogui.typewrite(密码)
        time.sleep(2)
        pyautogui.PAUSE = src_pause
        self.click('账号登录/登录')

        # 暂不支持自定义选区，只能使用默认区
        for i in range(15):
            if '区' in self.识别文本('账号登录2/点击选区'):
                time.sleep(2)
                break
        else:
            raise ValueError('登录失败')
        self.click('账号登录2/点击开始游戏')
        time.sleep(5)

    def 切换账号(self, 用户名, 密码):
        """ 在游戏中、在登录界面，都可以直接切换账号登录 """
        if self.识别文本('设置/设置'):
            self.click('设置/设置')
            self.click('设置/切换账号')
            self.click('设置2/确定离开游戏')
        time.sleep(3)
        self._登录账户(用户名, 密码)

    def 等待战斗结束(self):
        """ 战斗胜利返回1，失败返回0 """
        for i in range(180):  # 最多等3分钟
            t = self.识别文本('战盟秘境2/战斗胜利')
            if strfind(t, ['战斗', '通关']) != -1:
                time.sleep(3)
                if strfind(t, ['败', '失', '尖']) != -1:
                    return 0
                else:
                    return 1
            else:
                time.sleep(1)

    def add_todo(self, 时间, 说明):
        self.todolist.append([时间, self.cur_account_id, 说明])

    def 查看待办(self):
        self.todolist.sort()
        for i, x in enumerate(self.todolist):
            print(i, x[0].strftime('%Y-%m-%d %H:%M:%S'), x[1], x[2])

    def 打开攻略(self):
        self.click('主菜单/攻略')
        self.click('攻略/tab1')  # 老账号一般攻略按钮在这里
        self.click('攻略/tab2')  # 但新账号攻略一般在这里。因为老账号这里是空的没按钮，所以可以这样trick来处理。

    def 找子菜单位置(self, dst, start_tag=None):
        """ 找到位置后，返回其所在矩形区域位置 """
        def 寻找菜单项(方向, 参照值, 拖拽次数上限):
            for i in range(拖拽次数上限):
                im = self.loc.rect2img('赏金1/子菜单')
                x0, y0, _, _ = self.loc['赏金1/子菜单']['ltrb']

                shapes = self.xlapi.common_ocr(im)['shapes']
                for sp in shapes:
                    if dst in sp['label']['text']:
                        l, t, r, b = np.array(sp['points'], dtype=int).reshape(-1).tolist()
                        return dst, [l + x0, t + y0, r + x0, b + y0]
                    if 参照值 in sp['label']['text']:
                        l, t, r, b = np.array(sp['points'], dtype=int).reshape(-1).tolist()
                        return 参照值, [l + x0, t + y0, r + x0, b + y0]

                self.loc.drag_to('赏金1/子菜单', 方向)
                time.sleep(2)

        if start_tag:
            r = 寻找菜单项(1, start_tag, 10)
            if r[0] == dst:
                return r[1]

        r = 寻找菜单项(3, dst, 10)
        if r:
            return r[1]

    def __4_游戏自动化(self):
        pass

    def 领取离线奖励(self):
        if strfind(self.识别文本('主菜单/离线奖励'), ['离线', '奖励']) != -1:
            self.click('主菜单/离线奖励')
            self.write_log('离线奖励', self.识别文本('离线奖励/奖励内容'))
            self.click('离线奖励/领取')

    def 自动九幽入侵(self):
        self.click('主菜单/历练')
        self.click('九幽入侵/九幽入侵')

        # 1 剩余未打的九幽
        ls = PriorityQueue()
        for i in range(1, 6):
            t2 = self.识别时间(f'九幽入侵/九幽{i}')
            if t2 == -1:  # 遇到异常，即有了打了一半没打完的，不做处理和记录
                continue
            if t2 <= utc_now():
                if self.识别次数('九幽入侵/剩余次数') > 0:
                    self.click(f'九幽入侵/诛讨{i}')
                    self.等待战斗结束()
                    time.sleep(2)
            t2 = self.识别时间(f'九幽入侵/九幽{i}')
            if isinstance(t2, datetime.datetime):
                ls.put([t2, i])

        # 2 剩余次数
        m = self.识别次数('九幽入侵/剩余次数')

        # 3 后续规划
        if m:  # 还有次数，那么则是九幽cd还没到
            q = ls.get()  # 获得最近一次九幽刷新的时间
            self.add_todo(q[0], f'第{q[1]}个九幽已刷新')
        else:
            t3 = self.识别时间('九幽入侵/剩余次数')
            self.add_todo(t3, f'九幽入侵次数已刷新')

    def 自动九幽主宰(self):
        self.打开攻略()
        if '玩法' in self.识别文本('攻略/九幽主宰'):  # 玩法未开启
            self.click('攻略/关闭')
            return

        t = utc_now()
        t1 = datetime.datetime(t.year, t.month, t.day, 11, 45)
        t2 = datetime.datetime(t.year, t.month, t.day, 12, 15)
        t3 = datetime.datetime(t.year, t.month, t.day, 17, 45)
        t4 = datetime.datetime(t.year, t.month, t.day, 18, 15)

        def 打主宰():
            self.打开攻略()
            self.click('攻略/前往九幽主宰')
            while self.识别次数('九幽主宰/剩余次数'):
                self.click('九幽主宰/诛讨')
                self.等待战斗结束()
                time.sleep(3)

        if t < t1:
            self.add_todo(t1, '准备打中午的主宰')
        elif t < t2:
            打主宰()
            self.add_todo(t3, '准备打晚上的主宰')
        elif t < t3:
            self.add_todo(t3, '准备打晚上的主宰')
        elif t < t4:
            打主宰()
            self.add_todo(t1 + timedelta(days=1), '准备明天中午的主宰')
        else:
            self.add_todo(t1 + timedelta(days=1), '准备明天中午的主宰')

        self.click('攻略/关闭')

    def 自动刷新黑市(self):
        # 1 找到菜单
        self.click('主菜单/市场')
        rect = self.找子菜单位置('黑市', '银币')
        # 找不到黑市，可能小号还没开通这种功能呢
        if not rect:
            return

        l, t, r, b = rect
        self.click([(l + r) // 2, (t + b) // 2])

        # 2 购买资源
        def check(s):
            return any(['币】' in s,
                        '招募' in s and '150' in s,
                        '赏金' in s,
                        '三品' in s,
                        '夺宝' in s and '3折' in s,
                        ])

        def 购买页面内容(use_log=True):
            ls = []
            for i in range(1, 7):
                text = self.识别文本(f'黑市商店/商品{i}')
                ls.append(text)

                if check(text) and '日限】1/1' not in text:
                    self.click(f'黑市商店/商品{i}')
                    time.sleep(1)
                    self.click(f'黑市商店2/购买数量')
                    self.click(f'黑市商店2/确认')

            if use_log:
                self.write_log('黑市', ls)

        购买页面内容(False)
        n = self.识别次数('黑市商店/刷新次数')
        for i in range(n):
            self.click('黑市商店/刷新')
            购买页面内容()

        # 3 记录下次刷新时间
        self.add_todo(self.识别时间('黑市商店/刷新时间'), '黑市刷新')

    def 自动购买秘籍(self):
        # 1 找到菜单
        self.click('主菜单/市场')
        rect = self.找子菜单位置('秘籍')
        if not rect:
            return

        l, t, r, b = rect
        self.click([(l + r) // 2, (t + b) // 2])

        # 2 购买资源
        def check(s):
            return any([s.lstrip()[:2] != '银币' and '银币' in s,
                        '招募' in s and '150' in s,
                        '赏金' in s,
                        '三品' in s,
                        '夺宝' in s and '3折' in s,
                        ])

        def 购买页面内容(use_log=True):
            ls = []
            for i in range(1, 7):
                text = self.识别文本(f'黑市商店/商品{i}')
                ls.append(text)

                if check(text):
                    self.click(f'黑市商店/商品{i}')
                    self.click(f'黑市商店2/购买数量')
                    self.click(f'黑市商店2/确认')

            if use_log:
                self.write_log('秘籍', ls)

        购买页面内容(False)
        n = self.识别次数('黑市商店/刷新次数')
        for i in range(n):
            self.click('黑市商店/刷新')
            购买页面内容()

        # 3 记录下次刷新时间
        self.add_todo(self.识别时间('黑市商店/刷新时间'), '黑市刷新')

    def 记录天材地宝(self):
        self.更新游戏画面()
        ls = []
        for i in range(1, 5):
            名称 = self.识别文本(f'天材地宝/任务{i}')
            if 名称[0] == '品':
                名称 = '一' + 名称
            ls.append(名称)
            挑战 = self.识别文本(f'天材地宝/挑战{i}')
            print(i + 1, 名称, 挑战)
        self.write_log('天材地宝', ls)
        return ls

    def 读取前3个赏金(self):
        self.更新游戏画面()
        ls = []
        for i in range(1, 4):
            名称 = self.识别文本(f'赏金1/任务{i}')
            print(i + 1, 名称)
            ls.append(名称)
        return ls

    def 读取后3个赏金(self):
        self.更新游戏画面()
        ls = []
        for i in range(4, 7):
            名称 = self.识别文本(f'赏金2/任务{i}')
            print(i + 1, 名称)
            ls.append(名称)
        return ls

    def 自动刷新赏金(self):
        [(595, 319), (652, 378)]

    def write_log(self, title, data):
        """ 往日志写入信息 """
        self.db.insert_row('log', {'account': self.当前账号, 'update_time': utc_timestamp(8),
                                   'title': title, 'text': data})
        self.db.commit()

    def 自动战盟建设(self):
        self.click('主菜单/战盟')
        self.click('战盟/建设')
        m = self.识别次数('战盟建设/可捐次数')
        for i in range(m):
            self.click('战盟建设/初级建设')
            self.click('战盟建设/确认')

    def 自动战盟秘境(self):
        self.click('主菜单/战盟')
        self.click('战盟/秘境')
        while True:
            m = self.识别次数('战盟秘境1/挑战次数')  # 不用管战斗过程，只要挑战次数变0就行
            if m == 0:
                break
            else:
                self.click('战盟秘境1/挑战')
                time.sleep(2)

    def 自动通关(self):
        while True:
            t = self.识别文本('主菜单/通关')
            self.click('主菜单/挑战')
            r = self.等待战斗结束()
            time.sleep(4)
            print('挑战', t, '成功' if r else '失败')

    def 自动修仙历练(self):
        pass

    def __5_高级接口(self):
        pass

    def 自动托管(self):
        pass

        self.记录账号状态()

        self.领取离线奖励()
        self.自动九幽入侵()
        self.自动九幽主宰()

        # 自动升级核心

        self.自动刷新黑市()
        # self.自动购买秘籍()

        # 自动天材地宝
        # 自动势力争夺
        # 自动游历四方
        # 自动修仙历练

        # 自动赏金任务

        # 自动突破秘籍
        # 自动吃元气丹

        # 自动修真大会
        # 自动全服争霸

        # 自动竞技排行
        # 自动远征
        # 自动天梯争霸

        # 自动战盟
        # 自动通关
        # 自动副本
        # 自动天神塔

        # 自动霸业
        # 自动寻宝
        # 自动荣耀商店
        # 自动功勋商店

        # 切换账号
        # 自动兑换激活码

        self.add_todo(utc_now() + timedelta(hours=1), '没事也每隔1小时看一次')

    def 多账号托管(self, accounts):
        """ 需要有try机制解决登录冲突等异常情况
        并且要有强手段重新恢复另一个账号的登录
        然后至于出现异常的账号，可以一个小时候再去复查

        todo 加了-1标记，一些特殊性的提示。比如快20点要寻宝等，给语音提示。
        """

        # 1 先遍历标记所有账号
        self.accounts = accounts
        for i in range(len(self.accounts)):
            self.cur_account_id = i
            self.add_todo(utc_now(), '初始化首次运行')

        # 2 开始不断循环等待最近一次事件后，触发托管
        while True:
            # 等待最近一次事件
            self.todolist.sort()
            self.查看待办()

            # 等待最近事件触发时间
            _time, idx, comment = self.todolist[0]
            dt = _time - utc_now()
            if dt.total_seconds() > 10:
                time.sleep(dt.total_seconds() - 10)

            print('【触发托管】', utc_now(), idx, self.accounts[idx][0], comment)
            self.todolist = [x for x in self.todolist if x[1] != idx]  # 清除跟这个账号有关的所有任务记录，后续托管后重新重置
            self.cur_account_id = idx

            try:
                self.切换账号(self.accounts[idx][0], self.accounts[idx][1])
                self.自动托管()
            except Exception as e:
                print(f'账号{idx} 哎呀！程序炸了！')
                print(format_exception(e))
                # 可能会因为各种异常炸了，就重置监控
                self.add_todo(utc_now() + timedelta(hours=1), '程序炸了也1小时后来再看看')
                pass


if __name__ == '__main__':
    pass
