import datetime
import os
import time
from types import SimpleNamespace

import openpyxl
import pandas as pd

from pyxllib.xl import *
from pyxllib.ext.kq5034lib import *

classdata = """
第1课
""".strip()


class 梵呗暑假2023年08月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '8月梵呗'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DY3dVT0lHaXVLTEpU'  # 生成日报用
        self.开课日期 = '2023-08-07'
        self.视频返款 = [20, 16, 12, 8, 4, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4天/第5天，完成观看的依次返款额。
        self.打卡返款 = {1: 30, 4: 60, 7: 100}  # 打卡满1/4/7次的返款额

        self.课程链接 = ['',
                     'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64ad3f99e4b0d1e42e815d23',
                     'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64ad3f9ce4b0d1e42e815d25',
                     'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64ad3f9ee4b03e4b54d97e48',
                     'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64ad3fa0e4b03e4b54d97e4a',
                     'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64ad3fa2e4b03e4b54d97e4c',
                     'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64ad3fa4e4b0d1e42e815d27',
                     'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64ad3fa6e4b03e4b54d97e4e',
                     'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64ad3fa8e4b0b0bc2bfeb3bc',
                     'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64ad3faae4b0d1e42e815d29',
                     'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64ad3fabe4b0b0bc2bfeb3be']

        self._init(today)
        self.driver = None

        # 注意：最后计算打卡数据，需要补下载一个来自旁听教室的数据

    def 异常处理(self):
        # self.修订(1, 1, '完成当堂学习')
        pass

    def 剩余统计(self):
        n, a, x = 3, 500, 1364  # 初始报名人数, 每人报名金额, 总共返款额
        print(f'报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')


def rename():
    d = XlPath(r'数据表')
    for f in d.glob_files('*'):
        f.rename2(f.parent / f.name.replace('2023-08-17', '2023-08-16'))


if __name__ == '__main__':
    os.chdir(r'C:\home\chenkunze\data\5034考勤\2023年08月梵呗')

    # 1 下载考勤表
    classdata = pd.read_excel('考勤.xlsx', '课次数据')
    # 登录小鹅通('18850340559', 'Y3QpAVv2ZuPHnbz')
    # input()  # 这里要暂停，人工处理后再下一步
    # for idx, row in classdata.iterrows():
    #     m = re.search(r'\d+', row['课次'])
    #     if int(m.group()) <= 35:
    #         continue
    #     下载课次考勤数据(row['链接'], row['名称'])
    #     time.sleep(3)

    # 2 更新考勤数据
    wb = openpyxl.load_workbook('考勤.xlsx')
    ws = wb['考勤表']
    视频返款 = [20, 16, 12, 8, 4, 0]
    回放返款延迟天数 = sum(map(bool, 视频返款)) - 1

    cur_user2money = defaultdict(int)  # 当天应返款
    user2money = defaultdict(int)  # 总共应返款

    today = datetime.datetime.now().today().date()
    # today = datetime.datetime.now().today().date()
    for idx, clsdata in classdata.iterrows():
        data = 课次数据()
        data.add_files('数据表', f'*{clsdata["名称"]}*')

        if clsdata['课次'] == '第15课':
            要求在线分钟 = 0
        else:
            要求在线分钟 = 15

        for i in ws.iterrows('用户ID'):
            user_id = ws.cell2(i, '用户ID').value
            text, color, money = data.小鹅通考勤结果2(user_id, 视频返款, 要求在线分钟)

            # 3 TODO 异常修正

            ws.cell2(i, clsdata['课次']).set_rich_value(text, color)

            回放返款日期 = (clsdata['开始日期'] + datetime.timedelta(days=回放返款延迟天数)).date()
            if '当堂' in text:
                user2money[user_id] += money
                if today == clsdata['开始日期'].date():
                    cur_user2money[user_id] += money
            elif '回放' in text and 回放返款日期 <= today:
                user2money[user_id] += money
                if today == 回放返款日期:
                    cur_user2money[user_id] += money

    # 4 生成返款文件
    # 暂时只要考虑当堂完成数量就行
    ls = []
    for i in ws.iterrows('用户ID'):
        user_id = ws.cell2(i, '用户ID').value
        ws.cell2(i, ['已返款', '总计']).value = user2money.get(user_id, 0)
        订单号 = ws.cell2(i, '交易订单号').value
        if not 订单号 or '无' in 订单号:
            continue
        今日返款额 = cur_user2money.get(user_id, 0)
        if 今日返款额:
            ls.append(f'{订单号},{今日返款额},暑期梵呗打卡促学金,{订单号}_class11')
    XlPath('第11天 打卡.csv').write_text('\n'.join(ls))

    wb.save('考勤+.xlsx')

    # 聚合读取考勤数据(r'C:\home\chenkunze\data\5034考勤\2023年08月觉观\数据表', '第1堂')
    # 聚合读取考勤数据(r'C:\home\chenkunze\data\5034考勤\2023年08月觉观\数据表', '第1堂')

    # fire.Fire(梵呗03届2023年07月)

    # m = 梵呗暑假2023年08月()

    # raw字符串

    # os.startfile(m.表格路径)
    # browser(m.在线表格)

    # m.表格内容对齐('报名表', '账单', '交易单号')
    # m.表格内容对齐('分组表', 'Sheet1', '微信昵称')

    # m.匹配用户ID()

    # m.登录小鹅通('18850340559', 'Y3QpAVv2ZuPHnbz')
    # m.下载课次考勤数据()
    # m.考勤日报()

    # m.登录微信支付()
    # m.批量退款()
    # m.申请单条退款('RS0NYA-0OZRE8O-E6NU', '375', '第18届觉观第1~3,5~6课完成当堂学习')

    # m.剩余统计()

    # data = 课次数据()
    # data.add_files(r'C:\home\chenkunze\data\5034\考勤\2023年08月梵呗\s', '*1.真声体验*')

    pass
