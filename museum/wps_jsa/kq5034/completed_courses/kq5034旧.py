from collections import Counter
import os
import re

import fire
import pandas as pd

from pyxllib.xl import XlPath
from pyxllib.file.xlsxlib import openpyxl, XlWorksheet

from xlproject.kq5034lib import 网课考勤, TicToc
from xlproject.kq5034lib import KqDb


class 觉观10届2021年12月(网课考勤):
    def __init__(self, today=None):
        """
        报名166人*1046元=173636元。剩余19375元。
        """
        self.表格路径 = r'D:\localdata\5034\2021年12月觉观'
        self.在线表格 = 'https://docs.qq.com/sheet/DUk5uUWRORlRaUktV?tab=v7m83a'  # 生成日报用
        self.开课日期 = '2021-12-01'  # 通过开课时间，会自动判断出是念住课还是觉观禅课
        self.视频返款 = [43, 33, 22, 11, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4天/第5天，完成观看的依次返款额。
        self.打卡返款 = [30, 60, 100]  # 打卡满5/10/15次的返款额
        # 22课×42元+打卡100元=1046元
        self._init(today)


class 念住05届2022年01月(网课考勤):
    def __init__(self, today=None):
        """
        报名185人*620元=114700元，剩余26685元。
        """
        self.表格路径 = r'D:\localdata\5034\2022年01月念住\第05届念住考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUlF1UnRackJ2Vm5U'  # 生成日报用
        self.开课日期 = '2022-01-08'  # 通过开课时间，会自动判断出是念住课还是觉观禅课
        self.视频返款 = [20, 15, 10, 5, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4天/第5天，完成观看的依次返款额。
        self.打卡返款 = [100, 150, 200]  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元
        self._init(today)


class 觉观11届2022年02月(网课考勤):
    def __init__(self, today=None):
        """
        发潘宏铭老师：2022年2月觉观禅网课，报名125人*1134元=141750元，剩余24690元。
        """
        self.表格路径 = r'D:\home\chenkunze\data\5034\2022年02月觉观\考勤\第11届觉观考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUklHS3B4UUtudk9Q'  # 生成日报用
        self.开课日期 = '2022-02-01'  # 通过开课时间，会自动判断出是念住课还是觉观禅课
        self.视频返款 = [47, 36, 24, 12, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4天/第5天，完成观看的依次返款额。
        self.打卡返款 = [30, 60, 100]  # 打卡满5/10/15次的返款额
        # 22课×47元+打卡100元=1134元
        self._init(today)

    def 异常处理(self):
        for k in range(1, 9):
            self.修订(5, k, '完成当堂学习')
        self.修订(122, 13, '完成当堂学习')


class 念住06届2022年03月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR

        报名：https://m.dashengshan.cn/h-nd-510.html?_sc=1&checkWxLogin=true
        这次回放变7天了

        发潘宏铭老师：2022年3月念住网课，报名179人*620元=110980元，剩余25950元。
        """
        self.表格路径 = r'C:\home\chenkunze\data\5034\考勤\2022年03月\考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUmdyU055ZERkTWla'  # 生成日报用
        self.开课日期 = '2022-03-08'  # 通过开课时间，会自动判断出是念住课还是觉观禅课
        self.视频返款 = [20, 15, 10, 5, 0, 0, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = [100, 150, 200]  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元
        self._init(today)

        self.links = [
            # 1~7
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6223f659e4b066e9608c105f',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6223f7c5e4b0beaee430d046',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6223f872e4b066e9608c1096',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6223f98de4b066e9608c10b3',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6223fa30e4b04d7e2fd2a936',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6223fbffe4b04d7e2fd2a952',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6223fc87e4b04d7e2fd2a95c',
            # 8~14
            'https://admin.xiaoe-tech.com/live#/detail?id=l_62260262e4b04d7e2fd33a68',  # 3月15日
            'https://admin.xiaoe-tech.com/live#/detail?id=l_62240060e4b054255da4bc87',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_622400f8e4b066e9608c1159',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_62240199e4b066e9608c1163',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6224022fe4b054255da4bca8',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_622402a8e4b066e9608c1171',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6224033ee4b066e9608c1180',
            # 15~21
            'https://admin.xiaoe-tech.com/live#/detail?id=l_622403b7e4b04d7e2fd2a9e0',  # 3月22日
            'https://admin.xiaoe-tech.com/live#/detail?id=l_62240bd3e4b066e9608c12bb',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_62240cafe4b02b82585182b0',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_62240d1ee4b04d7e2fd2ab43',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_62240d81e4b054255da4be47',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_62240e2ae4b02b82585182df',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_62240ea3e4b066e9608c133d',
        ]

    def 异常处理(self):
        self.修订(11, 1, '完成当堂学习')
        self.修订(11, 1, '完成当堂学习')

        self.修订(107, 7, '完成当堂学习')
        self.修订(41, 8, '完成当堂学习')  # 220315周二10:22


class 觉观12届2022年04月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR

        报名：觉观禅技术公开课——第12届4月1日开课:
        https://m.dashengshan.cn/h-nd-515.html?checkWxLogin=true&openId=m0CqvOlfkFp%2Fbz3EteR7sWMOmnq0XYMS2gTQcy7Qmqo%3D&secondAuth=true

        发潘宏铭老师：2022年4月觉观网课，报名164人*1222元=200408元，剩余24115元。（有一位学员是潘老师3月30日退款的操作我已经算进来了）
        """
        self.返款标题 = '觉观禅第12届'
        self.表格路径 = r'C:\home\chenkunze\data\5034\考勤\2022年04月\考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUlNwWnF1QmZGdVBS'  # 生成日报用
        self.开课日期 = '2022-04-01'  # 通过开课时间，会自动判断出是念住课还是觉观禅课
        self.视频返款 = [51, 39, 26, 13, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~5天，完成观看的依次返款额。
        self.打卡返款 = [30, 60, 100]  # 打卡满5/10/15次的返款额
        self._init(today)

        self.links = [
            # 1~7
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a798e4b02b8258562328',  # 4月1日~4月6日
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a79de4b0beaee43565bc',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7a1e4b04d7e2fd7480e',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7a4e4b054255da95901',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7a8e4b066e96090a4b8',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7abe4b0beaee43565c3',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7afe4b066e96090a4bc',
            # 8~14
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7b2e4b066e96090a4c0',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7b6e4b02b825856233c',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7b9e4b02b8258562342',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7bde4b0beaee43565e5',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7c0e4b066e96090a4cf',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7c4e4b04d7e2fd74822',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7c8e4b0beaee43565ee',
            # 15~21
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7cbe4b0beaee43565f1',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7cfe4b066e96090a4f3',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7d2e4b054255da9592a',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7d6e4b066e96090a4f9',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7dae4b02b8258562353',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7dde4b054255da95932',
            'https://admin.xiaoe-tech.com/live#/detail?id=l_6232a7e1e4b054255da9593a',
        ]

    def 异常处理(self):
        self.修订(102, 1, '完成当堂学习')
        self.修订(107, 2, '完成当堂学习')
        self.修订(152, 2, '完成当堂学习')
        self.修订(157, 4, '完成当堂学习')

        # 她报名了但没人理她，错过了前4课，我给她补退促学金了
        for j in range(1, 5):
            self.修订(47, j, '完成当堂学习')

        self.修订(64, 6, '完成当堂学习')
        self.修订(127, 6, '完成当堂学习')
        self.修订(23, 8, '完成当堂学习')

        # 第9课因为后台师兄存的视频错了，全员返款
        for i in range(1, 165):
            try:
                self.修订(i, 9, '完成当堂学习')
            except:
                pass

        self.修订(23, 10, '完成当堂学习')
        self.修订(23, 11, '完成当堂学习')
        self.修订(122, 14, '完成当堂学习')
        self.修订(90, 14, '完成当堂学习')
        # self.修订(23, 16, '完成当堂学习')
        self.修订(23, 17, '完成当堂学习')

        self.修订(100, 6, '第1天回放')

        self.修订(66, 5, '第1天回放')


def _2022五一身心行修线上觉观营():
    """ 2022年五一相关测试，想尝试把数据写入数据库 """

    os.chdir(r'C:\home\chenkunze\data\5034\考勤\2022年05月心身行')

    db = KqDb()
    db.update_小鹅通数据()

    # 2 用户ID匹配
    # db.update_用户列表('小鹅通下载表/用户列表导出20220430094404214347.csv')
    # db.匹配用户ID('考勤.xlsx')

    # 3 考勤数据
    db.update_wb()


class 念住07届2022年05月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR
        """
        self.返款标题 = '第07届念住'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUmlRRlprZGRWSk9X'  # 生成日报用
        self.开课日期 = '2022-05-08'  # 通过开课时间，会自动判断出是念住课还是觉观禅课
        self.视频返款 = [20, 15, 10, 5, 0, 0, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = [100, 150, 200]  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元
        self._init(today)

    def 异常处理(self):
        self.修订(88, 1, '完成当堂学习')
        self.修订(87, 2, '完成当堂学习')
        self.修订(226, 1, '完成当堂学习')
        self.修订(224, 3, '完成当堂学习')
        self.修订(37, 3, '完成当堂学习')
        self.修订(172, 3, '第1天回放')

        # 185号提前退还了所有促学金，目前表格里保留考勤记录，最后计算余额的时候注意下就行

        self.修订(87, 4, '完成当堂学习')
        self.修订(87, 5, '完成当堂学习')

        self.修订(123, 2, '完成当堂学习')

        self.修订(224, 6, '完成当堂学习')

        self.修订(160, 8, '完成当堂学习')
        self.修订(201, 2, '完成当堂学习')
        self.修订(201, 3, '完成当堂学习')

        self.修订(78, 4, '完成当堂学习')
        self.修订(78, 8, '完成当堂学习')
        self.修订(78, 9, '完成当堂学习')
        self.修订(201, 10, '完成当堂学习')
        self.修订(201, 11, '完成当堂学习')
        self.修订(201, 12, '完成当堂学习')
        self.修订(78, 10, '完成当堂学习')
        self.修订(78, 11, '完成当堂学习')
        self.修订(223, 18, '完成当堂学习')
        self.修订(223, 19, '完成当堂学习')
        self.修订(223, 20, '完成当堂学习')
        self.修订(223, 2, '完成当堂学习')

    def 剩余统计(self):
        n, a, x = 229, 620, 109640  # 初始报名人数, 每人报名金额, 剩余报名人数里总共返款额
        print(f'2022年5月念住网课，报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')
        # 发潘宏铭老师：2022年5月念住网课，报名229人*620元=141980元，剩余32340元。


class 觉观13届2022年06月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR

        注意！注意！
        36号因为时区（utc-5）问题，在北京时间前13个小时，早课5:20，对方还在昨天16:20。最后要补足差额。
        238号捐功德箱是潘老师补的要给差额。
        """
        self.返款标题 = '第13届觉观'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUmVTb292T0ZIclNN'  # 生成日报用
        self.开课日期 = '2022-06-01'  # 通过开课时间，会自动判断出是念住课还是觉观禅课
        self.视频返款 = [55, 42, 28, 14, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~5天，完成观看的依次返款额。
        self.打卡返款 = [30, 60, 100]  # 打卡满5/10/15次的返款额
        # 22课×55元+打卡100元=1310元
        self._init(today)

    def 异常处理(self):
        """
        32号退课
        198号退课

        238号是额外加入的
        """
        # 第1天
        self.修订(196, 1, '完成当堂学习')

        # 第2天（周四）早上
        self.修订(166, 2, '完成当堂学习')

        # 晚上：36,66,83,106,195
        # self.修订(36, 2, '完成当堂学习')  # 沟通中，需要确认是不是回放完成
        # 66号，反馈切换设备，第2课有看完，但查询实际只有29分钟
        # 83号，听海，改user_id就行
        self.修订(106, 1, '完成当堂学习')
        self.修订(106, 2, '完成当堂学习')
        # 195号，中到大雨转小雨，反馈第2课有看
        self.修订(195, 2, '完成当堂学习')

        # 12号换账号，第1课补迁过来
        self.修订(12, 1, '第1天回放')

        self.修订(108, 2, '完成当堂学习')

        self.修订(88, 4, '完成当堂学习')

        self.修订(65, 4, '完成当堂学习')

        self.修订(68, 5, '完成当堂学习')
        self.修订(12, 5, '完成当堂学习')

        self.修订(133, 7, '完成当堂学习')
        self.修订(12, 7, '完成当堂学习')
        self.修订(171, 4, '完成当堂学习')
        self.修订(171, 7, '完成当堂学习')

        self.修订(68, 8, '完成当堂学习')

        self.修订(12, 9, '完成当堂学习')
        self.修订(57, 6, "第1天回放")

        self.修订(68, 11, '完成当堂学习')
        self.修订(133, 11, '完成当堂学习')

        self.修订(68, 17, '完成当堂学习')
        self.修订(5, 16, '完成当堂学习')

        self.修订(30, 14, '完成当堂学习')
        self.修订(119, 19, '完成当堂学习')

    def 剩余统计(self):
        n, a, x = 237, 1310, 263952  # 初始报名人数, 每人报名金额, 剩余报名人数里总共返款额
        print(f'2022年6月觉观网课，报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')
        # 发潘宏铭老师：2022年6月觉观网课，报名237人*1310元=310470元，剩余46518元。


class 念住08届2022年07月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR
        """
        self.返款标题 = '第8届念住'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUmplQ2xzeGF1RmVG'  # 生成日报用
        self.开课日期 = '2022-07-08'  # 通过开课时间，会自动判断出是念住课还是觉观禅课
        self.视频返款 = [20, 15, 10, 5, 0, 0, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = [100, 150, 200]  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元
        self._init(today)

    def 异常处理(self):
        """
        31号，Rita Zhang，第1天就说退课了
            M202206257296,620,第8届念住退课,M202206257296_plus
        44号，占毅楠，已退65元，第10~21课+打卡，现要补退剩余课程420元。然后从考勤表中移除。但是张枬师兄沟通改为课程结束适当返款。
            M202206267315,160,第8届念住剩余16课返一半促学金,M202206267315_plus
        93号，温温蘅，时差，最后要补足返款（第1~10课第1天回放->当堂，第11课第2天回放->第1天回放，共55元）
            M202207067534,55,第8届念住回放共11课时差补返款,M202207077595_plus
        """
        # self.修订(0, 0, '完成当堂学习')
        self.修订(72, 1, '完成当堂学习')
        self.修订(122, 1, '完成当堂学习')

        self.修订(72, 1, '完成当堂学习')
        self.修订(10, 2, '完成当堂学习')

    def 剩余统计(self):
        n, a, x = 128, 620, 56770  # 初始报名人数, 每人报名金额, 剩余报名人数里总共返款额
        print(f'2022年7月念住网课，报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')
        # 发潘宏铭老师：2022年7月念住网课，报名128人*620元=79360元，剩余22590元。


class 梵呗02届2022年07月:
    def __init__(self):
        self.有效时长 = 40
        self.视频返款 = [40, 30, 20, 20, 0, 0, 0, 0, 0]

    def get观看记录1(self):
        """ 基础考勤数据解析 """

        def parse_timespan(v):
            # 单位是分钟，向下取整
            v = v.strip()
            if v in ('未参与', '无'):
                return 0
            elif m := re.match(r'(\d{2})时(\d{2})分(\d{2})秒', v):
                hours, minutes, seconds = map(int, m.groups())
                return hours * 60 + minutes
            else:
                raise ValueError

        ls = []
        for f in XlPath('下载表').glob_files('第*课*.xls'):
            print(f.name)
            if f.name[0] == '~':
                continue
            names_cnt = Counter()  # 同一份表格里，相同姓名另外编号
            title = f.stem.split(' ')[0]
            df = pd.read_excel(f, skiprows=14)
            for idx, x in df.iterrows():
                if x['姓名'] == x['群昵称']:
                    name = x['姓名']
                else:
                    name = x['姓名'] + ',' + x['群昵称']

                # 昵称改动
                if name == '15 胡宗尚':
                    name = '15 胡宗尚,红梅'
                elif name == '陈赛男':
                    name = '陈秋菊'

                if name in names_cnt:
                    # name += str(names_cnt[name] + 1)
                    # print(f'重名：{name}')
                    pass  # 这次理论上没有重名的，进行合并
                names_cnt[name] += 1
                item = [title, name, parse_timespan(x['观看直播时长']), parse_timespan(x['观看总时长'])]

                # 第1课回放计算放宽要求
                if '第01课' in f.name or '第14课' in f.name:
                    if item[2]:
                        item[2] += 15
                    if item[3]:
                        item[3] += 15

                ls.append(item)

        df = pd.DataFrame.from_records(ls, columns=['课次', '钉钉姓名,群昵称', '直播分钟', '总观分钟'])
        ls = []
        for keys, values in df.groupby(['课次', '钉钉姓名,群昵称']):
            # print(keys, sum(values['直播分钟']), sum(values['总观分钟']))
            ls.append([keys[0], keys[1], sum(values['直播分钟']), sum(values['总观分钟'])])

        df = pd.DataFrame.from_records(ls, columns=['课次', '钉钉姓名,群昵称', '直播分钟', '总观分钟'])
        df.sort_values(['钉钉姓名,群昵称', '课次'], inplace=True)

        return df

    def get观看记录2(self):
        """ 按课次汇总情况 """
        df = self.get观看记录1()
        start_class_num, end_class_num = int(min(df['课次'])[1:3]), int(max(df['课次'])[1:3])  # 已有课次数
        ls = []
        columns = ['钉钉姓名,群昵称'] + [f'第{i:02}课' for i in range(start_class_num, end_class_num + 1)] + [
            '总应返款']
        for k, vs in df.groupby('钉钉姓名,群昵称'):
            # 每个人
            item = [k]
            money = 0
            for i in range(start_class_num, end_class_num + 1):
                # 每个课次
                title, status = f'第{i:02}课', '未开始学习'
                for _, v in vs.iterrows():
                    if not v['课次'].startswith(title):
                        continue
                    time_flag = max(int(v['课次'][4]), 1)
                    if v['直播分钟'] >= self.有效时长:
                        status = '完成当堂学习'
                        money += self.视频返款[0]
                        break
                    elif v['总观分钟'] >= self.有效时长:
                        status = f'第{time_flag}天回放'
                        money += self.视频返款[time_flag]
                        break
                    elif v['总观分钟'] and v['总观分钟'] < self.有效时长:
                        status = '不足40分钟'
                        break
                item.append(status)
            ls.append(item + [money])

        df = pd.DataFrame.from_records(ls, columns=columns)

        for idx, row in df.iterrows():
            if '吴婷' in row['钉钉姓名,群昵称']:
                df.loc[idx, '第03课'] = '完成当堂学习'
            elif '陈赛男' in row['钉钉姓名,群昵称']:
                df.loc[idx, '第03课'] = '完成当堂学习'

        return df

    def get打卡记录(self):
        """ 未完工 """
        for f in XlPath('下载表').glob_files('*作业*.xlsx'):
            if f.name[0] == '~':
                continue

            df = pd.read_excel(f, skiprows=2)

            for _, x in df.iterrows():
                pass

            print(df['学员姓名'])

            print(f)

            break

    def write观看记录2xlsx(self):

        def check_name2(微信关联昵称, 钉钉姓名群昵称):
            if 微信关联昵称 and 钉钉姓名群昵称:
                return 微信关联昵称.split(',')[0] == 钉钉姓名群昵称.split(',')[-1]

        def check_name3(姓名, 微信关联昵称, 钉钉姓名群昵称):
            if 微信关联昵称 and 钉钉姓名群昵称:
                srcs = 微信关联昵称.split(',')
                dsts = 钉钉姓名群昵称.split(',')

                if 姓名 in dsts:
                    return True
                return bool(set(srcs) & set(dsts))

        def write_item(i, idx, row):
            from pyxllib.cv.rgbfmt import RgbFormatter
            used_idxs.add(idx)
            ws.cell2(i, '钉钉姓名,群昵称').value = row['钉钉姓名,群昵称']
            ws.cell2(i, ['已返款', '总计']).value = row['总应返款']

            for j in range(1, 30):
                title = f'第{j:02}课'
                if title in row:
                    cel = ws.cell2(i, title)
                    value = cel.value = row[title]
                    color = None
                    if '完成当堂学习' in value:
                        color = RgbFormatter.from_name('鲜绿色')
                    elif '回放' in value:
                        color = RgbFormatter.from_name('黄色')
                        v1 = self.视频返款[0]
                        v2 = self.视频返款[int(re.search(r'第(\d+)天', value).group(1))]
                        if v2:
                            color = color.light((v1 - v2) / v2)  # 根据返款额度自动变浅
                        else:  # 如果无返款额度
                            color = RgbFormatter.from_name('灰色')
                    elif title <= '第00课':  # 这个手动设置，截止课次
                        cel.value = '未完成学习'
                        color = RgbFormatter.from_name('红色')

                    if color:
                        cel.fill_color(color)

        df = self.get观看记录2()
        wb = openpyxl.load_workbook('考勤.xlsx')
        ws: XlWorksheet = wb['考勤表']
        used_idxs = set()
        for i, x in ws.iterrows('姓名', to_dict=['姓名', '微信,关联昵称']):
            flag = False
            # 1 先找姓名完全匹配
            for idx, row in df.iterrows():
                if x['姓名'] == row['钉钉姓名,群昵称'].split(',')[0] and idx not in used_idxs:
                    write_item(i, idx, row)
                    flag = True
                    break
            # 2 再找昵称匹配
            if flag:
                continue
            for idx, row in df.iterrows():
                if check_name2(x['微信,关联昵称'], row['钉钉姓名,群昵称']) and idx not in used_idxs:
                    write_item(i, idx, row)
                    flag = True
                    break
            # 3 再不济，姓名昵称随便能等于也行
            if flag:
                continue
            for idx, row in df.iterrows():
                if check_name3(x['姓名'], x['微信,关联昵称'], row['钉钉姓名,群昵称']) and idx not in used_idxs:
                    write_item(i, idx, row)
                    break

        # 3 钉钉中未被匹配的其他考勤数据
        for idx, row in df.sort_values('总应返款', ascending=False).iterrows():
            if idx not in used_idxs:
                i += 1
                write_item(i, idx, row)

        # 【计算返款】
        def get_money(statu):
            if statu == '完成当堂学习':
                return self.视频返款[0]
            elif m := re.match(r'第(\d+)天回放', statu):
                return self.视频返款[int(m.group(1))]
            else:
                return 0

        直播返款课次, 回放返款课次 = [15, 15], [14, 14]
        n, m = max(直播返款课次), max(回放返款课次)
        ls = []  # 今日返款文件
        for i, x in ws.iterrows('姓名', to_dict=['交易订单号']):
            if not x['交易订单号']:
                continue

            total = 0  # 已返款
            for j in range(1, min(15, n + 1)):
                statu = ws.cell2(i, f'第{j:02}课').value
                if not statu: statu = '未开始学习'
                t = get_money(statu)
                if t:
                    if self.视频返款[0] == t and j <= 直播返款课次[1]:
                        total += t
                    elif t < self.视频返款[0] and j <= 回放返款课次[1]:
                        total += t

                    flag = False
                    if self.视频返款[0] == t and (直播返款课次[0] <= j <= 直播返款课次[1]):
                        flag = True
                    if t < self.视频返款[0] and (回放返款课次[0] <= j <= 回放返款课次[1]):
                        flag = True
                    if flag:
                        cols = [x['交易订单号'], t, f'第03届梵呗第{j}课{statu}', x['交易订单号'] + f'_class{j:02}']
                        ls.append(','.join(map(str, cols)))

            打卡返款 = ws.cell2(i, ['打卡返款', '返款']).value
            if 打卡返款:
                total += 打卡返款
                cols = [x['交易订单号'], str(打卡返款), f'第3届梵呗打卡返款', x['交易订单号'] + f'_journal']
                ls.append(','.join(map(str, cols)))
            ws.cell2(i, ['已返款', '总计']).value = total

        wb.save('考勤+.xlsx')
        XlPath(
            f'第{直播返款课次[0]}~{直播返款课次[1]}课直播+第{回放返款课次[0]}~{回放返款课次[1]}课回放返款.csv').write_text(
            '\n'.join(ls))

    def 剩余统计(self):
        n, a, x = 74, 660, 26420 + 4 * 660  # 初始报名人数, 每人报名金额, 剩余报名人数里总共返款额
        print(f'2022年7月梵呗网课，报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')
        # 发潘宏铭老师：2022年7月梵呗网课，报名74人*660元=48840元，剩余19780元。

    @staticmethod
    def 打卡数据():
        a = """晏婷婷	Re-YTT
    徐红梅	红梅,15 胡宗尚
    刘思竹	思竹轩
    嗣禅	嗣禅
    侯继平	吉修·花溪
    姜晓东	梵花
    翁叶玲	susansupplies
    蒋琳	琳
    沈秀红	沈
    王萍	ihappy_wp
    吴婷	吴婷
    林浪波	小林
    程艳丽	
    苏继红	静平
    李华	lihua710202
    王页文	王页文
    娄芸燕	伊芙
    王语浤	次第花开
    萌Helen	萌Helen
    谢勤	清风
    范润玉	小鱼
    孙玮	熊脸猫
    唐公子	唐公子
    于永帅	永帅
    李懿和	礼懿和
    大瑞	大瑞
    宗怡	适庐
    王继青	王继青
    刘军	刘军,爱美丽~刘军
    翟晓琼	晓琼
    释寂木	寂木
    郝宇博	夏阳·初绽
    张敏	绿豆
    杨薇	杨薇
    霍鹏飞	霍鹏飞
    石志宏	wamindfulgeorge,George 石志宏
    杨桂云	常乐
    张枬	张枬
    舒扬	YOUNG SHU
    冯敏	Mintie
    张晓萍	张晓萍
    赵志杰	耀依
    朱女士	zjf
    徐聂儿	尔尔尔好了
    汪晓霞	开水
    廷若	老七,阎霞
    邓相萍	嫣然
    李秋花	0零
    KL	KL
    吕晓楠	Bm
    许光	许光
    陳慧穎	陳慧穎
    吴谨君	吴谨君
    曲婉菲	曲婉菲
    江彩红	安康
    杨昌芳	明月清风
    陈秋菊	彼得拉,陈赛男
    杨炼	微尘
    叶凯萍	葉??·凯萍
    释演法	观心默然
    杨耀宏	一念心性
    秦诗迪	秦诗迪
    源法	源法
    靳晨鸣	定海神针
    周荣艳	周荣艳
    高钰	高格
    杨艳辉	语纾
    陈翠	ajjschen
    姚霖	姚四林"""
        b = """"0零	"	7
    "Stephany	"	5
    "阿白	"	9
    "安康	"	2
    "常乐	"	8
    "陈慧颖	"	8
    "陈赛男	"	9
    "翟晓琼	"	1
    "定海神针	"	4
    "梵花	"	8
    "菲菲	"	6
    "高格	"	6
    "观心默然	"	4
    "郝宇博	"	8
    "河南-侯继平	"	8
    "红梅	"	1
    "霍鹏飞	"	5
    "寂木师父	"	9
    "静平	"	5
    "李华  	"	6
    "刘军	"	7
    "萌Helen31	"	3
    "明月清风	"	3
    "墨	"	6
    "沈	"	8
    "适庐	"	6
    "思竹轩	"	7
    "嗣禅Jeniva	"	5
    "无量	"	5
    "吴婷	"	1
    "小鱼	"	4
    "延若（老七）	"	2
    "杨炼19112723190	"	2
    "杨薇（有事留言.急事电话）	"	1
    "杨耀宏	"	7
    "语纾	"	3"""
        b = b.replace('"', '')
        ys = set(b.splitlines())
        for x in a.splitlines():
            for y in ys:
                name, value = y.split()
                if name in x:
                    print(f'{x}\t{name}\t{value}')
                    ys -= {y}
                    break
            else:
                print(f'{x}\t\t')
        print('----')
        print('\n'.join(ys))

        # 小鹅通里，剩余未匹配的打卡数据
        # 菲菲		6


class 觉观14届2022年08月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR
        """
        self.返款标题 = '第14届觉观'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUnRsSnB0TU5LeWZt'  # 生成日报用
        self.开课日期 = '2022-08-01'  # 通过开课时间，会自动判断出是念住课还是觉观禅课
        self.视频返款 = [59, 45, 30, 15, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = [30, 60, 100]  # 打卡满5/10/15次的返款额
        # 22课×59元+打卡100元=1398元
        self._init(today)

    def 异常处理(self):
        """
        """
        # self.修订(0, 0, '完成当堂学习')
        self.修订(28, 8, '完成当堂学习')

    def 剩余统计(self):
        n, a, x = 96, 1398, 110964  # 初始报名人数, 每人报名金额, 剩余报名人数里总共返款额
        print(f'2022年6月觉观网课，报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')
        # 发潘宏铭老师：2022年8月觉观网课，报名96人*1398元=134208元，剩余23244元。


class 念住09届2022年09月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR
        """
        self.返款标题 = '第09届念住'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUndiSWhwc1VTTUJY'  # 生成日报用
        self.开课日期 = '2022-09-08'  # 通过开课时间，会自动判断出是念住课还是觉观禅课
        self.视频返款 = [20, 15, 10, 5, 0, 0, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = [100, 150, 200]  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元
        self._init(today)

    def 异常处理(self):
        self.修订(62, 1, '第1天回放')
        self.修订(36, 4, '完成当堂学习')

    def 剩余统计(self):
        n, a, x = 114, 620, 51380  # 初始报名人数, 每人报名金额, 剩余报名人数里总共返款额
        print(f'2022年9月念住网课，报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')
        # 发潘宏铭老师：2022年9月念住网课，报名114人*620元=70680元，剩余19300元。
        # 92号没有追回10元，所以报潘老师的余额其实不太对，剩余要再减10元


class 觉观15届2022年10月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR

        最终版-山中薪教室分组表 https://docs.qq.com/sheet/DSHhibnNHU05mc05y?tab=BB08J2&u=991349431c534ad49e9d044c4185f642
        """
        self.返款标题 = '第15届觉观'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUnZzSGllSGtwcEV3'  # 生成日报用
        self.开课日期 = '2022-10-01'  # 通过开课时间，会自动判断出是念住课还是觉观禅课
        self.视频返款 = [63, 48, 32, 16, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4天，完成观看的依次返款额。
        self.打卡返款 = [30, 60, 100]  # 打卡满5/10/15次的返款额
        # 22课×63元+打卡100元=1486元
        self._init(today)

    def 异常处理(self):
        self.修订(78, 14, '第1天回放')
        self.修订(78, 15, '完成当堂学习')
        self.修订(28, 15, '完成当堂学习')
        self.修订(22, 17, '完成当堂学习')
        self.修订(47, 17, '完成当堂学习')
        self.修订(30, 15, '完成当堂学习')

    def 剩余统计(self):
        n, a, x = 87, 1486, 119854  # 初始报名人数, 每人报名金额, 剩余报名人数里总共返款额
        print(f'2022年10月觉观网课，报名{n}人*{a}元-24元={n * a - 24}元，剩余{n * a - 24 - x}元。')
        # 发潘宏铭老师：2022年10月觉观网课，报名87人*1486元-24元=129258元，剩余9404元。
        # 减24元是因为发现有两位师兄报名费少交了，类似这样的细节情况，以及有时候有些全额退费不是我操作的没有记录，
        # 可能会导致我这里有时候汇总的结果跟实际会有些出入，不知道影响不影响。


class 念住10届2022年11月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR
        """
        self.返款标题 = '第10届念住'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUlZHVmZsQVN1dW9U'  # 生成日报用
        self.开课日期 = '2022-11-01'
        self.视频返款 = [20, 15, 10, 5, 0, 0, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = [100, 150, 200]  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元
        self._init(today)

    def 异常处理(self):
        pass
        # self.修订(0, 0, '完成当堂学习')
        # self.修订(0, 0, '第1天回放')
        # 请在下面书写实际修改，并去掉'# '前缀
        self.修订(36, 1, '完成当堂学习')
        self.修订(36, 2, '完成当堂学习')
        self.修订(36, 3, '完成当堂学习')

    def 剩余统计(self):
        n, a, x = 51, 620, 24785  # 初始报名人数, 每人报名金额, 剩余报名人数里总共返款额
        print(f'2022年11月念住网课，报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')
        # 发潘宏铭老师：2022年11月念住网课，报名51人*620元=31620元，剩余6835元。


class 觉观16届2022年12月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR

        最终版-山中薪教室分组表 https://docs.qq.com/sheet/DSHhibnNHU05mc05y?tab=BB08J2&u=991349431c534ad49e9d044c4185f642
        """
        self.返款标题 = '第16届觉观'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUnBJVm5yUHhKV2xU'  # 生成日报用
        self.开课日期 = '2022-12-01'  # 通过开课时间，会自动判断出是念住课还是觉观禅课
        self.视频返款 = [67, 51, 34, 17, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4天，完成观看的依次返款额。
        self.打卡返款 = [30, 60, 100]  # 打卡满5/10/15次的返款额
        # 22课×67元+打卡100元=1574元
        self._init(today)

    def 异常处理(self):
        self.修订(124, 1, '完成当堂学习')
        # 53号感觉在装糊涂~算了，还是简单处理了
        self.修订(53, 12, '完成当堂学习')
        self.修订(53, 14, '完成当堂学习')
        self.修订(160, 20, '完成当堂学习')

    def 剩余统计(self):
        n, a, x = 160, 1574, 221153 - 67 + 3 * 1574  # 初始报名人数, 每人报名金额, 剩余报名人数里总共返款额
        print(f'2022年12月觉观网课，报名{n}人*{a}元={n * a - 24}元，剩余{n * a - 24 - x}元。')
        # 发潘宏铭老师：2022年12月觉观网课，报名160人*1574元=251816元（含3名退课），剩余26008元。


class 念住11届2023年01月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR
        """
        self.返款标题 = '第11届念住'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUmVuSkxzZHJPWUJI'  # 生成日报用
        self.开课日期 = '2023-01-01'
        self.视频返款 = [20, 15, 10, 5, 0, 0, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = [100, 150, 200]  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元
        self._init(today)

    def 异常处理(self):
        self.修订(43, 6, '完成当堂学习')

    def 剩余统计(self):
        n, a, x = 43, 620, 18035  # 初始报名人数, 每人报名金额, 剩余报名人数里总共返款额
        print(f'报名{n}人*{a}元={n * a - 24}元，剩余{n * a - 24 - x}元。')


class 梵呗2023年01月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR
        """
        self.返款标题 = '2023年春节梵呗'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUlpUcVBkbk5SYVNT'  # 生成日报用
        self.开课日期 = '2023-01-22'
        self.视频返款 = [40, 32, 24, 16, 8, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第5天，完成观看的依次返款额。
        # self.打卡返款 = [30, 60, 100]  # 打卡满1/4/7次的返款额
        self.打卡返款 = {1: 30, 4: 60, 7: 100}  # 打卡满1/4/7次的返款额
        # 7课×40元+打卡100元=380元
        self._init(today)
        self.课次数 = 7

    def 异常处理(self):
        self.修订(75, 1, '完成当堂学习')
        self.修订(132, 1, '完成当堂学习')
        self.修订(16, 6, '完成当堂学习')  # 系统明明只记录了28分钟，不过算了，不跟她确认了

    def 剩余统计(self):
        """ 2023年1月春节梵呗考勤

        1、总订单139份，退课8人，实际追踪131人。131人*380元=49780元，理论剩余9248元。
        2、有两位少交80元报名费，有一位是分成180+200元两次交报名费。
        3、修正特殊情况后，实际剩余9112元。
        """


class 觉观17届2023年02月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR
        """
        self.返款标题 = '第17届觉观'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUlRDRHFmTGtzWHJa'  # 生成日报用
        self.开课日期 = '2023-02-01'
        self.视频返款 = [71, 54, 36, 18, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~5天，完成观看的依次返款额。
        # self.打卡返款 = [100, 150, 200]  # 打卡满5/10/15次的返款额
        self.打卡返款 = {5: 30, 10: 60, 15: 100}  # 打卡满5/10/15次的返款额
        # 22课×71元+打卡100元=1662元
        self._init(today)

    def 异常处理(self):
        self.修订(22, 2, '完成当堂学习')
        self.修订(22, 3, '完成当堂学习')
        self.修订(19, 3, '完成当堂学习')
        self.修订(19, 4, '第1天回放')
        self.修订(19, 5, '第1天回放')
        self.修订(19, 10, '第1天回放')

    def 觉观统计(self):
        n, a, x = 76, 1662, 112497 - 71  # 初始报名人数, 每人报名金额, 剩余报名人数里总共返款额
        print(f'2023年2月觉观网课，报名{n}人*{a}元={n * a - 24}元，剩余{n * a - 24 - x}元。')
        # 发潘宏铭老师：2022年12月觉观网课，报名160人*1574元=251816元（含3名退课），剩余26008元。


class 念住12届2023年03月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR
        """
        self.返款标题 = '第12届念住'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUlZDc3FiSGJvd0pn'  # 生成日报用
        self.开课日期 = '2023-03-01'
        self.视频返款 = [20, 15, 10, 5, 0, 0, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = {5: 100, 10: 150, 15: 200}  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元
        self._init(today)

    def 剩余统计(self):
        n, a, x = 42, 620, 17370 - 35  # 初始报名人数, 每人报名金额, 剩余报名人数里总共返款额
        print(f'报名{n}人*{a}元={n * a - 24}元，剩余{n * a - 24 - x}元。')


class 梵呗01届2023年03月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR
        """
        self.返款标题 = '3月梵呗'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUmNFZWRvV3lDeEtL'  # 生成日报用
        self.开课日期 = '2023-03-22'
        self.视频返款 = [40, 32, 24, 16, 8, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4天/第5天，完成观看的依次返款额。
        self.打卡返款 = {1: 30, 4: 60, 7: 100}  # 打卡满1/4/7次的返款额
        # 10课*40元+打卡100元=500元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6413d58ee4b0b0bc2bc8833e',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6413d650e4b0cf39e6ad4867',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6413d6c0e4b0b0bc2bc8844d',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6413f8cbe4b09d7237862326',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6413fa91e4b09d72378623b6',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6413faf8e4b0b2d1c3f98c99',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6413fbbbe4b0b0bc2bc89ac6',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6413fc1de4b0b0bc2bc89b24',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6413fd76e4b0f2aa7dcdc0a0',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6413fe31e4b0b0bc2bc89cb0']

        self._init(today)
        self.driver = None

        # 注意：最后计算打卡数据，需要补下载一个来自旁听教室的数据

    def 异常处理(self):
        self.修订(22, 1, '完成当堂学习')
        self.修订(32, 1, '完成当堂学习')

    def 剩余统计(self):
        n, a, x = 42, 500, 17308  # 初始报名人数, 每人报名金额, 总共返款额
        print(f'报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')


class 觉观18届2023年04月(网课考勤):
    def __init__(self, today=None):
        """
        考勤返款工作手册: https://docs.qq.com/doc/DUnhiZ3JiS2phdVdt
        考勤返款常见问题解答: https://docs.qq.com/doc/DUmdwa01IWHpFdHhR
        """
        self.返款标题 = '第18届觉观'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUk5Kc25GcVZOUm52'  # 生成日报用
        self.开课日期 = '2023-04-01'
        self.视频返款 = [75, 57, 38, 19, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~5天，完成观看的依次返款额。
        self.打卡返款 = {5: 30, 10: 60, 15: 100}  # 打卡满5/10/15次的返款额
        # 22课×75元+打卡100元=1750元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641ab420e4b0cf39e6afeb79',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641ab4c3e4b0b0bc2bcb28b1',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641adb75e4b0b2d1c3fc3d18',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641adc37e4b0b0bc2bcb4c77',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641ade5ee4b0f2aa7dd06e83',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641adfa7e4b0b2d1c3fc4010',  # 6
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641ae1fce4b0f2aa7dd070e2',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641ae2c2e4b09d7237877df2',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641ae3f7e4b0f2aa7dd071fe',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641ae45be4b0cf39e6b0149a',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641ae4c3e4b0b0bc2bcb526e',  # 11
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641ae513e4b09d7237877e5e',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641ae597e4b0b0bc2bcb52dc',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641ae611e4b0b0bc2bcb5311',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641aeac0e4b0b0bc2bcb5535',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641aeb7be4b0b0bc2bcb558f',  # 16
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641aebfce4b0b2d1c3fc4649',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641aec8de4b0b2d1c3fc4693',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641aecdce4b0b2d1c3fc46ad',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641aeda8e4b0cf39e6b01921',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_641aee26e4b0f2aa7dd076af']

        self._init(today)
        self.driver = None

    def 异常处理(self):
        self.修订(57, 1, '完成当堂学习')
        self.修订(48, 5, '完成当堂学习')
        for i in range(1, 80):
            # 第4课设置错了，就给大家统一返款了
            self.修订(i, 4, '完成当堂学习')
        self.修订(56, 20, '完成当堂学习')

    def 剩余统计(self):
        n, a, x = 76, 1750, 117571  # 初始报名人数, 每人报名金额, 总共返款额
        print(f'报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')


class 梵呗02届2023年05月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '5月梵呗'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUkxzSkVLc1duRnRX'  # 生成日报用
        self.开课日期 = '2023-05-10'
        self.视频返款 = [40, 32, 24, 16, 8, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4天/第5天，完成观看的依次返款额。
        self.打卡返款 = {1: 30, 4: 60, 7: 100}  # 打卡满1/4/7次的返款额
        # 10课*40元+打卡100元=500元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64572f0ee4b0b2d1c4124452',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64572f6ae4b0b2d1c4124468',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64572fcce4b0b0bc2be16a58',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_645734afe4b09d72379267c2',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6457351de4b0b0bc2be16bcc',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64573595e4b0cf39e6c60e2b',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_645735eee4b09d72379267ea',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_645736cce4b0b0bc2be16c1a',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6457374ee4b0b0bc2be16c40',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_645737a2e4b0b0bc2be16c5c']

        self._init(today)
        self.driver = None

        # 注意：最后计算打卡数据，需要补下载一个来自旁听教室的数据

    def 异常处理(self):
        pass

    def 剩余统计(self):
        n, a, x = 7, 500, 2686  # 初始报名人数, 每人报名金额, 总共返款额
        print(f'报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')


class 念住13届2023年05月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '第13届念住'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUmZDS0paVWxpSVh3'  # 生成日报用
        self.开课日期 = '2023-05-01'
        self.视频返款 = [20, 15, 10, 5, 0, 0, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = {5: 100, 10: 150, 15: 200}  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f338e4b09d723795745d',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f33ae4b0b0bc2be789a2',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f33ce4b0b2d1c4186852',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f33de4b0f2aa7dec92b1',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f33fe4b0b2d1c4186859',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f342e4b0f2aa7dec92b9',  # 6
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f344e4b0cf39e6cc2d58',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f345e4b0b2d1c4186860',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f347e4b0b2d1c4186862',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f34ae4b0f2aa7dec92bd',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f34ce4b0b0bc2be789b3',  # 11
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f34de4b0b0bc2be789b5',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f34fe4b0b0bc2be789b7',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f351e4b0b2d1c418686a',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f353e4b0b0bc2be789bc',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f355e4b0f2aa7dec92c7',  # 16
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f358e4b0f2aa7dec92c9',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f35ae4b0b2d1c4186870',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f35be4b0b0bc2be789c5',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f35de4b0cf39e6cc2d6c',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f35fe4b0b2d1c4186876']

        self._init(today)

    def 异常处理(self):
        self.修订(9, 1, '完成当堂学习')  # 9号第1天还没开通权限

    def 剩余统计(self):
        n, a, x = 9, 620, 4455  # 初始报名人数, 每人报名金额, 总共返款额
        print(f'报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')


class 觉观19届2023年06月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '第19届觉观'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUnRZQW9EQUpjbmF0'  # 生成日报用
        self.开课日期 = '2023-06-01'
        self.视频返款 = [79, 60, 40, 20, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~5天，完成观看的依次返款额。
        self.打卡返款 = {5: 30, 10: 60, 15: 100}  # 打卡满5/10/15次的返款额
        # 22课×79元+打卡100元=1838元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f338e4b09d723795745d',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f33ae4b0b0bc2be789a2',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f33ce4b0b2d1c4186852',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f33de4b0f2aa7dec92b1',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f33fe4b0b2d1c4186859',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f342e4b0f2aa7dec92b9',  # 6
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f344e4b0cf39e6cc2d58',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f345e4b0b2d1c4186860',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f347e4b0b2d1c4186862',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f34ae4b0f2aa7dec92bd',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f34ce4b0b0bc2be789b3',  # 11
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f34de4b0b0bc2be789b5',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f34fe4b0b0bc2be789b7',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f351e4b0b2d1c418686a',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f353e4b0b0bc2be789bc',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f355e4b0f2aa7dec92c7',  # 16
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f358e4b0f2aa7dec92c9',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f35ae4b0b2d1c4186870',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f35be4b0b0bc2be789c5',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f35de4b0cf39e6cc2d6c',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f35fe4b0b2d1c4186876']

        self._init(today)
        self.driver = None

    def 异常处理(self):
        self.修订(2, 1, '完成当堂学习')
        self.修订(15, 1, '完成当堂学习')

        self.修订(21, 3, '完成当堂学习')
        self.修订(21, 4, '第3天回放')
        self.修订(21, 5, '第3天回放')

        self.修订(4, 11, '完成当堂学习')

        self.修订(20, 16, '完成当堂学习')

    def 剩余统计(self):
        n, a, x = 69, 1838, 109855  # 初始报名人数, 每人报名金额, 总共返款额
        print(f'报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')


class 念住14届2023年07月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '第14届念住'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUkJ6d0RJTGNXdkxJ'  # 生成日报用
        self.开课日期 = '2023-07-01'
        self.视频返款 = [20, 15, 10, 5, 0, 0, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = {5: 100, 10: 150, 15: 200}  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca2fe4b0b2d1c41c866e',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca31e4b0b0bc2bebacec',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca33e4b0b2d1c41c8672',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca34e4b0cf39e6d04d52',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca36e4b0f2aa7df0b2f6',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca38e4b09d7237978564',  # 6
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca3ae4b0b0bc2bebacee',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca3ce4b0b0bc2bebacf2',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca3de4b0b2d1c41c8677',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca3fe4b0f2aa7df0b2fa',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca41e4b0f2aa7df0b2fc',  # 11
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca43e4b0cf39e6d04d57',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca44e4b0f2aa7df0b300',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca46e4b0cf39e6d04d59',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca48e4b0cf39e6d04d5b',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca4ae4b0b2d1c41c867b',  # 16
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca4ce4b0b0bc2bebacfb',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca4ee4b0b0bc2bebacff',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca50e4b0b2d1c41c867d',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca52e4b0b2d1c41c867f',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca53e4b0f2aa7df0b306']

        self._init(today)

    def 异常处理(self):
        # self.修订(9, 1, '完成当堂学习')  # 9号第1天还没开通权限
        pass

    def 剩余统计(self):
        n, a, x = 20, 620, 9470  # 初始报名人数, 每人报名金额, 总共返款额
        print(f'报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')


class 觉观20届2023年08月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '第20届觉观'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DY1B5RFBvTWJxZm5B'  # 生成日报用
        self.开课日期 = '2023-08-01'
        self.视频返款 = [83, 63, 42, 20, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~5天，完成观看的依次返款额。
        self.打卡返款 = {5: 30, 10: 60, 15: 100}  # 打卡满5/10/15次的返款额
        # 22课×83元+打卡100元=1926元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64679878e4b0f2aa7decf4e6',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6467987ae4b09d723795a535',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6467987ce4b0f2aa7decf4e8',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6467987ee4b0b2d1c418c91a',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6467987fe4b0cf39e6cc8f08',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64679881e4b0f2aa7decf4eb',  # 6
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64679883e4b0b0bc2be7ebaf',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64679885e4b0cf39e6cc8f0c',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64679887e4b09d723795a53a',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64679888e4b0b0bc2be7ebb1',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6467988ae4b0f2aa7decf4ef',  # 11
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6467988ce4b0b2d1c418c91c',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6467988ee4b0b0bc2be7ebb3',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64679890e4b0b0bc2be7ebb5',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64679892e4b0f2aa7decf4f2',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64679894e4b0f2aa7decf4f6',  # 16
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64679895e4b0b2d1c418c920',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64679897e4b0f2aa7decf4f9',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6466f35be4b0b0bc2be789c5',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64679899e4b09d723795a53c',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6467989be4b0b0bc2be7ebb7']

        self._init(today)
        self.driver = None

    def 异常处理(self):
        # self.修订(2, 1, '完成当堂学习')
        pass

    def 剩余统计(self):
        n, a, x = 69, 1838, 109855  # 初始报名人数, 每人报名金额, 总共返款额
        print(f'报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')


class 念住15届2023年09月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '第15届念住'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DY0VYUW1MYlpieUVk'  # 生成日报用
        self.开课日期 = '2023-09-01'
        self.视频返款 = [20, 15, 10, 5, 0, 0, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = {5: 100, 10: 150, 15: 200}  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64d8772de4b0d1e42e8d2643',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca31e4b0b0bc2bebacec',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca33e4b0b2d1c41c8672',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca34e4b0cf39e6d04d52',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca36e4b0f2aa7df0b2f6',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca38e4b09d7237978564',  # 6
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca3ae4b0b0bc2bebacee',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca3ce4b0b0bc2bebacf2',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca3de4b0b2d1c41c8677',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca3fe4b0f2aa7df0b2fa',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca41e4b0f2aa7df0b2fc',  # 11
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca43e4b0cf39e6d04d57',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca44e4b0f2aa7df0b300',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca46e4b0cf39e6d04d59',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca48e4b0cf39e6d04d5b',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca4ae4b0b2d1c41c867b',  # 16
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca4ce4b0b0bc2bebacfb',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca4ee4b0b0bc2bebacff',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca50e4b0b2d1c41c867d',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca52e4b0b2d1c41c867f',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca53e4b0f2aa7df0b306']

        self._init(today)
        self.driver = None


class 梵呗04届2023年09月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '9月梵呗'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://kdocs.cn/l/cggKZWOJK5T9'  # 生成日报用
        self.开课日期 = '2023-09-10'
        self.视频返款 = [40, 32, 24, 16, 8, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4天/第5天，完成观看的依次返款额。
        self.打卡返款 = {1: 30, 4: 60, 7: 100}  # 打卡满1/4/7次的返款额

        # https://admin.xiaoe-tech.com/t/live#/detail?id=l_64fa80c4e4b064a8374089c6&tab=student

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64fa80c4e4b064a8374089c6',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64fa8bd7e4b064a863b6ca8d',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64fa8d73e4b064a837408ec7',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64fa8ddee4b064a82f094f9e',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64fa8e32e4b064a82f094fb0',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64fa8e67e4b064a837408f07',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64fa8e9ce4b04c1014b63191',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64fa8f04e4b04c1014b631b4',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64fa89d9e4b064a82f094e53',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_64fa8374e4b064a837408b05']

        self._init(today)
        self.达标时长 = 20
        self.driver = None

        # 注意：最后计算打卡数据，需要补下载一个来自旁听教室的数据

    def 异常处理(self):
        # self.修订(1, 1, '完成当堂学习')
        for i in range(2, 8):
            self.修订(i, 2, '完成当堂学习')
        self.修订(2, 8, '完成当堂学习')

    def 剩余统计(self):
        n, a, x = 3, 500, 1364  # 初始报名人数, 每人报名金额, 总共返款额
        print(f'报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')


class 觉观21届2023年10月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '第21届觉观'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUmNiR0NRbmRicERz'  # 生成日报用
        self.开课日期 = '2023-10-01'
        self.视频返款 = [87, 66, 42, 22, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~5天，完成观看的依次返款额。
        self.打卡返款 = {5: 30, 10: 60, 15: 100}  # 打卡满5/10/15次的返款额
        # 22课×87元+打卡100元=2014元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d186e4b0b0bc2c1df2a9&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d189e4b0b0bc2c1df2ab&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d18be4b0b0bc2c1df2ad&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d18de4b0b0bc2c1df2af&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d18fe4b09d7237b0c906&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d191e4b09d7237b0c908&tab=student',  # 6
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d193e4b0b0bc2c1df2b1&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d195e4b0b0bc2c1df2b3&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d197e4b0b0bc2c1df2b5&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d199e4b0b0bc2c1df2b7&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d19be4b0b0bc2c1df2b9&tab=student',  # 11
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d19de4b0b0bc2c1df2bb&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d19fe4b0b0bc2c1df2bd&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d1a1e4b0b0bc2c1df2bf&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d1a3e4b0b0bc2c1df2c1&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d1a5e4b09d7237b0c90a&tab=student',  # 16
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d1a7e4b0b0bc2c1df2c3&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d1a8e4b0b0bc2c1df2c5&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d1aae4b0b0bc2c1df2c7&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d1ace4b0b0bc2c1df2c9&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6507d1aee4b09d7237b0c90c&tab=student']

        self._init(today)
        self.driver = None

    def 异常处理(self):
        # self.修订(2, 1, '完成当堂学习')

        for i in range(1, 9):
            self.修订(5, i, '完成当堂学习')

        self.修订(3, 1, '完成当堂学习')
        self.修订(15, 1, '完成当堂学习')
        self.修订(15, 2, '完成当堂学习')
        self.修订(15, 3, '完成当堂学习')
        self.修订(12, 3, '完成当堂学习')
        self.修订(12, 4, '完成当堂学习')


class 念住16届2023年11月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '第16届念住'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUkdPS0VMR0tjSVBs'  # 生成日报用
        self.开课日期 = '2023-11-01'
        self.视频返款 = [20, 15, 10, 5, 0, 0, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = {5: 100, 10: 150, 15: 200}  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6527e9d4e4b0b0bc2c27d950&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6527e9d6e4b0b0bc2c27d952&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6527e9d8e4b0b0bc2c27d954&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca34e4b0cf39e6d04d52',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca36e4b0f2aa7df0b2f6',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca38e4b09d7237978564',  # 6
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca3ae4b0b0bc2bebacee',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca3ce4b0b0bc2bebacf2',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca3de4b0b2d1c41c8677',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca3fe4b0f2aa7df0b2fa',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca41e4b0f2aa7df0b2fc',  # 11
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca43e4b0cf39e6d04d57',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca44e4b0f2aa7df0b300',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca46e4b0cf39e6d04d59',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca48e4b0cf39e6d04d5b',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca4ae4b0b2d1c41c867b',  # 16
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca4ce4b0b0bc2bebacfb',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca4ee4b0b0bc2bebacff',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca50e4b0b2d1c41c867d',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca52e4b0b2d1c41c867f',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6472ca53e4b0f2aa7df0b306']

        self._init(today)
        self.driver = None

    def 异常处理(self):
        # self.修订(9, 1, '完成当堂学习')  # 9号第1天还没开通权限
        # self.修订(72, 19, '完成当堂学习')
        pass


class 觉观22届2023年12月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '第22届觉观'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUlBaVWxOUGtsdWtW'  # 生成日报用
        self.开课日期 = '2023-12-01'
        self.视频返款 = [91, 69, 46, 23, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~5天，完成观看的依次返款额。
        self.打卡返款 = {5: 30, 10: 60, 15: 100}  # 打卡满5/10/15次的返款额
        # 22课×91元+打卡100元=2102元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf29e4b023c044fdbae0&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf2ce4b04c10386395ab&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf2ee4b04c100fc7a6f0&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf30e4b0694cd8ef814f&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf33e4b04c100fc7a6f2&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf35e4b04c1093fcfd0e&tab=student',  # 6
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf38e4b04c109d9fbf35&tab=student',  #
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf3ae4b0694cd8ef8153&tab=student',  # 8
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf3ce4b04c10386395af&tab=student',  # 9
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf3ee4b04c10386395b3&tab=student',  # 10
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf40e4b04c1093fcfd14&tab=student',  # 11
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf42e4b04c1093fcfd16&tab=student',  # 12
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf45e4b04c1093fcfd1c&tab=student',  # 13
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf47e4b04c109d9fbf39&tab=student',  # 14
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf49e4b04c1093fcfd1e&tab=studenta',  # 15
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf4be4b04c109d9fbf3b&tab=student',  # 16
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf4ee4b04c109d9fbf3d&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf50e4b04c100fc7a6fb&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf52e4b04c100fc7a6fd&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf54e4b023c044fdbaf2&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf56e4b04c1093fcfd2a&tab=student']

        self._init(today)
        self.driver = None

    def 异常处理(self):
        self.修订(80, 7, '完成当堂学习')
        self.修订(86, 9, '完成当堂学习')
        self.修订(79, 9, '第1天回放')
        pass


class 梵呗05届2023年11月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '11月梵呗'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUnNCYktJanJrTFhW'  # 生成日报用
        self.开课日期 = '2023-11-10'
        self.视频返款 = [40, 32, 24, 16, 8, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4天/第5天，完成观看的依次返款额。
        self.打卡返款 = {1: 30, 4: 60, 7: 100}  # 打卡满1/4/7次的返款额

        # https://admin.xiaoe-tech.com/t/live#/detail?id=l_64fa80c4e4b064a8374089c6&tab=student

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_654b0209e4b0694cd8ebcfb2&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_654b025be4b04c1093f94f59&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_654b029ce4b04c10385f90aa&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_654b02efe4b04c109d9b70d4&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_654b032de4b023c044f9ca6c&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_654b0360e4b023c044f9ca76&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_654b0395e4b023c044f9ca99&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_654b03d1e4b04c100fc3e170&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_654b040be4b023c044f9cb02&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_654b044ce4b023c044f9cb2e&tab=student']

        self._init(today)
        self.达标时长 = 20
        self.driver = None

        # 注意：最后计算打卡数据，需要补下载一个来自旁听教室的数据

    def 异常处理(self):
        # self.修订(1, 1, '完成当堂学习')
        # for i in range(2, 8):
        #     self.修订(i, 2, '完成当堂学习')
        # self.修订(2, 8, '完成当堂学习')
        pass

    def 剩余统计(self):
        n, a, x = 3, 500, 1364  # 初始报名人数, 每人报名金额, 总共返款额
        print(f'报名{n}人*{a}元={n * a}元，剩余{n * a - x}元。')


class 梵呗06届2024年01月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '1月梵呗'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUmVkTkFUcU5CZnNW'  # 生成日报用
        self.开课日期 = '2024-01-09'
        self.视频返款 = [40, 32, 24, 16, 8, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4天/第5天，完成观看的依次返款额。
        self.打卡返款 = {1: 30, 4: 60, 7: 100}  # 打卡满1/4/7次的返款额

        # https://admin.xiaoe-tech.com/t/live#/detail?id=l_64fa80c4e4b064a8374089c6&tab=student

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_659d1327e4b064a87c3058a6&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_659d1371e4b064a8fbe2939b&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_659d14cbe4b04c109af2af28&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_659d14ffe4b064a8fbe29564&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_659d1536e4b04c109af2af60&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_659d1573e4b0d3a962aa0229&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_659d15b1e4b064a87c305b90&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_659d15f7e4b0d3a962aa02ec&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_659d1630e4b064a8fbe29675&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_659d1678e4b064a8fbe296b1&tab=student']

        self._init(today)
        self.达标时长 = 20
        self.driver = None


class 觉观22届2023年12月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '第22届觉观'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUlBaVWxOUGtsdWtW'  # 生成日报用
        self.开课日期 = '2023-12-01'
        self.视频返款 = [91, 69, 46, 23, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~5天，完成观看的依次返款额。
        self.打卡返款 = {5: 30, 10: 60, 15: 100}  # 打卡满5/10/15次的返款额
        # 22课×91元+打卡100元=2102元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf29e4b023c044fdbae0&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf2ce4b04c10386395ab&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf2ee4b04c100fc7a6f0&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf30e4b0694cd8ef814f&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf33e4b04c100fc7a6f2&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf35e4b04c1093fcfd0e&tab=student',  # 6
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf38e4b04c109d9fbf35&tab=student',  #
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf3ae4b0694cd8ef8153&tab=student',  # 8
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf3ce4b04c10386395af&tab=student',  # 9
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf3ee4b04c10386395b3&tab=student',  # 10
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf40e4b04c1093fcfd14&tab=student',  # 11
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf42e4b04c1093fcfd16&tab=student',  # 12
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf45e4b04c1093fcfd1c&tab=student',  # 13
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf47e4b04c109d9fbf39&tab=student',  # 14
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf49e4b04c1093fcfd1e&tab=studenta',  # 15
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf4be4b04c109d9fbf3b&tab=student',  # 16
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf4ee4b04c109d9fbf3d&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf50e4b04c100fc7a6fb&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf52e4b04c100fc7a6fd&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf54e4b023c044fdbaf2&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_655abf56e4b04c1093fcfd2a&tab=student']

        self._init(today)
        self.driver = None

    def 异常处理(self):
        self.修订(80, 7, '完成当堂学习')
        self.修订(86, 9, '完成当堂学习')
        self.修订(79, 9, '第1天回放')
        pass


class 念住17届2024年01月(网课考勤):
    def __init__(self, today=None):
        self.返款标题 = '第17届念住'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DUmdPV21FY0toaFNE'  # 生成日报用
        self.开课日期 = '2024-01-01'
        self.视频返款 = [20, 15, 10, 5, 0, 0, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = {5: 100, 10: 150, 15: 200}  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6582a6e4e4b023c04a51824d&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_6582a6e7e4b0694c0e991655&tab=student',
                         '',
                         '',
                         '',
                         '',  # 6
                         '',
                         '',
                         '',
                         '',
                         '',  # 11
                         '',
                         '',
                         '',
                         '',
                         '',  # 16
                         '',
                         '',
                         '',
                         '',
                         '']

        self._init(today)


class 念住18届2024年03月(网课考勤2):
    def __init__(self, today=None):
        self.返款标题 = '第18届念住'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DSHBBVVJaeXZTeGFj'  # 生成日报用
        self.开课日期 = '2024-03-01'
        self.视频返款 = [20, 15, 10, 5, 0, 0, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = {5: 100, 10: 150, 15: 200}  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bea6e4b04c10a131fa00&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bea8e4b064a83b98af3f&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beaae4b064a8cb278380&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beace4b064a83b98af41&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beaee4b064a83b98af45&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beb0e4b064a83b98af48&tab=student',  # 6
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beb1e4b064a8cb278384&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beb3e4b04c10a131fa29&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beb5e4b04c10a131fa2b&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beb7e4b064a83b98af4e&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beb9e4b04c10a131fa2d&tab=student',  # 11
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bebbe4b064a8cb278388&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bebde4b064a8cb27838c&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bebee4b064a8cb27838f&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bec0e4b04c10a131fa3b&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bec2e4b064a83b98af5a&tab=student',  # 16
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bec4e4b064a83b98af5e&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bec6e4b064a83b98af60&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bec8e4b064a83b98af69&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bec9e4b064a8cb278397&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6becbe4b064a8cb27839b&tab=student'
                         ]
        self.打卡链接 = ['',
                         # 'https://admin.xiaoe-tech.com/t/clock_admin/index#/punchDetail/joinUser?activity_id=ac_65aa8ce34e677_AQornQlM&markType=calendar&active_name=joinUser&createType=0&courseId=&isGroupJoin=true',
                         ]
        self._init(today)

        self.driver = None

        self.app_id = 'apporrfwkpb5562'
        self.client_id = 'xopSPYlP2393519'
        self.secret_key = 'nfzlh4hA0XnTxDU0n6vILOotB3NCxkBG'
        self.login_xe()  # 获得token
        self.prfx = f"《{self.返款标题}网课—" + "{x}》-直播用户列表.{y}.csv"  # 文件名格式

    def 异常处理(self):
        # self.修订(9, 1, '完成当堂学习')  # 9号第1天还没开通权限
        pass


class 梵呗07届2024年03月(网课考勤2):
    def __init__(self, today=None):
        self.返款标题 = '2024.03梵呗'  # '3月梵呗'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DSGNQcW1KSkd3WlFa'  # 生成日报用
        self.开课日期 = '2024-03-09'
        self.视频返款 = [40, 32, 24, 16, 8, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4天/第5天，完成观看的依次返款额。
        self.打卡返款 = {1: 30, 4: 60, 7: 100}  # 打卡满1/4/7次的返款额

        # https://admin.xiaoe-tech.com/t/live#/detail?id=l_64fa80c4e4b064a8374089c6&tab=student

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65e6cb2be4b064a8cfe7c645&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65e6cce5e4b023c0f86da91b&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65e6cd2ee4b064a8cfe7c9ea&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65e6cdd2e4b023c0f86dab0a&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65e6ce08e4b064a8cfe7cb67&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65e6ce40e4b064a8cfe7cc2c&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65e6ce6ce4b023c0f86dad23&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65e6ceb4e4b064a8cfe7cd73&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65e6ceefe4b023c0f86daef3&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65e6cf5ae4b023c0f86db061&tab=student'
                         ]
        self.打卡链接 = ['',
                         # 'https://admin.xiaoe-tech.com/t/clock_admin/index#/punchDetail/joinUser?activity_id=ac_65aa8ce34e677_AQornQlM&markType=calendar&active_name=joinUser&createType=0&courseId=&isGroupJoin=true',

                         ]
        self._init(today)
        self.达标时长 = 20
        self.driver = None

        self.app_id = 'apporrfwkpb5562'
        self.client_id = 'xopSPYlP2393519'
        self.secret_key = 'nfzlh4hA0XnTxDU0n6vILOotB3NCxkBG'
        self.login_xe()  # 获得token
        self.prfx = f"《{self.返款标题}网课-本体音艺网课-" + "{x}》-直播用户列表.{y}.csv"  # 文件名格式

        # 注意：最后计算打卡数据，需要补下载一个来自旁听教室的数据

    def 异常处理(self):
        # self.修订(1, 1, '完成当堂学习')
        pass


class 觉观24届2024年04月(网课考勤2):
    def __init__(self, today=None):
        # super().__init__(today)
        self.返款标题 = '第24届觉观'
        # self.返款标题2 = f"《{self.返款标题}技术公益网课—" + "第{x}堂》-直播用户列表.{}.csv"
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://docs.qq.com/sheet/DSE1TVnlnZk1saXZm'  # 生成日报用
        self.开课日期 = '2024-04-01'
        self.视频返款 = [91, 69, 46, 23, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~5天，完成观看的依次返款额。
        self.打卡返款 = {5: 30, 10: 60, 15: 100}  # 打卡满5/10/15次的返款额
        # 22课×91元+打卡100元=2102元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc2172e4b0d84d784afe25&tab=student',  # 1
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc2174e4b0d84d784afe27&tab=student',  # 2
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc2177e4b092c1684c3aab&tab=student',  # 3
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc2179e4b0d84d784afe29&tab=student',  # 4
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc217be4b0694cfcd6b101&tab=student',  # 5
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc217ee4b092c1684c3aad&tab=student',  # 6
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc2180e4b023c0bea82e8e&tab=student',  # 7
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc2182e4b0694cc0500a63&tab=student',
                         # 8
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc2184e4b092c1684c3ab0&tab=student',
                         # 9
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc2187e4b0694cc0500a66&tab=student',
                         # 10
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc2189e4b023c0bea82e94&tab=student',  # 11
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc218be4b092c1684c3ab5&tab=student',
                         # 12
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc218ee4b0694ccd88ccdd&tab=student',  # 13
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc2190e4b092c1684c3ab9&tab=student',  # 14
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc2193e4b023c0bea82e9b&tab=student',
                         # 15
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc2195e4b092c1684c3abb&tab=student',  # 16
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc2197e4b0694cc0500a72&tab=student',  # 17
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc219ae4b0694cc0500a74&tab=student',  # 18
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc219ce4b0694cfcd6b10d&tab=student',
                         # 19
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc219ee4b0694cc0500a76&tab=student',  # 20
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65fc21a0e4b0694ccd88cce3&tab=student'  # 21
                         ]
        self.打卡链接 = ['',
                         # 'https://admin.xiaoe-tech.com/t/clock_admin/index#/punchDetail/joinUser?activity_id=ac_65aa8ce34e677_AQornQlM&markType=calendar&active_name=joinUser&createType=0&courseId=&isGroupJoin=true',

                         ]
        self._init(today)

        self.driver = None

        self.app_id = 'apporrfwkpb5562'
        self.client_id = 'xopSPYlP2393519'
        self.secret_key = 'nfzlh4hA0XnTxDU0n6vILOotB3NCxkBG'
        self.login_xe()  # 获得token
        self.prfx = f"《{self.返款标题}网课—" + "第{x}堂》-直播用户列表.{y}.csv"  # 文件名格式

    def 异常处理(self):
        self.修订(86, 1, '完成当堂学习')


class 念住19届2024年05月(网课考勤2):
    def __init__(self, today=None):
        self.返款标题 = '第19届念住'
        self.表格路径 = r'考勤.xlsx'
        self.在线表格 = 'https://kdocs.cn/l/caso5Qy2qHgI'  # 生成日报用
        self.开课日期 = '2024-05-01'
        self.视频返款 = [20, 15, 10, 5, 0, 0, 0, 0]  # 直播(当堂)/第1天（当天）/第2天/第3天/第4~7天，完成观看的依次返款额。
        self.打卡返款 = {5: 100, 10: 150, 15: 200}  # 打卡满5/10/15次的返款额
        # 21课×20元+打卡200元=620元

        self.课程链接 = ['',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bea6e4b04c10a131fa00&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bea8e4b064a83b98af3f&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beaae4b064a8cb278380&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beace4b064a83b98af41&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beaee4b064a83b98af45&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beb0e4b064a83b98af48&tab=student',  # 6
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beb1e4b064a8cb278384&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beb3e4b04c10a131fa29&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beb5e4b04c10a131fa2b&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beb7e4b064a83b98af4e&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6beb9e4b04c10a131fa2d&tab=student',  # 11
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bebbe4b064a8cb278388&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bebde4b064a8cb27838c&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bebee4b064a8cb27838f&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bec0e4b04c10a131fa3b&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bec2e4b064a83b98af5a&tab=student',  # 16
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bec4e4b064a83b98af5e&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bec6e4b064a83b98af60&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bec8e4b064a83b98af69&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6bec9e4b064a8cb278397&tab=student',
                         'https://admin.xiaoe-tech.com/t/live#/detail?id=l_65d6becbe4b064a8cb27839b&tab=student'
                         ]
        self.打卡链接 = ['',
                         # 'https://admin.xiaoe-tech.com/t/clock_admin/index#/punchDetail/joinUser?activity_id=ac_65aa8ce34e677_AQornQlM&markType=calendar&active_name=joinUser&createType=0&courseId=&isGroupJoin=true',
                         ]
        self._init(today)

        self.driver = None

        self.app_id = 'apporrfwkpb5562'
        self.client_id = 'xopSPYlP2393519'
        self.secret_key = 'nfzlh4hA0XnTxDU0n6vILOotB3NCxkBG'
        # self.login_xe()     # 获得token
        self.prfx = f"《{self.返款标题}网课—" + "{x}》-直播用户列表.{y}.csv"


def 禅宗3期2_3阶():
    kq = Kq5034()

    course_name = '禅宗3期2,3阶'
    user_list = """S8CVYO-0OZRE8O-6SGI	u_654062775a532_eLbcVlkDyA
S8CWFS-0OZRE8O-277B	u_653084d6681ff_Qr5Je2mLKT
S8AXEY-0OZRE8O-6C0Q	u_65311f2a4a8f6_aUuBoexqfj
S8AEDC-0OZRE8O-6HQU	u_64a4a52f4ea32_Hja5GtMTmw
S8CXSR-0OZRE8O-069B	u_64e8af543213d_1B348cSz7y
S8B2U4-0OZRE8O-43W8	u_652e96ae14794_716z0EP8lW
S8B4MR-0OZRE8O-65BN	u_64f0060f2aa45_le7OtOdciE
S8B2WW-0OZRE8O-FDH9	u_654065fd02528_CfU9m2SDia
S8CZ8C-0OZRE8O-7V7I	u_654065e899b25_uZPfadfSPW
S8AAV0-0OZRE8O-5UD6	u_64f00c013ea3c_vXErW0dwh2
S8AB3I-0OZRE8O-DQNB	u_643d437090c07_tc0x5zLqow
S8D1TK-0OZRE8O-9OVO	u_65110af459f47_oPqtJZRJId
S8D2MT-0OZRE8O-ICKD	u_654065cc49eed_yMwy0Xckjs
S8D3A1-0OZRE8O-I366	u_64a480c93023f_6Ucxwg4Gix
S8AO0S-0OZRE8O-2FAA	u_654065e899b73_AXWItFJdcT
S8D45N-0OZRE8O-BTVB	u_65312aff4ae3b_QMyc7W4RDn
S8D5VR-0OZRE8O-JDSI	u_642e77ce81bab_sI0EGqYodD
S8AAFC-0OZRE8O-37JO	u_6540627fafe74_y0un3LOceW
S8AVIX-0OZRE8O-238Z	u_64eff6be5cbfe_cPaHSOyzDB
S8DGNC-0OZRE8O-GOZ9	u_65311f7c1c97b_sJdpFNeEOl
S8DLLO-0OZRE8O-IISW	u_654062775a2f9_jPNCzy5dfB
请联系义工补充交易订单号	u_648e8c2d8e7be_rSXUUsJDgq
S8E1YQ-0OZRE8O-JVRT	u_6541c688492ee_wZ2zVbpY6U
S8B23S-0OZRE8O-E6XY	u_653b40d5f9478_qVxeqV8VIa
S8ADGA-0OZRE8O-73KY	u_64a4990634c74_BGqm07iwKL
S8B5K7-0OZRE8O-IDA5	u_65406616dc30d_5GIU6x5nel
S8H0RS-0OZRE8O-BYV8	u_65406287e63aa_Cod6ILV2md
S8H1TA-0OZRE8O-13IA	u_654062679405d_Z7ArWZQmV2
S8B23D-0OZRE8O-J0RM	u_64effd9ae4e6d_rJJThKyRnu
S8B35F-0OZRE8O-E5QB	u_654065fd026ad_Ofu6vjTu1N
S8AA1Q-0OZRE8O-DB9L	u_64e8b8b08e920_V8jpcmLZqk
S8APB6-0OZRE8O-2ZWD	u_642e76a74f8a9_MrowgpjttT
S8B5NB-0OZRE8O-BIMA	u_654062439e553_Rwj2Rw5R03
S8H3F5-0OZRE8O-79K7	u_6540660ec58ce_ooJz4MUyvZ
S8AZ0M-0OZRE8O-2O60	u_64eff97a37555_CNSDcoebRa
S8H70Z-0OZRE8O-FXW8	u_6530713919810_4E3SEli3Y0
S8AVJU-0OZRE8O-AX2A	u_6540625d771e7_AQq4LRXTQe
S8D3NP-0OZRE8O-KIQQ	u_6541a6a9f09e9_s1Yt1MHMQy
S8D4ZA-0OZRE8O-INYH	u_654065c349c69_yukV03XQQi
S8D8MA-0OZRE8O-27YV	u_642e8e8c1c586_OufB058FRz
请联系义工补充交易订单号	u_642e79313b88a_qMfRE0CDza
请联系义工补充交易订单号	u_642e7675241fb_hzJCLBnEYp
请联系义工补充交易订单号	u_64e89e8775a9c_XPkkGRBpzV
请联系义工补充交易订单号	u_64e818285df27_yhJtdTfOGZ
请联系义工补充交易订单号	u_642e8ef13bdcd_fiVhVvWaPH
请联系义工补充交易订单号	u_6433e6f23a7e1_WcuHdmt6fX
请联系义工补充交易订单号	u_642eb57c5029f_fxmdRahpqD
请联系义工补充交易订单号	u_64e89dc731dc0_2r2q7nxA94
请联系义工补充交易订单号	u_642e82dd3bb30_tURRlT2L6b
请联系义工补充交易订单号	u_64a4ad3b4eb1b_zrIvgFowlt
请联系义工补充交易订单号	u_64eff7117e0eb_2HBKKD9hlN
请联系义工补充交易订单号	u_64f016458d8ea_hoZFyWyID3
请联系义工补充交易订单号	u_64e8c30475ff1_lP2nknT8cg
请联系义工补充交易订单号	u_64e8b57651f39_SsqXdhMzM1
请联系义工补充交易订单号	u_64eff745da073_Co2FQwaRwA
请联系义工补充交易订单号	u_642e76da635c0_XMdDlHLwIi
请联系义工补充交易订单号	u_65e4c4b3a5263_e7gjz9sDQJ
请联系义工补充交易订单号	u_642e96ac6c819_zngQD8bwPB
""".splitlines()
    voucher_ids = [x.split('\t')[0] for x in user_list]
    user_id2s = [x.split('\t')[1] for x in user_list]

    # 1 视频数据更
    kq.xe2.switch_shop('宗门学府')
    # 1.1 更新课程数据（禅宗比较特别，可以先只考虑全量重置数据，不用考虑增量更新）
    # kq.update_lesson_data_table_by_course(course_name, update_mode=1)
    # kq.xldb.reset_table_item_id('lesson_data_table', counter_name='study_talbe_study_id_seq')

    # 1.2 查看课程数据
    # browser(kq.browser_lesson_data(course_name, user_id2s=user_id2s))

    # 2 打卡数据更新
    # 2.1 更新打卡数据
    # kq.update_clockin_table()
    kq.update_clockin('禅宗3期2,3阶共学打卡')
    kq.update_clockin('禅宗3期2,3阶共修打卡-日常修行')
    kq.update_clockin('禅宗3期2,3阶共修打卡-禅门')
    kq.update_clockin('禅宗3期2,3阶共修打卡-闻思门')
    kq.update_clockin('禅宗3期2,3阶共修打卡-人天福德门')

    kq.update_clockin('禅宗3期2,3阶共学打卡补')
    kq.update_clockin('禅宗3期2,3阶共修打卡-日常修行补')
    kq.update_clockin('禅宗3期2,3阶共修打卡-禅门补')
    kq.update_clockin('禅宗3期2,3阶共修打卡-闻思门补')
    kq.update_clockin('禅宗3期2,3阶共修打卡-人天福德门补')

    # kq.xldb.refine_clockin_data()

    # 2.2 计算打卡数据
    df = kq.browser_clockin_data(course_name,
                                 [
                                     '共学打卡',
                                     '共修打卡-闻思门', '共修打卡-人天福德门',
                                     '共修打卡-禅门', '共修打卡-日常修行',
                                     '共学打卡补',
                                     '共修打卡-闻思门补', '共修打卡-人天福德门补',
                                     '共修打卡-禅门补', '共修打卡-日常修行补',
                                 ],
                                 user_id2s=user_id2s)
    browser(df)

    # 3 返款数据
    # 3.1 更新返款数据（建议以月为单位更新）
    # kq.xldb.update_weipay_from_file(r"C:\Users\kzche\Downloads\1599622041基本账户2024-04-01_2024-04-30.csv")
    # kq.xldb.reset_table_item_id('weipay_table')

    # 3.2 查看返款数据
    # browser(kq.browser_weipay_data(voucher_ids))


class 已完结网课(Kq5034):
    def d231120禅宗1期4阶(self, update=False):
        # file = XlPath(r"D:/home/chenkunze/nut/m2112kq5034/2024年/2023年禅宗01期4阶/考勤.xlsx")
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤表[考勤表['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()
        course_name = '禅宗2期4阶'  # 借用2期4阶的课程数据，数据源是一样的

        # 1 视频数据更新
        # 1.1 更新课程数据（禅宗比较特别，可以先只考虑全量重置数据，不用考虑增量更新）
        if update:
            self.update_lesson_data_table_by_course(course_name, update_mode=2)

        # 1.2 查看课程数据
        df2 = self.browser_lesson_data(course_name, user_id2s=user_id2s)

        # 2 打卡数据更新
        # 2.1 更新打卡数据
        if update:
            self.update_clockin(course_name + '共学打卡')
            self.update_clockin(course_name + '共修打卡')

        # 2.2 计算打卡数据
        #  这里需要将course_name后面的打卡名称后缀，全部都要列出
        df1 = self.browser_clockin_data('禅宗1期4阶',
                                        [
                                            '共学打卡',
                                            '共修打卡'
                                        ],
                                        user_id2s=user_id2s)

        df = pd.merge(df1, df2, on='user_id2')
        browser(df)

    def d240401禅宗2期4阶(self):
        """ 每周六更新

        https://docs.qq.com/sheet/DUlRYQ1BsbWJlb3hw?tab=vpz1li

        todo 首次运行可能稳定性不够，建议开启预热下载一两课后，再重启重新运行
            遇到过第1课导出，第2课说没数据的bug，此时会下载最近1次第1课的数据作为第2课数据，就出问题了
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤表[考勤表['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()
        course_name = '禅宗2期4阶'

        # 1 视频数据更新
        # 1.1 更新课程数据（禅宗比较特别，可以先只考虑全量重置数据，不用考虑增量更新）
        self.update_lesson_data_table_by_course(course_name, update_mode=2)

        # 1.2 查看课程数据
        df2 = self.browser_lesson_data(course_name, user_id2s=user_id2s)

        # 2 打卡数据更新
        # 2.1 更新打卡数据
        self.update_clockin('禅宗1期4阶共学打卡')
        self.update_clockin('禅宗1期4阶共修打卡')
        self.update_clockin('禅宗2期4阶共学打卡')
        self.update_clockin('禅宗2期4阶共修打卡')

        # 2.2 计算打卡数据
        df1a = self.browser_clockin_data('禅宗1期4阶',
                                         [
                                             '共学打卡',
                                             '共修打卡'
                                         ],
                                         user_id2s=user_id2s)
        df1b = self.browser_clockin_data(course_name,
                                         [
                                             '共学打卡',
                                             '共修打卡'
                                         ],
                                         user_id2s=user_id2s)

        df1 = pd.merge(df1a, df1b, on='user_id2')
        df = pd.merge(df1, df2, on='user_id2')
        browser(df)

        # 3 查看返款数据
        # browser(self.browser_weipay_data(voucher_ids))

        """
1、大家好，这是四阶2期第18周截止周六的考勤数据：https://docs.qq.com/sheet/DUlRYQ1BsbWJlb3hw?tab=vpz1li，已按表中的统计进行了返款。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。
        """

    def d240501禅宗5期1阶(self, update=False):
        """ 每周六更新

        https://docs.qq.com/sheet/DUlRYQ1BsbWJlb3hw?tab=vpz1li

        todo 首次运行可能稳定性不够，建议开启预热下载一两课后，再重启重新运行
            遇到过第1课导出，第2课说没数据的bug，此时会下载最近1次第1课的数据作为第2课数据，就出问题了
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤表[考勤表['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()
        course_name = '禅宗5期1阶'

        # 1 视频数据更新
        # 1.1 更新课程数据（禅宗比较特别，可以先只考虑全量重置数据，不用考虑增量更新）
        if update:
            self.update_lesson_data_table_by_course(course_name, update_mode=2)

        # 1.2 查看课程数据
        df2 = self.browser_lesson_data(course_name, user_id2s=user_id2s)

        # 2 打卡数据更新
        # 2.1 更新打卡数据
        if update:
            self.update_clockin(course_name + '共学打卡')
            self.update_clockin(course_name + '共修打卡-随念门')
            self.update_clockin(course_name + '共修打卡-忏悔门')
            self.update_clockin(course_name + '共学打卡(补)')
            self.update_clockin(course_name + '共修打卡-随念门(补)')
            self.update_clockin(course_name + '共修打卡-忏悔门(补)')

        # 2.2 计算打卡数据
        #  这里需要将course_name后面的打卡名称后缀，全部都要列出
        df1 = self.browser_clockin_data(course_name,
                                        [
                                            '共学打卡',
                                            '共修打卡-随念门',
                                            '共修打卡-忏悔门',
                                            '共学打卡(补)',
                                            '共修打卡-随念门(补)',
                                            '共修打卡-忏悔门(补)'
                                        ],
                                        user_id2s=user_id2s)

        df = pd.merge(df1, df2, on='user_id2')
        browser(df)

        # 3 查看返款数据
        # browser(self.browser_weipay_data(voucher_ids))

    def d240419梵呗增益(self):
        """ 4月19日~5月4日

        https://www.kdocs.cn/l/cgrqXRGAvsS1?from=docs
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        报名表 = pd.read_excel(file, '报名表')
        user_id2s = 报名表['用户ID'].tolist()

        # 更新课程和打卡数据
        self.update_lesson_data_table(shop1=True)
        self.update_clockin('2404梵呗增益打卡')

        df1 = self.browser_clockin_data('2404梵呗增益', ['打卡'], user_id2s=user_id2s)
        df2 = self.browser_lesson_data('2404梵呗增益', user_id2s=user_id2s)
        df = pd.merge(df1, df2, on='user_id2')
        browser(df)

    def d240501第19届念住(self):
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤表[考勤表['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()

        # 更新课程和打卡数据
        self.update_lesson_data_table(shop1=True)
        self.update_clockin('第19届念住初阶网课日志【中心教室】')

        # 打卡数据
        titles = [f'第19届念住学修日志-{i:02}' for i in range(1, 22)]
        df1 = self.browser_clockin_data('第19届念住初阶网课日志【中心教室】', [''],
                                        user_id2s=user_id2s, titles=titles)
        # 视频数据
        df2 = self.browser_lesson_data('第19届念住初阶网课-', user_id2s=user_id2s)

        # 拼接显示
        df = pd.merge(df1, df2, on='user_id2')
        browser(df)

        """
        1、大家好，这是202405念住课第5天的考勤数据表：https://www.kdocs.cn/l/caso5Qy2qHgI，已按表中的统计进行了返款。
        2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。
        对常见问题我会在群里统一回复，有必要的情况下我会再主动私信各位核对数据。
        """

        # browser(self.browser_weipay_data(voucher_ids))

    def d240509第8届梵呗初级(self):
        """
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)
        user_id2s = 考勤表['用户ID'].tolist()

        # 更新课程和打卡数据
        self.update_lesson_data_table(shop1=True)
        self.update_clockin('2024.05梵呗-本体音艺网络班中心教室日志')
        self.update_clockin('2024.05梵呗-本体音艺网络班旁听教室日志')

        titles = [f'202405堂{i}-本体音艺初级班学修日志' for i in range(1, 11)]
        df1 = self.browser_clockin_data('2024.05梵呗-本体音艺网络班',
                                        ['中心教室日志', '旁听教室日志'],
                                        user_id2s=user_id2s, titles=titles)
        df2 = self.browser_lesson_data('2024.05梵呗-本体音艺网课', user_id2s=user_id2s)
        df = pd.merge(df1, df2, on='user_id2')
        browser(df)

    def d240601第25届觉观(self, update=False):
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤文件 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤文件[考勤文件['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()

        # 更新课程和打卡数据
        if update:
            self.update_lesson_data_table(shop1=True)
            self.update_clockin('第25届觉观技术公益网课【中心教室】日志')

        # 打卡数据
        titles = [f'【第25届中心教室】—第{i}课打卡' for i in range(1, 22)]
        df1 = self.browser_clockin_data('第25届觉观技术公益网课【中心教室】日志', [''],
                                        user_id2s=user_id2s, titles=titles)
        # 视频数据
        df2 = self.browser_lesson_data('第25届觉观技术公益网课(中心教室）-', user_id2s=user_id2s)

        # 拼接显示
        df = pd.merge(df1, df2, on='user_id2')
        browser(df)

        """
        1、大家好，这是202405念住课第5天的考勤数据表：https://www.kdocs.cn/l/caso5Qy2qHgI，已按表中的统计进行了返款。
        2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。
        对常见问题我会在群里统一回复，有必要的情况下我会再主动私信各位核对数据。
        """

        # browser(self.browser_weipay_data(voucher_ids))

    def d240609梵呗增益(self, update=False):
        """ 4月19日~5月4日

        https://www.kdocs.cn/l/cgrqXRGAvsS1?from=docs
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)
        user_id2s = 考勤表['用户ID'].tolist()

        # 更新课程和打卡数据
        if update:
            self.update_lesson_data_table(shop1=True)
            self.update_clockin('【202406梵呗增益学修班】打卡')

        df1 = self.browser_clockin_data('【202406梵呗增益学修班】', ['打卡'], user_id2s=user_id2s)
        df2 = self.browser_lesson_data('2406【增益', user_id2s=user_id2s)
        df = pd.merge(df1, df2, on='user_id2')
        browser(df)

    def d240701第20届念住(self, update=False):
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤表[考勤表['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()
        course_name = '第20届念住初阶网课'

        # 更新课程和打卡数据
        if update:
            self.update_lesson_data_table(shop1=True)
            self.update_clockin(course_name + '日志【中心教室】')

        # 打卡数据
        titles = [f'第20届念住学修日志-{i:02}' for i in range(1, 22)]
        df1 = self.browser_clockin_data(course_name, ['日志【中心教室】'],
                                        user_id2s=user_id2s, titles=titles)
        # 视频数据
        df2 = self.browser_lesson_data(f'{course_name}-', user_id2s=user_id2s)

        # 拼接显示
        df = pd.merge(df1, df2, on='user_id2')
        browser(df)

        """
1、大家好，这是202407念住初阶中心教室第27天的考勤数据表：https://kdocs.cn/l/cnSzLz0oW80f，已按表中的统计进行了返款。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。

请大家尽量通过问卷的方式反馈问题，有必要也可群里@我或私信我，但后者的消息回复可能不及时（最迟晚上才能回复）。
        """

        # browser(self.browser_weipay_data(voucher_ids))

    def d240706梵呗初阶(self, update=False):
        """ 7月6日~7月13日

        https://kdocs.cn/l/ch73CQ2w0eps
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)
        user_id2s = 考勤表['用户ID'].tolist()

        # 更新课程和打卡数据
        if update:
            self.update_lesson_data_table(shop1=True)
            self.update_clockin('【202407梵呗初阶网络班】打卡')

        df1 = self.browser_clockin_data('【202407梵呗初阶网络班】', ['打卡'], user_id2s=user_id2s)
        df2 = self.browser_lesson_data('2407【初阶', user_id2s=user_id2s)
        df = pd.merge(df1, df2, on='user_id2')
        browser(df)
        """
1、大家好，这是“202407本体音艺暑期初级网络班”课程的考勤数据表：https://kdocs.cn/l/ch73CQ2w0eps，已按表中的统计进行了返款。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。

请大家尽量通过问卷的方式反馈问题，有必要也可群里@我或私信我，但后者的消息回复可能不及时（最迟第二天晚上前才能回复）。
        """

    def d240729梵呗二阶(self, update=False):
        """ 2024/7/29  ~ 2024/8/4

        https://kdocs.cn/l/cudn7ksegohr
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)
        user_id2s = 考勤表['用户ID'].tolist()

        # 更新课程和打卡数据
        if update:
            self.update_lesson_data_table(shop1=True)
            self.update_clockin('【202408梵呗二阶网络班】打卡')

        df1 = self.browser_clockin_data('【202408梵呗二阶网络班】', ['打卡'], user_id2s=user_id2s)
        df2 = self.browser_lesson_data('2408【二阶', user_id2s=user_id2s)
        df = pd.merge(df1, df2, on='user_id2')

        # 修正用
        df.loc[df['user_id2'] == 'u_6608d5814eccf_HxoRJ51PtY', '2408【二阶3.6】梵呗的音乐哲学（1）'] = '当堂完成/100%'
        df.loc[df['user_id2'] == 'u_6608d5814eccf_HxoRJ51PtY', '2408【二阶6.2】比较声乐及其听觉训练（5）'] = '当堂完成/100%'

        browser(df)
        """
大家好，这是“2024暑期本体音艺二阶网络班”课程的考勤数据表：https://kdocs.cn/l/cudn7ksegohr。


1、大家好，这是“2024暑期本体音艺二阶网络班”课程的考勤数据表：https://kdocs.cn/l/cudn7ksegohr，已按表中的统计进行了返款。
2、若有缺漏或错误，可以私信我反馈（我每天早晚会统一看私信），请大家放心。


        """

    def d240801第26届觉观(self, update=False):
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤文件 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤文件[考勤文件['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()
        course_name = '第26届觉观技术公益网课'

        # 更新课程和打卡数据
        if update:
            self.update_lesson_data_table(shop1=True)
            self.update_clockin('第26届觉观技术公益网课【中心教室】日志')

        # 打卡数据
        titles = [f'【第26届中心教室】—第{i}课打卡' for i in range(1, 22)]
        df1 = self.browser_clockin_data('第26届觉观技术公益网课【中心教室】', ['日志'],
                                        user_id2s=user_id2s, titles=titles)
        # 视频数据
        df2 = self.browser_lesson_data('第26届觉观技术公益网课（中心教室）-', user_id2s=user_id2s)

        # 拼接显示
        df = pd.merge(df1, df2, on='user_id2')

        # 修正用
        df.loc[df['user_id2'] == 'u_66970c2fa7415_n1Nbl6GDDL', ['第26届觉观技术公益网课（中心教室）-第5课',
                                                                '第26届觉观技术公益网课（中心教室）-第14课']] = '当堂完成/100%'
        df.loc[df['user_id2'] == 'u_669ba60f6efc8_GNkteBZH4a', '第26届觉观技术公益网课（中心教室）-第6课'] = '当堂完成/100%'
        df.loc[df['user_id2'] == 'u_66a8b0a52f8d2_sqPJlXJ9CI', ['第26届觉观技术公益网课（中心教室）-第9课',
                                                                '第26届觉观技术公益网课（中心教室）-第11课',
                                                                '第26届觉观技术公益网课（中心教室）-第12课',
                                                                '第26届觉观技术公益网课（中心教室）-第13课']] = '当堂完成/100%'
        df.loc[df['user_id2'] == 'u_66a874bc52e3a_aIawPCVZl2', '第26届觉观技术公益网课（中心教室）-第18课'] = '当堂完成/100%'

        browser(df)

        """
3、截至目前，有少数学员没有任何观看、打卡记录。这可能是由于报名表中的信息有误，导致未能匹配到正确的用户ID。如果您实际观看了直播但考勤表中却未记录，请私信联系我。        
3、第22课答疑不考勤，已经给所有学员统一返款了。
今天是最后一天返款，若有缺漏或错误请私信我，我会配合修正，请大家放心。


1、大家好，这是"5034第26届觉观网课中心教室"第26天的考勤数据表：https://kdocs.cn/l/ce9QrC7Stfsl，已按表中的统计进行了返款。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。

请大家尽量通过问卷的方式反馈问题，有必要也可群里@我或私信我，但后者的消息回复可能不及时（预计晚上统一回复）。
        """

        # browser(self.browser_weipay_data(voucher_ids))

    def d240901第21届念住(self, update=False):
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤表[考勤表['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()
        course_name = '第21届念住初阶网课'

        # 更新课程和打卡数据
        if update:
            self.update_lesson_data_table(shop1=True)
            self.update_clockin(course_name + '日志【中心教室】')

        # 打卡数据
        titles = [f'第21届念住学修日志-{i:02}' for i in range(1, 22)]
        df1 = self.browser_clockin_data(course_name, ['日志【中心教室】'],
                                        user_id2s=user_id2s, titles=titles)
        # 视频数据
        df2 = self.browser_lesson_data(f'{course_name}-', user_id2s=user_id2s)

        # 拼接显示
        df = pd.merge(df1, df2, on='user_id2')

        # 修正用
        df.loc[df['user_id2'] == 'u_6673e2a13b7f4_lHMWyj4vJN', ['第21届念住初阶网课-3']] = '当堂完成/100%'

        browser(df)

        """
1、大家好，这是"202409 →中心教室念住初阶"第28天的考勤数据表：https://kdocs.cn/l/canu94Dmliao，已按表中的统计进行了返款。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。

请大家尽量通过问卷的方式反馈问题，有必要也可群里@我或私信我，但后者的消息回复可能不及时（最迟当天晚上才能回复）。
        """

        # browser(self.browser_weipay_data(voucher_ids))

    def d240909梵呗初阶(self, update=False):
        """ 9月9日~9月18日

        https://kdocs.cn/l/ccN2WVbcRpct
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)
        user_id2s = 考勤表['用户ID'].tolist()

        # 更新课程和打卡数据
        if update:
            self.update_lesson_data_table(shop1=True)
            self.update_clockin('【202409梵呗初阶网络班】打卡')

        titles = [f'学修日志{i:02}' for i in range(1, 11)]
        df1 = self.browser_clockin_data('【202409梵呗初阶网络班】', ['打卡'], user_id2s=user_id2s, titles=titles)
        df2 = self.browser_lesson_data('2409【初阶', user_id2s=user_id2s)
        df = pd.merge(df1, df2, on='user_id2')
        browser(df)
        """

1、大家好，这是“202409本体音艺初级”课程的考勤数据表：https://kdocs.cn/l/ccN2WVbcRpct，已按表中的统计进行了返款。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，也可群里@我或私信我，我会配合修正。

请大家尽量通过问卷的方式反馈问题，有必要也可群里@我或私信我，但后者的消息回复可能不及时（最迟第二天晚上前才能回复）。
        """

    def d240907禅宗6期1阶(self, update=False):
        """ 每周六更新

        https://kdocs.cn/l/catZWOYJu6n3

        todo 首次运行可能稳定性不够，建议开启预热下载一两课后，再重启重新运行
            遇到过第1课导出，第2课说没数据的bug，此时会下载最近1次第1课的数据作为第2课数据，就出问题了
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤表[考勤表['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()
        course_name = '禅宗6期1阶'

        # 1 视频数据更新
        # 1.1 更新课程数据（禅宗比较特别，可以先只考虑全量重置数据，不用考虑增量更新）
        if update:
            self.update_lesson_data_table_by_course(course_name, update_mode=2)

        # 1.2 查看课程数据
        df2 = self.browser_lesson_data(course_name, user_id2s=user_id2s)

        # 2 打卡数据更新
        # 2.1 更新打卡数据
        if update:
            self.update_clockin(course_name + '共学打卡')
            self.update_clockin(course_name + '共修打卡-随念门')
            self.update_clockin(course_name + '共修打卡-忏悔门')
            # self.update_clockin(course_name + '共学打卡(补)')
            # self.update_clockin(course_name + '共修打卡-随念门(补)')
            # self.update_clockin(course_name + '共修打卡-忏悔门(补)')

        # 2.2 计算打卡数据
        #  这里需要将course_name后面的打卡名称后缀，全部都要列出
        df1 = self.browser_clockin_data(course_name,
                                        [
                                            '共学打卡',
                                            '共修打卡-随念门',
                                            '共修打卡-忏悔门',
                                            # '共学打卡(补)',
                                            # '共修打卡-随念门(补)',
                                            # '共修打卡-忏悔门(补)'
                                        ],
                                        user_id2s=user_id2s)

        df = pd.merge(df1, df2, on='user_id2')

        # 修正用
        df.loc[df['user_id2'] == 'u_65379a1a8577e_svMOYzCAAO', ['第02周-第08课-印度佛教史4']] = '已完成'

        df.loc[df['user_id2'] == 'u_66d47e944a580_czbzN2lcSa', '共修打卡-忏悔门'] += 7
        df.loc[df['user_id2'] == 'u_66d47e944a580_czbzN2lcSa', '共学打卡'] -= 7
        browser(df)

        # 3 查看返款数据
        # browser(self.browser_weipay_data(voucher_ids))
        """
1、大家好，这是"202409禅宗6期1阶"课程的考勤数据表：https://kdocs.cn/l/catZWOYJu6n3，已按表中的统计进行了返款。
    1）「1.返款进度」：截至目前，学员完成的听课（当周课程若未在当周内完成，即使后续补看也不予返款）与打卡情况对应的返款进度。
    2）「2.考试资格」：自开课以来，学员累计完成的听课与打卡情况（不作为返款的依据）
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。
        """

    def d241001第27届觉观(self, update=False):
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤文件 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤文件[考勤文件['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()
        course_name = '第27届觉观技术公益网课'

        # 更新课程和打卡数据
        if update:
            self.update_lesson_data_table(shop1=True)
            self.update_clockin(course_name + '【中心教室】日志')

        # 打卡数据
        titles = [f'【第27届中心教室】—第{i}课打卡' for i in range(1, 22)]
        df1 = self.browser_clockin_data(f'{course_name}【中心教室】', ['日志'],
                                        user_id2s=user_id2s, titles=titles)
        # 视频数据
        df2 = self.browser_lesson_data(f'{course_name}-', user_id2s=user_id2s)

        # 拼接显示
        df = pd.merge(df1, df2, on='user_id2')

        # 修正用
        # df.loc[df['user_id2'] == 'u_66970c2fa7415_n1Nbl6GDDL', ['第26届觉观技术公益网课（中心教室）-第5课', '第26届觉观技术公益网课（中心教室）-第14课']] = '当堂完成/100%'
        # df.loc[df['user_id2'] == 'u_64d17c902f42e_Y5b7eaBYxa', '第27届觉观技术公益网课【中心教室】-第1课'] = '当堂完成/100%'

        user_ids = ['u_64d17c902f42e_Y5b7eaBYxa', 'u_66fc12727bad5_WkjDSk4BwB', 'u_66fbf046f61c4_Aml2LzZCMN',
                    'u_63337fbc1d095_tQ9stCgzT1']
        df.loc[df['user_id2'].isin(user_ids), '第27届觉观技术公益网课-1'] = '当堂完成/100%'
        df.loc[df['user_id2'] == 'u_66f909d46998b_eJ8VOr7ZGS', '第27届觉观技术公益网课-2'] = '第1天回放/100%'
        df.loc[df['user_id2'] == 'u_66fa925792acb_ILzoaVO9Rs', '第27届觉观技术公益网课-6'] = '当堂完成/100%'
        df.loc[df['user_id2'] == 'u_66f8caa664235_ltZ4Wg8kbG', '第27届觉观技术公益网课-9'] = '当堂完成/100%'
        browser(df)
        """
1、截至目前，有少数学员没有任何观看、打卡记录。这可能是由于报名表中的信息有误，导致未能匹配到正确的用户ID。如果您实际观看了直播但考勤表中却未记录，请私信联系我。        
2、第22课答疑不考勤，已经给所有学员统一返款了。
今天是最后一天返款，若有缺漏或错误请私信我，我会配合修正，请大家放心。
3、今晚第22课答疑不考勤，明天统一返款。


1、大家好，这是"5034第27届觉观网课中心教室"第27天的考勤数据表：https://kdocs.cn/l/clk9s1rdCmZd，已按表中的统计进行了返款。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。

请大家尽量通过问卷的方式反馈问题，有必要也可群里@我或私信我，但后者的消息回复可能不及时（预计晚上统一回复）。

        """
        # browser(self.browser_weipay_data(voucher_ids))

    def d241009梵呗增益(self, update=False):
        """ 2024/10/09  ~ 2024/11/02

        https://kdocs.cn/l/cnjjFDNJsLFd
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)
        user_id2s = 考勤表['用户ID'].tolist()

        # 更新课程和打卡数据
        if update:
            self.update_lesson_data_table(shop1=True)
            self.update_clockin('【202410梵呗二阶网络班】打卡')

        df1 = self.browser_clockin_data('【202410梵呗二阶网络班】', ['打卡'], user_id2s=user_id2s)
        df2 = self.browser_lesson_data('2410【增益堂', user_id2s=user_id2s)
        df = pd.merge(df1, df2, on='user_id2')

        # 修正用
        df.loc[df['user_id2'] == 'u_615a84c98ba83_jh7I8x2eXX', '2410【增益堂1】穿脱海青教学'] = '当堂完成/100%'
        # df.loc[df['user_id2'] == 'u_6608d5814eccf_HxoRJ51PtY', '2408【二阶6.2】比较声乐及其听觉训练（5）'] = '当堂完成/100%'

        browser(df)
        """
1、大家好，这是“202410本体音艺初阶增益班级群”课程的考勤数据表：https://kdocs.cn/l/cnjjFDNJsLFd，已按表中的统计进行了返款。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，有必要也可私信我反馈（我每天早晚会统一看私信），请大家放心。
        """

    def d241101第22届念住(self, update=False):
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤表[考勤表['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()
        course_name = '第22届念住初阶网课'

        # 更新课程和打卡数据
        if update:
            self.update_lesson_data_table(shop1=True)
            self.update_clockin(course_name + '日志【中心教室】')

        # 改数据库里数据
        self.xldb.patch_lesson_data('第22届念住初阶网课-1', 'u_67202620dbfc6_wwl2XfXBHs', 4)

        # 打卡数据
        titles = [f'第22届念住学修日志-{i:02}' for i in range(1, 22)]
        df1 = self.browser_clockin_data(course_name, ['日志【中心教室】'],
                                        user_id2s=user_id2s, titles=titles)
        # 视频数据
        df2 = self.browser_lesson_data(f'{course_name}-', user_id2s=user_id2s)

        # 拼接显示
        df = pd.merge(df1, df2, on='user_id2')

        # 修正用
        # df.loc[df['user_id2'] == 'u_6673e2a13b7f4_lHMWyj4vJN', ['第21届念住初阶网课-3']] = '当堂完成/100%'

        browser(df)

        """
3、统一回复下截止目前的问卷反馈，考勤数据不是实时更新的，而是以我每天上午发送通知的时间为更新点。所以下午晚上没看到数据，次日我发通知再确认就行了。
已反馈的25号、7号（错写为10号）数据均为此原因，已自动修正。还有一位反馈11号但姓名对应检索不到的未处理，估计是同性质问题，若检查今天数据后仍有疑惑可以私聊我。

1、大家好，这是"202411→ 念住初阶中心教室"第16天的考勤数据表：https://kdocs.cn/l/cgmoDIupTfOt，已按表中的统计进行了返款。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。
3、目前统计的考勤结果是截至我当日统计时间点前的最新进度。后续各位学员的新观看和打卡进度将会在明天的考勤统计中体现。

请大家尽量通过问卷的方式反馈问题，有必要也可群里@我或私信我，但后者的消息回复可能不及时（最迟当天晚上才能回复）。
        """

        # browser(self.browser_weipay_data(voucher_ids))

    def d241101第28届觉观(self, update=False):
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤文件 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤文件[考勤文件['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()
        course_name = '第28届觉观技术公益网课'

        # 更新课程和打卡数据
        if update:
            self.xe2.switch_shop('5034山中薪')
            self.update_lesson_data_table(shop1=True)
            self.update_clockin(course_name + '【中心教室】')

        # self.xldb.patch_lesson_data('第28届觉观技术公益网课-第1课', 'u_67217ff0aa538_CceLXOL3Fu', 1)
        # self.xldb.patch_lesson_data('第28届觉观技术公益网课-第1课', 'u_66b9683b5f560_B6yCBpc1xx', 1)
        # self.xldb.patch_lesson_data('第28届觉观技术公益网课-第1课', 'u_66b9683b5f560_B6yCBpc1xx', 4)
        # self.xldb.patch_lesson_data('第28届觉观技术公益网课-第9课', 'u_66b9683b5f560_B6yCBpc1xx', 0)

        # 打卡数据
        titles = [f'【第28届中心教室】—第{i}课打卡' for i in range(1, 22)]
        df1 = self.browser_clockin_data(f'{course_name}', ['【中心教室】'],
                                        user_id2s=user_id2s, titles=titles)
        # 视频数据
        df2 = self.browser_lesson_data(f'{course_name}-', user_id2s=user_id2s)

        # 拼接显示
        df = pd.merge(df1, df2, on='user_id2')

        browser(df)
        # browser(self.browser_weipay_data(voucher_ids))
        return df.to_dict(orient='split')

    def d241109梵呗初阶(self, update=False):
        """ 11月9日~11月18日

        https://kdocs.cn/l/ci3TL8cyshAJ
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)
        user_id2s = 考勤表['用户ID'].tolist()

        # 更新课程和打卡数据
        if update:
            self.update_lesson_data_table(shop1=True)
            self.update_clockin('【202411梵呗初阶网络班】打卡')

        titles = [f'学修日志{i:02}' for i in range(1, 11)]
        df1 = self.browser_clockin_data('【202411梵呗初阶网络班】', ['打卡'], user_id2s=user_id2s, titles=titles)
        df2 = self.browser_lesson_data('2411【初阶', user_id2s=user_id2s)
        df = pd.merge(df1, df2, on='user_id2')
        browser(df)
        """
1、大家好，这是“202411本体音艺初阶”课程的考勤数据表：https://kdocs.cn/l/ci3TL8cyshAJ

1、大家好，这是“202411本体音艺初阶”课程的考勤数据表：https://kdocs.cn/l/ci3TL8cyshAJ，已按表中的统计进行了返款。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，也可群里@我或私信我，我会配合修正。

请大家尽量通过问卷的方式反馈问题，有必要也可群里@我或私信我，但后者的消息回复可能不及时（最迟第二天晚上前才能回复）。
        """

    def d241012禅宗4期23阶(self, update=True):
        """ 每周六更新

        https://kdocs.cn/l/cc3iLFCYawr0

        todo 首次运行可能稳定性不够，建议开启预热下载一两课后，再重启重新运行
            遇到过第1课导出，第2课说没数据的bug，此时会下载最近1次第1课的数据作为第2课数据，就出问题了
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤表[考勤表['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()
        course_name = '禅宗4期2,3阶'

        # 1 视频数据更新
        # 1.1 更新课程数据（禅宗比较特别，可以先只考虑全量重置数据，不用考虑增量更新）
        if update:
            self.update_lesson_data_table_by_course(course_name, update_mode=2)

        # 1.2 查看课程数据
        df2 = self.browser_lesson_data(course_name, user_id2s=user_id2s)

        # 2 打卡数据更新
        # 2.1 更新打卡数据
        if update:
            self.update_clockin(course_name + '共学打卡')
            self.update_clockin(course_name + '共修打卡-闻思门')
            self.update_clockin(course_name + '共修打卡-人天福德门')
            self.update_clockin(course_name + '共修打卡-禅门')
            self.update_clockin(course_name + '共修打卡-日常修行')

        # 2.2 计算打卡数据
        #  这里需要将course_name后面的打卡名称后缀，全部都要列出
        df1 = self.browser_clockin_data(course_name,
                                        [
                                            '共学打卡',
                                            '共修打卡-闻思门',
                                            '共修打卡-人天福德门',
                                            '共修打卡-禅门',
                                            '共修打卡-日常修行',
                                        ],
                                        user_id2s=user_id2s)

        df = pd.merge(df1, df2, on='user_id2')

        # 修正用
        # df.loc[df['user_id2'] == 'u_65379a1a8577e_svMOYzCAAO', ['第02周-第08课-印度佛教史4']] = '已完成'
        # df.loc[df['user_id2'] == 'u_66d47e944a580_czbzN2lcSa', '共修打卡-忏悔门'] += 7
        # df.loc[df['user_id2'] == 'u_66d47e944a580_czbzN2lcSa', '共学打卡'] -= 7
        browser(df)

        # 3 查看返款数据
        # browser(self.browser_weipay_data(voucher_ids))
        """
1、大家好，这是"202410禅宗4期2,3阶"课程截止第4周的考勤数据表：https://kdocs.cn/l/cc3iLFCYawr0，已按表中的统计进行了返款。
    1）「1.返款进度」：截至目前，学员完成的听课（当周课程若未在当周内完成，即使后续补看也不予返款）与打卡情况对应的返款进度。
    2）「2.考试资格」：自开课以来，学员累计完成的听课与打卡情况（不作为返款的依据）
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。
        """

    def d240916禅宗3期4阶(self, update=True):
        """ 每周六更新

        https://kdocs.cn/l/cgG7BqIycu8o

        todo 首次运行可能稳定性不够，建议开启预热下载一两课后，再重启重新运行
            遇到过第1课导出，第2课说没数据的bug，此时会下载最近1次第1课的数据作为第2课数据，就出问题了
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径

        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤表[考勤表['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()
        course_name = '禅宗3期4阶'

        # 1 视频数据更新
        # 1.1 更新课程数据（禅宗比较特别，可以先只考虑全量重置数据，不用考虑增量更新）
        if update:
            self.update_lesson_data_table_by_course(course_name, update_mode=2)

        # 1.2 查看课程数据
        df2 = self.browser_lesson_data(course_name, user_id2s=user_id2s)

        # 2 打卡数据更新
        # 2.1 更新打卡数据
        if update:
            self.update_clockin(course_name + '共学打卡')
            self.update_clockin(course_name + '共修打卡')
            # self.update_clockin(course_name + '共修打卡-忏悔门')
            # self.update_clockin(course_name + '共学打卡(补)')
            # self.update_clockin(course_name + '共修打卡-随念门(补)')
            # self.update_clockin(course_name + '共修打卡-忏悔门(补)')

        # 2.2 计算打卡数据
        #  这里需要将course_name后面的打卡名称后缀，全部都要列出
        df1 = self.browser_clockin_data(course_name,
                                        [
                                            '共学打卡',
                                            '共修打卡',
                                            # '共修打卡-忏悔门',
                                            # '共学打卡(补)',
                                            # '共修打卡-随念门(补)',
                                            # '共修打卡-忏悔门(补)'
                                        ],
                                        user_id2s=user_id2s)

        df = pd.merge(df1, df2, on='user_id2')

        # 修正用
        # df.loc[df['user_id2'] == 'u_65379a1a8577e_svMOYzCAAO', ['第02周-第08课-印度佛教史4']] = '已完成'
        # df.loc[df['user_id2'] == 'u_66d47e944a580_czbzN2lcSa', '共修打卡-忏悔门'] += 7
        # df.loc[df['user_id2'] == 'u_66d47e944a580_czbzN2lcSa', '共学打卡'] -= 7
        browser(df)

        # 3 查看返款数据
        # browser(self.browser_weipay_data(voucher_ids))
        """
1、大家好，这是"禅宗修道普及班四阶3期中心教室"截止第9周的考勤数据表：https://kdocs.cn/l/cgG7BqIycu8o，已按表中的统计进行了返款。
    1）「1.返款进度」：截至目前，学员完成的听课（当周课程若未在当周内完成，即使后续补看也不予返款）与打卡情况对应的返款进度。
    2）「2.考试资格」：自开课以来，学员累计完成的听课与打卡情况（不作为返款的依据）
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。
        """

    def d241208禅宗6期23阶(self, update=True):
        """ 每周日更新

        https://kdocs.cn/l/ce0Poc7TrQiI

        todo 首次运行可能稳定性不够，建议开启预热下载一两课后，再重启重新运行
            遇到过第1课导出，第2课说没数据的bug，此时会下载最近1次第1课的数据作为第2课数据，就出问题了
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤表[考勤表['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()
        course_name = '禅宗6期2,3阶'

        # 1 视频数据更新
        # 1.1 更新课程数据（禅宗比较特别，可以先只考虑全量重置数据，不用考虑增量更新）
        if update:
            self.update_lesson_data_table_by_course(course_name, update_mode=0)

        # 1.2 查看课程数据
        df2 = self.browser_lesson_data(course_name, user_id2s=user_id2s)

        # 2 打卡数据更新
        # 2.1 更新打卡数据
        if update:
            self.update_clockin(course_name + '共学打卡')
            self.update_clockin(course_name + '共修打卡-闻思门')
            self.update_clockin(course_name + '共修打卡-人天福德门')
            self.update_clockin(course_name + '共修打卡-禅门')
            self.update_clockin(course_name + '共修打卡-日常修行')

        # 2.2 计算打卡数据
        #  这里需要将course_name后面的打卡名称后缀，全部都要列出
        df1 = self.browser_clockin_data(course_name,
                                        [
                                            '共学打卡',
                                            '共修打卡-闻思门',
                                            '共修打卡-人天福德门',
                                            '共修打卡-禅门',
                                            '共修打卡-日常修行',
                                        ],
                                        user_id2s=user_id2s)

        df = pd.merge(df1, df2, on='user_id2')

        # 修正用
        # df.loc[df['user_id2'] == 'u_65379a1a8577e_svMOYzCAAO', ['第02周-第08课-印度佛教史4']] = '已完成'
        # df.loc[df['user_id2'] == 'u_66d47e944a580_czbzN2lcSa', '共修打卡-忏悔门'] += 7
        # df.loc[df['user_id2'] == 'u_66d47e944a580_czbzN2lcSa', '共学打卡'] -= 7
        browser(df)

        # 3 查看返款数据
        # browser(self.browser_weipay_data(voucher_ids))
        """
1、大家好，这是"202410禅宗6期2,3阶"课程截止第1周的考勤数据表：https://kdocs.cn/l/ce0Poc7TrQiI，已按表中的统计进行了返款。
    1）「1.返款进度」：截至目前，学员完成的听课（当周课程若未在当周内完成，即使后续补看也不予返款）与打卡情况对应的返款进度。
    2）「2.考试资格」：自开课以来，学员累计完成的听课与打卡情况（不作为返款的依据）
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。
        """

    def d241208禅宗7期1阶(self, update=False):
        """ 每周六更新

        https://kdocs.cn/l/cdUe2cTHdEAU

        todo 首次运行可能稳定性不够，建议开启预热下载一两课后，再重启重新运行
            遇到过第1课导出，第2课说没数据的bug，此时会下载最近1次第1课的数据作为第2课数据，就出问题了
        """
        # 配置文件
        file = self.root / (func_input_message(depth=1)['funcname'] + '.xlsx')  # 这个函数名就是表格配置路径
        考勤表 = pd.read_excel(file, '考勤表', skiprows=1)  # 先用简单的读取方式，后续可以扩展更灵活的判断操作方式
        考勤表 = 考勤表[考勤表['用户ID'].notna()]
        user_id2s = 考勤表['用户ID'].tolist()
        voucher_ids = 考勤表['交易订单号'].tolist()
        course_name = '禅宗7期1阶'

        # 1 视频数据更新
        # 1.1 更新课程数据（禅宗比较特别，可以先只考虑全量重置数据，不用考虑增量更新）
        if update:
            self.update_lesson_data_table_by_course(course_name, update_mode=2)

        # 1.2 查看课程数据
        df2 = self.browser_lesson_data(course_name, user_id2s=user_id2s)

        # 2 打卡数据更新
        # 2.1 更新打卡数据
        if update:
            self.update_clockin(course_name + '共学打卡')
            self.update_clockin(course_name + '共修打卡-随念门')
            self.update_clockin(course_name + '共修打卡-忏悔门')
            # self.update_clockin(course_name + '共学打卡(补)')
            # self.update_clockin(course_name + '共修打卡-随念门(补)')
            # self.update_clockin(course_name + '共修打卡-忏悔门(补)')

        # 2.2 计算打卡数据
        #  这里需要将course_name后面的打卡名称后缀，全部都要列出
        df1 = self.browser_clockin_data(course_name,
                                        [
                                            '共学打卡',
                                            '共修打卡-随念门',
                                            '共修打卡-忏悔门',
                                            # '共学打卡(补)',
                                            # '共修打卡-随念门(补)',
                                            # '共修打卡-忏悔门(补)'
                                        ],
                                        user_id2s=user_id2s)

        df = pd.merge(df1, df2, on='user_id2')

        # 修正用
        # df.loc[df['user_id2'] == 'u_65379a1a8577e_svMOYzCAAO', ['第02周-第08课-印度佛教史4']] = '已完成'
        df.loc[df['user_id2'] == 'u_6753b73b8282d_O901HkAlCl', ['-第04周-第14课-二选一 法门通论随念门2']] = '已完成'
        # df.loc[df['user_id2'] == 'u_66d47e944a580_czbzN2lcSa', '共修打卡-忏悔门'] += 7
        # df.loc[df['user_id2'] == 'u_66d47e944a580_czbzN2lcSa', '共学打卡'] -= 7
        browser(df)

        # 3 查看返款数据
        # browser(self.browser_weipay_data(voucher_ids))
        """
1、大家好，这是"202412禅宗7期1阶"课程截止第1周的考勤数据表：https://kdocs.cn/l/cdUe2cTHdEAU，已按表中的统计进行了返款。
    1）「1.返款进度」：截至目前，学员完成的听课（当周课程若未在当周内完成，即使后续补看也不予返款）与打卡情况对应的返款进度。
    2）「2.考试资格」：自开课以来，学员累计完成的听课与打卡情况（不作为返款的依据）
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。
        """


if __name__ == '__main__':
    with TicToc():
        fire.Fire(念住07届2022年05月)

        # 每天批量退款链接：https://pay.weixin.qq.com/index.php/xphp/cbatchrefund/batch_refund#/pages/index/index
