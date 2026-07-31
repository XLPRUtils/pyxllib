from pyxllib.file.xlsxlib import get_column_letter

from xlsln.kq5034.ckz240412网课考勤 import *
from xlproject.code4101 import support_retry_process

from xlsln.kq5034.courses.base import 网课考勤Ex


class 网课考勤Exx(网课考勤Ex):

    def __init__(self):
        super().__init__()
        self.shop_id = 1
        self.course_name = 'd250301第32届觉观'
        self.wb = KqCourseBook('cdbUPz8HyZtN')
        self.status = None
        self.start_date = datetime.date(2025, 3, 1)
        self.回放天数 = 5

    def step2(self):
        """ 每个课程主要的差异在这里 """
        if self.get_status() >= 2:
            return
        logger.info('2 从数据库xldb3获取新的考勤数据并写回在线表格')

        # 250203周一，"一人多账号"问题，需要使用合并数据大法~
        self.xldb.merge_user('u_679b46026bd40_7KsAIYUOo7', 'u_679b212089b62_2F2rviCIyw')  # 3-14 王福巧

        # 1 获取用户清单
        user_id2s = self.wb.get_column_list('用户ID')

        # 2 拼接应该写入的数据
        # 打卡数据
        titles = [f'【打卡】第32届中心教室-{i}' for i in range(1, 23)]
        df1 = self.xldb.browser_clockin_data(f'{self.course_name}-', ['打卡数'],
                                             user_id2s=user_id2s, titles=titles)

        # 修正对应user_id2的考勤数据
        # target_user = 'u_6773ff3e6fdec_Wv1NqEcSbD'
        # mask = df1['user_id2'] == target_user
        # df1.loc[mask, '打卡数'] = 15

        # 视频数据
        df2 = self.browser_lesson_data(f'{self.course_name}-', user_id2s=user_id2s)
        # 删掉带有"测试"的数据列，是开课前的"测试1"、"测试2“
        df2 = df2.loc[:, ~df2.columns.str.contains('测试')]
        # 拼接数据
        df3 = self.拼接打卡视频数据(user_id2s, df1, df2)

        # 3 把数据写入表格
        self.write_rows_skip_empty(df3)

        logger.info('已将最新考勤数据文本写入在线表格')
        self.set_status(2)

    def step6(self):
        if self.get_status() >= 6:
            return
        logger.info('6 发送日报')
        wechat_lock_send('考勤中台', '@艳子')
        wechat_lock_send('考勤中台', self.get_daily('5034第32届觉观网课山中薪中心教室'))
        self.set_status(6)


@support_retry_process()
def main():
    kq = 网课考勤Exx()
    if not kq.get_daily():
        return
    kq.step1()
    kq.step2()
    kq.step3()
    kq.step4()
    kq.step5()
    kq.step6()


def 更新修订():
    """ 可能手动修正了一些异常，需要刷新显示数据的时候，可以用这个函数，强制重运行第2、3步 """
    kq = 网课考勤Exx()
    kq.status = 1
    kq.step2()
    kq.step3()


if __name__ == '__main__':
    os.chdir(get_xl_homedir())
    if len(sys.argv) > 1:
        fire.Fire()
    else:
        kq = 网课考勤Exx()
        kq.status = 1
        kq.step2()
        kq.step3()
