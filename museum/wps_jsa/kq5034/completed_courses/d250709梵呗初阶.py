from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(1,
                         XlPath(__file__).stem,
                         'coRKxDSwxut0',
                         'V2-5yLRQUdcUZjQCyLpt64yqw',
                         5,
                         1,
                         课程商品名='202507本体音艺线上班',
                         )

    def step2(self, *, status=2):
        """ 每个课程主要的差异在这里 """
        if self.get_status() >= status:
            return
        logger.info('2 从数据库xldb获取新的考勤数据并写回在线表格')

        # 1 获取用户清单
        self.更新用户匹配()
        self.把报名表的用户ID同步到考勤表()

        user_id2s = self.wb_get_column_list('用户ID')

        # 2 拼接应该写入的数据
        titles = [f'学修日志{i:02}' for i in range(1, 12)]
        df1 = self.kqdb.browser_clockin_data(f'{self.course_name}-', ['打卡数'],
                                             user_id2s=user_id2s, titles=titles)
        # 视频数据
        df2 = self.browser_lesson_data(f'{self.course_name}', user_id2s=user_id2s)
        # 拼接数据
        df3 = self.拼接打卡视频数据(user_id2s, df1, df2)

        # 3 把数据写入表格
        self.write_rows_skip_empty(df3)

        logger.info('已将最新考勤数据文本写入在线表格')
        self.set_status(status)

    def step6(self, *, status=6):
        if self.get_status() >= status:
            return
        logger.info('6 发送日报')
        wechat_lock_send('本体音艺考勤班委群', self.get_daily(f'20{self.course_name[1:5]}本体音艺初阶班级群', -1))
        # 梵呗情况特殊，需要把完成状态设置到昨天
        self.set_status(status)

    def 配置课次和打卡链接(self):
        # self.add_book_lessons_to_db()
        self.update_clockin('d250709梵呗初阶-打卡数',
                            'https://admin.xiaoe-tech.com/t/clock_admin/index#/punchDetail/diaryList?activity_id=ac_67583ac77ea39_XP8r9zBp&markType=&miniMiddleUrl=https%3A%2F%2Fapporrfwkpb5562.h5.xiaoeknow.com%2Fxiaoe_clock%2Fmini_middle%3Factivity_id%3Dac_67583ac77ea39_XP8r9zBp%26app_id%3Dapporrfwkpb5562')


def main_a():
    """ 当天晚上更新数据 """
    kq = 考勤课程()
    if not kq.get_daily():
        return
    kq.step1(status=7)
    kq.step2(status=8)
    kq.step3(status=9)


def main_b():
    """ 次日进行返款 """
    kq = 考勤课程()
    if not kq.get_daily(bias=-1):
        return
    # 由于梵呗执行main_b的时候都是新的一天，所以梵呗不能补跑123，否则会重复下载打卡数据等
    # kq.step1()
    # kq.step2()
    # kq.step3()

    # 主要是继续跑4、5、6
    kq.step4()
    kq.step5()
    kq.step6()


if __name__ == '__main__':
    os.chdir(get_xl_homedir())
    if len(sys.argv) > 1:
        fire.Fire()
    else:
        main_b()
