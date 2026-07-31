from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(1,
                         XlPath(__file__).stem,
                         'cq0o1Qi5xqNx',
                         'V2-5yLRQUdcUZjQCyLpt64yqw',
                         5)

    def step2(self):
        """ 每个课程主要的差异在这里 """
        if self.get_status() >= 2:
            return
        logger.info('2 从数据库xldb获取新的考勤数据并写回在线表格')

        # 1 获取用户清单
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
        self.set_status(2)

    def step6(self):
        if self.get_status() >= 6:
            return
        logger.info('6 发送日报')
        wechat_lock_send('本体音艺考勤班委群', self.get_daily(f'20{self.course_name[1:5]}本体音艺初阶班级群', -1))
        # 梵呗系列太特别，最后还是用-6的特殊标记；才能保证晚上的脚本可以继续增量更新。
        # todo 这个有隐患，以后才考虑更好的优化策略。
        self.set_status(-6)


def main_a():
    """ 当天晚上更新数据 """
    kq = 考勤课程()
    if not kq.get_daily():
        return
    kq.step1()
    kq.step2()
    kq.step3()


def main_b():
    """ 次日进行返款 """
    kq = 考勤课程()
    if not kq.get_daily(bias=-1):
        return
    kq.step1()
    kq.step2()
    kq.step3()

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

        # 更新修订()
