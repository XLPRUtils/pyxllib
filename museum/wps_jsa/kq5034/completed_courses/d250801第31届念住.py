from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(1,
                         XlPath(__file__).stem,
                         'cca6UOxftXrf',
                         'V2-6DtGBVDEeuArTd7mAnB0XG',
                         7,
                         课程商品名='第31届念住禅法（初阶）【中心教室】')

    def test_wb2(self):
        print(self.wb.run_func('locateTableRange', '考勤表', 4, ['视频应返款']))

    def step2(self):
        """ 每个课程主要的差异在这里 """
        if self.get_status() >= 2:
            return
        logger.info('2 从数据库xldb2获取新的考勤数据并写回在线表格')

        # 0 "一人多账号"问题，需要使用合并数据大法~
        # self.kqdb.merge_user('u_6839c09608380_TDO4Xyn9NC', 'u_683a347037c38_qnI4VlkXW5')  # 1-04

        # 1 获取用户清单
        self.更新用户匹配()
        self.把报名表的用户ID同步到考勤表()

        user_id2s = self.wb_get_column_list('用户ID')

        # 2 拼接应该写入的数据
        # 打卡数据
        titles = [f'念住学修日志-{i:02}' for i in range(1, 22)]
        df1 = self.browser_clockin_data(f'{self.course_name}-', ['打卡数'],
                                        user_id2s=user_id2s, titles=titles)
        df1.loc[df1['user_id2'] == 'u_66a1a6564eb76_7UxhLvXXQW', '打卡数'] += 2
        # 视频数据
        df2 = self.browser_lesson_data(f'{self.course_name}-', user_id2s=user_id2s)
        # 删掉带有"测试"的数据列，是开课前的"测试1"、"测试2“
        df2 = df2.loc[:, ~df2.columns.str.contains('测试')]

        # 处理国外学生
        # self.shift_international_students(df2, ['u_679c7f4716dca_ZAy3CiX3q2'])

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
        wechat_lock_send('考勤中台', '@如如')

        tag = self.course_name[1:5]
        if int(tag) % 2:
            wechat_lock_send('考勤中台', self.get_daily(f'20{tag}念住单月网课群'))
        else:
            wechat_lock_send('考勤中台', self.get_daily(f'20{tag}念住双月网课群'))

        self.set_status(6)


def main():
    kq = 考勤课程()
    if not kq.get_daily():
        return
    kq.step1()
    kq.step2()
    kq.step3()
    kq.step4()
    kq.step5()
    kq.step6()


if __name__ == '__main__':
    os.chdir(get_xl_homedir())
    if len(sys.argv) > 1:
        fire.Fire()
    else:
        kq = 考勤课程()
        # kq.status = 0
        # kq.step1()
        # kq.step2()
        # kq.step3()
        # kq.step4()
        # kq.step5()
        # kq.step6()
