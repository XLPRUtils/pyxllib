from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'ccubH8g6TnrX',
                         'V2-78OlydSkG9LMdZ9832LmxA',
                         课程商品名='四阶',
                         )

    def step2(self):
        """ 每个课程主要的差异在这里 """
        if self.get_status() >= 2:
            return
        logger.info('2 从数据库xldb3获取新的考勤数据并写回在线表格')

        # 1 获取用户清单
        self.更新用户匹配()
        self.把报名表的用户ID同步到考勤表()

        user_id2s = self.wb_get_column_list('用户ID')

        # 2 拼接应该写入的数据
        # todo 这个也要加titles过滤重复
        df1 = self.browser_clockin_data(f'{self.course_name}-', None, user_id2s=user_id2s)
        df1.loc[df1['user_id2'] == 'u_6753a43d484a8_shYEPQU77L', '共学打卡'] += 12

        # 视频数据
        df2 = self.browser_lesson_data(f'{self.course_name}', user_id2s=user_id2s)

        # 拼接数据
        df3 = self.拼接打卡视频数据(user_id2s, df1, df2)

        # 3 把数据写入表格
        self.write_rows_skip_empty(df3, find_start_col='共学打卡')

        logger.info('已将最新考勤数据文本写入在线表格')
        self.set_status(2)

    def step6(self):
        if self.get_status() >= 6:
            return
        logger.info('6 发送日报')
        name = re.search(r'\d+期.+?阶', self.course_name).group()
        wechat_lock_send('禅宗修道考勤管理', self.get_daily(f'禅宗修道普及班{name}中心教室'))
        self.set_status(6)

    def 从旧课程数据迁移配置(self):
        self.kqdb.禅宗从旧课程继承配置('d250317禅宗4阶46期', self.course_name)

    def 配置打卡数据(self):
        self.update_clockin('d250623禅宗7期4阶-共学打卡',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_655d7e11d7fe9_NJwCWJ&community_id=c_6559bc14e1748_2hIi6q392659&clock_id=ac_6860089849316_ZFzlgt4v&group_id=group_6860089894e6e_GpidW0bO&taskId=th_68600898866a8_xuRmwYHq&beginDate=247&totalDay=246&is_combine_task=246&management_entry_id=calendar_clock_management&component_name=clock_task_data')
        self.update_clockin('d250623禅宗7期4阶-共修打卡',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_655d816e28c27_qql1P9&community_id=c_6559bc14e1748_2hIi6q392659&clock_id=ac_686008c1bb6a0_YtVxNjFG&group_id=group_686008c2346a7_yKqc8rvF&taskId=th_686008c225d62_91einfrT&beginDate=247&totalDay=246&is_combine_task=246&management_entry_id=calendar_clock_management&component_name=clock_task_data')

    def 爬虫检查课程目录(self):
        url = ('https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course?'
               'communityId=c_6559bc14e1748_2hIi6q392659'
               '&course_id=course_2ysDL7wJraFZgrrngXlp1LT00ZH&type=manage')
        courses = self.xe2.爬虫获得禅宗课程目录(url)
        del courses['选修-24阿含-丁小平老师']

        print(courses)
        return courses

    def 配置课次数据(self):
        courses = self.爬虫检查课程目录()
        self.kqdb.添加禅宗课次配置数据(self.course_name, 15, courses)


def main_a():
    """ 凌晨先更新数据 """
    kq = 考勤课程()
    kq.step1()
    kq.step2()
    kq.step3()


def main_b():
    """ 上午再进行返款 """
    kq = 考勤课程()
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
        kq = 考勤课程()
        # kq.配置课次数据()

        kq.status = 0
        kq.step1()
        kq.step2()
        kq.step3()
        kq.step4()
        kq.step5()

        # main_b()
