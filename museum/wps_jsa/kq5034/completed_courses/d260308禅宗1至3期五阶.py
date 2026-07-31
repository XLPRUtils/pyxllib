# 已完结课程脚本，仅保留历史追溯。
from xlsln.kq5034.courses.kqcourse import *
from xlsln.kq5034.courses.zen_stage5_catalog import 获取禅宗五阶补充课次, 获取禅宗五阶一次性排课事故日期偏移


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'cqHyxKU6vBya',
                         'V2-78OlydSkG9LMdZ9832LmxA',
                         课程商品名='5阶',
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
        df1 = self.browser_clockin_data(f'{self.course_name}-', None, user_id2s=user_id2s)

        # 视频数据
        df2 = self.browser_lesson_data(f'{self.course_name}', user_id2s=user_id2s)

        # 拼接数据
        df3 = self.拼接打卡视频数据(user_id2s, df1, df2)

        # 3 把数据写入表格
        self.write_rows_skip_empty(df3, find_start_col='共学打卡')

        logger.info('已将最新考勤数据文本写入在线表格')
        self.set_status(2)

    def step6(self):
        self._ensure_previous_steps_completed(6, 6)
        if self.get_status() >= 6:
            return
        logger.info('6 发送日报')
        name = re.search(r'\d+期.+?阶', self.course_name).group().replace('点', '.')
        wechat_lock_send('线上修道班考勤管理', self.get_daily(f'禅宗修道普及班{name}中心教室'))
        self.set_status(6)

    def 爬虫检查课程目录(self):
        courses = 获取禅宗五阶补充课次(self.xe2)
        print(courses)
        return courses

    def 配置课次数据(self):
        courses = self.爬虫检查课程目录()
        self.kqdb.添加禅宗课次配置数据(
            self.course_name,
            20,
            courses,
            week_date_offsets=获取禅宗五阶一次性排课事故日期偏移(self.course_name),
        )

    def 配置打卡数据(self):
        gongxue_url = ('https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper'
                       '?apply_id=apy_68454a6be3b68_0ewWIx&community_id=c_6824b2a916fe1_jT7cZ3pb4838'
                       '&clock_id=ac_69ace94feb04f_GPLHveE6&group_id=group_69ace95943e40_Ww9UVFAs'
                       '&taskId=th_69ace95943e9c_5xoWb73r&beginDate=106&totalDay=105&is_combine_task=105'
                       '&management_entry_id=calendar_clock_management&component_name=clock_task_data')
        gongxiu_url = ('https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper'
                       '?apply_id=apy_6845525bdd131_8EdUBr&community_id=c_6824b2a916fe1_jT7cZ3pb4838'
                       '&clock_id=ac_69ace9035f9db_HeWDEBKY&group_id=group_69ace90f0af14_g2ZxwvZ2'
                       '&taskId=th_69ace90f0af83_P5GJ8iXF&beginDate=106&totalDay=105&is_combine_task=105'
                       '&management_entry_id=calendar_clock_management&component_name=clock_task_data')
        self.update_clockin(f'{self.course_name}-共学打卡',
                            url=gongxue_url,
                            download=False)
        self.update_clockin(f'{self.course_name}-共修打卡',
                            url=gongxiu_url,
                            download=False)

    def 从旧课程数据迁移配置(self):
        self.kqdb.禅宗从旧课程继承配置(
            'd260301禅宗46期五阶',
            self.course_name,
            week_date_offsets=获取禅宗五阶一次性排课事故日期偏移(self.course_name),
        )


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

    # d250920 明天先不跑返款，我看下如何补修打卡数据
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
        # kq.从旧课程数据迁移配置()
        # kq.配置打卡数据()

        # kq.status = 0
        # kq.step1()
        kq.step2()
        kq.step3()
        kq.step4()
        kq.step5()
        # kq.step6()

        # kq.配置课次数据()

        # main_b()
