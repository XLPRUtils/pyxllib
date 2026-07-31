from xlsln.kq5034.courses.kqcourse import *
from xlsln.kq5034.courses.zen_stage3_catalog import 获取禅宗三阶补充课次


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'cgQCbhITRJwA',
                         'V2-34ZicIDNmuN44rHJzWkioA',
                         课程商品名='三阶'
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
        name = re.search(r'\d+期.+?阶', self.course_name).group()
        wechat_lock_send('线上修道班考勤管理', self.get_daily(f'禅宗修道普及班{name}中心教室'))
        self.set_status(6)

    def 从旧课程数据迁移配置(self):
        raise RuntimeError('d260412禅宗9期三阶的视频课次请改用 配置课次数据() 爬取真实目录')

    def 爬虫检查课程目录(self):
        courses = 获取禅宗三阶补充课次(self.xe2)
        print(courses)
        return courses

    def 配置课次数据(self):
        courses = self.爬虫检查课程目录()
        self.kqdb.添加禅宗课次配置数据(self.course_name, 20, courses)

    def 配置打卡数据(self):
        course_name = self.course_name
        self.update_clockin(f'{course_name}-共学打卡',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_69da82d13fec7_A0biMG&community_id=c_69da74779e777_KWuZ4MFJ2307&clock_id=ac_69da82eee9e43_YWTB8J7b&group_id=group_69da82fc3f44e_iABh8M3Y&taskId=th_69da82fc3f4a3_3JukwRRr&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data',
                            download=False)
        self.update_clockin(f'{course_name}-共修打卡-闻思门',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_69dbcfbd5e92f_nrumMA&community_id=c_69da74779e777_KWuZ4MFJ2307&clock_id=ac_69dbcfd96d7ef_A1IsZ3EG&group_id=group_69dbcfe7ab6f4_LEc0jbVw&taskId=th_69dbcfe7ab74b_asywx556&beginDate=63&totalDay=62&is_combine_task=62&management_entry_id=calendar_clock_management&component_name=clock_task_data',
                            download=False)
        self.update_clockin(f'{course_name}-共修打卡-禅门',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_69e3bf1f62caa_xd3eeK&community_id=c_69da74779e777_KWuZ4MFJ2307&clock_id=ac_69e3bf369fb51_sSuFH7Rd&group_id=group_69e3bf3d6ba0a_z17J8jLj&taskId=th_69e3bf3d6ba5b_cv1j2Gwq&beginDate=57&totalDay=56&is_combine_task=56&management_entry_id=calendar_clock_management&component_name=clock_task_data',
                            download=False)
        self.update_clockin(f'{course_name}-共修打卡-日常行门',
                            url='',
                            download=False)


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
        # kq = 考勤课程()
        # kq.status = 0
        # kq.step3()
        # kq.step4()
        # kq.step5()
        # kq.step6()

        # kq.配置课次数据()
        # kq.配置打卡数据()

        main_b()
