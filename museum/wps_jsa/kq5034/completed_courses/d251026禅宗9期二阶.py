from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'chO73qp5jxk9',
                         'V2-34ZicIDNmuN44rHJzWkioA',
                         课程商品名='二阶',
                         打卡返款=False,
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
        df1.loc[df1['user_id2'] == 'u_6881f52818b30_RttH0nClSN', '共修打卡-阿含诵'] += 1

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

    def 爬虫检查课程目录(self):
        url = ('https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course?'
               'communityId=c_68fcd4f92ba7b_yx91ZF6s3843'
               '&course_id=course_34YouPwPrDWvHXy4v7OSXwAHhWe&type=manage')
        courses = self.xe2.爬虫获得禅宗课程目录(url)

        print(courses)
        return courses

    def 配置课次数据(self):
        courses = self.爬虫检查课程目录()
        self.kqdb.添加禅宗课次配置数据(self.course_name, 9, courses)

    def 配置打卡数据(self):
        course_name = self.course_name
        self.update_clockin(f'{course_name}-共学打卡',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_68fcd7fe73b1f_hdnHh1&community_id=c_68fcd4f92ba7b_yx91ZF6s3843&clock_id=ac_68fcd879e6ac5_MoV9UEQO&group_id=group_68fcd887e0524_F0dMlSPo&taskId=th_68fcd887e0579_PUy8uuxq&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data')
        self.update_clockin(f'{course_name}-共修打卡-持诵门',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_68fcd8c0be576_LIGSIe&community_id=c_68fcd4f92ba7b_yx91ZF6s3843&clock_id=ac_68fcd8e1a77a3_Ov8Z6LMr&group_id=group_68fcd8f1c3338_XtEKVJf0&taskId=th_68fcd8f1c338e_LqAfceAv&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data')
        self.update_clockin(f'{course_name}-共修打卡-阿含诵',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_69046e48d0ee4_mdYiOk&community_id=c_68fcd4f92ba7b_yx91ZF6s3843&clock_id=ac_69046e687baa8_JdgI5F6E&group_id=group_69046e7402c25_WXZQryXr&taskId=th_69046e7402c7c_uazKed34&beginDate=59&totalDay=58&is_combine_task=58&management_entry_id=calendar_clock_management&component_name=clock_task_data')
        self.update_clockin(f'{course_name}-共修打卡-人天福德门',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_69046ddca2cc6_wPsoDI&community_id=c_68fcd4f92ba7b_yx91ZF6s3843&clock_id=ac_69046e088ea0d_OuiPFllL&group_id=group_69046e2739399_JFpNgcYD&taskId=th_69046e27393ee_Dcsw50Bb&beginDate=59&totalDay=58&is_combine_task=58&management_entry_id=calendar_clock_management&component_name=clock_task_data')


def main_a():
    """ 凌晨先更新数据 """
    kq = 考勤课程()
    kq.step1()
    kq.step2()
    kq.step3()


def main_b():
    """ 上午再进行返款 """
    kq = 考勤课程()
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
        kq = 考勤课程()
        # kq.更新用户匹配()
        # kq.配置打卡数据()

        kq.status = 0
        kq.step2()
        # kq.step3()
        # kq.step4()
        # kq.step5()
        # kq.step6()

        # kq.配置打卡数据()

        # main_b()
