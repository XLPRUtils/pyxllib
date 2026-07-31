from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'cafhLQZFBPLN',
                         'V2-34ZicIDNmuN44rHJzWkioA',
                         课程商品名='二阶',
                         打卡返款=True,
                         )
# https://www.kdocs.cn/api/v3/ide/file/cafhLQZFBPLN/script/V2-34ZicIDNmuN44rHJzWkioA/sync_task
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
        if not self.打卡返款:  # 如果不统计打卡返款，所有打卡数据都设为空字符串
            for col in df1.columns:
                if col != 'user_id2':
                    df1[col] = ''

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
        self.kqdb.禅宗从旧课程继承配置('d251026禅宗9期二阶', self.course_name)

    def 配置课次数据(self):
        courses = self.爬虫检查课程目录()
        self.kqdb.添加禅宗课次配置数据(self.course_name, 9, courses)

    def 配置打卡数据(self):
        course_name = self.course_name
        self.update_clockin(f'{course_name}-共学打卡',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_68fcd7fe73b1f_hdnHh1&community_id=c_68fcd4f92ba7b_yx91ZF6s3843&clock_id=ac_69627ecfcd312_5jajS1er&group_id=group_69627ed011bd4_PAAnRTum&taskId=th_69627ed00e25f_zPDKT9Xv&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data',
                            download=False)
        self.update_clockin(f'{course_name}-共修打卡-持诵门',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_68fcd8c0be576_LIGSIe&community_id=c_68fcd4f92ba7b_yx91ZF6s3843&clock_id=ac_69627ef43c7df_IPMYHHz6&group_id=group_69627ef488145_rjjSmdUT&taskId=th_69627ef48306c_DUIMb597&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data',
                            download=False)
        self.update_clockin(f'{course_name}-共修打卡-阿含诵',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_69046e48d0ee4_mdYiOk&community_id=c_68fcd4f92ba7b_yx91ZF6s3843&clock_id=ac_69627e5fe3ad0_CZuuOYLM&group_id=group_69627e6abe16a_YFWIrFVj&taskId=th_69627e6abe1a6_o1XLiYCc&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data',
                            download=False)
        self.update_clockin(f'{course_name}-共修打卡-人天福德门',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_69046ddca2cc6_wPsoDI&community_id=c_68fcd4f92ba7b_yx91ZF6s3843&clock_id=ac_69627e919d0df_81fEXlTb&group_id=group_69627e984acdd_voHPJJCG&taskId=th_69627e984ad2c_hDYekDPa&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data',
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
        kq = 考勤课程()
        # kq.更新用户匹配()
        # kq.配置打卡数据()
        # kq.从旧课程数据迁移配置()

        kq.status = 0
        # kq.step1()
        # kq.step2()
        # kq.step3()
        kq.step4()
        kq.step5()
        # kq.step6()

        # kq.配置打卡数据()

        # main_a()
