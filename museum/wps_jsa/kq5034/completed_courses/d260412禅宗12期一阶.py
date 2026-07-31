from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'copnS6juyN2T',
                         'V2-4ihfz275sveOiNjWQChw3G',
                         课程商品名='一阶',
                         打卡返款=True,  # 这个参数结课了再开
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
        # user_id2为'u_686097ef3e45c_IRtTLGIAyi'，其'共修打卡-崇拜祈祝'次数加1
        # df1.loc[df1['user_id2'] == 'u_686097ef3e45c_IRtTLGIAyi', '共修打卡-崇拜祈祝'] += 1
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
        wechat_lock_send('线上修道班考勤管理', self.get_daily(f'禅宗修道普及班{name}中心教室'))
        self.set_status(6)

    def 从旧课程数据迁移配置(self):
        self.kqdb.禅宗从旧课程继承配置('d260104禅宗11期一阶', self.course_name)

    def 配置打卡数据(self):
        course_name = self.course_name
        # self.xe2.switch_shop('宗门学府')
        self.update_clockin(f'{course_name}-共学打卡',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_68609f4cb98c5_pOwF6g&community_id=c_68600a2f51cb6_Q4UahbvQ7554&clock_id=ac_69da8156cc895_Lqqkgz5I&group_id=group_69da815736ae7_sMv6Xx1e&taskId=th_69da815732d26_1PhjQ4to&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data',
                            download=False)

        self.update_clockin(f'{course_name}-共修打卡-随念门',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_686a5943a0976_IgjvcX&community_id=c_68600a2f51cb6_Q4UahbvQ7554&clock_id=ac_69da81f4a7628_xc9gjhRY&group_id=group_69da81fc26adc_2TSezhKi&taskId=th_69da81fc26b35_KBYrd5ev&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data',
                            download=False)

        self.update_clockin(f'{course_name}-共修打卡-忏悔门',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_686a586fe7471_sYYRTP&community_id=c_68600a2f51cb6_Q4UahbvQ7554&clock_id=ac_69da81c53bae9_peSjaa5N&group_id=group_69da81d0a4aea_vIHWasSQ&taskId=th_69da81d0a4b47_RYe5uLuV&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data',
                            download=False)

        self.update_clockin(f'{course_name}-共修打卡-崇拜祈祝',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_68609f8f9cb04_0xficG&community_id=c_68600a2f51cb6_Q4UahbvQ7554&clock_id=ac_69da8194efcb6_Dh6087s5&group_id=group_69da81956be68_IZakwAiZ&taskId=th_69da819566ca4_DpTbcYW4&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data',
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
        # kq.从旧课程数据迁移配置()
        # kq.配置打卡数据()

        kq.status = 0
        kq.step1()
        # kq.step2()
        # kq.step3()
        # kq.step4()
        # kq.step5()

        # main_b()
