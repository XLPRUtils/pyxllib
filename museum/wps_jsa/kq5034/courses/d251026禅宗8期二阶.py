from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'chO73qp5jxk9',
                         'V2-34ZicIDNmuN44rHJzWkioA',
                         课程商品名='二阶'
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
        if self.get_status() >= 6:
            return
        logger.info('6 发送日报')
        name = re.search(r'\d+期.+?阶', self.course_name).group()
        wechat_lock_send('线上修道班考勤管理', self.get_daily(f'禅宗修道普及班{name}中心教室'))
        self.set_status(6)

    def 从旧课程数据迁移配置(self):
        self.kqdb.禅宗从旧课程继承配置('禅宗7期2,3阶', self.course_name)

    def 配置打卡数据(self):
        course_name = self.course_name
        self.update_clockin(f'{course_name}-共学打卡',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_655d82977a330_FhyvxN&community_id=c_64b399e7bcc74_j112kTRI4127&clock_id=ac_68600744bba12_ewVsiK5K&group_id=group_6860074547827_0tVhY7NT&taskId=th_686007453fb9e_Y4nLQwXv&beginDate=93&totalDay=92&is_combine_task=92&management_entry_id=calendar_clock_management&component_name=clock_task_data')
        self.update_clockin(f'{course_name}-共修打卡-闻思门',
                            url='')
        self.update_clockin(f'{course_name}-共修打卡-人天福德门',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_655d83440cc91_cUzkHC&community_id=c_64b399e7bcc74_j112kTRI4127&clock_id=ac_686007c74f7f0_TtK21fuu&group_id=group_686007c7a1037_evzIpFID&taskId=th_686007c79a5a1_bRr9cpTf&beginDate=93&totalDay=92&is_combine_task=92&management_entry_id=calendar_clock_management&component_name=clock_task_data')
        self.update_clockin(f'{course_name}-共修打卡-禅门',
                            url='')
        self.update_clockin(f'{course_name}-共修打卡-日常修行',
                            url='')
        self.update_clockin(f'{course_name}-共修打卡-诵经',
                            url='')
        self.update_clockin(f'{course_name}-共修打卡-安般禅',
                            url='')


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
        # kq = 考勤课程()
        # kq.status = 0
        # kq.step3()
        # kq.step4()
        # kq.step5()
        # kq.step6()

        # kq.配置打卡数据()

        main_b()
