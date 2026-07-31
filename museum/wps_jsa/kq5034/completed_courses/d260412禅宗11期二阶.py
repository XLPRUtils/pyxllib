from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'cpHrvzZyzjiS',
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
        # 680: 李杰 04/20 阿含诵 Day1 前台已完成，但旧 PG 导出缺失，临时补 1 次。
        df1.loc[df1['user_id2'] == 'u_695957138b74b_xU07nvZyag', '共修打卡-阿含诵'] += 1
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
        self._ensure_previous_steps_completed(6, 6)
        if self.get_status() >= 6:
            return
        logger.info('6 发送日报')
        name = re.search(r'\d+期.+?阶', self.course_name).group()
        wechat_lock_send('线上修道班考勤管理', self.get_daily(f'禅宗修道普及班{name}中心教室'))
        self.set_status(6)

    def 从旧课程数据迁移配置(self):
        raise RuntimeError('d260412禅宗11期二阶的视频课次请改用 配置课次数据() 爬取真实目录')

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
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_68fcd7fe73b1f_hdnHh1&community_id=c_68fcd4f92ba7b_yx91ZF6s3843&clock_id=ac_69da822293da9_SDw2avRl&group_id=group_69da8222d6b0f_jTqhhyXu&taskId=th_69da8222d1e58_Bai3cLCr&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data',
                            download=False)
        self.update_clockin(f'{course_name}-共修打卡-持诵门',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_68fcd8c0be576_LIGSIe&community_id=c_68fcd4f92ba7b_yx91ZF6s3843&clock_id=ac_69da824d73425_D233q81R&group_id=group_69da824dde138_Raa49Zgz&taskId=th_69da824dd9659_73VmOcRY&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data',
                            download=False)
        self.update_clockin(f'{course_name}-共修打卡-阿含诵',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_69046e48d0ee4_mdYiOk&community_id=c_68fcd4f92ba7b_yx91ZF6s3843&clock_id=ac_69da82aa13d97_rRF3SASA&group_id=group_69da82aa693bf_xipgyDnd&taskId=th_69da82aa65230_bWv6C6gE&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data',
                            download=False)
        self.update_clockin(f'{course_name}-共修打卡-人天福德门',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_69046ddca2cc6_wPsoDI&community_id=c_68fcd4f92ba7b_yx91ZF6s3843&clock_id=ac_69da827a8b5cb_VVbYunbB&group_id=group_69da827ade3b1_dCgCXAKm&taskId=th_69da827ad9f6b_XSJv6njh&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data',
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
