from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'cbFs9XMaCgcF',
                         'V2-78OlydSkG9LMdZ9832LmxA',
                         课程商品名='4.5阶',
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
        def filter(df):
            # 确保 publish_time 是 datetime 类型
            df['publish_time'] = pd.to_datetime(df['publish_time'])

            # 过滤掉小于日期的数据
            start_date = pd.Timestamp('2025-06-15 00:00:00')
            df_filtered = df[df['publish_time'] >= start_date].copy()

            if df_filtered.empty:
                return df_filtered

            # 分成两组：共学打卡 和 共修打卡
            gongxue_mask = df_filtered['clockin_name'].str.contains('共学打卡', na=False)
            gongxiu_mask = df_filtered['clockin_name'].str.contains('共修打卡', na=False)
            gongxiu2_mask = df_filtered['clockin_name'].str.contains('共修实际打卡', na=False)

            # 共学打卡：只需要过滤时间，不需要其他处理
            gongxue_df = df_filtered[gongxue_mask].copy()

            # 共修打卡：需要按用户和7天周期去重
            gongxiu_df = df_filtered[gongxiu_mask].copy()
            gongxiu2_df = df_filtered[gongxiu2_mask].copy()

            if not gongxiu_df.empty:
                # 为共修打卡数据计算所属的周期（从2025-08-31开始，每7天为一个周期）
                def get_week_group(timestamp):
                    days_since_start = (timestamp - start_date).days
                    return days_since_start // 7

                gongxiu_df['week_group'] = gongxiu_df['publish_time'].apply(get_week_group)

                # 按 user_id2 和 week_group 分组，每组只保留一条记录
                # 这里保留每组中 publish_time 最早的一条记录
                gongxiu_df = gongxiu_df.groupby(['user_id2', 'week_group']).first().reset_index()

                # 删除辅助列
                gongxiu_df = gongxiu_df.drop('week_group', axis=1)

            # 合并数据
            result_df = pd.concat([gongxue_df, gongxiu_df, gongxiu2_df], ignore_index=True)

            # 按 publish_time 排序
            result_df = result_df.sort_values('publish_time').reset_index(drop=True)

            return result_df

        df1 = self.browser_clockin_data(f'{self.course_name}-', None,
                                        user_id2s=user_id2s)

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
        name = re.search(r'\d+期.+?阶', self.course_name).group().replace('点', '.')
        wechat_lock_send('禅宗修道考勤管理', self.get_daily(f'禅宗修道普及班{name}中心教室'))
        self.set_status(6)

    def 配置打卡数据(self):
        self.update_clockin(f'{self.course_name}-共学打卡',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_68454a6be3b68_0ewWIx&community_id=c_6824b2a916fe1_jT7cZ3pb4838&clock_id=ac_6845521b863e9_40guYEFG&group_id=group_68455225a2df0_Ro7J1H0x&taskId=th_68455225a2e4e_uAYvQbmV&beginDate=101&totalDay=100&is_combine_task=100&management_entry_id=calendar_clock_management&component_name=clock_task_data')
        self.update_clockin(f'{self.course_name}-共修打卡',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_6845525bdd131_8EdUBr&community_id=c_6824b2a916fe1_jT7cZ3pb4838&clock_id=ac_68455273be7f1_gqyn530D&group_id=group_6845528420dd4_Drb2IVjH&taskId=th_6845528420e29_9xMMluNj&beginDate=101&totalDay=100&is_combine_task=100&management_entry_id=calendar_clock_management&component_name=clock_task_data')
        self.update_clockin(f'{self.course_name}-共修实际打卡总数',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_6845525bdd131_8EdUBr&community_id=c_6824b2a916fe1_jT7cZ3pb4838&clock_id=ac_68455273be7f1_gqyn530D&group_id=group_6845528420dd4_Drb2IVjH&taskId=th_6845528420e29_9xMMluNj&beginDate=101&totalDay=100&is_combine_task=100&management_entry_id=calendar_clock_management&component_name=clock_task_data')

    def 重置_课次开始时间(self):
        """ d251009，应该基于每个课次设置开始时间，而不是整个禅宗课程，因此也需要先修改数据库中的数据 """
        from datetime import datetime, timedelta
        import re

        kqdb = self.kqdb  # 数据库连接对象
        # 目标表格：lesson_table
        # 检索字段：lesson_name，需要含有"禅宗"字眼
        #   用正则找'第(\d+)周'，比如'第3周'就设week=3
        # 然后读取start_date字段的日期，加上 timedelta(weeks=week-1)
        # 加个过滤条件 start_date>=2025-06-22

        # 查询符合条件的课程数据
        lessons = kqdb.exec2dict("""
            SELECT * FROM lesson_table 
            WHERE lesson_name LIKE '%禅宗%' AND start_date >= '2025-06-22'
        """)

        for lesson in lessons:
            # 从lesson_name中提取周数
            match = re.search(r'第(\d+)周', lesson['lesson_name'])
            if match:
                week = int(match.group(1))
                # 解析原开始日期
                start_date = datetime.strptime(lesson['start_date'], '%Y-%m-%d %H:%M:%S') if isinstance(
                    lesson['start_date'], str) else lesson['start_date']
                # 计算新的开始日期
                new_start_date = start_date + timedelta(weeks=week - 1)
                # 更新数据库中的开始日期
                kqdb.update_row('lesson_table',
                                {'start_date': new_start_date.strftime('%Y-%m-%d %H:%M:%S')},
                                {'lesson_id': lesson['lesson_id']})

        kqdb.commit()

    def 从旧课程数据迁移配置(self):
        self.kqdb.禅宗从旧课程继承配置('d250928禅宗3期4点5阶', self.course_name)

    def 从旧课程拷贝数据(self):
        self.kqdb.禅宗从旧课程拷贝数据('d250831禅宗46期4点5阶', self.course_name)

    def 修改数据时间(self):
        """ 实测不行，算出来全是延期数据，必须都改成当周

        操作流程
        1、lesson_table，找出lesson_name LIKE f'{self.course_name}-%'的数据
        2、对找到的每条lesson数据，在less_data_table里，把对应的update_time都改了
        3、改成lesson里的start_date加上7天10分钟
        """
        kqdb = self.kqdb  # 数据库连接对象

        # 1. 找出lesson_name LIKE f'{self.course_name}-%'的数据
        lessons = kqdb.exec2dict(f"""
            SELECT * FROM lesson_table 
            WHERE lesson_name LIKE '{self.course_name}-%'
        """)

        for lesson in lessons:
            # 解析lesson里的start_date
            start_date = datetime.strptime(lesson['start_date'], '%Y-%m-%d %H:%M:%S') if isinstance(
                lesson['start_date'], str) else lesson['start_date']
            # 计算新的时间：start_date加上7天10分钟
            new_time = start_date + timedelta(days=7, minutes=10)

            # 2. 在less_data_table里，把对应的update_time都改了
            kqdb.update_row('lesson_data_table',
                            {'update_time': new_time.strftime('%Y-%m-%d %H:%M:%S')},
                            {'lesson_id': lesson['lesson_id']})

        kqdb.commit()


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
        # kq.重置_课次开始时间()
        # kq.从旧课程数据迁移配置()
        # kq.从旧课程拷贝数据()
        # kq.配置打卡数据()
        # kq.修改数据时间()

        kq.status = 0
        # kq.step1()
        # kq.修改数据时间()
        # kq.step2()
        # kq.step3()
        kq.step4()
        kq.step5()

        # kq.配置课次数据()

        # main_b()
