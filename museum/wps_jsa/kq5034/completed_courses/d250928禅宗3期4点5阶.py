from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'ch5qvHDFKYUc',
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
            start_date = pd.Timestamp('2025-10-05 00:00:00')
            # df_filtered = df[df['publish_time'] >= start_date].copy()
            df_filtered = df.copy()

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
                                        user_id2s=user_id2s, filter=filter)
        # 统一额外加6次打卡
        for col in df1.columns:
            if col != 'user_id2':
                # 将空字符串转换为0，再统一加6
                df1[col] = pd.to_numeric(df1[col], errors='coerce').fillna(0) + 6
        df1.loc[df1['user_id2'] == 'u_6433e6f23a7e1_WcuHdmt6fX', '共学打卡'] += 1

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
        # self.update_clockin(f'{self.course_name}-共学打卡',
        #                     url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_68454a6be3b68_0ewWIx&community_id=c_6824b2a916fe1_jT7cZ3pb4838&clock_id=ac_68b9fb0a45abf_IbNnMg41&group_id=group_68b9fb0a8e8a9_4gHH6Iha&taskId=th_68b9fb0a86b21_80HxOgGh&beginDate=87&totalDay=86&is_combine_task=86&management_entry_id=calendar_clock_management&component_name=clock_task_data')
        # self.update_clockin(f'{self.course_name}-共修打卡',
        #                     url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_6845525bdd131_8EdUBr&community_id=c_6824b2a916fe1_jT7cZ3pb4838&clock_id=ac_68b9fae4b25a7_Ya8q2ckJ&group_id=group_68b9fae523967_Ei0j8Qcd&taskId=th_68b9fae51baa6_tAKnetGL&beginDate=87&totalDay=86&is_combine_task=86&management_entry_id=calendar_clock_management&component_name=clock_task_data')
        self.update_clockin(f'{self.course_name}-共修实际打卡总数',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_6845525bdd131_8EdUBr&community_id=c_6824b2a916fe1_jT7cZ3pb4838&clock_id=ac_68b9fae4b25a7_Ya8q2ckJ&group_id=group_68b9fae523967_Ei0j8Qcd&taskId=th_68b9fae51baa6_tAKnetGL&beginDate=87&totalDay=86&is_combine_task=86&management_entry_id=calendar_clock_management&component_name=clock_task_data')

    def 从旧课程数据迁移配置(self):
        self.kqdb.禅宗从旧课程继承配置('d250831禅宗46期4点5阶', self.course_name)


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
        # kq.从旧课程数据迁移配置()
        # kq.配置打卡数据()

        kq.status = 0
        # kq.step1()
        # kq.step2()
        # kq.step3()
        kq.step4()
        kq.step5()
        kq.step6()

        # kq.配置课次数据()

        # main_b()
