from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    早期共修来源课程 = 'd251130禅宗7期4点5阶'
    早期共修开始时间 = '2026-03-08'
    早期共修结束时间 = '2026-03-15'

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'cjEE1jEybxRO',
                         'V2-78OlydSkG9LMdZ9832LmxA',
                         课程商品名='4.5阶',
                         )

    @staticmethod
    def _sheet_count(value):
        if value is None or value == '':
            return 0
        try:
            if pd.isna(value):
                return 0
        except TypeError:
            pass
        return int(float(value))

    @staticmethod
    def _format_count(value):
        return '' if not value else int(value)

    def _读取早期共修实际数量(self, user_id2s):
        """读取开课初期仍落在7期4.5阶配置下的8期共修原始条数。"""
        base = pd.DataFrame({'user_id2': user_id2s})
        df = self.kqdb.exec2df(
            """
            SELECT user_id2, COUNT(*) AS 早期共修打卡
            FROM clockin_data_table
            WHERE clockin_name=%s
              AND publish_time >= %s
              AND publish_time < %s
              AND groupname LIKE %s
            GROUP BY user_id2
            """,
            params=[
                f'{self.早期共修来源课程}-共修打卡',
                self.早期共修开始时间,
                self.早期共修结束时间,
                '%8期%',
            ],
        )
        if df.empty:
            base['早期共修打卡'] = 0
            return base
        return base.merge(df, on='user_id2', how='left').fillna({'早期共修打卡': 0})

    def _补入早期共修实际数量(self, df, user_id2s):
        extra = self._读取早期共修实际数量(user_id2s)
        merged = df.merge(extra, on='user_id2', how='left')
        if '共修打卡' not in merged:
            merged['共修打卡'] = ''
        merged['共修打卡'] = [
            self._format_count(self._sheet_count(current) + self._sheet_count(extra_value))
            for current, extra_value in zip(merged['共修打卡'], merged['早期共修打卡'])
        ]
        return merged.drop(columns=['早期共修打卡'])

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

            df_filtered = df.copy()

            if df_filtered.empty:
                return df_filtered

            # 分成两组：共学打卡 和 共修打卡；共修按实际打卡数量统计，不按周去重。
            gongxue_mask = df_filtered['clockin_name'].str.contains('共学打卡', na=False)
            gongxiu_mask = df_filtered['clockin_name'].str.contains('共修打卡', na=False)

            gongxue_df = df_filtered[gongxue_mask].copy()
            gongxiu_df = df_filtered[gongxiu_mask].copy()

            # 合并数据
            result_df = pd.concat([gongxue_df, gongxiu_df], ignore_index=True)

            # 按 publish_time 排序
            result_df = result_df.sort_values('publish_time').reset_index(drop=True)

            return result_df

        df1 = self.browser_clockin_data(f'{self.course_name}-', None,
                                        user_id2s=user_id2s, filter=filter)
        df1 = self._补入早期共修实际数量(df1, user_id2s)

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
        old_course_name = 'd251130禅宗7期4点5阶'
        self.kqdb.禅宗从旧课程继承配置(old_course_name, self.course_name)
        # 迁移完后，可以刻意把end_date再往后偏移几个月，课程实际结束的时候再手动把结束时间改了


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

        # kq.status = 0
        # kq.step1()
        # kq.step2()
        # kq.step3()
        # kq.step4()
        # kq.step5()

        # kq.配置课次数据()

        main_b()
