from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'ceMaIfbUSeer',
                         'V2-78OlydSkG9LMdZ9832LmxA',
                         课程商品名='4.5阶',
                         )

    def step1(self, update=True, *, status=1):
        if self.get_status() >= status:
            return
        logger.info('1 下载小鹅通数据')
        if update:
            # 更新课程和打卡数据
            if self.shop_id == 1:
                self.xe2.switch_shop('5034山中薪')
                self.update_lesson_data_table(shop1=True)
                self.update_clockin(f'{self.course_name}-*')
            elif self.shop_id == 2:
                self.xe2.switch_shop('宗门学府')
                self.update_lesson_data_table(shop2=True)
                self.update_clockin(f'{self.course_name}-*')

                # + 插曲，第3周全员补打卡数据
                self.step1_打卡补丁()

            logger.info(f'"{self.course_name}"下载完小鹅通数据')
        self.set_status(status)

    def step1_打卡补丁(self):
        # 1 第3周要补全员打卡数据
        user_id2s = self.wb_get_column_list('用户ID')  # 为了实验，先只运行5个人
        clockins = [
            # clockin_id, name, 生成打卡数
            [140, 'd250831禅宗46期4点5阶-共学打卡', 1],
            [141, 'd250831禅宗46期4点5阶-共修打卡', 1],
            [142, 'd250831禅宗46期4点5阶-共修实际打卡总数', 3],
        ]
        for clockin_id, clockin_name, times in clockins:
            rows = []
            for user_id2 in user_id2s:
                row = [clockin_id, clockin_name, user_id2, '2025-09-20 12:00']
                rows += [row] * times
            self.kqdb.insert_rows('clockin_data_table',
                                  'clockin_id,clockin_name,user_id2,publish_time',
                                  rows)
            self.kqdb.commit_all()

        # 2 10月25日手动额外补部分打卡数据
        content = """u_66d7a3e677fcf_scam19c1Tu	0	0	0	0	0	0
u_66d429aa470d8_J04oCjiSM0	0	2	2	1	1	1
u_645d91722f43e_pj29BhnY5B	0	0	0	0	0	0
u_6541c6805d6b5_PWdvwqDrWx	0	3	3	2	2	2
u_66d410fcbb21d_RN1gdHkBQW	0	0	0	0	0	0
u_66d6b2a576242_jIwLULQUha	0	1	1	1	1	1
u_642e7c67392f7_8MN8q0mubA	0	0	0	0	0	0
u_66d3c1b97508e_xzI71TdV1H	0	2	2	1	2	2
u_65bb107c61b0a_NbXV0LTePY	0	0	0	0	0	0
u_658a98f942d26_LqWGmvfGuz	0	1	1	0	1	1
u_65c6456182b55_AxdgJrbTeH	0	0	0	0	0	0
u_642e77fe5aad2_sfgkJdmGXs	0	0	0	0	0	0
u_64eff6f9e7ce7_EzI41686iX	3	3	3	3	3	3
u_66d3e1af73dcf_Ra2QZSRQWg	0	0	0	0	0	0
u_66d3bda673977_C2rzUqJBDW	0	2	2	1	3	3
u_65379a1a8577e_svMOYzCAAO	0	2	2	1	2	2
u_66d552846805d_xdicjpOqW6	0	0	0	0	0	0
u_64e92e1f2653d_VXaivJzc3D	0	0	0	0	0	0
u_65312cde159fd_PHkUyHUNwP	0	3	3	1	2	2
u_66d42b8547e2b_6MrYaoF04u	0	1	1	1	2	2
u_642e828369eff_a8KnSPNWeH	0	0	0	0	0	0
u_66d3be2f75021_TVMGVBz0yH	0	0	0	1	0	0
u_673b19db7b2a4_uDpkppfEST	0	0	0	0	0	0
u_673b19523a7fa_LprgNHRcuH	0	0	0	1	1	1
u_661b35345800b_QzljTE30yD	0	0	0	0	0	0
u_642e767b5aa74_ZjUpOW5Vkr	0	0	0	0	0	0
u_653122c91de12_l7paJMwebW	0	1	1	1	2	2
u_66d4110f5a2ee_m2oKeLIDIV	1	2	2	0	1	1
u_66d6ad813877c_iJD08HTFzD	0	0	0	1	1	1
u_644d144c9c2c5_ZFkuAPKzxO	0	0	0	1	1	1
u_66d3c546750fa_KoYggLQOMv	0	0	0	1	1	1
u_66d47e944a580_czbzN2lcSa	0	2	2	2	3	3
u_66d3be6d73993_ytOcaF5wJR	0	0	0	1	3	3
u_66d3e04273da5_bMiCo5PuNL	1	1	1	1	2	2
u_66d4110f5a2f5_0JP2VQF3Ym	0	2	2	1	4	4
u_64e8a2d651c43_hZRHYf3Iwc	1	2	2	1	2	2
u_67d7ce4c47e83_gFRMzDAdb0	0	0	0	0	0	0
u_68b4bf6f16467_4mrA6s6xUZ	0	2	2	1	1	1
u_66d4110f5a2f2_0ElqcKcdgj	0	1	1	1	1	1
u_66d3c6b775125_EEC0OWV8qX	0	2	2	0	0	0
u_66d9a3871abb7_sLBw1w7YKQ	0	0	0	0	0	0
u_66d3d5cb73c72_RPdwPKg2Nz	0	2	2	1	1	1
u_64391a207645b_kulDkucZNz	2	2	2	4	4	4
u_66d6650e75a97_gci7pBAZ5W	0	0	0	0	0	0
u_65f4c6086f4a7_SjegdmeHyu	0	2	2	1	1	1
u_66d3d4d0752e9_zA1XMfGgPC	0	0	0	0	0	0
u_64eff779ce708_boAeJIcslw	0	1	1	1	2	2
u_673b1a183a866_BnvRanEpVb	1	2	2	0	0	0
u_673b194060a51_sejQRoCTVs	0	0	0	0	0	0"""
        # 前3列补10月17日数据，后3列补10月19日数据。第0列是user_id2。每3列一组的数据正好对应clockin_id的3个
        lines = content.strip().split('\n')
        for line in lines:
            parts = line.split('\t')
            user_id2 = parts[0]

            # 前3列补10月17日数据
            date1 = '2025-10-17 12:00'
            for i, (clockin_id, clockin_name, _) in enumerate(clockins):
                count = int(parts[i + 1])
                if count > 0:
                    rows = [[clockin_id, clockin_name, user_id2, date1]] * count
                    self.kqdb.insert_rows('clockin_data_table',
                                          'clockin_id,clockin_name,user_id2,publish_time',
                                          rows)

            # 后3列补10月19日数据
            date2 = '2025-10-19 12:00'
            for i, (clockin_id, clockin_name, _) in enumerate(clockins):
                count = int(parts[i + 4])
                if count > 0:
                    rows = [[clockin_id, clockin_name, user_id2, date2]] * count
                    self.kqdb.insert_rows('clockin_data_table',
                                          'clockin_id,clockin_name,user_id2,publish_time',
                                          rows)

        self.kqdb.commit_all()

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

            # 过滤掉小于 2025-08-31 00:00 的数据
            start_date = pd.Timestamp('2025-08-31 00:00:00')
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
                                        user_id2s=user_id2s, filter=filter)

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
        self.update_clockin('d250831禅宗1期4点5阶-共学打卡',
                            url='')
        self.update_clockin('d250831禅宗1期4点5阶-共修打卡',
                            url='')

    def 爬虫检查课程目录(self):
        url = ('https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course?'
               'communityId=c_6824b2a916fe1_jT7cZ3pb4838'
               '&course_id=course_2x5l8x8SwyhufMjaOr93Jyqj2FK&type=manage')
        course = self.xe2.爬虫获得禅宗课程目录(url, start_name='第15周')
        print(course)
        # return course

        # 临时的第16周数据情况比较特别，要AI魔改下后处理
        # 我需要key改成第16周，只留最后一条，'佛教概观4-文化现象2/2-惟海法师-20240831PM1835'改成'佛教概观4'

        # 1. 获取原始列表（安全起见用 .get 防止报错）
        original_list = course.get('第15周', [])

        # 2. 确保列表不为空，然后提取最后一条
        if original_list:
            last_item = original_list[-1]  # 取出最后一个列表项

            # last_item 结构是 [原名称, URL]
            original_url = last_item[1]

            # 3. 修改名称
            # 方式一：直接指定（如果你确定就是这个名字）
            new_name = '佛教概观4'

            # 方式二：自动截取（如果你想自动去掉“-”后面的内容，推荐这种）
            # new_name = last_item[0].split('-')[0]

            # 4. 组装新的 course2
            course2 = {
                '第16周': [
                    [new_name, original_url]
                ]
            }
        else:
            print("未找到第15周的数据")
            course2 = {}

        # ================== 打印结果验证 ==================
        print(course2)
        return course2

    def 配置课次数据(self):
        course = self.爬虫检查课程目录()
        # 实际只有16周，但是监控应该至少多加1周捕捉时间
        self.kqdb.添加禅宗课次配置数据(self.course_name, 16 + 1, course)


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
    kq.step4()  # 课程准备结束，不用显式返款了
    kq.step5()
    # kq.step6()


if __name__ == '__main__':
    os.chdir(get_xl_homedir())
    if len(sys.argv) > 1:
        fire.Fire()
    else:
        kq = 考勤课程()
        # kq.爬虫检查课程目录()
        # kq.配置课次数据()

        # kq.status = 0
        # kq.step1()
        # kq.step2()
        # kq.step3()
        # kq.step4()
        # kq.step5()

        # kq.配置课次数据()

        main_b()
