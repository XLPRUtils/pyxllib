from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'cnKsOsLnrhbh',
                         'V2-4ihfz275sveOiNjWQChw3G',
                         课程商品名='一阶',
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
        df1.loc[df1['user_id2'] == 'u_686097ef3e45c_IRtTLGIAyi', '共修打卡-崇拜祈祝'] += 1

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
               'communityId=c_68600a2f51cb6_Q4UahbvQ7554'
               '&course_id=course_2z8tWHRoKCvzUfD8KkRmdcrC2JL&type=manage')
        course = self.xe2.爬虫获得禅宗课程目录(url)

        courses = {}
        for k, v in course.items():
            x = re.search(r'\d+', k)
            if x:
                week = int(x.group())
                # course字典，只保留第5周开始的数据，即第5~9周
                # if week >= 5:
                #     courses[k] = v
                courses[k] = v

        print(courses)
        return courses

    def 配置课次数据(self):
        # 1 复制爬虫得到的字典，并手动调整下每节课的名称
        # d250719，已有的课程最好注释掉，不然会重置next_update导致一些更新时间点问题
        courses = {
            # '第1周':
            #     [['佛教概观1-1', 92,
            #       'https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page?course_id=course_2z8tWHRoKCvzUfD8KkRmdcrC2JL&resource_id=v_68600ae3e4b0694ca0d20a18&p_id=chap_2z8tXXCXyZO2cwqbgsbmdZ9yPwE&type=3&communityId=c_68600a2f51cb6_Q4UahbvQ7554&sub_course_id='],
            #      ['法门通论-导论1', 69,
            #       'https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page?course_id=course_2z8tWHRoKCvzUfD8KkRmdcrC2JL&resource_id=v_68600ae3e4b0694ca0d20a18&p_id=chap_2z8tXXCXyZO2cwqbgsbmdZ9yPwE&type=3&communityId=c_68600a2f51cb6_Q4UahbvQ7554&sub_course_id=']],
            # '第2周':
            #     [['佛教概观1-2', 91,
            #       'https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page?course_id=course_2z8tWHRoKCvzUfD8KkRmdcrC2JL&resource_id=v_68600b3de4b0694c5af59b38&p_id=chap_2z8trDQGArM9vOsGOF1AtD2muIp&type=3&communityId=c_68600a2f51cb6_Q4UahbvQ7554&sub_course_id='],
            #      ['法门通论-导论2', 60,
            #       'https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page?course_id=course_2z8tWHRoKCvzUfD8KkRmdcrC2JL&resource_id=v_68600b3de4b0694c5af59b38&p_id=chap_2z8trDQGArM9vOsGOF1AtD2muIp&type=3&communityId=c_68600a2f51cb6_Q4UahbvQ7554&sub_course_id='],
            #      ['佛教概观1-3', 80,
            #       'https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page?course_id=course_2z8tWHRoKCvzUfD8KkRmdcrC2JL&resource_id=v_68600b3de4b0694c5af59b38&p_id=chap_2z8trDQGArM9vOsGOF1AtD2muIp&type=3&communityId=c_68600a2f51cb6_Q4UahbvQ7554&sub_course_id=']],
            # '第3周':
            #     [['佛教概观4',
            #       'https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page?course_id=course_2z8tWHRoKCvzUfD8KkRmdcrC2JL&resource_id=v_6868c8a7e4b0694ca0d7e200&p_id=chap_2zRcjJqVw6ZidggIjDSPH925htS&type=3&communityId=c_68600a2f51cb6_Q4UahbvQ7554&sub_course_id='],
            #      ['法门通论-崇拜祈祝门',
            #       'https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page?course_id=course_2z8tWHRoKCvzUfD8KkRmdcrC2JL&resource_id=v_6873e1c6e4b0694ca0e007de&p_id=chap_2zRcjJqVw6ZidggIjDSPH925htS&type=3&communityId=c_68600a2f51cb6_Q4UahbvQ7554&sub_course_id='],
            #      ['礼仪文化1',
            #       'https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page?course_id=course_2z8tWHRoKCvzUfD8KkRmdcrC2JL&resource_id=v_6868c8a7e4b0694c5afb6aab&p_id=chap_2zRcjJqVw6ZidggIjDSPH925htS&type=3&communityId=c_68600a2f51cb6_Q4UahbvQ7554&sub_course_id='],
            #      ['礼仪文化2',
            #       'https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page?course_id=course_2z8tWHRoKCvzUfD8KkRmdcrC2JL&resource_id=v_6868c8a7e4b0694ca0d7e202&p_id=chap_2zRcjJqVw6ZidggIjDSPH925htS&type=3&communityId=c_68600a2f51cb6_Q4UahbvQ7554&sub_course_id=']],
            '第4周':
                [['《佛教史》史论',
                  'https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page?course_id=course_2z8tWHRoKCvzUfD8KkRmdcrC2JL&resource_id=v_686a880be4b0694ca0d8eecd&p_id=chap_2zVMJl6JfcQWZWqsQhvCh0OmKsI&type=3&communityId=c_68600a2f51cb6_Q4UahbvQ7554&sub_course_id='],
                 ['吉祥偈唱诵',
                  'https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page?course_id=course_2z8tWHRoKCvzUfD8KkRmdcrC2JL&resource_id=v_687d21dbe4b0694ca0e74508&p_id=chap_2zVMJl6JfcQWZWqsQhvCh0OmKsI&type=3&communityId=c_68600a2f51cb6_Q4UahbvQ7554&sub_course_id='],
                 ['礼仪文化3',
                  'https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page?course_id=course_2z8tWHRoKCvzUfD8KkRmdcrC2JL&resource_id=v_686a8701e4b0694ca0d8ee79&p_id=chap_2zVMJl6JfcQWZWqsQhvCh0OmKsI&type=3&communityId=c_68600a2f51cb6_Q4UahbvQ7554&sub_course_id='],
                 ['礼仪文化4',
                  'https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page?course_id=course_2z8tWHRoKCvzUfD8KkRmdcrC2JL&resource_id=v_686a8702e4b0694c5afc620a&p_id=chap_2zVMJl6JfcQWZWqsQhvCh0OmKsI&type=3&communityId=c_68600a2f51cb6_Q4UahbvQ7554&sub_course_id='],
                 ]
        }

        courses = self.爬虫检查课程目录()

        # 2 调用底层通用处理接口
        self.kqdb.添加禅宗课次配置数据(self.course_name, 9, courses)

    def 配置打卡数据(self):
        course_name = self.course_name
        self.update_clockin(f'{course_name}-共学打卡',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_68609f4cb98c5_pOwF6g&community_id=c_68600a2f51cb6_Q4UahbvQ7554&clock_id=ac_68609f6f5c9e3_lYmUwVP3&group_id=group_68609f7d69170_vqx2hsCc&taskId=th_68609f7d691d4_tIqO62eV&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data')

        self.update_clockin(f'{course_name}-共修打卡',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_68609f8f9cb04_0xficG&community_id=c_68600a2f51cb6_Q4UahbvQ7554&clock_id=ac_68609fb76717b_uyujhvmR&group_id=group_68609fc485985_yyKHwSGq&taskId=th_68609fc4859da_48WQBgU5&beginDate=64&totalDay=63&is_combine_task=63&management_entry_id=calendar_clock_management&component_name=clock_task_data')

        self.update_clockin(f'{course_name}-共修打卡-忏悔门',
                            url='')

        self.update_clockin(f'{course_name}-共修打卡-崇拜祈祝',
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
        # kq.配置课次数据()

        # kq.status = 0
        # kq.step1()
        # kq.step2()
        # kq.step3()

        main_b()
