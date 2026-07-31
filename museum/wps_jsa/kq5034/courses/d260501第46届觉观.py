from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(1,
                         XlPath(__file__).stem,
                         'coU9I2vqb5nK',
                         'V2-2AbgKCjFgq3p2fOaEOgKvg',
                         5,
                         课程商品名='第46届—觉观技术公益网课【中心教室】')

    def step2(self):
        """ 每个课程主要的差异在这里 """
        if self.get_status() >= 2:
            return
        logger.info('2 从数据库xldb3获取新的考勤数据并写回在线表格')

        # 250203周一，"一人多账号"问题，需要使用合并数据大法~
        # self.kqdb.merge_user('u_6952791a1a9ed_MKbbDswYX3', 'u_69360f6f8daf6_69MrVkc2aK')  # 3-14 王福巧

        # 1 获取用户清单
        self.更新用户匹配()
        self.把报名表的用户ID同步到考勤表()

        user_id2s = self.wb_get_column_list('用户ID')

        # 2 拼接应该写入的数据
        # 打卡数据
        titles = [f'【打卡】中心教室-{i}' for i in range(1, 23)]
        df1 = self.kqdb.browser_clockin_data(f'{self.course_name}-', ['打卡数'],
                                             user_id2s=user_id2s, titles=titles)

        # 修正对应user_id2的考勤数据
        # df1.loc[df1['user_id2'] == 'u_6773ff3e6fdec_Wv1NqEcSbD', '打卡数'] = 15

        # 视频数据
        df2 = self.browser_lesson_data(f'{self.course_name}-', user_id2s=user_id2s)
        # 删掉带有"测试"的数据列，是开课前的"测试1"、"测试2“
        df2 = df2.loc[:, ~df2.columns.str.contains('测试')]

        # 处理国外学生
        # self.shift_international_students(df2, ['u_692ab72eccdcc_iBjgAtf3o3', ])

        # 拼接数据
        df3 = self.拼接打卡视频数据(user_id2s, df1, df2)

        # 3 把数据写入表格
        self.write_rows_skip_empty(df3)

        logger.info('已将最新考勤数据文本写入在线表格')
        self.set_status(2)

    def step6(self):
        if self.get_status() >= 6:
            return
        logger.info('6 发送日报')
        wechat_lock_send('考勤中台', '@陈成')
        wechat_lock_send('考勤中台', self.get_daily(f'5034第{self.course_name[8:10]}届觉观网课山中薪中心教室'))
        self.set_status(6)

    def 开课配置(self):
        """
        文档：https://www.yuque.com/xlpr/pyxllib/nrg8piay6mt95rg0
        """
        # 0 在线表格
        # 拷贝最近的类似课程，比如9月梵呗初阶从7月梵呗初阶拷贝，另存为新表格
        # 复制表格url，配置到这个类初始化参数中
        # 表格替换配置最新的kqcourse.js内容

        # 1 爬虫获得课程清单
        # 总表：https://www.kdocs.cn/l/cguYugQWIRs1，Sheet"课次数据"，L2填写prefix="2509"，执行
        # 执行结果lesson_name改成"d250909梵呗初阶-第01课"， ...，一共11课
        # 简单检查下start_date、end_date是否对齐

        # 然后把配置导入数据库
        # self.add_book_lessons_to_db()

        # 2 DBeaver，数据库检查
        # 2.1 lesson_table，lesson_name检查格式"{course_name}-第01课"
        # 2.2 clockin_table同理：{course_name}-打卡数
        # 小鹅通/内容/打卡，搜索prefix，找到打卡的url
        url = ''
        self.update_clockin(f'{self.course_name}-打卡数', url)

        # 3 jsa

        # 复制该py脚本，修改init中对应参数（book_id，课程商品名）
        # 复制最新的kqcourse.js到在线表格

        # 然后依次执行：自动填充考勤表日期, 批量优化条件格式, 设置课次超链接
        # 手动再调整下第2行表头的样式格式
        # 清理已有的课程、打卡数据；第4行公式重置为初始未返款状态；第3行N列调整日期；O列调整返款标题
        # 修缮考勤表公式

        # 3 完善报名表

        # 4 结合分组表，完善考勤表


def main():
    kq = 考勤课程()
    if not kq.get_daily():
        return
    kq.step1()
    kq.step2()
    kq.step3()
    kq.step4()
    kq.step5()
    kq.step6()


if __name__ == '__main__':
    os.chdir(get_xl_homedir())
    if len(sys.argv) > 1:
        fire.Fire()
    else:
        # kq = 考勤课程()
        # kq.开课配置()

        # kq.status = 0
        # kq.step4()
        # kq.step5()
        # kq.step6()
        main()

        # print(kq.course_name[8:10])
