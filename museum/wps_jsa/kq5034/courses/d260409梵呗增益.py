from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(1,
                         XlPath(__file__).stem,
                         'cncoFAnlEY7K',
                         'V2-5jxerWcf5aKgIXzukoq48q',
                         5,
                         1,
                         课程商品名='202604本体音艺增益班',
                         )

    def step2(self, *, status=2):
        """ 每个课程主要的差异在这里 """
        if self.get_status() >= status:
            return
        logger.info('2 从数据库xldb3获取新的考勤数据并写回在线表格')

        # 1 获取用户清单
        self.更新用户匹配()
        self.把报名表的用户ID同步到考勤表()

        user_id2s = self.wb_get_column_list('用户ID')

        # 2 拼接应该写入的数据
        # todo 这个也要加titles过滤重复
        df1 = self.kqdb.browser_clockin_data(f'{self.course_name}-', ['打卡数'],
                                             user_id2s=user_id2s)
        # 视频数据
        df2 = self.browser_lesson_data(f'{self.course_name}', user_id2s=user_id2s)
        # 拼接数据
        df3 = self.拼接打卡视频数据(user_id2s, df1, df2)

        # 3 把数据写入表格
        self.write_rows_skip_empty(df3)

        logger.info('已将最新考勤数据文本写入在线表格')
        self.set_status(status)

    def step6(self, *, status=6):
        if self.get_status() >= status:
            return
        logger.info('6 发送日报')
        # {self.course_name[8:10]}
        wechat_lock_send('本体音艺考勤班委群',
                         self.get_daily(f'20{self.course_name[1:5]}本体音艺初阶增益网课班级群', -1))
        # 梵呗系列太特别，最后还是用-6的特殊标记；才能保证晚上的脚本可以继续增量更新。
        self.set_status(status)

        # todo d250705，发现日报可能有些问题，因为梵呗增益比较特别，最后两课是无回放，下次8月梵呗增益注意检查最后日报正确性

    def 开课配置(self):
        """
        文档：https://www.yuque.com/xlpr/pyxllib/nrg8piay6mt95rg0
        """
        # 1 DBeaver，数据库检查
        # lesson_table:lesson_name检查"d251009梵呗增益-第01课"这样格式的数据是否完整齐备
        # clockin_table同理：d251009梵呗增益-打卡数

        # 先搜索'2604'，爬虫找到课程。课次名若是“2604增益堂1-xxx”可自动适配为标准lesson_name。
        # 第21、22课结束日期是当天21点，仍要手动检查后再运行下述函数。
        # self.add_book_lessons_to_db()

        # 内容/打卡，搜索'202510本体音艺线上增益班'
        self.update_clockin('d260409梵呗增益-打卡数',
                            url='https://admin.xiaoe-tech.com/t/clock_admin/index#/punchDetail/diaryList?activity_id=ac_69d7321182eff_jryavJ0s&markType=&miniMiddleUrl=https%3A%2F%2Fapporrfwkpb5562.h5.xet.citv.cn%2Fxiaoe_clock%2Fmini_middle%3Factivity_id%3Dac_69d7321182eff_jryavJ0s%26app_id%3Dapporrfwkpb5562')

        # 2 jsa

        # 复制该py脚本，修改init中对应参数（book_id，课程商品名）
        # 复制最新的kqcourse.js到在线表格

        # 然后依次执行：自动填充考勤表日期, 批量优化条件格式, 设置课次超链接
        # 生成的日期，注意最后两天要手动去掉回放，变仅直播的处理。
        # 手动再调整下第2行表头的样式格式

        # 清理已有的课程、打卡数据；第4行公式重置为初始未返款状态；第3行N列调整日期；O列调整返款标题
        # 修缮考勤表公式

        # 3 完善报名表

        # 4 结合分组表，完善考勤表

        pass


def main_a():
    """ 当天晚上更新数据 """
    kq = 考勤课程()
    if not kq.get_daily():
        return
    # 21点预处理逻辑上归属次日返款周期，所以状态日期写到明天。
    kq.status_date_offset_days = 1
    kq.step1()
    kq.step2()
    kq.step3()


def main_b():
    """ 次日进行返款 """
    kq = 考勤课程()
    if not kq.get_daily(bias=-1):
        return
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

        # kq.step6()
        main_b()
        
