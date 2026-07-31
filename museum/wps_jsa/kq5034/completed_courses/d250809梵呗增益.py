from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(1,
                         XlPath(__file__).stem,
                         'crw4mD3m1Wu9',
                         'V2-5jxerWcf5aKgIXzukoq48q',
                         5,
                         1,
                         课程商品名='202508本体音艺增益班',
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
        # 文档：https://www.yuque.com/xlpr/pyxllib/nrg8piay6mt95rg0

        # 生成的日期，注意最后两天要手动去掉回放，变仅直播的处理。

        # 先搜索'2508'，爬虫找到课程，手动调整名称等适配后，运行下述函数。
        # self.add_book_lessons_to_db()

        # 内容/打卡，搜索'202508本体音艺线上增益班'
        self.update_clockin('d250809梵呗增益-打卡数',
                            url='https://admin.xiaoe-tech.com/t/clock_admin/index#/punchDetail/diaryList?activity_id=ac_675831c585c74_R4ntwnKl&markType=&miniMiddleUrl=https%3A%2F%2Fapporrfwkpb5562.h5.xiaoeknow.com%2Fxiaoe_clock%2Fmini_middle%3Factivity_id%3Dac_675831c585c74_R4ntwnKl%26app_id%3Dapporrfwkpb5562')

        # jsa运行：设置课次超链接

        pass


def main_a():
    """ 当天晚上更新数据 """
    kq = 考勤课程()
    if not kq.get_daily():
        return
    kq.step1(status=7)
    kq.step2(status=8)
    kq.step3(status=9)


def main_b():
    """ 次日进行返款 """
    kq = 考勤课程()
    if not kq.get_daily(bias=-1):
        return
    # 由于梵呗执行main_b的时候都是新的一天，所以梵呗不能补跑123，否则会重复下载打卡数据等
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
        # kq = 考勤课程()
        # kq.step6()
        main_a()
        