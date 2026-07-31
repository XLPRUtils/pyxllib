from pyxllib.prog.pupil import run_once
from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'coTdxDpsVECK',
                         'V2-78OlydSkG9LMdZ9832LmxA',
                         )

    def step2(self):
        """ 每个课程主要的差异在这里 """
        if self.get_status() >= 2:
            return
        logger.info('2 从数据库xldb3获取新的考勤数据并写回在线表格')

        # 1 获取用户清单
        user_id2s = self.wb_get_column_list('用户ID')

        # 2 拼接应该写入的数据
        # todo 这个也要加titles过滤重复
        df1 = self.browser_clockin_data(f'{self.course_name}-', None, user_id2s=user_id2s)

        # 视频数据
        df2 = self.browser_lesson_data(f'{self.course_name}', user_id2s=user_id2s)

        # 拼接数据
        df3 = self.拼接打卡视频数据(user_id2s, df1, df2)

        # 3 把数据写入表格
        self.write_rows_skip_empty(df3, find_start_col='4,6期共学打卡')

        logger.info('已将最新考勤数据文本写入在线表格')
        self.set_status(2)

    def step6(self):
        if self.get_status() >= 6:
            return
        logger.info('6 发送日报')
        wechat_lock_send('禅宗修道考勤管理', self.get_daily('禅宗修道普及班4阶4,6期中心教室'))
        self.set_status(6)

    def 数据补丁(self, sheet, bias_week, names):
        # 1 获取数据库课程数据情况
        lessons = self.kqdb.exec2dict(
            "SELECT * FROM lesson_table WHERE lesson_name LIKE 'd250317禅宗4阶46期-%' ORDER BY lesson_id")

        @run_once('str')
        def find_lesson_id(name):
            for lesson in lessons:
                if name in lesson['lesson_name']:
                    return lesson['lesson_id']
            return None

        # 2 遍历更新每个用户的课程数据时间点
        base_start = parse_datetime('2025-03-23 00:00:00')
        new_update_time = base_start + timedelta(minutes=1, weeks=bias_week - 1)

        cols = list(names)
        df = self.wb.sql_select(sheet, ['用户ID'] + cols, 4)
        df = df[['用户ID'] + cols]

        for idx, row in df.iterrows():
            if not row['用户ID']:
                continue

            for col in cols:
                if row[col] != '已完成':
                    continue

                lesson_id = find_lesson_id(col)
                if not lesson_id:
                    continue

                # 把lesson_data_table里，user_id2、lesson_id对应的所有条目的update_time都重置了
                self.kqdb.execute(
                    f"UPDATE lesson_data_table SET update_time = '{new_update_time}' "
                    f"WHERE user_id2 = '{row['用户ID']}' AND lesson_id = {lesson_id}")
                self.kqdb.commit()

        self.kqdb.commit()

    def 数据补丁1(self):
        self.数据补丁('1.返款进度', 1, ['阿含经导读1', '阿含经导读2', '阿含经导读3', '瑜伽菩萨戒1'])
        self.数据补丁('1.返款进度', 2, '阿含经导读4	瑜伽菩萨戒2	瑜伽菩萨戒3	瑜伽菩萨戒4'.split())
        self.数据补丁('1.返款进度', 3, '阿含经导读5	瑜伽菩萨戒5	瑜伽菩萨戒6	瑜伽菩萨戒7'.split())
        self.数据补丁('1.返款进度', 4,
                      '基础止观1	基础止观2	基础止观3	基础止观4	基础止观5	瑜伽菩萨戒8	瑜伽菩萨戒9'.split())
        self.数据补丁('1.返款进度', 5,
                      '基础止观6	瑜伽菩萨戒10	瑜伽菩萨戒11	瑜伽菩萨戒12	瑜伽菩萨戒13'.split())
        self.数据补丁('1.返款进度', 6, '基础止观7	基础止观8	般若经导读1	般若经导读2'.split())
        self.数据补丁('1.返款进度', 7, '基础止观9	基础止观10	般若经导读3'.split())
        self.数据补丁('1.返款进度', 8, '基础止观11	金刚经导选1'.split())
        self.数据补丁('1.返款进度', 9, '基础止观12	基础止观13	金刚经导选2'.split())
        self.数据补丁('1.返款进度', 10, '基础止观14	基础止观15	金刚经导选3'.split())
        self.数据补丁('1.返款进度', 11, '基础止观16	基础止观17	基础止观18	基础止观19	基础止观20'.split())
        self.数据补丁('1.返款进度', 12, '基础止观21	基础止观22	基础止观23	基础止观24	基础止观25'.split())
        self.数据补丁('1.返款进度', 13, '基础止观26	基础止观27	基础止观28	基础止观29'.split())
        self.数据补丁('1.返款进度', 14, '基础止观30	基础止观31	基础止观32'.split())

    def 数据补丁2(self):
        self.数据补丁('2.考试资格', 2, ['阿含经导读1', '阿含经导读2', '阿含经导读3', '瑜伽菩萨戒1'])
        self.数据补丁('2.考试资格', 3, '阿含经导读4	瑜伽菩萨戒2	瑜伽菩萨戒3	瑜伽菩萨戒4'.split())
        self.数据补丁('2.考试资格', 4, '阿含经导读5	瑜伽菩萨戒5	瑜伽菩萨戒6	瑜伽菩萨戒7'.split())
        self.数据补丁('2.考试资格', 5,
                      '基础止观1	基础止观2	基础止观3	基础止观4	基础止观5	瑜伽菩萨戒8	瑜伽菩萨戒9'.split())
        self.数据补丁('2.考试资格', 6,
                      '基础止观6	瑜伽菩萨戒10	瑜伽菩萨戒11	瑜伽菩萨戒12	瑜伽菩萨戒13'.split())
        self.数据补丁('2.考试资格', 7, '基础止观7	基础止观8	般若经导读1	般若经导读2'.split())
        self.数据补丁('2.考试资格', 8, '基础止观9	基础止观10	般若经导读3'.split())
        self.数据补丁('2.考试资格', 9, '基础止观11	金刚经导选1'.split())
        self.数据补丁('2.考试资格', 10, '基础止观12	基础止观13	金刚经导选2'.split())
        self.数据补丁('2.考试资格', 11, '基础止观14	基础止观15	金刚经导选3'.split())
        self.数据补丁('2.考试资格', 12, '基础止观16	基础止观17	基础止观18	基础止观19	基础止观20'.split())
        self.数据补丁('2.考试资格', 13, '基础止观21	基础止观22	基础止观23	基础止观24	基础止观25'.split())
        self.数据补丁('2.考试资格', 14, '基础止观26	基础止观27	基础止观28	基础止观29'.split())
        self.数据补丁('2.考试资格', 15, '基础止观30	基础止观31	基础止观32'.split())


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


def main_每天更新打卡():
    kq = 考勤课程()

    logger.info('1 下载小鹅通数据')
    kq.xe2.switch_shop('宗门学府')
    kq.update_clockin(f'{kq.course_name}-*')
    kq.set_status(1)

    kq.step2()


def 更新修订():
    """ 可能手动修正了一些异常，需要刷新显示数据的时候，可以用这个函数，强制重运行第2、3步 """
    kq = 考勤课程()
    kq.status = 1
    kq.step2()
    kq.step3()


if __name__ == '__main__':
    os.chdir(get_xl_homedir())
    if len(sys.argv) > 1:
        fire.Fire()
    else:
        # kq = 考勤课程()
        # kq.step4()
        # kq.step5()

        # kq.status = 0
        # kq.step3()

        # kq.考试资格计算()

        # main_b()

        main_每天更新打卡()
