from pyxllib.file.xlsxlib import get_column_letter

from xlsln.kq5034.ckz240412网课考勤 import *
from xlproject.code4101 import support_retry_process

from xlsln.kq5034.courses.base import 网课考勤Ex


class 网课考勤Exx(网课考勤Ex):

    def __init__(self):
        super().__init__()
        self.course_name = '2501梵呗初阶'
        self.wb = KqCourseBook('cs1qPreithBy')
        self.status = None
        self.start_date = datetime.date(2025, 1, 9)
        self.回放天数 = 5

    def step1(self, update=True):
        if self.get_status() >= 1:
            return
        logger.info('1 下载小鹅通数据')
        if update:
            # 更新课程和打卡数据
            self.xe2.switch_shop('5034山中薪')
            self.update_lesson_data_table(shop1=True)
            self.update_clockin('202501梵呗班学修日志')
            logger.info(f'"{self.course_name}"下载完小鹅通数据')
            self.xe2.close_if_exceeds_min_tabs()
        self.set_status(1)

    def step2(self):
        """ 每个课程主要的差异在这里 """
        if self.get_status() >= 2:
            return
        logger.info('2 从数据库xldb获取新的考勤数据并写回在线表格')

        # 1 获取用户清单
        user_id2s = self.wb.get_column_list('用户ID')

        # 2 拼接应该写入的数据
        df1 = self.browser_clockin_data(f'202501梵呗班学修日志', [''],
                                        user_id2s=user_id2s)
        # 视频数据
        df2 = self.browser_lesson_data(f'{self.course_name}', user_id2s=user_id2s)
        # 拼接数据
        df3 = self.拼接打卡视频数据(user_id2s, df1, df2)
        rows = [row[1:] for row in df3.to_dict(orient='split')['data']]

        # 3 把数据写入表格
        # 数据要从哪里开始写入
        start_col = self.wb.run_airscript("return findCol('打卡数', Sheets('考勤表').UsedRange)")
        col_name = get_column_letter(start_col)
        # 每次写入50行，如果数据很大比较特殊，怕速度慢，一次30秒的限制会超时，可以把50再改小
        self.wb.write_arr(rows, '考勤表', f'{col_name}4', 50)
        logger.info('已将最新考勤数据文本写入在线表格')
        self.set_status(2)

    def step6(self):
        if self.get_status() >= 6:
            return
        logger.info('6 发送日报')
        wechat_lock_send('考勤返款学习群', '@艳子')
        wechat_lock_send('考勤返款学习群', self.get_daily('202501本体音艺初阶班级群', -1))
        # 梵呗系列太特别，最后还是用-6的特殊标记；才能保证晚上的脚本可以继续增量更新。
        # todo 这个有隐患，以后才考虑更好的优化策略。
        self.set_status(-6)

    def step2_test(self):
        """ 250115周三16:36，临时debug调试用 """
        user_id2s = self.wb.get_column_list('用户ID')
        df2 = self.browser_lesson_data(f'{self.course_name}-', user_id2s=user_id2s)
        browser(df2)


@support_retry_process()
def main_a():
    """ 当天晚上更新数据 """
    kq = 网课考勤Exx()
    if not kq.get_daily():
        return
    kq.step1()
    kq.step2()
    kq.step3()


@support_retry_process()
def main_b():
    """ 次日进行返款 """
    kq = 网课考勤Exx()
    if not kq.get_daily(bias=-1):
        return
    kq.step1()
    kq.step2()
    kq.step3()

    # 主要是继续跑4、5、6
    kq.step4()
    kq.step5()
    kq.step6()


def 更新修订():
    """ 可能手动修正了一些异常，需要刷新显示数据的时候，可以用这个函数，强制重运行第2、3步 """
    kq = 网课考勤Exx()
    kq.status = 1
    kq.step2()
    kq.step3()


if __name__ == '__main__':
    os.chdir(get_xl_homedir())
    if len(sys.argv) > 1:
        fire.Fire()
    else:
        # main_b()

        更新修订()
