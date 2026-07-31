from pyxllib.file.xlsxlib import get_column_letter

from xlsln.kq5034.ckz240412网课考勤 import *
from xlproject.code4101 import support_retry_process

from xlsln.kq5034.courses.base import 网课考勤Ex


class 网课考勤Exx(网课考勤Ex):

    def __init__(self):
        super().__init__()
        self.course_name = '第30届觉观技术公益网课'
        self.wb = KqCourseBook('ccAiVkGqNdML')
        self.status = None
        self.start_date = datetime.date(2025, 1, 1)
        self.回放天数 = 5

    def step1(self, update=True):
        if self.get_status() >= 1:
            return
        logger.info('1 下载小鹅通数据')
        if update:
            # 更新课程和打卡数据
            self.xe2.switch_shop('5034山中薪')
            self.update_lesson_data_table(shop1=True)
            self.update_clockin(self.course_name + '【中心教室】')
            logger.info(f'"{self.course_name}"下载完小鹅通数据')
            self.xe2.tab.close()
        self.set_status(1)

    def step2(self):
        """ 每个课程主要的差异在这里 """
        if self.get_status() >= 2:
            return
        logger.info('2 从数据库xldb1获取新的考勤数据并写回在线表格')

        # 1 获取用户清单
        user_id2s = self.wb.get_column_list('用户ID')

        # 2 拼接应该写入的数据
        # 打卡数据
        titles = [f'【打卡】第30届中心教室-{i}' for i in range(1, 22)]
        df1 = self.browser_clockin_data(f'{self.course_name}', ['【中心教室】'],
                                        user_id2s=user_id2s, titles=titles)
        # 视频数据
        df2 = self.browser_lesson_data(f'{self.course_name}-', user_id2s=user_id2s)
        # 删掉带有"测试"的数据列，是开课前的"测试1"、"测试2“
        df2 = df2.loc[:, ~df2.columns.str.contains('测试')]
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
        wechat_lock_send('考勤返款学习群', '@有容')
        wechat_lock_send('考勤返款学习群', self.get_daily('5034第30届觉观网课中心教室'))
        self.set_status(6)


@support_retry_process()
def main():
    kq = 网课考勤Exx()
    if not kq.get_daily():
        return
    kq.step1()
    kq.step2()
    kq.step3()
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
        # main()

        kq = 网课考勤Exx()
        kq.status = 0
        kq.step1()
        kq.step2()
        kq.step3()
