from types import SimpleNamespace
from jinja2 import Template
from pyxllib.file.xlsxlib import get_column_letter

from xlsln.kq5034.ckz240412网课考勤 import *
from xlproject.code4101 import support_retry_process

from xlsln.kq5034.courses.base import 网课考勤Ex, jscodes

jscodes.step3 = r"""
const [ur, rows, cols] = locateTableRange('考勤表', 4, ['视频应返款'])

// 梵呗增益这段比较特别，是有点不一样的。目前是直播、回放都20元，没有梯度。可以手动设置字典。
const refundDict = {当堂:20, 第1天:20, 第2天:20, 第3天:20, 第4天:20, 第5天:20}

cols['第01课'] = findCol('第01课', ur.Rows(2), xlPart)
cols['第22课'] = findCol('第22课', ur.Rows(2), xlPart)
for (let i = {{rows_start}}; i <= {{rows_end}}; i++) {
    let totalRefund = 0
    for (let j = cols['第01课']; j <= cols['第22课']; j++)
        totalRefund += highlightCourseProgress(refundDict, ur.Cells(i, j))
    ur.Cells(i, cols['视频应返款']).Value2 = totalRefund
}
""".strip()


class 网课考勤Exx(网课考勤Ex):

    def __init__(self):
        super().__init__()
        self.course_name = '2412增益'
        self.wb = Wps考勤表('cvxZbpnhb1sI')
        self.status = None
        self.start_date = datetime.date(2024, 12, 9)
        self.回放天数 = 6

    def step1(self, update=True):
        if self.get_status() >= 1:
            return
        logger.info('1 下载小鹅通数据')
        if update:
            # 更新课程和打卡数据
            self.xe2.switch_shop('5034山中薪')
            self.update_lesson_data_table(shop1=True)
            self.update_clockin('202412本体音艺线上增益班')
            logger.info(f'"{self.course_name}"下载完小鹅通数据')
            self.xe2.close_if_exceeds_min_tabs()
        self.set_status(1)

    def step2(self):
        """ 每个课程主要的差异在这里 """
        if self.get_status() >= 2:
            return
        logger.info('2 从数据库xldb2获取新的考勤数据并写回在线表格')

        # 1 获取用户清单
        user_id2s = self.wb.get_column_list('用户ID')

        # 2 拼接应该写入的数据
        df1 = self.browser_clockin_data(f'202412本体音艺线上增益班', [''],
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

    def step4(self):
        if self.get_status() >= 4:
            return
        # 本次梵呗跳过日常返款流程
        logger.info('4 自动返款')
        lines = self.wb.get_column_list('返款配置')
        if lines:
            self.weipay.login(['考勤管理'])  # 等会如果要登录扫码，通过指定微信群获取
            自动返款促学金(lines, self.weipay)
        self.set_status(4)
        # self.weipay.tab.close()

    def step5(self):
        if self.get_status() >= 5:
            return
        # 本次梵呗跳过日常返款流程
        logger.info('5 更新已返款')
        self.wb.run_airscript(jscodes.step5)
        self.set_status(5)

    def step6(self):
        if self.get_status() >= 6:
            return
        logger.info('6 发送日报')
        wechat_lock_send('202412本体音艺初阶增益班委群', self.get_daily('202412本体音艺初阶增益网课班级群'))
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
        main()
