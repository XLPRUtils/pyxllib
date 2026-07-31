from types import SimpleNamespace
from jinja2 import Template
from pyxllib.file.xlsxlib import get_column_letter

from xlsln.kq5034.ckz240412网课考勤 import *
from xlproject.code4101 import support_retry_process

jscodes = SimpleNamespace()

jscodes.step3 = r"""
const [ur, rows, cols] = locateTableRange('考勤表', 4, ['视频应返款'])
const refundDict = parseRefundRules(findCel('视频应返款', ur).Offset(1, 0).Text)
cols['第01课'] = findCol('第01课', ur.Rows(2), xlPart)
cols['第21课'] = findCol('第21课', ur.Rows(2), xlPart)
for (let i = {{rows_start}}; i <= {{rows_end}}; i++) {
    let totalRefund = 0
    for (let j = cols['第01课']; j <= cols['第21课']; j++)
        totalRefund += highlightCourseProgress(refundDict, ur.Cells(i, j))
    ur.Cells(i, cols['视频应返款']).Value2 = totalRefund
}
""".strip()

jscodes.step5 = r"""
const [ur, rows, cols] = locateTableRange('考勤表', 4, ['订单金额', '已返款', '总应返款'])
for (let i = rows.start; i <= rows.end; i++) {
    const orderAmount = ur.Cells(i, cols['订单金额']).Value2 || 0
    if (orderAmount !== 0) {
        ur.Cells(i, cols['已返款']).Value2 = Math.max(ur.Cells(i, cols['总应返款']).Value2, ur.Cells(i, cols['已返款']).Value2)
    } else {
        ur.Cells(i, cols['已返款']).Value2 = 0
    }
}
""".strip()


class 网课考勤Ex(网课考勤):

    def __init__(self):
        super().__init__()
        self.course_name = '第28届觉观技术公益网课'
        # self.wb = Wps考勤表('chjoiIr4vc0y')
        self.wb = Wps考勤表('crnE9qT1q3wq')  # 测试用副本
        self.status = None

    def get_status(self):
        """ 获取已运行的状态，这个每次程序只要运行一次即可 """
        if self.status is None:
            text = self.wb.run_airscript("return findCel('返款配置', Sheets('考勤表').UsedRange).Offset(1, 0).Value2")
            status = 0
            try:
                # 解析文本，假设格式为 '最近运行更新时间：\nYYYY/MM/DD hh:mm:ss,状态编号'
                parts = text.split('\n')[1].split(',')
                date_str = parts[0].strip()  # 获取日期字符串
                status = int(parts[1]) if len(parts) > 1 else 6  # 获取状态编号，默认为6
                # 解析日期并判断是否是今天
                date = datetime.datetime.strptime(date_str.split()[0], "%Y/%m/%d").date()
                today = datetime.date.today()
                if date != today:
                    status = -status  # 如果不是今天，取负数
            except Exception as e:
                pass
            self.status = status
        return self.status

    def set_status(self, status):
        """ 设置运行的状态 """
        tag = rf'最近运行更新时间：\n{utc_timestamp().replace('-', '/')}'
        if status < 6:
            tag += f',{status}'
        self.wb.run_airscript(f"findCel('返款配置', Sheets('考勤表').UsedRange).Offset(1, 0).Value2 = '{tag}'")

    def step1(self, update=True):
        if self.get_status() >= 1:
            return
        logger.info('1 下载小鹅通数据')
        if update:
            # 更新课程和打卡数据
            self.xe2.switch_shop('5034山中薪')
            self.update_lesson_data_table(shop1=True)
            self.update_clockin(self.course_name + '【中心教室】')
            wechat_lock_send('考勤管理', f'"{self.course_name}"考勤数据更新完成')
        self.set_status(1)

    def step2(self):
        """ 每个课程主要的差异在这里 """
        if self.get_status() >= 2:
            return
        logger.info('2 从数据库xldb2获取新的考勤数据并写回在线表格')

        # 1 获取用户清单
        user_id2s = self.wb.get_column_list('用户ID')

        # 2 拼接应该写入的数据
        # 打卡数据
        titles = [f'【第28届中心教室】—第{i}课打卡' for i in range(1, 22)]
        df1 = self.browser_clockin_data(f'{self.course_name}', ['【中心教室】'],
                                        user_id2s=user_id2s, titles=titles)
        # 视频数据
        df2 = self.browser_lesson_data(f'{self.course_name}-', user_id2s=user_id2s)
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

    def step3(self):
        if self.get_status() >= 3:
            return
        logger.info('3 更新应返款')
        # 获得数据范围，rows['start']、rows['end']存储了整体数据的起始、终止行
        rows = self.wb.run_airscript("return locateTableRange('考勤表', 4, ['视频应返款'])[1]")
        batch_size = 50  # 每次只处理50行
        for r in range(rows['start'], rows['end'], batch_size):
            vars = {
                'rows_start': r,
                'rows_end': min(r + batch_size - 1, rows['end']),
            }
            jscode = Template(jscodes.step3).render(vars)
            # print(content)
            self.wb.run_airscript(jscode)
        logger.info('应返款更新完成')
        self.set_status(3)

    def step4(self):
        if self.get_status() >= 4:
            return
        logger.info('4 自动返款')
        lines = self.wb.get_column_list('返款配置')
        if lines:
            自动返款促学金(lines)
        self.set_status(4)

    def step5(self):
        if self.get_status() >= 5:
            return
        logger.info('5 更新已返款')
        self.wb.run_airscript(jscodes.step5)
        self.set_status(5)

    def step6(self):
        if self.get_status() >= 6:
            return
        logger.info('6 发送日报')
        # "当前应返款"的下一个单元格就是日期值（找某个参数名对应配置值的写法）
        day = self.wb.run_airscript("return findCel('当前应返款', Sheets('考勤表').UsedRange).Offset(1, 0).Value2")
        发送日报('5034第28届觉观网课中心教室', day, f'https://kdocs.cn/l/{self.wb.file_id}')
        self.set_status(6)


@support_retry_process()
def main():
    kq = 网课考勤Ex()
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
        main()
