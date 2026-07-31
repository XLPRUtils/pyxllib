import re

from xlsln.kq5034.courses.kqcourse import *


class 考勤课程(KqCourse):

    def __init__(self):
        super().__init__(1,
                         XlPath(__file__).stem,
                         'casO8HhNVtlt',
                         'V2-6DtGBVDEeuArTd7mAnB0XG',
                         7,
                         课程商品名='第40届念住禅法（初阶）【中心教室】')

    def test_wb2(self):
        print(self.wb.run_func('locateTableRange', '考勤表', 4, ['视频应返款']))

    def _正课打卡标题(self):
        rows = self.kqdb.exec2dict(
            "SELECT DISTINCT update_title FROM clockin_data_table "
            "WHERE clockin_name=%s",
            [f"{self.course_name}-打卡数"],
        ).fetchall()
        titles = []
        for row in rows:
            title = row.get("update_title")
            text = str(title or "").strip()
            match = re.match(r"^第\s*(\d{1,2})\s*天[：:]", text)
            if match:
                titles.append((int(match.group(1)), text))
        return [
            title
            for _day, title in sorted(set(titles), key=lambda item: item[0])
        ]

    def step2(self):
        """ 每个课程主要的差异在这里 """
        if self.get_status() >= 2:
            return
        logger.info('2 从数据库xldb2获取新的考勤数据并写回在线表格')

        # 1 获取用户清单
        self.更新用户匹配()
        self.把报名表的用户ID同步到考勤表()

        user_id2s = self.wb_get_column_list('用户ID')

        # 2 拼接应该写入的数据
        # 打卡数据
        titles = self._正课打卡标题()
        df1 = self.browser_clockin_data(f'{self.course_name}-', ['打卡数'],
                                        user_id2s=user_id2s, titles=titles)
        # 视频数据
        df2 = self.browser_lesson_data(f'{self.course_name}-', user_id2s=user_id2s)
        # 删掉带有"测试"的数据列，是开课前的"测试1"、"测试2"
        df2 = df2.loc[:, ~df2.columns.str.contains('测试')]

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
        wechat_lock_send('考勤中台', '@如如')

        tag = self.course_name[1:5]
        if int(tag) % 2:
            wechat_lock_send('考勤中台', self.get_daily(f'20{tag}念住单月网课群'))
        else:
            wechat_lock_send('考勤中台', self.get_daily(f'20{tag}念住双月网课群'))

        self.set_status(6)


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


def 更新修订():
    kq = 考勤课程()
    kq.status = 1
    kq.step2()
    kq.step3()


if __name__ == '__main__':
    os.chdir(get_xl_homedir())
    if len(sys.argv) > 1:
        fire.Fire()
    else:
        更新修订()
