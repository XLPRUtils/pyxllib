"""已完结：20260301禅宗46期五阶。"""

import json
from urllib.parse import urlencode

from xlsln.kq5034.courses.kqcourse import *
from xlsln.kq5034.courses.zen_stage5_catalog import 获取禅宗五阶补充课次, 获取禅宗五阶一次性排课事故日期偏移


class 考勤课程(KqCourse):
    打卡返款单项上限 = 16
    打卡返款单价 = 5

    # ==========================================================================
    # 重要：下面这个“动态补打卡”是 46期五阶在 2026-06-21 后遇到小鹅通无法补
    # 共修/共学打卡时的事故专用插件。它没有通用迁移性。
    # 新建、复制、继承、集成其它课程配置时，必须清空/删除这个插件配置；
    # 只有用户明确要求某个课程启用动态补打卡时，才允许重新按该课程规则配置。
    # ==========================================================================
    动态补打卡插件 = {
        'community_id': 'c_69a3a337724fc_TJoXKy8E2351',
        'feed_url': ('https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/'
                     'community_manage/content_settings/feed_list'
                     '?communityId=c_69a3a337724fc_TJoXKy8E2351&type=manage'),
        'start_at': pd.Timestamp('2026-06-21 00:00:00'),
        'page_size': 100,
        'max_pages': 50,
        'clockin_names': {
            '共学': '共学打卡',
            '共修': '共修打卡',
        },
    }

    def __init__(self):
        super().__init__(2,
                         XlPath(__file__).stem,
                         'ctMnRgPB3Hm4',
                         'V2-78OlydSkG9LMdZ9832LmxA',
                         课程商品名='5阶',
                         )

    @staticmethod
    def _sheet_count(value):
        if value is None or value == '':
            return 0
        try:
            if pd.isna(value):
                return 0
        except TypeError:
            pass
        return int(float(value))

    @staticmethod
    def _format_count(value):
        return '' if not value else int(value)

    @classmethod
    def _计算打卡应返款(cls, 共学打卡, 共修打卡):
        """共学、共修各最多返 16 次，每次 5 元，全期最多 160 元。"""
        return (
            min(cls._sheet_count(共学打卡), cls.打卡返款单项上限)
            + min(cls._sheet_count(共修打卡), cls.打卡返款单项上限)
        ) * cls.打卡返款单价

    @staticmethod
    def _动态补打卡内容(row):
        content = row.get('content')
        if isinstance(content, dict):
            content = content.get('text')
        return str(content or '').strip()

    @staticmethod
    def _动态补打卡时间(row):
        for key in ['created_at', 'publish_time']:
            value = row.get(key)
            if value is None or value == '':
                continue
            dt = pd.to_datetime(value, errors='coerce')
            if not pd.isna(dt):
                return dt
        return None

    @staticmethod
    def _动态补打卡去重键(feed_row, user_id2, clockin_name, publish_time, content):
        # 不能按天去重：学员可能在同一天用多条动态补完多次共修/共学。
        # 这里仅防止接口分页或重试时重复返回同一条动态。
        for key in ['id', 'feed_id', 'feeds_id', 'dynamic_id']:
            value = str(feed_row.get(key) or '').strip()
            if value:
                return 'feed', value
        return 'event', user_id2, clockin_name, publish_time.isoformat(), content

    def _请求动态列表页(self, tab, page_index):
        plugin = self.动态补打卡插件
        params = {
            'search_content': '',
            'page_index': page_index,
            'page_size': plugin['page_size'],
            'community_id': plugin['community_id'],
            'feeds_list_type': -1,
            'nick_name': '',
            'start_date': plugin['start_at'].date().isoformat(),
            'end_date': '',
            'roles': '',
        }
        url = '/small_community/b_feeds_list?' + urlencode(params)
        js = """
const xhr = new XMLHttpRequest();
xhr.open('GET', arguments[0], false);
xhr.withCredentials = true;
xhr.send(null);
return {status: xhr.status, text: xhr.responseText};
"""
        response = tab.run_js(js, url)
        if not isinstance(response, dict):
            raise RuntimeError('动态列表接口返回异常')

        status = int(response.get('status') or 0)
        if status >= 400:
            raise RuntimeError(f'动态列表接口请求失败：HTTP {status}')

        payload = json.loads(response.get('text') or '{}')
        if int(payload.get('code') or 0) != 0:
            raise RuntimeError(f"动态列表接口返回失败：{payload.get('msg') or payload.get('message') or payload.get('code')}")
        data = payload.get('data') or {}
        return data if isinstance(data, dict) else {}

    def _读取动态补打卡原始数据(self):
        plugin = self.动态补打卡插件
        rows = []
        with self.xe2.临时工作标签页(url=plugin['feed_url'], wait_seconds=3) as tab:
            for page_index in range(1, plugin['max_pages'] + 1):
                data = self._请求动态列表页(tab, page_index)
                page_rows = [dict(row) for row in data.get('list') or [] if isinstance(row, dict)]
                if not page_rows:
                    break
                rows.extend(page_rows)

                total_count = int(data.get('total_count') or data.get('total') or 0)
                if total_count and len(rows) >= total_count:
                    break
                oldest_time = self._动态补打卡时间(page_rows[-1])
                if oldest_time is not None and oldest_time < plugin['start_at']:
                    break
        return rows

    def _计算动态补打卡明细(self, user_id2s):
        plugin = self.动态补打卡插件
        valid_user_id2s = {str(x or '').strip() for x in user_id2s if str(x or '').strip()}
        duplicate_keys = set()
        rows = []
        summary = {
            'feed_rows': 0,
            'candidate_rows': 0,
            'inserted_rows': 0,
            'skipped_before_start_rows': 0,
            'skipped_missing_user_rows': 0,
            'skipped_not_in_sheet_rows': 0,
            'skipped_empty_content_rows': 0,
            'skipped_missing_time_rows': 0,
            'skipped_duplicate_rows': 0,
            'inserted_by_clockin_name': {},
        }

        feed_rows = self._读取动态补打卡原始数据()
        summary['feed_rows'] = len(feed_rows)
        for feed_row in feed_rows:
            publish_time = self._动态补打卡时间(feed_row)
            if publish_time is None:
                summary['skipped_missing_time_rows'] += 1
                continue
            if publish_time < plugin['start_at']:
                summary['skipped_before_start_rows'] += 1
                continue

            user_id2 = str(feed_row.get('user_id') or '').strip()
            if not user_id2:
                summary['skipped_missing_user_rows'] += 1
                continue
            if valid_user_id2s and user_id2 not in valid_user_id2s:
                summary['skipped_not_in_sheet_rows'] += 1
                continue

            content = self._动态补打卡内容(feed_row)
            if not content:
                summary['skipped_empty_content_rows'] += 1
                continue

            kind = '共修' if '共修' in content else '共学'
            clockin_name = plugin['clockin_names'][kind]
            duplicate_key = self._动态补打卡去重键(feed_row, user_id2, clockin_name, publish_time, content)
            summary['candidate_rows'] += 1
            if duplicate_key in duplicate_keys:
                summary['skipped_duplicate_rows'] += 1
                continue
            duplicate_keys.add(duplicate_key)

            rows.append({
                'user_id2': user_id2,
                'clockin_name': clockin_name,
                'publish_time': publish_time,
                'content': content,
            })
            summary['inserted_rows'] += 1
            summary['inserted_by_clockin_name'][clockin_name] = summary['inserted_by_clockin_name'].get(clockin_name, 0) + 1

        return pd.DataFrame(rows), summary

    def _补入动态补打卡数量(self, df, user_id2s):
        extra_df, summary = self._计算动态补打卡明细(user_id2s)
        logger.info(f'46期五阶动态补打卡试算：{summary}')
        if extra_df.empty:
            return df

        counts = extra_df.groupby(['user_id2', 'clockin_name']).size().reset_index(name='动态补打卡')
        for clockin_name in self.动态补打卡插件['clockin_names'].values():
            if clockin_name not in df:
                df[clockin_name] = ''
            col_counts = counts[counts['clockin_name'] == clockin_name][['user_id2', '动态补打卡']]
            df = df.merge(col_counts, on='user_id2', how='left')
            df['动态补打卡'] = df['动态补打卡'].fillna(0).astype(int)
            df[clockin_name] = [
                self._format_count(self._sheet_count(current) + self._sheet_count(extra_value))
                for current, extra_value in zip(df[clockin_name], df['动态补打卡'])
            ]
            df.drop(columns=['动态补打卡'], inplace=True)
        return df

    def 试算动态补打卡(self):
        """只读试算动态补打卡，不写 WPS，也不修改数据库。"""
        user_id2s = self.wb_get_column_list('用户ID')
        extra_df, summary = self._计算动态补打卡明细(user_id2s)
        if not extra_df.empty:
            summary['users_with_supplement'] = int(extra_df['user_id2'].nunique())
        else:
            summary['users_with_supplement'] = 0
        return summary

    def step2(self):
        """ 每个课程主要的差异在这里 """
        if self.get_status() >= 2:
            return
        logger.info('2 从数据库xldb3获取新的考勤数据并写回在线表格')

        # 1 获取用户清单
        self.更新用户匹配()
        self.把报名表的用户ID同步到考勤表()

        user_id2s = self.wb_get_column_list('用户ID')
        df1 = self.browser_clockin_data(f'{self.course_name}-', None, user_id2s=user_id2s)
        df1 = self._补入动态补打卡数量(df1, user_id2s)

        # 视频数据
        df2 = self.browser_lesson_data(f'{self.course_name}', user_id2s=user_id2s)

        # 拼接数据
        df3 = self.拼接打卡视频数据(user_id2s, df1, df2)

        # 3 把数据写入表格
        self.write_rows_skip_empty(df3, find_start_col='共学打卡')

        logger.info('已将最新考勤数据文本写入在线表格')
        self.set_status(2)

    def step6(self):
        self._ensure_previous_steps_completed(6, 6)
        if self.get_status() >= 6:
            return
        logger.info('6 发送日报')
        name = re.search(r'\d+期.+?阶', self.course_name).group().replace('点', '.')
        wechat_lock_send('线上修道班考勤管理', self.get_daily(f'禅宗修道普及班{name}中心教室'))
        self.set_status(6)

    def 爬虫检查课程目录(self):
        courses = 获取禅宗五阶补充课次(self.xe2)
        print(courses)
        return courses

    def 配置课次数据(self):
        courses = self.爬虫检查课程目录()
        self.kqdb.添加禅宗课次配置数据(
            self.course_name,
            20,
            courses,
            week_date_offsets=获取禅宗五阶一次性排课事故日期偏移(self.course_name),
        )

    def 配置打卡数据(self):
        self.update_clockin(f'{self.course_name}-共学打卡',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_69a45ca81ccc6_vLTG9a&community_id=c_69a3a337724fc_TJoXKy8E2351&clock_id=ac_69a45d0234f97_rHFfClYC&group_id=group_69a45d11a005e_nbhqhyB6&taskId=th_69a45d11a00b3_ogCHr9lg&beginDate=113&totalDay=112&is_combine_task=112&management_entry_id=calendar_clock_management&component_name=clock_task_data')
        self.update_clockin(f'{self.course_name}-共修打卡',
                            url='https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/micro_wrapper?apply_id=apy_69a45d28b1c00_v91o4j&community_id=c_69a3a337724fc_TJoXKy8E2351&clock_id=ac_69a45d419ee00_PGQ5wYTC&group_id=group_69a45d4c20029_XC45tZmj&taskId=th_69a45d4c20086_w6vaOYYy&beginDate=113&totalDay=112&is_combine_task=112&management_entry_id=calendar_clock_management&component_name=clock_task_data')


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

    # d250920 明天先不跑返款，我看下如何补修打卡数据
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
        # kq.从旧课程数据迁移配置()
        # kq.配置打卡数据()

        # kq.status = 0
        # kq.step1()
        # kq.step2()
        # kq.step3()
        kq.step4()
        kq.step5()
        # kq.step6()

        # kq.配置课次数据()

        # main_b()
