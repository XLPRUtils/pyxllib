"""KQ5034 业务编排入口。"""

from .common import *  # noqa: F403
from .db import KqBook, KqDb, get_kqdb, get_用户列表
from .order_ops import find_order_in_db, sync_kqbook_order_sheet
from .weipay import Weipay
from .xiaoetong import XiaoetongApi, XiaoetongWeb

class KqTools:
    root = xlhome_dir('data/m2112kq5034')

    def __init__(self):
        self._xe = None
        self._xe2 = None
        self._weipay = None
        self._kqdb = None
        self._kqbook = None

    def __0_附属工具(self):
        pass

    @property
    def xe(self):
        """ api管理中心：https://admin.cloud.xiaoe-tech.com/#/account """
        if self._xe is None:
            xe = XiaoetongApi()
            xe.login()
            self._xe = xe
        return self._xe

    @property
    def xe2(self):
        """ 小鹅通网页端爬虫 """
        if self._xe2 is None:
            self._xe2 = XiaoetongWeb()
        return self._xe2

    @property
    def weipay(self):
        """ 微信支付 """
        if self._weipay is None:
            self._weipay = Weipay(['考勤后台'])
        return self._weipay

    @property
    def kqdb(self) -> KqDb:
        """ 数据库 """
        if self._kqdb is None:
            self._kqdb = get_kqdb()
        return self._kqdb

    @property
    def kqbook(self) -> KqBook:
        """ 考勤总表 """
        if self._kqbook is None:
            self._kqbook = KqBook()
        return self._kqbook

    def __1_开课前配置工具(self):
        pass

    def update_user_table(self,
                          shop1_file=None,
                          shop2_file=None,
                          shop1_api=False):
        """ 更新用户清单表
        api虽然可以做，但如果想要实时更新数据，代价成本非常大，加上禅宗目前是不支持api的，所以我觉得还是用这个drissionlib解决实在

        :param shop1_file: "5034山中薪"的用户清单文件
            输入"本地文件"：表示直接上传本地某份文件的数据
            None：表示不处理
            True: 表示从网站下载最新的数据自动进行更新
        :param shop2_file: "宗门学府"的用户清单文件，参数格式同file1
        :param shop1_api: 是否使用小鹅通api更新shop1的用户清单数据
        """

        # 1 先更新5034山中薪的
        if shop1_file is True:
            self.xe2.switch_shop('5034山中薪')
            shop1_file = self.xe2.export_user_list()
            # 把下载的文件，放到特定的目录下
            if shop1_file is not None:
                shop1_file = shutil.copy(shop1_file, get_xl_homedir() / 'data/m2112kq5034/数据表' / shop1_file.name)
        if shop1_file is not None:
            self.kqdb.update_user_table_from_file(shop1_file, 1)

        # 2 再更新宗门学府的
        if shop2_file is True:
            self.xe2.switch_shop('宗门学府')
            shop2_file = self.xe2.export_user_list()
            if shop2_file is not None:
                shop2_file = shutil.copy(shop2_file, get_xl_homedir() / 'data/m2112kq5034/数据表' / shop2_file.name)
        if shop2_file is not None:
            self.kqdb.update_user_table_from_file(shop2_file, 2)

        # 3 使用api更新shop1的用户数据
        if shop1_api:
            last_users = self.kqdb.get_last_user_ids(1)
            new_users = self.xe.get_new_user_list_with_detail(exists_users=last_users)
            self.kqdb.insert_new_users(new_users)

    def browser_users(self, user_id2s):
        return self.kqdb.browser_users(user_id2s)

    def add_book_lessons_to_db(self, data_row=4):
        return self.kqdb.add_book_lessons_to_db(data_row)

    def update_shop1_all_lesson_playback_settings(self, force_reset_lessons=None):
        """ 将没有next_update的店铺1的课程，全部补上开始、结束、更新节点等各数据 """
        # 1 强制重置一些条目
        force_reset_lessons = force_reset_lessons or []

        # 遍历找里面的值，如果是字符串类型，则要从lesson_name映射回lesson_id的数值
        force_reset_lessons2 = []
        for x in force_reset_lessons:
            if isinstance(x, str):
                lesson_id = self.kqdb.exec2one("SELECT lesson_id FROM lesson_table "
                                               f"WHERE lesson_name='{x}'")
                force_reset_lessons2.append(lesson_id)
            else:
                force_reset_lessons2.append(x)

        # 把force_reset_lessons2这些的next_update置空
        if force_reset_lessons2:
            self.kqdb.execute(f"UPDATE lesson_table SET next_update=NULL "
                              f"WHERE shop_id=1 AND lesson_id IN ({','.join([str(x) for x in force_reset_lessons2])})")
            self.kqdb.commit_all()

        # 2 补充全部缺失数据
        lesson_id2s = self.kqdb.exec2col("SELECT lesson_id2 FROM lesson_table "
                                         "WHERE lesson_id2 NOT LIKE 'https%' AND "
                                         "(next_update IS NULL OR video_duration IS NULL) "
                                         "ORDER BY lesson_id")
        print(lesson_id2s)
        for lesson_id2 in tqdm(lesson_id2s, desc='更新课次回放设置'):
            print(lesson_id2)
            row = self.xe2.get_leeson_playback_settings(lesson_id2)
            del row['lesson_name']
            self.kqdb.update_row('lesson_table', row,
                                 {'lesson_id2': lesson_id2}, commit=True)

    def update_clockin_table(self):
        """ 更新打卡表，主要是对一些缺失开始结束日期的打卡，配置信息进行更新 """
        ls = self.kqdb.exec2dict('SELECT clockin_id, url FROM clockin_table '
                                 'WHERE start_date IS NULL OR end_date IS NULL '
                                 'ORDER BY clockin_id')
        for x in ls:
            url2 = re.sub(r'group_id=.+$', '', x['url'])
            url2 += 'management_entry_id=calendar_clock_management&component_name=clock_task'
            row = self.xe2.find_clockin_message(url2)
            if row:
                self.kqdb.update_row('clockin_table', row,
                                     {'clockin_id': x['clockin_id']},
                                     commit=True)

    def __2_日常更新数据(self):
        pass

    def update_weipay_data(self, today=None):
        return self.kqdb.update_weipay_data(today)

    @classmethod
    def _计算课次下一次需要更新的时间点(cls, row):
        """ 输入数据库课程配置的一个条目，计算下一次应该更新的时间 """
        from datetime import datetime, timedelta

        # 获取当前时间
        now = datetime.now()

        # 计算视频结束时间
        if '闯关' in row['lesson_name']:  # 闯关类课程，不用计算视频时长的偏差
            video_end_time = row['start_date']
        else:
            video_end_time = row['start_date'] + timedelta(seconds=row['video_duration'] or 0)

        # 确定更新间隔（这里可以对各种新的课程逻辑配置规则，目前禅宗间隔7天，普通课程间隔1天）
        update_interval = timedelta(days=7 if '禅宗' in row['lesson_name'] else 1)

        # 如果当前时间在视频结束时间之前，下次更新就是视频结束时间
        if now < video_end_time:
            row['next_update'] = video_end_time
            return row['next_update']

        if not row.get('end_date'):
            row['end_date'] = datetime(9999, 12, 31, 23, 59, 59)

        # 如果当前时间已经超过结束日期，设置为最大时间
        if now >= row['end_date']:
            row['next_update'] = datetime(9999, 12, 31, 23, 59, 59)
            return row['next_update']

        # 从视频结束时间开始，计算下一个更新时间点
        # if 'd250317禅宗4阶' in row['lesson_name']:
        if '禅宗' in row['lesson_name']:
            # 禅宗比较特别，直接用next_update
            next_time = row['next_update']
        else:
            next_time = video_end_time

        while next_time <= now:
            next_time += update_interval

        # 如果计算出的下次更新时间超过了结束日期，就用结束日期
        if next_time > row['end_date']:
            next_time = row['end_date']

        row['next_update'] = next_time
        return row['next_update']

    def update_lesson_data_table(self, shop1=False, shop2=False):
        """ 更新学习数据表

        :param shop1: 是否更新"5034山中薪"的学习数据
        :param shop2: 是否更新"宗门学府"的学习数据
        """
        # 1 要更新的店铺数据
        shop_ids = []
        if shop1:
            shop_ids.append(1)
        if shop2:
            shop_ids.append(2)
        if not shop_ids:
            return

        # 2 遍历店铺、获取课次数据
        total_num = 0
        for shop_id in shop_ids:
            # 1 筛选出待更新的课次
            cur_dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # 数据库里next_update为空的不会被检索出来
            lessons = self.kqdb.exec2dict('SELECT * FROM lesson_table '
                                          f"WHERE next_update<='{cur_dt}' "
                                          f"AND shop_id={shop_id} "
                                          f"ORDER BY lesson_id").fetchall()
            num = len(lessons)
            total_num += num

            # 2 遍历更新每个课次
            for i, row in enumerate(lessons, start=1):
                logger.info(f'shop={shop_id} {i}/{num}：' + row['lesson_name'])

                try:
                    file = self.xe2.export_lesson_data(row)
                except DrissionPage.errors.PageDisconnectedError:
                    logger.warning('DrissionPage连接断开，尝试重启浏览器')
                    if self._xe2:
                        self._xe2.quit()
                        self._xe2 = None
                    file = self.xe2.export_lesson_data(row)

                if file is not None:
                    self.kqdb.update_lesson_data_from_file(row['lesson_id'], file)

                # 有数据，或者闯关类课程，更新update_time
                # 闯关课程，无论数据下载成功与否，都要强制到下个节点再更新
                if file or '闯关' in row['lesson_name']:
                    self.kqdb.update_row('lesson_table',
                                         {'next_update': self._计算课次下一次需要更新的时间点(row)},
                                         {'lesson_id': row['lesson_id']},
                                         commit=True)

            尝试关闭重复页面(self.xe2.browser, timeout=1, reason=f'shop={shop_id}课次下载后兜底清理',
                     keep_tab_ids=[getattr(self.xe2.tab, 'tab_id', None)])
                
            return total_num

    def update_lesson_data_table_by_course(self, course_name, update_mode=0):
        """ 更新对应课次的数据（目前算是禅宗专用）
        等禅宗的课次更新逻辑统一到shop_id=1系列，这个函数应该就可以删掉了

        :param course_name: 课程名
        :paran slice: 只更新部分课次，slice为切片参数
        :param update_mode:
            0, 已存在就不更新了
            1, 增量更新
            2, 重置更新  （目前都是用这种模式）
        """
        lessons = self.kqdb.exec2dict('SELECT lesson_id,lesson_name,lesson_id2 FROM lesson_table '
                                      f"WHERE lesson_name LIKE '%{course_name}%' "
                                      'ORDER BY lesson_id').fetchall()

        for lesson in tqdm(lessons, desc=f'{course_name} 课次数据更新'):
            lesson_id, lesson_name, url = lesson['lesson_id'], lesson['lesson_name'], lesson['lesson_id2']

            if update_mode == 0:
                if self.kqdb.exec2one('SELECT COUNT(1) FROM lesson_data_table WHERE lesson_id=%s LIMIT 1',
                                      [lesson_id]):
                    continue
            elif update_mode == 2:
                self.kqdb.execute('DELETE FROM lesson_data_table WHERE lesson_id=%s', [lesson_id, ])
                self.kqdb.commit()

            file = self.xe2.export_lesson_data(lesson)  # 导出数据，下载数据，获得文件路径file
            if file is None:
                continue
            self.kqdb.update_lesson_data_from_file(lesson_id, file)  # 把文件数据上传数据库
        尝试关闭重复页面(self.xe2.browser, timeout=1, reason=f'{course_name}课次批量下载后收尾',
                 keep_tab_ids=[getattr(self.xe2.tab, 'tab_id', None)])

    def update_clockin(self, clockin_name, url=None, download=True):
        """ shop2的打卡数据

        :param clockin_name: 打卡名称
        :param url: 打卡url
        :param bool download: 是否下载数据
        """
        # 1 如果输入了url，检查clockin_table中是否已经存在这个打卡配置，没有的话则添加
        if url is not None:
            # 这个函数接口就是自动增量更新的机制
            self.kqdb.add_clockin(clockin_name, url)

        # 2 如果clockin_name使用了'*'，则表示是要模式匹配数据库里已有的打卡名称
        clockin_names = self._resolve_clockin_names(clockin_name)

        # 3 从网页端下载最新的打卡数据文件
        if download:
            for name in clockin_names:
                self._process_single_clockin(name)

    def _resolve_clockin_names(self, clockin_name):
        """ 解析打卡名称，支持通配符 """
        if '*' in clockin_name:
            pattern = clockin_name.replace('*', '%')
            # 必须要按照clockin_id从小到大排序
            return self.kqdb.exec2col(f"SELECT name FROM clockin_table WHERE name LIKE '{pattern}' "
                                      "ORDER BY clockin_id")
        return [clockin_name]

    def _process_single_clockin(self, clockin_name):
        """ 处理单个打卡数据的下载和更新 """
        logger.info(f'处理打卡数据：{clockin_name}')
        x = self.kqdb.exec2dict('SELECT url, start_date FROM clockin_table WHERE name=%s',
                                [clockin_name, ]).fetchall()[0]

        if not x['url']:  # 空url配置
            return

        urls = self._parse_clockin_urls(x['url'])
        if not urls:
            return

        # step1上游有店铺的提前确认切换，这里理论上直接到处没问题。不放心的话也可以加一步检查。
        files, failed_urls = self._download_clockin_files(urls, x['start_date'])
        if not files:
            return
        if failed_urls:
            for f in files:
                with contextlib.suppress(Exception):
                    f.delete()
            failed_msg = '\n'.join([f'{u}: {e}' for u, e in failed_urls])
            logger.error(f'打卡多来源下载不完整，已跳过本次数据库覆盖：{clockin_name}\n{failed_msg}')
            return

        try:
            if len(files) == 1:
                self.kqdb.update_clockin_data_from_file(clockin_name, files[0])
            else:
                self._merge_and_update_clockin_files(clockin_name, files)
        finally:
            for f in files:
                f.delete()

    def _parse_clockin_urls(self, url_str):
        """ 解析URL字符串，返回URL列表 """
        try:
            urls = json.loads(url_str)
            if isinstance(urls, list):
                return urls
        except (json.JSONDecodeError, TypeError):
            pass
        return [url_str]

    def _download_clockin_files(self, urls, start_date):
        """ 下载所有URL对应的打卡文件；多URL场景只要有一段失败，就应放弃本轮覆盖 """
        files = []
        failed_urls = []
        for u in urls:
            last_error = None
            for retry in range(3):
                try:
                    f = self.xe2.export_clockin_data(u, start_date=start_date)
                    if f:
                        files.append(f)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"下载打卡数据失败（第{retry + 1}次） {u}: {e}")
                    time.sleep(2)
            if last_error is not None:
                failed_urls.append((u, last_error))
        return files, failed_urls

    def _merge_and_update_clockin_files(self, clockin_name, files):
        """ 合并多个文件并更新数据库 """
        dfs = []
        for f in files:
            try:
                # 尝试读取 Excel 或 CSV
                try:
                    df = pd.read_excel(f)
                except ValueError:
                    df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                logger.error(f"读取打卡数据文件失败 {f}: {e}")

        if not dfs:
            return

        merged_df = pd.concat(dfs, ignore_index=True)
        # 保存为临时文件以便复用 update_clockin_data_from_file 接口
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            merged_df.to_excel(tmp_path, index=False)
            self.kqdb.update_clockin_data_from_file(clockin_name, tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def __3_日常查看各统计报表(self):
        pass

    def browser_lesson_data(self, course_name, lesson_names=None, user_id2s=None):
        return self.kqdb.browser_lesson_data(course_name, lesson_names, user_id2s)

    def browser_clockin_data(self, course_name, clockin_names, user_id2s=None, titles=None, filter=None):
        return self.kqdb.browser_clockin_data(course_name, clockin_names, user_id2s, titles, filter)

    def browser_weipay_data(self, voucher_ids):
        return self.kqdb.browser_weipay_data(voucher_ids)

    def 拼接打卡视频数据(self, user_id2s, df1, df2):
        """ 按照给定的user_id2s的用户id顺序，展示打卡、视频数据
        user_id2s支持空值、重复值
        """
        # 拼接显示
        df1.drop_duplicates(subset=['user_id2'], inplace=True)
        df2.drop_duplicates(subset=['user_id2'], inplace=True)
        df = pd.merge(df1, df2, on='user_id2')

        # 对齐user_ids
        template_df = pd.DataFrame({'user_id2': user_id2s})
        aligned_df = pd.merge(template_df, df, on='user_id2', how='left')
        aligned_df.fillna('', inplace=True)

        return aligned_df

    def __4_订单功能(self):
        pass

    def 在数据库中查找订单(self, 订单号):
        return find_order_in_db(订单号, kqdb=self.kqdb)

    def kqbook_检查已返款(self, 需要退款=False, *, file_id=None, script_id=None):
        if file_id:
            kqbook = KqBook(file_id=file_id, script_id=script_id)
        else:
            kqbook = self.kqbook

        return sync_kqbook_order_sheet(
            need_refund=需要退款,
            kqbook=kqbook,
            weipay=self.weipay,
        )

    def kqbook_执行退款(self, **kwargs):
        self.kqbook_检查已返款(True, **kwargs)

    @classmethod
    def 过滤有效返款促学金(cls, lines):
        lines2 = []
        for line in lines:
            # 注意：支持自动过滤掉空行、无效的订单号格式数据
            line2 = line.strip() if line else ''
            if re.match(r'\w{6}-\w{7}-\w{4},', line2) or re.match(r'MA\d{22},', line2):
                lines2.append(line2)

        return lines2

    @classmethod
    def 自动返款促学金(cls, lines, weipay=None):
        # 1 解析行数据
        lines2 = cls.过滤有效返款促学金(lines)
        # 空数据就不用继续处理了
        if not lines2:
            return {'submitted': False, 'completed': False, 'reason': 'no_lines'}

        # 2 计算标题
        titles = {x.split(',')[2] for x in lines2}
        if len(titles) == 1:
            title = titles.pop()
        else:
            title = '返款表'

        # 3 按对应文件名保存
        today = datetime.date.today()
        subdir_name = f'{today.year}年{today.month:02}月'
        dfmt = today.strftime('%y%m%d')
        file = cls.root / '返款表' / subdir_name / f'd{dfmt}-{title}.csv'
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('\n'.join(lines2))

        # 4 自动执行返款
        if weipay is None:
            weipay = Weipay(['考勤后台'])
        return weipay.request_file_refund(file)
