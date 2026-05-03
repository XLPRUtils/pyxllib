"""数据库与 WPS 总表相关实现。"""

from .common import *  # noqa: F403

class KqDb(XlprDb):
    """ 考勤数据库类 """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __1_课程课次配置(self):
        pass

    def add_course(self,
                   course_name,
                   start_date,
                   message=None,
                   *,
                   end_date=None,
                   course_id2=None,
                   course_type=None,
                   shop_id=None):
        """ 添加课程数据

        :param course_name: 注意course_name最好经过精心设计，不要有重复值
        :param start_date: 课程开始日期
        :param message: 课程相关的数据描述字典，各种不同的课程，可以有特殊的适配格式
        :param end_date: 课程结束日期，有些课程可以自动计算，无法计算的也可以作为可选参数，不填
        :param course_id2: 小鹅通的课程id
        :param course_type: 课程类型，一般可以从course_name自动推算
        :param shop_id: 店铺id，一般可以从course_name自动推算
            觉观、念住、梵呗 为 shop_id=1
            禅宗 为 shop_id=2
        """
        # course_name允许已存在，如果已经存在，由插入改为更新操作
        pass

    def browser_course(self, course_id):
        """ 浏览课程，查看课程当前的观看情况 """
        pass

    def _standardize_lesson_names(self, df):
        r""" 标准化课程名称
        自动转换规则：
        "第(\d{2})届觉观.+?-(\d+)" -> "d260101第\1届觉观-第{\2:02}课"
        "2603堂1-xxx" -> "d260309梵呗初阶-第01课"
        "2604增益堂1-xxx" -> "d260409梵呗增益-第01课"
        其中日期前缀取自该届课程最早的start_date
        """
        if df.empty or 'lesson_name' not in df.columns or 'start_date' not in df.columns:
            return df

        # 1. 提取批次信息并计算每批次的起始日期
        # 这里的批次标识暂定为"第xx届" + 课程类型
        batch_start_dates = {}

        def parse_course_batch(name):
            """提取课程批次和课次信息。"""
            if not isinstance(name, str):
                return None

            name = name.strip()
            if not name:
                return None

            if m := re.match(r'^第(\d+)届(觉观|念住).*?-(\d+)$', name):
                return {
                    'batch_id': m.group(1),
                    'course_type': m.group(2),
                    'lesson_num': int(m.group(3)),
                }

            if m := re.match(r'^(?:20\d{4})?(觉观|念住).*?[（\(](\d+)[）\)].*?-(\d+)$', name):
                return {
                    'batch_id': m.group(2),
                    'course_type': m.group(1),
                    'lesson_num': int(m.group(3)),
                }

            # 梵呗初阶这批课程当前在小鹅通里是“2603堂1-xxx”这种命名。
            if m := re.match(r'^(?P<batch>\d{4})堂(?P<lesson>\d+)(?:-|$)', name):
                return {
                    'batch_id': m.group('batch'),
                    'course_type': '梵呗初阶',
                    'lesson_num': int(m.group('lesson')),
                }

            # 梵呗增益这批课程当前在小鹅通里是“2604增益堂1-xxx”这种命名。
            if m := re.match(r'^(?:20)?(?P<batch>\d{4})增益堂(?P<lesson>\d+)(?:-|$)', name):
                return {
                    'batch_id': m.group('batch'),
                    'course_type': '梵呗增益',
                    'lesson_num': int(m.group('lesson')),
                }

            return None

        for _, row in df.iterrows():
            name = row['lesson_name']
            info = parse_course_batch(name)
            if not info:
                continue

            key = (info['batch_id'], info['course_type'])
            s_date = row['start_date']

            if pd.isna(s_date):
                continue

            if key not in batch_start_dates or s_date < batch_start_dates[key]:
                batch_start_dates[key] = s_date
        
        # 2. 应用转换
        def convert(row):
            name = row['lesson_name']
            info = parse_course_batch(name)
            if not info:
                return name

            key = (info['batch_id'], info['course_type'])
            if key in batch_start_dates:
                date_prefix = batch_start_dates[key].strftime('%y%m%d')
                lesson_num = info['lesson_num']
                course_type = info['course_type']

                if course_type in ('觉观', '念住'):
                    new_name = f"d{date_prefix}第{info['batch_id']}届{course_type}-第{lesson_num:02d}课"
                elif course_type == '梵呗初阶':
                    new_name = f"d{date_prefix}梵呗初阶-第{lesson_num:02d}课"
                elif course_type == '梵呗增益':
                    new_name = f"d{date_prefix}梵呗增益-第{lesson_num:02d}课"
                else:
                    return name

                return new_name.replace(' ', '')

            return name

        df['lesson_name'] = df.apply(convert, axis=1)
        return df

    def _preprocess_lesson_data(self, df):
        """ 预处理课程数据：补全缺失值、修正逻辑错误
        目前包含：
        1. 觉观/念住系列课程：如果 end_date 为空，自动设置为 next_update + 5/7天
        2. 梵呗增益：前20课回放5天，最后2课无回放，结束时间取直播结束时间
        """
        if df.empty: return df

        def _is_missing_end_date(value):
            return pd.isna(value) or str(value).strip() == '' or str(value).lower() in ('nat', 'nan', 'none')

        # 1. 补全 end_date
        def fill_end_date(row):
            name = str(row['lesson_name'])
            val = row.get('end_date')
            if not _is_missing_end_date(val):
                return val

            # 觉观/念住使用 next_update + 固定天数
            offset_days = 0
            if '觉观' in name:
                offset_days = 5
            elif '念住' in name:
                offset_days = 7

            if offset_days > 0 and pd.notna(row.get('next_update')):
                try:
                    nu = pd.to_datetime(row['next_update'])
                    new_end = nu + timedelta(days=offset_days)
                    logger.info(f"自动补全 end_date: {name} -> {new_end} (next_update + {offset_days} days)")
                    return new_end
                except Exception as e:
                    logger.error(f"自动补全 end_date 失败: {name}, next_update={row.get('next_update')}, error={e}")

            # 梵呗增益前20课默认有5天回放，最后2课无回放
            if '梵呗增益' in name:
                try:
                    start_dt = pd.to_datetime(row['start_date'])
                    lesson_num = int(re.search(r'-第(\d+)课$', name).group(1))
                    if lesson_num <= 20:
                        new_end = (start_dt + timedelta(days=5)).replace(hour=23, minute=59, second=59, microsecond=0)
                        logger.info(f"自动补全 end_date: {name} -> {new_end} (start_date + 5 days)")
                        return new_end

                    if pd.notna(row.get('video_duration')):
                        new_end = start_dt + timedelta(seconds=int(row['video_duration']))
                        logger.info(f"自动补全 end_date: {name} -> {new_end} (start_date + video_duration)")
                        return new_end

                    if pd.notna(row.get('next_update')):
                        new_end = pd.to_datetime(row['next_update'])
                        logger.info(f"自动补全 end_date: {name} -> {new_end} (live-only next_update)")
                        return new_end
                except Exception as e:
                    logger.error(f"自动补全 end_date 失败: {name}, start_date={row.get('start_date')}, error={e}")

            return row.get('end_date')

        # 应用补全逻辑
        # 使用 apply 逐行处理，确保逻辑覆盖
        df['end_date'] = df.apply(fill_end_date, axis=1)
        
        return df

    def _validate_lessons_logic(self, df):
        """ 校验课次数据的逻辑一致性 """
        if df.empty: return

        # 按批次分组校验
        # 提取批次名，例如 "d260101第42届觉观"
        # 假设名称格式已经标准化为：d{date_prefix}第{batch_id}届{type}-第{lesson_num}课
        
        # 临时添加辅助列
        def parse_info(name):
            m = re.match(r'^(d\d{6}.+?)-第(\d+)课$', name)
            if m:
                return m.group(1), int(m.group(2))
            return None, None
            
        # 这里只校验符合标准格式的数据，不符合的已经在前面报错了或者被忽略
        parsed = df['lesson_name'].apply(parse_info)
        df['_batch_name'] = parsed.apply(lambda x: x[0])
        df['_lesson_num'] = parsed.apply(lambda x: x[1])
        
        error_msgs = []
        
        # 按批次分组处理
        for batch_name, group in df.groupby('_batch_name'):
            if pd.isna(batch_name): continue
            
            # 按课次号排序
            group = group.sort_values('_lesson_num')
            lessons = group.to_dict('records')
            
            # 1. 校验课次连续性
            nums = [x['_lesson_num'] for x in lessons]
            # 检查是否从1开始（可选？用户没强制说必须从1开始，但通常是）
            # 检查是否连续
            for i in range(len(nums) - 1):
                if nums[i+1] != nums[i] + 1:
                    error_msgs.append(f"批次【{batch_name}】课次不连续：第{nums[i]}课 -> 第{nums[i+1]}课")
            
            # 2. 校验日期逻辑
            for i in range(len(lessons)):
                curr = lessons[i]
                
                # 校验 start_date 连续性（如果是每日一课）
                # 觉观通常是每日一课，但也可能有例外，用户要求：第1课是1号，第10课是10号
                # 这意味着 start_date - 第1课start_date = (lesson_num - 1) days
                if i > 0:
                    first_lesson = lessons[0]
                    expected_date = first_lesson['start_date'] + timedelta(days=curr['_lesson_num'] - first_lesson['_lesson_num'])
                    if curr['start_date'].date() != expected_date.date():
                        error_msgs.append(f"批次【{batch_name}】第{curr['_lesson_num']}课 start_date 异常：期望 {expected_date.date()}，实际 {curr['start_date'].date()}")
                
                # 3. 校验 end_date
                # 觉观/念住可以在next_update的基础上加5/7天
                offset_days = 0
                if '觉观' in batch_name:
                    offset_days = 5
                elif '念住' in batch_name:
                    offset_days = 7
                    
                if offset_days > 0:
                    if pd.isna(curr['next_update']):
                        # 如果 next_update 为空，可能需要根据业务补全？或者报错？
                        # 暂时跳过基于next_update的检查，或者假设next_update存在
                        pass
                    else:
                        # 注意 next_update 在数据库读取出来可能是 str 或 datetime
                        # 这里 df 是从 sql_select 出来的，可能是 str
                        next_update_dt = pd.to_datetime(curr['next_update'])
                        expected_end = next_update_dt + timedelta(days=offset_days)
                        
                        if pd.isna(curr['end_date']):
                            # 如果为空，自动补全（修改原df）
                            # 注意：这里修改的是切片副本还是原df？groupby通常是副本，需要回写
                            # 简单起见，直接定位修改原df
                            idx = group.index[i]
                            df.at[idx, 'end_date'] = expected_end
                            logger.info(f"自动补全 end_date: {batch_name} 第{curr['_lesson_num']}课 -> {expected_end} (next_update + {offset_days} days)")
                        else:
                            # 如果有值，检查是否符合
                            curr_end = pd.to_datetime(curr['end_date'])
                            # 允许一定误差？还是严格相等？用户说“要检查是否符合这些逻辑”
                            # 考虑到时间部分可能不同，只比较日期？或者允许几秒误差？
                            # 觉观/念住的 end_date 通常是精确的
                            if abs((curr_end - expected_end).total_seconds()) > 3600: # 允许1小时误差？
                                error_msgs.append(f"批次【{batch_name}】第{curr['_lesson_num']}课 end_date 异常：期望 {expected_end}，实际 {curr_end}")

        # 清理临时列
        df.drop(columns=['_batch_name', '_lesson_num'], inplace=True)
        
        if error_msgs:
            for msg in error_msgs:
                logger.error(msg)
            raise ValueError(f"数据逻辑校验失败，共发现 {len(error_msgs)} 处错误，请检查日志。")

    def add_book_lessons_to_db(self, data_row=4, commit=True, df=None, check_name_format=True):
        """ 将在线表格课次数据清单添加到数据库
        
        功能包含：
        1. 获取或使用传入的课次数据
        2. 数据预处理：
           - 按 start_date 升序排序
           - 自动标准化课程名称（dYYMMDD第xx届觉观/念住-第xx课）
           - 自动补全缺失数据（如觉观/念住系列自动计算 end_date）
        3. 数据校验：
           - 校验课程名称格式
           - 校验课次连续性
           - 校验日期逻辑（start_date 连续性、end_date 准确性）
        4. 数据库操作：
           - 检查数据库中是否存在重复 lesson_name
           - 插入新数据
        
        :param data_row: 在线表格数据起始行，默认4
        :param commit: 是否提交事务，默认为True。设为False可用于测试校验逻辑
        :param df: 可选，直接传入 DataFrame 数据，用于测试
        :param check_name_format: 是否校验课程名称格式，默认为True
        """
        # 1 获取待新建的课次数据
        if df is None:
            wb = KqBook()
            df = wb.sql_select('课次数据', ['shop_id', 'lesson_id2', 'lesson_name',
                                            'start_date', 'next_update', 'video_duration', 'end_date'],
                               data_row)

        # 1.2 按日期排序
        if not df.empty and 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'])
            df.sort_values(by='start_date', inplace=True)
            
            # 1.3 名称自动标准化转换
            df = self._standardize_lesson_names(df)
            
            # 1.3.2 预处理：补全缺失值
            df = self._preprocess_lesson_data(df)
            
        # 1.4 名称格式校验
        if check_name_format and not df.empty:
            invalid_names = []
            for name in df['lesson_name']:
                if not re.match(r'd\d{6}.+?-第\d{2}课', name):
                    invalid_names.append(name)
            
            if invalid_names:
                logger.error(r"以下lesson_name不符合规范(d\d{6}.+?-第\d{2}课)：")
                for name in invalid_names:
                    logger.error(name)
                raise ValueError("存在不符合规范的lesson_name，程序终止。")

        # 1.5 逻辑一致性校验
        self._validate_lessons_logic(df)

        # 2 全局重复检查
        # 获得已有的lesson_names
        exists_names = set(self.exec2col('SELECT lesson_name FROM lesson_table'))

        # 新的lesson_name推荐按照"课程名+编号"的模式，确保全局名称唯一，否则可能有风险
        # 其实觉观等系列命名就是规范的，但有些网课不是那么规范，容易产生歧义，最好加上唯一标识前缀

        # 将df中的lesson_name和exists_names合并为一个列表
        all_lesson_names = list(df['lesson_name']) + list(exists_names)

        # 检查是否有重复的lesson_name
        name_counter = Counter(all_lesson_names)
        duplicates = {name: count for name, count in name_counter.items() if count > 1}

        if duplicates:
            # 如果有重复，打印日志并抛出错误
            logger.warning("发现重复的lesson_name，请检查以下名称：")
            for name, count in duplicates.items():
                logger.warning(f"名称: {name}, 重复次数: {count}")
            raise ValueError("存在重复的lesson_name，程序终止。")

        # 4 如果没有重复，进入for循环插入数据
        for _, row in df.iterrows():
            row2 = row.to_dict()
            if not row2['end_date']:
                del row2['end_date']
            
            if commit:
                self.insert_row('lesson_table', row2, commit=-1)
            else:
                logger.info(f"待写入数据: {row2}")
        
        if commit:
            self.commit_all()
        else:
            logger.warning("注意：当前为测试模式(commit=False)，数据未写入数据库。")

    def 添加禅宗课次配置数据(self, course_name, week_num, courses):
        """
        :param course_name: 课程名称，例如'd250629禅宗9期一阶'
        :param week_num: 持续周数
        :param courses: 课次数据
        :return:


        Q：如果我新写入"第15周"的数据，虽然数据库已有第15周，但是我lesson_name有点变化了，不会覆盖掉已有的第15周数据吧？
        A：这取决于你修改后的名称和数据库里原有名称的包含关系。add_book_lessons_to_db
        
            这段代码使用的不是“完全相等”匹配，而是**“包含匹配”（且逻辑是：看数据库里的旧名字是否包含在新传入的名字**里）。
        """
        from datetime import datetime, timedelta

        # 1 配置参数
        # 从course_name提取出start_date
        formatted_str = f"20{course_name[1:3]}-{course_name[3:5]}-{course_name[5:7]}"
        start_date = datetime.strptime(formatted_str, "%Y-%m-%d")
        # print(start_date)  # 输出: 2025-06-29 00:00:00
        end_date = start_date + timedelta(weeks=week_num)

        # 2 遍历课程数据
        def update_row(week_id, prefix, name, lesson_id2, video_duration):
            # 1 配置数据条目
            row = {
                'start_date': (start_date + timedelta(weeks=week_id - 1)).strftime("%Y%m%d %H:%M:%S"),
                'end_date': end_date.strftime("%Y%m%d %H:%M:%S"),
                'next_update': (start_date + timedelta(weeks=week_id)).strftime("%Y%m%d %H:%M:%S"),
                'lesson_id2': lesson_id2,
                'lesson_name': f'{prefix}={name}',
                'shop_id': 2,
            }
            if video_duration:
                row['video_duration'] = video_duration * 60  # 分钟转换成秒数

            # 2 提前抓取所有匹配前缀的数据库条目
            existing_lessons = self.execute("SELECT lesson_id, lesson_name FROM lesson_table "
                                            f"WHERE lesson_name LIKE '{prefix}=%'").fetchall()

            # 3 查找匹配的条目
            def is_name_match(full_name, db_name):
                """
                检查完整名称是否匹配数据库中的精简名称
                参考数字匹配的严谨性要求
                """
                # 如果数据库名称不在完整名称中，直接返回False
                if db_name not in full_name:
                    return False

                # 检查数字匹配的严谨性
                # 找到db_name在full_name中的位置
                start_pos = full_name.find(db_name)
                if start_pos == -1:
                    return False

                end_pos = start_pos + len(db_name)

                # 检查db_name后面是否直接跟着数字（这种情况不算匹配）
                if end_pos < len(full_name) and full_name[end_pos].isdigit():
                    # 还需要检查db_name的最后一个字符是否是数字
                    if db_name and db_name[-1].isdigit():
                        return False

                return True

            matched_lesson_id = None
            current_lesson_name = row['lesson_name']  # 完整的原始名称

            for lesson_id, db_lesson_name in existing_lessons:
                # 提取数据库中的name部分（去掉prefix=前缀）
                if '=' in db_lesson_name:
                    db_name = db_lesson_name.split('=', 1)[1]  # 只按第一个=分割

                    # 检查当前lesson_name是否包含数据库中的name，并考虑数字匹配的严谨性
                    if is_name_match(current_lesson_name, db_name):
                        matched_lesson_id = lesson_id
                        break

            # 4 根据匹配结果决定插入还是更新
            if matched_lesson_id:
                row['lesson_name'] = db_lesson_name
                self.update_row('lesson_table', row, {'lesson_id': matched_lesson_id})
            else:
                self.insert_row('lesson_table', row)
            self.commit_all()

        for week_tag, week_courses in courses.items():
            week_tag = chinese2digits(week_tag)
            week_id = int(re.search(r'第(\d+)周', week_tag).group(1))
            for course in week_courses:
                if len(course) == 2:
                    name, lesson_id2 = course
                    video_duration = None
                else:
                    name, video_duration, lesson_id2 = course

                prefix = f'{course_name}-第{week_id}周'
                update_row(week_id, prefix, name, lesson_id2, video_duration)

    def __2_用户数据(self):
        pass

    @staticmethod
    def parse_timedesc_to_seconds(timedelte_desc):
        seconds = 0
        m0 = re.search(r'(\d+)小时', timedelte_desc)
        if m0:
            seconds += int(m0.group(1)) * 3600
        m1 = re.search(r'(\d+)分钟', timedelte_desc)
        if m1:
            seconds += int(m1.group(1)) * 60
        m2 = re.search(r'(\d+)(秒|$)', timedelte_desc)
        if m2:
            seconds += int(m2.group(1))
        return seconds

    def update_user_table_from_file(self, file, shop_id=1):
        """ 读取数据，更新到数据库 """

        zhname2en = {
            '用户ID': 'user_id2',
            '昵称': 'user_nickname',
            '来源渠道': 'from_channel',
            '姓名': 'name',
            '账户绑定手机号': 'bind_phone',
            '最近采集手机号': 'collect_phone',
            '账号状态': 'is_seal',
            '邮箱': 'wx_email',
            '微信号': 'wx_account',
            '年龄': 'age',
            '省份': 'province',
            '行业': 'industry',
            '兴趣': 'interest',
            '店铺标签': 'app_id',
            '所属推广员': 'promoter',
            '职位': 'job',
            '地址': 'address',
            '国家': 'country',
            '性别': 'gender',
            '生日': 'birth',
            '语言': 'language',
            '城市': 'city',
            '公司': 'company',
            '购买商品': 'buy_goods',
            '今日学习时长': 'today_study_time',
            '首次支付时间': 'first_pay_time',
            '累计消费金额': 'consume_money',
            '最近购买商品': 'last_buy_goods',
            '累计学习时长': 'total_study_time',
            '最近消费时间': 'total_consume_time',
            '最近访问时间': 'last_visited_at',
            '完成课程数': 'finish_course_num',
            '最近收货手机号': 'last_receive_phone',
            '备注名': 'remark_name',
            '注册时间': 'user_created_at',
            '累计消费次数': 'consume_times',
            '描述': 'desc',
            '累计有效消费金额': 'valid_consume_money',
            '总积分': 'total_score',
            # '优惠券总数': 'total_coupon_num',
            # '优惠码总数': 'total_voucher_num',
            # '兑换码总数': 'total_exchange_num',
            # '邀请码总数': 'total_invite_num',
        }

        def 读取用户导出表(path):
            with open(path, 'rb') as fh:
                if fh.read(4).startswith(b'PK\x03\x04'):
                    raise ValueError(f'用户导出文件不是CSV，疑似误下载了Excel文件：file={path}')

            last_columns = []
            for attempt in range(1, 4):
                try:
                    df = pd.read_csv(path, low_memory=False)
                except UnicodeDecodeError:
                    df = pd.read_csv(path, low_memory=False, encoding='gbk')

                df.columns = [str(x).replace('\ufeff', '').strip() for x in df.columns]
                last_columns = list(df.columns)
                if '用户ID' in df.columns:
                    return df

                if attempt < 3:
                    logger.warning(f'用户导出文件缺少关键列，准备重试读取 {attempt}/3：file={path} columns={last_columns}')
                    time.sleep(2)

            raise KeyError(f'用户导出文件缺少关键列“用户ID”：file={path} columns={last_columns}')

        df = 读取用户导出表(file)
        if '注册时间' in df.columns:
            df.sort_values('注册时间', ascending=True, inplace=True)
        elif '序号' in df.columns:
            logger.warning(f'用户导出文件缺少“注册时间”列，改按“序号”排序：file={file}')
            df.sort_values('序号', ascending=True, inplace=True)
        else:
            logger.warning(f'用户导出文件缺少“注册时间/序号”列，保留原始顺序：file={file}')

        if '累计学习时长' in df.columns:
            df['累计学习时长'] = df['累计学习时长'].map(self.parse_timedesc_to_seconds)
        if '今日学习时长' in df.columns:
            df['今日学习时长'] = df['今日学习时长'].map(self.parse_timedesc_to_seconds)

        # 从旧的账号到新的账号逐步往数据库加，可能是插入，也可能是更新
        exists_users = set(self.exec2col(f'SELECT user_id2 FROM user_table WHERE shop_id={shop_id}'))
        for idx, row in tqdm(df.iterrows(), total=len(df), desc='添加/更新用户数据'):
            row2 = {'shop_id': shop_id}
            # 把不是pd.isna的数值都加到row2
            for k, v in row.items():
                if k in ('序号', '描述'):  # todo 以后要研究下这个“描述”怎么添加数据库
                    continue
                if not pd.isna(v):
                    if k in ('累计消费金额', '累计有效消费金额'):
                        v = str(v).replace(',', '')
                    if k in zhname2en:
                        row2[zhname2en[k]] = v.strip() if isinstance(v, str) else v
                    else:
                        pass  # todo 需要提示用户，有新字段没有处理
                    # 使用这段的话可以支持额外扩展的字段，不过有风险，不建议采用。而且user_table里暂时没有预设extra字段。
                    # if k in zhname2en:
                    #     row2[zhname2en[k]] = v.strip() if isinstance(v, str) else v
                    # else:
                    #     row2['extra'][k] = v.strip() if isinstance(v, str) else v
            if row2['user_id2'] in exists_users:
                self.update_row('user_table', row2,
                                {'user_id2': row2['user_id2']}, commit=-1)
            else:
                self.insert_row('user_table', row2, commit=-1)
            if idx % 1000 == 0:
                self.commit_all()
        self.commit_all()

    def get_last_user_ids(self, shop_id=1):
        """ 获取数据库中，指定店铺最新的已存在用户 """
        sql_str = f'SELECT user_id2 FROM user_table ' \
                  f'WHERE user_created_at = (SELECT MAX(user_created_at) FROM user_table where shop_id={shop_id})'
        user_ids = set(self.exec2col(sql_str))
        return user_ids

    def search_users(self, phone=None, name=None, nickname=None, phone2=None, shop_id=None):
        """ 在数据库里搜索用户

        :param name: 一般是真实姓名
        :param nickname: 其他昵称
        :param phone: 手机号
        :param phone2: 候选匹配手机号

        240413周六23:57，可以用这个检查后台手机号是否重复：
        SELECT shop_id,bind_phone, COUNT(1) FROM user_table GROUP BY shop_id,bind_phone ORDER BY COUNT(1) DESC;

        理论上不同店铺，手机号是不会重复的

        240414周日00:07，感觉用手机号就可以匹配，根本没必要这么复杂~ 但是这个函数先留着吧
        """
        # 这里为了准确性，牺牲一定的速度性能，尽量进行了全量遍历
        sql = 'SELECT user_id2,user_nickname,name,' \
              'bind_phone,collect_phone,last_receive_phone,' \
              'is_seal FROM user_table'
        if shop_id:
            sql += f' WHERE shop_id={shop_id}'

        data = self.exec2dict(sql)
        ls = []  # 有权重的匹配数据
        for row in data:
            # 有可能会有这种格式的手机号：+49-15902155569，所以使用"in"语法判断匹配
            score = 0
            if phone:  # 手机号匹配是最高权重: 100
                if phone in row['bind_phone']:
                    score += 100
                if phone in row['collect_phone']:
                    score += 50
                if phone in row['last_receive_phone']:
                    score += 10

            if phone2:  # 候选手机号相对权重减半: 50
                if phone2 in row['bind_phone']:
                    score += 50
                if phone2 in row['collect_phone']:
                    score += 25
                if phone2 in row['last_receive_phone']:
                    score += 5

            if name:  # 30
                if row['name'] == name:
                    score += 30
                if row['user_nickname'] == name:
                    score += 10

            if nickname:  # 20
                if row['user_nickname'] == nickname:
                    score += 20
                if row['name'] == nickname:
                    score += 5

            if score:
                if row['is_seal'] == '已注销':
                    score -= 200
                elif row['is_seal'] == '待激活':
                    score -= 100
                row['score'] = score
                ls.append(row)

        ls.sort(key=lambda x: x['score'], reverse=True)
        return ls

    def insert_new_users(self, users):
        """ 从表格文件中添加用户 """
        tprint(f'数据库添加{len(users)}个新用户')
        tt = TicToc()
        for user in users:
            self.insert_row('user_table', user, commit=-1)
            if tt.tocvalue() > 10:
                self.commit_all()
                tt.tic()
        self.commit_all()

    def browser_users(self, user_id2s):
        """ 查看特定清单的用户数据 """
        user_id2s2 = [f"'{x}'" for x in user_id2s]
        df = self.exec2df(f'SELECT user_id2, name, user_nickname, bind_phone, user_created_at FROM user_table '
                          f"WHERE user_id2 IN ({','.join(user_id2s2)})")
        return df

    def __x_查找用户(self):
        pass

    def 查看指定用户的观看记录(self, user_id2):
        # 生成sql，在lesson_data_table里，找user_id2，且按lesson_id去重的数据
        # lesson_id可以在lesson_table从lesson_id映射到lesson_name的名称（使用join in直接衔接）
        # 最终返回清单，每条是lesson_name，count
        sql = f"""SELECT DISTINCT lt.lesson_name, ldt.lesson_id
FROM lesson_data_table ldt
JOIN lesson_table lt ON ldt.lesson_id = lt.lesson_id
WHERE ldt.user_id2 = '{user_id2}'
ORDER BY ldt.lesson_id DESC;
"""
        ls = self.execute(sql).fetchall()
        ls = [x[0] for x in ls]
        return ls

    def 用户信息摘要(self, x):
        """ 输入series类型的一个条目，输出其摘要信息

        :param x: 可以输入已经从用户列表提取到的字典信息，而可以是user_id2的字符串值
        """

        def fmt(k, v):
            if v is None:
                return ''
            elif isinstance(v, float) and math.isnan(v):
                return ''
            else:
                if k == 'buy_goods':
                    v = v.split(',')
                    v = '\n\t'.join(v)
                    return f'{k}={v}'
                elif k == '观看记录':
                    v = '\n\t'.join(v)
                return f'{k}={v}'

        if isinstance(x, str):
            x = self.exec2dict(f"SELECT * FROM user_table WHERE user_id2='{x}'").fetchone()
        if '观看记录' not in x:
            x['观看记录'] = self.查看指定用户的观看记录(x['user_id2'])
        ls = [fmt(k, v) for k, v in x.items()]

        return '\n'.join([x for x in ls if x])

    @classmethod
    def 标准化昵称手机号参数(cls, 昵称, 手机号):
        def tolist(x):
            if x and not isinstance(x, list):
                x = [x]
            x = [str(a) for a in x if a]
            return x

        昵称 = tolist(昵称)

        手机号 = tolist(手机号)
        手机号 = [f'{x}'.lstrip('`') for x in 手机号 if (x and x != 'None')]
        手机号 = [x for x in 手机号 if x]

        return 昵称, 手机号

    def 查找用户(self, 昵称='', 手机号='', 课程标准名='', 课程商品名='', shop_id=1, return_mode=2):
        """
        :param str|list[str] 昵称: 昵称可以输入一个列表，会在目标昵称、真实姓名同时做检索
            也支持输入user_id2，进行用户id的精确匹配
        :param str|list[str] 手机号: 手机也可以输入一个列表，因为有些人可能报名时手机号填错，可以增加一些匹配规则
        :param str 课程商品名: 本次课程售卖的名称，可选参数，如果输入可以提高匹配精确度权重
        :param return_mode:
            1，仅第1项匹配的user_id2和权重，如果为空则返回['', 0]
            2，匹配全文
        :return:
        """

        # 1 统一输入参数格式
        昵称, 手机号 = self.标准化昵称手机号参数(昵称, 手机号)

        # 2 查找所有可能匹配的项目
        # 在数据库self.kqdb的user_table检索，取出bind_phone、collect_phone和`手机号`有重叠的所有条目
        rows = self.exec2dict('SELECT * FROM user_table '
                              f'WHERE ARRAY[bind_phone, collect_phone] && ARRAY{手机号} '
                              f'AND is_seal != \'已注销\' '
                              f'AND shop_id={shop_id}').fetchall()

        if not rows:
            return '', -1

        # 暂时跳过候选项不唯一的
        if len(rows) > 1:
            return '', len(rows)

        x = rows[0]
        return x['user_id2'], 90

    def __3_各网课定制的不同进度算法(self):
        pass

    def delete_lesson_data_items(self, lesson_data_ids):
        """ 删除给定标记的数据 """
        if lesson_data_ids:
            tag = ','.join(map(str, lesson_data_ids))
            self.execute(f'DELETE FROM lesson_data_table WHERE lesson_data_id IN ({tag})')

    def update_lesson_data_item(self, item):
        """ 更新某个条目的数据 """
        data = {}
        for c in ['stay_seconds', 'cum_seconds', 'studio_seconds', 'playback_seconds', 'study_state', 'progress']:
            data[c] = item[c]
        self.update_row('lesson_data_table', data,
                        {'lesson_data_id': item['lesson_data_id']})

    def user_study_result_a(self, lesson, items, reduce_db=True):
        """ A类普通课程的进度计算 """
        # 1 找到进度最大的一条
        # 1.1 排序，填充默认值
        items.sort_values('update_time', inplace=True)
        custom_fillna(items, 0, numeric_fill_value=0)

        # 1.2 确保studio_seconds、playback_seconds只能单调递增
        last_studio_seconds = 0
        last_playback_seconds = 0

        for idx in items.index:
            # 获取当前行的值
            curr_studio = items.at[idx, 'studio_seconds']
            curr_playback = items.at[idx, 'playback_seconds']

            # 检查是否需要更新studio_seconds
            if curr_studio < last_studio_seconds:
                items.at[idx, 'studio_seconds'] = last_studio_seconds
            else:
                last_studio_seconds = curr_studio

            # 检查是否需要更新playback_seconds
            if curr_playback < last_playback_seconds:
                items.at[idx, 'playback_seconds'] = last_playback_seconds
            else:
                last_playback_seconds = curr_playback

            # 重新计算cum_seconds
            items.at[idx, 'cum_seconds'] = items.at[idx, 'studio_seconds'] + items.at[idx, 'playback_seconds']

        # 1.3 计算出每条的progress
        total_seconds = lesson['video_duration']  # 视频完整时长
        items['progress'] = items['cum_seconds'].apply(lambda x: int(x / total_seconds * 100))

        # 1.4 找出进度最大的条目
        max_progress_item = items.loc[items['progress'].idxmax()]

        # 2 还未达到指标
        if max_progress_item['progress'] < 50:
            if reduce_db:
                ids = set(items['lesson_data_id']) - {max_progress_item['lesson_data_id']}
                if ids:
                    self.delete_lesson_data_items(ids)
                    self.update_lesson_data_item(max_progress_item)
                    self.commit_all()
            return f'学习中/{max_progress_item["progress"]}%'

        # 3 找到最早满足指标的条目
        first_finished_item = items[items['progress'] >= 50].iloc[0]
        # 最早达标和最大进度不是同一条时，需要把最大进度的信息更新到最早一条
        if first_finished_item['lesson_data_id'] != max_progress_item['lesson_data_id']:
            for c in ['stay_seconds', 'cum_seconds', 'studio_seconds', 'playback_seconds', 'study_state', 'progress']:
                first_finished_item[c] = max_progress_item[c]

        # 4 更新数据库
        if reduce_db:
            ids = set(items['lesson_data_id']) - {first_finished_item['lesson_data_id']}
            if ids:
                self.delete_lesson_data_items(ids)
                self.update_lesson_data_item(first_finished_item)
                self.commit_all()

        # 5 计算并返回完成进度
        item = first_finished_item

        # 当堂完成
        if item['studio_seconds'] / total_seconds >= 0.5:
            return f'当堂完成/{max_progress_item["progress"]}%'

        # 回放完成
        dt1 = lesson['start_date'] + timedelta(seconds=lesson['video_duration'])  # 直播结束时间
        dt2 = item['update_time']
        delta_d = max(int((dt2 - dt1).total_seconds() / (3600 * 24)), 1)
        return f'第{int(delta_d)}天回放/{max_progress_item["progress"]}%'

    def user_study_result_b(self, lesson, items, reduce_db=True):
        """ B类的禅宗课程的进度计算

        1、找出进度最大，且时间最早的条目，这条记为x
        2、除了x，其他items都可以删除
        3、根据x的时间节点，参考lesson的数据，计算是"准时完成"，还是"延几周完成"
        4、没完成的课也另外显示进度情况

        注：要确保每周都有采集到数据，否则5月的课，6月才采集，不会按照5月只是漏采集然后当周完成，而是算成延期1个月
            如果确实出现这种要补救的情况，可以在上游线修改数据库数据。修改器对应的update_time
        """
        # 1 数据预处理

        # 排序，填充默认值
        items.sort_values('update_time', inplace=True)
        custom_fillna(items, 0, numeric_fill_value=0)  # d250629, 这步运行非常慢，但我现在不敢贸然去掉

        # 如果播放时间超过半小时，或者study_state显示已完成，进度也强制改为100
        def 更新单条进度(item):
            # 这段isna的判断应该是不需要的，但以防万一先留着
            if pd.isna(item['cum_seconds']):
                item['cum_seconds'] = 0
            if pd.isna(item['progress']):
                item['progress'] = 0

            if item['progress'] < 100 and (item['cum_seconds'] >= 1800 or '已完成' in item['study_state']):
                item['progress'] = 100

            return item

        items = items.apply(更新单条进度, axis=1)

        # 2 找最早进度达到100%的条目
        x = items.loc[items['progress'].idxmax()]

        # 删除除了x外的条目
        if reduce_db:
            ids = set(items['lesson_data_id']) - {x['lesson_data_id']}
            if ids:
                self.delete_lesson_data_items(ids)
                self.commit_all()

        # 3 完成情况下的进度展示
        if x['progress'] >= 100:
            start_date = pd.to_datetime(lesson['start_date'])  # 课程开课时间
            update_time = pd.to_datetime(x['update_time'])  # 用户完成时间
            # 计算目标完成时间
            target_complete_date = start_date + pd.Timedelta(days=0.5 + 7)
            diff = update_time - target_complete_date
            # 判断是否准时完成
            if diff.days < 1:  # d251214: 不满足1天的延迟也算准时完成，这种一般是我处理出bug了，第2天补充修复
                return '准时完成'
            else:
                # 此时 diff.days 至少为 1，进入正式的周数计算
                # 此时 delay_days 最小是 1 - 0.5 = 0.5
                # 0.5 // 7 = 0, + 1 = 1 (延1周)，逻辑恢复正常
                delay_days = diff.days - 0.5
                delay_weeks = int(delay_days // 7) + 1  # 向上取整到周
                return f'延{delay_weeks}周完成'

        # 4 未完成情况下的进度展示
        progress, cum_seconds = x['progress'], x['cum_seconds']
        if progress:
            return f'进度{progress}%'
        elif cum_seconds:
            return f'观看{cum_seconds // 60}分钟'
        else:
            return ''  # 未开始

    def user_study_result_c(self, lesson, items, reduce_db=True):
        """ C类的闯关课程的进度计算 """
        # 1 找到进度最大的一条
        # 排序，填充默认值
        items.sort_values('update_time', inplace=True)
        custom_fillna(items, 0, numeric_fill_value=0)

        # 计算出每条的progress
        total_seconds = lesson['video_duration']  # 视频完整时长
        # 完全采用停留时长作为进度，不用小鹅通自带的进度标记
        items['progress'] = items['cum_seconds'].apply(lambda x: int(x / total_seconds * 100))

        # 2 找出进度最大的条目。闯关情况下不用考虑update_time影响
        item = items.loc[items['progress'].idxmax()]
        if reduce_db:
            ids = set(items['lesson_data_id']) - {item['lesson_data_id']}
            if ids:
                self.delete_lesson_data_items(ids)
                self.commit_all()

        # 3 完成进度
        if item['progress'] <= 0:
            return ''

        if item['progress'] < 90:
            return f'学习中/{item["progress"]}%'

        # 进度达到90/150/200，分别算完成1遍/2遍/3遍
        if item['progress'] >= 200:
            times = 3
        elif item['progress'] >= 150:
            times = 2
        else:
            times = 1
        return f'{times}遍/{item["progress"]}%'

    def user_study_result(self, lesson, items, reduce_db=True):
        """ 某个学生在lesson的课次配置上，多条items学习记录汇总后的学习状态描述

        :param lesson: 课次配置信息
        :param items: 学员数据
        :param reduce_db: 是否对数据库的存量数据进行精简，避免存储冗余
        """
        if '禅宗' in lesson['lesson_name']:
            return self.user_study_result_b(lesson, items, reduce_db)
        elif '念住闯关' in lesson['lesson_name']:
            return self.user_study_result_c(lesson, items, reduce_db)
        else:  # 其他是有回放机制的
            return self.user_study_result_a(lesson, items, reduce_db)

    def __4_课程视频进度(self):
        pass

    def update_lesson_data_from_file(self, lesson_id, file, tz=None):
        """ 更新课次数据

        :param lesson_id: 课次id
        :param file: 本地数据文件
        :param tz: 时间，默认就是当前时间
        """
        df = pd.read_csv(file)
        zhname2en = {
            '用户ID': 'user_id2',
            '参与状态': 'study_state',
            '播放进度': 'progress',
            '累计观看时长(秒)': 'cum_seconds',  # shop1的格式
            '累计播放时长（秒）': 'cum_seconds',  # shop2的格式
            '上次播放时间': 'last_play_time',
            '完成时间': 'finish_time',
            '直播间停留时长(秒)': 'stay_seconds',
            '直播观看时长(秒)': 'studio_seconds',
            '回放观看时长(秒)': 'playback_seconds',
            '评论次数': 'comment_times',
            '直播间成交金额': 'money',
        }
        if '累计观看时长(秒)' not in df and ('累计观看时长' in df and '累计播放时长（秒）' not in df):
            df['累计播放时长（秒）'] = df['累计观看时长'].map(self.parse_timedesc_to_seconds)

        if '播放进度' in df:  # 这个字段是可能没有的，没有的时候怎么算已完成，就是下游的事啦，这里可不管
            df['播放进度'] = df['播放进度'].map(lambda x: int(x.strip('%')))

        def parse_time(x):
            if pd.isna(x):
                return None
            return None if ('--' in x) else x.strip()

        for name in ['上次播放时间', '完成时间']:
            if name in df:
                df[name] = df[name].map(parse_time)

        if tz is None:
            tz = utc_timestamp()
        for idx, row in df.iterrows():
            row2 = {'lesson_id': lesson_id, 'update_time': tz}
            for k, v in zhname2en.items():
                if k in row:
                    row2[v] = row[k].strip() if isinstance(row[k], str) else row[k]

            # 修复 last_play_time 可能为 float(nan) 导致 psycopg 报错的问题
            for k in ['last_play_time', 'finish_time']:
                if k in row2 and isinstance(row2[k], float):
                    row2[k] = None

            self.insert_row('lesson_data_table', row2, commit=-1)
        self.commit_all()

    def browser_lesson_data(self, course_name, lesson_names=None, user_id2s=None):
        """ 查看视频课程数据并保持用户ID的顺序及处理空值和无效值

        :param course_name: 获取对应课程的课次数据
        :param lesson_names: 可以限定只获取课程下某些课次的数据，默认获取全部课次
            写的时候不要带 f'{course_name}-' 的前缀
        :param user_id2s: 如果提供了用户清单（允许重复、空行），按照用户清单的行顺序返回df
        """
        # 1 获取课程相关的所有课时数据
        lessons = self.exec2dict('SELECT * FROM lesson_table '
                                 f"WHERE lesson_name LIKE '%{course_name}%' "
                                 f"ORDER BY lesson_id").fetchall()
        for x in lessons:
            x['brief_lesson_name'] = x['lesson_name'].replace(course_name, '')
        if lesson_names is None:
            lesson_names = [lesson['brief_lesson_name'] for lesson in lessons]

        # 2 构建所有课时DataFrame的列表
        dfs = []
        for lesson in lessons:
            if lesson['brief_lesson_name'] in lesson_names:
                # 获得课次的全部数据
                df = self.exec2df('SELECT * '
                                  'FROM lesson_data_table '
                                  f"WHERE lesson_id='{lesson['lesson_id']}'")
                # 按学员分组
                df2 = df.groupby('user_id2')
                # 每个学员的进度情况是单独计算的
                res = df2.apply(lambda items: self.user_study_result(lesson, items), include_groups=False)
                df3 = pd.DataFrame(res, columns=[lesson['brief_lesson_name']])
                dfs.append(df3)

        # 3 如果未提供user_id2s，从数据中提取所有唯一的user_id2
        if user_id2s is None and dfs:
            user_id2s = pd.concat(dfs).index.unique()

        # 4 保持原始user_id2s，即使它们在数据集中不存在
        df_user_id2s = pd.DataFrame(user_id2s, columns=['user_id2'])
        df_user_id2s.reset_index(drop=True, inplace=True)  # 重置索引以保证合并时顺序不变

        # 5 合并所有结果，使用merge保持user_id2s的顺序
        final_df = df_user_id2s
        for df in dfs:
            final_df = final_df.merge(df, on='user_id2', how='left', sort=False)

        custom_fillna(final_df, '', numeric_fill_value='')  # 未开始
        return final_df

    def __5_打卡数据(self):
        pass

    def add_clockin(self, name, url, start_date=None, end_date=None):
        """ 添加一个打卡配置 （主要适配shop2，还不清楚shop1情况，后续再兼容）

        :param name: 取一个名字，最好具有全局唯一性
        :param url: 可以找到这个打卡精确数据所在的url链接

        todo 这个跟课次数据一样，其实也需要配置定期更新的机制的。但目前先手动执行更新，问题也不太大。
        """
        clockin_id = self.exec2one(f'SELECT clockin_id FROM clockin_table WHERE name=%s', [name, ])

        row = {
            'name': name,
            'url': url,
            'start_date': start_date,
            'end_date': end_date
        }
        row = {k: v for k, v in row.items() if (v is not None)}

        if clockin_id:
            self.update_row('clockin_table', row, {'clockin_id': clockin_id})
        else:
            self.insert_row('clockin_table', row)
        self.commit_all()

    def update_clockin_data_from_file(self, name, file):
        """

        :param name: 打卡配置名
        :param file: 本地文件路径
        """
        clockin_id = self.exec2one(r'SELECT clockin_id FROM clockin_table WHERE name=%s', [name, ])
        file = str(file)
        existing_count = self.exec2one(
            'SELECT COUNT(*) FROM clockin_data_table WHERE clockin_id=%s',
            [clockin_id]
        )

        # 1 先确认新文件可读，再覆盖旧数据
        zhname2en = {
            '用户ID': 'user_id2',  # 1 以下是禅宗体系
            '用户昵称': 'nickname',
            '分组': 'groupname',
            '发布时间': 'publish_time',
            '动态内容': 'update_content',
            '动态标题': 'update_title',
            '动态类型': 'update_type',
            '标签': 'tags',
            '阅读人数': 'read_num',
            '点赞数': 'like_num',
            '评论数': 'comment_num',
            '精华主题': 'is_essence',
            '分享次数': 'share_num',
            '动态链接': 'update_url',
            'user_id': 'user_id2',  # 2 以下是梵呗增益打卡
            '是否补打卡': 'is_repair',
            '打卡日历': 'task_date',  # 哪天的打卡数据
            '打卡时间': 'publish_time',
            '所属主题': 'update_title',
            '日记链接': 'update_url',
            '文字内容': 'update_content',
            '是否精选': 'is_essence',
            '所属作业': 'update_title',
            '用户id': 'user_id2',  # 3 念住打卡

            '任务名称': 'update_title',
        }

        try:
            df = pd.read_excel(file)
        except ValueError as e1:
            try:
                df = pd.read_csv(file)
            except ValueError as e2:
                logger.info(f"Excel读取失败: {file}, {e1}")
                logger.info(f"CSV读取失败: {file}, {e2}")
                return  # 结束函数执行
            except Exception as e2:
                logger.info(f"Excel读取失败: {file}, {e1}")
                logger.info(f"CSV读取异常: {file}, {e2}")
                return
        except Exception as e1:
            logger.info(f"Excel读取异常: {file}, {e1}")
            return

        recognized_cols = [col for col in df.columns if col in zhname2en]
        if not recognized_cols:
            logger.info(f"打卡文件表头异常，已跳过覆盖: {file}, cols={list(df.columns)}")
            return
        if df.empty and existing_count:
            logger.info(
                f"打卡文件为空，已跳过覆盖已有数据: {file}, "
                f"clockin_id={clockin_id}, existing_count={existing_count}"
            )
            return

        custom_fillna(df, '', numeric_fill_value='')

        for bool_col in ['精华主题', '是否补打卡', '是否精选']:
            if bool_col in df:
                df[bool_col] = df[bool_col].map({'是': True, '否': False})

        if '文字内容' in df:
            # 可能会有些增补字段
            for idx, row in df.iterrows():
                text = str(row['文字内容'])
                for ext_col in ['图片内容', '语音内容', '视频内容']:
                    if ext_col in row:
                        text += f'\n{ext_col}：' + str(row[ext_col])
                row['文字内容'] = text

        # 2 新文件已确认可读，再覆盖旧数据
        self.execute(f'DELETE FROM clockin_data_table WHERE clockin_id={clockin_id}')
        self.commit()

        for idx, row in df.iterrows():
            row2 = {'clockin_name': name, 'clockin_id': clockin_id, 'extra': {}}
            for zhname in row.keys():
                if zhname in zhname2en:
                    row2[zhname2en[zhname]] = row[zhname]
                else:
                    row2['extra'][zhname] = row[zhname]
            self.insert_row('clockin_data_table', row2, commit=-1)
        # with TicToc('更新打卡数据'):
        self.commit_all()

    def refine_clockin_data(self):
        # 1 数据去重
        self.execute("""WITH RankedClockins AS (SELECT clockin_data_id,
                                                       clockin_name,
                                                       nickname,
                                                       publish_time,
                                                       update_content,
                                                       ROW_NUMBER() OVER (PARTITION BY clockin_name, nickname, publish_time, update_content ORDER BY clockin_data_id) as rn
                                                FROM clockin_data_table)
        DELETE
        FROM clockin_data_table
        WHERE clockin_data_id IN (SELECT clockin_data_id
                                  FROM RankedClockins
                                  WHERE rn > 1)""")
        self.commit()

        # 2 重置编号
        self.reset_table_item_id('clockin_data_table')

    def get_clockin_data(self, clockin_name, user_id2s=None):
        """ 查看打卡数据

        :param clockin_name: 这个可以不是精确名字，而是包含名字的字符串，都会被筛选匹配出来
        :param user_id2s: 指定用户id清单顺序
        """
        df = self.exec2df('SELECT * FROM clockin_count_view '
                          f"WHERE clockin_name LIKE '%{clockin_name}%' "
                          'ORDER BY clockin_count DESC')
        if user_id2s:
            df = df[df['user_id2'].isin(user_id2s)]

        df2 = xlpivot(df, index=['user_id2'], columns=['clockin_name'],
                      values={'clockin_count': lambda items: items.iloc[0]['clockin_count'] if len(items) else 0})
        custom_fillna(df2, 0, numeric_fill_value=0)
        # 将df2的字段变成int类型
        for col in df2.columns:
            df2[col] = df2[col].astype(int)

        return df2

    def browser_clockin_data(self, course_name, clockin_names=None, user_id2s=None, titles=None, filter=None):
        """ 查看打卡数据，保持用户ID的顺序并处理空值和无效值

        :param titles: 是否只在限定打卡任务名内统计，并且去重
        :param filter: 对初步获取的df，自定义的过滤处理函数
        """
        # 1 获取课程相关的所有打卡数据
        if clockin_names is None:  # 说明要自动获取
            clockin_names = self.exec2col('SELECT name FROM clockin_table '
                                          f"WHERE name LIKE '{course_name}%' ORDER BY clockin_id")
            clockin_names = [name[len(course_name):] for name in clockin_names]

        # 1 新的更严谨的打卡计算方法
        # 1.1 获得原始打卡数据表
        sql_query = (
            'SELECT user_id2, clockin_name, update_title, task_date, publish_time FROM clockin_data_table '
            f"WHERE clockin_name LIKE '{course_name}%' "
        )
        df = self.exec2df(sql_query)

        if filter is not None:
            df = filter(df)

        # 1.2 分用户统计打卡次数
        def count_func(items):
            # 去掉update_title以"测试-"为前缀的情况
            items = items[~items['update_title'].str.startswith('测试-')]
            if titles:  # 只能在指定titles名称内
                items = items[items['update_title'].isin(titles)]
                # 这种情况下，每种名字只能算一次
                items = items.drop_duplicates(subset=['update_title'])
            # items转正常的df结构
            items = items[['task_date', 'update_title']]

            # 增加一个辅助列，是task_date和update_title的拼接
            cnt = 0
            tags = []
            for _, row in items.iterrows():
                if not row['task_date']:
                    row['task_date'] = cnt
                    cnt += 1
                tags.append(str(row['task_date']) + row['update_title'])
            items['task_date_title'] = tags

            # 按照task_date_title分组，每组取第一个
            items = items.duplicated(subset='task_date_title', keep='first')

            return len(items)

        # 先按照user_id2, clockin_name分组
        df_grouped = df.groupby(['user_id2', 'clockin_name'])
        # 每组执行count_func得到clockin_count列
        ls, columns = [], ['user_id2', 'clockin_name', 'clockin_count']
        # update_title默认填充空字符串
        df['update_title'] = df['update_title'].fillna('')
        for (user_id2, clockin_name), items in df_grouped:
            ls.append([user_id2, clockin_name, count_func(items)])
        df = pd.DataFrame.from_records(ls, columns=columns)

        # 2 检查user_id2s是否提供，若未提供，则从数据中提取所有唯一的user_id2
        if user_id2s is None:
            user_id2s = df['user_id2'].unique()
        # 保持原始user_id2s，即使它们在df中不存在
        df_user_id2s = pd.DataFrame(user_id2s, columns=['user_id2'])
        df_user_id2s.reset_index(drop=True, inplace=True)  # 重置索引以保证合并时顺序不变

        # 3 为每个clockin_name添加特定的后缀名
        df['specific_clockin_name'] = df['clockin_name'].apply(lambda x: x.replace(course_name, ''))
        df = df[df['specific_clockin_name'].isin(clockin_names)]  # 确保只处理指定的打卡名称

        # 4 构建结果DataFrame
        results = []
        for name in clockin_names:
            sub_df = df[df['specific_clockin_name'] == name][['user_id2', 'clockin_count']]
            sub_df.columns = ['user_id2', name]  # 重命名列以匹配clockin_names
            results.append(sub_df)

        # 5 合并所有结果，使用merge保持user_id2s的顺序
        final_df = df_user_id2s
        for result in results:
            final_df = final_df.merge(result, on='user_id2', how='left', sort=False)

        # 6 填充空值
        # 所有打卡列转为整数
        for name in clockin_names:
            final_df[name] = final_df[name].fillna(0).astype(int)
        # 把值为0的单元格值为空字符串
        final_df.replace(0, '', inplace=True)
        return final_df

    def __6_支付数据(self):
        pass

    def update_weipay_from_file(self, file):
        """ 更新微信支付数据

        数据库里有voucher_id字段进行了去重逻辑，所以可以重复插入数据，会自动去重
        """
        df = pd.read_csv(file)
        custom_fillna(df, '', numeric_fill_value='')
        # 删除所有列左边的"`"
        for col in df.columns:
            df[col] = df[col].map(lambda x: x[1:] if x.startswith('`') else x)

        for idx, row in df.iterrows():
            row2 = {}
            row2['datetime'] = row['记账时间']
            row2['business_order'] = row['微信支付业务单号']  # 每个交易的唯一标记
            row2['flow_order'] = row['资金流水单号']

            m = re.search(r'含手续费\s*(\d+(?:\.\d+)?)', row['备注'])
            if m:
                fee = m.group(1)
            else:
                fee = '0.00'

            if row['业务类型'] == '交易':
                row2['money'] = row['收支金额(元)']
                fee = '-' + fee  # 作为收入的时候，是要付手续费，是负值。但是退款的时候，同时退回手续费，那时候是正值
            elif row['业务类型'] == '退款':
                # 此时金额要从备注里获取
                m = re.search(r'总金额\s*(\d+(?:\.\d+)?)', row['备注'])
                if m:
                    row2['money'] = '-' + m.group(1)
                else:
                    row2['money'] = '-' + row['收支金额(元)']
            else:
                continue

            row2['balance'] = row['账户结余(元)']
            row2['submitter'] = row['资金变更提交申请人']
            row2['fee'] = fee
            # 这个在数据库中建立了索引，会产生唯一标记。收到账单的时候是原始凭证号
            # 我返款的时候，会写特殊标签标记
            row2['voucher_id'] = row['业务凭证号']

            self.insert_row('weipay_table', row2, commit=-1)
        self.commit_all()

        # 更新另一个关联的实体化视图
        self.execute('REFRESH MATERIALIZED VIEW weipay_matview')
        self.commit()

    def update_weipay_data(self, today=None):
        """ 更新微信支付数据 """
        from .weipay import Weipay

        logger.info(f'开始更新微信支付数据：today={today}')
        weipay = Weipay(['考勤后台'])
        try:
            # 下载这个月账单数据，每个月1、2号，也会补下载上个月的账单数据。本地会有备份文件，兼容一些旧框架算法。
            files = weipay.daily_update(today)
            logger.info(f'微信支付账单下载完成：file_count={len(files)} files={files}')
            # 下载的文件数据会更新到数据库，推荐以后使用数据库进行相关的逻辑处理。
            for f in files:
                self.update_weipay_from_file(f)
                logger.info(f'微信支付账单导入完成：file={f}')
            logger.info('微信支付数据更新完成')
        finally:
            try:
                if getattr(weipay, 'tab', None):
                    weipay.tab.close()
                    logger.info('微信支付账单更新标签页已自动关闭')
            except Exception as exc:
                logger.warning(f'关闭微信支付账单更新标签页失败，已忽略：{exc}')

    def browser_weipay_data(self, voucher_ids):
        """ 查看微信支付数据，保持voucher_id的顺序并处理空值和无效值。"""
        # 1 从数据库中获取微信支付数据
        voucher_ids2 = [f"'{voucher_id}'" for voucher_id in voucher_ids]
        df = self.exec2df(f'SELECT * FROM weipay_matview WHERE voucher_id IN ({",".join(voucher_ids2)})')

        # 2 检查voucher_ids是否提供，若未提供，则从数据中提取所有唯一的voucher_id
        if not voucher_ids:
            voucher_ids = df['voucher_id'].unique()

        # 保持原始voucher_ids，即使它们在df中不存在
        df_voucher_ids = pd.DataFrame(voucher_ids, columns=['voucher_id'])
        df_voucher_ids.reset_index(drop=True, inplace=True)  # 重置索引以保证合并时顺序不变

        # 3 构建结果DataFrame
        results = []
        for voucher_id in voucher_ids:
            sub_df = df[df['voucher_id'] == voucher_id]
            if not sub_df.empty:
                sub_df = sub_df.iloc[0]  # 取第一行数据，因为每个voucher_id对应一行
                sub_df = sub_df.to_frame().T  # 转换为单行DataFrame
            else:
                sub_df = pd.DataFrame([{}], columns=df.columns)  # 如果没有找到，创建一个空的DataFrame
            results.append(sub_df)

        # 4 合并所有结果，使用concat保持voucher_ids的顺序
        final_df = pd.concat(results, ignore_index=True)

        # 5 填充空值
        # 将所有列转换为适当的数据类型，例如，将数值列转换为int或float，将日期列转换为datetime
        # 这里假设所有列都应该是字符串类型，实际情况可能需要根据列的实际数据类型进行调整
        final_df = final_df.fillna('')  # 用空字符串填充空值

        return final_df

    def __6_汇总报表(self):
        pass

    def __7_一些修复工具(self):
        pass

    def merge_user(self, src_user_id, dst_user_id):
        """ 账号合并

        不过这种改法目前还有小概率会出问题。
        比如第1课旧A账号看了20分钟，新B账号看了10分钟，那么我实际只能取最大值取出20分钟，而不是累加的30分钟
        """
        sql = SqlBuilder('lesson_data_table')
        sql.where(f"user_id2='{src_user_id}'")
        sql.set(f"user_id2='{dst_user_id}'")
        self.execute(sql.build_update())
        sql.table = 'clockin_data_table'
        self.execute(sql.build_update())
        self.commit_all()

    def find_lesson_name(self, keys):
        """ 在数据库中检索最匹配的课程
        :param list[str] keys: 输入若干关键词
        :return: 返回最匹配的一个课程它的名称lesson_name
        
        这个函数一般是配合patch_lesson_data使用，从在线表格获得课程名、课次名等信息，找到数据库里的标准名称

        基本原理，就是先把包含keys的所有lesson_names都先找出来，然后取lesson_id最大的那一条
        """
        # key要做预处理，如果有"\d{2}:\d{2}~\d{2}:\d{2}\s*"这样的时间前缀要删掉; 去掉月日类的key
        keys = [re.sub(r'^\d{2}:\d{2}~\d{2}:\d{2}\s*', '', k)
                for k in keys if not re.search('月.+日', k)]

        def match_all_keys(name):
            for k in keys:
                # 有个难点是数字类的匹配，不能简单用字符串in算法匹配。
                # 比如'中国佛教史1' in '中国佛教史12'，但这两者其实根本不算匹配
                if k not in name or re.search(rf'{re.escape(k)}\d', name):
                    return False
            return True

        # 构建初步过滤的SQL查询
        first_key = keys[0]
        sql = f"SELECT lesson_id, lesson_name FROM lesson_table WHERE lesson_name LIKE '%{first_key}%'"

        # 获取初步过滤后的课程列表
        all_lessons = self.exec2dict(sql).fetchall()

        # 存储匹配的课程
        matched_lessons = [lesson for lesson in all_lessons if match_all_keys(lesson['lesson_name'])]

        # 如果有匹配的课程，返回lesson_id最大的那个的名称
        if matched_lessons:
            return max(matched_lessons, key=lambda x: x['lesson_id'])['lesson_name']

        # 如果没有匹配到任何课程，返回空字符串
        return ''

    def patch_lesson_data(self, lesson_name, user_id, finished_day=0):
        """
        :param finished_day: 目前主要是支持补修当堂完成的记录，对回放完成的补充，可能会有些瑕疵逻辑不严谨
            但只会宽松不会严格，就是如果有用户要补充"第2天回放完成"，那最后修复可能会变成"第1天回放完成"
        """
        lesson = self.exec2dict(f"SELECT * FROM lesson_table WHERE lesson_name='{lesson_name}'").fetchone()

        row = {
            'user_id2': user_id,
            'stay_seconds': lesson['video_duration'],  # 停留时间默认就是课程视频时间
            'cum_seconds': lesson['video_duration'],
            'studio_seconds': 0,
            'playback_seconds': 0,
            'lesson_id': lesson['lesson_id'],
            'update_time': lesson['start_date'] + timedelta(days=finished_day, seconds=lesson['video_duration']),
        }

        if finished_day > 0:
            row['playback_seconds'] = lesson['video_duration']
        else:
            row['studio_seconds'] = lesson['video_duration']

        self.insert_row('lesson_data_table', row, commit=True)

    def patch_lesson_data(self, lesson_name, user_id, finished_day=0):
        """
        :param finished_day: 目前主要是支持补修当堂完成的记录，对回放完成的补充，可能会有些瑕疵逻辑不严谨
            但只会宽松不会严格，就是如果有用户要补充"第2天回放完成"，那最后修复可能会变成"第1天回放完成"
        """
        lesson = self.exec2dict(f"SELECT * FROM lesson_table WHERE lesson_name='{lesson_name}'").fetchone()

        row = {
            'user_id2': user_id,
            'stay_seconds': lesson['video_duration'],  # 停留时间默认就是课程视频时间
            'cum_seconds': lesson['video_duration'],
            'studio_seconds': 0,
            'playback_seconds': 0,
            'lesson_id': lesson['lesson_id'],
            'update_time': lesson['start_date'] + timedelta(days=finished_day, seconds=lesson['video_duration']),
        }

        if finished_day > 0:
            row['playback_seconds'] = lesson['video_duration']
        else:
            row['studio_seconds'] = lesson['video_duration']

        self.insert_row('lesson_data_table', row, commit=True)

    def 禅宗从旧课程继承配置(self, old_course_name, new_course_name):
        from datetime import datetime

        # 1 从self的lesson_table，找lesson_name以old_lesson_prefix为前缀的数据，按lesson_id排序
        old_lesson_prefix = f'{old_course_name}-'
        old_lessons = self.exec2dict(f"""
            SELECT * FROM lesson_table 
            WHERE lesson_name LIKE '{old_lesson_prefix}%'
            ORDER BY lesson_id
        """).fetchall()

        # 2 新的开课时间
        # new_course_name是类似'd250615禅宗12期4点5阶'这样的值，需要通过前缀解析出其日期'2025-06-15，填到下述字符串里

        def extract_date_from_course_name(course_name):
            """从课程名称中提取日期信息"""
            # 匹配类似'd250615'这样的日期格式（d后接6位数字）
            date_match = re.search(r'd(\d{6})', course_name)
            date_str = date_match.group(1)  # 获取250615
            # 将250615转换为2025-06-15
            year = '20' + date_str[0:2]  # 25 -> 2025
            month = date_str[2:4]  # 06
            day = date_str[4:6]  # 15
            return f"{year}-{month}-{day} 00:00:00"

        start_date = datetime.strptime(extract_date_from_course_name(new_course_name), '%Y-%m-%d %H:%M:%S')

        # 3 新的结课时间
        # end_date可以比实际完结多一些周次，到时候可以手动设置实际完结时间
        # 从旧课程中找出最大的周次
        max_week = max(
            (
                int(re.search(r'第(\d+)周', old_lesson['lesson_name']).group(1))
                for old_lesson in old_lessons
            ),
            default=0  # 当old_lessons为空或没有匹配项时，返回0
        )
        end_date = start_date + timedelta(weeks=max_week + 10)

        # 4 从旧课程参考配置，设置新课程数据
        new_lesson_prefix = f'{new_course_name}-'
        new_lessons = []

        for old_lesson in old_lessons:
            new_lesson = {
                'start_date': start_date.strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': end_date.strftime('%Y-%m-%d %H:%M:%S'),
                'lesson_id2': old_lesson['lesson_id2'],
                'shop_id': old_lesson['shop_id'],
                'video_duration': old_lesson['video_duration'],
                # lesson_name直接采用，但替换掉前缀
                'lesson_name': old_lesson['lesson_name'].replace(old_lesson_prefix, new_lesson_prefix)
            }

            # 从lesson_name正则匹配出"第\d+周"
            week_str = re.search(r'第(\d+)周', new_lesson['lesson_name']).group(1)
            week = int(week_str)
            new_lesson['start_date'] = (start_date + timedelta(weeks=week - 1)).strftime('%Y-%m-%d %H:%M:%S')
            new_lesson['next_update'] = (start_date + timedelta(weeks=week)).strftime('%Y-%m-%d %H:%M:%S')

            new_lessons.append(new_lesson)

        # 5 过滤重复数据，确保增量插入更新，不会插入已经配置过配置好的课次
        # 先查询数据库中，已经存在的、以新前缀开头的课程名称
        existing_names = set(self.exec2col(f"""
            SELECT lesson_name FROM lesson_table 
            WHERE lesson_name LIKE '{new_lesson_prefix}%'
        """))

        # 6 过滤列表，只保留数据库中没有的
        for lesson in new_lessons:
            if lesson['lesson_name'] not in existing_names:
                self.insert_row('lesson_table', lesson)

        self.commit()

    def 禅宗从旧课程拷贝数据(self, old_course_name, new_course_name):
        """
        1、首先在lesson_table遍历获取两个课程的全部课次数据，用lesson_id2的相同值匹配可以一一关联对应上
        比如old_course_name的第1课lesson_id=20686，对应上new_course_name第1课lesson_id=20963
        所有匹配的课都要处理，以下只举例第1课的匹配
        2、在lesson_data_table，找出lesson_id=20686的所有数据，全部拷贝一个副本，lesson_id改为20963的数据
        但注意lesson_data_id这个字段是不用拷贝的，是用默认自增运算即可
        """
        # 获取旧课程和新课程的课次数据
        old_lessons = self.exec2dict(
            'SELECT lesson_id, lesson_id2 FROM lesson_table '
            f"WHERE lesson_name LIKE '{old_course_name}%' "
            'ORDER BY lesson_id'
        ).fetchall()

        new_lessons = self.exec2dict(
            'SELECT lesson_id, lesson_id2 FROM lesson_table '
            f"WHERE lesson_name LIKE '{new_course_name}%' "
            'ORDER BY lesson_id'
        ).fetchall()

        # 创建lesson_id2到新旧lesson_id的映射
        old_lesson_map = {item['lesson_id2']: item['lesson_id'] for item in old_lessons}
        new_lesson_map = {item['lesson_id2']: item['lesson_id'] for item in new_lessons}

        # 找出匹配的课次
        common_lesson_ids2 = set(old_lesson_map.keys()) & set(new_lesson_map.keys())

        for lesson_id2 in common_lesson_ids2:
            old_lesson_id = old_lesson_map[lesson_id2]
            new_lesson_id = new_lesson_map[lesson_id2]

            # 获取旧课次的学习数据
            lesson_data = self.exec2dict(
                'SELECT * FROM lesson_data_table WHERE lesson_id = %s',
                [old_lesson_id]
            ).fetchall()

            if lesson_data:
                # 准备插入的新数据，移除lesson_data_id字段
                for item in lesson_data:
                    item.pop('lesson_data_id', None)
                    item['lesson_id'] = new_lesson_id
                    self.insert_row('lesson_data_table', item)
                self.commit()


class KqBook(WpsOnlineBook):
    """ 考勤总表：https://www.kdocs.cn/l/cguYugQWIRs1 """

    def __init__(self, file_id='cguYugQWIRs1', script_id='V2-1ZdbCT3SwkZS2cYf9nunLl'):
        # https://www.kdocs.cn/api/v3/ide/file/cguYugQWIRs1/script/V2-1ZdbCT3SwkZS2cYf9nunLl/sync_task
        super().__init__(file_id, script_id)


def __2_复合功能类():
    pass


def get_kqdb():
    # KqTools().kqdb似乎有缓存问题，程序退出后并不会正常结束数据库连接事务，所以打算换成这样的函数试试
    # 250228周五20:26，已排查，问题大概率并不在这里，而是底层的析构函数，目前应该已解决
    return get_xldb3('kq5034', 'st', cls=KqDb)


@uni_cache(ttl=1800)
def get_用户列表(shop_id=1):
    kqdb = get_kqdb()
    data = kqdb.exec2dict(
        f'SELECT * FROM user_table WHERE shop_id={shop_id} ORDER BY user_id').fetchall()
    return pd.DataFrame(data)
