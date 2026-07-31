import datetime
import inspect
import math
import os
import re
import sys
import time

import fire
import numpy as np
import pandas as pd
from pyxllib.ext.wpsapi import WpsOnlineBook
from pyxllib.file.xlpath import XlPath
from pyxllib.file.xlsxlib import get_column_letter
from pyxllib.prog.xlenv import get_xl_homedir
from tqdm import tqdm

from kq5034.common import logger, wechat_lock_send
from kq5034.tools import KqTools
from xlsln.kq5034.courses.user_alias import RELATED_USER_ID_FIELD, build_user_alias_map


def __1_日报模板():
    pass


念住日报 = """
第1~21天
1、大家好，这是"{群名}"第{天次}天的考勤数据：{链接}。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，义工有空时会统一处理。
请大家尽量通过问卷的方式反馈问题，方便集中处理避免遗漏消息，有必要也可群里@我或私信我。

第22~27天
1、大家好，这是"{群名}"第{天次}天(第{结课次}课回放结束)的考勤数据：{链接}。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，义工有空时会统一处理。
请大家尽量通过问卷的方式反馈问题，方便集中处理避免遗漏消息，有必要也可群里@我或私信我。

第28天
1、大家好，这是"{群名}"第{天次}天(第{结课次}课回放结束)的考勤数据：{链接}。
2、本届念住考勤工作已全部完成，大家核对中若有缺漏或错误，近期依然可以填写问卷反馈：https://code4101.com/attendance-feedback，义工有空时会统一处理。
请大家尽量通过问卷的方式反馈问题，方便集中处理避免遗漏消息，有必要也可群里@我或私信我。
""".strip()

觉观日报 = """
第1~21天
1、大家好，这是"{群名}"第{天次}天的考勤数据：{链接}。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，义工有空时会统一处理。
请大家尽量通过问卷的方式反馈问题，方便集中处理避免遗漏消息，有必要也可群里@我或私信我。

第22~25天
1、大家好，这是"{群名}"第{天次}天(第{结课次}课回放结束)的考勤数据：{链接}。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，义工有空时会统一处理。
请大家尽量通过问卷的方式反馈问题，方便集中处理避免遗漏消息，有必要也可群里@我或私信我。

第26天
1、大家好，这是"{群名}"第{天次}天(第{结课次}课回放结束)的考勤数据：{链接}。
2、本届觉观考勤工作已全部完成，大家核对中若有缺漏或错误，近期依然可以填写问卷反馈：https://code4101.com/attendance-feedback，义工有空时会统一处理。
请大家尽量通过问卷的方式反馈问题，方便集中处理避免遗漏消息，有必要也可群里@我或私信我。
""".strip()

梵呗初阶日报 = """
第1~11天
1、大家好，这是"{群名}"截止{月日}晚，第{天次}天的考勤数据表：{链接}，已按表中统计进行返款。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，义工有空时会统一处理。
请大家尽量通过问卷的方式反馈问题，方便集中处理避免遗漏消息，有必要也可群里@我或私信我。

第12~15天
1、大家好，这是"{群名}"截止{月日}晚，第{天次}天(第{结课次}课回放结束)的考勤数据表：{链接}，已按表中统计进行返款。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，义工有空时会统一处理。
请大家尽量通过问卷的方式反馈问题，方便集中处理避免遗漏消息，有必要也可群里@我或私信我。

第16天
1、大家好，这是"{群名}"截止{月日}晚，第{天次}天(第{结课次}课回放结束)的考勤数据表：{链接}，已按表中统计进行返款。
2、本届本体音艺初阶考勤返款工作已全部完成，大家核对中若有缺漏或错误，近期依然可以填写问卷反馈：https://code4101.com/attendance-feedback，义工有空时会统一处理。
请大家尽量通过问卷的方式反馈问题，方便集中处理避免遗漏消息，有必要也可群里@我或私信我。
""".strip()

梵呗增益日报 = """
第1~22天
1、大家好，这是"{群名}"截止{月日}晚，第{天次}天的考勤数据表：{链接}，已按表中统计进行返款。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，义工有空时会统一处理。
请大家尽量通过问卷的方式反馈问题，方便集中处理避免遗漏消息，有必要也可群里@我或私信我。

第23~24天
1、大家好，这是"{群名}"截止{月日}晚，第{天次}天(第{结课次}课回放结束)的考勤数据表：{链接}，已按表中统计进行返款。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，义工有空时会统一处理。
请大家尽量通过问卷的方式反馈问题，方便集中处理避免遗漏消息，有必要也可群里@我或私信我。

第25天
1、大家好，这是"{群名}"截止{月日}晚，第{天次}天(第{结课次}课回放结束)的考勤数据表：{链接}，已按表中统计进行返款。
2、本届本体音艺初阶增益考勤返款工作已全部完成，大家核对中若有缺漏或错误，近期依然可以填写问卷反馈：https://code4101.com/attendance-feedback，义工有空时会统一处理。
请大家尽量通过问卷的方式反馈问题，方便集中处理避免遗漏消息，有必要也可群里@我或私信我。
""".strip()

念住闯关日报 = """
第1~天
1、大家好，这是"{群名}"{月日}的考勤数据：{链接}。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，义工有空时会统一处理。
请大家尽量通过问卷的方式反馈问题，方便集中处理避免遗漏消息，有必要也可群里@我或私信我。
""".strip()

禅宗周报 = """
第1~天
1、大家好，这是"{群名}"截止第{周次}周的考勤数据表：{链接}，已按表中的统计进行了返款。
1）本考勤数据采集时间统一为每周六24点，每周日早上返款并发周报。
2）数据在「考勤表」统一展示，当周以内准时完成的课程才给予返款；回放完成，延期若干周完成的课程，仅影响考试资格，不作为返款依据。
2、若有缺漏或错误，可以填写问卷反馈：https://code4101.com/attendance-feedback，我在下次考勤更新前会统一处理。
"""


def 获取日报模板(templates_text, day: int) -> str:
    # 1) 用正则匹配出 「第X~Y天」「第X~天」 或者 「第Z天」 这样的段落
    pattern = re.compile(
        r'第(\d+)(?:~(\d*))?天\n'  # 第(\d+)(~(\d*))?天\n
        r'(.*?)(?=\n第\d)',  # 懒惰匹配后续内容，一直到下一个 "第\d" 为止
        re.DOTALL  # 让 '.' 可以匹配换行
    )

    matches = pattern.finditer(templates_text + '\n\n第999天')
    # 这里加一个 "\n第999天"，只是为了保证最后一段能被提取出来

    # 2) 解析出每个匹配片段的天数范围及内容，并存储
    #    matches的元素形如 (start_day, end_day, content_str)
    #    注意：end_day 可能是 None，例如 “第26天”；也可能是空字符串，例如 “第26~天”
    templates = []

    for match in matches:
        start_day_str, end_day_str, content = match.group(1), match.group(2), match.group(3)
        start_day = int(start_day_str)
        if end_day_str is None:
            # 若没有 ~，就说明是 “第26天” 这种
            end_day = start_day
        elif end_day_str:
            end_day = int(end_day_str)
        else:
            # “第26~天” 表示没有截至日期
            end_day = None

        # strip 去掉首尾多余空行
        content = content.strip()

        # 存储为一个 tuple 或者 dict
        templates.append({
            'start': start_day,
            'end': end_day,
            'content': content
        })

    # 这样就将三个块都解析到 templates 这个列表里了
    # 举个例子，templates 可能长这样：
    # [
    #   {'start': 1, 'end': 21, 'content': "1、大家好..."},
    #   {'start': 22, 'end': 25, 'content': "1、大家好..."},
    #   {'start': 26, 'end': 26, 'content': "1、大家好..."},
    # ]

    # 3) 根据 day 在 templates 里找匹配区间
    for block in templates:
        if block['start'] <= day and (block['end'] is None or day <= block['end']):
            return block['content']
    # 找不到符合条件的返回 None
    return None


def __2_核心功能():
    pass


def shift_day_text(text: str) -> str:
    """ 处理国际化学生数据，往前偏移一天

    将字符串中的“回放”天数提前一天：
      - 当堂完成 保持不变
      - 第1天回放 -> 当堂完成
      - 第x天回放 (x>1) -> 第(x-1)天回放
    保留结尾的“/xxx%”部分。
    """
    # 如果本身包含“当堂完成”，直接返回
    # （假设同一个字符串里不会同时出现“当堂完成”和“第x天回放”）
    if "当堂完成" in text:
        return text

    # 使用正则匹配“第(\d+)天回放”，并对数字做处理
    pattern = re.compile(r"第(\d+)天回放")

    def _replace(m: re.Match) -> str:
        day_num = int(m.group(1))
        if day_num == 1:
            # 第1天回放 => 当堂完成
            return "当堂完成"
        else:
            # 第x天回放 => 第(x-1)天回放
            return f"第{day_num - 1}天回放"

    new_text = pattern.sub(_replace, text)
    return new_text


class KqCourse(KqTools):

    def __init__(self, shop_id=1, course_name='第x届网课',
                 book_id='123456', script_id='xyz',
                 回放天数=5, 返款类型=0, 课程商品名='', 打卡返款=True):
        super().__init__()

        self.shop_id = shop_id
        self.course_name = course_name
        self.wb = WpsOnlineBook(book_id, script_id)

        self.status = None
        self.status_date_offset_days = 0
        self.start_date = self.parse_date_from_course_name(self.course_name)
        self.回放天数 = 回放天数

        # 0表示当天返款，1表示次日返款，会影响状态标记逻辑
        self.返款类型 = 返款类型

        self.课程商品名 = 课程商品名
        
        self.打卡返款 = 打卡返款

    @staticmethod
    def _is_blank_cell(value):
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ''
        if isinstance(value, list | tuple | dict):
            return False
        try:
            result = pd.isna(value)
        except TypeError:
            return False
        return bool(result) if isinstance(result, bool | int | float | np.bool_) else False

    @staticmethod
    def _sheet_number(value):
        if KqCourse._is_blank_cell(value):
            return 0.0
        if isinstance(value, int | float):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return 0.0
            return float(value)
        text = str(value).strip().replace(',', '').replace('￥', '').replace('元', '')
        if text in {'', '--'}:
            return 0.0
        try:
            return float(text)
        except ValueError:
            return 0.0

    @staticmethod
    def _normalize_order_id(value):
        if KqCourse._is_blank_cell(value):
            return ''
        return str(value or '').strip().lstrip("`'")

    def __1_预备工作(self):
        pass

    def 更新订单匹配(self):
        # 1 获取数据
        cols = ['微信支付订单号', '订单日期', '商户订单号', '订单金额', '已返款']
        df = self.wb.sql_select('报名表', cols, 4)

        # 2 更新订单数据
        for i, row in df.iterrows():
            # 1 如果'微信支付订单号'的字符串长度小于19，那肯定是错误的，直接跳过
            if len(str(row['微信支付订单号'])) < 19:
                continue

            # 2 如果后4个字段有任意一个字段是空字符串，则需要更新订单数据
            texts = [str(row[col]) for col in cols[1:]]
            if all(texts):
                continue

            row2 = self.在数据库中查找订单(row['微信支付订单号'])

            # 3 如果有数据，更新后continue
            if row2['商户订单号']:
                for col in cols:
                    val = row2[col]
                    if val is not None and not isinstance(val, str):
                        val = str(val)
                    df.loc[i, col] = val
                continue

            # 4 否则需要使用爬虫检索这条订单数据
            row2 = self.weipay.search_refund(row['微信支付订单号'])
            if 'error' not in row2:
                df.loc[i, '微信支付订单号'] = '`' + row2['支付单号']
                df.loc[i, '订单日期'] = datetime.datetime.strptime(row2['交易时间'], '%Y-%m-%d %H:%M:%S').strftime(
                    '%Y%m')
                df.loc[i, '商户订单号'] = row2['商户订单号']
                df.loc[i, '订单金额'] = str(row2['订单金额'])
                df.loc[i, '已返款'] = str(row2['已返款'])
            else:
                df.loc[i, '订单金额'] = str(row2['error'])
                df.loc[i, '已返款'] = ''

        # 3 写回表格
        df = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)
        arr = df[cols].values.tolist()
        # 二次清洗
        for r_idx, row in enumerate(arr):
            for c_idx, val in enumerate(row):
                if isinstance(val, float) and (val != val or val == float('inf') or val == float('-inf')):
                    arr[r_idx][c_idx] = None
        self.wb.write_arr(arr, '报名表!H4', 50)
        self.wb.run_func('releaseMutexLock', '报名表', '商户订单号')

    def 更新用户匹配(self):
        # 1 获取数据
        课程标准名, 课程商品名 = self.course_name, self.课程商品名
        cols = ['姓名', '微信昵称', '手机号', '错误手机号',
                '微信支付订单号', '订单日期', '商户订单号', '订单金额', '已返款',  # 这一行是展现表格结构，实际不需要使用的字段
                '用户ID', '匹配得分']
        df = self.wb.sql_select('报名表', cols, 4)

        # 2 更新订单数据
        has_update = False
        for i, r in tqdm(df.iterrows(), total=len(df)):
            # 如果已经有用户ID，跳过
            if not self._is_blank_cell(r['用户ID']):
                continue
            has_update = True

            # 否则先用数据库检索用户ID
            昵称, 手机号 = [r['姓名'], r['微信昵称']], [r['手机号'], r['错误手机号']]
            user_id2, weight = self.kqdb.查找用户(昵称, 手机号,
                                                  课程标准名=课程标准名, 课程商品名=课程商品名,
                                                  shop_id=self.shop_id, return_mode=1)

            # 如果检索不到再用爬虫检索ID
            if not user_id2:
                self.xe2.switch_shop(self.shop_id)
                user_id2 = self.xe2.查找用户(昵称, 手机号, 课程标准名, 课程商品名)
                if user_id2:
                    weight = 95

            df.loc[i, '用户ID'] = user_id2
            df.loc[i, '匹配得分'] = str(weight) if weight is not None else None

        # 3 写回表格
        if has_update:
            df['手机号'] = "'" + df['手机号']
            df = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)
            arr = df[cols].values.tolist()
            # 二次清洗
            for r_idx, row in enumerate(arr):
                for c_idx, val in enumerate(row):
                    if isinstance(val, float) and (val != val or val == float('inf') or val == float('-inf')):
                        arr[r_idx][c_idx] = None
            self.wb.write_arr(arr, '报名表!D4', 50)
            self.wb.run_func('releaseMutexLock', '报名表', '商户订单号')

    def 把报名表的用户ID同步到考勤表(self):
        # 1 获取数据
        cols = ['姓名', '商户订单号', '用户ID']
        df1 = self.wb.sql_select('报名表', cols, 4)
        df2 = self.wb.sql_select('考勤表', cols, 4)

        dict1 = df1.set_index(['姓名', '商户订单号'])['用户ID'].to_dict()
        
        # 2 遍历df2的'用户ID'，如果是空值，从dcit1检索填写
        has_update = False
        for i, r in df2.iterrows():
            if self._is_blank_cell(r['用户ID']):
                key = (r['姓名'], r['商户订单号'])
                user_id2 = dict1.get(key)
                if not self._is_blank_cell(user_id2):
                    df2.loc[i, '用户ID'] = user_id2
                    has_update = True

        # 3 更新回表格
        if has_update:
            arr = df2[['用户ID']].values.tolist()  # 按cols的顺序转成list数组数据
            self.wb.write_arr(arr, '考勤表!F4', 50)
            self.wb.run_func('releaseMutexLock', '考勤表')

    def get_user_alias_map(self):
        """从报名表读取关联用户ID，并以考勤表用户ID作为主账号。"""
        try:
            df1 = self.wb.sql_select('报名表', ['姓名', '商户订单号', RELATED_USER_ID_FIELD], 4)
        except Exception as e:
            if RELATED_USER_ID_FIELD in str(e):
                return {}
            raise

        df2 = self.wb.sql_select('考勤表', ['姓名', '商户订单号', '用户ID'], 4)
        related_map = df1.set_index(['姓名', '商户订单号'])[RELATED_USER_ID_FIELD].to_dict()
        rows = []
        for _, row in df2.iterrows():
            key = (row['姓名'], row['商户订单号'])
            rows.append({
                '用户ID': row['用户ID'],
                RELATED_USER_ID_FIELD: related_map.get(key, ''),
            })

        alias_map = build_user_alias_map(rows)
        if alias_map:
            logger.info(f'读取到{len(alias_map)}个{RELATED_USER_ID_FIELD}映射，将在统计时虚拟合并')
        return alias_map

    def _2_考勤六步(self):
        pass

    def parse_date_from_course_name(self, course_name):
        # 用正则提取日期部分，格式为 d + 6位数字
        match = re.match(r"^d(\d{6})", course_name)
        if not match:
            raise ValueError("课程名格式不符合要求")

        date_str = match.group(1)
        # 解析年份（25 → 2025）、月份、日期
        year = 2000 + int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])

        return datetime.date(year, month, day)

    @staticmethod
    def _parse_status_text(text, *, bias_hours=0):
        status = 0
        try:
            # 解析文本，假设格式为 '最近运行更新时间：\nYYYY/MM/DD hh:mm:ss,状态编号'
            parts = text.split('\n')[1].split(',')
            date_str = parts[0].strip()
            status = int(parts[1]) if len(parts) > 1 else 100
            date = datetime.datetime.strptime(date_str.split()[0], "%Y/%m/%d").date()
            today = datetime.datetime.now() + datetime.timedelta(hours=bias_hours)
            if date != today.date():
                status = -abs(status)
        except Exception:
            pass
        return status

    def _read_persisted_status(self, bias_hours=0):
        return self._parse_status_text(self.wb.run_func('getStatus'), bias_hours=bias_hours)

    def get_status(self, bias_hours=0):
        """ 获取已运行的状态，这个每次程序只要运行一次即可

        :param bias_hours: 默认在当前时刻获取状态。但有些特殊情况，需要伪造特定时间点来获取状态。
            一般是每天早上梵呗返款执行step4~step6，需要伪造成是10小时前获取状态。
        """
        if self.status is None:
            bias_hours = bias_hours or 24 * int(self.status_date_offset_days or 0)
            self.status = self._read_persisted_status(bias_hours=bias_hours)
        return self.status

    def set_status(self, status, *, allow_rewind=False):
        """ 设置运行的状态 """
        current = self._read_persisted_status(bias_hours=24 * int(self.status_date_offset_days or 0))
        if not allow_rewind and current >= 0 and status < current:
            raise RuntimeError(
                f'禁止回退在线表步骤状态：当前={current}，目标={status}。'
                '强制重跑只能修改进程内状态，不能把已成功步骤写成未完成。'
            )
        self.wb.run_func('setStatus', status, int(self.status_date_offset_days or 0))
        self.status = status

    def _ensure_previous_steps_completed(self, target_step, target_status):
        previous_step = int(target_step) - 1
        required_status = int(target_status) - 1
        if previous_step <= 0 or required_status <= 0:
            return self.get_status()

        current_status = self.get_status()
        if current_status >= required_status:
            return current_status

        logger.warning(
            '检测到考勤步骤跳跃，先补跑前置步骤：'
            f'当前状态={current_status}，目标 step{target_step}(status={target_status}) '
            f'需要先完成 step{previous_step}(status={required_status})'
        )
        func = getattr(self, f'step{previous_step}', None)
        if func is None:
            raise RuntimeError(f'缺少 step{previous_step}，无法补跑到状态 {required_status}')
        if 'status' in inspect.signature(func).parameters:
            func(status=required_status)
        else:
            func()
        self.status = None
        current_status = self.get_status()

        if current_status < required_status:
            raise RuntimeError(
                '考勤前置步骤补跑失败，已停止后续步骤：'
                f'当前状态={current_status}，需要={required_status}'
            )
        return current_status

    def step1(self, update=True, *, status=1):
        if self.get_status() >= status:
            return
        logger.info('1 下载小鹅通数据')
        if update:
            # 更新课程和打卡数据
            if self.shop_id == 1:
                self.xe2.switch_shop('5034山中薪')
                self.update_lesson_data_table(shop1=True)
                if self.打卡返款:
                    self.update_clockin(f'{self.course_name}-*')
            elif self.shop_id == 2:
                self.xe2.switch_shop('宗门学府')
                self.update_lesson_data_table(shop2=True)
                if self.打卡返款:
                    self.update_clockin(f'{self.course_name}-*')
            logger.info(f'"{self.course_name}"下载完小鹅通数据')
        self.set_status(status)

    def step2(self, *, status=2):
        raise NotImplementedError

    def wb_get_column_list(self, col, *, filter_empty_rows=False):
        return self.wb.sql_select('考勤表', [col], 4,
                                  filter_empty_rows=filter_empty_rows, return_mode='json')[col]

    def _读取返款额度校验表(self):
        cols = ['姓名', '商户订单号', '视频应返款', '打卡应返款', '总应返款', '当前应返款', '已返款']
        try:
            df = self.wb.sql_select('考勤表', cols, 4, filter_empty_rows=False)
            if '打卡应返款' in df.columns:
                return df
        except Exception as exc:
            logger.warning(f'读取含打卡应返款的返款额度校验表失败，按无打卡返款表兼容：{type(exc).__name__}: {exc}')

        cols = ['姓名', '商户订单号', '视频应返款', '当前应返款', '已返款']
        df = self.wb.sql_select('考勤表', cols, 4, filter_empty_rows=False)
        df['打卡应返款'] = 0
        return df

    def _校验返款配置不超过剩余额度(self, lines):
        """校验返款CSV配置不会超过本轮可返额度。

        :param list[str] lines: 返款配置CSV行。
        """
        if not lines:
            return

        requested = {}
        for line in lines:
            item = self.解析返款促学金行(line)
            order_id = self._normalize_order_id(item['订单号'])
            if not order_id:
                raise RuntimeError(f'返款配置缺少商户订单号，已阻断自动返款：{line!r}')
            requested.setdefault(order_id, {'amount': 0.0, 'items': []})
            requested[order_id]['amount'] += item['金额']
            requested[order_id]['items'].append(item)

        df = self._读取返款额度校验表()
        rows_by_order = {
            self._normalize_order_id(row.get('商户订单号')): row
            for _, row in df.iterrows()
            if self._normalize_order_id(row.get('商户订单号'))
        }

        missing_orders = []
        violations = []
        for order_id, request in requested.items():
            row = rows_by_order.get(order_id)
            if row is None:
                missing_orders.append(order_id)
                continue

            total_due = self._sheet_number(row.get('总应返款'))
            if not total_due:
                total_due = self._sheet_number(row.get('视频应返款')) + self._sheet_number(row.get('打卡应返款'))
            refunded = self._sheet_number(row.get('已返款'))
            current_due = self._sheet_number(row.get('当前应返款'))
            total_remaining = round(total_due - refunded, 2)
            remaining = round(min(current_due, total_remaining), 2) if current_due > 0 else 0.0
            request_amount = round(request['amount'], 2)
            if request_amount > max(remaining, 0.0) + 0.01:
                violations.append({
                    'name': str(row.get('姓名') or '').strip(),
                    'order_id': order_id,
                    'request_amount': request_amount,
                    'remaining': remaining,
                    'total_due': round(total_due, 2),
                    'refunded': round(refunded, 2),
                    'current_due': round(current_due, 2),
                    'voucher_ids': [x['业务单号'] for x in request['items']],
                })

        if missing_orders or violations:
            messages = []
            if missing_orders:
                messages.append(f'找不到商户订单号：{", ".join(missing_orders[:5])}')
            for item in violations[:10]:
                messages.append(
                    f"{item['name']} {item['order_id']} 本次{item['request_amount']}元 > "
                    f"可返{item['remaining']}元"
                    f"（当前应返{item['current_due']}，总应返{item['total_due']}，已返{item['refunded']}），"
                    f"业务单号：{', '.join(item['voucher_ids'][:3])}"
                )
            if len(violations) > 10:
                messages.append(f'另有 {len(violations) - 10} 条超额返款配置未展示')
            raise RuntimeError('返款配置额度校验失败，已阻断自动返款：\n' + '\n'.join(messages))

    def _校验返款配置等于当前应返款(self, lines):
        """防止 step5 把未实际提交的金额累加到已返款。"""
        if not lines:
            return

        requested = {}
        for line in lines:
            item = self.解析返款促学金行(line)
            order_id = self._normalize_order_id(item['订单号'])
            if not order_id:
                raise RuntimeError(f'返款配置缺少商户订单号，已阻断更新已返款：{line!r}')
            requested[order_id] = round(requested.get(order_id, 0.0) + self._sheet_number(item['金额']), 2)

        df = self._读取返款额度校验表()
        rows_by_order = {
            self._normalize_order_id(row.get('商户订单号')): row
            for _, row in df.iterrows()
            if self._normalize_order_id(row.get('商户订单号'))
        }

        mismatches = []
        for order_id, request_amount in requested.items():
            row = rows_by_order.get(order_id)
            if row is None:
                mismatches.append(f'{order_id} 找不到在线表订单行，返款配置={request_amount}')
                continue
            current_due = round(self._sheet_number(row.get('当前应返款')), 2)
            if abs(request_amount - current_due) > 0.01:
                name = str(row.get('姓名') or '').strip()
                mismatches.append(f'{name} {order_id} 返款配置={request_amount}，当前应返款={current_due}')

        if mismatches:
            raise RuntimeError(
                '返款配置金额与当前应返款不一致，已阻断 step5，避免已返款错账：\n'
                + '\n'.join(mismatches[:10])
            )

    def shift_international_students(self, df2: pd.DataFrame, user_list: list) -> pd.DataFrame:
        """
        对指定的国际学生（user_list），将 df2 中的“回放”时间节点提前一格：
          - 当堂完成 -> 保持不变
          - 第1天回放 -> 当堂完成
          - 第x天回放 (x>1) -> 第(x-1)天回放

        注意：此操作会直接在传入的 df2 上进行修改。
        """
        # 需要处理的列（排除 user_id2 列）
        columns_to_process = [col for col in df2.columns if col != "user_id2"]

        # 按指定学生逐行逐列执行替换
        for user_id in user_list:
            mask = df2["user_id2"] == user_id
            for col in columns_to_process:
                df2.loc[mask, col] = df2.loc[mask, col].apply(shift_day_text)

        # 原地修改，df2 已发生变化，按需可返回
        return df2

    @staticmethod
    def _wps_table_soft_timeout_seconds():
        return float(os.getenv('KQ_WPS_TABLE_SOFT_TIMEOUT_SECONDS') or os.getenv('WPS_TABLE_SOFT_TIMEOUT_SECONDS') or '25')

    def _run_wps_row_func_adaptive(self, func, start_row, end_row, batch_size, *, soft_timeout_seconds=None):
        """按行分批运行 WPS 函数，超时或失败时自动二分缩小批次。"""
        if start_row > end_row:
            return []
        soft_timeout_seconds = soft_timeout_seconds or self._wps_table_soft_timeout_seconds()
        batch_size = max(1, int(batch_size))
        unit_retry_count = max(1, int(os.getenv('KQ_WPS_UNIT_RETRY_COUNT') or '2'))
        retry_delay = max(1.0, float(os.getenv('KQ_STEP3_BATCH_RETRY_DELAY_SECONDS') or '10'))
        done_segments = []

        def run_segment(a, b):
            size = b - a + 1
            for attempt in range(1, unit_retry_count + 1):
                try:
                    self.wb.run_func(func, a, b, soft_timeout_seconds=soft_timeout_seconds)
                    done_segments.append((a, b))
                    logger.info(f'WPS行批次完成：{func} rows={a}-{b} size={size}')
                    return
                except Exception:
                    if size > 1:
                        mid = (a + b) // 2
                        logger.warning(f'WPS行批次失败，自动拆分：{func} rows={a}-{b} -> {a}-{mid}, {mid + 1}-{b}')
                        run_segment(a, mid)
                        run_segment(mid + 1, b)
                        return
                    if attempt >= unit_retry_count:
                        logger.exception(f'WPS单行批次失败，状态不会推进：{func} row={a}')
                        raise
                    logger.warning(f'WPS单行批次失败，准备重试 {attempt}/{unit_retry_count}: {func} row={a}')
                    time.sleep(retry_delay * attempt)

        for r in range(start_row, end_row + 1, batch_size):
            run_segment(r, min(r + batch_size - 1, end_row))
        return done_segments

    def write_rows_skip_empty(self, df3, *, batch_size=50, find_start_col='打卡数'):
        """ 将df3的考勤数据写入表格，跳过user_id2为空的行
        df3的第一列必须是user_id2

        250312周三19:40，由于念住闯关随到随学会有重复学员的情况，不能删掉旧的已学数据
            需要通过在表格删掉user_id2的机制，来停止新数据的覆盖
            旧的写入，是暴力写进空数据
            现在的机制，是会跳过user_id2的行不处理，保留里面可能以前程序写过的数据结果
        """
        start_col = self.wb.run_func('findCol', find_start_col, '考勤表!2:2')
        col_name = get_column_letter(start_col)

        current_batch = []
        current_row = 4  # 起始行号
        batch_start_row = current_row

        for row in df3.to_dict(orient='split')['data']:
            if pd.isna(row[0]) or not row[0]:  # user_id2为空
                # 遇到空行后，需要把之前已有的数据先全部写进表格
                if current_batch:
                    self.wb.write_arr(current_batch, f'考勤表!{col_name}{batch_start_row}', batch_size)
                    current_batch = []
            else:
                if not current_batch:  # 新批次的第一行
                    batch_start_row = current_row
                current_batch.append(row[1:])

                if len(current_batch) >= batch_size:
                    self.wb.write_arr(current_batch, f'考勤表!{col_name}{batch_start_row}', batch_size)
                    current_batch = []
            current_row += 1

        # 处理剩余的行
        if current_batch:
            self.wb.write_arr(current_batch, f'考勤表!{col_name}{batch_start_row}', batch_size)

    def step3(self, skip_rows=0, *, status=3):
        """
        :param skip_rows: 跳过前面部分条目的数据不处理。一般用来兼容新旧考勤规则不同的算法逻辑。
            其实不建议使用该参数，而是推荐把新建数据分成两个sheet进行隔离
        :return:
        """
        self._ensure_previous_steps_completed(3, status)
        if self.get_status() >= status:
            return
        logger.info('3 更新应返款')
        # 获得数据范围，rows['start']、rows['end']存储了整体数据的起始、终止行
        rows = self.wb.run_func('locateTableRange', '考勤表', 4, ['视频应返款'])[1]
        batch_size = 50
        if '禅宗' in self.course_name:
            batch_size = 10  # 禅宗的情况改小点，这个一行可能就有60多个课
            if '4点5阶' in self.course_name or '4.5阶' in self.course_name:
                batch_size = 1  # 4.5阶表很宽，WPS sync_task 批量多行容易 500 超时

        if '禅宗' in self.course_name:
            func = f'step3_计算视频应返款_禅宗'
        elif '念住闯关' in self.course_name:
            func = f'step3_计算视频应返款_念住闯关'
        elif '梵呗初阶' in self.course_name:
            func = f'step3_计算视频应返款_梵呗初阶'
        elif re.search('增益堂|梵呗增益', self.course_name):
            func = f'step3_计算视频应返款_梵呗增益'
        else:
            func = f'step3_计算视频应返款_觉观念住'
        self._run_wps_row_func_adaptive(
            func,
            rows['start'] + skip_rows,
            rows['end'],
            batch_size,
            soft_timeout_seconds=self._wps_table_soft_timeout_seconds(),
        )
        logger.info('应返款更新完成')
        self.set_status(status)
        time.sleep(30)  # 第3步数据涉及公式重算有延迟，建议等半分钟继续

    def step4(self, *, status=4, force_submit=False):
        self._ensure_previous_steps_completed(4, status)
        status_before = self.get_status()
        if status_before >= status and not force_submit:
            return
        logger.info('4 自动返款')
        if force_submit:
            logger.warning(f'启用强制返款提交：将忽略当前状态{status_before}，自动重写CSV文件名和返款业务单号')
        lines = self.wb_get_column_list('返款配置')
        lines = self.过滤有效返款促学金(lines)
        if lines:
            self._校验返款配置不超过剩余额度(lines)
            if force_submit:
                result = self.自动返款促学金(lines, self.weipay, force_submit=True)
            else:
                result = self.自动返款促学金(lines, self.weipay)
            if result and result.get('reason') == 'submit_marker_exists':
                logger.warning(f'检测到返款文件已提交过，step4 直接按已完成处理：{result.get("marker", {}).get("file")}')
        if status_before < status:
            self.set_status(status)

    def step5(self, *, status=5):
        self._ensure_previous_steps_completed(5, status)
        if self.get_status() >= status:
            return
        logger.info('5 更新已返款')
        lines = self.wb_get_column_list('返款配置')
        lines = self.过滤有效返款促学金(lines)
        self._校验返款配置等于当前应返款(lines)
        self.wb.run_func('step5_更新已返款', soft_timeout_seconds=self._wps_table_soft_timeout_seconds())
        self.set_status(status)

    def get_daily(self, 群名='', bias=0):
        """ 获取日报

        :param bias: 偏差天次，主要用于梵呗，需要次日更新昨晚的数据
        """
        # 1 获取模板
        today = datetime.date.today() + datetime.timedelta(days=bias)
        天次 = (today - self.start_date).days + 1
        周次 = (today - self.start_date).days // 7 + 1
        采集星期 = 返款星期 = ''  # 只有禅宗采用

        if '禅宗' in self.course_name:
            日报 = 禅宗周报
            周次 = self.wb.run_func('get禅宗周次')

            采集星期 = "周六"
            返款星期 = "周日"

        elif '觉观' in self.course_name:
            日报 = 觉观日报
        elif '念住闯关' in self.course_name:
            日报 = 念住闯关日报
        elif '念住' in self.course_name:
            日报 = 念住日报
        elif '梵呗增益' in self.course_name:
            日报 = 梵呗增益日报
        elif '梵呗初阶' in self.course_name:
            日报 = 梵呗初阶日报
        else:
            raise NotImplementedError

        template = 获取日报模板(日报, 天次)
        if template is None:
            return

        # 2 填充内容
        月日 = f'{today.month}月{today.day}日'
        链接 = f'https://kdocs.cn/l/{self.wb.book_id}'
        结课次 = 天次 - self.回放天数
        content = template.format(群名=群名, 月日=月日, 周次=周次, 天次=天次,
                                  结课次=结课次, 链接=链接,
                                  采集星期=采集星期, 返款星期=返款星期)
        return content

    def step6(self):
        raise NotImplementedError

    def 更新修订(self):
        """ 可能手动修正了一些异常，需要刷新显示数据的时候，可以用这个函数，强制重运行第2、3步 """
        self.status = 1
        self.step2()
        self.step3()


if __name__ == '__main__':
    os.chdir(get_xl_homedir())
    if len(sys.argv) > 1:
        fire.Fire(KqCourse)
    else:
        pass
