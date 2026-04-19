"""微信支付相关实现。"""

from .common import *  # noqa: F403
from .wechat_runtime import KqWechat

class Weipay(DpWebBase):
    """ 微信支付 """

    def __init__(self, users=None):
        """
        :param users: 如果要扫码登录等操作，是否要发送到指定的一些微信群
        """
        super().__init__('https://pay.weixin.qq.com')
        self.user = None  # 微信支付的用户名记录(这个要放在login()之前！)
        if users:
            # 执行登录逻辑，如果尚未登录则调用二维码登录函数
            self.login(users)

    def login(self, users=None):
        """
        检查并处理登录逻辑，通过二维码登录页面直到用户扫码成功。

        :param users: 发给哪些微信用户帮忙扫码
            如 ['文件传输助手', '陈坤泽', '孙浩']
            None, 如果未写，默认等待当前机器面前的值守人员直接扫码
        """
        tab = self.tab
        if not tab.url == 'https://pay.weixin.qq.com/index.php/core/info':
            tab.get('https://pay.weixin.qq.com')

            # 设置一个标记变量，用于控制微信消息至少发送一次
            message_sent = False

            # 使用 while 循环，直到到达指定的 URL 地址
            while not tab.url == 'https://pay.weixin.qq.com/index.php/core/info':
                # 1 二维码如果过期需要刷新
                div = tab('tag:div@@class=qrcode-img')
                try:
                    is_invalid = div('tag:div@@class=alt@@text():二维码失效', timeout=3)
                except DrissionPage.errors.ContextLostError:  # 一般是登录后，元素被重置了
                    is_invalid = None

                if is_invalid:  # 需要刷新二维码
                    logger.info(self.get_recive('二维码已过期，请发送任意消息，重新触发获取最新二维码'))

                    tab.refresh()
                    message_sent = False

                if message_sent:
                    time.sleep(5)
                    # 如果已经发送过消息，就 continue 等待用户扫码
                    continue

                # 2 获取二维码图片
                div = tab('tag:div@@id=IDQrcodeImg')
                # 241223周一21:16，这个今早炸锅了，感觉可能和微信二维码的什么掩码等机制有关，我还是换原来的方式吧，原来都没出过问题
                file = div('tag:img').save(XlPath.tempdir(), 'qrcode')  # dp提供对有src属性的图片自动下载的功能

                # 3 发送微信消息，提醒用户扫码登录
                if users:
                    for user in users:
                        wechat_lock_send(user, '考勤工作需要，快帮我扫码登录微信支付', files=[file])
                    time.sleep(5)  # 发完图片一般要等会
                    with get_autogui_lock():
                        KqWechat.扫码登录微信支付(users[0])
                else:
                    print('>> 请扫码登录首页后，程序会自动继续运行...')

                # 设置标记变量为 True，确保至少发送一次消息
                message_sent = True

        self.user = tab('tag:a@@class=username').text.split('@')[0]

    def __1_基础功能(self):
        pass

    def 重连标签页(self):
        """DrissionPage 的 tab 对象偶尔会在页面跳转后失效，重新取一个当前可用 tab。"""
        try:
            tab = get_latest_not_dev_tab(self.browser)
            if tab:
                self.tab = tab
                return tab
        except Exception:
            pass

        self.tab = self.browser.latest_tab
        return self.tab

    def download_monthly_records(self, month, save_dir=True):
        """ 下载月份资金账单

        :param month: 月份，格式为"2024-03"
            本接只支持按整月下载
        :param save_dir: 保存在某目录下
            True, 保存到默认为止
            str, 报错到指定目录

        >> Weipay().download_monthly_records('2024-07')
        """
        tab = self.tab
        tab.get('https://pay.weixin.qq.com/index.php/xphp/cfund_bill_nc/funds_bill_nc#/')  # 资金账单

        start_day = month + '-01'
        end_day = month + f'-{str(pd.Period(month).end_time.day)}'

        tab.wait(3)
        tab.wait.ele_displayed('tag:input@@placeholder=开始日期')
        tab.action_type('tag:input@@placeholder=开始日期', start_day)
        tab.action_type('tag:input@@placeholder=结束日期', end_day + '\n')
        tab.wait(3)
        tab('tag:button@@text()=查询').click()
        tab.wait(5)

        if save_dir:
            tab('tag:a@@class=popups download').click()
            ele0 = tab('tag:div@@class=el-dialog__wrapper new-capital-down-dialog@@text():账单打包完成，请确认下载')
            src_file = XlPath(ele0('.el-button el-button--primary', timeout=60).click.to_download().wait(show=False))

            if save_dir is True:
                save_dir = xlhome_dir('data/m2112kq5034/数据表')
            else:
                save_dir = XlPath(save_dir)
            name = re.sub(r'_\d+\.csv$', '.csv', src_file.name)  # 删除后缀可能带有的多版本数字标记
            dst_file = save_dir / name
            shutil.copy(src_file, dst_file)
            return dst_file

    def daily_update(self, today=None):
        """ 每天更新账单数据

        - 如果是1号或2号，下载上个月和当月的账单
        - 其他日期只下载当月账单
        """
        dst_files = []

        # 1 计算今天所属年份、月份、日期
        today = today or pd.Timestamp.now()
        current_year = today.year
        current_month = today.month
        current_day = today.day

        # 2 如果日期是1号或2号，要下载上个月的数据
        if current_day in [1, 2]:
            # 计算上个月的年月
            if current_month == 1:
                last_month_year = current_year - 1
                last_month = 12
            else:
                last_month_year = current_year
                last_month = current_month - 1

            # 下载上个月数据
            last_month_str = f"{last_month_year}-{str(last_month).zfill(2)}"
            # 网页可能不稳定，可以多试几次
            for i in range(4):
                try:
                    dst_file = self.download_monthly_records(last_month_str)
                    break
                except DrissionPage.errors.NoRectError:
                    pass
            else:
                dst_file = self.download_monthly_records(last_month_str)

            if dst_file is not None:
                dst_files.append(dst_file)

        # 3 下载当月数据
        if current_day == 1:  # 当月第1天暂时不能下载，要再等一天
            return dst_files

        current_month_str = f"{current_year}-{str(current_month).zfill(2)}"
        for i in range(4):
            try:
                dst_file = self.download_monthly_records(current_month_str)
                break
            except DrissionPage.errors.NoRectError:
                pass
        else:
            dst_file = self.download_monthly_records(current_month_str)

        if dst_file is not None:
            dst_files.append(dst_file)

        return dst_files

    def search_refund(self, voucher_id):
        """ 查找订单退款信息 """
        # 1 查找订单
        tab = self.tab
        tab.get('https://pay.weixin.qq.com/index.php/core/trade/search_new')

        voucher_id = str(voucher_id).lstrip("`'")
        input_name = 'mmpay_order_id' if re.match(r'\d+$', voucher_id) else 'merchant_order_id'
        tab.find_ele_with_refresh(f't:input@@name={input_name}').input(voucher_id)
        tab('t:a@@id=idQueryButton').click()

        # 2 报错
        tips = tab('t:div@@class=tips-error').text
        # 请至少填一个查询单号
        # 查询失败:微信订单号输入不正确
        if tips:  # 报错
            return {'error': tips}

        # 3 订单信息
        html = tab.find_ele_with_refresh('t:div@@class=table-wrp with-border')('t:table').html
        # 用beautifulsoup解析这段html，把每个th对应的td以字典方式解析出来
        soup = BeautifulSoup(html, 'lxml')
        trs = soup.find_all('tr')
        row = {}
        for tr in trs:
            th = tr.find('th')
            if th:
                th = th.text.strip()
                td = tr.find('td')
                if td:
                    td = td.text.strip()
                    row[th] = td

        # {'商户订单号': 'SX2TGC-0OZRE8O-EFG9',
        # '支付单号': '4200002706202505304477731244',
        # '交易状态': '部分退款完成',  # 买家已支付, 全额退款完成
        # '订单金额': '620.00元',
        # '交易时间': '2025-05-30 21:29:06'
        row['订单金额'] = float(row['订单金额'].strip('元'))

        # 4 已返款
        match row['交易状态']:
            case '买家已支付':
                row['已返款'] = 0
            case '部分退款完成':
                tab2 = tab('t:a@@id=reqReturnBn').click.for_new_tab()
                row['已返款'] = float(tab2('t:div@@class=form').eles('t:div@@class=form-item')[3]('t:span').text)
                tab2.close()
            case '全额退款完成':
                row['已返款'] = row['订单金额']
            case _:
                logger.warning('未知交易状态：' + row['交易状态'])
                row['已返款'] = row['交易状态']

        尝试关闭重复页面(self.browser, reason='查询微信支付订单后收尾',
                 keep_tab_ids=[getattr(self.tab, 'tab_id', None)])
        return row

    def request_single_refund(self, voucher_id, refund_amount=0, refund_reason=''):
        """ 申请单条退款
        :param voucher_id: 订单号或者商户订单号都支持

        >> wp.request_single_refund('SFW1WL-0OZRE8O-KX63', 0.01, '测试退款')
        >> wp.request_single_refund('4200002199202406302648230239', 0.01, '测试退款')
        """
        # 1 查询订单
        tab = self.tab
        tab.get('https://pay.weixin.qq.com/index.php/core/refundapply')
        input_name = 'wxOrderNum' if re.match(r'\d+$', voucher_id) else 'mchOrderNum'

        tab.find_ele_with_refresh(f't:input@@name={input_name}').input(voucher_id)
        tab('#applyRefundBtn').click()

        # '请输入正确格式的交易单号'  # 格式错误
        # '记录不存在'  # 上面错误
        # '订单不存在'  # 下面错误
        # '订单已全额退款'

        # 2 填写返款表格
        tab.find_ele_with_refresh('t:input@@name=refund_amount').input(refund_amount)
        tab('#textInput').input(refund_reason)
        tab('#commitRefundApplyBtn').click(by_js=True)
        tab.wait(2)

        # 3 验证
        self.填写密码与验证码(tab)

        # 4 等待微信侧真正处理完成后再返回，避免上层误记“已退款”
        self.wait_refund_completion()

    def get_recive(self, content):
        """ 获得任意反馈内容 """

        with WeChatSingletonLock(120) as wx:
            recive_msg = None
            wx.SendMsg(content, '考勤后台')

            while recive_msg is None:
                wx._show()
                wx.ChatWith('考勤后台')
                msgs = wx.GetAllMessage()

                for msg in msgs[::-1]:
                    # if msg.msg_type != 'receive':  # 先关掉接收消息的约束，方便codepc_aw上用主微信账号调试
                    #     continue

                    # 只能找刚才提问后的新内容
                    if msg.content == content and msg.sender == 'Self':
                        break

                    recive_msg = msg.content
                    if recive_msg:
                        break

                time.sleep(3)

        return recive_msg

    def get_vcode(self):
        """ 拿到短信验证码 """

        with WeChatSingletonLock(120) as wx:
            vcode = None

            content = f'@{self.user} 微信支付往你手机发了一个支付验证码，请查看回复下'
            wx.SendMsg(content, '考勤后台')

            while True:
                wx._show()
                wx.ChatWith('考勤后台')
                msgs = wx.GetAllMessage()

                for msg in msgs[::-1]:
                    # if msg.msg_type != 'receive':  # 先关掉接收消息的约束，方便codepc_aw上用主微信账号调试
                    #     continue
                    vals = re.findall(r'\d+', msg.content)
                    # 过滤掉长度不是6的
                    vals = [v for v in vals if len(v) == 6]
                    if vals:
                        vcode = vals[0]
                        # we.send_text(f'已收到验证码：{vals[0]}')
                        break

                    # 只能找刚才提问后的新内容
                    if msg.content == content and msg.sender == 'Self':
                        break

                if vcode:
                    break

                time.sleep(3)

        return vcode

    def wait_refund_completion(self, timeout=300):
        """等待返款完成。

        调试这段逻辑时，失败要直接抛错，不能再无限刷新重试，否则真实现场会被掩盖。
        """
        tab = self.tab
        deadline = time.time() + timeout
        last_status_text = ''

        while time.time() < deadline:
            if ele := tab('t:div@@class=dialog@@text():提交成功'):
                ele('t:a@@text():确认').click()
                tab.wait(1)

            try:
                body_text = tab('tag:body').text
            except Exception:
                body_text = ''

            table = tab.ele('tag:table@@class=table', timeout=3)
            if table:
                rows = table.eles('tag:tr')
                row_texts = [row.text for row in rows[1:4]]
                status_text = '\n'.join(row_texts) if row_texts else body_text
            else:
                status_text = body_text

            last_status_text = status_text[:300]

            if '已处理' in status_text:
                return
            if any(x in status_text for x in ['处理中', '提交成功']):
                time.sleep(2)
                continue
            if table:
                time.sleep(2)
                continue

            time.sleep(1)

        raise RuntimeError(
            '微信支付批量退款结果未完成，'
            f'url={tab.url}，title={tab.title}，状态摘要={last_status_text!r}'
        )

    def 尝试点击返款提交后的提示按钮(self, tab, timeout=15):
        """微信支付返款提交后，偶尔会多出一层确认弹窗；有则处理，没有就继续。"""
        locators = [
            'tag:a@@class=btn btn-primary close-dialog JSCloseDG',
            'tag:a@@id=FinishProtocolBn',
            'tag:a@@text()=进入商户平台',
            'tag:a@@tabindex=2',
            'tag:a@@text()=完成',
            'tag:button@@text()=完成',
            'tag:a@@text()=确认',
            'tag:button@@text()=确认',
            'tag:span@@text()=确认',
            'tag:a@@text()=知道了',
            'tag:a@@text()=返回',
            'tag:button@@text()=返回',
            'tag:i@@class=el-dialog__close el-icon el-icon-close',
            'tag:i@@class=el-dialog__close',
        ]
        deadline = time.time() + timeout
        clicked_any = False
        while time.time() < deadline:
            clicked_this_round = False
            for locator in locators:
                try:
                    for btn in tab.eles(locator):
                        if not btn.states.is_displayed:
                            continue
                        btn.click(by_js=True)
                        tab.wait(1)
                        clicked_any = True
                        clicked_this_round = True
                        break
                except Exception:
                    continue
                if clicked_this_round:
                    break

            if clicked_this_round:
                continue

            try:
                body_text = tab('tag:body').text
            except Exception:
                body_text = ''

            if any(x in body_text for x in ['批量退款批次查询', '处理中', '已处理']):
                return clicked_any
            time.sleep(1)
        return clicked_any

    def 填写密码与验证码(self, tab):
        # 1 填写密码
        inputs = tab.eles('tag:input@@class=real-input')
        passwd = XlEnv.get(f'XL_KQ_PAY_PASSWORD_{self.user}', decoding=True)
        if not passwd:
            passwd = XlEnv.get('XL_KQ_PAY_PASSWORD', decoding=True)
        if passwd:
            inputs[0].input(passwd, clear=True)

        # 2 短信验证码
        if len(inputs) > 1:
            tab('tag:a@@text():发送短信').click()
            # vcode = self.get_vcode()
            time.sleep(10)
            with get_autogui_lock():
                vcode = KqWechat.从懒人转发获得短信内容()
            inputs[1].input(vcode, clear=True)

        # 3 确认按钮
        time.sleep(1)
        tab('tag:a@@text()=确定@@class=btn btn-primary align-center').click()

        # 4 有些页面会额外弹一次提示，有就处理，没有就直接继续。
        self.尝试点击返款提交后的提示按钮(tab)

    def request_file_refund(self, file=None):
        """ 通过文件进行批量退款

        :param file:
            str, 上传指定文件
            None, 自动找到最新的返款文件

        """
        # 1 找到本地最新的返款文件
        if file is None:
            d = xlhome_dir('data/m2112kq5034/返款表')
            # 找到目录下更新时间最新的文件
            files = list(d.glob_files('*.csv'))
            files.sort(key=lambda f: f.mtime())
            file = files[-1]

        # 2 上传文件
        tab = self.tab
        tab.get('https://pay.weixin.qq.com/index.php/xphp/cbatchrefund/batch_refund#/pages/index/index')
        tab.wait(2)
        tab('tag:a@@title=上传文件').click.to_upload(file)
        tab.wait(2)
        tab('tag:a@@text():确定@@class=btn btn-primary@@href=javascript:void(0);').click()
        tab.wait(2)

        # 3 填写密码
        self.填写密码与验证码(tab)

        # 4 等待返款完成
        self.wait_refund_completion()

    def __2_订单功能(self):
        pass

    @classmethod
    def _生成0O替换组合(cls, 订单号):
        """生成将数字0替换为字母O的所有可能组合"""
        # 找到所有数字0的位置
        zero_positions = [i for i, char in enumerate(订单号) if char == '0']

        if not zero_positions:
            return [订单号]

        combinations = []
        # 用位操作生成所有可能的组合
        # 0表示保持为'0'，1表示替换为'O'
        for i in range(2 ** len(zero_positions)):
            订单号_list = list(订单号)
            for j, pos in enumerate(zero_positions):
                if i & (1 << j):  # 检查第j位是否为1
                    订单号_list[pos] = 'O'
            combinations.append(''.join(订单号_list))

        return combinations

    @classmethod
    def 生成候选订单清单(cls, 订单号):
        订单号 = str(订单号).lstrip("`'")
        候选订单清单 = [订单号]
        if '-' in 订单号 and '0' in 订单号:
            # 订单号中的每个0都有可能是O，请枚举所有情况进行查找，找到第一个满足就退出
            候选订单清单 = cls._生成0O替换组合(订单号)
        return 候选订单清单

    @classmethod
    def 优化订单格式(cls, row):
        row2 = {}
        row2['订单日期'] = row['datetime'].strftime('%Y%m') if row.get('datetime') else ''
        row2['微信支付订单号'] = ('`' + row['flow_order']) if row.get('flow_order') else ''
        row2['商户订单号'] = row['voucher_id'] if row.get('voucher_id') else ''
        row2['订单金额'] = float(row['money']) if row.get('money') else ''
        row2['已返款'] = str(row['refund']) if row.get('refund') else 0
        return row2
