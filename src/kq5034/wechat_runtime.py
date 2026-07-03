"""微信桌面运行时相关实现。"""

from .common import *  # noqa: F403

class KqWechat:
    @staticmethod
    def 创建微信实例():
        """ wxautox 初始化时会直接 print，某些控制台环境下会触发 stdout flush 异常 """
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return WeChat()

    @staticmethod
    def 扫码登录微信支付(user):
        """
        :param user: 微信群名/图片二维码存放的群位置
        """
        # 0 打开图片
        wx = KqWechat.创建微信实例()
        wx.ChatWith(user)
        messages = wx.GetAllMessage()
        msg = messages[-1]
        msg.click()  # wxautox才有click方法，wxauto基础版没有

        # 1 前置条件是已经用微信打开需要使用的二维码图片
        image = WeChatImage()
        # 使用微信内置的识别二维码功能
        image.t_qrcode.Click(move=False, simulateMove=False, return_pos=False)

        # 2 会弹出一个新的小程序窗口
        def calculate_relative_point(ltrb, dst_val):
            # todo 位置也是根据已有经验推断相对坐标的，也不太准，最好也是后期改成基于ocr的通用逻辑
            left, top, right, bottom = ltrb
            # 计算x轴中点
            x_center = (left + right) / 2
            # 计算y轴相对于原位置的偏移比例
            # (原目标y值 - 原top) / (原bottom - 原top)
            y_offset_ratio = (dst_val - 42) / (814 - 42)
            # 应用到新矩形
            new_height = bottom - top
            y_position = top + y_offset_ratio * new_height
            return (x_center, y_position)

        time.sleep(10)  # todo 暴力等待不太合理，后续可以考虑引入ocr来智能判定
        ct1 = uia.PaneControl(Name='微信支付商家助手', searchDepth=1)
        ct1 = UiCtrlNode(ct1, build_depth=5)
        ct1.activate()  # 必须把窗口激活到最前面

        # 3 点击进入商店，以及点击退出小程序窗口
        rect = ct1.BoundingRectangle
        ltrb = [rect.left, rect.top, rect.right, rect.bottom]
        # 进入商店的位置
        pyautogui.click(*calculate_relative_point(ltrb, 300))
        # 退出小程序的位置
        time.sleep(5)
        pyautogui.click(*calculate_relative_point(ltrb, 650))

        # 4 关闭窗口
        image.Close()

    @staticmethod
    def 从懒人转发获得短信内容(time_window=5, check_interval=1, timeout=300):
        """等待懒人转发的微信支付短信验证码，超时后抛异常交给外层重试。

        :param int time_window: 短信来电时间的有效窗口，单位分钟。
        :param float check_interval: 轮询微信消息的间隔，单位秒。
        :param timeout: 等待超时时间，单位秒；传入None表示不限制等待时间。
        :return str: 6位短信验证码。
        """
        from datetime import datetime, timedelta

        if check_interval <= 0:
            raise ValueError(f'check_interval必须大于0：{check_interval!r}')
        if timeout is not None and timeout < 0:
            raise ValueError(f'timeout不能为负数：{timeout!r}')

        def extract_verification_code(text):
            """从文本中提取6位验证码"""
            match = re.search(r'验证码【(\d{6})】', text)
            return match.group(1) if match else None

        def extract_call_time(text):
            """从文本中提取来电时间"""
            match = re.search(r'来电时间：(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', text)
            return match.group(1) if match else None

        def is_recent_time(time_str, time_window):
            """验证时间是否在指定时间窗口（分钟）内"""
            try:
                msg_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                return False

            current_time = datetime.now()
            time_diff = current_time - msg_time
            return timedelta(minutes=0) <= time_diff <= timedelta(minutes=time_window)

        def validate_message(text, time_window=5):
            """ 综合验证短信有效性 """
            code = extract_verification_code(text)
            time_str = extract_call_time(text)

            if not code or not time_str:
                return None

            return code if is_recent_time(time_str, time_window) else None

        service_name = '懒人信息转发服务'

        def message_text(message):
            """兼容 wxautox 的 Message 对象；短信文本可能在 sender/info 中。"""
            if message is None:
                return ''
            if isinstance(message, str):
                return message
            parts = []
            for attr in ('sender', 'content', 'text'):
                value = getattr(message, attr, None)
                if value:
                    parts.append(str(value))
            try:
                info = message.info
            except Exception:
                info = None
            if isinstance(info, (list, tuple)):
                parts.extend(str(x) for x in info if x)
            elif info:
                parts.append(str(info))
            if not parts:
                parts.append(str(message))
            return ' '.join(dict.fromkeys(parts))

        def collect_current_chat_texts(wx):
            """只读目标聊天当前已加载消息，避免 GetSession 递归扫描整棵会话树。"""
            texts = []
            try:
                messages = wx.GetAllMessage()
            except Exception as exc:
                logger.warning(f'读取微信短信转发会话消息失败：{exc!r}')
                return texts
            for message in reversed(messages):
                text = message_text(message)
                if '验证码' in text or '95017' in text:
                    texts.append(text)
            return texts

        wx = KqWechat.创建微信实例()
        chat_opened = False
        last_error = None

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            # 新短信提醒来电号码：验证码【644651】95017(微信支付)来电时间：2025-04-02 09:21:27
            if not chat_opened:
                try:
                    wx.ChatWith(service_name, timeout=5, exact=False)
                    chat_opened = True
                except Exception as exc:
                    last_error = exc
                    logger.warning(f'打开微信短信转发会话失败：{exc!r}')
            if chat_opened:
                for content in collect_current_chat_texts(wx):
                    if valid_code := validate_message(content, time_window):
                        logger.info('微信支付短信验证码已从微信目标会话消息获取')
                        return valid_code

            # 若目标会话读取失败，重置后下一轮重新打开，避免长期停在错误聊天。
            if not chat_opened:
                for content in collect_current_chat_texts(wx):
                    if valid_code := validate_message(content, time_window):
                        logger.info('微信支付短信验证码已从当前微信会话消息获取')
                        return valid_code

            now = time.monotonic()
            if deadline is not None and now >= deadline:
                detail = f'，last_error={last_error!r}' if last_error else ''
                raise TimeoutError(f'等待懒人信息转发服务短信验证码超时：timeout={timeout}s，time_window={time_window}min{detail}')

            if deadline is None:
                time.sleep(check_interval)
            else:
                time.sleep(min(check_interval, max(0, deadline - now)))
