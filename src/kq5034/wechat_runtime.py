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
    def 从懒人转发获得短信内容(time_window=5, check_interval=1):
        from datetime import datetime, timedelta

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

        wx = KqWechat.创建微信实例()
        wx.ChatWith('懒人信息转发服务')

        while True:
            # 取到最后条短信内容
            messages = wx.GetAllMessage()
            content = messages[-1].info[0]
            # 新短信提醒来电号码：验证码【644651】95017(微信支付)来电时间：2025-04-02 09:21:27

            # 验证是否符合格式，符合则返回值，否则等待接收短信
            if valid_code := validate_message(content, time_window):
                return valid_code

            time.sleep(check_interval)
