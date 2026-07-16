# -*- coding: utf-8 -*-
"""微信支付最小可用实现。"""

import json
import tempfile
from pathlib import Path

from .common import *  # noqa: F403
from .wechat_runtime import KqWechat


class Weipay(DpWebBase):
    def __init__(self, users=None):
        super().__init__('https://pay.weixin.qq.com')
        self.user = None
        if users:
            self.login(users)

    def login(self, users=None):
        tab = self.tab
        if tab.url != 'https://pay.weixin.qq.com/index.php/core/info':
            raw_timeout = os.getenv('KQ_WEIPAY_LOGIN_TIMEOUT_SECONDS', '240')
            try:
                login_timeout = max(30, int(float(raw_timeout)))
            except (TypeError, ValueError):
                login_timeout = 240
            login_deadline = time.time() + login_timeout
            tab.get('https://pay.weixin.qq.com')
            message_sent = False
            while tab.url != 'https://pay.weixin.qq.com/index.php/core/info':
                if time.time() >= login_deadline:
                    raise TimeoutError(f'微信支付登录超时：等待 {login_timeout} 秒后仍未进入商户首页')
                div = tab('tag:div@@class=qrcode-img')
                try:
                    is_invalid = div('tag:div@@class=alt@@text():二维码失效', timeout=3)
                except DrissionPage.errors.ContextLostError:
                    is_invalid = None
                if is_invalid:
                    logger.info(self.get_recive('二维码已过期，请发送任意消息，重新触发获取最新二维码'))
                    tab.refresh()
                    message_sent = False
                if message_sent:
                    time.sleep(5)
                    continue
                div = tab('tag:div@@id=IDQrcodeImg')
                file = div('tag:img').save(XlPath.tempdir(), 'qrcode')
                if users:
                    for user in users:
                        wechat_lock_send(user, '考勤工作需要，快帮我扫码登录微信支付', files=[file])
                    time.sleep(5)
                    with get_autogui_lock():
                        KqWechat.扫码登录微信支付(users[0], assume_current_chat=True)
                else:
                    print('>> 请扫码登录首页后，程序会自动继续运行...')
                message_sent = True
        self.user = tab('tag:a@@class=username').text.split('@')[0]

    def 重连标签页(self):
        try:
            tab = get_latest_not_dev_tab(self.browser)
            if tab:
                self.tab = tab
                return tab
        except Exception:
            pass
        self.tab = self.browser.latest_tab
        return self.tab

    @staticmethod
    def _clear_page_selection(tab):
        js = r"""
try {
  const selection = window.getSelection && window.getSelection();
  if (selection) selection.removeAllRanges();
  if (document.selection && document.selection.empty) document.selection.empty();
  return true;
} catch (err) {
  return false;
}
"""
        try:
            return bool(tab.run_js(js))
        except Exception:
            return False

    def get_recive(self, content):
        with WeChatSingletonLock(120) as wx:
            recive_msg = None
            wx.SendMsg(content, '考勤后台')
            while recive_msg is None:
                wx._show()
                wx.ChatWith('考勤后台')
                msgs = wx.GetAllMessage()
                for msg in msgs[::-1]:
                    if msg.content == content and msg.sender == 'Self':
                        break
                    recive_msg = msg.content
                    if recive_msg:
                        break
                time.sleep(3)
        return recive_msg

    def _fill_visible_inputs(self, tab, values, *, minimum_count=None, timeout=45):
        values = [str(v) for v in values]
        minimum_count = minimum_count or len(values)
        js = r"""
const values = JSON.parse(arguments[0] || '[]');
const minimumCount = arguments[1] || values.length;
try {
  const selection = window.getSelection && window.getSelection();
  if (selection) selection.removeAllRanges();
  if (document.selection && document.selection.empty) document.selection.empty();
} catch (err) {}
const isVisible = (el) => {
  if (!el) return false;
  let p = el;
  while (p) {
    const style = getComputedStyle(p);
    const cls = (p.className || '').toString();
    if (cls.includes('hide') || style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) {
      return false;
    }
    p = p.parentElement;
  }
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
};
const inputs = [...document.querySelectorAll('input')].filter((el) => isVisible(el) && !el.disabled);
if (inputs.length < minimumCount) return `BAD_INPUTS:${inputs.length}`;
for (let i = 0; i < values.length; i++) {
  const input = inputs[i];
  input.focus();
  input.value = '';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  input.dispatchEvent(new Event('change', {bubbles: true}));
  input.value = values[i];
  input.dispatchEvent(new Event('input', {bubbles: true}));
  input.dispatchEvent(new Event('change', {bubbles: true}));
  input.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}));
  input.dispatchEvent(new KeyboardEvent('keyup', {key: 'Enter', bubbles: true}));
  input.blur();
  input.dispatchEvent(new Event('blur', {bubbles: true}));
}
if (inputs.length) {
  document.body.click();
}
try {
  const selection = window.getSelection && window.getSelection();
  if (selection) selection.removeAllRanges();
  if (document.selection && document.selection.empty) document.selection.empty();
} catch (err) {}
return 'OK';
"""
        deadline = time.time() + timeout
        last_result = None
        while time.time() < deadline:
            result = tab.run_js(js, json.dumps(values, ensure_ascii=False), minimum_count)
            if result == 'OK':
                return
            last_result = result
            if not str(result).startswith('BAD_INPUTS:'):
                break
            time.sleep(1)

        try:
            title = tab.title
        except Exception:
            title = ''
        try:
            body_text = self._normalize_page_text(tab('tag:body').text)[:200]
        except Exception:
            body_text = ''
        raise RuntimeError(
            f'页面输入框填写失败：{last_result} url={tab.url} title={title!r} body={body_text!r}'
        )

    @staticmethod
    def _snapshot_download_dir():
        d = Path(tempfile.gettempdir())
        data = {}
        for f in d.iterdir():
            if not f.is_file():
                continue
            try:
                stat = f.stat()
                data[str(f)] = (stat.st_size, stat.st_mtime)
            except OSError:
                continue
        return data

    def _wait_for_new_download_file(self, before_files, timeout=120):
        d = Path(tempfile.gettempdir())
        allow_suffixes = {'.csv', '.xls', '.xlsx', '.zip'}
        deadline = time.time() + timeout
        while time.time() < deadline:
            candidates = []
            for f in d.iterdir():
                if not f.is_file():
                    continue
                if f.suffix.lower() not in allow_suffixes:
                    continue
                try:
                    stat = f.stat()
                except OSError:
                    continue
                key = str(f)
                signature = (stat.st_size, stat.st_mtime)
                if before_files.get(key) == signature:
                    continue
                candidates.append((stat.st_mtime, f, stat.st_size))

            candidates.sort(reverse=True)
            for _, f, size0 in candidates:
                time.sleep(0.8)
                try:
                    size1 = f.stat().st_size
                except OSError:
                    continue
                if size1 > 0 and size1 == size0:
                    return XlPath(f)
            time.sleep(1)
        raise RuntimeError('等待微信支付账单下载文件超时')

    @staticmethod
    def _iter_visible_tip_dialogs(tab):
        try:
            dialogs = tab.eles('t:div@@aria-label=温馨提示')
        except Exception:
            return []

        visible_dialogs = []
        for dialog in dialogs:
            try:
                if not dialog.states.has_rect:
                    continue
            except Exception:
                continue
            visible_dialogs.append(dialog)
        return visible_dialogs

    def _confirm_bill_download_dialog(self, tab, timeout=90):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                body_text = self._normalize_page_text(tab('tag:body').text)
            except Exception:
                body_text = ''

            if body_text:
                # 资金账单页 body.text 会长期包含隐藏模板中的“未开通资金账户”等文案，
                # 这里不能直接拿整页文本做权限判错，否则会误伤正常的下载确认流程。
                if any(x in body_text for x in ['请使用微信扫码登录', '二维码失效', '微信扫一扫登录', '请扫码登录', '登录超时，请重新登录']):
                    raise RuntimeError('微信支付登录态已失效，请重新扫码登录后再执行批量退款')
                if '暂时无该功能权限' in body_text and '请联系本商户员工管理员' in body_text:
                    raise RuntimeError('微信支付当前登录态缺少访问权限，请重新扫码或完成安全验证后再试')

            for dialog in self._iter_visible_tip_dialogs(tab):
                try:
                    text = self._normalize_page_text(dialog.text)
                except Exception:
                    continue
                if '当前商户号还未开通资金账户' in text and '无法查看资金账单' in text:
                    raise RuntimeError('微信支付当前商户号未开通资金账户，无法下载资金账单')
                if '账单打包完成' not in text and '请确认下载' not in text:
                    continue

                btn = None
                for selector in (
                        't:button@@class:el-button--primary@@text():确 定',
                        't:button@@class:el-button--primary',
                ):
                    try:
                        btn = dialog.ele(selector, timeout=1)
                    except Exception:
                        btn = None
                    if btn:
                        break
                if not btn:
                    continue

                btn.click(by_js=True)
                try:
                    dialog.wait.hidden()
                except Exception:
                    pass
                return True

            time.sleep(0.5)
        return False

    def download_monthly_records(self, month, save_dir=True):
        tab = self.tab
        logger.info(f'开始下载微信支付账单：month={month} save_dir={save_dir}')
        tab.get('https://pay.weixin.qq.com/index.php/xphp/cfund_bill_nc/funds_bill_nc#/')

        start_day = month + '-01'
        end_day = month + f'-{str(pd.Period(month).end_time.day)}'

        tab.wait(3)
        self._fill_visible_inputs(tab, [start_day, end_day], minimum_count=2)

        # 账单页左侧有“已结算查询”入口，这里只点主查询按钮。
        query_btn = None
        for selector in (
                't:button@@class:el-button--primary@@text()=查询',
                't:button@@class:el-button--primary@@text():查询',
                't:button@@text()=查询',
        ):
            try:
                query_btn = tab.ele(selector, timeout=3)
            except Exception:
                query_btn = None
            if query_btn:
                break
        if not query_btn:
            raise RuntimeError('未找到微信支付账单查询按钮')
        query_btn.click(by_js=True)
        tab.wait(5)

        if not save_dir:
            return None

        before_files = self._snapshot_download_dir()

        download_btn = None
        for selector in (
                't:a@@class=popups download@@text():业务明细账单',
                't:a@@class=popups download',
                't:a@@text()=下载',
        ):
            try:
                download_btn = tab.ele(selector, timeout=5)
            except Exception:
                download_btn = None
            if download_btn:
                break
        if not download_btn:
            raise RuntimeError('未找到微信支付账单下载入口')
        download_btn.click(by_js=True)

        if not self._confirm_bill_download_dialog(tab, timeout=90):
            raise RuntimeError('未找到微信支付账单下载确认弹窗')

        src_file = self._wait_for_new_download_file(before_files, timeout=120)
        if save_dir is True:
            save_dir = xlhome_dir('data/m2112kq5034/数据表')
        else:
            save_dir = XlPath(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        name = re.sub(r'_\d+\.csv$', '.csv', src_file.name)
        dst_file = save_dir / name
        shutil.copy(src_file, dst_file)
        logger.info(f'微信支付账单下载完成：month={month} file={dst_file}')
        return dst_file

    def daily_update(self, today=None):
        dst_files = []
        today = today or pd.Timestamp.now()
        current_year = today.year
        current_month = today.month
        current_day = today.day

        months = []
        if current_day in [1, 2]:
            if current_month == 1:
                months.append(f'{current_year - 1}-12')
            else:
                months.append(f'{current_year}-{str(current_month - 1).zfill(2)}')
        if current_day != 1:
            months.append(f'{current_year}-{str(current_month).zfill(2)}')

        logger.info(f'开始执行微信支付账单日更：today={today} months={months}')
        for month in months:
            dst_file = None
            last_error = None
            for i in range(4):
                try:
                    dst_file = self.download_monthly_records(month)
                    last_error = None
                    break
                except DrissionPage.errors.NoRectError as exc:
                    last_error = exc
                    logger.warning(f'月份账单下载遇到 NoRectError，准备重试：month={month} attempt={i + 1}/4 error={exc}')
                    time.sleep(1)
            if dst_file is not None:
                dst_files.append(dst_file)
            elif last_error is not None:
                raise last_error

        logger.info(f'微信支付账单日更完成：file_count={len(dst_files)} files={dst_files}')
        return dst_files

    @staticmethod
    def _normalize_page_text(text):
        return re.sub(r'\s+', ' ', str(text or '')).strip()

    @staticmethod
    def _coerce_money(value, default=0.0):
        text = re.sub(r'[^\d.\-]', '', str(value or ''))
        if not text:
            return default
        try:
            return float(text)
        except Exception:
            return default

    @staticmethod
    def _normalize_refund_query_type(voucher_id, query_type='auto'):
        query_type = str(query_type or 'auto').strip().lower()
        if query_type != 'auto':
            return query_type

        voucher_id = str(voucher_id or '').lstrip("`'").strip()
        if re.fullmatch(r'\d+', voucher_id):
            return 'pay_order' if voucher_id.startswith('42') else 'refund_id'
        return 'merchant_order'

    @staticmethod
    def _extract_summary_pairs(text):
        text = Weipay._normalize_page_text(text)
        known_keys = ['交易单号', '商户单号', '退款完成时间', '商户订单号', '支付单号', '交易时间']
        keyed_pattern = re.compile(
            r'(' + '|'.join(map(re.escape, known_keys)) + r')[：:]\s*(.*?)(?=\s+(?:' + '|'.join(map(re.escape, known_keys)) + r')[：:]|$)'
        )
        pairs = {key.strip(): value.strip() for key, value in keyed_pattern.findall(text)}
        if pairs:
            return pairs

        fallback = {}
        pattern = re.compile(r'([^\s:：]+)[：:]\s*(.*?)(?=\s+[^\s:：]+[：:]|$)')
        for key, value in pattern.findall(text):
            fallback[key.strip()] = value.strip()
        return fallback

    @staticmethod
    def _basename_stem(path):
        text = str(path or '').replace('\\', '/').rstrip('/')
        text = text.split('/')[-1]
        return re.sub(r'\.[^.]+$', '', text)

    @staticmethod
    def _get_element_render_state(ele):
        js = r"""
const el = this;
let hiddenAncestor = false;
let zIndex = 0;
let p = el;
while (p) {
  const style = getComputedStyle(p);
  const cls = (p.className || '').toString();
  if (cls.includes('hide') || style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) {
    hiddenAncestor = true;
    break;
  }
  const zi = parseInt(style.zIndex, 10);
  if (!Number.isNaN(zi)) zIndex = Math.max(zIndex, zi);
  p = p.parentElement;
}
const rect = el.getBoundingClientRect();
return `${hiddenAncestor ? 1 : 0}|${rect.width}|${rect.height}|${zIndex}`;
"""
        try:
            raw = ele.run_js(js)
            hidden_flag, width, height, z_index = str(raw).split('|', 3)
            return {'hidden_ancestor': hidden_flag == '1', 'width': float(width), 'height': float(height), 'z_index': int(float(z_index or 0))}
        except Exception:
            return {'hidden_ancestor': True, 'width': 0.0, 'height': 0.0, 'z_index': -1}

    def _is_element_really_visible(self, ele):
        state = self._get_element_render_state(ele)
        return not state['hidden_ancestor'] and state['width'] > 0 and state['height'] > 0

    @staticmethod
    def _dom_click(ele):
        js = r"""
const el = this;
if (!el) return false;
try {
  const selection = window.getSelection && window.getSelection();
  if (selection) selection.removeAllRanges();
  if (document.selection && document.selection.empty) document.selection.empty();
} catch (err) {}
['mouseover', 'mousedown', 'mouseup', 'click'].forEach((name) => {
  el.dispatchEvent(new MouseEvent(name, {bubbles: true, cancelable: true, view: window}));
});
if (typeof el.click === 'function') el.click();
return true;
"""
        try:
            return bool(ele.run_js(js))
        except Exception:
            return False

    def _click_submit_success_dialog_primary(self, tab):
        js = r"""
const normalize = (value) => String(value || '').replace(/\s+/g, ' ').trim();
const isVisible = (el) => {
  if (!el) return false;
  let p = el;
  while (p) {
    const style = getComputedStyle(p);
    const cls = (p.className || '').toString();
    if (cls.includes('hide') || style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) {
      return false;
    }
    p = p.parentElement;
  }
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
};
const dialogs = [...document.querySelectorAll('.dialog')].filter(isVisible);
const dialog = dialogs.find((node) => {
  const text = normalize(node.innerText || node.textContent);
  return text.includes('\u63d0\u4ea4\u6210\u529f') || text.includes('\u9000\u6b3e\u7533\u8bf7\u5df2\u63d0\u4ea4\u6210\u529f');
});
if (!dialog) return '';
const selectors = [
  '.dialog-ft a.btn.btn-primary.popups',
  'a.btn.btn-primary.popups',
  '.dialog-ft a.btn.btn-primary',
  'a.btn.btn-primary'
];
for (const selector of selectors) {
  const btn = dialog.querySelector(selector);
  if (btn && isVisible(btn)) {
    btn.click();
    return normalize(btn.innerText || btn.textContent || '');
  }
}
return '';
"""
        try:
            return str(tab.run_js(js) or '').strip()
        except Exception:
            return ''

    def _has_visible_weipay_security_dialog(self, tab):
        js = r"""
const normalize = (value) => String(value || '').replace(/\s+/g, ' ').trim();
const isVisible = (el) => {
  if (!el) return false;
  let p = el;
  while (p) {
    const style = getComputedStyle(p);
    const cls = (p.className || '').toString();
    if (cls.includes('hide') || style.display === 'none' || style.visibility === 'hidden') return false;
    p = p.parentElement;
  }
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
};
const dialogs = [...document.querySelectorAll('.dialog,.el-dialog,.el-message-box,.modal,[role="dialog"]')].filter(isVisible);
const dialog = dialogs.find((node) => normalize(node.innerText || node.textContent).includes('\u5b89\u5168\u9a8c\u8bc1'));
return dialog ? normalize(dialog.innerText || dialog.textContent).slice(0, 300) : '';
"""
        try:
            return str(tab.run_js(js) or '').strip()
        except Exception:
            return ''

    def _click_visible_element_js(self, ele):
        js = r"""
const el = this;
if (!el) return false;
try {
  const selection = window.getSelection && window.getSelection();
  if (selection) selection.removeAllRanges();
  if (document.selection && document.selection.empty) document.selection.empty();
} catch (err) {}
try {
  el.scrollIntoView({block: 'center', inline: 'center'});
} catch (err) {}
const rect = el.getBoundingClientRect();
const x = rect.left + rect.width / 2;
const y = rect.top + rect.height / 2;
for (const name of ['mouseover', 'mousemove', 'mousedown', 'mouseup', 'click']) {
  el.dispatchEvent(new MouseEvent(name, {
    bubbles: true,
    cancelable: true,
    view: window,
    clientX: x,
    clientY: y,
  }));
}
if (typeof el.click === 'function') el.click();
return true;
"""
        try:
            return bool(ele.run_js(js))
        except Exception:
            return False

    def _wait_after_weipay_confirm_click(self, tab, *, timeout=20):
        deadline = time.time() + timeout
        last_body = ''
        while time.time() < deadline:
            try:
                body_text = self._normalize_page_text(tab('tag:body').text)
            except Exception:
                body_text = ''
            last_body = body_text
            if '提交成功' in body_text or '退款申请已提交成功' in body_text:
                if self.尝试点击返款提交后的提示按钮(tab, timeout=3):
                    return {'ok': True, 'reason': 'submit_success_popup', 'body': body_text[:300]}
                return {'ok': True, 'reason': 'submit_success_visible', 'body': body_text[:300]}
            if '/cbatchrefund/refund#/pages/refund_list/refund_list' in str(getattr(tab, 'url', '')):
                return {'ok': True, 'reason': 'refund_result_page', 'body': body_text[:300]}
            if not self._has_visible_weipay_security_dialog(tab):
                # The risk-control dialog may close before the success dialog is rendered.
                time.sleep(0.5)
                try:
                    body_text = self._normalize_page_text(tab('tag:body').text)
                except Exception:
                    body_text = ''
                last_body = body_text or last_body
                if '提交成功' in body_text or '退款申请已提交成功' in body_text:
                    if self.尝试点击返款提交后的提示按钮(tab, timeout=3):
                        return {'ok': True, 'reason': 'submit_success_popup', 'body': body_text[:300]}
                    return {'ok': True, 'reason': 'submit_success_visible', 'body': body_text[:300]}
            time.sleep(0.5)
        return {'ok': False, 'reason': 'confirm_click_no_downstream_signal', 'body': last_body[:300]}

    def _click_weipay_confirm_and_wait(self, tab, confirm_action, *, submit_file=None, timeout=20):
        errors = []
        strategies = [
            ('native', lambda: confirm_action.click()),
            ('dom-events', lambda: self._dom_click(confirm_action)),
            ('center-js', lambda: self._click_visible_element_js(confirm_action)),
            ('by-js', lambda: confirm_action.click(by_js=True)),
        ]
        for name, clicker in strategies:
            self._clear_page_selection(tab)
            try:
                result = clicker()
                if result is False:
                    errors.append(f'{name}: false')
                    continue
            except Exception as exc:
                errors.append(f'{name}: {exc!r}')
                continue
            state = self._wait_after_weipay_confirm_click(tab, timeout=timeout)
            if state.get('ok'):
                logger.info(f'微信支付安全验证确认按钮点击成功：strategy={name} reason={state.get("reason")} file={submit_file!s}')
                return state
            logger.warning(
                f'微信支付安全验证确认按钮点击后未观察到下游信号，尝试下一种：'
                f'strategy={name} file={submit_file!s} state={state!r}'
            )
        try:
            body_text = self._normalize_page_text(tab('tag:body').text)
        except Exception:
            body_text = ''
        visible_actions = self._snapshot_visible_action_texts(tab)
        raise RuntimeError(
            '微信支付确认弹窗提交按钮点击后未进入成功状态'
            f'，submit_file={submit_file!s} errors={errors} visible_actions={visible_actions} body={body_text[:300]!r}'
        )

    def _click_weipay_security_confirm_once(self, tab, *, submit_file=None):
        self._clear_page_selection(tab)
        confirm_action = self._find_visible_confirm_action(tab, timeout=10)
        if confirm_action is None:
            visible_actions = self._snapshot_visible_action_texts(tab)
            raise RuntimeError(
                '微信支付确认弹窗未找到可见的提交按钮'
                f'，submit_file={submit_file!s} visible_actions={visible_actions}'
            )
        errors = []
        for name, clicker in [
            ('native', lambda: confirm_action.click()),
            ('dom-events', lambda: self._dom_click(confirm_action)),
            ('center-js', lambda: self._click_visible_element_js(confirm_action)),
            ('by-js', lambda: confirm_action.click(by_js=True)),
        ]:
            self._clear_page_selection(tab)
            try:
                result = clicker()
                if result is False:
                    errors.append(f'{name}: false')
                    continue
                logger.info(f'微信支付安全验证确认按钮已点击：strategy={name} file={submit_file!s}')
                return True
            except Exception as exc:
                errors.append(f'{name}: {exc!r}')
        raise RuntimeError(f'微信支付确认弹窗提交按钮点击失败，submit_file={submit_file!s} errors={errors}')

    def _wait_weipay_security_input_count(self, tab, *, minimum_count=1, timeout=20):
        deadline = time.time() + timeout
        last_inputs = []
        while time.time() < deadline:
            inputs = self._find_visible_dialog_inputs(tab)
            last_inputs = inputs
            if len(inputs) >= minimum_count:
                return inputs
            try:
                body_text = self._normalize_page_text(tab('tag:body').text)
            except Exception:
                body_text = ''
            if '提交成功' in body_text or '退款申请已提交成功' in body_text:
                return []
            time.sleep(0.5)
        logger.warning(
            f'等待微信支付安全验证输入框数量超时：minimum_count={minimum_count} '
            f'current_count={len(last_inputs)}'
        )
        return last_inputs

    def _click_visible_text_action_js(self, tab, texts):
        if not texts:
            return ''

        js = r"""
const targets = arguments[0] || [];
try {
  const selection = window.getSelection && window.getSelection();
  if (selection) selection.removeAllRanges();
  if (document.selection && document.selection.empty) document.selection.empty();
} catch (err) {}
const normalize = (value) => String(value || '').replace(/\s+/g, ' ').trim();
const isVisible = (el) => {
  if (!el) return false;
  let p = el;
  while (p) {
    const style = getComputedStyle(p);
    const cls = (p.className || '').toString();
    if (cls.includes('hide') || style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) {
      return false;
    }
    p = p.parentElement;
  }
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
};
const zIndexOf = (el) => {
  let best = 0;
  let p = el;
  while (p) {
    const zi = parseInt(getComputedStyle(p).zIndex || '0', 10);
    if (!Number.isNaN(zi)) best = Math.max(best, zi);
    p = p.parentElement;
  }
  return best;
};
const resolveAction = (el) => {
  const action = el.closest('a,button,[role="button"],.btn,.el-button,.popups,.close-dialog,.JSCloseDG,[tabindex]');
  if (action && isVisible(action)) return action;
  return el;
};
let best = null;
for (const node of document.querySelectorAll('body *')) {
  if (!isVisible(node)) continue;
  const text = normalize(node.innerText || node.textContent);
  if (!text || text.length > 40) continue;
  for (let i = 0; i < targets.length; i++) {
    const target = normalize(targets[i]);
    if (!target) continue;
    if (!(text === target || text.startsWith(target) || text.includes(target))) continue;
    const action = resolveAction(node);
    if (!isVisible(action)) continue;
    const rect = action.getBoundingClientRect();
    const inDialog = action.closest('.dialog,.el-dialog,.el-message-box,.modal,.layui-layer,.ui-dialog,[role="dialog"],.popups') ? 1 : 0;
    const score = inDialog * 100000 + zIndexOf(action) * 1000 + (text === target ? 300 : 0) + rect.width * rect.height - i;
    if (!best || score > best.score) {
      best = {target, action, score};
    }
  }
}
if (!best) return '';
['mousedown', 'mouseup', 'click'].forEach((name) => {
  best.action.dispatchEvent(new MouseEvent(name, {bubbles: true, cancelable: true, view: window}));
});
if (typeof best.action.click === 'function') best.action.click();
return best.target;
"""
        try:
            return str(tab.run_js(js, list(texts)) or '').strip()
        except Exception:
            return ''

    def _snapshot_visible_action_texts(self, tab):
        js = r"""
const normalize = (value) => String(value || '').replace(/\s+/g, ' ').trim();
const isVisible = (el) => {
  if (!el) return false;
  let p = el;
  while (p) {
    const style = getComputedStyle(p);
    const cls = (p.className || '').toString();
    if (cls.includes('hide') || style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) {
      return false;
    }
    p = p.parentElement;
  }
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
};
const rows = [];
for (const node of document.querySelectorAll('a,button,[role="button"],.btn,.el-button,.popups,[tabindex]')) {
  if (!isVisible(node)) continue;
  const text = normalize(node.innerText || node.textContent || node.value || '');
  if (!text || text.length > 50) continue;
  rows.push(text);
}
return [...new Set(rows)].slice(0, 20);
"""
        try:
            return tab.run_js(js) or []
        except Exception:
            return []

    def _click_visible_text_action(self, tab, texts):
        clicked_text = self._click_visible_text_action_js(tab, texts)
        if clicked_text:
            return clicked_text
        for target in texts:
            candidates = []
            for locator in [f'tag:a@@text()={target}', f'tag:button@@text()={target}', f'tag:span@@text()={target}', f'tag:a@@text():{target}', f'tag:button@@text():{target}', f'tag:span@@text():{target}']:
                try:
                    for ele in tab.eles(locator):
                        state = self._get_element_render_state(ele)
                        if state['hidden_ancestor'] or state['width'] <= 0 or state['height'] <= 0:
                            continue
                        candidates.append((state['z_index'], state['width'] * state['height'], ele))
                except Exception:
                    continue
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                candidates[0][2].click(by_js=True)
                return target
        return ''

    def _find_visible_upload_action(self, tab, *, timeout=30):
        deadline = time.time() + timeout
        self._clear_page_selection(tab)
        js = r"""
return (() => {
  const wanted = ['选择文件', '上传文件', '上传'];
  try {
    const selection = window.getSelection && window.getSelection();
    if (selection) selection.removeAllRanges();
    if (document.selection && document.selection.empty) document.selection.empty();
  } catch (err) {}
  const nodes = [...document.querySelectorAll('a,button,label,span,div,[role="button"]')];
  for (const node of document.querySelectorAll('[data-kq-upload-action]')) {
    node.removeAttribute('data-kq-upload-action');
  }
  const isVisible = (node) => {
    const rect = node.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    for (let cur = node; cur; cur = cur.parentElement) {
      const style = getComputedStyle(cur);
      if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
    }
    return true;
  };
  const scoreTag = (node) => ({A: 50, BUTTON: 40, LABEL: 30, SPAN: 20, DIV: 10}[node.tagName] || 0);
  const scored = [];
  for (const node of nodes) {
    if (!isVisible(node)) continue;
    const text = (node.innerText || node.value || node.getAttribute('aria-label') || node.getAttribute('title') || '').trim();
    if (!wanted.some(x => text === x || text.includes(x))) continue;
    const rect = node.getBoundingClientRect();
    scored.push([scoreTag(node), rect.width * rect.height, node]);
  }
  if (!scored.length) return false;
  scored.sort((a, b) => (b[0] - a[0]) || (b[1] - a[1]));
  scored[0][2].setAttribute('data-kq-upload-action', '1');
  return true;
})();
"""
        try:
            if tab.run_js(js):
                ele = tab.ele('css:[data-kq-upload-action="1"]', timeout=1)
                if ele:
                    return ele
        except Exception:
            pass
        locators = [
            'tag:a@@title=上传文件',
            'tag:a@@text()=选择文件',
            'tag:a@@text():选择文件',
            'tag:button@@text()=选择文件',
            'tag:button@@text():选择文件',
            'tag:label@@text()=选择文件',
            'tag:label@@text():选择文件',
            'tag:button@@text()=选择文件',
            'tag:a@@text()=选择文件',
            'tag:label@@text()=选择文件',
            'tag:span@@text()=选择文件',
            'tag:div@@text()=选择文件',
            'tag:button@@text():选择文件',
            'tag:a@@text():选择文件',
            'tag:label@@text():选择文件',
            'tag:span@@text():选择文件',
            'tag:div@@text():选择文件',
            'tag:a@@title=上传文件',
            'tag:button@@title=上传文件',
            'tag:label@@title=上传文件',
            'tag:div@@title=上传文件',
            'tag:span@@title=上传文件',
            'tag:a@@text()=上传文件',
            'tag:button@@text()=上传文件',
            'tag:label@@text()=上传文件',
            'tag:span@@text()=上传文件',
            'tag:div@@text()=上传文件',
            'tag:a@@text():上传文件',
            'tag:button@@text():上传文件',
            'tag:label@@text():上传文件',
            'tag:span@@text():上传文件',
            'tag:div@@text():上传文件',
            'tag:a@@text()=上传',
            'tag:button@@text()=上传',
            'tag:label@@text()=上传',
            'tag:span@@text()=上传',
            'tag:div@@text()=上传',
            'tag:a@@text():上传',
            'tag:button@@text():上传',
            'tag:label@@text():上传',
            'tag:span@@text():上传',
            'tag:div@@text():上传',
        ]

        while time.time() < deadline:
            candidates = []
            for order, locator in enumerate(locators):
                try:
                    for ele in tab.eles(locator):
                        state = self._get_element_render_state(ele)
                        if state['hidden_ancestor'] or state['width'] <= 0 or state['height'] <= 0:
                            continue
                        tag_name = ''
                        try:
                            tag_name = str(ele.tag).lower()
                        except Exception:
                            pass
                        tag_score = {'a': 50, 'button': 40, 'label': 30, 'span': 20, 'div': 10}.get(tag_name, 0)
                        candidates.append((tag_score, -order, state['z_index'], state['width'] * state['height'], ele, locator))
                except Exception:
                    continue
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
                return candidates[0][4]
            time.sleep(1)
        return None

    def _count_file_inputs_with_files(self, tab):
        js = r"""
const rows = [];
for (const node of document.querySelectorAll('input[type="file"]')) {
  const files = node.files ? node.files.length : 0;
  const text = String(node.value || '');
  rows.push({files, value: text, accept: String(node.accept || ''), className: String(node.className || '')});
}
return rows;
"""
        try:
            return tab.run_js(js) or []
        except Exception:
            return []

    def _has_uploaded_file(self, tab):
        states = self._count_file_inputs_with_files(tab)
        return any(int(item.get('files') or 0) > 0 for item in states if isinstance(item, dict))

    def _has_selected_upload_file(self, tab, file):
        file_name = Path(file).name
        if not file_name:
            return False
        js = r"""
const fileName = arguments[0];
const norm = value => String(value || '').replace(/\s+/g, ' ').trim();
const selectors = ['#upload-button', '.file-upload', '.form-item', '.content-bd'];
for (const selector of selectors) {
  for (const node of document.querySelectorAll(selector)) {
    if (norm(node.innerText || node.textContent).includes(fileName)) return true;
  }
}
return norm(document.body && (document.body.innerText || document.body.textContent)).includes(fileName);
"""
        try:
            return bool(tab.run_js(js, file_name))
        except Exception:
            return False

    def _find_file_input(self, tab):
        locators = [
            'tag:input@@type=file@@class:el-upload__input',
            'tag:input@@type=file',
        ]
        for locator in locators:
            try:
                elements = tab.eles(locator)
            except Exception:
                continue
            if elements:
                return elements[0]
        return None

    def _dispatch_file_input_change(self, tab):
        js = r"""
const isVisible = (el) => {
  if (!el) return false;
  let p = el;
  while (p) {
    const style = getComputedStyle(p);
    const cls = (p.className || '').toString();
    if (cls.includes('hide') || style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) {
      return false;
    }
    p = p.parentElement;
  }
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
};
for (const input of document.querySelectorAll('input[type="file"]')) {
  const host = input.closest('.el-upload,.upload-wrapper,[class*="upload"]') || input.parentElement;
  if (host && !isVisible(host)) continue;
  if (!input.files || !input.files.length) continue;
  input.dispatchEvent(new Event('input', {bubbles: true}));
  input.dispatchEvent(new Event('change', {bubbles: true}));
  return true;
}
return false;
"""
        try:
            return bool(tab.run_js(js))
        except Exception:
            return False

    def _wait_upload_bound(self, tab, file=None, timeout=5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._has_uploaded_file(tab):
                self._dispatch_file_input_change(tab)
                return True
            if file is not None and self._has_selected_upload_file(tab, file):
                return True
            time.sleep(0.5)
        return False

    def _upload_via_file_input(self, tab, file):
        try:
            anchor = tab.ele('tag:a@@title=上传文件', timeout=1)
        except Exception:
            anchor = None
        input_ele = self._find_file_input(tab)
        strategies = []
        if input_ele is not None:
            strategies.append((
                'input.set_file_input',
                lambda: (input_ele._set_file_input(str(file)), self._dispatch_file_input_change(tab)),
            ))
        if anchor is not None and self._is_element_really_visible(anchor):
            strategies.extend([
                ('anchor.to_upload', lambda: anchor.click.to_upload(file)),
                ('anchor.to_upload(by_js)', lambda: anchor.click.to_upload(file, by_js=True)),
                ('anchor.click', lambda: (tab.set.upload_files(file), anchor.click(), tab.wait.upload_paths_inputted())),
                ('anchor.click(by_js)', lambda: (tab.set.upload_files(file), anchor.click(by_js=True), tab.wait.upload_paths_inputted())),
            ])
        if input_ele is not None:
            strategies.extend([
                ('input.click(by_js)', lambda: (tab.set.upload_files(file), input_ele.click(by_js=True), tab.wait.upload_paths_inputted())),
                ('input.click', lambda: (tab.set.upload_files(file), input_ele.click(), tab.wait.upload_paths_inputted())),
            ])

        for name, action in strategies:
            try:
                action()
                if self._wait_upload_bound(tab, file=file, timeout=6):
                    logger.info(f'微信支付上传文件成功：strategy={name} file={str(file)!r}')
                    return True
                logger.warning(f'微信支付上传后页面未显示目标文件：strategy={name} file={str(file)!r}')
            except Exception as exc:
                logger.warning(f'微信支付上传策略失败，尝试下一种：strategy={name} file={str(file)!r} error={exc}')
        return False

    def _find_visible_dialog_inputs(self, tab):
        js = r"""
const input = this;
const normalize = (value) => String(value || '').replace(/\s+/g, ' ').trim();
const isVisible = (el) => {
  if (!el) return false;
  for (let p = el; p; p = p.parentElement) {
    const style = getComputedStyle(p);
    const cls = (p.className || '').toString();
    if (/(^|\s)hide(\s|$)/.test(cls) || style.display === 'none' || style.visibility === 'hidden') return false;
  }
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
};
if (!input || input.disabled || input.readOnly) return null;
const dialog = input.closest('.dialog,.el-dialog,.el-message-box,.modal,[role="dialog"]');
if (!dialog || !isVisible(dialog)) return null;
if (!normalize(dialog.innerText || dialog.textContent).includes('\u5b89\u5168\u9a8c\u8bc1')) return null;
const rect = input.getBoundingClientRect();
const formItem = input.closest('.form-item') || input.parentElement;
return {
  type: String(input.type || '').toLowerCase(),
  class_name: String(input.className || ''),
  placeholder: normalize(input.getAttribute('placeholder') || ''),
  parent_text: normalize(formItem && (formItem.innerText || formItem.textContent)),
  width: rect.width,
  height: rect.height,
};
"""
        candidates = []
        try:
            inputs = tab.eles('tag:input')
        except Exception:
            inputs = []
        for ele in inputs:
            try:
                row = ele.run_js(js)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            candidates.append({
                'ele': ele,
                'z_index': 0,
                'area': float(row.get('width') or 0) * float(row.get('height') or 0),
                'type': row.get('type') or '',
                'class_name': row.get('class_name') or '',
                'placeholder': row.get('placeholder') or '',
                'parent_text': row.get('parent_text') or '',
            })
        return candidates

    def _fill_weipay_risk_real_inputs(self, tab, values, *, minimum_count=None):
        """Fill WeChat Pay's transparent six-digit risk-control inputs.

        The visible six small boxes are disabled display inputs. The actual
        Vue-bound controls are ``input.real-input`` with opacity 0.
        """
        values = [str(v or '') for v in values]
        minimum_count = minimum_count or len(values)
        js = r"""
const values = JSON.parse(arguments[0] || '[]');
const minimumCount = arguments[1] || values.length;
const normalize = (value) => String(value || '').replace(/\s+/g, ' ').trim();
const isVisible = (el) => {
  if (!el) return false;
  let p = el;
  while (p) {
    const style = getComputedStyle(p);
    const cls = (p.className || '').toString();
    if (cls.includes('hide') || style.display === 'none' || style.visibility === 'hidden') return false;
    p = p.parentElement;
  }
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
};
try {
  const selection = window.getSelection && window.getSelection();
  if (selection) selection.removeAllRanges();
  if (document.selection && document.selection.empty) document.selection.empty();
} catch (err) {}
const dialogs = [...document.querySelectorAll('.dialog')].filter(isVisible)
  .filter((node) => normalize(node.innerText || node.textContent).includes('\u5b89\u5168\u9a8c\u8bc1'));
const dialog = dialogs[dialogs.length - 1];
if (!dialog) return {ok: false, reason: 'NO_SECURITY_DIALOG', count: 0, lens: []};
const inputs = [...dialog.querySelectorAll('input.real-input')].filter((el) => !el.disabled && !el.readOnly);
if (inputs.length < minimumCount) {
  return {ok: false, reason: 'BAD_INPUT_COUNT', count: inputs.length, lens: inputs.map((el) => String(el.value || '').length)};
}
const descriptor = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
const setter = descriptor && descriptor.set;
for (let i = 0; i < values.length; i += 1) {
  const input = inputs[i];
  const value = String(values[i] || '').slice(0, Number(input.maxLength || 6) || 6);
  if (!value) continue;
  input.focus();
  input.click();
  if (setter) setter.call(input, '');
  else input.value = '';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  if (setter) setter.call(input, value);
  else input.value = value;
  input.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'insertText', data: value}));
  input.dispatchEvent(new Event('change', {bubbles: true}));
  input.dispatchEvent(new KeyboardEvent('keyup', {bubbles: true, key: value.slice(-1) || '0'}));
}
const lens = inputs.map((el, i) => ({
  i,
  len: String(el.value || '').length,
  parent: normalize((el.closest('.form-item') && el.closest('.form-item').innerText) || ''),
}));
return {ok: lens.slice(0, values.length).every((item, i) => item.len === String(values[i] || '').slice(0, 6).length), count: inputs.length, lens};
"""
        try:
            result = tab.run_js(js, json.dumps(values, ensure_ascii=False), minimum_count)
        except Exception as exc:
            logger.warning(f'微信支付安全验证 real-input 填写失败：error={exc!r}')
            return None
        if isinstance(result, dict) and result.get('ok'):
            return result
        logger.warning(f'微信支付安全验证 real-input 填写未确认成功：result={result!r}')
        return None

    def _click_weipay_risk_send_sms(self, tab, *, timeout=8):
        send_btn = self._find_visible_text_action(tab, ['发送短信', '发送验证码', '获取验证码'])
        if send_btn is None:
            logger.warning('微信支付确认弹窗未找到明显的发送短信按钮')
            return False
        clicked = False
        try:
            send_btn.click()
            clicked = True
        except Exception:
            if self._dom_click(send_btn):
                clicked = True
            else:
                try:
                    send_btn.click(by_js=True)
                    clicked = True
                except Exception:
                    clicked = False
        if not clicked:
            return False

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                body_text = self._normalize_page_text(tab('tag:body').text)
            except Exception:
                body_text = ''
            if any(x in body_text for x in ['重新发送', '获取验证码', '验证码已发送']):
                return True
            time.sleep(0.5)
        return False

    def _find_visible_confirm_action(self, tab, *, timeout=10):
        deadline = time.time() + timeout
        self._clear_page_selection(tab)
        js = r"""
return (() => {
  const wanted = ['确定', '确认', '提交'];
  try {
    const selection = window.getSelection && window.getSelection();
    if (selection) selection.removeAllRanges();
    if (document.selection && document.selection.empty) document.selection.empty();
  } catch (err) {}
  for (const node of document.querySelectorAll('[data-kq-confirm-action]')) {
    node.removeAttribute('data-kq-confirm-action');
  }
  const isVisible = (node) => {
    const rect = node.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    for (let cur = node; cur; cur = cur.parentElement) {
      const style = getComputedStyle(cur);
      const cls = (cur.className || '').toString();
      if (cls.includes('hide') || style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
    }
    return true;
  };
  const zIndexOf = (node) => {
    let best = 0;
    for (let cur = node; cur; cur = cur.parentElement) {
      const zi = Number.parseInt(getComputedStyle(cur).zIndex || '0', 10);
      if (!Number.isNaN(zi)) best = Math.max(best, zi);
    }
    return best;
  };
  const scored = [];
  for (const node of document.querySelectorAll('a,button,div,span,[role="button"]')) {
    if (!isVisible(node)) continue;
    const text = (node.innerText || node.value || node.getAttribute('aria-label') || node.getAttribute('title') || '').trim();
    if (!wanted.some(x => text === x || text.includes(x))) continue;
    const rect = node.getBoundingClientRect();
    const inDialog = node.closest('.dialog,.el-dialog,.el-message-box,.modal,.layui-layer,.ui-dialog,[role="dialog"],.popups') ? 1 : 0;
    const cls = (node.className || '').toString();
    const actionable = ['A', 'BUTTON'].includes(node.tagName) || node.getAttribute('role') === 'button' || /(^|\s)(btn|button|primary|submit)(\s|$)/i.test(cls) ? 1 : 0;
    const exact = wanted.includes(text) ? 1 : 0;
    const compact = text.length <= 8 ? 1 : 0;
    scored.push([inDialog, exact, actionable, compact, zIndexOf(node), -(rect.width * rect.height), node]);
  }
  if (!scored.length) return false;
  scored.sort((a, b) => (b[0] - a[0]) || (b[1] - a[1]) || (b[2] - a[2]) || (b[3] - a[3]) || (b[4] - a[4]) || (b[5] - a[5]));
  scored[0][6].setAttribute('data-kq-confirm-action', '1');
  return true;
})();
"""
        try:
            if tab.run_js(js):
                ele = tab.ele('css:[data-kq-confirm-action="1"]', timeout=1)
                if ele:
                    return ele
        except Exception:
            pass
        locators = [
            'tag:a@@text()=确定',
            'tag:button@@text()=确定',
            'tag:div@@text()=确定',
            'tag:span@@text()=确定',
            'tag:a@@text():确定',
            'tag:button@@text():确定',
            'tag:div@@text():确定',
            'tag:span@@text():确定',
            'tag:a@@text()=确认',
            'tag:button@@text()=确认',
            'tag:div@@text()=确认',
            'tag:span@@text()=确认',
            'tag:a@@text():确认',
            'tag:button@@text():确认',
            'tag:div@@text():确认',
            'tag:span@@text():确认',
            'tag:a@@text()=提交',
            'tag:button@@text()=提交',
            'tag:div@@text()=提交',
            'tag:span@@text()=提交',
            'tag:a@@text():提交',
            'tag:button@@text():提交',
            'tag:div@@text():提交',
            'tag:span@@text():提交',
            'tag:a@@class=btn btn-primary align-center@@text()=确定',
            'tag:button@@class=btn btn-primary align-center@@text()=确定',
        ]
        while time.time() < deadline:
            candidates = []
            for locator in locators:
                try:
                    for ele in tab.eles(locator):
                        state = self._get_element_render_state(ele)
                        if state['hidden_ancestor'] or state['width'] <= 0 or state['height'] <= 0:
                            continue
                        candidates.append((state['z_index'], state['width'] * state['height'], ele))
                except Exception:
                    continue
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                return candidates[0][2]
            time.sleep(1)
        return None

    def _find_visible_text_action(self, tab, texts):
        self._clear_page_selection(tab)
        js = rf"""
return (() => {{
  const wanted = {json.dumps(list(texts), ensure_ascii=False)};
  try {{
    const selection = window.getSelection && window.getSelection();
    if (selection) selection.removeAllRanges();
    if (document.selection && document.selection.empty) document.selection.empty();
  }} catch (err) {{}}
  const nodes = [...document.querySelectorAll('a,button,label,span,div,[role="button"]')];
  for (const node of document.querySelectorAll('[data-kq-text-action]')) {{
    node.removeAttribute('data-kq-text-action');
  }}
  const isVisible = (node) => {{
    const rect = node.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    for (let cur = node; cur; cur = cur.parentElement) {{
      const style = getComputedStyle(cur);
      if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
    }}
    return true;
  }};
  const scored = [];
  for (const node of nodes) {{
    if (!isVisible(node)) continue;
    const text = (node.innerText || node.value || node.getAttribute('aria-label') || node.getAttribute('title') || '').trim();
    if (!wanted.some(x => text === x || text.includes(x))) continue;
    const rect = node.getBoundingClientRect();
    const cls = (node.className || '').toString();
    const inDialog = node.closest('.dialog,.el-dialog,.el-message-box,.modal,.layui-layer,.ui-dialog,[role="dialog"],.popups') ? 1 : 0;
    const actionable = ['A', 'BUTTON', 'LABEL'].includes(node.tagName) || node.getAttribute('role') === 'button' || /(^|\s)(btn|button|primary|send|sms)(\s|$)/i.test(cls) ? 1 : 0;
    const exact = wanted.includes(text) ? 1 : 0;
    const compact = text.length <= 12 ? 1 : 0;
    scored.push([inDialog, exact, actionable, compact, Number.parseInt(getComputedStyle(node).zIndex || '0', 10) || 0, -(rect.width * rect.height), node]);
  }}
  if (!scored.length) return false;
  scored.sort((a, b) => (b[0] - a[0]) || (b[1] - a[1]) || (b[2] - a[2]) || (b[3] - a[3]) || (b[4] - a[4]) || (b[5] - a[5]));
  scored[0][6].setAttribute('data-kq-text-action', '1');
  return true;
}})();
"""
        try:
            if tab.run_js(js):
                ele = tab.ele('css:[data-kq-text-action="1"]', timeout=1)
                if ele:
                    return ele
        except Exception:
            pass
        locators = []
        for target in texts:
            locators.extend([
                f'tag:a@@text()={target}',
                f'tag:button@@text()={target}',
                f'tag:label@@text()={target}',
                f'tag:span@@text()={target}',
                f'tag:div@@text()={target}',
                f'tag:a@@text():{target}',
                f'tag:button@@text():{target}',
                f'tag:label@@text():{target}',
                f'tag:span@@text():{target}',
                f'tag:div@@text():{target}',
            ])
        candidates = []
        for locator in locators:
            try:
                for ele in tab.eles(locator):
                    state = self._get_element_render_state(ele)
                    if state['hidden_ancestor'] or state['width'] <= 0 or state['height'] <= 0:
                        continue
                    candidates.append((state['z_index'], state['width'] * state['height'], ele))
            except Exception:
                continue
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    def _raise_if_weipay_auth_invalid(self, body_text):
        text = self._normalize_page_text(body_text)
        if any(x in text for x in ['请使用微信扫码登录', '二维码失效', '微信扫一扫登录', '请扫码登录', '登录超时，请重新登录']):
            raise RuntimeError('微信支付登录态已失效，请重新扫码登录后再执行批量退款')
        if '暂时无该功能权限' in text and '请联系本商户员工管理员' in text:
            raise RuntimeError('微信支付当前登录态缺少访问权限，请重新扫码或完成安全验证后再试')
        if any(x in text for x in ['你没有操作此功能的权限', '你没有此页面的查看操作权限', '所需权限', '请联系本商户员工管理员修改权限']):
            raise RuntimeError('微信支付当前账号缺少退款操作权限，请使用具备权限的商户员工管理员或超级管理员账号')
        if any(x in text for x in ['你不是超级管理员', '请用超级管理员帐号登录执行此操作']):
            raise RuntimeError('微信支付当前账号不是超级管理员，无法执行该退款操作，请切换超级管理员账号后再试')
        if '安全验证' in text and '进入账户概况' in text:
            raise RuntimeError('微信支付当前流程被安全验证拦截，请先在本机完成安全验证后再执行批量退款')
        if '当前商户号还未开通资金账户' in text and '无法查看资金账单' in text:
            raise RuntimeError('微信支付当前商户号未开通资金账户，无法下载资金账单')

    @staticmethod
    def _batch_refund_submit_marker_path(file):
        file = XlPath(file)
        return file.parent / f'{file.stem}.submitted.json'

    def _load_batch_refund_submit_marker(self, file):
        marker = self._batch_refund_submit_marker_path(file)
        if not marker.is_file():
            return None
        try:
            data = json.loads(marker.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                data.setdefault('marker_file', str(marker))
                return data
        except Exception as exc:
            logger.warning(f'批量退款提交标记读取失败，忽略并继续：file={file!s}，marker={marker!s}，error={exc}')
        return None

    def _save_batch_refund_submit_marker(self, file, *, stage, status_text=''):
        file = XlPath(file)
        marker = self._batch_refund_submit_marker_path(file)
        payload = {
            'file': str(file),
            'file_name': file.name,
            'stage': stage,
            'status_text': self._normalize_page_text(status_text)[:500],
            'saved_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return payload

    def 尝试点击返款提交后的提示按钮(self, tab, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                body_text = self._normalize_page_text(tab('tag:body').text)
            except Exception:
                body_text = ''

            if '提交成功' in body_text or '退款申请已提交成功' in body_text:
                clicked_text = self._click_submit_success_dialog_primary(tab)
                if clicked_text:
                    tab.wait(1)
                    return True

                try:
                    btn = tab('tag:a@@class=btn btn-primary popups@@text()=确认', timeout=1)
                except Exception:
                    btn = None
                if btn and self._is_element_really_visible(btn):
                    if not self._dom_click(btn):
                        btn.click(by_js=True)
                    tab.wait(1)
                    return True

            time.sleep(0.5)

        return False

    def 填写密码与验证码(self, tab, submit_file=None, sms_timeout=300):
        self._clear_page_selection(tab)
        inputs = self._find_visible_dialog_inputs(tab)
        try:
            page_user = self._normalize_page_text(tab('tag:a@@class=username').text)
        except Exception:
            page_user = ''
        page_user_key = (page_user or '').split('@')[0]
        password_keys = []
        if self.user:
            password_keys.append(f'XL_KQ_PAY_PASSWORD_{self.user}')
        if page_user_key and page_user_key != self.user:
            password_keys.append(f'XL_KQ_PAY_PASSWORD_{page_user_key}')
        password_keys.append('XL_KQ_PAY_PASSWORD')
        passwd = ''
        for key in password_keys:
            passwd = XlEnv.get(key, decoding=True)
            if passwd:
                break
        if not passwd:
            user_hint = page_user_key or str(self.user or '')
            raise RuntimeError(
                '微信支付安全验证需要操作密码，但本机未配置密码环境变量'
                f'，请配置 XL_KQ_PAY_PASSWORD_{user_hint} 或 XL_KQ_PAY_PASSWORD 后重试，submit_file={submit_file!s}'
            )
        if not inputs:
            visible_actions = self._snapshot_visible_action_texts(tab)
            try:
                body_text = self._normalize_page_text(tab('tag:body').text)
            except Exception:
                body_text = ''
            if '安全验证' not in body_text:
                self._raise_if_weipay_auth_invalid(body_text)
            raise RuntimeError(
                '微信支付提交确认弹窗未找到可见输入框'
                f'，submit_file={submit_file!s} visible_actions={visible_actions} body={body_text[:300]!r}'
            )

        password_input = None
        sms_input = None
        for item in inputs:
            text = f"{item['type']} {item['class_name']} {item['placeholder']} {item.get('parent_text', '')}".lower()
            if password_input is None and any(x in text for x in ['password', 'passwd', 'pay', '密码']):
                password_input = item['ele']
                continue
            if sms_input is None and any(x in text for x in ['sms', 'code', '验证码', '短信']):
                sms_input = item['ele']

        if password_input is None:
            password_input = inputs[0]['ele']
        if sms_input is None and len(inputs) > 1:
            for item in inputs:
                if item['ele'] is not password_input:
                    sms_input = item['ele']
                    break

        if not self._fill_weipay_risk_real_inputs(tab, [passwd], minimum_count=1):
            logger.warning(f'微信支付操作密码 real-input 填充失败，回退到可见输入：submit_file={submit_file!s}')
            self._clear_page_selection(tab)
            password_input.input(passwd, clear=True)

        if sms_input is None:
            self._click_weipay_security_confirm_once(tab, submit_file=submit_file)
            inputs = self._wait_weipay_security_input_count(tab, minimum_count=2, timeout=20)
            password_input = None
            sms_input = None
            for item in inputs:
                text = f"{item['type']} {item['class_name']} {item['placeholder']} {item.get('parent_text', '')}".lower()
                if password_input is None and any(x in text for x in ['password', 'passwd', 'pay', '密码']):
                    password_input = item['ele']
                    continue
                if sms_input is None and any(x in text for x in ['sms', 'code', '验证码', '短信']):
                    sms_input = item['ele']
            if sms_input is None and len(inputs) > 1:
                sms_input = inputs[-1]['ele']
            if sms_input is None:
                try:
                    body_text = self._normalize_page_text(tab('tag:body').text)
                except Exception:
                    body_text = ''
                if '提交成功' in body_text or '退款申请已提交成功' in body_text:
                    popup_confirmed = self.尝试点击返款提交后的提示按钮(tab)
                    if popup_confirmed and submit_file is not None:
                        self._save_batch_refund_submit_marker(
                            submit_file,
                            stage='submit_success_popup',
                            status_text='提交成功',
                        )
                    return popup_confirmed
                raise RuntimeError(
                    '微信支付操作密码确认后未进入短信验证阶段'
                    f'，submit_file={submit_file!s} body={body_text[:300]!r}'
                )

        if sms_input is not None:
            self._clear_page_selection(tab)
            if not self._click_weipay_risk_send_sms(tab):
                logger.warning(f'微信支付确认弹窗未找到明显的发送短信按钮，继续等待验证码：submit_file={submit_file!s}')
            time.sleep(10)
            try:
                with get_autogui_lock():
                    vcode = KqWechat.从懒人转发获得短信内容(timeout=sms_timeout)
            except TimeoutError as err:
                raise TimeoutError(
                    f'微信支付短信验证码等待超过{sms_timeout}s，返款未提交：submit_file={submit_file!s}'
                ) from err
            if not self._fill_weipay_risk_real_inputs(tab, [passwd or '', vcode], minimum_count=2):
                logger.warning(f'微信支付验证码 real-input 填充失败，回退到旧定位：submit_file={submit_file!s}')
                self._clear_page_selection(tab)
                sms_input.input(vcode, clear=True)
        time.sleep(1)
        self._clear_page_selection(tab)
        confirm_action = self._find_visible_confirm_action(tab, timeout=10)
        if confirm_action is None:
            visible_actions = self._snapshot_visible_action_texts(tab)
            raise RuntimeError(
                '微信支付确认弹窗未找到可见的提交按钮'
                f'，submit_file={submit_file!s} visible_actions={visible_actions}'
            )
        click_state = self._click_weipay_confirm_and_wait(tab, confirm_action, submit_file=submit_file)
        popup_confirmed = bool(click_state.get('ok'))
        if popup_confirmed and submit_file is not None:
            self._save_batch_refund_submit_marker(
                submit_file,
                stage=click_state.get('reason') or 'submit_confirmed',
                status_text=click_state.get('body') or '提交成功',
            )
        return popup_confirmed

    @staticmethod
    def _parse_trade_search_result_html(html):
        soup = BeautifulSoup(html, 'lxml')
        row = {}
        for tr in soup.find_all('tr'):
            th = tr.find('th')
            td = tr.find('td')
            if th and td:
                row[Weipay._normalize_page_text(th.get_text(' ', strip=True))] = Weipay._normalize_page_text(td.get_text(' ', strip=True))
        return row

    def search_refund(self, voucher_id):
        tab = self.tab
        voucher_id = str(voucher_id or '').lstrip("`'").strip()
        if not voucher_id:
            return {'error': '订单号不能为空'}

        tab.get('https://pay.weixin.qq.com/index.php/core/trade/search_new')
        input_name = 'mmpay_order_id' if re.fullmatch(r'\d+', voucher_id) else 'merchant_order_id'
        input_ele = tab.ele(f'tag:input@@name={input_name}', timeout=15)
        if not input_ele:
            return {'error': '微信支付订单查询页未加载完成'}
        input_ele.input(voucher_id, clear=True)

        query_btn = tab.ele('tag:a@@id=idQueryButton', timeout=5) or tab.ele('tag:button@@text()=查询', timeout=5)
        if not query_btn:
            return {'error': '未找到微信支付订单查询按钮'}
        query_btn.click(by_js=True)

        deadline = time.time() + 20
        tips_text = ''
        table = None
        while time.time() < deadline:
            tips = tab.ele('tag:div@@class=tips-error', timeout=1)
            tips_text = self._normalize_page_text(tips.text if tips else '')
            if tips_text:
                return {'error': tips_text}
            table = tab.ele('tag:div@@class=table-wrp with-border', timeout=1)
            if table and any(k in table.text for k in ['支付单号', '交易单号', '商户订单号', '商户单号', '订单金额']):
                break
            time.sleep(0.5)
        if not table:
            return {'error': '微信支付订单查询结果未加载完成'}

        html = table('tag:table').html
        raw = self._parse_trade_search_result_html(html)
        row = dict(raw)

        alias_map = {
            '交易单号': '支付单号',
            '微信支付订单号': '支付单号',
            '商户单号': '商户订单号',
            '支付时间': '交易时间',
            '交易状态': '订单状态',
            '实付金额': '订单金额',
            '已申请退款金额': '已返款',
            '已退款金额': '已返款',
        }
        for source_key, target_key in alias_map.items():
            if source_key in row and target_key not in row:
                row[target_key] = row[source_key]

        row['订单金额'] = self._coerce_money(row.get('订单金额'))
        refunded_amount = self._coerce_money(row.get('已返款'))
        if refunded_amount <= 0 and (row.get('支付单号') or row.get('商户订单号')):
            try:
                details = self.search_refund_details(row.get('商户订单号') or row.get('支付单号'), query_type='auto', raise_err=False)
            except Exception:
                details = []
            if details:
                refunded_amount = round(sum(self._coerce_money(item.get('退款金额')) for item in details), 2)
        row['已返款'] = refunded_amount
        return row

    def _fill_precise_refund_query(self, voucher_id, query_type='auto'):
        voucher_id = str(voucher_id or '').lstrip("`'").strip()
        query_type = self._normalize_refund_query_type(voucher_id, query_type)
        input_index_map = {
            'pay_order': 0,
            'merchant_order': 1,
            'refund_id': 2,
        }
        if query_type not in input_index_map:
            raise ValueError(f'不支持的退款查询类型：{query_type}')

        tab = self.tab
        tab.get('https://pay.weixin.qq.com/index.php/core/refundquery')
        tab.wait(2)
        precise_btn = tab.ele('tag:a@@id=preciseRefundSearchBtn', timeout=10) or tab.ele('tag:a@@text()=精确查询', timeout=5)
        if not precise_btn:
            raise RuntimeError('未找到微信支付退款精确查询入口')
        precise_btn.click(by_js=True)
        tab.wait(1)

        js = r"""
const voucherId = arguments[0];
const targetIndex = arguments[1];
const normalize = (value) => String(value || '').replace(/\s+/g, ' ').trim();
const isVisible = (el) => {
  if (!el) return false;
  let p = el;
  while (p) {
    const style = getComputedStyle(p);
    const cls = (p.className || '').toString();
    if (cls.includes('hide') || style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) {
      return false;
    }
    p = p.parentElement;
  }
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
};
const box = [...document.querySelectorAll('.preciseRefundSearch,.preciseQuery,[class*="precise"]')].find(isVisible);
if (!box) return 'NO_BOX';
const inputs = [...box.querySelectorAll('input')].filter(isVisible);
if (inputs.length < 3) return `BAD_INPUTS:${inputs.length}`;
for (const input of inputs) {
  input.focus();
  input.value = '';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  input.dispatchEvent(new Event('change', {bubbles: true}));
}
const target = inputs[targetIndex];
target.focus();
target.value = voucherId;
target.dispatchEvent(new Event('input', {bubbles: true}));
target.dispatchEvent(new Event('change', {bubbles: true}));
const btn = [...box.querySelectorAll('a,button')].find((el) => isVisible(el) && normalize(el.innerText || el.textContent).includes('查询'));
if (!btn) return 'NO_QUERY_BTN';
btn.click();
return 'OK';
"""
        result = tab.run_js(js, voucher_id, input_index_map[query_type])
        if result != 'OK':
            raise RuntimeError(f'退款精确查询表单填写失败：{result}')

        deadline = time.time() + 20
        while time.time() < deadline:
            table = tab.ele('tag:div@@class=table-wrp with-border table-receive', timeout=2)
            if not table:
                time.sleep(0.5)
                continue
            table_text = self._normalize_page_text(table.text)
            if '正在查询' in table_text:
                time.sleep(0.5)
                continue
            return
        raise RuntimeError('退款精确查询结果页未在预期时间内加载完成')

    @staticmethod
    def _parse_refund_query_table_html(html):
        soup = BeautifulSoup(html, 'lxml')
        records = []
        for tbody in soup.select('tbody'):
            rows = tbody.find_all('tr', recursive=False)
            if len(rows) < 2:
                continue

            summary_row, detail_row = rows[0], rows[1]
            summary_pairs = Weipay._extract_summary_pairs(summary_row.get_text(' ', strip=True))
            cells = detail_row.find_all('td', recursive=False)
            if len(cells) < 5:
                continue

            records.append({
                '交易单号': summary_pairs.get('交易单号', ''),
                '商户单号': summary_pairs.get('商户单号', ''),
                '退款完成时间': summary_pairs.get('退款完成时间', ''),
                '退款单号': Weipay._normalize_page_text(cells[0].get_text(' ', strip=True)),
                '退款金额': Weipay._coerce_money(cells[1].get_text(' ', strip=True)),
                '退款状态': Weipay._normalize_page_text(cells[2].get_text(' ', strip=True)),
                '申请人': Weipay._normalize_page_text(cells[3].get_text(' ', strip=True)),
                '提交时间': Weipay._normalize_page_text(cells[4].get_text(' ', strip=True)),
            })
        return records

    def _get_refund_query_page_state(self):
        tab = self.tab
        pager = tab.ele('tag:div@@class=pagination fr', timeout=2)
        if not pager:
            return 1, 1

        labels = pager.eles('tag:label')
        if len(labels) >= 2:
            try:
                return int(labels[0].text.strip()), int(labels[1].text.strip())
            except Exception:
                pass
        return 1, 1

    def _goto_next_refund_query_page(self, previous_first_refund_id=''):
        tab = self.tab
        next_btn = tab.ele('tag:a@@class=btn page-next', timeout=5)
        if not next_btn:
            return False

        next_btn.click(by_js=True)
        deadline = time.time() + 15
        while time.time() < deadline:
            table = tab.ele('tag:div@@class=table-wrp with-border table-receive', timeout=2)
            if not table:
                time.sleep(0.5)
                continue
            records = self._parse_refund_query_table_html(table.html)
            if records and records[0]['退款单号'] != previous_first_refund_id:
                return True
            time.sleep(0.5)
        raise RuntimeError('退款详情查询翻页后结果未刷新')

    def search_refund_details(self, voucher_id, query_type='auto', raise_err=True):
        try:
            self._fill_precise_refund_query(voucher_id, query_type)
            tab = self.tab
            body_text = self._normalize_page_text(tab('tag:body').text)
            if '没有查询结果' in body_text or '暂无数据' in body_text:
                return []

            all_records = []
            seen_refund_ids = set()
            while True:
                table = tab.ele('tag:div@@class=table-wrp with-border table-receive', timeout=5)
                if not table:
                    break

                records = self._parse_refund_query_table_html(table.html)
                for row in records:
                    refund_id = row['退款单号']
                    if refund_id in seen_refund_ids:
                        continue
                    seen_refund_ids.add(refund_id)
                    all_records.append(row)

                current_page, total_pages = self._get_refund_query_page_state()
                if current_page >= total_pages:
                    break
                first_refund_id = records[0]['退款单号'] if records else ''
                self._goto_next_refund_query_page(first_refund_id)

            all_records.sort(key=lambda row: pd.to_datetime(row['退款完成时间']) if row['退款完成时间'] else pd.Timestamp.max)
            return all_records
        except Exception:
            if raise_err:
                raise
            return []

    def wait_refund_completion(self, timeout=300, voucher_id=None, expected_refund_amount=None, baseline_refunded_amount=0):
        if not voucher_id:
            return self.wait_batch_refund_completion(timeout=timeout)

        tab = self.tab
        deadline = time.time() + timeout
        last_status_text = ''
        while time.time() < deadline:
            try:
                body_text = tab('tag:body').text
            except Exception:
                body_text = ''
            if any(x in body_text for x in ['退款申请已提交成功', '提交成功']):
                self.尝试点击返款提交后的提示按钮(tab, timeout=3)
                tab.wait(1)

            try:
                row = self.search_refund(voucher_id)
            except Exception as exc:
                last_status_text = f'订单轮询失败：{exc}'
                time.sleep(2)
                continue

            if 'error' in row:
                last_status_text = str(row['error'])
                time.sleep(2)
                continue

            refunded_amount = float(row.get('已返款') or 0)
            trade_status = str(row.get('订单状态') or row.get('交易状态') or '')
            last_status_text = f'订单状态={trade_status} 已返款={refunded_amount}'
            target_amount = float(baseline_refunded_amount or 0) + float(expected_refund_amount or 0)
            if refunded_amount + 1e-9 >= target_amount:
                return
            if trade_status in ['退款成功', '全额退款', '已退款'] and not expected_refund_amount:
                return
            time.sleep(2)

        raise RuntimeError(f'微信支付退款结果未完成，url={tab.url}，title={tab.title}，状态摘要={last_status_text!r}')

    def request_single_refund(self, voucher_id, refund_amount=0, refund_reason=''):
        baseline_refunded_amount = 0
        try:
            row = self.search_refund(voucher_id)
            if 'error' not in row:
                baseline_refunded_amount = float(row.get('已返款') or 0)
        except Exception:
            pass

        tab = self.tab
        tab.get('https://pay.weixin.qq.com/index.php/core/refundapply')
        input_name = 'wxOrderNum' if re.fullmatch(r'\d+', str(voucher_id or '').lstrip("`'")) else 'mchOrderNum'

        order_input = tab.ele(f'tag:input@@name={input_name}', timeout=15)
        if not order_input:
            raise RuntimeError('未找到单笔退款申请页的订单输入框')
        order_input.input(str(voucher_id).lstrip("`'"), clear=True)

        apply_btn = tab.ele('tag:a@@id=applyRefundBtn', timeout=5) or tab.ele('tag:button@@text()=申请退款', timeout=5)
        if not apply_btn:
            raise RuntimeError('未找到单笔退款申请按钮')
        apply_btn.click(by_js=True)

        refund_amount_input = tab.ele('tag:input@@name=refund_amount', timeout=15)
        if not refund_amount_input:
            raise RuntimeError('未找到退款金额输入框')
        refund_amount_input.input(refund_amount, clear=True)

        reason_input = tab.ele('#textInput', timeout=5) or tab.ele('tag:textarea', timeout=5)
        if reason_input:
            reason_input.input(refund_reason)

        commit_btn = tab.ele('#commitRefundApplyBtn', timeout=5) or tab.ele('tag:button@@text()=提交申请', timeout=5)
        if not commit_btn:
            raise RuntimeError('未找到提交退款申请按钮')
        commit_btn.click(by_js=True)
        tab.wait(2)

        self.填写密码与验证码(tab)
        self.wait_refund_completion(
            voucher_id=voucher_id,
            expected_refund_amount=refund_amount,
            baseline_refunded_amount=baseline_refunded_amount,
        )

    def _extract_batch_refund_status(self, submit_started_at=None, file_name=''):
        tab = self.tab
        js = r"""
const normalize = (value) => String(value || '').replace(/\s+/g, ' ').trim();
const isVisible = (el) => {
  if (!el) return false;
  let p = el;
  while (p) {
    const style = window.getComputedStyle(p);
    const cls = (p.className || '').toString();
    if (cls.includes('hide') || style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) return false;
    p = p.parentElement;
  }
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
};
return [...document.querySelectorAll('table')].filter(isVisible).map((table, tableIndex) => {
  let headers = [...table.querySelectorAll('thead th, thead td')].map((cell) => normalize(cell.innerText || cell.textContent));
  const tbodyRows = [...table.querySelectorAll('tbody tr')];
  let dataRows = tbodyRows;
  if (!headers.length && tbodyRows.length) {
    headers = [...tbodyRows[0].querySelectorAll('th,td')].map((cell) => normalize(cell.innerText || cell.textContent));
    dataRows = tbodyRows.slice(1);
  }
  if (!dataRows.length) {
    const allRows = [...table.querySelectorAll('tr')];
    if (!headers.length && allRows.length) {
      headers = [...allRows[0].querySelectorAll('th,td')].map((cell) => normalize(cell.innerText || cell.textContent));
      dataRows = allRows.slice(1);
    } else {
      dataRows = allRows;
    }
  }
  const rows = dataRows.map((row, rowIndex) => {
    const cells = [...row.querySelectorAll('td,th')].map((cell) => normalize(cell.innerText || cell.textContent));
    const record = {};
    headers.forEach((header, index) => {
      if (header) record[header] = cells[index] || '';
    });
    return {rowIndex, cells, record, text: normalize(row.innerText || row.textContent)};
  }).filter((row) => row.text);
  return {tableIndex, headers, rows, text: normalize(table.innerText || table.textContent)};
}).filter((table) => table.rows.length);
"""
        try:
            tables = tab.run_js(js) or []
        except Exception:
            return None

        file_marker = self._basename_stem(file_name)

        for table in tables:
            headers = table.get('headers') or []
            if '批次状态' not in headers:
                continue

            rows = table.get('rows') or []
            matched_rows = []
            if file_marker:
                matched_rows = [row for row in rows if file_marker in self._normalize_page_text((row.get('record') or {}).get('文件名') or row.get('text'))]
            target_rows = matched_rows or rows
            if not target_rows:
                continue

            row = target_rows[0]
            record = row.get('record') or {}
            status_text = self._normalize_page_text(record.get('批次状态') or row.get('text'))
            row_text = self._normalize_page_text(row.get('text'))
            if '处理失败' in status_text or '部分失败' in status_text or '退款失败' in status_text:
                kind = 'failure'
            elif '已处理' in status_text or '处理完成' in status_text or '已完成' in status_text:
                kind = 'success'
            elif '处理中' in status_text or '待处理' in status_text:
                kind = 'processing'
            else:
                kind = 'unknown'

            return {
                'status_kind': kind,
                'status_text': status_text,
                'row_text': row_text,
                'table_index': table.get('tableIndex'),
                'row_index': row.get('rowIndex'),
                'record': record,
            }

        return None

    def _goto_batch_refund_query_view(self, timeout=10):
        tab = self.tab
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                body_text = self._normalize_page_text(tab('tag:body').text)
            except Exception:
                body_text = ''
            if '批量退款批次查询' in body_text and self._extract_batch_refund_status():
                return True
            for locator in ['tag:a@@text():批量退款批次查询', 'tag:button@@text()=批量退款批次查询', 'tag:span@@text()=批量退款批次查询']:
                try:
                    for ele in tab.eles(locator):
                        if not self._is_element_really_visible(ele):
                            continue
                        ele.click(by_js=True)
                        tab.wait(1)
                        return True
                except Exception:
                    continue
            clicked_text = self._click_visible_text_action(tab, ['批量退款批次查询'])
            if clicked_text:
                tab.wait(1)
                return True
            time.sleep(1)
        return False

    def _refresh_batch_refund_query_view(self, result_page_url=None):
        tab = self.tab
        result_page_url = result_page_url or 'https://pay.weixin.qq.com/index.php/xphp/cbatchrefund/refund#/pages/refund_list/refund_list'

        # 只允许停留/回到批量退款结果页，不再在整页范围内盲点“查询”，否则容易误跳到别的结算查询模块。
        if '/refund#/pages/refund_list/refund_list' not in tab.url:
            tab.get(result_page_url)
            tab.wait(2)
            return True

        try:
            tab.refresh()
        except Exception:
            tab.get(result_page_url)
        tab.wait(2)
        return True

    def wait_batch_refund_completion(self, timeout=300, submit_started_at=None, file_name='',
                                     initial_popup_result=None, submit_soft_timeout=60,
                                     submit_confirmed=False):
        tab = self.tab
        result_page_url = 'https://pay.weixin.qq.com/index.php/xphp/cbatchrefund/refund#/pages/refund_list/refund_list'
        deadline = time.time() + timeout
        last_status_text = ''
        submit_observed_at = time.time()
        last_refresh_at = 0
        submitted = bool(submit_confirmed)

        if '/refund#/pages/refund_list/refund_list' not in tab.url:
            tab.get(result_page_url)
            tab.wait(2)

        while time.time() < deadline:
            try:
                body_text = tab('tag:body').text
            except Exception:
                body_text = ''
            normalized_body = self._normalize_page_text(body_text)
            try:
                self._raise_if_weipay_auth_invalid(normalized_body)
            except Exception:
                if submitted:
                    logger.warning(
                        f'批量退款提交后检测阶段遇到登录态/权限问题，按已提交继续后续流程：'
                        f'file={file_name!r}，url={tab.url}，title={tab.title}，状态摘要={normalized_body[:300]!r}'
                    )
                    return {'submitted': True, 'completed': False, 'status_text': normalized_body[:300], 'reason': 'post_submit_auth_invalid'}
                raise

            if '提交成功' in normalized_body or '退款申请已提交成功' in normalized_body:
                submitted = True
                self.尝试点击返款提交后的提示按钮(tab, timeout=2)
                if '/refund#/pages/refund_list/refund_list' not in tab.url:
                    tab.get(result_page_url)
                    tab.wait(2)
                continue

            batch_state = self._extract_batch_refund_status(submit_started_at=submit_started_at, file_name=file_name)

            if batch_state:
                submitted = True
                last_status_text = batch_state.get('row_text', '')[:300]
                if batch_state['status_kind'] == 'success':
                    logger.info(f'批量退款处理完成：{last_status_text}')
                    return {'submitted': True, 'completed': True, 'status_text': last_status_text, 'reason': 'batch_success'}
                if batch_state['status_kind'] == 'failure':
                    raise RuntimeError(f'微信支付批量退款批次处理失败，file={file_name!r}，url={tab.url}，title={tab.title}，状态摘要={last_status_text!r}')
                if batch_state['status_kind'] == 'processing':
                    if submit_soft_timeout and time.time() - submit_observed_at >= submit_soft_timeout:
                        logger.warning(
                            f'批量退款已提交且处理中超过{submit_soft_timeout}s，按软超时继续后续流程：'
                            f'file={file_name!r}，url={tab.url}，title={tab.title}，状态摘要={last_status_text!r}'
                        )
                        return {'submitted': True, 'completed': False, 'status_text': last_status_text, 'reason': 'processing_soft_timeout'}
                elif time.time() - submit_observed_at >= 10:
                    logger.warning(f'批量退款结果页未识别出明确批次状态，继续轮询：file={file_name!r}，状态摘要={last_status_text!r}')
            else:
                last_status_text = normalized_body[:300]
                if submitted and submit_soft_timeout and time.time() - submit_observed_at >= submit_soft_timeout:
                    logger.warning(
                        f'批量退款已提交，结果页超过{submit_soft_timeout}s仍未识别到批次状态，按软超时继续后续流程：'
                        f'file={file_name!r}，url={tab.url}，title={tab.title}，状态摘要={last_status_text!r}'
                    )
                    return {'submitted': True, 'completed': False, 'status_text': last_status_text, 'reason': 'unknown_soft_timeout'}

            if time.time() - last_refresh_at >= 5:
                self._refresh_batch_refund_query_view(result_page_url)
                last_refresh_at = time.time()
            time.sleep(1)
        if submitted:
            logger.warning(
                f'批量退款提交后检测超时，但该文件已确认提交，按保护策略继续后续流程：'
                f'file={file_name!r}，url={tab.url}，title={tab.title}，状态摘要={last_status_text!r}'
            )
            return {'submitted': True, 'completed': False, 'status_text': last_status_text, 'reason': 'post_submit_hard_timeout'}
        raise RuntimeError(f'微信支付批量退款结果未完成，file={file_name!r}，url={tab.url}，title={tab.title}，状态摘要={last_status_text!r}')

    def request_file_refund(self, file=None):
        if file is None:
            d = xlhome_dir('data/m2112kq5034/返款表')
            files = list(d.glob_files('*.csv'))
            files.sort(key=lambda f: f.mtime())
            file = files[-1]
        file = XlPath(file)
        marker = self._load_batch_refund_submit_marker(file)
        if marker:
            logger.warning(
                f'批量退款文件已存在提交标记，跳过再次提交：'
                f'file={str(file)!r}，stage={marker.get("stage")!r}，saved_at={marker.get("saved_at")!r}'
            )
            return {'submitted': True, 'completed': False, 'status_text': marker.get('status_text', ''), 'reason': 'submit_marker_exists', 'marker': marker}
        tab = self.tab
        tab.get('https://pay.weixin.qq.com/index.php/xphp/cbatchrefund/batch_refund#/pages/index/index')
        upload_button = self._find_visible_upload_action(tab, timeout=30)
        if upload_button is None:
            visible_actions = self._snapshot_visible_action_texts(tab)
            raise RuntimeError(
                '微信支付批量退款页未找到可见的文件选择入口'
                f'，url={tab.url} title={tab.title} visible_actions={visible_actions}'
            )
        uploaded = self._upload_via_file_input(tab, file)
        if not uploaded:
            upload_button.click.to_upload(file)
        tab.wait(2)
        file_input_states = self._count_file_inputs_with_files(tab)
        if not (self._has_uploaded_file(tab) or self._has_selected_upload_file(tab, file)):
            visible_actions = self._snapshot_visible_action_texts(tab)
            raise RuntimeError(
                '微信支付页面未接收到上传文件'
                f'，file={str(file)!r} visible_actions={visible_actions} file_inputs={file_input_states}'
            )
        confirm_button = None
        for button in tab.eles('tag:a@@text():确定', timeout=10):
            try:
                width, height = button.rect.size
            except Exception:
                continue
            if width > 0 and height > 0:
                confirm_button = button
                break
        if confirm_button is None:
            raise RuntimeError('微信支付批量退款页未找到可见的“确定”按钮')
        confirm_button.click()
        tab.wait(2)
        submit_started_at = pd.Timestamp.now()
        popup_confirmed = self.填写密码与验证码(tab, submit_file=file)
        if popup_confirmed:
            self._save_batch_refund_submit_marker(file, stage='submit_success_popup', status_text='提交成功')
        result = self.wait_batch_refund_completion(
            submit_started_at=submit_started_at,
            file_name=str(file),
            submit_confirmed=bool(popup_confirmed),
        )
        if result and result.get('submitted'):
            self._save_batch_refund_submit_marker(file, stage=result.get('reason', 'submitted'), status_text=result.get('status_text', ''))
        return result

