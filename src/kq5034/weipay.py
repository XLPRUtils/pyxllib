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
            tab.get('https://pay.weixin.qq.com')
            message_sent = False
            while tab.url != 'https://pay.weixin.qq.com/index.php/core/info':
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
                        KqWechat.扫码登录微信支付(users[0])
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

    def _fill_visible_inputs(self, tab, values, *, minimum_count=None):
        values = [str(v) for v in values]
        minimum_count = minimum_count or len(values)
        js = r"""
const values = JSON.parse(arguments[0] || '[]');
const minimumCount = arguments[1] || values.length;
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
return 'OK';
"""
        result = tab.run_js(js, json.dumps(values, ensure_ascii=False), minimum_count)
        if result != 'OK':
            raise RuntimeError(f'页面输入框填写失败：{result}')

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
  return text.includes('????') || text.includes('?????????');
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

    def _click_visible_text_action_js(self, tab, texts):
        if not texts:
            return ''

        js = r"""
const targets = arguments[0] || [];
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

    def _raise_if_weipay_auth_invalid(self, body_text):
        text = self._normalize_page_text(body_text)
        if any(x in text for x in ['请使用微信扫码登录', '二维码失效', '微信扫一扫登录', '请扫码登录', '登录超时，请重新登录']):
            raise RuntimeError('微信支付登录态已失效，请重新扫码登录后再执行批量退款')
        if '暂时无该功能权限' in text and '请联系本商户员工管理员' in text:
            raise RuntimeError('微信支付当前登录态缺少访问权限，请重新扫码或完成安全验证后再试')
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

    def 填写密码与验证码(self, tab, submit_file=None):
        inputs = tab.eles('tag:input@@class=real-input')
        passwd = XlEnv.get(f'XL_KQ_PAY_PASSWORD_{self.user}', decoding=True) or XlEnv.get('XL_KQ_PAY_PASSWORD', decoding=True)
        if passwd:
            inputs[0].input(passwd, clear=True)
        if len(inputs) > 1:
            tab('tag:a@@text():发送短信').click()
            time.sleep(10)
            with get_autogui_lock():
                vcode = KqWechat.从懒人转发获得短信内容()
            inputs[1].input(vcode, clear=True)
        time.sleep(1)
        tab('tag:a@@text()=确定@@class=btn btn-primary align-center').click()
        if submit_file is not None:
            self._save_batch_refund_submit_marker(
                submit_file,
                stage='submit_clicked',
                status_text='已点击提交，等待后续确认',
            )
        return self.尝试点击返款提交后的提示按钮(tab)

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
        tab.wait(2)
        tab('tag:a@@title=上传文件').click.to_upload(file)
        tab.wait(2)
        tab('tag:a@@text():确定@@class=btn btn-primary@@href=javascript:void(0);').click()
        tab.wait(2)
        submit_started_at = pd.Timestamp.now()
        popup_confirmed = self.填写密码与验证码(tab, submit_file=file)
        if popup_confirmed:
            self._save_batch_refund_submit_marker(file, stage='submit_success_popup', status_text='提交成功')
        result = self.wait_batch_refund_completion(
            submit_started_at=submit_started_at,
            file_name=str(file),
            submit_confirmed=True,
        )
        if result and result.get('submitted'):
            self._save_batch_refund_submit_marker(file, stage=result.get('reason', 'submitted'), status_text=result.get('status_text', ''))
        return result

