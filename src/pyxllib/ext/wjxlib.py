#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2024/11/19

"""问卷星相关工具。"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from pyxllib.prog.lazyimport import lazy_import

try:
    from DrissionPage import Chromium
except ModuleNotFoundError:
    Chromium = lazy_import("from DrissionPage import Chromium")

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import("pandas")

logger = logging.getLogger(__name__)

DEFAULT_COURSE_QUESTION_TITLE = "所属课程"
WJX_USERNAME_ENV = "WJX_USERNAME"
WJX_PASSWORD_ENV = "WJX_PASSWORD"


class WjxAutomationError(RuntimeError):
    """Raised when WJX automation cannot complete safely."""


def resolve_wjx_credentials(username: str | None = None, password: str | None = None) -> tuple[str, str]:
    """Resolve WJX credentials from explicit arguments or environment variables."""

    resolved_username = username or os.getenv(WJX_USERNAME_ENV)
    resolved_password = password or os.getenv(WJX_PASSWORD_ENV)
    if not resolved_username or not resolved_password:
        raise RuntimeError(f"未设置环境变量 {WJX_USERNAME_ENV} / {WJX_PASSWORD_ENV}，无法自动登录问卷星")
    return resolved_username, resolved_password


class WjxWeb:
    """问卷星网页的爬虫。"""

    HOME_URL = "https://www.wjx.cn/"
    LOGIN_URL = "https://www.wjx.cn/login.aspx"
    RESULT_LIMIT_URL = "https://www.wjx.cn/wjx/activitystat/resultlimit.aspx"

    def __init__(
        self,
        url: str = HOME_URL,
        *,
        login_username: str | None = None,
        password: str | None = None,
        auto_login: bool = True,
    ):
        self._login_username = login_username
        self._password = password
        self.browser = Chromium()
        self.browser.set.download_path(tempfile.gettempdir())
        parsed_url = urlparse(url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}" if parsed_url.scheme and parsed_url.netloc else None
        self.tab = self.browser.new_tab(url)
        if auto_login:
            self.login()
        if url and url != self.LOGIN_URL and self.tab.url != url:
            self.tab.get(url)

    def close_if_exceeds_min_tabs(self, min_tabs_to_keep: int = 1) -> None:
        try:
            if self.tab and self.base_url and len(self.browser.get_tabs(url=self.base_url)) > min_tabs_to_keep:
                self.tab.close()
        except Exception:
            pass

    def _ensure_credentials(
        self,
        username: str | None = None,
        password: str | None = None,
    ) -> tuple[str, str]:
        return resolve_wjx_credentials(username or self._login_username, password or self._password)

    def _wait_login_result(self, timeout: float = 60, interval: float = 1) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.tab.url.startswith(self.LOGIN_URL):
                return True
            time.sleep(interval)
        return False

    def login(
        self,
        wait_timeout: float = 60,
        *,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        tab = self.tab

        if tab.url.startswith(self.RESULT_LIMIT_URL):
            tab("t:a@@text():登录").click()
            time.sleep(2)

        if tab.url.startswith(self.LOGIN_URL):
            username, password = self._ensure_credentials(username, password)
            time.sleep(2)
            tab("t:input@@name=UserName").input(username, clear=True)
            time.sleep(2)
            tab("t:input@@name=Password").input(password, clear=True)
            time.sleep(2)
            tab("t:input@@type=submit").click()
            if not self._wait_login_result(timeout=wait_timeout):
                raise RuntimeError("问卷星登录后仍停留在登录页，可能需要人工完成验证码或额外验证")

    def get_page_num(self) -> tuple[int, int]:
        """返回当前页编号和总页数。"""

        idx, num = map(int, self.tab("tag:span@@class=paging-num").text.split("/"))
        return idx, num

    def prev_page(self) -> None:
        self.tab("tag:a@@class=go-pre").click()

    def next_page(self) -> None:
        self.tab("tag:a@@class=go-next").click()

    def _parse_table(self):
        """处理并解析网页中的表格数据。"""

        self.tab.find_ele_with_refresh("t:table")
        table_html = self.tab.eles("t:table")[-1].html
        df = pd.read_html(io.StringIO(table_html))[0]
        df.columns = [col.replace("\ue645", "") for col in df.columns]
        df.replace("\ue66b", "", regex=True, inplace=True)
        df.replace("\ue6a3\ue6d4", "", regex=True, inplace=True)
        return df

    def set_num_of_page(self, num_of_page: int) -> None:
        """查看数据页面，设置每页显示多少条记录。"""

        select = self.tab("tag:span@@text():每页显示").next("tag:select")
        select.click()
        opt = select(f"tag:option@@text()={num_of_page}")
        if opt.attr("selected") != "selected":
            opt.click()
        else:
            select.click()

    def get_df(self, all_pages: bool = False):
        """获得当前页面的表格数据。"""

        dfs = [self._parse_table()]
        if all_pages:
            current_idx, total_pages = self.get_page_num()
            while current_idx < total_pages:
                self.next_page()
                time.sleep(2)
                dfs.append(self._parse_table())
                current_idx, total_pages = self.get_page_num()

        return pd.concat(dfs, ignore_index=True) if all_pages else dfs[0]


@dataclass
class WjxSession:
    browser: Any
    tab: Any

    def close(self) -> None:
        try:
            self.tab.close()
        except Exception:
            logger.debug("Failed to close WJX tab")


def _normalize_unique(items: list[str] | None) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for item in items or []:
        value = (item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _get_text(ele: Any) -> str:
    return (getattr(ele, "value", None) or getattr(ele, "text", "") or "").strip()


def _wait_until(checker, *, timeout: float = 10, interval: float = 0.2, desc: str = "目标状态"):
    end_time = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < end_time:
        try:
            value = checker()
            if value:
                return value
        except Exception as exc:  # pragma: no cover - browser polling
            last_error = exc
        time.sleep(interval)
    if last_error is not None:
        raise TimeoutError(f"等待{desc}超时，最后错误：{last_error}") from last_error
    raise TimeoutError(f"等待{desc}超时")


def _wait_element(tab: Any, selectors: str | list[str], *, timeout: float = 10, desc: str = "元素"):
    selector_list = [selectors] if isinstance(selectors, str) else selectors

    def checker():
        for selector in selector_list:
            try:
                ele = tab.ele(selector, timeout=1)
            except Exception:
                continue
            if ele:
                return ele
        return None

    return _wait_until(checker, timeout=timeout, desc=desc)


def _wait_text_button(tab: Any, texts: list[str], *, timeout: float = 10, desc: str = "按钮"):
    selectors: list[str] = []
    for text in texts:
        selectors.extend(
            [
                f'xpath://a[contains(normalize-space(.), "{text}")]',
                f'xpath://button[contains(normalize-space(.), "{text}")]',
                f'xpath://span[contains(normalize-space(.), "{text}")]',
                f'xpath://input[contains(@value, "{text}")]',
            ]
        )
    return _wait_element(tab, selectors, timeout=timeout, desc=desc)


def _wait_url_contains(tab: Any, keyword: str, *, timeout: float = 15, desc: str = "页面跳转"):
    return _wait_until(lambda: keyword in tab.url and tab.url, timeout=timeout, desc=desc)


def _run_js_json(tab: Any, script: str) -> dict[str, Any] | list[Any] | None:
    result = tab.run_js(script)
    return json.loads(result) if result else None


def clean_wjx_popups(tab: Any, *, max_rounds: int = 3, wait_seconds: float = 0.4) -> int:
    script = r"""
const visible = (el) => {
  if (!el) return false;
  const style = window.getComputedStyle(el);
  const rect = el.getBoundingClientRect();
  return style.display !== 'none'
    && style.visibility !== 'hidden'
    && style.opacity !== '0'
    && rect.width >= 0
    && rect.height >= 0;
};

let count = 0;
const seen = new Set();
const selectors = [
  'i.closeAd',
  '.closeAd',
  '[class*="closeAd"]',
  '[class*="popup-close"]',
  '[class*="dialog-close"]',
  '[aria-label*="关闭"]',
  '[title*="关闭"]'
];

for (const selector of selectors) {
  for (const el of document.querySelectorAll(selector)) {
    if (seen.has(el) || !visible(el)) continue;
    seen.add(el);
    try {
      el.click();
      count += 1;
    } catch (e) {}
  }
}

for (const el of document.querySelectorAll('.wjx_adWrap')) {
  if (!visible(el)) continue;
  el.style.display = 'none';
  count += 1;
}

return count;
"""
    total = 0
    for _ in range(max_rounds):
        try:
            current = int(tab.run_js(script) or 0)
        except Exception:  # pragma: no cover - browser-specific
            logger.debug("Failed to clear WJX popups")
            break
        total += current
        if not current:
            break
        time.sleep(wait_seconds)
    return total


def _cleanup_extra_wjx_tabs(session: WjxSession) -> int:
    try:
        tabs = session.browser.get_tabs(url="wjx.cn")
    except Exception:
        return 0

    current_id = getattr(session.tab, "tab_id", None)
    closed = 0
    for tab in tabs:
        if current_id is not None and getattr(tab, "tab_id", None) == current_id:
            continue
        try:
            tab.close()
            closed += 1
        except Exception:
            logger.debug("Failed to close extra WJX tab")
    return closed


def create_wjx_session() -> WjxSession:
    browser = Chromium()
    tab = browser.new_tab()
    return WjxSession(browser=browser, tab=tab)


def _click_element(ele: Any) -> None:
    clickers = [
        lambda: ele.click.left(),
        lambda: ele.click(),
        lambda: ele.click(by_js=True),
    ]
    last_error: Exception | None = None
    for clicker in clickers:
        try:
            clicker()
            return
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise WjxAutomationError("元素点击失败")


def _click_login_submit(tab: Any) -> None:
    selectors = [
        'css:input[type="submit"]',
        "#LoginButton",
        'xpath://input[@type="submit" and contains(translate(@value, " ", ""), "登录")]',
        'xpath://button[contains(translate(normalize-space(.), " ", ""), "登录")]',
        'xpath://a[contains(@class, "submit") and contains(translate(normalize-space(.), " ", ""), "登录")]',
    ]
    for selector in selectors:
        try:
            button = tab.ele(selector, timeout=1)
        except Exception:
            continue
        if button:
            _click_element(button)
            return
    raise WjxAutomationError("未找到问卷星登录提交按钮")


def _fill_login_form_if_needed(session: WjxSession, username: str, password: str) -> None:
    tab = session.tab
    lower_url = (tab.url or "").lower()
    if "login" not in lower_url and not tab.ele('xpath://button[contains(normalize-space(.), "登录")]', timeout=1):
        return

    username_selectors = [
        'xpath://input[contains(@placeholder, "手机号")]',
        'xpath://input[contains(@placeholder, "账号")]',
        'xpath://input[contains(@placeholder, "用户名")]',
        'css:input[type="text"]',
    ]
    password_selectors = [
        'xpath://input[contains(@placeholder, "密码")]',
        'css:input[type="password"]',
    ]

    username_input = _wait_element(tab, username_selectors, timeout=8, desc="问卷星账号输入框")
    password_input = _wait_element(tab, password_selectors, timeout=8, desc="问卷星密码输入框")
    username_input.input(username, clear=True)
    password_input.input(password, clear=True)

    try:
        agree = tab.ele('xpath://label[contains(normalize-space(.), "同意") or contains(normalize-space(.), "协议")]', timeout=1)
        if agree:
            agree.click(by_js=True)
    except Exception:
        logger.debug("No WJX agreement checkbox found")

    _click_login_submit(tab)


def ensure_logged_in(
    session: WjxSession,
    *,
    username: str | None = None,
    password: str | None = None,
    target_url: str,
) -> None:
    username, password = resolve_wjx_credentials(username, password)

    tab = session.tab
    tab.get(target_url)
    try:
        tab.handle_alert(accept=True, timeout=1)
    except Exception:
        pass
    clean_wjx_popups(tab)
    _cleanup_extra_wjx_tabs(session)

    if "login" not in (tab.url or "").lower():
        return

    _fill_login_form_if_needed(session, username, password)

    def login_finished():
        current = (tab.url or "").lower()
        if "login" in current:
            return False
        return current

    try:
        _wait_until(login_finished, timeout=30, interval=0.5, desc="问卷星登录完成")
    except TimeoutError as exc:
        raise WjxAutomationError("问卷星登录未完成，可能需要额外验证或页面结构已变化") from exc

    tab.get(target_url)
    clean_wjx_popups(tab)


def _design_url(activity_id: str | int) -> str:
    return f"https://www.wjx.cn/wjx/design/designstart.aspx?activity={activity_id}"


def _run_button(tab: Any):
    return _wait_element(
        tab,
        [
            "#ctl02_ContentPlaceHolder1_btnRun",
            'xpath://input[contains(@value, "恢复运行") or contains(@value, "暂停接收答卷")]',
            'xpath://button[contains(normalize-space(.), "恢复运行") or contains(normalize-space(.), "暂停接收答卷")]',
            'xpath://a[contains(normalize-space(.), "恢复运行") or contains(normalize-space(.), "暂停接收答卷")]',
        ],
        timeout=5,
        desc="问卷运行按钮",
    )


def _run_state(tab: Any) -> str:
    text = _get_text(_run_button(tab))
    if text == "恢复运行":
        return "paused"
    if text == "暂停接收答卷":
        return "running"
    raise WjxAutomationError(f"无法识别问卷运行按钮文案：{text!r}")


def _confirm_popup(tab: Any) -> bool:
    selectors = [
        'tag:input@@value:确认',
        'tag:button@@text():确认',
        'tag:input@@value:确定',
        'tag:button@@text():确定',
        'tag:a@@text():确认',
        'tag:a@@text():确定',
        'tag:span@@text():确认',
        'tag:span@@text():确定',
    ]
    for selector in selectors:
        try:
            ele = tab(selector, timeout=2)
        except Exception:
            continue
        if ele:
            _click_element(ele)
            return True
    return False


def _confirm_resume_popup(tab: Any) -> bool:
    result = _run_js_json(
        tab,
        r"""
return JSON.stringify((() => {
    const modal = Array.from(document.querySelectorAll('div,section')).find(el => {
        const txt = (el.innerText || el.textContent || '').trim();
        return txt.includes('确认恢复运行吗');
    });
    if (!modal) return {clicked: false};
    const confirmBtn = Array.from(modal.querySelectorAll('input,button,a,span,div')).find(el => {
        return (el.innerText || el.textContent || el.value || '').trim() === '确定';
    });
    if (!confirmBtn) return {clicked: false, modal_text: (modal.innerText || modal.textContent || '').trim()};
    confirmBtn.click();
    return {
        clicked: true,
        modal_text: (modal.innerText || modal.textContent || '').trim(),
        confirm_text: (confirmBtn.innerText || confirmBtn.textContent || confirmBtn.value || '').trim()
    };
})())
""",
    )
    return bool(result and result.get("clicked"))


def _is_design_page(tab: Any, activity_id: str | int) -> bool:
    url = (tab.url or "").lower()
    return "designstart.aspx" in url and f"activity={activity_id}".lower() in url


def open_design_page(session: WjxSession, *, activity_id: str | int) -> None:
    target = _design_url(activity_id)
    if not _is_design_page(session.tab, activity_id):
        try:
            session.tab.handle_alert(accept=True, next_one=True)
        except Exception:
            pass
        session.tab.get(target)
        try:
            session.tab.handle_alert(accept=True, timeout=1)
        except Exception:
            pass
    clean_wjx_popups(session.tab)
    _cleanup_extra_wjx_tabs(session)


def pause_responses(session: WjxSession, *, activity_id: str | int) -> None:
    open_design_page(session, activity_id=activity_id)
    tab = session.tab
    before_text = _get_text(_run_button(tab))
    if before_text == "恢复运行":
        return
    if before_text != "暂停接收答卷":
        raise WjxAutomationError(f"未找到预期的暂停按钮，当前文案：{before_text!r}")
    _run_button(tab).click(by_js=True)
    time.sleep(0.3)
    _confirm_popup(tab)
    _wait_until(
        lambda: _get_text(_run_button(tab)) == "恢复运行" and True,
        timeout=10,
        desc="暂停后恢复运行按钮出现",
    )


def resume_responses(session: WjxSession, *, activity_id: str | int) -> None:
    open_design_page(session, activity_id=activity_id)
    tab = session.tab
    before_text = _get_text(_run_button(tab))
    if before_text == "暂停接收答卷":
        return
    if before_text != "恢复运行":
        raise WjxAutomationError(f"未找到预期的恢复运行按钮，当前文案：{before_text!r}")

    _click_element(_run_button(tab))

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        current = _get_text(_run_button(tab))
        if current == "暂停接收答卷":
            return
        _confirm_resume_popup(tab)
        _confirm_popup(tab)
        time.sleep(0.4)

    _wait_until(
        lambda: _get_text(_run_button(tab)) == "暂停接收答卷" and True,
        timeout=10,
        desc="恢复后暂停按钮出现",
    )


def _wait_edit_ready(tab: Any, *, question_title: str) -> None:
    def checker():
        info = _run_js_json(
            tab,
            f"""
return JSON.stringify((() => {{
    const holders = window.questionHolder || [];
    const title = {json.dumps(question_title, ensure_ascii=False)};
    for (let i = 0; i < holders.length; i++) {{
        const dn = holders[i] && holders[i].dataNode;
        if (dn && dn._title === title) {{
            return {{
                ready: true,
                topic: dn._topic || i + 1,
                title: dn._title,
                select_len: (dn._select || []).length
            }};
        }}
    }}
    return {{ready: false}};
}})())
""",
        )
        return info if info and info.get("ready") else False

    _wait_until(checker, timeout=20, desc="问卷编辑页就绪")
    clean_wjx_popups(tab)


def open_edit_page(
    session: WjxSession,
    *,
    activity_id: str | int,
    question_title: str = DEFAULT_COURSE_QUESTION_TITLE,
) -> None:
    tab = session.tab
    if "designnew.aspx" in tab.url and f"curid={activity_id}" in tab.url:
        _wait_edit_ready(tab, question_title=question_title)
        return

    if not ("editquestionnaire.aspx" in tab.url and f"activity={activity_id}" in tab.url):
        open_design_page(session, activity_id=activity_id)
        edit_selectors = [
            "css:#ctl02_ContentPlaceHolder1_hrefEdit",
            f'xpath://a[contains(@href, "/newwjx/design/editquestionnaire.aspx?activity={activity_id}")]',
            'xpath://a[contains(normalize-space(.), "编辑问卷")]',
        ]
        _click_element(_wait_element(tab, edit_selectors, timeout=10, desc="编辑问卷入口"))
        _wait_until(
            lambda: ("editquestionnaire.aspx" in tab.url or "designnew.aspx" in tab.url) and tab.url,
            timeout=15,
            desc="进入问卷编辑模式",
        )

    if "editquestionnaire.aspx" in tab.url:
        _click_element(_wait_text_button(tab, ["下一步"], timeout=10, desc="修改模式下一步"))
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and "designnew.aspx" not in (tab.url or ""):
            _confirm_popup(tab)
            time.sleep(0.5)

    _wait_url_contains(tab, "designnew.aspx", timeout=20, desc="正式编辑页")
    _wait_edit_ready(tab, question_title=question_title)


def _get_question_info(tab: Any, *, question_title: str) -> dict[str, Any]:
    info = _run_js_json(
        tab,
        f"""
return JSON.stringify((() => {{
    const holders = window.questionHolder || [];
    const title = {json.dumps(question_title, ensure_ascii=False)};
    for (let i = 0; i < holders.length; i++) {{
        const dn = holders[i] && holders[i].dataNode;
        if (dn && dn._title === title) {{
            return {{
                topic: dn._topic || i + 1,
                title: dn._title,
                type: dn._type,
                select_len: (dn._select || []).length
            }};
        }}
    }}
    return null;
}})())
""",
    )
    if not info:
        raise WjxAutomationError(f"未找到题目“{question_title}”")
    info["topic"] = int(info["topic"])
    info["select_len"] = int(info.get("select_len") or 0)
    return info


def enter_course_question_edit_mode(
    session: WjxSession,
    *,
    question_title: str = DEFAULT_COURSE_QUESTION_TITLE,
) -> None:
    tab = session.tab
    info = _get_question_info(tab, question_title=question_title)
    result = _run_js_json(
        tab,
        f"""
return JSON.stringify((() => {{
    const textOf = el => ((el && (el.innerText || el.textContent)) || '').trim();
    const title = {json.dumps(question_title, ensure_ascii=False)};
    const q = typeof getDivByTopic === 'function' ? getDivByTopic({info["topic"]}) : null;
    if (!q) return {{ok: false, error: 'question_not_found'}};
    q.scrollIntoView({{block: 'center'}});
    q.click();
    let edit = [...q.querySelectorAll('span, a, button')].find(el => textOf(el) === '编辑');
    if (!edit) {{
        edit = [...document.querySelectorAll('.div_question.qactive span, .div_question.qactive a, .div_question.qactive button')]
            .find(el => textOf(el) === '编辑');
    }}
    const alreadyEditing = !!(window.cur && window.cur.dataNode && window.cur.dataNode._title === title
        && document.querySelectorAll('.item_title').length > 0);
    if (edit) edit.click();
    return {{ok: !!edit || alreadyEditing, alreadyEditing}};
}})())
""",
    )
    if not result or not result.get("ok"):
        raise WjxAutomationError(f"进入“{question_title}”编辑态失败：{result}")
    _wait_until(
        lambda: len(read_course_options(session, question_title=question_title)["all_items"]) > 0,
        timeout=10,
        desc="课程选项加载",
    )


def read_course_options(
    session: WjxSession,
    *,
    question_title: str = DEFAULT_COURSE_QUESTION_TITLE,
) -> dict[str, Any]:
    info = _get_question_info(session.tab, question_title=question_title)
    data = _run_js_json(
        session.tab,
        f"""
return JSON.stringify((() => {{
    const dn = typeof getDataNodeByTopic === 'function'
        ? getDataNodeByTopic({info["topic"]})
        : (window.questionHolder && window.questionHolder[{info["topic"] - 1}] ? window.questionHolder[{info["topic"] - 1}].dataNode : null);
    if (!dn) return null;
    const items = [];
    const selects = dn._select || [];
    for (let i = 1; i < selects.length; i++) {{
        const item = selects[i];
        if (!item) continue;
        const name = (item._item_title || '').trim();
        if (!name) continue;
        items.push({{
            order: i,
            name: name,
            hidden: item._item_relation === '-1',
            relation_value: item._item_relation || '',
            option_value: item._item_value || ''
        }});
    }}
    return {{
        topic: dn._topic || {info["topic"]},
        all_items: items,
        visible_names: items.filter(item => !item.hidden).map(item => item.name)
    }};
}})())
""",
    )
    if not data:
        raise WjxAutomationError(f"读取“{question_title}”选项失败")
    return data


def _validate_additions(add_names: list[str], existing_names: set[str]) -> None:
    if not add_names:
        return
    duplicated_in_input: list[str] = []
    seen: set[str] = set()
    for item in add_names:
        if item in seen and item not in duplicated_in_input:
            duplicated_in_input.append(item)
        seen.add(item)
    if duplicated_in_input:
        raise WjxAutomationError(f"新增课程列表里有重复项：{duplicated_in_input}")
    duplicated_existing = [item for item in add_names if item in existing_names]
    if duplicated_existing:
        raise WjxAutomationError(f"新增课程与已有选项重名，不能重复新增：{duplicated_existing}")


def _apply_course_hides(
    session: WjxSession,
    *,
    hide_names: list[str] | None,
    question_title: str = DEFAULT_COURSE_QUESTION_TITLE,
) -> dict[str, Any]:
    hide_names = _normalize_unique(hide_names)
    payload = json.dumps({"hide": hide_names, "title": question_title}, ensure_ascii=False)
    result = _run_js_json(
        session.tab,
        f"""
return JSON.stringify((() => {{
    const payload = {payload};
    const title = payload.title;
    const holders = window.questionHolder || [];
    let dn = null;
    for (let i = 0; i < holders.length; i++) {{
        const node = holders[i] && holders[i].dataNode;
        if (node && node._title === title) {{
            dn = node;
            break;
        }}
    }}
    if (!dn) throw new Error('未找到所属课程题目');
    if (!window.cur) throw new Error('当前未激活所属课程编辑态');

    const result = {{
        hidden_applied: [],
        hidden_skipped: [],
        hidden_missing: [],
        added_applied: []
    }};
    const indexByName = {{}};
    const selects = dn._select || [];
    for (let i = 1; i < selects.length; i++) {{
        const item = selects[i];
        if (item && item._item_title) indexByName[item._item_title.trim()] = i;
    }}

    for (const name of payload.hide || []) {{
        const idx = indexByName[name];
        if (!idx) {{
            result.hidden_missing.push(name);
            continue;
        }}
        const item = selects[idx];
        if (item._item_relation === '-1') {{
            result.hidden_skipped.push(name);
            continue;
        }}
        if (item._item_relation && item._item_relation !== '-1') {{
            throw new Error(`课程“${{name}}”已设置选项关联，不能直接自动隐藏`);
        }}
        item._item_relation = '-1';
        result.hidden_applied.push(name);
    }}

    if (window.cur.updateItem) window.cur.updateItem();
    if (window.cur.updateReferQ) window.cur.updateReferQ();
    if (window.cur.setRandomText) window.cur.setRandomText();
    if (window.cur.checkItemTitle && !window.cur.checkItemTitle()) {{
        throw new Error('所属课程选项校验未通过');
    }}

    return result;
}})())
""",
    )
    if not result:
        raise WjxAutomationError("批量隐藏所属课程失败")
    return result


def _add_course_option(
    session: WjxSession,
    *,
    name: str,
    question_title: str = DEFAULT_COURSE_QUESTION_TITLE,
) -> dict[str, Any]:
    payload = json.dumps({"name": name, "title": question_title}, ensure_ascii=False)
    prepared = _run_js_json(
        session.tab,
        f"""
return JSON.stringify((() => {{
    const payload = {payload};
    const title = payload.title;
    const holders = window.questionHolder || [];
    let dn = null;
    for (let i = 0; i < holders.length; i++) {{
        const node = holders[i] && holders[i].dataNode;
        if (node && node._title === title) {{
            dn = node;
            break;
        }}
    }}
    if (!dn) throw new Error('未找到所属课程题目');
    if (!window.cur || !window.cur.dataNode || window.cur.dataNode._title !== title) {{
        throw new Error('当前未激活所属课程编辑态');
    }}

    const cur = window.cur;
    const beforeOptionLength = cur.option_radio ? cur.option_radio.length : 0;
    const beforeSelectLength = dn._select ? dn._select.length : 0;
    const oldConfirm = window.confirm;
    window.confirm = () => true;
    try {{
        cur.addNewItem();
    }} finally {{
        window.confirm = oldConfirm;
    }}

    const afterOptionLength = cur.option_radio ? cur.option_radio.length : 0;
    const afterSelectLength = dn._select ? dn._select.length : 0;
    if (afterOptionLength <= beforeOptionLength || afterSelectLength <= beforeSelectLength) {{
        throw new Error(`新增课程“${{payload.name}}”后，问卷星未生成新的选项行`);
    }}

    const idx = afterOptionLength - 1;
    const optionRow = cur.option_radio[idx];
    const titleEl = optionRow && optionRow.get_item_title ? optionRow.get_item_title() : null;
    if (!titleEl) {{
        throw new Error(`新增课程“${{payload.name}}”后，未找到新增选项的标题控件`);
    }}

    document.querySelectorAll('[data-codeyun-new-option="1"]').forEach(el => el.removeAttribute('data-codeyun-new-option'));
    titleEl.setAttribute('data-codeyun-new-option', '1');
    titleEl.scrollIntoView({{ block: 'center' }});
    titleEl.click();
    titleEl.focus();

    return {{
        idx: idx,
        default_title: ((titleEl.innerText || titleEl.textContent || titleEl.value || '')).trim(),
        option_value: dn._select[idx] ? (dn._select[idx]._item_value || '') : ''
    }};
}})())
""",
    )
    if not prepared:
        raise WjxAutomationError(f"新增课程“{name}”前置准备失败")

    title_ele = _wait_element(session.tab, 'css:[data-codeyun-new-option="1"]', timeout=5, desc=f'新增课程“{name}”标题框')
    title_ele.input(name, clear=True)
    title_ele.run_js("this.blur();")

    verified = _wait_until(
        lambda: _run_js_json(
            session.tab,
            f"""
return JSON.stringify((() => {{
    const title = {json.dumps(question_title, ensure_ascii=False)};
    const name = {json.dumps(name, ensure_ascii=False)};
    const holders = window.questionHolder || [];
    let dn = null;
    for (let i = 0; i < holders.length; i++) {{
        const node = holders[i] && holders[i].dataNode;
        if (node && node._title === title) {{
            dn = node;
            break;
        }}
    }}
    if (!dn || !window.cur) return null;
    const idx = window.cur.option_radio ? window.cur.option_radio.length - 1 : -1;
    if (idx < 1) return null;
    const optionRow = window.cur.option_radio[idx];
    const titleEl = optionRow && optionRow.get_item_title ? optionRow.get_item_title() : null;
    const dataTitle = ((dn._select && dn._select[idx] && dn._select[idx]._item_title) || '').trim();
    const domText = ((titleEl && (titleEl.innerText || titleEl.textContent)) || '').trim();
    const domValue = ((titleEl && titleEl.value) || '').trim();
    if (dataTitle != name || domText != name) return false;
    return {{
        idx: idx,
        data_title: dataTitle,
        dom_text: domText,
        dom_value: domValue,
        option_value: dn._select && dn._select[idx] ? (dn._select[idx]._item_value || '') : ''
    }};
}})())
""",
        ),
        timeout=5,
        interval=0.2,
        desc=f'新增课程“{name}”命名同步',
    )
    if not verified:
        raise WjxAutomationError(f"新增课程“{name}”命名后未能同步到问卷数据")

    session.tab.run_js(
        """
document.querySelectorAll('[data-codeyun-new-option="1"]').forEach(el => el.removeAttribute('data-codeyun-new-option'));
"""
    )
    return verified


def apply_course_changes(
    session: WjxSession,
    *,
    hide_names: list[str] | None,
    add_names: list[str] | None,
    question_title: str = DEFAULT_COURSE_QUESTION_TITLE,
) -> dict[str, Any]:
    hide_names = _normalize_unique(hide_names)
    add_names = _normalize_unique(add_names)
    result = _apply_course_hides(session, hide_names=hide_names, question_title=question_title)
    for name in add_names:
        _add_course_option(session, name=name, question_title=question_title)
        result["added_applied"].append(name)

    final_check = _run_js_json(
        session.tab,
        """
return JSON.stringify((() => {
    if (!window.cur) return { valid: false, error: 'no_cur' };
    if (window.cur.updateItem) window.cur.updateItem();
    if (window.cur.updateReferQ) window.cur.updateReferQ();
    if (window.cur.setRandomText) window.cur.setRandomText();
    if (window.cur.checkItemTitle && !window.cur.checkItemTitle()) {
        return { valid: false, error: 'checkItemTitle_failed' };
    }
    return { valid: true };
})())
""",
    )
    if not final_check or not final_check.get("valid"):
        raise WjxAutomationError(f"所属课程选项校验未通过：{final_check}")
    return result


def finish_editing(session: WjxSession, *, activity_id: str | int) -> None:
    _wait_element(session.tab, "css:#hrefFiQ", timeout=10, desc="完成编辑按钮").click()
    _wait_until(
        lambda: "designstart.aspx" in session.tab.url and f"activity={activity_id}" in session.tab.url and True,
        timeout=30,
        desc="返回设计向导页",
    )
    clean_wjx_popups(session.tab)


def inspect_template(
    *,
    login_username: str | None = None,
    password: str | None = None,
    activity_id: str | int,
    question_title: str = DEFAULT_COURSE_QUESTION_TITLE,
) -> dict[str, Any]:
    login_username, password = resolve_wjx_credentials(login_username, password)
    session = create_wjx_session()
    try:
        ensure_logged_in(session, username=login_username, password=password, target_url=_design_url(activity_id))
        running_before = _run_state(session.tab) == "running"
        open_edit_page(session, activity_id=activity_id, question_title=question_title)
        data = read_course_options(session, question_title=question_title)
        return {
            "action": "inspect",
            "activity_id": str(activity_id),
            "question_title": question_title,
            "was_running": running_before,
            "run_state_unchanged": True,
            "all_items": data["all_items"],
            "visible_names": data["visible_names"],
        }
    finally:
        session.close()


def apply_template_changes(
    *,
    login_username: str | None = None,
    password: str | None = None,
    activity_id: str | int,
    hide_names: list[str] | None,
    add_names: list[str] | None,
    question_title: str = DEFAULT_COURSE_QUESTION_TITLE,
) -> dict[str, Any]:
    login_username, password = resolve_wjx_credentials(login_username, password)
    session = create_wjx_session()
    hide_names = _normalize_unique(hide_names)
    add_names = _normalize_unique(add_names)
    overlapping = sorted(set(hide_names) & set(add_names))
    if overlapping:
        raise WjxAutomationError(f"同一课程不能同时出现在隐藏和新增中：{overlapping}")

    try:
        ensure_logged_in(session, username=login_username, password=password, target_url=_design_url(activity_id))
        running_before = _run_state(session.tab) == "running"
        open_edit_page(session, activity_id=activity_id, question_title=question_title)
        enter_course_question_edit_mode(session, question_title=question_title)
        before = read_course_options(session, question_title=question_title)
        existing_names = {item["name"] for item in before["all_items"]}
        _validate_additions(add_names, existing_names)

        changed = apply_course_changes(
            session,
            hide_names=hide_names,
            add_names=add_names,
            question_title=question_title,
        )
        after = read_course_options(session, question_title=question_title)
        finish_editing(session, activity_id=activity_id)
        resumed = False
        if _run_state(session.tab) == "paused":
            resume_responses(session, activity_id=activity_id)
            resumed = True

        return {
            "action": "apply",
            "activity_id": str(activity_id),
            "question_title": question_title,
            "was_running": running_before,
            "resumed": resumed,
            "run_state_unchanged": False,
            "hidden_applied": changed["hidden_applied"],
            "hidden_skipped": changed["hidden_skipped"],
            "hidden_missing": changed["hidden_missing"],
            "added_applied": changed["added_applied"],
            "before": before,
            "after": after,
        }
    finally:
        session.close()


def execute_wjx_template_action(
    *,
    login_username: str | None = None,
    password: str | None = None,
    activity_id: str | int,
    action: str,
    hide_names: list[str] | None = None,
    add_names: list[str] | None = None,
    question_title: str = DEFAULT_COURSE_QUESTION_TITLE,
) -> dict[str, Any]:
    normalized_action = (action or "").strip().lower()
    if normalized_action == "inspect":
        return inspect_template(
            login_username=login_username,
            password=password,
            activity_id=activity_id,
            question_title=question_title,
        )
    if normalized_action == "apply":
        return apply_template_changes(
            login_username=login_username,
            password=password,
            activity_id=activity_id,
            hide_names=hide_names,
            add_names=add_names,
            question_title=question_title,
        )
    raise WjxAutomationError(f"不支持的问卷星动作：{action}")


__all__ = [
    "DEFAULT_COURSE_QUESTION_TITLE",
    "WjxAutomationError",
    "WjxSession",
    "WjxWeb",
    "apply_course_changes",
    "apply_template_changes",
    "clean_wjx_popups",
    "create_wjx_session",
    "ensure_logged_in",
    "enter_course_question_edit_mode",
    "execute_wjx_template_action",
    "finish_editing",
    "inspect_template",
    "open_design_page",
    "open_edit_page",
    "pause_responses",
    "read_course_options",
    "resolve_wjx_credentials",
    "resume_responses",
]
