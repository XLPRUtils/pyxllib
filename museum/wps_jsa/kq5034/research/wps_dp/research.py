#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""用 DrissionPage 研究 WPS/KDocs 网页版页面结构与菜单操作。"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from DrissionPage import Chromium
from DrissionPage.common import Keys

try:
    from .artifacts import (
        DEFAULT_ARTIFACT_ROOT,
        DEFAULT_RUNS_ROOT,
        build_requested_actions,
        resolve_save_dir as resolve_artifact_dir,
    )
except ImportError:  # pragma: no cover - 兼容直接运行 research.py
    from artifacts import (  # type: ignore
        DEFAULT_ARTIFACT_ROOT,
        DEFAULT_RUNS_ROOT,
        build_requested_actions,
        resolve_save_dir as resolve_artifact_dir,
    )


DEFAULT_URL = "https://www.kdocs.cn/l/condDXMoWDSW"
FILE_MENU_PANEL = "css:.file-more-panel"
SAVE_SUBMENU_PANEL = "css:.header-more-block-submenu"
COPY_SUCCESS_TITLE = "text:创建副本成功"
OPEN_NOW_BUTTON = "text:立即打开"
TITLE_INFO_BOX = "css:.component-header-file-info"
TITLE_EDIT_INPUT = 'css:.component-header-file-info input[maxlength="240"]'
SHARE_PANEL = "css:.collaboration-share-panel"
SHARE_BUTTON = "css:.cooperation-open-btn button"
SHARE_BOX_AREA = "css:.share-box-area"
SHARE_SWITCH = "css:.kds-switch"
SHARE_PERMISSION_DESC = "css:.kds-permission-desc"
SHARE_PERMISSION_BUTTON = "css:.kds-dropdown-button"
SHARE_PERMISSION_POPOVER = "css:.kds-popover"
SHARE_PERMISSION_LIST = "css:.kds-link-perm__list"
SHARE_PERMISSION_ITEM = "css:.kds-link-perm__list-item"

SHARE_SCOPE_LABELS = {
    "all": "所有人",
    "specified": "指定人",
}
SHARE_PERMISSION_LABELS = {
    "edit": "编辑",
    "comment": "查看和评论",
    "view": "查看",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="打开一份 WPS/KDocs 文档，并输出基础探测信息。"
    )
    parser.add_argument("url", nargs="?", default=DEFAULT_URL, help="目标文档 URL")
    parser.add_argument("--port", type=int, default=9222, help="浏览器调试端口")
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=8,
        help="打开页面后额外等待几秒，方便页面脚本和登录态稳定",
    )
    parser.add_argument(
        "--save-dir",
        help="显式指定保存目录；不传时自动写入 --save-root 下的时间戳运行目录",
    )
    parser.add_argument(
        "--save-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="未传 --save-dir 时，运行结果根目录",
    )
    parser.add_argument(
        "--label",
        help="追加到自动生成目录名中的简短标签",
    )
    parser.add_argument(
        "--create-copy",
        action="store_true",
        help="执行 文件操作 -> 另存 -> 创建副本",
    )
    parser.add_argument(
        "--open-now",
        action="store_true",
        help="点击创建副本成功弹窗里的“立即打开”按钮",
    )
    parser.add_argument(
        "--rename-title",
        help="将当前文档标题重命名为指定字符串",
    )
    parser.add_argument(
        "--share-enabled",
        choices=["on", "off"],
        help="配置分享总开关，on 表示开启，off 表示关闭",
    )
    parser.add_argument(
        "--share-scope",
        choices=["all", "specified"],
        help="链接权限范围，all=所有人，specified=指定人",
    )
    parser.add_argument(
        "--share-permission",
        choices=["edit", "comment", "view"],
        help="链接权限角色，edit/comment/view",
    )
    return parser


def connect_browser(port: int = 9222) -> Chromium:
    return Chromium(port)


def is_displayed(ele) -> bool:
    return bool(ele and ele.states.is_displayed)


def find_visible(tab, locator: str, timeout: float = 0):
    end_time = time.time() + timeout
    while True:
        for ele in tab.eles(locator):
            if is_displayed(ele):
                return ele
        if time.time() >= end_time:
            return None
        time.sleep(0.2)


def wait_until(
    predicate,
    timeout: float = 10,
    interval: float = 0.2,
    err_msg: str = "等待超时",
):
    end_time = time.time() + timeout
    while time.time() < end_time:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise TimeoutError(err_msg)


def click_with_retry(ele, name: str) -> None:
    try:
        ele.click()
        return
    except Exception:
        pass

    try:
        ele.click(by_js=True)
        return
    except Exception as exc:  # pragma: no cover - 调试时保留具体报错
        raise RuntimeError(f"点击失败：{name}") from exc


def get_kdocs_tabs(browser: Chromium) -> list:
    tabs = []
    for tab in browser.get_tabs():
        try:
            if "kdocs.cn" in (tab.url or ""):
                tabs.append(tab)
        except Exception:
            continue
    return tabs


def describe_kdocs_tabs(browser: Chromium) -> list[dict]:
    data = []
    for tab in get_kdocs_tabs(browser):
        data.append(
            {
                "tab_id": getattr(tab, "tab_id", ""),
                "title": tab.title,
                "url": tab.url,
                "has_open_now": bool(find_visible(tab, OPEN_NOW_BUTTON, timeout=0.2)),
                "has_copy_success": bool(find_visible(tab, COPY_SUCCESS_TITLE, timeout=0.2)),
            }
        )
    return data


def pick_kdocs_tab(
    browser: Chromium,
    *,
    url_contains: str = "kdocs.cn",
    title_contains: str | None = None,
    prefer_locator: str | None = None,
):
    candidates = []
    for tab in browser.get_tabs():
        try:
            url = tab.url or ""
            title = tab.title or ""
        except Exception:
            continue

        if url_contains and url_contains not in url:
            continue
        if title_contains and title_contains not in title:
            continue
        candidates.append(tab)

    if not candidates:
        raise RuntimeError("未找到符合条件的 KDocs 标签页")

    if prefer_locator:
        for tab in candidates:
            if find_visible(tab, prefer_locator, timeout=0.5):
                return tab

    return candidates[0]


def pick_kdocs_tab_by_url(
    browser: Chromium,
    url: str,
    *,
    prefer_locator: str | None = None,
):
    token = extract_doc_token(url)
    candidates = []
    for tab in browser.get_tabs():
        try:
            if token in (tab.url or ""):
                candidates.append(tab)
        except Exception:
            continue

    if not candidates:
        raise RuntimeError("未找到符合 url 的 KDocs 标签页")

    if prefer_locator:
        for tab in candidates:
            if find_visible(tab, prefer_locator, timeout=0.5):
                return tab

    # 同一文档可能被开了多个标签页，优先返回未停在标题编辑态的那个。
    for tab in candidates:
        if not get_visible_title_input(tab):
            return tab

    return candidates[0]


def open_target_tab(browser: Chromium, url: str, wait_seconds: float):
    tab = browser.new_tab()
    print(f"opening url: {url}", flush=True)
    tab.get(url)
    print("load started", flush=True)
    tab.wait.load_start()
    print(f"sleep {wait_seconds}s for page stabilization", flush=True)
    time.sleep(wait_seconds)
    return tab


def collect_probe(tab) -> dict:
    visible_text = tab.run_js(
        "return document.body ? document.body.innerText.slice(0, 2000) : '';"
    )
    page_html = tab.html or ""
    title = tab.title
    iframe_info = tab.run_js(
        """
        return Array.from(document.querySelectorAll('iframe')).map((frame, index) => ({
            index,
            id: frame.id || '',
            name: frame.name || '',
            src: frame.src || ''
        }));
        """
    )
    login_texts = []
    for text in ("登录", "微信登录", "手机号登录", "打开App", "保存到我的云文档"):
        if find_visible(tab, f"text:{text}"):
            login_texts.append(text)

    return {
        "title": title,
        "url": tab.url,
        "html_length": len(page_html),
        "visible_text_preview": visible_text,
        "iframe_count": len(iframe_info or []),
        "iframes": iframe_info or [],
        "possible_login_hints": login_texts,
    }


def get_title_box(tab):
    box = tab.ele(TITLE_INFO_BOX, timeout=3)
    if not box:
        raise RuntimeError("未找到标题容器")
    return box


def get_visible_title_input(tab):
    for ele in tab.eles(TITLE_EDIT_INPUT):
        if is_displayed(ele):
            return ele
    return None


def get_title_text(tab) -> str:
    input_ele = get_visible_title_input(tab)
    if input_ele:
        return input_ele.attr("value") or ""

    box = get_title_box(tab)
    return (box.text or "").strip()


def inspect_title_area(tab) -> dict:
    box = get_title_box(tab)
    title_input = get_visible_title_input(tab)
    info = {
        "title_text": get_title_text(tab),
        "tab_title": tab.title,
        "url": tab.url,
        "container_class": box.attr("class"),
        "container_rect": {
            "x": box.rect.location[0],
            "y": box.rect.location[1],
            "width": box.rect.size[0],
            "height": box.rect.size[1],
        },
        "mode": "edit" if title_input else "display",
        "input_value": title_input.attr("value") if title_input else None,
    }
    return info


def save_json(data: dict, path: Path) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_probe_files(tab, probe: dict, save_dir: Path) -> tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = save_dir / "page.png"
    probe_path = save_dir / "probe.json"

    tab.get_screenshot(path=str(screenshot_path))
    save_json(probe, probe_path)
    return screenshot_path, probe_path


def save_title_area_info(tab, save_dir: Path) -> Path:
    path = save_dir / "title_area.json"
    save_json(inspect_title_area(tab), path)
    return path


def prepare_save_dir(doc: "KDocsDocument", save_dir: Path) -> Path:
    doc.save_dir = Path(save_dir)
    doc.save_dir.mkdir(parents=True, exist_ok=True)
    return doc.save_dir


def normalize_share_scope(scope: str | None) -> str | None:
    if scope is None:
        return None

    mapping = {
        "all": "all",
        "everyone": "all",
        "所有人": "all",
        "specified": "specified",
        "specific": "specified",
        "指定人": "specified",
    }
    key = scope.strip().lower()
    return mapping.get(key, mapping.get(scope.strip()))



def normalize_share_permission(permission: str | None) -> str | None:
    if permission is None:
        return None

    mapping = {
        "edit": "edit",
        "编辑": "edit",
        "comment": "comment",
        "查看和评论": "comment",
        "评论": "comment",
        "view": "view",
        "查看": "view",
        "access": "view",
    }
    key = permission.strip().lower()
    return mapping.get(key, mapping.get(permission.strip()))



def infer_share_state_from_desc(desc: str) -> dict:
    state = {"scope": None, "permission": None}
    if not desc:
        return state

    if "所有人" in desc:
        state["scope"] = "all"
    elif "指定人" in desc or "仅指定人" in desc:
        state["scope"] = "specified"

    if "编辑" in desc:
        state["permission"] = "edit"
    elif "评论" in desc:
        state["permission"] = "comment"
    elif "查看" in desc or "访问" in desc:
        state["permission"] = "view"

    return state



def get_share_button(tab):
    button = tab.ele(SHARE_BUTTON, timeout=3)
    if button and is_displayed(button):
        return button

    for ele in tab.eles("text:分享"):
        if is_displayed(ele):
            parent = ele.parent()
            if parent and parent.tag == "button":
                return parent
            return ele

    raise RuntimeError("未找到分享按钮")



def get_share_panel(tab, timeout: float = 1):
    panel = tab.ele(SHARE_PANEL, timeout=timeout)
    if panel and is_displayed(panel):
        return panel
    return None


def click_share_box_area(tab) -> bool:
    box = tab.ele(SHARE_BOX_AREA, timeout=1)
    if not (box and is_displayed(box)):
        return False

    # 新版 WPS 的分享浮层绑定在外层 share-box-area 上，普通 click 有时不会触发。
    return bool(
        tab.run_js(
            """
            const el = document.querySelector('.share-box-area');
            if (!el) return false;
            const fire = (type, detail = 1) => el.dispatchEvent(new MouseEvent(type, {
                bubbles: true,
                cancelable: true,
                view: window,
                detail,
            }));
            fire('mouseover', 0);
            fire('mouseenter', 0);
            fire('pointerover', 0);
            fire('pointerenter', 0);
            fire('pointerdown', 1);
            fire('mousedown', 1);
            fire('mouseup', 1);
            fire('pointerup', 1);
            fire('click', 1);
            return true;
            """
        )
    )



def open_share_panel(tab):
    panel = get_share_panel(tab, timeout=0.5)
    if panel:
        return panel

    deadline = time.time() + 8
    while time.time() < deadline:
        panel = get_share_panel(tab, timeout=0.2)
        if panel:
            return panel
        try:
            click_with_retry(get_share_button(tab), "打开分享面板")
        except RuntimeError:
            pass
        time.sleep(0.3)
        panel = get_share_panel(tab, timeout=0.2)
        if panel:
            return panel
        if click_share_box_area(tab):
            time.sleep(0.3)
            panel = get_share_panel(tab, timeout=0.2)
            if panel:
                return panel
        time.sleep(0.3)
    raise TimeoutError("分享面板未出现")



def close_share_panel(tab) -> None:
    if not get_share_panel(tab, timeout=0.2):
        return

    deadline = time.time() + 5
    while time.time() < deadline:
        if not get_share_panel(tab, timeout=0.2):
            return
        click_with_retry(get_share_button(tab), "关闭分享面板")
        time.sleep(0.3)
    raise TimeoutError("分享面板未关闭")



def get_share_switch(tab):
    panel = open_share_panel(tab)
    switch = panel.ele(SHARE_SWITCH, timeout=3)
    if switch and is_displayed(switch):
        return switch
    raise RuntimeError("未找到分享开关")



def is_share_enabled(tab) -> bool:
    switch = get_share_switch(tab)
    aria_checked = switch.attr("aria-checked")
    if aria_checked in ("true", "false"):
        return aria_checked == "true"
    return "is-checked" in (switch.attr("class") or "")



def set_share_enabled(tab, enabled: bool) -> bool:
    current = is_share_enabled(tab)
    if current == enabled:
        return False

    click_with_retry(get_share_switch(tab), "切换分享开关")
    wait_until(
        lambda: is_share_enabled(tab) == enabled,
        timeout=10,
        err_msg="分享开关状态未更新",
    )
    return True



def get_share_permission_desc(tab) -> str:
    panel = open_share_panel(tab)
    desc = panel.ele(SHARE_PERMISSION_DESC, timeout=3)
    if desc and is_displayed(desc):
        return desc.text.strip()
    return ""



def get_share_permission_button(tab):
    panel = open_share_panel(tab)
    for ele in panel.eles(SHARE_PERMISSION_BUTTON):
        if not is_displayed(ele):
            continue
        text = (ele.text or "").strip()
        if not text or text == "文档创建者":
            continue
        return ele
    raise RuntimeError("未找到链接权限按钮")



def get_share_permission_popover(tab, timeout: float = 1):
    popover = tab.ele(SHARE_PERMISSION_POPOVER, timeout=timeout)
    if popover and is_displayed(popover) and "kds-link-perm__list" in (popover.html or ""):
        return popover
    return None



def open_share_permission_menu(tab):
    if not is_share_enabled(tab):
        raise RuntimeError("分享开关未开启，无法配置链接权限")

    popover = get_share_permission_popover(tab, timeout=0.5)
    if popover:
        return popover

    deadline = time.time() + 8
    while time.time() < deadline:
        popover = get_share_permission_popover(tab, timeout=0.2)
        if popover:
            return popover
        click_with_retry(get_share_permission_button(tab), "打开链接权限菜单")
        time.sleep(0.3)
    raise TimeoutError("链接权限菜单未出现")



def close_share_permission_menu(tab) -> None:
    if not get_share_permission_popover(tab, timeout=0.2):
        return

    deadline = time.time() + 5
    while time.time() < deadline:
        if not get_share_permission_popover(tab, timeout=0.2):
            return
        try:
            click_with_retry(get_share_permission_button(tab), "关闭链接权限菜单")
        except Exception:
            pass
        time.sleep(0.3)
        if not get_share_permission_popover(tab, timeout=0.2):
            return
        try:
            click_with_retry(get_share_button(tab), "回收链接权限菜单")
        except Exception:
            pass
        time.sleep(0.3)
    raise TimeoutError("链接权限菜单未关闭")



def get_share_permission_lists(tab):
    popover = open_share_permission_menu(tab)
    lists = [ele for ele in popover.eles(SHARE_PERMISSION_LIST) if is_displayed(ele)]
    if len(lists) < 2:
        raise RuntimeError("链接权限菜单结构异常，未找到范围和权限两组配置")
    return lists



def is_share_menu_item_selected(item) -> bool:
    return "<svg" in (item.html or "").lower()



def get_selected_share_menu_label(list_ele) -> str | None:
    for item in list_ele.eles(SHARE_PERMISSION_ITEM):
        if is_displayed(item) and is_share_menu_item_selected(item):
            return item.text.strip()
    return None



def inspect_share_settings(tab) -> dict:
    desc = get_share_permission_desc(tab)
    state = {
        "enabled": is_share_enabled(tab),
        "description": desc,
        "scope": None,
        "permission": None,
    }
    state.update(infer_share_state_from_desc(desc))

    if state["enabled"] and (state["scope"] is None or state["permission"] is None):
        try:
            scope_list, permission_list = get_share_permission_lists(tab)
            if state["scope"] is None:
                state["scope"] = normalize_share_scope(get_selected_share_menu_label(scope_list))
            if state["permission"] is None:
                state["permission"] = normalize_share_permission(
                    get_selected_share_menu_label(permission_list)
                )
        finally:
            try:
                close_share_permission_menu(tab)
            except TimeoutError:
                pass

    return state



def save_share_settings_info(tab, save_dir: Path) -> Path:
    path = save_dir / "share_settings.json"
    save_json(inspect_share_settings(tab), path)
    return path



def select_share_permission_option(tab, group_index: int, target_label: str) -> bool:
    lists = get_share_permission_lists(tab)
    list_ele = lists[group_index]
    current_label = get_selected_share_menu_label(list_ele)
    if current_label == target_label:
        close_share_permission_menu(tab)
        return False

    for item in list_ele.eles(SHARE_PERMISSION_ITEM):
        if is_displayed(item) and item.text.strip() == target_label:
            click_with_retry(item, f"选择{target_label}")

            def _is_done():
                popover = get_share_permission_popover(tab, timeout=0.2)
                if not popover:
                    return True
                lists2 = [ele for ele in popover.eles(SHARE_PERMISSION_LIST) if is_displayed(ele)]
                if len(lists2) <= group_index:
                    return False
                return get_selected_share_menu_label(lists2[group_index]) == target_label

            wait_until(
                _is_done,
                timeout=8,
                err_msg=f"选择{target_label}后权限状态未更新",
            )
            if get_share_permission_popover(tab, timeout=0.2):
                close_share_permission_menu(tab)
            return True

    close_share_permission_menu(tab)
    raise RuntimeError(f"未找到权限选项：{target_label}")



def ensure_share_settings(
    tab,
    *,
    enabled: bool = True,
    scope: str = "all",
    permission: str = "view",
    save_dir: Path | None = None,
) -> dict:
    target_scope = normalize_share_scope(scope)
    target_permission = normalize_share_permission(permission)
    if target_scope is None:
        raise ValueError(f"不支持的分享范围：{scope}")
    if target_permission is None:
        raise ValueError(f"不支持的分享权限：{permission}")

    before = inspect_share_settings(tab)
    changed = False

    if before["enabled"] != enabled:
        changed = set_share_enabled(tab, enabled) or changed

    current = inspect_share_settings(tab)
    if not enabled:
        result = {
            "before": before,
            "after": current,
            "changed": changed or before != current,
        }
        if save_dir is not None:
            save_json(result, save_dir / "share_result.json")
            save_share_settings_info(tab, save_dir)
            tab.get_screenshot(path=str(save_dir / "share_panel.png"))
        return result

    if current["scope"] != target_scope:
        changed = select_share_permission_option(tab, 0, SHARE_SCOPE_LABELS[target_scope]) or changed
        current = inspect_share_settings(tab)

    if current["permission"] != target_permission:
        changed = (
            select_share_permission_option(tab, 1, SHARE_PERMISSION_LABELS[target_permission])
            or changed
        )
        current = inspect_share_settings(tab)

    result = {
        "before": before,
        "after": current,
        "changed": changed or before != current,
    }
    if save_dir is not None:
        save_json(result, save_dir / "share_result.json")
        save_share_settings_info(tab, save_dir)
        tab.get_screenshot(path=str(save_dir / "share_panel.png"))
    return result



class KDocsSession:
    """管理 DrissionPage 浏览器连接和 KDocs 标签页选择。"""

    def __init__(self, port: int = 9222, browser: Chromium | None = None):
        self.port = port
        self.browser = browser or connect_browser(port)

    def describe_tabs(self) -> list[dict]:
        return describe_kdocs_tabs(self.browser)

    def activate_tab(self, tab):
        self.browser.activate_tab(tab)
        time.sleep(1)
        return tab

    def pick_document(
        self,
        *,
        url_contains: str = "kdocs.cn",
        title_contains: str | None = None,
        prefer_locator: str | None = None,
        save_dir: str | Path = DEFAULT_ARTIFACT_ROOT,
        wait_seconds: float = 8,
    ) -> "KDocsDocument":
        tab = pick_kdocs_tab(
            self.browser,
            url_contains=url_contains,
            title_contains=title_contains,
            prefer_locator=prefer_locator,
        )
        doc = KDocsDocument(
            session=self,
            url=tab.url,
            tab=tab,
            save_dir=save_dir,
            wait_seconds=wait_seconds,
        )
        doc.activate()
        return doc

    def pick_document_by_url(
        self,
        url: str,
        *,
        prefer_locator: str | None = None,
        save_dir: str | Path = DEFAULT_ARTIFACT_ROOT,
        wait_seconds: float = 8,
    ) -> "KDocsDocument":
        tab = pick_kdocs_tab_by_url(self.browser, url, prefer_locator=prefer_locator)
        doc = KDocsDocument(
            session=self,
            url=url,
            tab=tab,
            save_dir=save_dir,
            wait_seconds=wait_seconds,
        )
        doc.activate()
        return doc

    def open_document(
        self,
        url: str,
        *,
        save_dir: str | Path = DEFAULT_ARTIFACT_ROOT,
        wait_seconds: float = 8,
    ) -> "KDocsDocument":
        doc = KDocsDocument(
            session=self,
            url=url,
            save_dir=save_dir,
            wait_seconds=wait_seconds,
        )
        doc.open()
        return doc

    def get_document(
        self,
        url: str,
        *,
        reuse_existing: bool = True,
        prefer_locator: str | None = None,
        save_dir: str | Path = DEFAULT_ARTIFACT_ROOT,
        wait_seconds: float = 8,
    ) -> "KDocsDocument":
        if reuse_existing:
            try:
                return self.pick_document_by_url(
                    url,
                    prefer_locator=prefer_locator,
                    save_dir=save_dir,
                    wait_seconds=wait_seconds,
                )
            except RuntimeError:
                pass

        return self.open_document(
            url,
            save_dir=save_dir,
            wait_seconds=wait_seconds,
        )


class KDocsDocument:
    """面向对象封装单个 WPS/KDocs 文档操作。"""

    def __init__(
        self,
        *,
        session: KDocsSession,
        url: str,
        tab=None,
        save_dir: str | Path = DEFAULT_ARTIFACT_ROOT,
        wait_seconds: float = 8,
    ):
        self.session = session
        self.url = url
        self.tab = tab
        self.save_dir = Path(save_dir)
        self.wait_seconds = wait_seconds

    @property
    def browser(self) -> Chromium:
        return self.session.browser

    def __repr__(self) -> str:
        title = self.tab.title if self.tab else "<not-opened>"
        return f"KDocsDocument(title={title!r}, url={self.url!r})"

    def activate(self) -> "KDocsDocument":
        if not self.tab:
            raise RuntimeError("文档标签页尚未绑定，不能激活")
        self.session.activate_tab(self.tab)
        return self

    def open(self) -> "KDocsDocument":
        self.tab = open_target_tab(self.browser, self.url, self.wait_seconds)
        return self

    def ensure_tab(
        self,
        *,
        reuse_existing: bool = True,
        prefer_locator: str | None = None,
    ) -> "KDocsDocument":
        if self.tab:
            self.activate()
            return self

        if reuse_existing:
            try:
                picked = self.session.pick_document_by_url(
                    self.url,
                    prefer_locator=prefer_locator,
                    save_dir=self.save_dir,
                    wait_seconds=self.wait_seconds,
                )
                self.tab = picked.tab
                return self
            except RuntimeError:
                pass

        return self.open()

    def collect_probe(self) -> dict:
        return collect_probe(self.tab)

    def save_probe_files(self, probe: dict | None = None) -> tuple[Path, Path]:
        probe = probe or self.collect_probe()
        return save_probe_files(self.tab, probe, self.save_dir)

    def inspect_title_area(self) -> dict:
        return inspect_title_area(self.tab)

    def save_title_area_info(self) -> Path:
        return save_title_area_info(self.tab, self.save_dir)

    def snapshot(self) -> dict:
        probe = self.collect_probe()
        screenshot_path, probe_path = self.save_probe_files(probe)
        title_area_path = self.save_title_area_info()
        return {
            "probe": probe,
            "screenshot_path": screenshot_path,
            "probe_path": probe_path,
            "title_area_path": title_area_path,
        }

    def enter_title_edit_mode(self):
        return enter_title_edit_mode(self.tab)

    def open_file_menu(self) -> None:
        open_file_menu(self.tab)

    def open_save_submenu(self) -> None:
        open_save_submenu(self.tab)

    def create_copy(self, *, open_now: bool = False) -> dict:
        return run_create_copy_flow(
            self.tab,
            self.save_dir,
            browser=self.browser,
            open_now=open_now,
        )

    def open_now_from_modal(self) -> dict:
        return open_copied_doc_from_modal(self.tab, self.browser, self.save_dir)

    def rename_title(self, new_title: str) -> dict:
        return rename_document(self.tab, new_title, self.save_dir)

    def open_share_panel(self):
        return open_share_panel(self.tab)

    def close_share_panel(self) -> None:
        close_share_panel(self.tab)

    def inspect_share_settings(self) -> dict:
        return inspect_share_settings(self.tab)

    def save_share_settings_info(self) -> Path:
        return save_share_settings_info(self.tab, self.save_dir)

    def ensure_share_settings(
        self,
        *,
        enabled: bool = True,
        scope: str = "all",
        permission: str = "view",
    ) -> dict:
        return ensure_share_settings(
            self.tab,
            enabled=enabled,
            scope=scope,
            permission=permission,
            save_dir=self.save_dir,
        )


def get_file_menu_button(tab):
    locators = [
        "text:文件操作",
        "css:.app-header-more-btn button",
        "css:.app-header-more-btn",
        "css:.kd-icon-menu",
    ]
    for locator in locators:
        ele = find_visible(tab, locator, timeout=1)
        if ele:
            return ele
    raise RuntimeError("未找到文件操作按钮")


def enter_title_edit_mode(tab):
    input_ele = get_visible_title_input(tab)
    if input_ele:
        return input_ele

    box = get_title_box(tab)
    click_with_retry(box, "点击标题区进入改名态")
    input_ele = find_visible(tab, TITLE_EDIT_INPUT, timeout=2)
    if input_ele:
        return input_ele

    # 新版 WPS 表格页常见为“单击无效、标题文本节点需要接近真实双击的事件序列”。
    for selector in (
        "#filename",
        ".component-header-file-info .filename",
        ".component-header-file-info .filename-wrap",
        ".component-header-file-info",
    ):
        tab.run_js(
            f"""
            const el = document.querySelector({selector!r});
            if (!el) return false;
            const fire = (type, detail) => el.dispatchEvent(new MouseEvent(type, {{
                bubbles: true,
                cancelable: true,
                view: window,
                detail,
            }}));
            fire('mouseover', 0);
            fire('mousedown', 1);
            fire('mouseup', 1);
            fire('click', 1);
            fire('mousedown', 2);
            fire('mouseup', 2);
            fire('click', 2);
            fire('dblclick', 2);
            return true;
            """
        )
        input_ele = find_visible(tab, TITLE_EDIT_INPUT, timeout=1)
        if input_ele:
            return input_ele

    return wait_until(
        lambda: get_visible_title_input(tab),
        timeout=8,
        err_msg="点击标题后未进入改名态",
    )


def ensure_file_menu_closed(tab) -> None:
    panel = find_visible(tab, FILE_MENU_PANEL, timeout=0.5)
    if not panel:
        return

    btn = get_file_menu_button(tab)
    click_with_retry(btn, "关闭文件操作菜单")
    wait_until(
        lambda: not find_visible(tab, FILE_MENU_PANEL, timeout=0.2),
        timeout=5,
        err_msg="文件操作菜单未能关闭",
    )


def open_file_menu(tab) -> None:
    btn = get_file_menu_button(tab)
    click_with_retry(btn, "打开文件操作菜单")
    wait_until(
        lambda: find_visible(tab, FILE_MENU_PANEL, timeout=0.2),
        timeout=8,
        err_msg="文件操作菜单未出现",
    )


def get_save_entry(tab):
    candidates = tab.eles("css:.file-more-panel .block-menu-item")
    for ele in candidates:
        if is_displayed(ele) and "另存" in ele.text:
            return ele

    candidates = tab.eles("css:.file-more-panel .component-menu-item.sub-menu-item")
    for ele in candidates:
        if is_displayed(ele) and "另存" in ele.text:
            return ele

    ele = find_visible(tab, "text:另存", timeout=1)
    if ele:
        return ele
    raise RuntimeError("未找到“另存”入口")


def open_save_submenu(tab) -> None:
    entry = get_save_entry(tab)
    click_with_retry(entry, "打开另存子菜单")
    wait_until(
        lambda: find_visible(tab, SAVE_SUBMENU_PANEL, timeout=0.2),
        timeout=8,
        err_msg="另存子菜单未出现",
    )


def get_create_copy_entry(tab):
    candidates = tab.eles("css:.header-more-block-submenu .component-menu-item")
    for ele in candidates:
        if is_displayed(ele) and "创建副本" in ele.text:
            return ele

    ele = find_visible(tab, "text:创建副本", timeout=1)
    if ele:
        return ele
    raise RuntimeError("未找到“创建副本”入口")


def wait_copy_success(tab):
    title_ele = wait_until(
        lambda: find_visible(tab, COPY_SUCCESS_TITLE, timeout=0.2),
        timeout=20,
        err_msg="未等到“创建副本成功”提示",
    )

    def find_path_text():
        for ele in tab.eles("xpath://div[contains(text(), '我的云文档')]"):
            if is_displayed(ele):
                return ele.text
        return None

    path_text = wait_until(
        find_path_text,
        timeout=10,
        err_msg="未等到副本保存路径提示",
    )
    return title_ele.text, path_text


def get_open_now_button(tab):
    button = find_visible(tab, OPEN_NOW_BUTTON, timeout=1)
    if button:
        return button
    raise RuntimeError("未找到“立即打开”按钮")


def wait_opened_copy_tab(browser: Chromium, source_url: str, before_tab_ids: set[str]):
    def find_new_tab():
        for tab in get_kdocs_tabs(browser):
            tab_id = getattr(tab, "tab_id", "")
            if tab_id and tab_id not in before_tab_ids:
                return tab

        for tab in get_kdocs_tabs(browser):
            if (tab.url or "") != source_url and "副本" in (tab.title or ""):
                return tab
        return None

    return wait_until(
        find_new_tab,
        timeout=20,
        err_msg="点击“立即打开”后未出现副本文档标签页",
    )


def open_copied_doc_from_modal(tab, browser: Chromium, save_dir: Path | None = None) -> dict:
    before_tab_ids = set(browser.tab_ids)
    source_url = tab.url
    button = get_open_now_button(tab)
    click_with_retry(button, "点击立即打开")

    opened_tab = wait_opened_copy_tab(browser, source_url, before_tab_ids)
    time.sleep(2)

    result = {
        "opened_title": opened_tab.title,
        "opened_url": opened_tab.url,
        "source_title": tab.title,
        "source_url": source_url,
    }
    if save_dir is not None:
        save_json(result, save_dir / "open_now_result.json")
        opened_tab.get_screenshot(path=str(save_dir / "opened_copy.png"))
    return result


def rename_document(tab, new_title: str, save_dir: Path | None = None) -> dict:
    old_title = get_title_text(tab) or tab.title
    if old_title == new_title and tab.title == new_title:
        result = {
            "old_title": old_title,
            "new_title": new_title,
            "changed": False,
            "url": tab.url,
        }
        if save_dir is not None:
            save_json(result, save_dir / "rename_result.json")
            save_title_area_info(tab, save_dir)
        return result

    input_ele = enter_title_edit_mode(tab)
    click_with_retry(input_ele, "聚焦标题输入框")
    time.sleep(0.3)
    tab.actions.key_down(Keys.CTRL).type("a").key_up(Keys.CTRL)
    time.sleep(0.2)
    tab.actions.type(new_title)
    time.sleep(0.4)
    tab.actions.type(Keys.ENTER)

    wait_until(
        lambda: (tab.title or "") == new_title and get_title_text(tab) == new_title,
        timeout=20,
        err_msg="未等到标题更新完成",
    )

    result = {
        "old_title": old_title,
        "new_title": new_title,
        "changed": True,
        "url": tab.url,
        "tab_title": tab.title,
    }
    if save_dir is not None:
        save_json(result, save_dir / "rename_result.json")
        save_title_area_info(tab, save_dir)
        tab.get_screenshot(path=str(save_dir / "renamed_page.png"))
    return result


def run_create_copy_flow(
    tab,
    save_dir: Path,
    *,
    browser: Chromium | None = None,
    open_now: bool = False,
) -> dict:
    ensure_file_menu_closed(tab)
    open_file_menu(tab)
    open_save_submenu(tab)

    copy_entry = get_create_copy_entry(tab)
    click_with_retry(copy_entry, "点击创建副本")
    success_title, saved_path = wait_copy_success(tab)

    result = {
        "success_title": success_title,
        "saved_path": saved_path,
        "url": tab.url,
        "title": tab.title,
    }
    save_json(result, save_dir / "copy_result.json")
    tab.get_screenshot(path=str(save_dir / "copy_success.png"))

    if open_now:
        if browser is None:
            raise ValueError("open_now=True 时必须传入 browser")
        result["opened_copy"] = open_copied_doc_from_modal(tab, browser, save_dir)
        save_json(result, save_dir / "copy_result.json")

    return result


def main() -> None:
    args = build_parser().parse_args()
    share_requested = any(
        value is not None
        for value in (args.share_enabled, args.share_scope, args.share_permission)
    )
    requested_actions = build_requested_actions(
        create_copy=args.create_copy,
        open_now=args.open_now,
        rename_title=args.rename_title,
        share_requested=share_requested,
    )

    print(f"connecting browser on port {args.port} ...", flush=True)
    session = KDocsSession(args.port)
    browser = session.browser
    print("browser connected", flush=True)

    if args.open_now and not args.create_copy:
        doc = session.pick_document(
            prefer_locator=OPEN_NOW_BUTTON,
            save_dir=DEFAULT_ARTIFACT_ROOT,
            wait_seconds=args.wait_seconds,
        )
        save_dir = prepare_save_dir(
            doc,
            resolve_artifact_dir(
                explicit_save_dir=args.save_dir,
                save_root=args.save_root,
                url=doc.url,
                actions=requested_actions,
                label=args.label,
            ),
        )
        save_json(
            {
                "requested_url": args.url,
                "resolved_url": doc.url,
                "actions": requested_actions,
                "label": args.label,
                "save_dir": str(save_dir),
                "port": args.port,
            },
            save_dir / "run_context.json",
        )
        print(f"save_dir: {save_dir}", flush=True)
        print(f"using existing tab: {doc.tab.title}", flush=True)
        result = doc.open_now_from_modal()
        save_json(
            {
                "requested_actions": requested_actions,
                "requested_url": args.url,
                "resolved_url": doc.url,
                "current_tab_title": doc.tab.title,
                "current_tab_url": doc.tab.url,
                "results": {"open": result},
            },
            save_dir / "run_summary.json",
        )
        print("open_now_result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    doc = session.get_document(
        args.url,
        reuse_existing=bool(args.rename_title or share_requested),
        save_dir=DEFAULT_ARTIFACT_ROOT,
        wait_seconds=args.wait_seconds,
    )
    save_dir = prepare_save_dir(
        doc,
        resolve_artifact_dir(
            explicit_save_dir=args.save_dir,
            save_root=args.save_root,
            url=doc.url,
            actions=requested_actions,
            label=args.label,
        ),
    )
    save_json(
        {
            "requested_url": args.url,
            "resolved_url": doc.url,
            "actions": requested_actions,
            "label": args.label,
            "save_dir": str(save_dir),
            "port": args.port,
        },
        save_dir / "run_context.json",
    )
    print(f"save_dir: {save_dir}", flush=True)
    if args.rename_title or share_requested:
        print(f"using existing tab: {doc.tab.title}", flush=True)

    print("collecting probe ...", flush=True)
    probe = doc.collect_probe()
    print("saving probe files ...", flush=True)
    screenshot_path, probe_path = doc.save_probe_files(probe)
    title_area_path = doc.save_title_area_info()

    print("WPS 页面探测完成")
    print(f"title: {probe['title']}")
    print(f"url: {probe['url']}")
    print(f"html_length: {probe['html_length']}")
    print(f"iframe_count: {probe['iframe_count']}")
    if probe["possible_login_hints"]:
        print("possible_login_hints:", ", ".join(probe["possible_login_hints"]))
    print(f"screenshot: {screenshot_path}")
    print(f"probe_json: {probe_path}")
    print(f"title_area_json: {title_area_path}")
    print("visible_text_preview:")
    print(probe["visible_text_preview"][:1000])

    copy_result = None
    if args.create_copy:
        print("running create-copy flow ...", flush=True)
        copy_result = doc.create_copy(open_now=args.open_now)
        print("create_copy_result:")
        print(json.dumps(copy_result, ensure_ascii=False, indent=2))

    rename_result = None
    if args.rename_title:
        print("running rename flow ...", flush=True)
        rename_result = doc.rename_title(args.rename_title)
        print("rename_result:")
        print(json.dumps(rename_result, ensure_ascii=False, indent=2))

    share_result = None
    if share_requested:
        print("running share flow ...", flush=True)
        share_result = doc.ensure_share_settings(
            enabled=args.share_enabled != "off" if args.share_enabled else True,
            scope=args.share_scope or "all",
            permission=args.share_permission or "view",
        )
        print("share_result:")
        print(json.dumps(share_result, ensure_ascii=False, indent=2))

    run_summary = {
        "requested_actions": requested_actions,
        "requested_url": args.url,
        "resolved_url": doc.url,
        "current_tab_title": doc.tab.title,
        "current_tab_url": doc.tab.url,
        "results": {
            "probe": {
                "title": probe["title"],
                "url": probe["url"],
                "html_length": probe["html_length"],
                "iframe_count": probe["iframe_count"],
            },
        },
    }
    if copy_result is not None:
        run_summary["results"]["copy"] = copy_result
    if rename_result is not None:
        run_summary["results"]["rename"] = rename_result
    if share_result is not None:
        run_summary["results"]["share"] = share_result
    save_json(run_summary, save_dir / "run_summary.json")


if __name__ == "__main__":
    main()
