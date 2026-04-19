#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2025/06/22

""" 惟海法师网课考勤工具之基础工具库
"""

import contextlib
import io
import json
import base64
from collections import Counter
from datetime import timedelta
import datetime
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from urllib.parse import urlencode, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

import DrissionPage.errors
from DrissionPage import Chromium
from tqdm import tqdm

from pyxllib.prog.xltime import parse_datetime
from pyxllib.prog.debug import func_input_message, tprint
from pyxllib.prog.time import utc_timestamp, TicToc
from pyxllib.prog.browser import browser
from pyxllib.file.xlpath import XlPath, cache_file
from pyxllib.ext.drissionlib import get_latest_not_dev_tab, XlTab, DpWebBase, close_duplicate_tabs, get_dp_tab

from pyxllib.data.pglib import XlprDb, SqlBuilder, get_xldb3
from pyxllib.algo.stat import custom_fillna, xlpivot
from pyxllib.prog.uni_cache import uni_cache
from pyxllib.prog.xlenv import XlEnv, get_xl_homedir, get_xl_hostname, xlhome_dir, xlhome_wkdir
from pyxllib.prog.debug import format_exception
from pyxllib.autogui.wxautolib import wechat_lock_send, WeChatSingletonLock, wechat_logger, WeChat
from pyxllib.text.levenshtein import get_levenshtein_similar
from pyxllib.ext.wpsapi import WpsOnlineBook
from pyxllib.text.convert import chinese2digits
from pyxllib.cv.slidercaptcha import SliderCaptchaLocator
from pyxllib.prog.filelock import get_autogui_lock
try:
    from loguru import logger
except ModuleNotFoundError:
    import logging

    logger = logging.getLogger(__name__)

try:
    import xlproject.loadenv  # noqa: F401
except Exception:
    pass

import pyautogui

try:  # 旧版vip买的wxautox
    from wxautox.elements import WeChatImage
except ImportError:  # pypi的wxautox，这个跟前面的那个wxautox还是有区别的
    from wxautox.ui.component import WeChatImage

from wxautox import uia
from pyxllib.autogui.uiautolib import UiCtrlNode


def __1_单件功能类():
    """ 一共有五大件：小鹅通（api版/爬虫版）、微信、微信支付、数据库、考勤总表
    """
    pass


def 清理问卷星干扰弹窗(tab, max_rounds=3, wait_seconds=0.5):
    """ 关闭问卷星里偶发的活动弹窗、推广浮层；没有则直接跳过 """
    js = r"""
const visible = (el) => {
  if (!el) return false;
  const style = window.getComputedStyle(el);
  const rect = el.getBoundingClientRect();
  return style.display !== 'none'
    && style.visibility !== 'hidden'
    && style.opacity !== '0'
    && rect.width >= 0
    && rect.height >= 0
    && rect.bottom >= 0
    && rect.right >= 0;
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

for (const sel of selectors) {
  for (const el of document.querySelectorAll(sel)) {
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
            cnt = tab.run_js(js) or 0
        except Exception as e:
            logger.warning(f'清理问卷星干扰弹窗异常：{e}')
            break
        total += cnt
        if not cnt:
            break
        time.sleep(wait_seconds)

    if total:
        logger.info(f'已清理问卷星干扰弹窗/浮层：{total}处')
    return total


def 尝试关闭重复页面(browser=None, timeout=3, reason='', keep_tab_ids=None):
    """ 轻量尝试关闭重复页面，避开 get_tabs() 这条不稳定链路 """
    if browser is None:
        return False
    keep_tab_ids = {x for x in (keep_tab_ids or []) if x}

    try:
        target_infos = browser._run_cdp('Target.getTargets').get('targetInfos', [])
    except Exception as e:
        extra = f'，原因={reason}' if reason else ''
        logger.warning(f'获取标签页目标信息失败，已跳过本次清理{extra}: {e}')
        return False

    tab_infos = []
    for info in target_infos:
        if info.get('type') != 'page':
            continue
        tab_id = info.get('targetId')
        url = info.get('url') or ''
        if not tab_id:
            continue
        tab_infos.append({'tab_id': tab_id, 'url': url})

    if len(tab_infos) <= 1:
        return True

    close_ids = []
    seen_domains = set()
    chrome_like_ids = []

    for info in tab_infos:
        if info['tab_id'] not in keep_tab_ids:
            continue
        url = info['url']
        if url.startswith('chrome://newtab') or url.startswith('chrome://omnibox-popup'):
            continue
        domain = urlparse(url).netloc or url
        seen_domains.add(domain)

    for info in tab_infos:
        if info['tab_id'] in keep_tab_ids:
            continue
        url = info['url']
        if url.startswith('chrome://newtab') or url.startswith('chrome://omnibox-popup'):
            chrome_like_ids.append(info['tab_id'])
            continue

        domain = urlparse(url).netloc or url
        if domain in seen_domains:
            close_ids.append(info['tab_id'])
        else:
            seen_domains.add(domain)

    remaining_after_dup = len(tab_infos) - len(close_ids)
    if remaining_after_dup > 1:
        close_ids.extend(chrome_like_ids)

    closed = 0
    for info in tab_infos:
        if info['tab_id'] not in close_ids:
            continue
        try:
            browser._run_cdp('Target.closeTarget', targetId=info['tab_id'])
            closed += 1
        except Exception:
            try:
                browser.close_tabs(info['tab_id'])
                closed += 1
            except Exception as e:
                logger.warning(f'关闭标签页失败，已跳过 tab={info["tab_id"]}: {e}')

    if closed:
        extra = f'，原因={reason}' if reason else ''
        logger.info(f'已关闭重复页面：{closed}个{extra}')
    return True
