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


def _patch_drissionpage_download_move():
    """兼容新版 Chrome 下载完成后不再保留 guid 临时文件的情况。"""

    try:
        from DataRecorder.tools import get_usable_path
        from DrissionPage._units.downloader import DownloadManager
    except Exception:
        return

    if getattr(DownloadManager, '_kq5034_missing_tmp_file_patched', False):
        return

    original_on_progress = DownloadManager._onDownloadProgress

    def is_recent_download_file(path):
        try:
            return (
                path.is_file()
                and not path.name.endswith('.crdownload')
                and time.time() - path.stat().st_mtime <= 300
            )
        except OSError:
            return False

    def on_download_progress(self, **kwargs):
        try:
            return original_on_progress(self, **kwargs)
        except FileNotFoundError as err:
            mission = self._missions.get(kwargs.get('guid'))
            if kwargs.get('state') != 'completed' or mission is None:
                raise

            tmp_dir = Path(mission.tmp_path)
            folder = Path(mission.folder)
            default_to_path = folder / mission.name

            candidates = [
                tmp_dir / mission.id,
                Path(tempfile.gettempdir()) / mission.name,
                tmp_dir / mission.name,
                default_to_path,
            ]
            source = next((p for p in candidates if is_recent_download_file(p)), None)
            if source is None:
                logger.warning(f'DrissionPage下载完成但临时文件缺失，未找到兜底文件：{mission.name} err={err}')
                self.set_done(mission, 'canceled')
                return

            try:
                mission.received_bytes = kwargs.get('receivedBytes', mission.received_bytes)
                mission.total_bytes = kwargs.get('totalBytes', mission.total_bytes)
                if source.resolve() == default_to_path.resolve() or mission._overwrite is not None:
                    to_path = default_to_path
                else:
                    to_path = Path(get_usable_path(str(default_to_path)))
                folder.mkdir(parents=True, exist_ok=True)
                if source.resolve() != to_path.resolve():
                    for _ in range(10):
                        try:
                            shutil.move(str(source), str(to_path))
                            break
                        except PermissionError:
                            time.sleep(.5)
                    else:
                        shutil.copy2(str(source), str(to_path))
                self.set_done(mission, 'completed', final_path=str(to_path))
                logger.warning(f'DrissionPage下载临时文件缺失，已按文件名兜底搬运：{source} -> {to_path}')
            except Exception as fallback_err:
                logger.warning(f'DrissionPage下载兜底搬运失败：{mission.name} err={fallback_err}')
                self.set_done(mission, 'canceled')

    DownloadManager._onDownloadProgress = on_download_progress
    DownloadManager._kq5034_missing_tmp_file_patched = True


_patch_drissionpage_download_move()

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
