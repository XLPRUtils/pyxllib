#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2025/06/05
import os
import time

pyxllib_start_time = time.time()

import datetime
import requests

import itertools
from unittest.mock import MagicMock

from loguru import logger
from notion_client import Client as NotionClient
import notion_client.errors
from dotenv import load_dotenv

from pyxllib.xl import *
from pyxllib.file.specialist import cache_file
from pyxllib.file.xlsxlib import *

from pyxllib.xlcv import *

from pyxllib.ext.unixlib import XlSSHClient, get_ssh
from pyxllib.data.pglib import XlprDb, SqlBuilder, get_xldb1, get_xldb2, get_xldb3
import psycopg

from pyxllib.ext.drissionlib import get_dp_page, get_dp_tab

from pyxlpr.openai2 import Chat, Chat2, CompressContent, DifyChat
from pyxllib.file.xlsyncfile import SyncFileClient, XlSyncFileClient

from pyxllib.text.jscode import get_airscript_head2

if sys.platform == 'win32':
    import win32com.client as win32

from pyxllib.ext.yuquelib import Yuque, LakeImage, XlLakeImage
from pyxllib.ext.wpsapi2 import WpsOnlineBook

from pyxllib.prog.multiprogs import run_python_module, support_retry_process

from pyxllib.prog.xlenv import *

# 如果当前工作目录有.env文件，则读取
if Path('.env').exists():
    load_dotenv('.env')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fire.Fire()
        exit()

    xlpr0 = get_ssh('titan2')
    print(xlpr0.exec('ls'))
