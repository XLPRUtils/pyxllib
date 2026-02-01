""" 每个目录下的api.py算是比较全量的对外接口全集，方便有时不想找嵌套具体位置，要快速使用的时候 """

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

# from pyxllib.ai.chat import Chat, CompressContent, DifyChat
from pyxllib.file.xlsyncfile import SyncFileClient, XlSyncFileClient

from pyxllib.text.jscode import get_airscript_head2

if sys.platform == 'win32':
    import win32com.client as win32

from pyxllib.ext.yuquelib import Yuque, LakeImage, XlLakeImage
from pyxllib.ext.wpsapi import WpsOnlineBook

from pyxllib.prog.scheduler import run_python_module, support_retry_process

from pyxllib.prog.xlenv import *
