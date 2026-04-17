# cython: language_level=3
from .wxauto import WeChat
from .elements import WxParam, LoginWnd
from .utils import *

__version__ = VERSION

__all__ = [
    'WeChat', 
    'VERSION',
    'WxParam',
    'OpenWeChat',
    'LoginWnd'
]
