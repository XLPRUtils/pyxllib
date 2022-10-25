#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/08/01 15:28

import time
import sys

____press_key = """
TODO 有待进一步封装整理

简化按键映射、简化PressKey和ReleaseKey过程
https://www.icode9.com/content-1-787701.html

简洁版可以使用: https://pypi.org/project/VirtualKey/
看源码原理基本相同，但实测还是有失灵的时候（HD3_Launcher.exe打开的游戏无法接受按键信息）
所以我还是要自己留一套版本
"""

if sys.platform == 'win32':
    import ctypes

    KEY_MAPPING = {'num1': 2, 'num2': 3, 'num3': 4, 'num4': 5, 'num5': 6,
                   'num6': 7, 'num7': 8, 'num8': 9, 'num9': 10, 'num0': 11,
                   'escape': 1, 'equal': 13, 'backspace': 14, 'tab': 15, 'q': 16,
                   'w': 17, 'e': 18, 'r': 19, 't': 20, 'y': 21,
                   'u': 22, 'i': 23, 'o': 24, 'p': 25, 'enter': 28,
                   'lcontrol': 29, 'a': 30, 's': 31, 'd': 32, 'f': 33,
                   'g': 34, 'h': 35, 'j': 36, 'k': 37, 'l': 38,
                   'z': 44, 'x': 45, 'c': 46, 'v': 47, 'b': 48,
                   'n': 49, 'm': 50, 'shift': 54, 'multiply': 55, 'space': 57,
                   'capital': 58, 'f1': 59, 'f2': 60, 'f3': 61, 'f4': 62,
                   'f5': 63, 'f6': 64, 'f7': 65, 'f8': 66, 'f9': 67,
                   'f10': 68, 'numlock': 69, 'f11': 87, 'f12': 88, 'divide': 181,
                   'home': 199, 'up': 200, 'prior': 201, 'left': 203, 'right': 205,
                   'end': 207, 'down': 208, 'next': 209, 'insert': 210, 'delete': 211}

    SendInput = ctypes.windll.user32.SendInput

    # C struct redefinitions
    PUL = ctypes.POINTER(ctypes.c_ulong)


    class KeyBdInput(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort),
                    ("wScan", ctypes.c_ushort),
                    ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong),
                    ("dwExtraInfo", PUL)]


    class HardwareInput(ctypes.Structure):
        _fields_ = [("uMsg", ctypes.c_ulong),
                    ("wParamL", ctypes.c_short),
                    ("wParamH", ctypes.c_ushort)]


    class MouseInput(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long),
                    ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong),
                    ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong),
                    ("dwExtraInfo", PUL)]


    class Input_I(ctypes.Union):
        _fields_ = [("ki", KeyBdInput),
                    ("mi", MouseInput),
                    ("hi", HardwareInput)]


    class Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong),
                    ("ii", Input_I)]


    # Actuals Functions

    def PressKey(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


    def ReleaseKey(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


    def down_up(ch, t=0.5):
        ch = KEY_MAPPING[ch]
        PressKey(ch)
        time.sleep(t)
        ReleaseKey(ch)
        return 1
