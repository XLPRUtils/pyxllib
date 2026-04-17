# cython: language_level=3
from datetime import datetime, timedelta
from . import uiautomation as uia
from PIL import Image
from ctypes import windll
import win32clipboard
import win32process
import subprocess
import win32gui
import win32api
import win32con
import win32ui
import pyperclip
import tenacity
import ctypes
import psutil
import shutil
import winreg
import logging
import hashlib
import time
import sys
import os
import re

ORIVERSION = "3.9.8.15"
VERSION = "3.9.11.17"
AGREEMENTHASH = 'c155ea310562359e0975cfe317386951e2a8297e3a3e9ab5064998c4465c41142e32ea66e82669d351f83be211623aa1e1bbfee65d3226d951553b8f2182f363'

AGREEMENT_CONTENT = """用户协议
最后更新日期：2024年12月7日

感谢您使用 wxauto(x)（以下简称“本项目”）。为明确用户责任，特制定本用户协议（以下简称“协议”）。请在使用前仔细阅读并同意以下条款。您使用本项目即视为您已接受并同意遵守本协议。

许可与使用限制
1. 合法用途
用户应仅将本项目用于合法用途，包括但不限于：
- 个人学习和研究。
- 在不违反适用法律法规及第三方协议（如微信用户协议）的情况下个人使用。

2. 禁止行为
- 不得私自删除该协议中任何内容。
- 用户不得将本项目用于以下用途：
- 开发、分发或使用任何违反法律法规的工具或服务。
- 开发、分发或使用任何违反第三方平台规则（如微信用户协议）的工具或服务。
- 从事任何危害他人权益、平台安全或公共利益的行为。

3. 风险与责任
用户在使用本项目时，须自行确保其行为的合法性及合规性。
任何因使用本项目而产生的法律风险、责任及后果，由用户自行承担。
"""

def set_cursor_pos(x, y):
    win32api.SetCursorPos((x, y))
    
def Click(rect):
    x = (rect.left + rect.right) // 2
    y = (rect.top + rect.bottom) // 2
    set_cursor_pos(x, y)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
    
def GetPathByHwnd(hwnd):
    try:
        thread_id, process_id = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(process_id)
        return process.exe()
    except Exception as e:
        print(f"Error: {e}")
        return None

def GetVersionByPath(file_path):
    try:
        info = win32api.GetFileVersionInfo(file_path, '\\')
        version = "{}.{}.{}.{}".format(win32api.HIWORD(info['FileVersionMS']),
                                        win32api.LOWORD(info['FileVersionMS']),
                                        win32api.HIWORD(info['FileVersionLS']),
                                        win32api.LOWORD(info['FileVersionLS']))
    except:
        version = None
    return version

def capture(hwnd, bbox):
    # 获取窗口的屏幕坐标
    window_rect = win32gui.GetWindowRect(hwnd)
    win_left, win_top, win_right, win_bottom = window_rect
    win_width = win_right - win_left
    win_height = win_bottom - win_top

    # 获取窗口的设备上下文
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    # 创建位图对象保存整个窗口截图
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, win_width, win_height)
    saveDC.SelectObject(saveBitMap)

    # 使用PrintWindow捕获整个窗口（包括被遮挡或最小化的窗口）
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)

    # 转换为PIL图像
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    im = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    # 释放资源
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    # 计算bbox相对于窗口左上角的坐标
    bbox_left, bbox_top, bbox_right, bbox_bottom = bbox
    # 转换为截图图像中的相对坐标
    crop_left = bbox_left - win_left
    crop_top = bbox_top - win_top
    crop_right = bbox_right - win_left
    crop_bottom = bbox_bottom - win_top

    # 裁剪目标区域
    cropped_im = im.crop((crop_left, crop_top, crop_right, crop_bottom))
    
    return cropped_im

def IsRedPixel(uicontrol):
    rect = uicontrol.BoundingRectangle
    hwnd = uicontrol.GetAncestorControl(lambda x,y:x.ClassName=='WeChatMainWndForPC').NativeWindowHandle
    bbox = (rect.left, rect.top, rect.right, rect.bottom)
    img = capture(hwnd, bbox)
    return any(p[0] > p[1] and p[0] > p[2] for p in img.getdata())

class DROPFILES(ctypes.Structure):
    _fields_ = [
    ("pFiles", ctypes.c_uint),
    ("x", ctypes.c_long),
    ("y", ctypes.c_long),
    ("fNC", ctypes.c_int),
    ("fWide", ctypes.c_bool),
    ]

pDropFiles = DROPFILES()
pDropFiles.pFiles = ctypes.sizeof(DROPFILES)
pDropFiles.fWide = True
matedata = bytes(pDropFiles)

def SetClipboardText(text: str):
    pyperclip.copy(text)
    # if not isinstance(text, str):
    #     raise TypeError(f"参数类型必须为str --> {text}")
    # t0 = time.time()
    # while True:
    #     if time.time() - t0 > 10:
    #         raise TimeoutError(f"设置剪贴板超时！ --> {text}")
    #     try:
    #         win32clipboard.OpenClipboard()
    #         win32clipboard.EmptyClipboard()
    #         win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
    #         break
    #     except:
    #         pass
    #     finally:
    #         try:
    #             win32clipboard.CloseClipboard()
    #         except:
    #             pass

try:
    from anytree import Node, RenderTree

    def PrintAllControlTree(ele):
        def findall(ele, n=0, node=None):
            nn = '\n'
            nodename = f"[{ele.ControlTypeName} {n}](\"{ele.ClassName}\", \"{ele.Name.replace(nn, '')}\", \"{''.join([str(i) for i in ele.GetRuntimeId()])}\")"
            if not node:
                node1 = Node(nodename)
            else:
                node1 = Node(nodename, parent=node)
            eles = ele.GetChildren()
            for ele1 in eles:
                findall(ele1, n+1, node1)
            return node1
        tree = RenderTree(findall(ele))
        for pre, fill, node in tree:
            print(f"{pre}{node.name}")
except:
    pass

def now_time(fmt='%Y%m%d%H%M%S%f'):
    return datetime.now().strftime(fmt)

def FindPid(process_name):
    procs = psutil.process_iter(['pid', 'name'])
    for proc in procs:
        if process_name in proc.info['name']:
            return proc.info['pid']
        
def OpenWeChat(wxpath=None):
    if wxpath is None:
        registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Tencent\WeChat", 0, winreg.KEY_READ)
        path, _ = winreg.QueryValueEx(registry_key, "InstallPath")
        winreg.CloseKey(registry_key)
        wxpath = os.path.join(path, "WeChat.exe")
    if os.path.exists(wxpath):
        os.system(f'start "" "{wxpath}"')

def transver(ver):
    vers = ver.split('.')
    res = '6' + vers[0]
    for v in vers[1:]:
        res += hex(int(v))[2:].rjust(2, '0')
    return int(res, 16)

def Mver(pid):
    exepath = psutil.Process(pid).exe()
    if GetVersionByPath(exepath) != ORIVERSION:
        Warning(f"该修复方法仅适用于版本号为{ORIVERSION}的微信！")
        return
    if not uia.Control(ClassName='WeChatLoginWndForPC', searchDepth=1).Exists(maxSearchSeconds=2):
        Warning("请先打开微信启动页面再次尝试运行该方法！")
        return
    path = os.path.join(os.path.dirname(__file__), 'a.dll')
    dll = ctypes.WinDLL(path)
    dll.GetDllBaseAddress.argtypes = [ctypes.c_uint, ctypes.c_wchar_p]
    dll.GetDllBaseAddress.restype = ctypes.c_void_p
    dll.WriteMemory.argtypes = [ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong]
    dll.WriteMemory.restype = ctypes.c_bool
    dll.GetMemory.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
    dll.GetMemory.restype = ctypes.c_ulong
    mname = 'WeChatWin.dll'
    tar = transver(VERSION)
    base_address = dll.GetDllBaseAddress(pid, mname)
    address = base_address + 64761648
    if dll.GetMemory(pid, address) != tar:
        dll.WriteMemory(pid, address, tar)
    handle = ctypes.c_void_p(dll._handle)
    ctypes.windll.kernel32.FreeLibrary(handle)

def FixVersionError():
    """修复版本低无法登录的问题"""
    pid = FindPid('WeChat.exe')
    if pid:
        Mver(pid)
        return
    else:
        try:
            registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Tencent\WeChat", 0, winreg.KEY_READ)
            path, _ = winreg.QueryValueEx(registry_key, "InstallPath")
            winreg.CloseKey(registry_key)
            wxpath = os.path.join(path, "WeChat.exe")
            if os.path.exists(wxpath):
                os.system(f'start "" "{wxpath}"')
                time.sleep(1.5)
                FixVersionError()
            else:
                raise Exception('nof found')
        except WindowsError:
            Warning("未找到微信安装路径，请先打开微信启动页面再次尝试运行该方法！")

def GetAllControlList(ele):
    def findall(ele, n=0, text=[]):
        if ele.Name:
            text.append(ele)
        eles = ele.GetChildren()
        for ele1 in eles:
            text = findall(ele1, n+1, text)
        return text
    text_list = findall(ele)
    return text_list

def GetAllControl(ele):
    def findall(ele, n=0, controls=[]):
        # if ele.Name:
        controls.append(ele)
        eles = ele.GetChildren()
        for ele1 in eles:
            controls = findall(ele1, n+1, controls)
        return controls
    text_list = findall(ele)[1:]
    return text_list

def SetClipboardFiles(paths):
    for file in paths:
        if not os.path.exists(file):
            raise FileNotFoundError(f"file ({file}) not exists!")
    files = ("\0".join(paths)).replace("/", "\\")
    data = files.encode("U16")[2:]+b"\0\0"
    t0 = time.time()
    while True:
        if time.time() - t0 > 10:
            raise TimeoutError(f"设置剪贴板文件超时！ --> {paths}")
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_HDROP, matedata+data)
            break
        except:
            pass
        finally:
            try:
                win32clipboard.CloseClipboard()
            except:
                pass

def PasteFile(folder):
    folder = os.path.realpath(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)

    t0 = time.time()
    while True:
        if time.time() - t0 > 10:
            raise TimeoutError(f"读取剪贴板文件超时！")
        try:
            win32clipboard.OpenClipboard()
            if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_HDROP):
                files = win32clipboard.GetClipboardData(win32clipboard.CF_HDROP)
                for file in files:
                    filename = os.path.basename(file)
                    dest_file = os.path.join(folder, filename)
                    shutil.copy2(file, dest_file)
                    return True
            else:
                print("剪贴板中没有文件")
                return False
        except:
            pass
        finally:
            win32clipboard.CloseClipboard()

def parse_msg(control: uia.Control):
    if control.ControlTypeName == 'PaneControl':
        return None
    
    info = {}
    # Text
    if control.Name == '':
        info['type'] = 'Text'
        info['sender'] = control.ButtonControl().Name
        info['sender_remark'] = control.GetProgenyControl(4, control_type='TextControl').Name
        info['time'] = control.GetProgenyControl(4, 1, control_type='TextControl').Name
        info['content'] = control.GetProgenyControl(6).Name

    # Image
    elif control.Name == '[图片]':
        info['type'] = 'Image'
        info['sender'] = control.GetProgenyControl(2, control_type='ButtonControl').Name
        info['sender_remark'] = control.GetProgenyControl(4, control_type='TextControl').Name
        info['time'] = control.GetProgenyControl(4, 1, control_type='TextControl').Name
        info['content'] = ""

    # Video
    elif control.Name == '[视频]':
        info['type'] = 'Video'
        info['sender'] = control.GetProgenyControl(2, control_type='ButtonControl').Name
        info['sender_remark'] = control.GetProgenyControl(4, control_type='TextControl').Name
        info['time'] = control.GetProgenyControl(4, 1, control_type='TextControl').Name
        info['content'] = ""

    # Location
    elif control.Name == '[位置]':
        info['type'] = 'Location'
        info['sender'] = control.GetProgenyControl(2, control_type='ButtonControl').Name
        info['sender_remark'] = control.GetProgenyControl(4, control_type='TextControl').Name
        info['time'] = control.GetProgenyControl(4, 1, control_type='TextControl').Name
        info['content'] = control.GetProgenyControl(7, 1, control_type='TextControl').Name + control.GetProgenyControl(7, control_type='TextControl').Name

    # File
    elif control.Name == '[文件]':
        info['type'] = 'File'
        info['sender'] = control.GetProgenyControl(2, control_type='ButtonControl').Name
        info['sender_remark'] = control.GetProgenyControl(4, control_type='TextControl').Name
        info['time'] = control.GetProgenyControl(4, 1, control_type='TextControl').Name
        info['content'] = control.GetProgenyControl(8, control_type='TextControl').Name

    # Voice
    elif control.Name.startswith('[语音]'):
        info['type'] = 'Voice'
        info['sender'] = control.GetProgenyControl(2, control_type='ButtonControl').Name
        info['sender_remark'] = control.GetProgenyControl(4, control_type='TextControl').Name
        info['time'] = control.GetProgenyControl(4, 1, control_type='TextControl').Name
        info['content'] = control.Name

    # Sticker
    elif control.Name == '[动画表情]':
        info['type'] = 'Sticker'
        info['sender'] = control.GetProgenyControl(2, control_type='ButtonControl').Name
        info['sender_remark'] = control.GetProgenyControl(4, control_type='TextControl').Name
        info['time'] = control.GetProgenyControl(4, 1, control_type='TextControl').Name
        info['content'] = ""

    # VideoChannel
    elif control.Name == '[视频号]':
        info['type'] = 'VideoChannel'
        info['source'] = control.GetProgenyControl(8, -1, control_type='TextControl').Name
        info['sender'] = control.GetProgenyControl(2, control_type='ButtonControl').Name
        info['sender_remark'] = control.GetProgenyControl(4, control_type='TextControl').Name
        info['time'] = control.GetProgenyControl(4, 1, control_type='TextControl').Name
        info['content'] = control.Name

    # Link
    elif control.Name == '链接':
        info['type'] = 'Link'
        info['describe'] = control.GetProgenyControl(7, control_type='TextControl').Name
        info['sender'] = control.GetProgenyControl(2, control_type='ButtonControl').Name
        info['sender_remark'] = control.GetProgenyControl(4, control_type='TextControl').Name
        info['time'] = control.GetProgenyControl(4, 1, control_type='TextControl').Name
        info['content'] = control.GetProgenyControl(6, control_type='TextControl').Name

    # Music
    elif control.Name == '[音乐]':
        info['type'] = 'Music'
        info['describe'] = control.GetProgenyControl(8, -1, control_type='TextControl').Name
        info['sender'] = control.GetProgenyControl(2, control_type='ButtonControl').Name
        info['sender_remark'] = control.GetProgenyControl(4, control_type='TextControl').Name
        info['time'] = control.GetProgenyControl(4, 1, control_type='TextControl').Name
        info['content'] = control.GetProgenyControl(8, control_type='TextControl').Name

    else:
        # MiniProgram
        tempcontrol = control.GetProgenyControl(7, -1, control_type='TextControl')
        if tempcontrol and tempcontrol.Name == '小程序':
            info['type'] = 'MiniProgram'
            info['source'] = control.GetProgenyControl(7, control_type='TextControl').Name
            info['sender'] = control.GetProgenyControl(2, control_type='ButtonControl').Name
            info['sender_remark'] = control.GetProgenyControl(4, control_type='TextControl').Name
            info['time'] = control.GetProgenyControl(4, 1, control_type='TextControl').Name
            info['content'] = control.Name
    
        # Advertisement

    return info

def GetText(HWND):
    length = win32gui.SendMessage(HWND, win32con.WM_GETTEXTLENGTH)*2
    buffer = win32gui.PyMakeBuffer(length)
    win32api.SendMessage(HWND, win32con.WM_GETTEXT, length, buffer)
    address, length_ = win32gui.PyGetBufferAddressAndLen(buffer[:-1])
    text = win32gui.PyGetString(address, length_)[:int(length/2)]
    buffer.release()
    return text

def GetAllWindowExs(HWND):
    if not HWND:
        return
    handles = []
    win32gui.EnumChildWindows(
        HWND, lambda hwnd, param: param.append([hwnd, win32gui.GetClassName(hwnd), GetText(hwnd)]),  handles)
    return handles

def FindWindow(classname=None, name=None) -> int:
    return win32gui.FindWindow(classname, name)

def FindWinEx(HWND, classname=None, name=None) -> list:
    hwnds_classname = []
    hwnds_name = []
    def find_classname(hwnd, classname):
        classname_ = win32gui.GetClassName(hwnd)
        if classname_ == classname:
            if hwnd not in hwnds_classname:
                hwnds_classname.append(hwnd)
    def find_name(hwnd, name):
        name_ = GetText(hwnd)
        if name in name_:
            if hwnd not in hwnds_name:
                hwnds_name.append(hwnd)
    if classname:
        win32gui.EnumChildWindows(HWND, find_classname, classname)
    if name:
        win32gui.EnumChildWindows(HWND, find_name, name)
    if classname and name:
        hwnds = [hwnd for hwnd in hwnds_classname if hwnd in hwnds_name]
    else:
        hwnds = hwnds_classname + hwnds_name
    return hwnds

def ClipboardFormats(unit=0, *units):
    units = list(units)
    retry_count = 5
    while retry_count > 0:
        try:
            win32clipboard.OpenClipboard()
            try:
                u = win32clipboard.EnumClipboardFormats(unit)
            finally:
                win32clipboard.CloseClipboard()
            break
        except Exception as e:
            retry_count -= 1
    units.append(u)
    if u:
        units = ClipboardFormats(u, *units)
    return units

def ReadClipboardData():
    Dict = {}
    formats = ClipboardFormats()

    for i in formats:
        if i == 0:
            continue

        retry_count = 5
        while retry_count > 0:
            try:
                win32clipboard.OpenClipboard()
                try:
                    data = win32clipboard.GetClipboardData(i)
                    Dict[str(i)] = data
                finally:
                    win32clipboard.CloseClipboard()
                break
            except Exception as e:
                retry_count -= 1
    return Dict

# def ReadClipboardData():
#     Dict = {}
#     for i in ClipboardFormats():
#         if i == 0:
#             continue
#         win32clipboard.OpenClipboard()
#         try:
#             filenames = win32clipboard.GetClipboardData(i)
#             win32clipboard.CloseClipboard()
#         except:
#             win32clipboard.CloseClipboard()
#             raise ValueError
#         Dict[str(i)] = filenames
#     return Dict

def ParseWeChatTime(time_str):
    """
    时间格式转换函数

    Args:
        time_str: 输入的时间字符串

    Returns:
        转换后的时间字符串
    """
    time_str = time_str.replace('星期天', '星期日')
    match = re.match(r'^(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$', time_str)
    if match:
        month, day, hour, minute, second = match.groups()
        current_year = datetime.now().year
        return datetime(current_year, int(month), int(day), int(hour), int(minute), int(second)).strftime('%Y-%m-%d %H:%M:%S')
    
    match = re.match(r'^(\d{1,2}):(\d{1,2})$', time_str)
    if match:
        hour, minute = match.groups()
        return datetime.now().strftime('%Y-%m-%d') + f' {hour}:{minute}:00'

    match = re.match(r'^昨天 (\d{1,2}):(\d{1,2})$', time_str)
    if match:
        hour, minute = match.groups()
        yesterday = datetime.now() - timedelta(days=1)
        return yesterday.strftime('%Y-%m-%d') + f' {hour}:{minute}:00'

    match = re.match(r'^星期([一二三四五六日]) (\d{1,2}):(\d{1,2})$', time_str)
    if match:
        weekday, hour, minute = match.groups()
        weekday_num = ['一', '二', '三', '四', '五', '六', '日'].index(weekday)
        today_weekday = datetime.now().weekday()
        delta_days = (today_weekday - weekday_num) % 7
        target_day = datetime.now() - timedelta(days=delta_days)
        return target_day.strftime('%Y-%m-%d') + f' {hour}:{minute}:00'

    match = re.match(r'^(\d{4})年(\d{1,2})月(\d{1,2})日 (\d{1,2}):(\d{1,2})$', time_str)
    if match:
        year, month, day, hour, minute = match.groups()
        return datetime(*[int(i) for i in [year, month, day, hour, minute]]).strftime('%Y-%m-%d %H:%M:%S')
    
    match = re.match(r'^(\d{2})-(\d{2}) (上午|下午) (\d{1,2}):(\d{2})$', time_str)
    if match:
        month, day, period, hour, minute = match.groups()
        current_year = datetime.now().year
        hour = int(hour)
        if period == '下午' and hour != 12:
            hour += 12
        elif period == '上午' and hour == 12:
            hour = 0
        return datetime(current_year, int(month), int(day), hour, int(minute)).strftime('%Y-%m-%d %H:%M:%S')

    return time_str

def RollIntoView(win, ele, equal=False, bias=0):
    while ele.BoundingRectangle.ycenter() < win.BoundingRectangle.top + bias or ele.BoundingRectangle.ycenter() >= win.BoundingRectangle.bottom - bias:
        if ele.BoundingRectangle.ycenter() < win.BoundingRectangle.top + bias:
            # 上滚动
            while True:
                if not ele.Exists(0):
                    return 'not exist'
                win.WheelUp(wheelTimes=1)
                time.sleep(0.1)
                if equal:
                    if ele.BoundingRectangle.ycenter() >= win.BoundingRectangle.top + bias:
                        break
                else:
                    if ele.BoundingRectangle.ycenter() > win.BoundingRectangle.top + bias:
                        break

        elif ele.BoundingRectangle.ycenter() >= win.BoundingRectangle.bottom - bias:
            # 下滚动
            while True:
                if not ele.Exists(0):
                    return 'not exist'
                win.WheelDown(wheelTimes=1)
                time.sleep(0.1)
                if equal:
                    if ele.BoundingRectangle.ycenter() <= win.BoundingRectangle.bottom - bias:
                        break
                else:
                    if ele.BoundingRectangle.ycenter() < win.BoundingRectangle.bottom - bias:
                        break
        time.sleep(0.3)


def ensure_agreement():
    # 定义协议文件和确认状态的路径
    # agreement_file = "AGREEMENT"  # 确保文件与模块放在同一目录
    # agreement_file_path = os.path.join(os.path.dirname(__file__), agreement_file)
    confirmation_folder = os.path.join(os.getenv("USERPROFILE"), ".wxauto")
    confirmation_file = os.path.join(confirmation_folder, "WXAUTO_AGREED")

    # 检查是否已确认协议
    if not os.path.exists(confirmation_folder):
        os.makedirs(confirmation_folder)

    if os.path.exists(confirmation_file):
        with open(confirmation_file, "r", encoding="utf-8") as file:
            now_agreement_hash = file.read()
        
        if now_agreement_hash == AGREEMENTHASH:
            return 
        else:
            print("协议文件已更新，重新确认协议。")

    # 读取协议内容
    # if not os.path.exists(agreement_file_path):
    #     raise FileNotFoundError(f"协议文件不存在。请联系开发者。")

    # with open(agreement_file_path, "r", encoding="utf-8") as file:
    #     agreement_content = file.read()

    agreementhash = hashlib.sha256(AGREEMENT_CONTENT.encode("utf-8")).hexdigest()
    if agreementhash != AGREEMENTHASH[:64]:
        raise ValueError(f"协议文件已被篡改。请联系开发者。")
    
    print("=== 请阅读以下协议，您使用本项目即视为您已接受并同意遵守本协议。 ===\n")
    print(AGREEMENT_CONTENT)
    
    with open(confirmation_file, "w", encoding="utf-8") as file:
        file.write(AGREEMENTHASH)

    # # 显示协议并要求用户确认
    # attempts = 0
    # max_attempts = 5
    # while attempts < max_attempts:
    #     print("=== 请阅读并确认以下协议 ===\n")
    #     print(agreement_content)
    #     print("\n输入回车键以表示您已阅读并同意协议。")

    #     user_input = input("请输入您的选择：").strip()
    #     if user_input == "":
    #         # 用户确认协议
    #         with open(confirmation_file, "w", encoding="utf-8") as file:
    #             file.write(AGREEMENTHASH)
    #         print("协议已确认，感谢您的配合。")
    #         return
    #     else:
    #         attempts += 1
    #         print(f"输入无效，您还有 {max_attempts - attempts} 次机会。\n")

    # # 超过最大尝试次数，自动卸载模块
    # print("您未能正确确认协议，将卸载模块 wxautox。")
    # subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "wxautox"])
    # sys.exit("由于协议未被确认，wxautox 模块已被卸载。")

# 在模块首次导入时调用
ensure_agreement()
# def RollIntoView(win, ele, equal=False):
#     while ele.BoundingRectangle.ycenter() < win.BoundingRectangle.top or ele.BoundingRectangle.ycenter() >= win.BoundingRectangle.bottom:
#         if ele.BoundingRectangle.ycenter() < win.BoundingRectangle.top:
#             # 上滚动
#             while True:
#                 win.WheelUp(wheelTimes=1)
#                 time.sleep(0.1)
#                 if equal:
#                     if ele.BoundingRectangle.ycenter() >= win.BoundingRectangle.top:
#                         break
#                 else:
#                     if ele.BoundingRectangle.ycenter() > win.BoundingRectangle.top:
#                         break

#         elif ele.BoundingRectangle.ycenter() >= win.BoundingRectangle.bottom:
#             # 下滚动
#             while True:
#                 win.WheelDown(wheelTimes=1)
#                 time.sleep(0.1)
#                 if equal:
#                     if ele.BoundingRectangle.ycenter() <= win.BoundingRectangle.bottom:
#                         break
#                 else:
#                     if ele.BoundingRectangle.ycenter() < win.BoundingRectangle.bottom:
#                         break
#         time.sleep(0.3)

wxlog = logging.getLogger('wxautox')
wxlog.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# file_handler = logging.FileHandler(
#     filename='wxautox.log',        # 日志文件名
#     mode='a',                  # 追加模式（默认）
#     encoding='utf-8'           # 文件编码
# )
# file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s')
console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)
wxlog.addHandler(console_handler)
# wxlog.addHandler(file_handler)
wxlog.propagate = False

def set_debug(debug: bool):
    if debug:
        wxlog.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
    else:
        wxlog.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
