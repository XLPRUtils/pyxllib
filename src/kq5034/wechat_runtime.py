"""微信桌面运行时相关实现。"""

import json
import subprocess
import threading
import ctypes
from ctypes import wintypes

from .common import *  # noqa: F403
from pyxllib.prog import process_runtime
from wxautox.utils import RollIntoView


_微信主窗口类名 = {'WeChatMainWndForPC', 'WeChatLoginWndForPC'}
_微信二级窗口类名 = {
    'ImagePreviewWnd',
    'ChatWnd',
    'ChatRecordWnd',
    'ChatRoomAnnouncementWnd',
    'ContactProfileWnd',
    'FileListMgrWnd',
    'MsgFileWnd',
    'SelectContactWnd',
    'SessionChatRoomDetailWnd',
    'SnsWnd',
}
_微信二级窗口名称 = {'微信支付商家助手', '商家助手'}
_微信支付商家助手窗口标题 = {'微信支付商家助手', '商家助手'}


class _Win32RectProxy:
    def __init__(self, rect):
        self.left, self.top, self.right, self.bottom = rect


class _Win32HelperControl:
    """微信小程序宿主窗口的轻量控件代理，避免 UIA 枚举 Chromium 子树卡死。"""

    ControlTypeName = 'PaneControl'
    AutomationId = ''

    def __init__(self, item):
        self._item = dict(item)
        self.Name = str(item.get('title') or '微信支付商家助手')
        self.ClassName = str(item.get('class') or 'Chrome_WidgetWin_0')
        self.NativeWindowHandle = item.get('hwnd')
        self.ProcessId = item.get('pid')

    @property
    def BoundingRectangle(self):
        return _Win32RectProxy(self._item.get('rect') or [0, 0, 0, 0])

    def GetRuntimeId(self):
        return [int(self.NativeWindowHandle or 0)]

    def GetChildren(self):
        return []

    def activate(self):
        hwnd = self.NativeWindowHandle
        if not hwnd:
            return
        user32 = ctypes.windll.user32
        try:
            user32.ShowWindow(wintypes.HWND(hwnd), 9)
            user32.SetForegroundWindow(wintypes.HWND(hwnd))
        except Exception as exc:
            logger.warning(f'激活微信支付商家助手Win32窗口失败：hwnd={hwnd} err={exc!r}')


def _微信二维码诊断目录(stage):
    safe_stage = re.sub(r'[^0-9A-Za-z_\-\u4e00-\u9fff]+', '_', str(stage or 'unknown')).strip('_')
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = Path(tempfile.gettempdir()) / 'codeyun' / 'kq5034_wechat_qrcode' / f'{ts}_{safe_stage}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _控件摘要(ctrl):
    try:
        rect = ctrl.BoundingRectangle
        ltrb = [rect.left, rect.top, rect.right, rect.bottom]
    except Exception:
        ltrb = None
    parts = []
    for attr in ('Name', 'ControlTypeName', 'ClassName', 'AutomationId'):
        try:
            value = getattr(ctrl, attr, '')
        except Exception:
            value = ''
        if value:
            parts.append(f'{attr}={value!r}')
    parts.append(f'BoundingRectangle={ltrb!r}')
    return ' '.join(parts)


def _控件属性(ctrl):
    data = {}
    for attr in ('Name', 'ControlTypeName', 'ClassName', 'AutomationId', 'NativeWindowHandle', 'ProcessId'):
        try:
            value = getattr(ctrl, attr, '')
        except Exception:
            value = ''
        data[attr] = value
    return data


def _微信顶层窗口角色(data, *, main_process_ids=None):
    main_process_ids = {x for x in (main_process_ids or set()) if x}
    name = str(data.get('Name') or '')
    class_name = str(data.get('ClassName') or '')
    process_id = data.get('ProcessId')

    if class_name in _微信主窗口类名:
        return 'main'
    if name in _微信二级窗口名称:
        return 'secondary:name'
    if class_name in _微信二级窗口类名:
        return 'secondary:class'
    # 微信内置浏览器/小程序常见顶层窗口。只处理标题明确为“微信”的窗口，
    # 避免误关用户正常打开的 Chrome 或其他 Chromium 程序。
    if class_name == 'Chrome_WidgetWin_0' and name == '微信':
        return 'secondary:wechat-browser'
    if process_id in main_process_ids:
        return 'secondary:same-process'
    return ''


def _枚举Win32顶层窗口():
    user32 = ctypes.windll.user32
    enum_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    class Rect(ctypes.Structure):
        _fields_ = [
            ('left', ctypes.c_long),
            ('top', ctypes.c_long),
            ('right', ctypes.c_long),
            ('bottom', ctypes.c_long),
        ]

    def window_text(hwnd):
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value

    def class_name(hwnd):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

    rows = []

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        rect = Rect()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        rows.append({
            'hwnd': int(hwnd),
            'title': window_text(hwnd),
            'class': class_name(hwnd),
            'pid': int(pid.value),
            'rect': [rect.left, rect.top, rect.right, rect.bottom],
        })
        return True

    user32.EnumWindows(enum_proc_type(callback), 0)
    return rows


def _关闭微信支付商家助手Win32(*, timeout_ms=3000):
    """关闭微信小程序宿主窗口；这类窗口有时不在微信主进程内，UIA 同进程兜底抓不到。"""
    user32 = ctypes.windll.user32
    wm_close = 0x0010
    smto_abort_if_hung = 0x0002
    closed = []
    for item in _枚举Win32顶层窗口():
        title = str(item.get('title') or '')
        class_name = str(item.get('class') or '')
        if class_name != 'Chrome_WidgetWin_0':
            continue
        if title not in _微信支付商家助手窗口标题 and '商家助手' not in title:
            continue
        result = ctypes.c_size_t()
        ok = user32.SendMessageTimeoutW(
            wintypes.HWND(item['hwnd']),
            wm_close,
            0,
            0,
            smto_abort_if_hung,
            max(300, int(timeout_ms)),
            ctypes.byref(result),
        )
        record = dict(item)
        record['send_message_timeout_ok'] = bool(ok)
        closed.append(record)
    if closed:
        logger.info(f'Win32已请求关闭微信支付商家助手窗口：{closed}')
    return closed


def _微信二维码链路Win32角色(item):
    title = str(item.get('title') or '')
    class_name = str(item.get('class') or '')

    if class_name in _微信主窗口类名:
        return ''
    if class_name in _微信二级窗口类名:
        return f'secondary:{class_name}'
    if class_name == 'Chrome_WidgetWin_0':
        if title in _微信支付商家助手窗口标题 or '商家助手' in title:
            return 'secondary:wechat-pay-helper'
        if title == '微信':
            return 'secondary:wechat-browser'
    return ''


def _微信支付商家助手Win32窗口():
    helpers = [
        item for item in _枚举Win32顶层窗口()
        if _微信二维码链路Win32角色(item) == 'secondary:wechat-pay-helper'
    ]
    helpers.sort(
        key=lambda item: (
            (item['rect'][2] - item['rect'][0]) * (item['rect'][3] - item['rect'][1]),
            item.get('hwnd') or 0,
        ),
        reverse=True,
    )
    return helpers[0] if helpers else None


def _关闭微信二维码链路Win32(*, timeout_ms=1000):
    """用 Win32 快速关闭二维码识别链路窗口，避免 UIA 枚举卡住登录主流程。"""
    user32 = ctypes.windll.user32
    wm_close = 0x0010
    smto_abort_if_hung = 0x0002
    closed = []
    for item in _枚举Win32顶层窗口():
        role = _微信二维码链路Win32角色(item)
        if not role:
            continue
        result = ctypes.c_size_t()
        ok = user32.SendMessageTimeoutW(
            wintypes.HWND(item['hwnd']),
            wm_close,
            0,
            0,
            smto_abort_if_hung,
            max(300, int(timeout_ms)),
            ctypes.byref(result),
        )
        record = dict(item)
        record['role'] = role
        record['send_message_timeout_ok'] = bool(ok)
        closed.append(record)
    if closed:
        logger.info(f'Win32已请求关闭微信二维码链路窗口：{closed}')
    return closed


def _关闭微信支付商家助手窗口(ctrl, *, timeout_ms=1000):
    """关闭当前微信支付商家助手窗口；成功页没有业务价值，不能残留占用后续扫码。"""
    try:
        hwnd = int(getattr(ctrl, 'NativeWindowHandle', 0) or 0)
    except Exception:
        hwnd = 0
    if hwnd:
        user32 = ctypes.windll.user32
        wm_close = 0x0010
        smto_abort_if_hung = 0x0002
        result = ctypes.c_size_t()
        ok = user32.SendMessageTimeoutW(
            wintypes.HWND(hwnd),
            wm_close,
            0,
            0,
            smto_abort_if_hung,
            max(300, int(timeout_ms)),
            ctypes.byref(result),
        )
        logger.info(f'微信支付扫码登录：已请求关闭商家助手窗口 hwnd={hwnd} ok={bool(ok)}')
        return bool(ok)
    try:
        ctrl.Close()
        logger.info('微信支付扫码登录：已通过UIA关闭商家助手窗口')
        return True
    except Exception as exc:
        logger.warning(f'微信支付扫码登录：关闭商家助手窗口失败，改为关闭二维码链路窗口：{exc!r}')
        return bool(_关闭微信二维码链路Win32(timeout_ms=timeout_ms))


def _关闭微信图片预览Win32(*, timeout_ms=1000):
    """只关闭二维码图片预览窗口，保留已经拉起的商家助手小程序。"""
    user32 = ctypes.windll.user32
    wm_close = 0x0010
    smto_abort_if_hung = 0x0002
    closed = []
    for item in _枚举Win32顶层窗口():
        if str(item.get('class') or '') != 'ImagePreviewWnd':
            continue
        result = ctypes.c_size_t()
        ok = user32.SendMessageTimeoutW(
            wintypes.HWND(item['hwnd']),
            wm_close,
            0,
            0,
            smto_abort_if_hung,
            max(300, int(timeout_ms)),
            ctypes.byref(result),
        )
        record = dict(item)
        record['send_message_timeout_ok'] = bool(ok)
        closed.append(record)
    if closed:
        logger.info(f'Win32已请求关闭微信二维码图片预览窗口：{closed}')
    return closed


def _存在微信支付商家助手Win32():
    return _微信支付商家助手Win32窗口() is not None


def _微信图片预览Win32窗口():
    previews = [
        item for item in _枚举Win32顶层窗口()
        if str(item.get('class') or '') == 'ImagePreviewWnd'
    ]
    previews.sort(
        key=lambda item: (
            (item['rect'][2] - item['rect'][0]) * (item['rect'][3] - item['rect'][1]),
            item.get('hwnd') or 0,
        ),
        reverse=True,
    )
    return previews[0] if previews else None


def _快速重置微信二维码窗口状态(*, close_seconds=3, timeout_ms=1000):
    """只用 Win32 关闭二维码图片预览、小程序/商家助手等窗口，保留微信主窗口。"""
    deadline = time.time() + max(0.5, float(close_seconds))
    closed = []
    remaining = []
    while True:
        closed.extend(_关闭微信二维码链路Win32(timeout_ms=timeout_ms))
        time.sleep(0.15)
        remaining = []
        for item in _枚举Win32顶层窗口():
            role = _微信二维码链路Win32角色(item)
            if role:
                record = dict(item)
                record['role'] = role
                remaining.append(record)
        if not remaining or time.time() >= deadline:
            break

    kept = [
        item for item in _枚举Win32顶层窗口()
        if str(item.get('class') or '') in _微信主窗口类名
    ]
    result = {
        'closed_count': len(closed),
        'closed': closed,
        'kept_count': len(kept),
        'kept': kept,
        'remaining_count': len(remaining),
        'remaining': remaining,
    }
    if closed or remaining:
        logger.info(
            '微信二维码窗口状态快速重置：'
            f'closed={len(closed)} remaining={len(remaining)} kept={len(kept)}'
        )
    return result


def _列出微信顶层窗口():
    changed_timeout = False
    try:
        try:
            uia.SetGlobalSearchTimeout(1)
            changed_timeout = True
        except Exception:
            pass
        root = uia.GetRootControl()
        controls = list(root.GetChildren())
    except Exception as exc:
        logger.warning(f'列出微信顶层窗口失败：{exc!r}')
        return []
    finally:
        if changed_timeout:
            try:
                uia.SetGlobalSearchTimeout(10)
            except Exception:
                pass

    rows = []
    main_process_ids = set()
    for ctrl in controls:
        try:
            data = _控件属性(ctrl)
            data['control'] = ctrl
            rows.append(data)
            if str(data.get('ClassName') or '') in _微信主窗口类名 and data.get('ProcessId'):
                main_process_ids.add(data.get('ProcessId'))
        except Exception as exc:
            logger.warning(f'读取顶层窗口属性失败：{exc!r}')

    windows = []
    for data in rows:
        try:
            role = _微信顶层窗口角色(data, main_process_ids=main_process_ids)
            if not role:
                continue
            data['role'] = role
            data['summary'] = _控件摘要(data['control'])
            windows.append(data)
        except Exception as exc:
            logger.warning(f'识别微信顶层窗口失败：{exc!r}')
    return windows


def _关闭微信二级窗口(ctrl, *, wait=0.5):
    try:
        ctrl.Close(wait)
        return True, ''
    except TypeError:
        try:
            ctrl.Close()
            time.sleep(wait)
            return True, ''
        except Exception as exc:
            close_error = exc
    except Exception as exc:
        close_error = exc

    try:
        ctrl.SetActive(0.1)
    except Exception:
        pass
    try:
        ctrl.SendKeys('{Esc}')
        time.sleep(wait)
        return True, f'Close失败后已发送Esc：{close_error!r}'
    except Exception as exc:
        return False, f'Close失败：{close_error!r}；Esc失败：{exc!r}'


def _重置微信二维码窗口状态(*, close_seconds=3):
    """只保留微信主窗口，关闭图片预览、商家助手等二维码链路二级窗口。"""
    deadline = time.time() + max(0.5, float(close_seconds))
    win32_closed = _关闭微信二维码链路Win32()
    closed = []
    errors = []
    kept = []
    while time.time() < deadline:
        windows = _列出微信顶层窗口()
        secondaries = [item for item in windows if str(item.get('role', '')).startswith('secondary')]
        kept = [item for item in windows if item.get('role') == 'main']
        if not secondaries:
            break
        for item in secondaries:
            ctrl = item.pop('control', None)
            if ctrl is None:
                continue
            ok, detail = _关闭微信二级窗口(ctrl, wait=0.3)
            record = {key: item.get(key) for key in ('Name', 'ClassName', 'ControlTypeName', 'NativeWindowHandle', 'role', 'summary')}
            if ok:
                if detail:
                    record['detail'] = detail
                closed.append(record)
            else:
                record['error'] = detail
                errors.append(record)
        time.sleep(0.2)

    win32_closed.extend(_关闭微信二维码链路Win32(timeout_ms=1000))
    remain = []
    for item in _列出微信顶层窗口():
        item.pop('control', None)
        remain.append(item)
    result = {
        'win32_closed_count': len(win32_closed),
        'win32_closed': win32_closed,
        'closed_count': len(closed),
        'closed': closed,
        'errors': errors,
        'kept_count': len(kept),
        'remaining': remain,
    }
    if win32_closed or closed or errors:
        logger.info(
            '微信二维码窗口状态重置：'
            f'win32_closed={len(win32_closed)} closed={len(closed)} errors={len(errors)} remaining={len(remain)}'
        )
    return result


def _子控件摘要(ctrl, *, limit=40):
    lines = []
    try:
        children = list(ctrl.GetChildren())
    except Exception as exc:
        return [f'children_error={exc!r}']
    for child in children[:limit]:
        try:
            lines.append(f'  {_控件摘要(child)}')
        except Exception as exc:
            lines.append(f'  <child summary failed: {exc!r}>')
    if len(children) > limit:
        lines.append(f'  ... {len(children) - limit} more children')
    return lines


def _采集微信二维码诊断(stage, err=None, *, include_uia=True):
    """采集微信二维码识别失败时的桌面证据，避免只留下 wxautox 栈。"""
    diag_dir = _微信二维码诊断目录(stage)
    try:
        screenshot = pyautogui.screenshot()
        screenshot.save(diag_dir / 'desktop.png')
    except Exception as exc:
        logger.warning(f'微信二维码诊断截图失败：{exc!r}')

    lines = []
    if err is not None:
        lines.append(f'error={err!r}')

    if not include_uia:
        try:
            (diag_dir / 'uia.txt').write_text('\n'.join(lines), encoding='utf-8')
        except Exception as exc:
            logger.warning(f'微信二维码诊断写入失败：{exc!r}')
        logger.warning(f'微信二维码诊断已采集：{diag_dir}')
        return diag_dir

    try:
        root = uia.GetRootControl()
        lines.append('[top_windows]')
        for ctrl in root.GetChildren():
            try:
                lines.append(_控件摘要(ctrl))
            except Exception as exc:
                lines.append(f'<window summary failed: {exc!r}>')
    except Exception as exc:
        lines.append(f'top_windows_error={exc!r}')

    for label, factory in (
            ('微信', lambda: uia.WindowControl(Name='微信', ClassName='WeChatMainWndForPC', searchDepth=1)),
            ('微信支付商家助手', lambda: uia.PaneControl(Name='微信支付商家助手', searchDepth=1)),
    ):
        try:
            ctrl = factory()
            lines.append(f'[{label}] {_控件摘要(ctrl)}')
            lines.extend(_子控件摘要(ctrl))
        except Exception as exc:
            lines.append(f'{label}_error={exc!r}')

    try:
        image_ctrl = uia.WindowControl(ClassName='ImagePreviewWnd', searchDepth=1)
        lines.append(f'[ImagePreviewWnd] {_控件摘要(image_ctrl)}')
        lines.extend(_子控件摘要(image_ctrl))
    except Exception as exc:
        lines.append(f'ImagePreviewWnd_error={exc!r}')

    try:
        (diag_dir / 'uia.txt').write_text('\n'.join(lines), encoding='utf-8')
    except Exception as exc:
        logger.warning(f'微信二维码诊断写入失败：{exc!r}')

    logger.warning(f'微信二维码诊断已采集：{diag_dir}')
    return diag_dir


def _微信图片二维码按钮候选(image):
    candidates = []
    for attr in ('t_qrcode',):
        try:
            ctrl = getattr(image, attr)
            if ctrl is not None:
                candidates.append((attr, ctrl))
        except Exception:
            pass

    tools_box = getattr(image, 'ToolsBox', None)
    if tools_box is not None:
        try:
            for ctrl in tools_box.GetChildren():
                name = getattr(ctrl, 'Name', '') or ''
                ctrl_type = getattr(ctrl, 'ControlTypeName', '') or ''
                if ctrl_type == 'ButtonControl' and any(key in name for key in ('二维码', 'QR', 'Code')):
                    candidates.append((f'ToolsBox child {name!r}', ctrl))
        except Exception:
            pass

    deduped = []
    seen = set()
    for label, ctrl in candidates:
        try:
            runtime_id = tuple(ctrl.GetRuntimeId())
        except Exception:
            runtime_id = (id(ctrl),)
        if runtime_id in seen:
            continue
        seen.add(runtime_id)
        deduped.append((label, ctrl))
    return deduped


def _控件可用(ctrl):
    try:
        return bool(ctrl.Exists(0.2))
    except Exception:
        try:
            rect = ctrl.BoundingRectangle
            return rect.right > rect.left and rect.bottom > rect.top
        except Exception:
            return False


def _激活控件窗口(ctrl):
    try:
        ctrl.SetActive(0.2)
        return True
    except Exception:
        pass
    try:
        hwnd = getattr(ctrl, 'NativeWindowHandle', None)
        if hwnd:
            user32 = ctypes.windll.user32
            user32.ShowWindow(wintypes.HWND(hwnd), 9)
            user32.SetForegroundWindow(wintypes.HWND(hwnd))
            return True
    except Exception:
        pass
    return False


def _点击控件(ctrl):
    try:
        ctrl.Click(move=False, simulateMove=False, return_pos=False)
        return
    except TypeError:
        ctrl.Click()


def _遍历控件树(root, *, max_nodes=300):
    stack = [root]
    visited = set()
    count = 0
    while stack and count < max_nodes:
        ctrl = stack.pop()
        try:
            runtime_id = tuple(ctrl.GetRuntimeId())
        except Exception:
            runtime_id = (id(ctrl),)
        if runtime_id in visited:
            continue
        visited.add(runtime_id)
        count += 1
        yield ctrl
        try:
            children = list(ctrl.GetChildren())
        except Exception:
            children = []
        stack.extend(reversed(children))


def _点击匹配控件(root, keywords, *, control_types=None):
    """优先按控件语义点击，避免只靠经验坐标。"""
    keywords = [str(x) for x in keywords if x]
    if not keywords:
        return False

    candidates = []
    for ctrl in _遍历控件树(root):
        try:
            name = getattr(ctrl, 'Name', '') or ''
            ctrl_type = getattr(ctrl, 'ControlTypeName', '') or ''
        except Exception:
            continue
        if control_types and ctrl_type not in control_types:
            continue
        if not any(key in name for key in keywords):
            continue
        if not _控件可用(ctrl):
            continue
        candidates.append((len(name), ctrl_type, name, ctrl))

    for _, ctrl_type, name, ctrl in sorted(candidates, key=lambda x: (x[0], x[1], x[2])):
        try:
            logger.info(f'微信支付扫码登录：点击匹配控件 name={name!r} type={ctrl_type!r}')
            _点击控件(ctrl)
            return True
        except Exception as exc:
            logger.warning(f'微信支付扫码登录：点击匹配控件失败 name={name!r} type={ctrl_type!r} err={exc!r}')
    return False


def _点击微信二维码消息(msg):
    """尽量点击消息里的图片本体，避免 wxautox 的头像偏移点击落空。"""
    control = getattr(msg, 'control', None)
    content = str(getattr(msg, 'content', '') or '')
    if control is None:
        msg.click()
        return 'message.click:no-control'

    try:
        RollIntoView(getattr(msg, 'chatbox').ListControl(), control, equal=True)
    except Exception:
        pass
    try:
        _激活控件窗口(control.GetTopLevelControl())
    except Exception:
        _激活控件窗口(control)

    candidates = []
    try:
        for ctrl in _遍历控件树(control, max_nodes=80):
            try:
                name = getattr(ctrl, 'Name', '') or ''
                ctrl_type = getattr(ctrl, 'ControlTypeName', '') or ''
                rect = ctrl.BoundingRectangle
                width = rect.right - rect.left
                height = rect.bottom - rect.top
            except Exception:
                continue
            if width < 40 or height < 40:
                continue
            if ctrl_type in {'ImageControl', 'ButtonControl'}:
                candidates.append((width * height, ctrl_type, name, ctrl))
    except Exception:
        pass

    for _, ctrl_type, name, ctrl in sorted(candidates, key=lambda x: x[0], reverse=True):
        try:
            logger.info(f'微信支付扫码登录：点击消息内图片候选 type={ctrl_type!r} name={name!r} summary={_控件摘要(ctrl)}')
            _点击控件(ctrl)
            return f'inner:{ctrl_type}:{name}'
        except Exception as exc:
            logger.warning(f'微信支付扫码登录：点击消息内图片候选失败 type={ctrl_type!r} name={name!r} err={exc!r}')

    try:
        rect = control.BoundingRectangle
        x = rect.left + int((rect.right - rect.left) * 0.72)
        y = rect.top + int((rect.bottom - rect.top) * 0.5)
        logger.info(f'微信支付扫码登录：使用消息区域坐标兜底打开二维码图片 x={x} y={y} content={content!r}')
        pyautogui.click(x, y)
        return 'coordinate:message-region'
    except Exception as exc:
        logger.warning(f'微信支付扫码登录：消息区域坐标兜底失败 err={exc!r}')

    msg.click()
    return 'message.click:fallback'


def _控件树文本(root, *, max_nodes=300):
    texts = []
    for ctrl in _遍历控件树(root, max_nodes=max_nodes):
        for attr in ('Name', 'Value'):
            try:
                value = getattr(ctrl, attr, '') or ''
            except Exception:
                value = ''
            if value:
                texts.append(str(value))
    return ' '.join(dict.fromkeys(texts))


def _控件矩形(ctrl):
    rect = ctrl.BoundingRectangle
    return [rect.left, rect.top, rect.right, rect.bottom]


def _激活微信支付商家助手窗口(ctrl, *, reason=''):
    try:
        if isinstance(ctrl, _Win32HelperControl):
            ctrl.activate()
        else:
            UiCtrlNode(ctrl, build_depth=1).activate()
    except Exception as exc:
        logger.warning(f'微信支付扫码登录：激活商家助手窗口失败 reason={reason!r} err={exc!r}')


def _点击窗口比例位置(ctrl, x_ratio, y_ratio, *, label):
    _激活微信支付商家助手窗口(ctrl, reason=label)
    ltrb = _控件矩形(ctrl)
    left, top, right, bottom = ltrb
    width = max(1, right - left)
    height = max(1, bottom - top)
    x = left + width * x_ratio
    y = top + height * y_ratio
    logger.info(f'微信支付扫码登录：{label}，按窗口比例点击 x={x:.1f} y={y:.1f} ratio={[x_ratio, y_ratio]} window={ltrb}')
    pyautogui.click(x, y)
    return True


def _ocr_result_payload(result):
    if hasattr(result, 'json'):
        try:
            return result.json.get('res') or {}
        except Exception:
            return {}
    return result if isinstance(result, dict) else {}


def _OCR标注文本(label):
    if isinstance(label, dict):
        return str(label.get('text') or label.get('label') or '')
    text = str(label or '')
    if text.startswith('{'):
        try:
            payload = json.loads(text)
        except Exception:
            return text
        if isinstance(payload, dict):
            return str(payload.get('text') or payload.get('label') or '')
    return text


def _OCR标注框(points):
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except Exception:
        return None
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _微信支付OCR文本框列表(payload):
    payload = _ocr_result_payload(payload)
    rows = []
    for text, box in zip(payload.get('rec_texts') or [], payload.get('rec_boxes') or []):
        try:
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
        except Exception:
            continue
        rows.append({'text': str(text or ''), 'box': [x1, y1, x2, y2]})

    document = payload.get('document') if isinstance(payload.get('document'), dict) else payload
    for shape in document.get('shapes') or []:
        if not isinstance(shape, dict):
            continue
        box = _OCR标注框(shape.get('points') or [])
        if not box:
            continue
        rows.append({'text': _OCR标注文本(shape.get('label')), 'box': box})
    return rows


def _微信支付商家助手截图(left, top, width, height):
    try:
        from pyxllib.autogui.anlib import _screenshot_region

        return _screenshot_region([int(left), int(top), width, height])
    except Exception as exc:
        logger.warning(f'微信支付扫码登录：AnLib多屏截图失败，退回pyautogui截图：{exc!r}')
        return pyautogui.screenshot(region=[int(left), int(top), width, height])


def _微信支付商户行OCR匹配(rows):
    matches = []
    for row in rows:
        text = str(row.get('text') or '').replace(' ', '')
        if not text or not any(key in text for key in ('武陵禅寺客堂', '客堂', '1599622041')):
            continue
        x1, y1, x2, y2 = row['box']
        matches.append({
            'text': text,
            'box': [x1, y1, x2, y2],
            'score': (2 if '1599622041' in text else 0) + (3 if '武陵禅寺客堂' in text else 1 if '客堂' in text else 0),
        })
    if not matches:
        return None
    return sorted(matches, key=lambda item: (-item['score'], item['box'][1], item['box'][0]))[0]


def _微信支付商家助手OCR文本框(ctrl, *, request_timeout=5):
    _激活微信支付商家助手窗口(ctrl, reason='OCR识别微信支付商家助手')
    left, top, right, bottom = _控件矩形(ctrl)
    width = max(1, int(right - left))
    height = max(1, int(bottom - top))
    if width < 200 or height < 200:
        return []
    screenshot = _微信支付商家助手截图(left, top, width, height)
    try:
        from pyxllib.autogui.anlib import get_xlapi

        rows = _微信支付OCR文本框列表(
            get_xlapi().common_ocr(screenshot, request_timeout=request_timeout, request_retries=0)
        )
    except Exception as exc:
        logger.warning(f'微信支付扫码登录：common_ocr识别商家助手失败，尝试本地PaddleOCR：{exc!r}')
        try:
            from pyxllib.ai.ocr import ocr_text

            rows = _微信支付OCR文本框列表(ocr_text(screenshot, model='basic'))
        except Exception as local_exc:
            logger.warning(f'微信支付扫码登录：商家助手本地PaddleOCR失败：{local_exc!r}')
            return []
    for row in rows:
        row['window'] = [left, top, right, bottom]
        row['width'] = width
        row['height'] = height
    return rows


def _微信支付商户行OCR文本框(ctrl):
    _激活微信支付商家助手窗口(ctrl, reason='OCR识别微信支付商户行')
    left, top, right, bottom = _控件矩形(ctrl)
    width = max(1, int(right - left))
    height = max(1, int(bottom - top))
    if width < 200 or height < 200:
        return None

    raw_timeout = os.getenv('KQ_WECHAT_PAY_MERCHANT_OCR_TIMEOUT_SECONDS', '8')
    try:
        timeout = min(15.0, max(2.0, float(raw_timeout)))
    except (TypeError, ValueError):
        timeout = 8.0

    last_texts = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            screenshot = _微信支付商家助手截图(left, top, width, height)
        except Exception as exc:
            logger.warning(f'微信支付扫码登录：商户选择页截图失败，转入坐标兜底：{exc!r}')
            return None

        try:
            from pyxllib.autogui.anlib import get_xlapi

            rows = _微信支付OCR文本框列表(get_xlapi().common_ocr(screenshot, request_timeout=5, request_retries=0))
        except Exception as exc:
            logger.warning(f'微信支付扫码登录：common_ocr识别商户选择页失败，尝试本地PaddleOCR：{exc!r}')
            rows = []

        last_texts = [row['text'] for row in rows if row.get('text')]
        best = _微信支付商户行OCR匹配(rows)
        if best:
            logger.info(f"微信支付扫码登录：OCR命中商户行 text={best['text']!r} box={best['box']!r}")
            return {
                'window': [left, top, right, bottom],
                'width': width,
                'height': height,
                **best,
            }

        time.sleep(0.5)

    try:
        screenshot = _微信支付商家助手截图(left, top, width, height)
        from pyxllib.ai.ocr import ocr_text

        rows = _微信支付OCR文本框列表(ocr_text(screenshot, model='basic'))
        last_texts = [row['text'] for row in rows if row.get('text')]
        best = _微信支付商户行OCR匹配(rows)
        if best:
            logger.info(f"微信支付扫码登录：本地PaddleOCR命中商户行 text={best['text']!r} box={best['box']!r}")
            return {
                'window': [left, top, right, bottom],
                'width': width,
                'height': height,
                **best,
            }
    except Exception as exc:
        logger.warning(f'微信支付扫码登录：商户选择页本地PaddleOCR失败，转入坐标兜底：{exc!r}')

    if last_texts:
        logger.info(f'微信支付扫码登录：OCR等待超时仍未命中客堂商户行，识别文本={last_texts[:12]!r}')
    return None

def _点击微信支付商户行OCR(ctrl):
    match = _微信支付商户行OCR文本框(ctrl)
    if not match:
        return False
    left, top, right, bottom = match['window']
    x1, y1, x2, y2 = match['box']
    click_x = left + (x1 + x2) / 2
    click_y = top + (y1 + y2) / 2
    logger.info(
        f'微信支付扫码登录：按OCR商户行文本中心点击 '
        f'x={click_x:.1f} y={click_y:.1f} box={match["box"]!r} text={match["text"]!r}'
    )
    pyautogui.click(click_x, click_y)
    return True


def _点击微信支付商户行(ctrl):
    """选择“武陵禅寺客堂 / 1599622041”商户。

    微信小程序窗口有时只能拿到 Win32 顶层代理，UIA 子树为空；此时必须
    明确点击商户选择页的卡片行，不能把该状态当成扫码失败后重试二维码。
    """
    if not isinstance(ctrl, _Win32HelperControl):
        if _点击匹配控件(ctrl, ['1599622041', '武陵禅寺客堂']):
            return True
    if _点击微信支付商户行OCR(ctrl):
        return True
    if isinstance(ctrl, _Win32HelperControl):
        # OCR失败时才使用兜底坐标；正常路径必须由当前真实窗口OCR决定点击点。
        if _点击窗口比例位置(ctrl, 0.22, 0.335, label='选择武陵禅寺客堂商户行(Win32 OCR失败兜底)'):
            return True
    return _点击窗口比例位置(ctrl, 0.28, 0.345, label='选择武陵禅寺客堂商户行')


def _点击微信支付确认区域(ctrl):
    return _点击窗口比例位置(ctrl, 0.5, 0.79, label='确认登录区域')


def _点击微信支付确认OCR(ctrl):
    rows = _微信支付商家助手OCR文本框(ctrl, request_timeout=3)
    texts = [str(row.get('text') or '') for row in rows if row.get('text')]
    for row in rows:
        text = str(row.get('text') or '').replace(' ', '')
        if not any(key in text for key in ('确认登录', '允许登录', '同意登录', '登录', '允许')):
            continue
        x1, y1, x2, y2 = row['box']
        left, top, _right, _bottom = row['window']
        click_x = left + (x1 + x2) / 2
        click_y = top + (y1 + y2) / 2
        logger.info(f'微信支付扫码登录：OCR命中确认按钮 text={text!r} x={click_x:.1f} y={click_y:.1f}')
        pyautogui.click(click_x, click_y)
        return True, texts
    return False, texts


def _点击微信支付完成OCR(ctrl):
    rows = _微信支付商家助手OCR文本框(ctrl, request_timeout=3)
    texts = [str(row.get('text') or '') for row in rows if row.get('text')]
    has_success = any('登录成功' in text.replace(' ', '') for text in texts)
    for row in rows:
        text = str(row.get('text') or '').replace(' ', '')
        if '完成' not in text:
            continue
        x1, y1, x2, y2 = row['box']
        left, top, _right, _bottom = row['window']
        click_x = left + (x1 + x2) / 2
        click_y = top + (y1 + y2) / 2
        logger.info(
            f'微信支付扫码登录：OCR命中完成按钮 text={text!r} '
            f'x={click_x:.1f} y={click_y:.1f} success_text={has_success}'
        )
        pyautogui.click(click_x, click_y)
        return True, texts
    return False, texts


def _等待微信支付商户选择生效():
    raw_seconds = os.getenv('KQ_WECHAT_PAY_MERCHANT_SELECT_SETTLE_SECONDS', '5')
    try:
        seconds = min(30.0, max(3.0, float(raw_seconds)))
    except (TypeError, ValueError):
        seconds = 5.0
    logger.info(f'微信支付扫码登录：商户行已点击，等待页面切换 seconds={seconds}')
    time.sleep(seconds)


def _启动微信二维码诊断看门狗(stage, *, timeout=90):
    def capture():
        try:
            _采集微信二维码诊断(stage)
        except Exception as exc:
            logger.warning(f'微信二维码诊断看门狗采集失败：{exc!r}')

    timer = threading.Timer(timeout, capture)
    timer.daemon = True
    timer.start()
    return timer


def _点击微信图片识别二维码(image, *, timeout=20):
    timeout = min(8, max(3, float(timeout)))
    try:
        fallback_delay = min(3, max(0.8, float(os.getenv('KQ_WECHAT_QRCODE_COORDINATE_FALLBACK_DELAY_SECONDS', '1.2'))))
    except (TypeError, ValueError):
        fallback_delay = 1.2
    started_at = time.time()
    deadline = time.time() + timeout
    last_error = None
    clicked_coordinate_fallback = False
    while time.time() < deadline:
        for label, ctrl in _微信图片二维码按钮候选(image):
            try:
                if not _控件可用(ctrl):
                    continue
                logger.info(f'微信支付扫码登录：命中二维码识别按钮 {label} {_控件摘要(ctrl)}')
                _点击控件(ctrl)
                return
            except Exception as exc:
                last_error = exc
        if not clicked_coordinate_fallback and time.time() - started_at >= fallback_delay:
            try:
                rect = image.api.BoundingRectangle
                width = rect.right - rect.left
                x = rect.left + min(374, max(40, width - 40))
                y = rect.top + 16
                logger.info(f'微信支付扫码登录：使用图片预览工具栏坐标兜底点击识别二维码 x={x} y={y}')
                pyautogui.click(x, y)
                clicked_coordinate_fallback = True
                return
            except Exception as exc:
                last_error = exc
        time.sleep(0.5)

    diag_dir = _采集微信二维码诊断('识别图中二维码按钮失败', last_error)
    raise RuntimeError(f'微信图片未找到“识别图中二维码”按钮，诊断目录：{diag_dir}') from last_error


def _等待微信支付商家助手窗口(*, timeout=45, raise_on_timeout=True):
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        helper = _微信支付商家助手Win32窗口()
        if helper is not None:
            node = _Win32HelperControl(helper)
            _关闭微信图片预览Win32(timeout_ms=500)
            node.activate()
            logger.info(f'微信支付扫码登录：Win32命中商家助手窗口 rect={helper.get("rect")}')
            return node
        time.sleep(0.2)

    if not raise_on_timeout:
        logger.info(f'微信支付商家助手窗口未出现，跳过小程序内点击：last_error={last_error!r}')
        return None
    diag_dir = _采集微信二维码诊断('微信支付商家助手窗口失败', last_error)
    raise RuntimeError(f'微信支付商家助手窗口未出现，诊断目录：{diag_dir}') from last_error


def _等待微信图片预览或商家助手(*, timeout=8, stable_seconds=0.8, raise_on_timeout=True):
    """等待打开二维码后的下一稳定状态：图片预览就绪，或微信已自动拉起商家助手。"""
    timeout = max(2.0, float(timeout))
    stable_seconds = max(0.2, float(stable_seconds))
    deadline = time.time() + timeout
    last_rect = None
    stable_since = None
    last_error = None

    while time.time() < deadline:
        if _存在微信支付商家助手Win32():
            ct1 = _等待微信支付商家助手窗口(timeout=2, raise_on_timeout=False)
            if ct1 is not None:
                logger.info('微信支付扫码登录：打开二维码后已自动出现商家助手窗口')
                return None, ct1

        preview = _微信图片预览Win32窗口()
        if preview is not None:
            rect = preview.get('rect') or [0, 0, 0, 0]
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            if width > 200 and height > 200:
                now = time.time()
                if rect == last_rect:
                    if stable_since is not None and now - stable_since >= stable_seconds:
                        try:
                            image = WeChatImage()
                            logger.info(f'微信支付扫码登录：二维码图片预览已稳定 rect={rect}')
                            return image, None
                        except Exception as exc:
                            last_error = exc
                            logger.warning(f'微信支付扫码登录：图片预览窗口已出现但 WeChatImage 初始化未就绪：{exc!r}')
                else:
                    last_rect = list(rect)
                    stable_since = now

        time.sleep(0.2)

    if not raise_on_timeout:
        return None, None
    diag_dir = _采集微信二维码诊断('微信二维码图片预览未稳定', last_error)
    raise RuntimeError(f'微信二维码图片预览未稳定，诊断目录：{diag_dir}') from last_error


def _打开最近微信二维码图片(wx, *, max_messages=8, open_timeout=4, stable_seconds=0.6, after_text=None):
    messages = wx.GetAllMessage()
    if not messages:
        diag_dir = _采集微信二维码诊断('微信会话无消息')
        raise RuntimeError(f'微信会话没有可点击消息，诊断目录：{diag_dir}')

    marker_index = None
    if after_text:
        needle = str(after_text)
        for index in range(len(messages) - 1, -1, -1):
            content = str(getattr(messages[index], 'content', '') or '')
            if needle in content:
                marker_index = index
                break
        if marker_index is None:
            diag_dir = _采集微信二维码诊断('未找到本轮二维码消息标记', include_uia=False)
            raise RuntimeError(f'微信会话未找到本轮二维码消息标记：{needle!r}，诊断目录：{diag_dir}')
        messages = messages[marker_index + 1:]
        if not messages:
            diag_dir = _采集微信二维码诊断('本轮二维码标记后无图片消息', include_uia=False)
            raise RuntimeError(f'本轮二维码消息标记后没有可点击消息：{needle!r}，诊断目录：{diag_dir}')

    last_error = None
    tried = 0
    for offset, msg in enumerate(reversed(messages[-max(1, int(max_messages)):]), start=1):
        tried += 1
        try:
            click_method = _点击微信二维码消息(msg)
            logger.info(
                '微信支付扫码登录：已尝试打开最近消息 '
                f'offset_from_end={offset} method={click_method!r} content={str(getattr(msg, "content", ""))[:80]!r}'
            )
        except Exception as exc:
            last_error = exc
            logger.warning(f'微信支付扫码登录：点击最近消息失败 offset_from_end={offset} err={exc!r}')
            continue

        image, ct1 = _等待微信图片预览或商家助手(
            timeout=open_timeout,
            stable_seconds=stable_seconds,
            raise_on_timeout=False,
        )
        if image is not None or ct1 is not None:
            return image, ct1, {
                'message_offset_from_end': offset,
                'tried_messages': tried,
                'after_text': after_text,
                'marker_index': marker_index,
            }
        _快速重置微信二维码窗口状态(close_seconds=1)

    diag_dir = _采集微信二维码诊断('最近消息未打开二维码图片', last_error)
    raise RuntimeError(f'最近 {tried} 条消息未打开二维码图片，诊断目录：{diag_dir}') from last_error


def _规范化微信支付商家助手窗口(ctrl):
    """把超出桌面的商家助手窗口拉回当前屏幕，保证相对坐标可点击。"""
    try:
        rect = ctrl.BoundingRectangle
        screen_width, screen_height = pyautogui.size()
        if not (
            rect.left < 0
            or rect.top < 0
            or rect.right > screen_width
            or rect.bottom > screen_height
        ):
            _激活微信支付商家助手窗口(ctrl, reason='窗口已在屏幕内')
            return ctrl
        logger.info(
            '微信支付扫码登录：商家助手窗口超出桌面，先激活并最大化 '
            f'window={[rect.left, rect.top, rect.right, rect.bottom]} screen={[screen_width, screen_height]}'
        )
        _激活微信支付商家助手窗口(ctrl, reason='窗口超出屏幕')
        pyautogui.hotkey('win', 'up')
        time.sleep(2)
        return _等待微信支付商家助手窗口(timeout=5, raise_on_timeout=True)
    except Exception as exc:
        logger.warning(f'微信支付扫码登录：规范化商家助手窗口失败，继续使用当前窗口：{exc!r}')
        return ctrl


class KqWechat:
    @staticmethod
    def 创建微信实例():
        """ wxautox 初始化时会直接 print，某些控制台环境下会触发 stdout flush 异常 """
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return WeChat()

    @staticmethod
    def 诊断微信窗口状态():
        windows = []
        for item in _列出微信顶层窗口():
            item.pop('control', None)
            windows.append(item)
        return {
            'window_count': len(windows),
            'windows': windows,
        }

    @staticmethod
    def 重置微信二维码窗口状态(close_seconds=3):
        return _重置微信二维码窗口状态(close_seconds=close_seconds)

    @staticmethod
    def 快速重置微信二维码窗口状态(close_seconds=3):
        return _快速重置微信二维码窗口状态(close_seconds=close_seconds)

    @staticmethod
    def 单测微信二维码打开识别关闭(user=None, *, repeat=3, assume_current_chat=True):
        raw_timeout = os.getenv('KQ_WECHAT_QRCODE_PROBE_TIMEOUT_SECONDS', '90')
        try:
            timeout = max(20, int(float(raw_timeout)))
        except (TypeError, ValueError):
            timeout = 90

        payload = json.dumps({
            'user': user,
            'repeat': repeat,
            'assume_current_chat': assume_current_chat,
        }, ensure_ascii=False)
        cmd = [
            sys.executable,
            '-c',
            (
                'from kq5034.wechat_runtime import KqWechat; '
                'import json, sys; '
                'kw=json.loads(sys.argv[1]); '
                'print(json.dumps(KqWechat._单测微信二维码打开识别关闭本进程(**kw), ensure_ascii=False))'
            ),
            payload,
        ]
        env = os.environ.copy()
        env.update({
            'PYTHONUTF8': '1',
            'PYTHONIOENCODING': 'utf-8',
        })
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process_runtime.terminate_process_tree(proc.pid, timeout=3)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except OSError:
                    pass
                stdout, stderr = '', ''
            diag_dir = _采集微信二维码诊断('单测_微信二维码打开识别关闭子进程超时', exc, include_uia=False)
            return {
                'status': 'timeout',
                'timeout': timeout,
                'diag_dir': str(diag_dir),
                'stdout_tail': stdout[-2000:],
                'stderr_tail': stderr[-2000:],
            }

        if proc.returncode != 0:
            diag_dir = _采集微信二维码诊断('单测_微信二维码打开识别关闭子进程失败', include_uia=False)
            return {
                'status': 'failed',
                'exit_code': proc.returncode,
                'diag_dir': str(diag_dir),
                'stdout_tail': stdout[-2000:],
                'stderr_tail': stderr[-2000:],
            }

        try:
            result = json.loads(stdout.strip().splitlines()[-1])
        except Exception:
            result = {'status': 'bad_output', 'stdout_tail': stdout[-2000:], 'stderr_tail': stderr[-2000:]}
        if stderr:
            result['stderr_tail'] = stderr[-4000:]
        result.setdefault('status', 'ok')
        return result

    @staticmethod
    def _单测微信二维码打开识别关闭本进程(user=None, *, repeat=3, assume_current_chat=True):
        repeat = min(10, max(1, int(repeat)))
        result = {
            'status': 'ok',
            'user': user,
            'repeat': repeat,
            'rounds': [],
        }
        total_started = time.perf_counter()
        wx = KqWechat.创建微信实例()
        if user:
            wx.ChatWith(user)
        elif not assume_current_chat:
            raise ValueError('user 为空时必须 assume_current_chat=True')

        for index in range(1, repeat + 1):
            row = {'round': index}
            round_started = time.perf_counter()
            image = None
            try:
                t0 = time.perf_counter()
                row['reset_before'] = _快速重置微信二维码窗口状态(close_seconds=2)
                row['reset_before_seconds'] = round(time.perf_counter() - t0, 3)

                if user:
                    t0 = time.perf_counter()
                    wx.ChatWith(user)
                    row['open_chat_seconds'] = round(time.perf_counter() - t0, 3)
                else:
                    try:
                        row['current_chat'] = str(wx.CurrentChat())
                    except Exception as exc:
                        row['current_chat_error'] = repr(exc)

                t0 = time.perf_counter()
                image, ct1, open_meta = _打开最近微信二维码图片(wx)
                row.update(open_meta)
                row['open_and_stabilize_seconds'] = round(time.perf_counter() - t0, 3)
                row['auto_helper'] = ct1 is not None

                if ct1 is None:
                    t0 = time.perf_counter()
                    _点击微信图片识别二维码(image)
                    row['recognize_click_seconds'] = round(time.perf_counter() - t0, 3)
                    t0 = time.perf_counter()
                    ct1 = _等待微信支付商家助手窗口(timeout=15, raise_on_timeout=False)
                    row['wait_helper_seconds'] = round(time.perf_counter() - t0, 3)
                else:
                    row['recognize_click_seconds'] = 0
                    row['wait_helper_seconds'] = 0

                row['helper_detected'] = ct1 is not None
                row['success'] = ct1 is not None
                if ct1 is not None:
                    try:
                        rect = ct1.BoundingRectangle
                        row['helper_rect'] = [rect.left, rect.top, rect.right, rect.bottom]
                    except Exception as exc:
                        row['helper_rect_error'] = repr(exc)
            except Exception as exc:
                row['success'] = False
                row['error'] = repr(exc)
            finally:
                try:
                    if image is not None:
                        image.Close()
                except Exception as exc:
                    row['image_close_error'] = repr(exc)
                t0 = time.perf_counter()
                row['reset_after'] = _快速重置微信二维码窗口状态(close_seconds=3)
                row['reset_after_seconds'] = round(time.perf_counter() - t0, 3)
                row['round_seconds'] = round(time.perf_counter() - round_started, 3)
                result['rounds'].append(row)

        result['success_count'] = sum(1 for row in result['rounds'] if row.get('success'))
        result['total_seconds'] = round(time.perf_counter() - total_started, 3)
        successful = [row['round_seconds'] for row in result['rounds'] if row.get('success')]
        if successful:
            result['avg_success_round_seconds'] = round(sum(successful) / len(successful), 3)
        return result

    @staticmethod
    def 扫码登录微信支付(user, *, assume_current_chat=False, after_text=None):
        if os.getenv('KQ_WECHAT_QRCODE_CHILD') == '1':
            return KqWechat._扫码登录微信支付本进程(user, assume_current_chat=assume_current_chat, after_text=after_text)

        raw_timeout = os.getenv(
            'KQ_WECHAT_QRCODE_CHILD_TIMEOUT_SECONDS',
            os.getenv('KQ_WEIPAY_LOGIN_TIMEOUT_SECONDS', '75'),
        )
        try:
            timeout = min(120, max(30, int(float(raw_timeout))))
        except (TypeError, ValueError):
            timeout = 75

        cmd = [
            sys.executable,
            '-c',
            (
                'from kq5034.wechat_runtime import KqWechat; '
                'import json, sys; '
                'kw=json.loads(sys.argv[1]); '
                'KqWechat._扫码登录微信支付本进程(**kw)'
            ),
            json.dumps({
                'user': str(user),
                'assume_current_chat': assume_current_chat,
                'after_text': after_text,
            }, ensure_ascii=False),
        ]
        env = os.environ.copy()
        env.update({
            'KQ_WECHAT_QRCODE_CHILD': '1',
            'PYTHONUTF8': '1',
            'PYTHONIOENCODING': 'utf-8',
        })
        logger.info(f'微信支付扫码登录：启动隔离子进程 timeout={timeout}s user={user!r}')
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process_runtime.terminate_process_tree(proc.pid, timeout=3)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except OSError:
                    pass
                stdout, stderr = '', ''
            diag_dir = _采集微信二维码诊断('扫码登录微信支付子进程超时', exc, include_uia=False)
            if stdout:
                logger.warning(f'微信支付扫码登录子进程超时 stdout：{stdout[-4000:]}')
            if stderr:
                logger.warning(f'微信支付扫码登录子进程超时 stderr：{stderr[-4000:]}')
            try:
                reset_result = _快速重置微信二维码窗口状态(close_seconds=3)
                logger.warning(f'微信支付扫码登录子进程超时后已重置微信二维码窗口状态：{reset_result}')
            except Exception as reset_exc:
                logger.warning(f'微信支付扫码登录子进程超时后重置微信二维码窗口状态失败：{reset_exc!r}')
            raise TimeoutError(f'微信支付扫码登录子进程超时：timeout={timeout}s，诊断目录：{diag_dir}') from exc

        if proc.returncode != 0:
            diag_dir = _采集微信二维码诊断('扫码登录微信支付子进程失败', include_uia=False)
            if stdout:
                logger.error(f'微信支付扫码登录子进程失败 stdout：{stdout[-4000:]}')
            if stderr:
                logger.error(f'微信支付扫码登录子进程失败 stderr：{stderr[-4000:]}')
            raise RuntimeError(f'微信支付扫码登录子进程失败：exit_code={proc.returncode}，诊断目录：{diag_dir}')

        if stdout:
            logger.info(f'微信支付扫码登录子进程 stdout：{stdout[-2000:]}')
        if stderr:
            logger.info(f'微信支付扫码登录子进程 stderr：{stderr[-4000:]}')

    @staticmethod
    def _扫码登录微信支付本进程(user, *, assume_current_chat=False, after_text=None):
        """
        :param user: 微信群名/图片二维码存放的群位置
        """
        raw_watchdog_timeout = os.getenv('KQ_WECHAT_QRCODE_TIMEOUT_SECONDS', '55')
        try:
            watchdog_timeout = min(45, max(15, int(float(raw_watchdog_timeout)) - 5))
        except (TypeError, ValueError):
            watchdog_timeout = 45
        watchdog = _启动微信二维码诊断看门狗('扫码登录微信支付卡住', timeout=watchdog_timeout)
        # 0 打开图片
        logger.info(
            '微信支付扫码登录：准备打开二维码图片 '
            f'user={user!r} assume_current_chat={assume_current_chat} after_text={after_text!r}'
        )
        try:
            image = None
            _快速重置微信二维码窗口状态(close_seconds=2)
            wx = KqWechat.创建微信实例()
            if assume_current_chat:
                try:
                    current_chat = wx.CurrentChat()
                except Exception as exc:
                    current_chat = ''
                    logger.warning(f'读取当前微信会话失败：{exc!r}')
                if user not in str(current_chat):
                    diag_dir = _采集微信二维码诊断('微信当前会话不匹配', include_uia=False)
                    raise RuntimeError(f'微信当前会话不是目标：target={user!r} current={current_chat!r}，诊断目录：{diag_dir}')
                logger.info(f'微信支付扫码登录：复用当前微信会话 current={current_chat!r}')
            else:
                logger.info(f'微信支付扫码登录：打开会话 user={user!r}')
                wx.ChatWith(user)
            raw_preview_timeout = os.getenv('KQ_WECHAT_IMAGE_PREVIEW_READY_TIMEOUT_SECONDS', '15')
            raw_preview_stable = os.getenv('KQ_WECHAT_IMAGE_PREVIEW_STABLE_SECONDS', '1.2')
            try:
                preview_timeout = min(45, max(15, float(raw_preview_timeout)))
            except (TypeError, ValueError):
                preview_timeout = 15
            try:
                preview_stable = min(3, max(0.3, float(raw_preview_stable)))
            except (TypeError, ValueError):
                preview_stable = 1.2

            logger.info('微信支付扫码登录：打开最近二维码图片')
            # 新版微信可能在打开图片后自动识别二维码并直接拉起商家助手，
            # 此时 ImagePreviewWnd 已关闭。先认最终状态，避免把成功误判成
            # “找不到识别图中二维码按钮”。
            image, ct1, open_info = _打开最近微信二维码图片(
                wx,
                max_messages=int(os.getenv('KQ_WECHAT_QRCODE_OPEN_MAX_MESSAGES', '8')),
                open_timeout=preview_timeout,
                stable_seconds=preview_stable,
                after_text=after_text,
            )
            logger.info(f'微信支付扫码登录：二维码图片打开结果 {open_info}')
            if ct1 is None:
                logger.info('微信支付扫码登录：点击微信图片“识别图中二维码”')
                _点击微信图片识别二维码(image)

            # 2 会弹出一个新的小程序窗口
            def current_helper_control(default_ctrl=None, *, timeout=2):
                ctrl = _等待微信支付商家助手窗口(timeout=timeout, raise_on_timeout=False)
                if ctrl is None:
                    return default_ctrl if _存在微信支付商家助手Win32() else None
                return _规范化微信支付商家助手窗口(ctrl)

            def helper_ltrb(ctrl):
                return _控件矩形(ctrl)

            logger.info('微信支付扫码登录：等待微信支付商家助手窗口')
            if ct1 is None:
                ct1 = _等待微信支付商家助手窗口(timeout=30, raise_on_timeout=False)
            if ct1 is None:
                raise RuntimeError('微信支付商家助手窗口未出现，二维码可能已失效或微信识别未完成')

            # 3 点击进入商店，以及点击退出小程序窗口
            ct1 = _规范化微信支付商家助手窗口(ct1)
            rect = ct1.BoundingRectangle
            ltrb = [rect.left, rect.top, rect.right, rect.bottom]
            logger.info(f'微信支付扫码登录：微信支付商家助手窗口位置={ltrb}')
            merchant_selected = False
            raw_helper_ready_timeout = os.getenv('KQ_WECHAT_PAY_HELPER_READY_TIMEOUT_SECONDS', '30')
            try:
                helper_ready_timeout = min(90, max(30, int(float(raw_helper_ready_timeout))))
            except (TypeError, ValueError):
                helper_ready_timeout = 30
            if isinstance(ct1, _Win32HelperControl):
                logger.info('微信支付扫码登录：商家助手使用Win32代理，直接点击商户选择页客堂商户行')
                merchant_selected = _点击微信支付商户行(ct1)
            else:
                helper_ready_deadline = time.time() + helper_ready_timeout
                while time.time() < helper_ready_deadline:
                    assistant_text = _控件树文本(ct1, max_nodes=500)
                    if any(key in assistant_text for key in ('系统繁忙', '网络繁忙', '稍后再试', '服务异常')):
                        raise RuntimeError(f'微信支付商家助手异常页面：{assistant_text[:300]!r}')
                    merchant_selected = _点击匹配控件(ct1, ['1599622041', '武陵禅寺客堂'])
                    if merchant_selected:
                        break
                    refreshed = _点击匹配控件(ct1, ['刷新', '重新获取', '重新加载'], control_types={'ButtonControl'})
                    if refreshed:
                        time.sleep(3)
                        continue
                    time.sleep(1)
            if not merchant_selected:
                _采集微信二维码诊断('微信支付商家助手未命中语义商户行')
                logger.info('微信支付扫码登录：未命中语义商户行，按商户选择页比例点击客堂商户行')
                merchant_selected = _点击微信支付商户行(ct1)

            if merchant_selected:
                _等待微信支付商户选择生效()
                # 选择商户后，微信小程序经常会切换为居中的确认窗口。
                # 确认阶段必须重新读取窗口位置，不能沿用商户列表阶段坐标。
                raw_confirm_timeout = os.getenv('KQ_WECHAT_PAY_CONFIRM_TIMEOUT_SECONDS', '30')
                try:
                    confirm_timeout = min(90, max(30, int(float(raw_confirm_timeout))))
                except (TypeError, ValueError):
                    confirm_timeout = 30
                raw_confirm_settle_seconds = os.getenv('KQ_WECHAT_PAY_CONFIRM_SETTLE_SECONDS', '30')
                try:
                    confirm_settle_seconds = min(90, max(30, int(float(raw_confirm_settle_seconds))))
                except (TypeError, ValueError):
                    confirm_settle_seconds = 30
                confirm_deadline = time.time() + confirm_timeout
                confirm_done = False

                def wait_confirm_clicked_settle(base_ctrl):
                    logger.info(f'微信支付扫码登录：确认按钮已点击，等待确认结果 seconds={confirm_settle_seconds}')
                    settle_deadline = time.time() + confirm_settle_seconds
                    last_text = ''
                    last_ocr_texts = []
                    while time.time() < settle_deadline:
                        ctrl = current_helper_control(base_ctrl, timeout=1)
                        if ctrl is None:
                            logger.info('微信支付扫码登录：确认点击后商家助手窗口已关闭')
                            return True
                        text = _控件树文本(ctrl, max_nodes=500)
                        last_text = text
                        if any(key in text for key in ('系统繁忙', '网络繁忙', '稍后再试', '服务异常')):
                            raise RuntimeError(f'微信支付商家助手确认后异常页面：{text[:300]!r}')
                        if '登录成功' in text:
                            logger.info('微信支付扫码登录：确认点击后商家助手显示登录成功，关闭成功弹窗')
                            _关闭微信支付商家助手窗口(ctrl)
                            return True
                        clicked_done, ocr_texts = _点击微信支付完成OCR(ctrl)
                        last_ocr_texts = ocr_texts
                        if any(key in ''.join(ocr_texts) for key in ('系统繁忙', '网络繁忙', '稍后再试', '服务异常')):
                            raise RuntimeError(f'微信支付商家助手确认后OCR异常页面：{ocr_texts[:12]!r}')
                        if clicked_done:
                            time.sleep(2)
                            if current_helper_control(ctrl, timeout=1) is not None:
                                _关闭微信支付商家助手窗口(ctrl)
                            return True
                        if any('登录成功' in item.replace(' ', '') for item in ocr_texts):
                            logger.info(f'微信支付扫码登录：OCR已看到登录成功但未命中完成按钮，强制关闭弹窗 texts={ocr_texts[:12]!r}')
                            _关闭微信支付商家助手窗口(ctrl)
                            return True
                        time.sleep(1)
                    logger.info(
                        '微信支付扫码登录：确认点击后仍未出现成功/关闭，'
                        f'末次文本={last_text[:300]!r} 末次OCR={last_ocr_texts[:12]!r}'
                    )
                    return False

                while time.time() < confirm_deadline:
                    ct2 = current_helper_control(ct1, timeout=1)
                    if ct2 is None:
                        logger.info('微信支付扫码登录：商家助手窗口已关闭，确认阶段结束')
                        confirm_done = True
                        break
                    assistant_text = _控件树文本(ct2, max_nodes=500)
                    if any(key in assistant_text for key in ('系统繁忙', '网络繁忙', '稍后再试', '服务异常')):
                        raise RuntimeError(f'微信支付商家助手异常页面：{assistant_text[:300]!r}')
                    if '登录成功' in assistant_text:
                        logger.info('微信支付扫码登录：商家助手已显示登录成功，结束确认阶段等待浏览器跳转')
                        confirm_done = True
                        break
                    if _点击匹配控件(ct2, ['确认登录', '允许登录', '同意登录', '登录', '允许'], control_types={'ButtonControl'}):
                        confirm_done = wait_confirm_clicked_settle(ct2)
                        break
                    if isinstance(ct2, _Win32HelperControl):
                        clicked_confirm, ocr_texts = _点击微信支付确认OCR(ct2)
                        if clicked_confirm:
                            confirm_done = wait_confirm_clicked_settle(ct2)
                            break
                        logger.info(f'微信支付扫码登录：确认阶段等待OCR确认按钮，当前文本={ocr_texts[:12]!r}')
                        time.sleep(1)
                        continue
                    if _点击匹配控件(ct2, ['1599622041', '武陵禅寺客堂']):
                        time.sleep(2)
                        continue
                    ltrb2 = helper_ltrb(ct2)
                    logger.info(f'微信支付扫码登录：确认阶段未命中语义控件，点击当前窗口确认区域={ltrb2}')
                    _点击微信支付确认区域(ct2)
                    confirm_done = wait_confirm_clicked_settle(ct2)
                    break
                if not confirm_done:
                    diag_dir = _采集微信二维码诊断('微信支付商家助手确认未完成')
                    raise RuntimeError(f'微信支付商家助手确认未完成，诊断目录：{diag_dir}')
        except Exception as exc:
            _采集微信二维码诊断('扫码登录微信支付失败', exc)
            try:
                _快速重置微信二维码窗口状态(close_seconds=3)
            except Exception as reset_exc:
                logger.warning(f'微信支付扫码登录失败后重置窗口状态失败：{reset_exc!r}')
            raise
        finally:
            watchdog.cancel()
            try:
                if image is not None:
                    image.Close()
            except UnboundLocalError:
                pass
            except Exception as exc:
                logger.warning(f'关闭微信二维码图片窗口失败：{exc!r}')

    @staticmethod
    def 诊断微信二维码处理(user=None, *, open_latest=False, click_qrcode=False):
        raw_timeout = os.getenv('KQ_WECHAT_QRCODE_PROBE_TIMEOUT_SECONDS', '60')
        try:
            timeout = max(10, int(float(raw_timeout)))
        except (TypeError, ValueError):
            timeout = 60

        payload = json.dumps({
            'user': user,
            'open_latest': open_latest,
            'click_qrcode': click_qrcode,
        }, ensure_ascii=False)
        cmd = [
            sys.executable,
            '-c',
            (
                'from kq5034.wechat_runtime import KqWechat; '
                'import json, sys; '
                'kw=json.loads(sys.argv[1]); '
                'print(json.dumps(KqWechat._诊断微信二维码处理本进程(**kw), ensure_ascii=False))'
            ),
            payload,
        ]
        env = os.environ.copy()
        env.update({
            'PYTHONUTF8': '1',
            'PYTHONIOENCODING': 'utf-8',
        })
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process_runtime.terminate_process_tree(proc.pid, timeout=3)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except OSError:
                    pass
                stdout, stderr = '', ''
            diag_dir = _采集微信二维码诊断('诊断_微信二维码处理子进程超时', exc, include_uia=False)
            return {
                'status': 'timeout',
                'timeout': timeout,
                'diag_dir': str(diag_dir),
                'stdout_tail': stdout[-2000:],
                'stderr_tail': stderr[-2000:],
            }

        if proc.returncode != 0:
            diag_dir = _采集微信二维码诊断('诊断_微信二维码处理子进程失败', include_uia=False)
            return {
                'status': 'failed',
                'exit_code': proc.returncode,
                'diag_dir': str(diag_dir),
                'stdout_tail': stdout[-2000:],
                'stderr_tail': stderr[-2000:],
            }

        try:
            result = json.loads(stdout.strip().splitlines()[-1])
        except Exception:
            result = {'status': 'bad_output', 'stdout_tail': stdout[-2000:], 'stderr_tail': stderr[-2000:]}
        result.setdefault('status', 'ok')
        return result

    @staticmethod
    def _诊断微信二维码处理本进程(user=None, *, open_latest=False, click_qrcode=False):
        """只诊断微信图片二维码处理链路。

        open_latest=True 时会进入指定会话并打开最新消息图片；
        click_qrcode=True 才会真正点击“识别图中二维码”，默认只检查按钮可见性。
        """
        if open_latest:
            if not user:
                raise ValueError('open_latest=True 时必须提供 user')
            wx = KqWechat.创建微信实例()
            logger.info(f'微信二维码诊断：打开会话 user={user!r}')
            wx.ChatWith(user)
            messages = wx.GetAllMessage()
            if not messages:
                diag_dir = _采集微信二维码诊断('诊断_微信会话无消息')
                raise RuntimeError(f'微信会话没有可点击消息：user={user!r}，诊断目录：{diag_dir}')
            messages[-1].click()

        image = WeChatImage()
        candidates = []
        for label, ctrl in _微信图片二维码按钮候选(image):
            candidates.append({
                'label': label,
                'available': _控件可用(ctrl),
                'summary': _控件摘要(ctrl),
            })

        diag_dir = _采集微信二维码诊断('诊断_微信二维码处理')
        result = {
            'diag_dir': str(diag_dir),
            'candidate_count': len(candidates),
            'candidates': candidates,
        }
        logger.info(f'微信二维码诊断结果：{result}')

        if click_qrcode:
            _点击微信图片识别二维码(image)
            result['clicked'] = True
        else:
            result['clicked'] = False
        return result

    @staticmethod
    def 从懒人转发获得短信内容(time_window=5, check_interval=1, timeout=300):
        """等待懒人转发的微信支付短信验证码，超时后抛异常交给外层重试。

        :param int time_window: 短信来电时间的有效窗口，单位分钟。
        :param float check_interval: 轮询微信消息的间隔，单位秒。
        :param timeout: 等待超时时间，单位秒；传入None表示不限制等待时间。
        :return str: 6位短信验证码。
        """
        from datetime import datetime, timedelta

        if check_interval <= 0:
            raise ValueError(f'check_interval必须大于0：{check_interval!r}')
        if timeout is not None and timeout < 0:
            raise ValueError(f'timeout不能为负数：{timeout!r}')
        if timeout is not None:
            # 实际短信链路经常超过 5 分钟，时间窗过短会把已到达的验证码当成过期消息漏掉。
            time_window = max(time_window, int((timeout + 59) // 60) + 1)

        def extract_verification_code(text):
            """从文本中提取6位验证码"""
            match = re.search(r'验证码【(\d{6})】', text)
            return match.group(1) if match else None

        def extract_call_time(text):
            """从文本中提取来电时间"""
            match = re.search(r'来电时间：(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', text)
            return match.group(1) if match else None

        def is_recent_time(time_str, time_window):
            """验证时间是否在指定时间窗口（分钟）内"""
            try:
                msg_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                return False

            current_time = datetime.now()
            time_diff = current_time - msg_time
            return timedelta(minutes=0) <= time_diff <= timedelta(minutes=time_window)

        def validate_message(text, time_window=5):
            """ 综合验证短信有效性 """
            code = extract_verification_code(text)
            time_str = extract_call_time(text)

            if not code or not time_str:
                return None

            return code if is_recent_time(time_str, time_window) else None

        service_name = '懒人信息转发服务'

        def message_text(message):
            """兼容 wxautox 的 Message 对象；短信文本可能在 sender/info 中。"""
            if message is None:
                return ''
            if isinstance(message, str):
                return message
            parts = []
            for attr in ('sender', 'content', 'text'):
                value = getattr(message, attr, None)
                if value:
                    parts.append(str(value))
            try:
                info = message.info
            except Exception:
                info = None
            if isinstance(info, (list, tuple)):
                parts.extend(str(x) for x in info if x)
            elif info:
                parts.append(str(info))
            if not parts:
                parts.append(str(message))
            return ' '.join(dict.fromkeys(parts))

        def collect_current_chat_texts(wx):
            """只读目标聊天当前已加载消息，避免 GetSession 递归扫描整棵会话树。"""
            texts = []
            try:
                messages = wx.GetAllMessage()
            except Exception as exc:
                logger.warning(f'读取微信短信转发会话消息失败：{exc!r}')
                return texts
            for message in reversed(messages):
                text = message_text(message)
                if '验证码' in text or '95017' in text:
                    texts.append(text)
            return texts

        wx = KqWechat.创建微信实例()
        chat_opened = False
        last_error = None
        last_progress_log_at = None
        latest_candidate_time = None
        latest_candidate_text = ''
        matched_candidate_count = 0

        deadline = None if timeout is None else time.monotonic() + timeout
        logger.info(
            '微信支付短信验证码等待开始：'
            f'service={service_name} timeout={timeout}s time_window={time_window}min check_interval={check_interval}s'
        )
        while True:
            # 新短信提醒来电号码：验证码【644651】95017(微信支付)来电时间：2025-04-02 09:21:27
            if not chat_opened:
                try:
                    wx.ChatWith(service_name, timeout=5, exact=False)
                    chat_opened = True
                except Exception as exc:
                    last_error = exc
                    logger.warning(f'打开微信短信转发会话失败：{exc!r}')
            if chat_opened:
                for content in collect_current_chat_texts(wx):
                    if time_str := extract_call_time(content):
                        latest_candidate_time = time_str
                        latest_candidate_text = content[-120:]
                    if extract_verification_code(content):
                        matched_candidate_count += 1
                    if valid_code := validate_message(content, time_window):
                        logger.info('微信支付短信验证码已从微信目标会话消息获取')
                        return valid_code

            # 若目标会话读取失败，重置后下一轮重新打开，避免长期停在错误聊天。
            if not chat_opened:
                for content in collect_current_chat_texts(wx):
                    if time_str := extract_call_time(content):
                        latest_candidate_time = time_str
                        latest_candidate_text = content[-120:]
                    if extract_verification_code(content):
                        matched_candidate_count += 1
                    if valid_code := validate_message(content, time_window):
                        logger.info('微信支付短信验证码已从当前微信会话消息获取')
                        return valid_code

            now = time.monotonic()
            elapsed = None if deadline is None else max(0, timeout - max(0, deadline - now))
            if last_progress_log_at is None or now - last_progress_log_at >= 60:
                remaining = None if deadline is None else max(0, int(round(deadline - now)))
                logger.info(
                    '微信支付短信验证码等待中：'
                    f'elapsed={0 if elapsed is None else int(round(elapsed))}s '
                    f'remaining={remaining if remaining is not None else "unbounded"} '
                    f'chat_opened={chat_opened} matched_candidates={matched_candidate_count} '
                    f'latest_candidate_time={latest_candidate_time!r} latest_candidate_tail={latest_candidate_text!r}'
                )
                last_progress_log_at = now
            if deadline is not None and now >= deadline:
                detail = f'，last_error={last_error!r}' if last_error else ''
                latest_detail = ''
                if latest_candidate_time or latest_candidate_text:
                    latest_detail = (
                        f'，latest_candidate_time={latest_candidate_time!r}'
                        f'，latest_candidate_tail={latest_candidate_text!r}'
                        f'，matched_candidates={matched_candidate_count}'
                    )
                raise TimeoutError(
                    f'等待懒人信息转发服务短信验证码超时：timeout={timeout}s，time_window={time_window}min'
                    f'{detail}{latest_detail}'
                )

            if deadline is None:
                time.sleep(check_interval)
            else:
                time.sleep(min(check_interval, max(0, deadline - now)))
