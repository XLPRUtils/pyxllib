"""微信桌面运行时相关实现。"""

import subprocess
import threading
import ctypes
from ctypes import wintypes

from .common import *  # noqa: F403
from pyxllib.prog import process_runtime


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


def _存在微信支付商家助手Win32():
    return any(
        _微信二维码链路Win32角色(item) == 'secondary:wechat-pay-helper'
        for item in _枚举Win32顶层窗口()
    )


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
        try:
            ctrl = uia.PaneControl(Name='微信支付商家助手', searchDepth=1)
            node = UiCtrlNode(ctrl, build_depth=5)
            node.activate()
            return node
        except Exception as exc:
            last_error = exc
            time.sleep(1)

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


def _打开最近微信二维码图片(wx, *, max_messages=8, open_timeout=4, stable_seconds=0.6):
    messages = wx.GetAllMessage()
    if not messages:
        diag_dir = _采集微信二维码诊断('微信会话无消息')
        raise RuntimeError(f'微信会话没有可点击消息，诊断目录：{diag_dir}')

    last_error = None
    tried = 0
    for offset, msg in enumerate(reversed(messages[-max(1, int(max_messages)):]), start=1):
        tried += 1
        try:
            msg.click()
        except Exception as exc:
            last_error = exc
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
            return ctrl
        logger.info(
            '微信支付扫码登录：商家助手窗口超出桌面，先激活并最大化 '
            f'window={[rect.left, rect.top, rect.right, rect.bottom]} screen={[screen_width, screen_height]}'
        )
        UiCtrlNode(ctrl, build_depth=1).activate()
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
    def 扫码登录微信支付(user, *, assume_current_chat=False):
        if os.getenv('KQ_WECHAT_QRCODE_CHILD') == '1':
            return KqWechat._扫码登录微信支付本进程(user, assume_current_chat=assume_current_chat)

        raw_timeout = os.getenv('KQ_WECHAT_QRCODE_TIMEOUT_SECONDS', '55')
        try:
            timeout = min(60, max(20, int(float(raw_timeout))))
        except (TypeError, ValueError):
            timeout = 55

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
    def _扫码登录微信支付本进程(user, *, assume_current_chat=False):
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
        logger.info(f'微信支付扫码登录：准备打开二维码图片 user={user!r} assume_current_chat={assume_current_chat}')
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
            messages = wx.GetAllMessage()
            if not messages:
                diag_dir = _采集微信二维码诊断('微信会话无消息')
                raise RuntimeError(f'微信会话没有可点击消息：user={user!r}，诊断目录：{diag_dir}')
            msg = messages[-1]
            logger.info('微信支付扫码登录：打开最新二维码图片')
            msg.click()  # wxautox才有click方法，wxauto基础版没有

            # 新版微信可能在打开图片后自动识别二维码并直接拉起商家助手，
            # 此时 ImagePreviewWnd 已关闭。先认最终状态，避免把成功误判成
            # “找不到识别图中二维码按钮”。
            raw_preview_timeout = os.getenv('KQ_WECHAT_IMAGE_PREVIEW_READY_TIMEOUT_SECONDS', '8')
            raw_preview_stable = os.getenv('KQ_WECHAT_IMAGE_PREVIEW_STABLE_SECONDS', '0.8')
            try:
                preview_timeout = min(20, max(3, float(raw_preview_timeout)))
            except (TypeError, ValueError):
                preview_timeout = 8
            try:
                preview_stable = min(3, max(0.3, float(raw_preview_stable)))
            except (TypeError, ValueError):
                preview_stable = 0.8
            image, ct1 = _等待微信图片预览或商家助手(
                timeout=preview_timeout,
                stable_seconds=preview_stable,
            )
            if ct1 is None:
                logger.info('微信支付扫码登录：点击微信图片“识别图中二维码”')
                _点击微信图片识别二维码(image)

            # 2 会弹出一个新的小程序窗口
            def calculate_relative_point(ltrb, dst_val):
                # 位置是根据已有经验推断的相对坐标，失败时诊断截图会保留现场。
                left, top, right, bottom = ltrb
                x_center = (left + right) / 2
                y_offset_ratio = (dst_val - 42) / (814 - 42)
                new_height = bottom - top
                y_position = top + y_offset_ratio * new_height
                return (x_center, y_position)

            logger.info('微信支付扫码登录：等待微信支付商家助手窗口')
            if ct1 is None:
                ct1 = _等待微信支付商家助手窗口(timeout=15, raise_on_timeout=False)
            if ct1 is None:
                raise RuntimeError('微信支付商家助手窗口未出现，二维码可能已失效或微信识别未完成')

            # 3 点击进入商店，以及点击退出小程序窗口
            ct1 = _规范化微信支付商家助手窗口(ct1)
            rect = ct1.BoundingRectangle
            ltrb = [rect.left, rect.top, rect.right, rect.bottom]
            logger.info(f'微信支付扫码登录：微信支付商家助手窗口位置={ltrb}')
            clicked_semantic = False
            raw_helper_ready_timeout = os.getenv('KQ_WECHAT_PAY_HELPER_READY_TIMEOUT_SECONDS', '8')
            try:
                helper_ready_timeout = min(15, max(3, int(float(raw_helper_ready_timeout))))
            except (TypeError, ValueError):
                helper_ready_timeout = 8
            helper_ready_deadline = time.time() + helper_ready_timeout
            while time.time() < helper_ready_deadline:
                assistant_text = _控件树文本(ct1, max_nodes=500)
                if any(key in assistant_text for key in ('系统繁忙', '网络繁忙', '稍后再试', '服务异常')):
                    raise RuntimeError(f'微信支付商家助手异常页面：{assistant_text[:300]!r}')
                clicked_semantic = _点击匹配控件(ct1, ['1599622041', '武陵禅寺客堂'])
                if clicked_semantic:
                    time.sleep(3)
                    break
                clicked_semantic = _点击匹配控件(ct1, ['刷新', '重新获取', '重新加载'], control_types={'ButtonControl'})
                if clicked_semantic:
                    time.sleep(3)
                    continue
                time.sleep(1)
            if not clicked_semantic:
                _采集微信二维码诊断('微信支付商家助手未命中语义控件')
                logger.info('微信支付扫码登录：未命中语义控件，回退到经验坐标点击')
                pyautogui.click(*calculate_relative_point(ltrb, 300))
                time.sleep(5)
                pyautogui.click(*calculate_relative_point(ltrb, 650))
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

        deadline = None if timeout is None else time.monotonic() + timeout
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
                    if valid_code := validate_message(content, time_window):
                        logger.info('微信支付短信验证码已从微信目标会话消息获取')
                        return valid_code

            # 若目标会话读取失败，重置后下一轮重新打开，避免长期停在错误聊天。
            if not chat_opened:
                for content in collect_current_chat_texts(wx):
                    if valid_code := validate_message(content, time_window):
                        logger.info('微信支付短信验证码已从当前微信会话消息获取')
                        return valid_code

            now = time.monotonic()
            if deadline is not None and now >= deadline:
                detail = f'，last_error={last_error!r}' if last_error else ''
                raise TimeoutError(f'等待懒人信息转发服务短信验证码超时：timeout={timeout}s，time_window={time_window}min{detail}')

            if deadline is None:
                time.sleep(check_interval)
            else:
                time.sleep(min(check_interval, max(0, deadline - now)))
