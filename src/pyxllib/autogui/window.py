# -*- coding: utf-8 -*-
"""
Windows 窗口管理与布局工具 (Window Management & Layout Tool)
整合了 window_walker (遍历与筛选) 和 window_layout (布局检查与优化) 的功能。

功能：
1. Walker: 遍历、筛选、分析 Windows 窗口，支持链式调用和生成报告。
2. Layout: 检查布局 (Inspect) 和 基于 BSP 的自动平铺 (Optimize)。

依赖：
- pywin32 (win32gui, win32api, win32con, win32process)
- fire (用于 CLI)
- pyxllib (pstr, document, etc.)
- pandas (可选)

用法：
    python window.py           # 默认模式：Inspect
    python window.py inspect   # 检查布局
    python window.py optimize  # 优化布局 (模拟)
    python window.py optimize --apply # 执行优化
"""

import sys
import math
import inspect
import ctypes
from ctypes import wintypes
from typing import List, Tuple, Callable, Optional, Iterator, Union, Any, NamedTuple
from enum import Enum

import win32api
import win32con
import win32gui
import win32process
import fire

from pyxllib.text.pstr import PStr
from pyxllib.prog.lazyimport import lazy_import
from pyxllib.text.document import Document, DocumentableMixin

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import("pandas")


# ==============================================================================
# 1. 基础常量与工具
# ==============================================================================

# DWM Window Attributes
DWMWA_CLOAKED = 14
DWMWA_EXTENDED_FRAME_BOUNDS = 9

# DPI Awareness
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def is_window_cloaked(hwnd):
    """检查窗口是否被 DWM Cloaked (例如虚拟桌面、挂起的 UWP 应用)

    :param int hwnd: 窗口句柄
    :return bool: 如果被 Cloaked 返回 True，否则返回 False
    """
    try:
        cloaked = ctypes.c_int(0)
        res = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
        )
        return res == 0 and cloaked.value != 0
    except Exception:
        return False


def get_extended_frame_bounds(hwnd):
    try:
        rect = wintypes.RECT()
        ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect))
        return (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:
        return win32gui.GetWindowRect(hwnd)


def get_center(rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
    return (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2


def calculate_visible_area(current_rect, upper_rects):
    """计算 current_rect 在被 upper_rects 遮挡后的实际可见面积。
    使用坐标压缩/扫描线算法 (无第三方依赖)。

    :param tuple current_rect: 当前窗口矩形 (x1, y1, x2, y2)
    :param list upper_rects: 上层遮挡窗口矩形列表 [(x1, y1, x2, y2), ...]
    :return int: 可见面积（像素数）
    """
    x1, y1, x2, y2 = current_rect
    if x1 >= x2 or y1 >= y2:
        return 0

    # 1. 预过滤：只保留与当前窗口有交集的遮挡物
    relevant_rects = []
    for ur in upper_rects:
        ux1, uy1, ux2, uy2 = ur
        # 计算交集
        ix1, iy1 = max(x1, ux1), max(y1, uy1)
        ix2, iy2 = min(x2, ux2), min(y2, uy2)

        # 如果有有效的交集区域
        if ix1 < ix2 and iy1 < iy2:
            relevant_rects.append((ix1, iy1, ix2, iy2))

    # 如果没有任何遮挡，直接返回自身面积
    if not relevant_rects:
        return (x2 - x1) * (y2 - y1)

    # 2. 坐标网格化 (Coordinate Compression)
    # 提取所有相关的 X 和 Y 坐标，排序并去重，形成网格线
    xs = sorted(list(set([x1, x2] + [r[0] for r in relevant_rects] + [r[2] for r in relevant_rects])))
    ys = sorted(list(set([y1, y2] + [r[1] for r in relevant_rects] + [r[3] for r in relevant_rects])))

    total_area = 0

    # 遍历每个网格单元
    for i in range(len(xs) - 1):
        w = xs[i + 1] - xs[i]
        if w <= 0:
            continue

        cx = (xs[i] + xs[i + 1]) / 2  # 网格中心 X

        for j in range(len(ys) - 1):
            h = ys[j + 1] - ys[j]
            if h <= 0:
                continue

            cy = (ys[j] + ys[j + 1]) / 2  # 网格中心 Y

            # 核心逻辑：检查当前网格中心点是否在任何一个遮挡矩形内
            # 如果被遮挡，则该网格不可见；否则可见并计入总面积
            is_covered = False
            for ux1, uy1, ux2, uy2 in relevant_rects:
                if ux1 <= cx <= ux2 and uy1 <= cy <= uy2:
                    is_covered = True
                    break

            if not is_covered:
                total_area += w * h

    return total_area


def get_monitor_info_dicts():
    """获取所有显示器的详细信息 (Dict format for Walker)"""
    monitors = []
    try:
        for i, (hMonitor, hdcMonitor, pyRect) in enumerate(win32api.EnumDisplayMonitors()):
            info = win32api.GetMonitorInfo(hMonitor)
            # pyRect is (left, top, right, bottom)
            width = pyRect[2] - pyRect[0]
            height = pyRect[3] - pyRect[1]

            monitors.append(
                {
                    "Index": i + 1,
                    "Device": info.get("Device", ""),
                    "size": f"{height}x{width}",
                    "rect": str(pyRect),
                    "Work Area": str(info.get("Work", "")),
                    "Flags": info.get("Flags", 0),
                    "Primary": "Yes" if info.get("Flags", 0) & win32con.MONITORINFOF_PRIMARY else "",
                }
            )
    except Exception:
        pass
    return monitors


# ==============================================================================
# 2. 窗口对象封装 (WindowWalker Core)
# ==============================================================================

class WindowEntry:
    """窗口对象封装，类似于 os.DirEntry

    提供窗口的各种属性访问，包括句柄、标题、类名、PID、几何信息等。
    支持懒加载以提高性能。
    """

    def __init__(self, hwnd: int):
        self.hwnd = hwnd
        self._title = None
        self._class_name = None
        self._pid = None
        self._tid = None
        self._rect = None
        self._is_visible = None
        self._upper_rects = None
        self._real_visible_area = None
        self._is_cloaked = None
        self._extended_frame_bounds = None

    def set_context(self, upper_rects):
        """注入上下文信息（如上层遮挡窗口），用于计算可见性

        :param list upper_rects: 上层窗口矩形列表
        """
        self._upper_rects = upper_rects

    @property
    def real_visible_area(self) -> int:
        """实际可见面积（扣除遮挡）"""
        if self._real_visible_area is None:
            if not self.is_visible:
                self._real_visible_area = 0
            else:
                # 如果没有上下文，默认没有遮挡（或者应该警告？）
                # 这里为了兼容性，如果没有注入 upper_rects，假设没有遮挡
                upper = getattr(self, "_upper_rects", []) or []
                self._real_visible_area = calculate_visible_area(self.rect, upper)
        return self._real_visible_area

    @property
    def real_visible_ratio(self) -> float:
        """实际可见比例 (0-100)"""
        area = self.area
        if area <= 0:
            return 0.0
        return self.real_visible_area / area * 100

    @property
    def title(self) -> str:
        """窗口标题"""
        if self._title is None:
            self._title = win32gui.GetWindowText(self.hwnd)
        return self._title

    @property
    def name(self) -> str:
        """兼容 DirWalker，别名 title"""
        return self.title

    @property
    def class_name(self) -> str:
        """窗口类名"""
        if self._class_name is None:
            self._class_name = win32gui.GetClassName(self.hwnd)
        return self._class_name

    @property
    def pid(self) -> int:
        """进程 ID"""
        if self._pid is None:
            self._tid, self._pid = win32process.GetWindowThreadProcessId(self.hwnd)
        return self._pid

    @property
    def tid(self) -> int:
        """线程 ID"""
        if self._tid is None:
            self._tid, self._pid = win32process.GetWindowThreadProcessId(self.hwnd)
        return self._tid

    @property
    def is_visible(self) -> bool:
        """窗口是否可见（这个是不考虑Z轴实际遮挡情况的）"""
        if self._is_visible is None:
            self._is_visible = bool(win32gui.IsWindowVisible(self.hwnd))
        return self._is_visible

    @property
    def is_cloaked(self) -> bool:
        """窗口是否被 DWM Cloaked"""
        if self._is_cloaked is None:
            self._is_cloaked = is_window_cloaked(self.hwnd)
        return self._is_cloaked

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """窗口矩形 (left, top, right, bottom)"""
        if self._rect is None:
            try:
                self._rect = win32gui.GetWindowRect(self.hwnd)
            except Exception:
                self._rect = (0, 0, 0, 0)
        return self._rect

    @property
    def extended_frame_bounds(self) -> Tuple[int, int, int, int]:
        """窗口视觉边界 (去除阴影) (left, top, right, bottom)"""
        if self._extended_frame_bounds is None:
            self._extended_frame_bounds = get_extended_frame_bounds(self.hwnd)
        return self._extended_frame_bounds

    @property
    def center(self) -> Tuple[int, int]:
        """窗口中心坐标 (x, y)"""
        left, top, right, bottom = self.rect
        return (left + right) // 2, (top + bottom) // 2

    @property
    def width(self) -> int:
        """窗口宽度"""
        left, top, right, bottom = self.rect
        return max(0, right - left)

    @property
    def height(self) -> int:
        """窗口高度"""
        left, top, right, bottom = self.rect
        return max(0, bottom - top)

    @property
    def size(self) -> str:
        """窗口分辨率 (高x宽)"""
        return f"{self.height}x{self.width}"

    @property
    def area(self) -> int:
        """窗口面积"""
        return self.width * self.height

    @property
    def process_name(self) -> str:
        """进程名"""
        if not hasattr(self, "_process_name"):
            self._process_name = ""
            try:
                # 尝试获取进程句柄
                h_process = win32api.OpenProcess(
                    win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, self.pid
                )
                if h_process:
                    try:
                        # 获取进程路径
                        path = win32process.GetModuleFileNameEx(h_process, 0)
                        self._process_name = path.split("\\")[-1]
                    finally:
                        win32api.CloseHandle(h_process)
            except Exception:
                pass
        return self._process_name

    def __repr__(self):
        return f"<WindowEntry hwnd={self.hwnd} title='{self.title}' class='{self.class_name}'>"


# 定义 Predicate 类型别名
Predicate = Callable[[WindowEntry], bool]


class WindowFilterFactory:
    """生成窗口筛选的判断函数"""

    @classmethod
    def custom(cls, func: Predicate):
        """自定义判断函数"""
        return func

    @classmethod
    def is_visible(cls, min_area=0):
        """只匹配可见窗口

        :param int min_area: 最小面积
        """
        return lambda e: e.is_visible and e.area >= min_area

    @classmethod
    def is_real_visible(cls, min_area=2, min_ratio=0, limit_coord=30000):
        """匹配实际可见性（考虑遮挡）

        :param int min_area: 最小可见像素数 (建议默认值用2，可以把explorer.exe有特殊的1像素的对象过滤掉)
        :param float min_ratio: 最小可见比例 (0-100, 默认0)。如果设70，表示如果这个窗口被遮挡的部分超过30%，也不满足条件。
        :param int limit_coord: 坐标限制 (默认30000)。如果窗口的坐标绝对值超过此值，视为不可见。
                            (Windows最小化窗口坐标通常为-32000，需要过滤)
        """

        def _check(e):
            if limit_coord > 0:
                l, t, r, b = e.rect
                if max(abs(l), abs(t), abs(r), abs(b)) > limit_coord:
                    return False
            return (e.real_visible_area >= min_area) and (e.real_visible_ratio >= min_ratio)

        return _check

    @classmethod
    def match_title(cls, patterns, ignore_case=True):
        """匹配窗口标题

        :param str|list patterns: 匹配模式，支持通配符
        :param bool ignore_case: 是否忽略大小写
        """
        if isinstance(patterns, str):
            patterns = [patterns]
        processed_patterns = [PStr.auto(p, ignore_case=ignore_case) for p in patterns]
        return lambda e: any(p.match(e.title) for p in processed_patterns)

    @classmethod
    def match_class_name(cls, patterns, ignore_case=True):
        """匹配窗口类名

        :param str|list patterns: 匹配模式
        :param bool ignore_case: 是否忽略大小写
        """
        if isinstance(patterns, str):
            patterns = [patterns]
        processed_patterns = [PStr.auto(p, ignore_case=ignore_case) for p in patterns]
        return lambda e: any(p.match(e.class_name) for p in processed_patterns)

    @classmethod
    def match_process_name(cls, patterns, ignore_case=True):
        """匹配进程名

        :param str|list patterns: 匹配模式
        :param bool ignore_case: 是否忽略大小写
        """
        if isinstance(patterns, str):
            patterns = [patterns]
        processed_patterns = [PStr.auto(p, ignore_case=ignore_case) for p in patterns]
        return lambda e: any(p.match(e.process_name) for p in processed_patterns)

    @classmethod
    def match_pid(cls, pids):
        """匹配 PID

        :param int|list pids: 进程 ID 或 ID 列表
        """
        if isinstance(pids, int):
            pids = {pids}
        else:
            pids = set(pids)
        return lambda e: e.pid in pids

    @classmethod
    def match_area(cls, min_area=0, max_area=None):
        """匹配窗口面积

        :param int min_area: 最小面积
        :param int|None max_area: 最大面积 (None表示不限制上限)
        :return: 判断函数
        """
        return lambda e: (min_area <= e.area) and (max_area is None or e.area <= max_area)

    @staticmethod
    def check_explorer_system_window(e: WindowEntry) -> bool:
        """判断是否为 explorer.exe 的系统窗口"""
        if e.process_name.lower() != "explorer.exe":
            return False

        cname = e.class_name
        # 1. Shell_*TrayWnd (Shell_TrayWnd, Shell_SecondaryTrayWnd)
        if cname.startswith("Shell_") and cname.endswith("TrayWnd"):
            return True

        # 2. Progman - Program Manager
        if cname == "Progman" and e.title == "Program Manager":
            return True

        return False

    @classmethod
    def explorer_system_window(cls):
        """匹配 explorer.exe 的系统窗口 (任务栏、桌面等)
        包括: Shell_*TrayWnd, Progman(Program Manager)
        """
        return cls.check_explorer_system_window


class WindowRuleBuilder:
    """规则构建器，支持链式调用"""

    def __init__(self, parent, rules, target_action: bool, base_condition: Optional[Predicate] = None):
        self.parent = parent
        self.rules = rules
        self.target_action = target_action
        self.base_condition = base_condition

    def __getattr__(self, method):
        if not hasattr(WindowFilterFactory, method):
            raise AttributeError(f"WindowFilterFactory has no attribute '{method}'")

        rule_factory = getattr(WindowFilterFactory, method)

        def wrapper(*args, **kwargs):
            predicate = rule_factory(*args, **kwargs)

            if self.base_condition:
                original_predicate = predicate
                base_cond = self.base_condition

                def composite_predicate(e: WindowEntry) -> bool:
                    return base_cond(e) and original_predicate(e)

                final_predicate = composite_predicate
            else:
                final_predicate = predicate

            self.rules.append((final_predicate, self.target_action))
            return self.parent

        return wrapper


class WindowWalker(DocumentableMixin):
    """窗口遍历和筛选工具"""

    def __init__(self, select=False):
        """
        :param bool select: 默认是否选中窗口。
                       False: 默认不选中，需通过 include 添加白名单。
                       True: 默认选中，需通过 exclude 添加黑名单。
        """
        self.default_select = select
        # rules: list of (predicate, action)
        # action: True for include, False for exclude
        self.select_rules: List[Tuple[Predicate, bool]] = []

    @property
    def include(self) -> WindowRuleBuilder:
        """添加包含规则"""
        return WindowRuleBuilder(self, self.select_rules, True)

    @property
    def exclude(self) -> WindowRuleBuilder:
        """添加排除规则"""
        return WindowRuleBuilder(self, self.select_rules, False)

    @property
    def include_visible(self) -> WindowRuleBuilder:
        """选中：必须是可见窗口 AND 满足后续条件"""
        return WindowRuleBuilder(self, self.select_rules, True, base_condition=lambda e: e.is_visible)

    @property
    def exclude_visible(self) -> WindowRuleBuilder:
        """排除：必须是可见窗口 AND 满足后续条件"""
        return WindowRuleBuilder(self, self.select_rules, False, base_condition=lambda e: e.is_visible)

    def should_select(self, entry: WindowEntry) -> bool:
        """判断是否选中该窗口"""
        decision = self.default_select
        for predicate, action in self.select_rules:
            if decision == action:
                continue
            if predicate(entry):
                decision = action
        return decision

    def iter(self) -> Iterator[WindowEntry]:
        """遍历所有顶层窗口"""
        windows = []

        def enum_handler(hwnd, ctx):
            windows.append(hwnd)

        win32gui.EnumWindows(enum_handler, None)

        upper_rects = []
        for hwnd in windows:
            entry = WindowEntry(hwnd)
            entry.set_context(list(upper_rects))  # Pass a copy to capture current state

            if self.should_select(entry):
                yield entry

            # Update upper_rects for occlusion calculation of subsequent windows
            # Only consider visible, non-minimized, non-cloaked windows as occluders
            if win32gui.IsWindowVisible(hwnd) and not is_window_cloaked(hwnd):
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                    # Simple check for minimized or invalid windows
                    if rect[0] > -30000 and (rect[2] - rect[0]) > 0 and (rect[3] - rect[1]) > 0:
                        upper_rects.append(rect)
                except Exception:
                    pass

    def _generate_layout_image(self, monitors, system_windows, normal_windows, visualize_scale=True):
        try:
            from PIL import Image, ImageDraw
            import ast
            import io
        except ImportError:
            return None

        # 1. Parse monitor rects and labels
        monitor_rects = []
        monitor_labels = []
        for i, m in enumerate(monitors):
            try:
                # m['rect'] is "(left, top, right, bottom)"
                r = ast.literal_eval(m["rect"])
                monitor_rects.append(r)
                monitor_labels.append(m.get("Device", f"Monitor {i + 1}"))
            except:
                pass

        if not monitor_rects:
            return None

        # 2. Calculate bounds
        xs = [r[0] for r in monitor_rects] + [r[2] for r in monitor_rects]
        ys = [r[1] for r in monitor_rects] + [r[3] for r in monitor_rects]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = max_x - min_x
        height = max_y - min_y

        if width <= 0 or height <= 0:
            return None

        # 3. Setup canvas with scaling
        scale = 1.0

        # Handle visualize_scale parameter
        if isinstance(visualize_scale, (int, float)) and not isinstance(visualize_scale, bool):
            if visualize_scale > 1:
                # Treat as percentage, e.g., 50 -> 0.5
                scale = visualize_scale / 100.0
            elif visualize_scale > 0:
                # Treat as ratio, e.g., 0.5 -> 0.5
                scale = visualize_scale
            else:
                # <= 0 means disabled, but this method shouldn't be called if disabled
                return None
        else:
            # Default auto-scaling logic (max 1200px)
            MAX_DIM = 1200
            if width > MAX_DIM or height > MAX_DIM:
                scale = min(MAX_DIM / width, MAX_DIM / height)

        padding = 40
        canvas_width = int(width * scale) + padding * 2
        canvas_height = int(height * scale) + padding * 2

        img = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        def to_canvas(rect):
            x1, y1, x2, y2 = rect
            nx1 = int((x1 - min_x) * scale) + padding
            ny1 = int((y1 - min_y) * scale) + padding
            nx2 = int((x2 - min_x) * scale) + padding
            ny2 = int((y2 - min_y) * scale) + padding
            return nx1, ny1, nx2, ny2

        # 4. Draw Monitors (Red outline, drawn first)
        for i, r in enumerate(monitor_rects):
            coords = to_canvas(r)
            draw.rectangle(coords, outline=(255, 0, 0), width=3)

            # Label
            label = monitor_labels[i]
            cx, cy = (coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2

            try:
                # Use textbbox if available (Pillow >= 8.0.0)
                bbox = draw.textbbox((0, 0), label)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                # Fallback for older Pillow
                w, h = draw.textsize(label)

            # Draw text with white background for readability
            text_bg = (cx - w / 2 - 2, cy - h / 2 - 2, cx + w / 2 + 2, cy + h / 2 + 2)
            draw.rectangle(text_bg, fill=(255, 255, 255))
            draw.text((cx - w / 2, cy - h / 2), label, fill=(255, 0, 0))

        # 5. Draw System Windows (Blue, drawn second, with shrink)
        if system_windows:
            shrink = 3  # shrink pixels to avoid overlapping with monitor borders
            for w in system_windows:
                # Only draw Taskbar (TrayWnd), ignore Desktop (Progman)
                if "TrayWnd" not in w.class_name:
                    continue

                coords = to_canvas(w.rect)
                if coords[2] > coords[0] and coords[3] > coords[1]:
                    # Apply shrink
                    sx1 = coords[0] + shrink
                    sy1 = coords[1] + shrink
                    sx2 = coords[2] - shrink
                    sy2 = coords[3] - shrink

                    if sx2 > sx1 and sy2 > sy1:
                        draw.rectangle((sx1, sy1, sx2, sy2), outline=(0, 0, 255), width=2)

        # 6. Normal Windows are not drawn (as requested to keep it clean)
        pass

        # 7. Return bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def to_document(
        self, fields: List[str] = None, split_system_windows=True, title="WindowWalker", visualize=True
    ) -> Document:
        """生成窗口列表的文档对象

        :param list fields: 包含的字段列表
        :param bool split_system_windows: 是否分离系统窗口
        :param str title: 文档标题
        :param bool|int|float visualize: 是否生成可视化布局图。
                          True: 默认自动缩放 (最大尺寸1200px)
                          False/None/0: 不生成
                          int/float > 1: 缩放百分比 (e.g. 50 -> 50%)
                          0 < float <= 1: 缩放比例 (e.g. 0.5 -> 50%)
        :return Document: 生成的文档对象
        """
        if fields is None:
            fields = ["hwnd", "pid", "process_name", "title"]

        # 获取所有符合条件的窗口对象
        entries = list(self.iter())

        # 分组逻辑
        system_windows = []
        normal_windows = []

        if split_system_windows:
            check_sys = WindowFilterFactory.check_explorer_system_window
            for e in entries:
                if check_sys(e):
                    system_windows.append(e)
                else:
                    normal_windows.append(e)
        else:
            normal_windows = entries

        # 辅助函数：将 entries 转为 DataFrame
        def _to_df(ents):
            data = []
            for entry in ents:
                row = []
                for f in fields:
                    val = getattr(entry, f, "")
                    if isinstance(val, (list, tuple)):
                        val = str(val)
                        row.append(val)
                data.append(row)
            return pd.DataFrame(data, columns=fields)

        # 获取屏幕信息
        monitors = get_monitor_info_dicts()
        df_monitors = pd.DataFrame(monitors)

        # 使用 Document 构建报告
        doc = Document(title=title)

        doc.add_header(f"1 显示器信息 (Monitor Info)", level=2)

        # 判断是否需要可视化
        should_visualize = False
        if visualize:
            if isinstance(visualize, bool) and visualize:
                should_visualize = True
            elif isinstance(visualize, (int, float)) and visualize > 0:
                should_visualize = True

        if should_visualize:
            doc.add_header("1.1 详细参数 (Table Info)", level=3)
            doc.add_table(df_monitors, row_index=False)

            img_bytes = self._generate_layout_image(monitors, system_windows, normal_windows, visualize_scale=visualize)
            if img_bytes:
                doc.add_header("1.2 布局可视化 (Layout Visualization)", level=3)
                doc.add_image(img_bytes)
        else:
            # 无可视化时，直接展示表格，不分小节
            doc.add_table(df_monitors, row_index=False)

        doc.add_header(f"2 窗口列表 (Window List)", level=2)

        if split_system_windows and system_windows:
            df_sys = _to_df(system_windows)
            doc.add_header(f"2.1 系统窗口 (System Windows)", level=3)
            doc.add_table(df_sys, row_index=1)

            df_norm = _to_df(normal_windows)
            doc.add_header(f"2.2 应用窗口 (Application Windows)", level=3)
            doc.add_table(df_norm, row_index=1)
        else:
            df_windows = _to_df(normal_windows)
            doc.add_table(df_windows, row_index=1)

        return doc


# ==============================================================================
# 3. 布局管理核心 (Window Layout / Optimizer)
# ==============================================================================


class MonitorInfo(NamedTuple):
    handle: int
    rect: Tuple[int, int, int, int]
    work_area: Tuple[int, int, int, int]
    index: int
    area_px: int
    device_name: str


class WindowInfo(NamedTuple):
    hwnd: int
    title: str
    class_name: str
    rect: Tuple[int, int, int, int]  # (Left, Top, Right, Bottom) - 窗口实际边界
    visual_rect: Tuple[int, int, int, int]  # (Left, Top, Right, Bottom) - 视觉边界 (去阴影)
    frame_padding: Tuple[int, int, int, int]  # (L, T, R, B) - 边框/阴影厚度
    center: Tuple[int, int]  # (cx, cy)
    width: int
    height: int
    monitor_index: int
    visible_area_px: int  # 可见像素 (扣除遮挡)
    visible_percent_screen: float  # 屏幕占比 (0-100)
    weight: float  # 归一化权重 (0.0-1.0, 用于 BSP)
    is_cloaked: bool  # 是否被 DWM 隐藏


def get_monitors() -> List[MonitorInfo]:
    monitors = []
    try:
        for i, (hMonitor, hdcMonitor, pyRect) in enumerate(win32api.EnumDisplayMonitors()):
            info = win32api.GetMonitorInfo(hMonitor)
            # info['Monitor'] is (left, top, right, bottom)
            w = info["Monitor"][2] - info["Monitor"][0]
            h = info["Monitor"][3] - info["Monitor"][1]
            monitors.append(
                MonitorInfo(
                    handle=hMonitor,
                    rect=info["Monitor"],
                    work_area=info["Work"],
                    index=i,
                    area_px=w * h,
                    device_name=info["Device"],
                )
            )
    except Exception as e:
        print(f"获取显示器信息失败: {e}")
    return monitors


def get_windows_info(monitors: List[MonitorInfo]) -> Tuple[List[WindowInfo], int]:
    """
    扫描所有窗口，计算可见性、归属显示器等详细信息。
    返回: (windows_list, total_screen_area)
    """
    windows = []
    total_screen_area = sum(m.area_px for m in monitors)

    def find_monitor(rect):
        cx, cy = get_center(rect)
        for m in monitors:
            if m.rect[0] <= cx < m.rect[2] and m.rect[1] <= cy < m.rect[3]:
                return m
        return None

    # 初始化 WindowWalker
    ww = WindowWalker(select=False)

    # 基础筛选：必须是“真实可见”的窗口
    # 这会过滤掉完全被遮挡的窗口，以及坐标异常的窗口
    ww.include.is_real_visible()

    # 自定义筛选逻辑 (exclude 模式)
    def exclude_filter(e):
        # 过滤无效/背景窗口
        if e.width <= 0 or e.height <= 0:
            return True
        if e.class_name in ["Progman", "WorkerW"]:
            return True
        if e.title == "Program Manager":  # Explicitly check title
            return True
        if not e.title and e.class_name == "Shell_TrayWnd":
            return True  # 任务栏
        if not e.title and e.width < 100 and e.height < 100:
            return True  # 小且无标题
        return False

    ww.exclude.custom(exclude_filter)

    # 遍历并转换
    for e in ww.iter():
        # Cloaked 检查 (WindowWalker 的 iter 内部已经计算了 visible_area，但我们需要显式处理 cloaked 状态)
        # 注意：WindowWalker 计算 visible_area 时已经考虑了 cloaked (cloaked 窗口不算遮挡物)，
        # 但 e.real_visible_area 是基于 upper_rects 计算的。
        # 如果 e 本身是 cloaked，它的 visible_area 应该是 0 (layout_tool 的逻辑)。

        is_cloaked_val = e.is_cloaked
        if is_cloaked_val:
            vis_area = 0
        else:
            vis_area = e.real_visible_area

        # 确定显示器
        monitor = find_monitor(e.rect)
        mon_idx = monitor.index if monitor else -1

        # 计算屏幕占比
        vis_percent = (vis_area / total_screen_area * 100) if total_screen_area > 0 else 0

        # 获取视觉边界
        visual_rect = e.extended_frame_bounds
        rect = e.rect

        pad_l = visual_rect[0] - rect[0]
        pad_t = visual_rect[1] - rect[1]
        pad_r = rect[2] - visual_rect[2]
        pad_b = rect[3] - visual_rect[3]

        # 简单校验 padding
        if pad_l < 0 or pad_r < 0 or pad_t < 0 or pad_b < 0:
            pad_l = pad_t = pad_r = pad_b = 0
            visual_rect = rect

        windows.append(
            WindowInfo(
                hwnd=e.hwnd,
                title=e.title,
                class_name=e.class_name,
                rect=rect,
                visual_rect=visual_rect,
                frame_padding=(pad_l, pad_t, pad_r, pad_b),
                center=e.center,
                width=e.width,
                height=e.height,
                monitor_index=mon_idx,
                visible_area_px=vis_area,
                visible_percent_screen=vis_percent,
                weight=0.0,  # 稍后在上下文中计算
                is_cloaked=is_cloaked_val,
            )
        )

    return windows, total_screen_area


def bsp_tiling(container_rect: Tuple[int, int, int, int], windows: List[WindowInfo]) -> List[Tuple[WindowInfo, Tuple[int, int, int, int]]]:
    """
    递归分割算法
    container_rect: (x, y, w, h)
    windows: List[WindowInfo] (已分配权重)
    """
    if not windows:
        return []

    # 基准情况：只剩一个窗口
    if len(windows) == 1:
        return [(windows[0], container_rect)]

    cx, cy, cw, ch = container_rect
    split_vertical = cw > ch  # 沿长边切割

    # 排序
    if split_vertical:
        sorted_wins = sorted(windows, key=lambda w: w.center[0])  # 按 X 排序
    else:
        sorted_wins = sorted(windows, key=lambda w: w.center[1])  # 按 Y 排序

    # 分割点策略: 二分数量 + 权重比例
    split_idx = len(sorted_wins) // 2
    left_group = sorted_wins[:split_idx]
    right_group = sorted_wins[split_idx:]

    weight_left = sum(w.weight for w in left_group)
    weight_right = sum(w.weight for w in right_group)

    if weight_left + weight_right == 0:
        ratio = 0.5
    else:
        ratio = weight_left / (weight_left + weight_right)
    
    # 限制 ratio 范围，防止分割过小
    ratio = max(0.2, min(0.8, ratio))

    # 计算子区域
    if split_vertical:
        split_w = int(cw * ratio)
        rect_left = (cx, cy, split_w, ch)
        rect_right = (cx + split_w, cy, cw - split_w, ch)
    else:
        split_h = int(ch * ratio)
        rect_top = (cx, cy, cw, split_h)
        rect_bottom = (cx, cy + split_h, cw, ch - split_h)
        rect_left = rect_top
        rect_right = rect_bottom

    results = []
    results.extend(bsp_tiling(rect_left, left_group))
    results.extend(bsp_tiling(rect_right, right_group))
    return results


class WindowAction(Enum):
    IGNORE = "IGNORE"  # 忽略：不参与任何操作 (黑名单)
    TILE = "TILE"  # 平铺：参与 BSP 布局优化 (默认)
    KEEP_SIZE = "KEEP_SIZE"  # 保持尺寸：可以移动位置，但保持原有尺寸 (暂作为浮动处理或特殊平铺)


class LayoutConfig:
    """布局配置管理"""

    def __init__(self):
        self.rules = []
        # 初始化默认规则
        # 使用 PStr.literal 确保明确意图，或者直接 PStr()
        # 注意：PStr 默认构造即为 Literal
        self.add_rule(PStr("NVIDIA GeForce Overlay"), WindowAction.IGNORE)
        self.add_rule(PStr("Program Manager"), WindowAction.IGNORE)
        self.add_rule(PStr("Task Manager"), WindowAction.IGNORE) # 任务管理器通常最好不要被动

    def add_rule(self, pattern, action):
        """添加一条规则"""
        self.rules.append((pattern, action))

    def get_action(self, title):
        """根据标题匹配规则，返回对应的动作"""
        # 遍历规则
        for pattern, action in self.rules:
            # 使用 search 以支持子串匹配 (对于 Literal)
            if pattern.search(title):
                return action

        return WindowAction.TILE


# 全局配置实例
global_config = LayoutConfig()


# ==============================================================================
# 4. CLI 接口类
# ==============================================================================


class WindowTool:
    """窗口管理工具集"""

    def inspect(self, limit: int = -1, monitor: Optional[int] = None):
        """
        [Inspect] 检查并打印当前所有窗口的布局信息。
        :param limit: 仅显示前 N 个窗口 (默认全部)
        :param monitor: 仅显示指定显示器的窗口索引
        """
        monitors = get_monitors()
        print(f"{'=' * 20} 显示器概览 {'=' * 20}")
        for m in monitors:
            w = m.rect[2] - m.rect[0]
            h = m.rect[3] - m.rect[1]
            print(f"Monitor {m.index}: {w}x{h} | Area: {m.rect} | Device: {m.device_name}")

        windows, total_screen_area = get_windows_info(monitors)
        print(f"Total Screen Area: {total_screen_area} px^2")

        # 筛选
        display_windows = windows
        if monitor is not None:
            display_windows = [w for w in display_windows if w.monitor_index == monitor]

        if limit > 0:
            display_windows = display_windows[:limit]
            print(f"\n{'=' * 20} 窗口列表 (Top {limit}, Z-Order) {'=' * 20}")
        else:
            print(f"\n{'=' * 20} 窗口列表 (All, Z-Order) {'=' * 20}")

        print(
            f"{'IDX':<3} | {'Mon':<3} | {'Screen%':>7} | {'RECT (L, T, R, B)':<22} | {'SIZE (WxH)':<12} | {'TITLE / CLASS'}"
        )
        print("-" * 110)

        total_vis_percent = 0.0
        for i, win in enumerate(display_windows):
            mon_str = str(win.monitor_index) if win.monitor_index >= 0 else "?"
            vis_str = f"{win.visible_percent_screen:.1f}"
            rect_str = f"({win.rect[0]},{win.rect[1]},{win.rect[2]},{win.rect[3]})"
            size_str = f"{win.width}x{win.height}"

            name = win.title if win.title else f"<{win.class_name}>"
            if win.is_cloaked:
                name = f"[Cloaked] {name}"
            if len(name) > 50:
                name = name[:47] + "..."
            
            # 清理标题中的换行符等
            name = name.replace('\n', ' ').replace('\r', '')

            print(f"{i:<3} | {mon_str:<3} | {vis_str:>7} | {rect_str:<22} | {size_str:<12} | {name}")

            if not win.is_cloaked and win.monitor_index >= 0:
                total_vis_percent += win.visible_percent_screen

        print("-" * 110)
        print(f"Total Visible Area Coverage (Displayed): {total_vis_percent:.1f}%")

    def optimize(self, monitor: int = 0, dry_run: bool = True, apply: bool = False, filter: Optional[str] = None, margin: int = 0, include_obscured: bool = False):
        """
        [Optimize] 自动优化窗口布局 (BSP 平铺)。
        :param monitor: 目标显示器索引 (默认 0)
        :param dry_run: 默认为 True (仅模拟)。设为 False 或使用 --apply 来实际执行。
        :param apply: 显式指定执行优化 (相当于 dry_run=False)
        :param filter: 窗口标题关键词过滤
        :param margin: 窗口间距 (像素)
        :param include_obscured: 是否包含被遮挡严重的窗口 (默认过滤掉 < 1% 可见的窗口)
        """
        # 参数处理
        if apply:
            dry_run = False

        print(f"\n>>> 正在初始化优化... (显示器: {monitor}, 模式: {'模拟' if dry_run else '执行'})")

        monitors = get_monitors()
        if monitor >= len(monitors):
            print(f"错误: 显示器索引 {monitor} 超出范围。")
            return

        target_monitor = monitors[monitor]
        print(f"目标工作区: {target_monitor.work_area} | 间距: {margin}px")

        # 获取窗口
        all_windows, _ = get_windows_info(monitors)

        # 筛选候选窗口
        candidates = []
        screen_total_pixels = target_monitor.area_px
        total_candidate_area = 0

        print(f"\n正在筛选显示器 {monitor} 上的窗口...")
        print(f"{'句柄':<10} | {'占比%':<7} | {'状态':<10} | {'标题'}")
        print("-" * 80)

        for w in all_windows:
            if w.monitor_index != monitor:
                continue

            # 标题过滤
            if filter and filter.lower() not in w.title.lower():
                continue

            # 规则检查
            action = global_config.get_action(w.title)

            if action == WindowAction.IGNORE:
                print(f"{w.hwnd:<10} | {w.visible_percent_screen:>6.1f}% | [黑名单]   | {w.title[:40]}")
                continue
            elif action == WindowAction.KEEP_SIZE:
                # TODO: 实现“移动但不调整大小”的逻辑 (例如 Bin Packing 或 级联)
                # 目前暂且跳过，保持原样
                print(f"{w.hwnd:<10} | {w.visible_percent_screen:>6.1f}% | [保持尺寸] | {w.title[:40]}")
                continue

            # action == TILE, 继续后续检查

            # 遮挡检查
            window_area = w.width * w.height
            visibility_ratio = w.visible_area_px / window_area if window_area > 0 else 0
            is_obscured = visibility_ratio < 0.99

            status = "OK"
            should_include = True

            if w.is_cloaked:
                status = "Cloaked"
                should_include = False
            elif not include_obscured and is_obscured:
                status = f"遮挡 {visibility_ratio:.0%}"
                should_include = False
            elif w.visible_percent_screen < 1.0:
                status = "<1% Vis"
                should_include = False

            if should_include:
                candidates.append(w)
                total_candidate_area += w.visible_area_px
                print(f"{w.hwnd:<10} | {w.visible_percent_screen:>6.1f}% | [保留]     | {w.title[:40]}")
            else:
                print(f"{w.hwnd:<10} | {w.visible_percent_screen:>6.1f}% | [忽略:{status:<4}] | {w.title[:40]}")

        if not candidates:
            print("没有发现可优化的窗口。")
            return

        # 计算权重
        weighted_windows = []
        for w in candidates:
            # 权重基于候选窗口的总可见面积重新归一化
            weight = w.visible_area_px / total_candidate_area if total_candidate_area > 0 else 0
            weighted_windows.append(w._replace(weight=weight))

        # 执行 BSP
        wa = target_monitor.work_area
        container_rect = (wa[0], wa[1], wa[2] - wa[0], wa[3] - wa[1])  # (x, y, w, h)

        layout_plan = bsp_tiling(container_rect, weighted_windows)

        # 输出方案
        print(f"\n优化方案 ({len(layout_plan)} 个窗口):")
        print(f"{'句柄':<10} | {'原位置':<20} | {'目标位置':<20} | {'标题'}")
        print("-" * 90)

        for win, target_rect in layout_plan:  # target_rect is (x, y, w, h)
            # 计算最终 SetWindowPos 参数 (应用 margin 和 padding)
            half_margin = margin // 2

            vt_x = target_rect[0] + half_margin
            vt_y = target_rect[1] + half_margin
            vt_w = max(1, target_rect[2] - margin)
            vt_h = max(1, target_rect[3] - margin)

            # 补偿阴影
            pad = win.frame_padding
            final_x = vt_x - pad[0]
            final_y = vt_y - pad[1]
            final_w = vt_w + pad[0] + pad[2]
            final_h = vt_h + pad[1] + pad[3]

            # 打印展示 (目标位置显示为 L,T,R,B)
            target_ltrb = (
                target_rect[0],
                target_rect[1],
                target_rect[0] + target_rect[2],
                target_rect[1] + target_rect[3],
            )
            print(f"{win.hwnd:<10} | {str(win.rect):<20} | {str(target_ltrb):<20} | {win.title[:30]}")

            if not dry_run:
                flags = win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE
                win32gui.SetWindowPos(win.hwnd, 0, final_x, final_y, final_w, final_h, flags)

        if not dry_run:
            print("\n布局应用成功。")
        else:
            print("\n(模拟模式) 未应用更改。使用 --apply 参数执行。")


if __name__ == "__main__":
    # 如果没有提供参数，默认执行 inspect
    if len(sys.argv) == 1:
        sys.argv.append("inspect")
    fire.Fire(WindowTool)
