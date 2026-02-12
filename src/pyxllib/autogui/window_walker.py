"""window_walker.py - Window Walker Utility based on pywin32
基于 pywin32 的 Windows 窗口遍历与筛选工具

功能介绍：
    模仿 pyxllib.file.walker.DirWalker 的机制，提供了一套灵活的 API 用于遍历、筛选和分析 Windows 窗口。
    主要解决了以下痛点：
    1. 简化 win32gui/win32api 的复杂调用。
    2. 提供“真实可见性”判断（calculate_visible_area），能够排除被其他窗口遮挡的情况。
    3. 支持链式调用构建筛选规则（include/exclude）。
    4. 集成 pandas 和 Document 输出，方便生成报告。

核心用法示例：
    # 1. 初始化 Walker (默认不选中任何窗口)
    ww = WindowWalker(select=False)

    # 2. 添加筛选规则：包含所有真实可见的窗口
    ww.include.is_real_visible()

    # 3. 排除特定的系统窗口（可选）
    ww.exclude.explorer_system_window()

    # 4. 遍历并获取结果
    for entry in ww.iter():
        print(f"[{entry.pid}] {entry.process_name}: {entry.title}")

    # 5. 生成可视化报告 (会在浏览器打开)
    ww.browse(fields=["process_name", "title", "rect", "real_visible_ratio"])
"""

import inspect
import win32gui
import win32process
import win32api
import win32con
import ctypes
from ctypes import wintypes

from typing import List, Tuple, Callable, Optional, Iterator, Union, Any
from pyxllib.text.pstr import PStr
from pyxllib.prog.lazyimport import lazy_import
from pyxllib.text.document import Document, DocumentableMixin

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = lazy_import("pandas")


def __1_基础工具函数():
    """ DWM属性、显示器信息、几何计算等 """
    pass


# DWM Window Attributes
DWMWA_CLOAKED = 14


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


def get_monitor_info():
    """获取所有显示器的详细信息

    :return list: 包含每个显示器详细信息的字典列表
    """
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


def calculate_visible_area(current_rect, upper_rects):
    """计算 current_rect 在被 upper_rects 遮挡后的实际可见面积。
    使用坐标压缩/扫描线算法 (无第三方依赖)。

    :param tuple current_rect: 当前窗口矩形 (x1, y1, x2, y2)
    :param list upper_rects: 上层遮挡窗口矩形列表 [(x1, y1, x2, y2), ...]
    :return int: 可见面积（像素数）

    >>> calculate_visible_area((0, 0, 100, 100), [])  # 无遮挡
    10000
    >>> calculate_visible_area((0, 0, 100, 100), [(0, 0, 50, 100)])  # 左半边被遮挡
    5000
    >>> calculate_visible_area((0, 0, 100, 100), [(0, 0, 100, 100)])  # 完全遮挡
    0
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


def __2_窗口对象封装():
    """ WindowEntry 类定义 """
    pass


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
    def rect(self) -> Tuple[int, int, int, int]:
        """窗口矩形 (left, top, right, bottom)"""
        if self._rect is None:
            try:
                self._rect = win32gui.GetWindowRect(self.hwnd)
            except Exception:
                self._rect = (0, 0, 0, 0)
        return self._rect

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


def __3_筛选规则工厂():
    """ WindowFilterFactory & WindowRuleBuilder """
    pass


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


def __4_窗口遍历器():
    """ WindowWalker 主类 """
    pass


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
        monitors = get_monitor_info()
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


if __name__ == "__main__":  # 示例：查找所有可见的
    ww = WindowWalker(select=False)
    ww.include.is_real_visible()
    ww.browse(fields=["process_name", "class_name", "title", "rect", "size"])
