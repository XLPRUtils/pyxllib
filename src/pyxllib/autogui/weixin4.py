"""微信 4.x 的文本发送兼容层。

这里只适配旧 ``wechat_lock_send`` 所需的最小接口。业务层的收件人、
文案、Step 6 和调度机制仍由原代码决定。
"""

from __future__ import annotations

import ctypes
import os
import time
from pathlib import Path


class Weixin4Error(RuntimeError):
    pass


def _normalize_text(value: str) -> str:
    return "".join(str(value).split()).replace("（", "(").replace("）", ")")


class Weixin4TextClient:
    """对微信 4.x 主窗口提供旧发送器需要的最小文本接口。"""

    window_class = "Qt51514QWindowIcon"
    # Business code keeps the historical mi15 recipient name.  WeChat 4 on mf
    # exposes the same operational conversation under this local name.
    user_aliases = {"考勤中台": "考勤后台"}

    def __init__(self, *, evidence_dir: str | Path | None = None):
        self.hwnd = self._find_main_window()
        root = evidence_dir or Path(os.environ.get("TEMP", ".")) / "codeyun" / "attendance-wechat-mf"
        self.evidence_dir = Path(root)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _find_main_window() -> int:
        import win32gui

        matches: list[int] = []

        def collect(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            if win32gui.GetClassName(hwnd) != Weixin4TextClient.window_class:
                return
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            if right - left >= 900 and bottom - top >= 700:
                matches.append(hwnd)

        win32gui.EnumWindows(collect, None)
        if not matches:
            raise Weixin4Error("未找到已登录的微信 4.x 主窗口")
        return max(
            matches,
            key=lambda hwnd: (
                win32gui.GetWindowRect(hwnd)[2] - win32gui.GetWindowRect(hwnd)[0]
            ) * (
                win32gui.GetWindowRect(hwnd)[3] - win32gui.GetWindowRect(hwnd)[1]
            ),
        )

    def _show(self):
        ctypes.windll.user32.ShowWindow(self.hwnd, 9)
        ctypes.windll.user32.SwitchToThisWindow(self.hwnd, True)
        time.sleep(0.6)

    def _rect(self, hwnd: int | None = None):
        import win32gui

        return win32gui.GetWindowRect(hwnd or self.hwnd)

    def _capture_size(self, hwnd: int | None = None) -> tuple[int, int]:
        """Return the bitmap size expected by ``PrintWindow``.

        ``pyautogui`` opts the process into DPI awareness when it is imported.
        On a 150% display, ``GetWindowRect`` then reports physical pixels while
        Qt's ``PrintWindow`` output still uses logical pixels.  Allocating the
        physical size leaves a large black area and shifts every OCR crop.
        """
        import win32gui

        user32 = ctypes.windll.user32
        set_context = getattr(user32, "SetThreadDpiAwarenessContext", None)
        old_context = None
        try:
            if set_context:
                set_context.restype = ctypes.c_void_p
                # PrintWindow renders this Qt window in DPI-unaware logical
                # coordinates even when pyautogui made the caller DPI-aware.
                old_context = set_context(ctypes.c_void_p(-1))
            left, top, right, bottom = win32gui.GetWindowRect(hwnd or self.hwnd)
        finally:
            if set_context and old_context:
                set_context(ctypes.c_void_p(old_context))
        width, height = right - left, bottom - top
        return width, height

    def _click_relative(self, x_ratio: float, y_ratio: float, *, hwnd: int | None = None):
        import pyautogui

        left, top, right, bottom = self._rect(hwnd)
        pyautogui.click(left + (right - left) * x_ratio, top + (bottom - top) * y_ratio)

    def _capture(self, name: str, *, hwnd: int | None = None):
        import win32gui
        import win32ui
        from PIL import Image

        target_hwnd = hwnd or self.hwnd
        width, height = self._capture_size(target_hwnd)
        window_dc = win32gui.GetWindowDC(target_hwnd)
        source_dc = win32ui.CreateDCFromHandle(window_dc)
        memory_dc = source_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(source_dc, width, height)
        memory_dc.SelectObject(bitmap)
        try:
            ok = ctypes.windll.user32.PrintWindow(target_hwnd, memory_dc.GetSafeHdc(), 3)
            if not ok:
                raise Weixin4Error("微信窗口截图失败")
            info = bitmap.GetInfo()
            bits = bitmap.GetBitmapBits(True)
            image = Image.frombuffer(
                "RGB",
                (info["bmWidth"], info["bmHeight"]),
                bits,
                "raw",
                "BGRX",
                0,
                1,
            )
            path = self.evidence_dir / name
            image.save(path)
            return image
        finally:
            win32gui.DeleteObject(bitmap.GetHandle())
            memory_dc.DeleteDC()
            source_dc.DeleteDC()
            win32gui.ReleaseDC(target_hwnd, window_dc)

    @staticmethod
    def _ocr_payload(image) -> dict:
        # PaddleX 启动时的联网探测会拖慢本地已缓存模型的加载。
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from pyxllib.ai.ocr import ocr_text

        result = ocr_text(image)
        return result.json["res"] if hasattr(result, "json") else result

    @classmethod
    def _ocr_text(cls, image) -> str:
        return "\n".join(cls._ocr_payload(image).get("rec_texts", []))

    @classmethod
    def _resolve_user(cls, user: str) -> str:
        return cls.user_aliases.get(user, user)

    @staticmethod
    def _find_search_popup() -> int:
        import win32gui

        matches: list[int] = []

        def collect(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            if win32gui.GetClassName(hwnd) != "Qt51514QWindowToolSaveBits":
                return
            if win32gui.GetWindowText(hwnd) != "Weixin":
                return
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            if right - left >= 250 and bottom - top >= 150:
                matches.append(hwnd)

        win32gui.EnumWindows(collect, None)
        if not matches:
            raise Weixin4Error("未找到微信搜索结果浮层")
        return max(matches, key=lambda h: (win32gui.GetWindowRect(h)[2] - win32gui.GetWindowRect(h)[0]) * (win32gui.GetWindowRect(h)[3] - win32gui.GetWindowRect(h)[1]))

    def _select_search_result(self, user: str):
        popup = self._find_search_popup()
        image = self._capture("weixin4-search-result.png", hwnd=popup)
        payload = self._ocr_payload(image)
        texts = payload.get("rec_texts", [])
        boxes = payload.get("rec_boxes", [])
        expected = _normalize_text(user)
        for text, box in zip(texts, boxes):
            candidate = _normalize_text(text)
            # Small green group names occasionally turn 勤 into 勒 in OCR.
            # Permit one glyph error only for equal-length names; the active
            # chat title is still independently verified after the click.
            is_near = (
                len(expected) >= 4
                and len(candidate) == len(expected)
                and sum(a != b for a, b in zip(candidate, expected)) <= 1
            )
            if candidate != expected and not is_near:
                continue
            x1, y1, x2, y2 = map(float, box)
            self._click_relative(
                ((x1 + x2) / 2) / image.width,
                ((y1 + y2) / 2) / image.height,
                hwnd=popup,
            )
            return
        raise Weixin4Error(f"搜索结果中没有精确会话：{user!r}")

    def _current_chat(self) -> str:
        image = self._capture("weixin4-current-chat.png")
        width, height = image.size
        # The left conversation/search pane ends around x=30%.  Starting the
        # crop to its right is essential: otherwise a search result containing
        # ``user`` can be mistaken for the active chat title.
        header = image.crop((int(width * 0.31), 0, int(width * 0.88), int(height * 0.085)))
        return self._ocr_text(header)

    @staticmethod
    def _matches(actual: str, expected: str) -> bool:
        normalized_actual = _normalize_text(actual)
        normalized_expected = _normalize_text(expected)
        return bool(normalized_expected and normalized_expected in normalized_actual)

    def ChatWith(self, user: str, timeout: float = 5):
        import pyautogui
        import pyperclip

        actual_user = self._resolve_user(user)
        self._show()
        if self._matches(self._current_chat(), actual_user):
            return user

        old_clipboard = pyperclip.paste()
        try:
            self._click_relative(0.16, 0.05)
            pyautogui.hotkey("ctrl", "a")
            pyperclip.copy(actual_user)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(min(max(timeout / 2, 1.5), 3.0))
            # WeChat 4's Enter key can submit the search phrase into the
            # existing chat instead of opening a result.  Click the exact OCR
            # line in the separate search popup, then verify the right title.
            self._select_search_result(actual_user)
            time.sleep(min(max(timeout / 2, 2.0), 3.0))
        finally:
            pyperclip.copy(old_clipboard)

        actual = self._current_chat()
        return user if self._matches(actual, actual_user) else actual

    def SendMsg(self, text, user: str | None = None, **kwargs):
        import pyautogui
        import pyperclip

        message = str(text)
        actual_user = self._resolve_user(user) if user else None
        if actual_user and not self._matches(self._current_chat(), actual_user):
            raise Weixin4Error(f"发送前目标校验失败：期望 {user!r}")

        old_clipboard = pyperclip.paste()
        inserted = False
        sent = False
        try:
            self._click_relative(0.62, 0.86)
            # The unattended sender owns this draft area.  Always remove a
            # stale draft left by an interrupted/failed previous attempt.
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("backspace")
            pyperclip.copy(message)
            pyautogui.hotkey("ctrl", "v")
            inserted = True
            time.sleep(0.8)
            # 粘贴后、真正发送前再校验一次，错群时绝不按回车。
            if actual_user and not self._matches(self._current_chat(), actual_user):
                pyautogui.hotkey("ctrl", "a")
                pyautogui.press("backspace")
                raise Weixin4Error(f"落键前目标校验失败：期望 {user!r}")

            before = self._capture("weixin4-before-send.png")
            width, height = before.size
            input_area = before.crop(
                (int(width * 0.29), int(height * 0.73), width, height)
            )
            # Long tracebacks keep their tail visible in WeChat's scrolled
            # input box; the beginning may be outside the viewport.  Validate
            # with the ending marker so real error notifications can pass.
            marker = _normalize_text(message)[-8:]
            if marker and marker not in _normalize_text(self._ocr_text(input_area)):
                raise Weixin4Error("发送前输入框文本校验失败")

            # OCR 期间前台焦点可能被用户切走；重新激活后直接点击发送按钮。
            self._show()
            self._click_relative(0.95, 0.965)
            time.sleep(1.8)
            after = self._capture("weixin4-after-send.png")
            input_after = after.crop(
                (int(width * 0.29), int(height * 0.73), width, height)
            )
            # 消息气泡可能因长列表裁切导致 OCR 漏识别；输入框已由微信清空，
            # 才是稳定且不会把“已发送”误报成失败的提交结果证据。
            if marker and marker in _normalize_text(self._ocr_text(input_after)):
                raise Weixin4Error("点击发送后输入框仍保留原消息")
            sent = True
        finally:
            if inserted and not sent:
                # Fail closed: never leave the intended body as a live draft
                # that a later search/click can accidentally submit.
                try:
                    self._show()
                    self._click_relative(0.62, 0.86)
                    pyautogui.hotkey("ctrl", "a")
                    pyautogui.press("backspace")
                except Exception:
                    pass
            pyperclip.copy(old_clipboard)

    def AtAll(self, text, user):
        raise Weixin4Error("微信 4.x 兼容层暂不支持 @所有人")

    def SendFiles(self, files, user=None, **kwargs):
        raise Weixin4Error("微信 4.x 兼容层暂不支持文件发送")

    def SendUrlCard(self, url, user=None):
        raise Weixin4Error("微信 4.x 兼容层暂不支持链接卡片发送")
