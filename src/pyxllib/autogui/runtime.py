from __future__ import annotations

import time
from typing import Any

from .model import CurView, Shape, View


class Runtime:
    """行为树运行时上下文基类。

    子类负责实现截图、OCR、点击、数据读取等运行期能力；View/Shape 只借助
    Runtime 完成匹配和动作。
    """

    default_wait_click_timeout: float | None = None

    def get_cur_view(self, update: bool = False) -> CurView | View | None:
        raise NotImplementedError

    def get_views(self, group: str = "", recursive: bool = False) -> list[View]:
        raise NotImplementedError

    def find_view(self, group: str = "") -> View | None:
        for view in self.get_views(group):
            if view.is_match(self, include_descendants=bool(group)):
                return view
        return None

    def wait_view(self, *views: CurView | View | int, timeout: float | None = None, interval_action: int = 1):
        """等待任一指定 view 匹配当前画面。

        这是生成器动作：未命中时让出一次行动力，下一轮 tick 再刷新识别。
        """

        target_views = [self._resolve_wait_view(view) for view in views]
        target_views = [view for view in target_views if view is not None]
        start = time.monotonic()
        while True:
            self.get_cur_view(update=True)
            for view in target_views:
                if view.is_match(self):
                    return view
            if timeout is not None and time.monotonic() - start >= float(timeout):
                expected = ", ".join(str(view.id or view.title or view.filename) for view in target_views) or "<empty>"
                raise TimeoutError(f"等待 view 超时：{expected}")
            yield interval_action

    def goto_view(self, view: CurView | View | int) -> Any:
        """场景移动到目标 view。

        具体路径规划、点击后的落点学习和跳转频次维护由业务 Runtime 实现；实现中
        应复用 wait_view 等待目标或中间落点，避免点击后立即判定造成不稳定。
        """

        raise NotImplementedError

    def match_shape(self, shape: Shape) -> bool:
        raise NotImplementedError

    def click_shape(self, view: View, shape: Shape) -> Any:
        raise NotImplementedError

    def wait_click_shape(self, view: View, shape: Shape, *, timeout: float | None = None, interval_action: int = 1) -> Any:
        """等待 shape 在当前画面可匹配后点击。

        子类的 ``match_shape`` 可缓存本轮定位结果，``click_shape`` 再复用该结果。
        """

        effective_timeout = self.default_wait_click_timeout if timeout is None else timeout
        start = time.monotonic()
        while True:
            matched = shape.is_match(self)
            self.on_wait_click_poll(view, shape, matched)
            if matched:
                return self.click_shape(view, shape)
            if effective_timeout is not None and time.monotonic() - start >= float(effective_timeout):
                raise TimeoutError(f"等待点击超时：{shape.title or shape.raw.get('id') or '<shape>'}")
            yield interval_action

    def on_wait_click_poll(self, view: View, shape: Shape, matched: bool) -> None:
        """等待点击每轮匹配后的观测钩子。"""

        return None

    def get_view(self, view: CurView | View | int) -> View | None:
        if isinstance(view, CurView):
            return view.view
        if isinstance(view, View):
            return view
        for candidate in self.get_views("", recursive=True):
            if candidate.id == int(view):
                return candidate
        return None

    def _resolve_wait_view(self, view: CurView | View | int) -> View | None:
        return self.get_view(view)

    def load_shape(self, shape: Shape, *, ratio: float = 0.5, duration: float = 1.5):
        attrs = self._attrs()
        attrs["load_new"] = False
        if not shape.content_direction:
            return
        before = self.shape_load_signature(shape)
        self.drag_shape_content(shape, ratio=ratio, duration=duration)
        yield 1
        after = self.shape_load_signature(shape)
        attrs["load_new"] = self.shape_content_loaded_new(shape, before, after)

    def drag_shape_content(self, shape: Shape, *, ratio: float = 0.5, duration: float = 1.5) -> Any:
        raise NotImplementedError

    def shape_load_signature(self, shape: Shape) -> Any:
        return None

    def shape_content_loaded_new(self, shape: Shape, before: Any, after: Any) -> bool:
        if before is None or after is None:
            return True
        return before != after

    def _attrs(self) -> dict[str, Any]:
        attrs = getattr(self, "attrs", None)
        if not isinstance(attrs, dict):
            attrs = {}
            setattr(self, "attrs", attrs)
        return attrs
