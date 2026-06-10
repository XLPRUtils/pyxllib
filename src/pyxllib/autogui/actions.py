from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import flatten_shapes, frame_size


@dataclass(frozen=True)
class CloseActionPlanner:
    """从标注列表中选择关闭/退出动作。"""

    title_priorities: tuple[str, ...] = ("关闭", "退出", "空白", "确认")
    independent_exit_target: str = "-1"

    def choose_close_shape(
        self,
        shapes: list[dict[str, Any]],
        *,
        include_groups: bool = False,
        include_independent_exit: bool = False,
    ) -> dict[str, Any] | None:
        candidates = [shape for shape in flatten_shapes(shapes) if include_groups or shape.get("kind") != "group"]
        for title in self.title_priorities:
            for shape in candidates:
                if str(shape.get("title") or "").strip() == title:
                    return shape
        if include_independent_exit:
            for shape in candidates:
                if self.is_independent_exit_shape(shape):
                    return shape
        return None

    def is_independent_exit_shape(self, shape: dict[str, Any]) -> bool:
        return str(shape.get("sceneJumpTarget") or "").strip() == self.independent_exit_target




@dataclass(frozen=True)
class ActionPlanner:
    """把帧树坐标转换为运行时点击/拖拽动作参数。"""

    input_backend: str = "adb"
    mode: str = "screen"
    area: str = "client"
    rotate: str = "0"

    def shape_box(self, image: dict[str, Any], shape: dict[str, Any]) -> dict[str, float | str]:
        width, height = frame_size(image)
        return {
            "name": str(shape.get("title") or ""),
            "x": float(shape.get("x") or 0) * width,
            "y": float(shape.get("y") or 0) * height,
            "w": max(1, float(shape.get("w") or 0) * width),
            "h": max(1, float(shape.get("h") or 0) * height),
        }

    def shape_center(self, image: dict[str, Any], shape: dict[str, Any]) -> tuple[float, float]:
        box = self.shape_box(image, shape)
        return (
            float(box.get("x") or 0) + float(box.get("w") or 0) / 2,
            float(box.get("y") or 0) + float(box.get("h") or 0) / 2,
        )

    def click_shape_payload(self, image: dict[str, Any], shape: dict[str, Any]) -> dict[str, Any]:
        x, y = self.shape_center(image, shape)
        return self.click_point_payload(image, x, y, clamp=False)

    def click_point_payload(self, image: dict[str, Any], x: float, y: float, *, clamp: bool = True) -> dict[str, Any]:
        width, height = frame_size(image)
        click_x = self._clamp(x, width) if clamp else float(x)
        click_y = self._clamp(y, height) if clamp else float(y)
        return {
            "x": click_x,
            "y": click_y,
            "mode": self.mode,
            "area": self.area,
            "rotate": self.rotate,
            "fixed_width": width,
            "fixed_height": height,
            "frame_width": width,
            "frame_height": height,
            "input_backend": self.input_backend,
        }

    def drag_point_payload(
        self,
        image: dict[str, Any],
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        *,
        duration_ms: int = 300,
    ) -> dict[str, Any]:
        width, height = frame_size(image)
        return {
            "start_x": self._clamp(start_x, width),
            "start_y": self._clamp(start_y, height),
            "end_x": self._clamp(end_x, width),
            "end_y": self._clamp(end_y, height),
            "duration_ms": duration_ms,
            "mode": self.mode,
            "area": self.area,
            "rotate": self.rotate,
            "fixed_width": width,
            "fixed_height": height,
            "frame_width": width,
            "frame_height": height,
            "input_backend": self.input_backend,
        }

    def drag_shape_content_points(
        self,
        image: dict[str, Any],
        shape: dict[str, Any],
        *,
        direction: str = "down",
        ratio: float = 0.5,
    ) -> tuple[float, float, float, float]:
        box = self.shape_box(image, shape)
        left = float(box.get("x") or 0)
        top = float(box.get("y") or 0)
        width = float(box.get("w") or 0)
        height = float(box.get("h") or 0)
        ratio = max(0.0, min(1.0, float(ratio)))
        direction = str(direction or "down").strip().lower()
        if direction in {"up", "down"}:
            x = left + width / 2
            if direction == "up":
                return x, top + height * (0.5 - ratio / 2), x, top + height * (0.5 + ratio / 2)
            return x, top + height * (0.5 + ratio / 2), x, top + height * (0.5 - ratio / 2)
        y = top + height / 2
        if direction == "left":
            return left + width * (0.5 - ratio / 2), y, left + width * (0.5 + ratio / 2), y
        return left + width * (0.5 + ratio / 2), y, left + width * (0.5 - ratio / 2), y

    def drag_shape_content_payload(
        self,
        image: dict[str, Any],
        shape: dict[str, Any],
        *,
        direction: str = "down",
        ratio: float = 0.5,
        duration: float = 1.5,
    ) -> dict[str, Any]:
        start_x, start_y, end_x, end_y = self.drag_shape_content_points(image, shape, direction=direction, ratio=ratio)
        return self.drag_point_payload(
            image,
            start_x,
            start_y,
            end_x,
            end_y,
            duration_ms=max(0, int(float(duration) * 1000)),
        )

    def _clamp(self, value: float, size: int) -> float:
        return max(0.0, min(float(size - 1), float(value)))
