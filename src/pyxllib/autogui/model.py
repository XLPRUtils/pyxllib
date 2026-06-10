from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable


class MatchRole(IntEnum):
    """标注匹配角色。"""

    off = 0
    required = 1
    decisive = 2


def normalize_match_role(value: Any, default: MatchRole = MatchRole.off) -> MatchRole:
    """把前端/旧数据中的匹配角色归一成 :class:`MatchRole`。"""

    if isinstance(value, MatchRole):
        return value
    if isinstance(value, bool):
        return MatchRole.required if value else MatchRole.off
    if isinstance(value, int):
        try:
            return MatchRole(value)
        except ValueError:
            return default
    text = str(value or "").strip().lower()
    aliases = {
        "": default,
        "off": MatchRole.off,
        "none": MatchRole.off,
        "无": MatchRole.off,
        "0": MatchRole.off,
        "required": MatchRole.required,
        "must": MatchRole.required,
        "必": MatchRole.required,
        "1": MatchRole.required,
        "decisive": MatchRole.decisive,
        "any": MatchRole.decisive,
        "定": MatchRole.decisive,
        "2": MatchRole.decisive,
    }
    return aliases.get(text, default)


@dataclass(frozen=True)
class Shape:
    """帧树中的一个标注框。"""

    raw: dict[str, Any]
    parent_view: View | None = None
    parent_shape: Shape | None = None

    @property
    def title(self) -> str:
        return str(self.raw.get("title") or "").strip()

    @property
    def kind(self) -> str:
        return str(self.raw.get("kind") or "").strip()

    @property
    def attrs(self) -> dict[str, Any]:
        return self.raw

    @property
    def content_direction(self) -> str:
        value = self.raw.get("contentDirection", self.raw.get("content_direction", self.raw.get("内容方向", "")))
        text = str(value or "").strip().lower()
        aliases = {
            "": "",
            "none": "",
            "off": "",
            "无": "",
            "down": "down",
            "下": "down",
            "up": "up",
            "上": "up",
            "right": "right",
            "右": "right",
            "left": "left",
            "左": "left",
        }
        return aliases.get(text, text)

    @property
    def is_scene_identity(self) -> bool:
        return self.scene_identity_role is not MatchRole.off

    @property
    def scene_identity_role(self) -> MatchRole:
        legacy_default = MatchRole.required if bool(self.raw.get("isSceneIdentity")) else MatchRole.off
        return normalize_match_role(self.raw.get("sceneIdentityRole"), legacy_default)

    @property
    def image_match_role(self) -> MatchRole:
        default = self.scene_identity_role if self.scene_identity_role is not MatchRole.off else MatchRole.off
        return normalize_match_role(self.raw.get("imageMatchRole"), default)

    @property
    def ocr_match_role(self) -> MatchRole:
        default = MatchRole.required if bool(self.raw.get("ocrEnabled")) and str(self.raw.get("ocrText") or "").strip() else MatchRole.off
        return normalize_match_role(self.raw.get("ocrMatchRole"), default)

    @property
    def scene_jump_target(self) -> str:
        return str(self.raw.get("sceneJumpTarget") or "").strip()

    def children(self) -> list[Shape]:
        children = self.raw.get("children")
        if not isinstance(children, list):
            return []
        return [Shape(child, parent_view=self.parent_view, parent_shape=self) for child in children if isinstance(child, dict)]

    def descendants(self, *, include_self: bool = True) -> list[Shape]:
        result: list[Shape] = [self] if include_self else []
        for child in self.children():
            result.extend(child.descendants(include_self=True))
        return result

    def is_match(self, runtime: Any) -> bool:
        """判断当前 shape 是否匹配 runtime 的当前画面。"""

        matcher = getattr(runtime, "match_shape", None)
        if matcher is None:
            raise RuntimeError("runtime 缺少 match_shape(shape) 能力")
        return bool(matcher(self))

    def click(self, runtime: Any) -> Any:
        """点击当前 shape。

        具体点击实现属于 Runtime；Shape 只携带静态标注和所属 View。
        """

        if not isinstance(self.parent_view, View):
            raise RuntimeError("shape 缺少 parent_view，无法点击")
        clicker = getattr(runtime, "click_shape", None)
        if clicker is None:
            raise RuntimeError("runtime 缺少 click_shape(view, shape) 能力")
        return clicker(self.parent_view, self)

    def wait_click(self, runtime: Any, *, timeout: float | None = None) -> Any:
        """等待当前 shape 命中后点击。

        这是生成器动作。Runtime 负责等待、定位结果复用和实际点击。
        """

        if not isinstance(self.parent_view, View):
            raise RuntimeError("shape 缺少 parent_view，无法等待点击")
        wait_clicker = getattr(runtime, "wait_click_shape", None)
        if wait_clicker is None:
            raise RuntimeError("runtime 缺少 wait_click_shape(view, shape) 能力")
        return (yield from wait_clicker(self.parent_view, self, timeout=timeout))

    def load(self, runtime: Any, ratio: float = 0.5, duration: float = 1.5):
        """滚动加载当前 shape 表示的内容窗口。

        runtime.attrs["load_new"] 表示本次滚动后是否仍有新内容。底层如何截图、
        拖拽、比较内容变化由 Runtime 决定。
        """

        loader = getattr(runtime, "load_shape", None)
        if loader is None:
            raise RuntimeError("runtime 缺少 load_shape(shape, ratio, duration) 能力")
        yield from loader(self, ratio=ratio, duration=duration)

    def box(self, image: View | dict[str, Any] | None = None) -> dict[str, float | str]:
        view = image if isinstance(image, View) else self.parent_view
        raw_image = view.raw if isinstance(view, View) else image
        width, height = frame_size(raw_image if isinstance(raw_image, dict) else None)
        return {
            "name": self.title,
            "x": float(self.raw.get("x") or 0) * width,
            "y": float(self.raw.get("y") or 0) * height,
            "w": max(1, float(self.raw.get("w") or 0) * width),
            "h": max(1, float(self.raw.get("h") or 0) * height),
        }


@dataclass(frozen=True)
class View:
    """一帧画面标注。

    View 保存数据库/帧树中的静态标注；当前画面、点击、截图、OCR 等运行期能力
    由 Runtime 提供。
    """

    raw: dict[str, Any] | None = None

    @property
    def id(self) -> int | None:
        return image_number(self.raw)

    @property
    def title(self) -> str:
        raw = self.raw if isinstance(self.raw, dict) else {}
        return str(raw.get("title") or "").strip()

    @property
    def filename(self) -> str:
        raw = self.raw if isinstance(self.raw, dict) else {}
        return str(raw.get("filename") or "").strip()

    def get_shapes(self, *, include_groups: bool = True, include_descendants: bool = True, **filters: Any) -> list[Shape]:
        """返回当前 view 中的 shape 标注。

        目前基础设施先提供 include_groups 和显式属性筛选；更复杂的“场景标识>0”
        这类表达式可以由业务层/查询层逐步补。
        """

        raw = self.raw if isinstance(self.raw, dict) else {}
        shapes = raw.get("shapes")
        if not isinstance(shapes, list):
            return []
        result: list[Shape] = []
        for item in shapes:
            if not isinstance(item, dict):
                continue
            shape = Shape(item, parent_view=self)
            if include_descendants:
                result.extend(shape.descendants(include_self=True))
            else:
                result.append(shape)
        if not include_groups:
            result = [shape for shape in result if shape.kind != "group"]
        for key, value in filters.items():
            result = [shape for shape in result if shape.raw.get(key) == value]
        return result

    def get_shape(self, title: str | None = None, *, contains: bool = False, **filters: Any) -> Shape | None:
        titles = [title] if title is not None else []
        for shape in self.get_shapes(**filters):
            if not shape.title:
                continue
            if not titles:
                return shape
            if any((contains and target in shape.title) or (not contains and shape.title == target) for target in titles):
                return shape
        return None

    def is_match(self, runtime: Any, *, include_descendants: bool = True) -> bool:
        """借助场景标识 shape 判断 runtime 当前画面是否匹配当前 view。"""

        shapes = [
            shape
            for shape in self.get_shapes(include_groups=False, include_descendants=include_descendants)
            if shape.is_scene_identity
        ]
        if not shapes:
            return False

        decisive_shapes = [shape for shape in shapes if shape.scene_identity_role is MatchRole.decisive]
        if any(shape.is_match(runtime) for shape in decisive_shapes):
            return True

        required_shapes = [shape for shape in shapes if shape.scene_identity_role is MatchRole.required]
        return bool(required_shapes) and all(shape.is_match(runtime) for shape in required_shapes)

    def close(self, runtime: Any) -> Any:
        """尝试关闭当前帧。

        语义上 close 动作发生在 runtime 上，View 只提供“该怎么关”的静态标注依据。
        """

        from .actions import CloseActionPlanner

        shape = CloseActionPlanner().choose_close_shape([shape.raw for shape in self.get_shapes()])
        if shape is not None:
            return runtime.click_shape(self, Shape(shape, parent_view=self))
        raise RuntimeError(f"帧「{self.title or self.id}」缺少可关闭标注")


@dataclass(frozen=True)
class DbView(View):
    """数据库/标注树中的静态帧。

    ``View`` 早期已经按静态帧设计；保留它作为通用基类，新增 ``DbView`` 用来
    在业务代码和伪代码里明确区分“数据库帧”和“当前运行帧”。
    """


@dataclass(frozen=True)
class CurView:
    """Runtime 识别到的当前帧。

    CurView 只描述一次运行期观测结果；静态标注仍由 ``view`` 指向的 DbView/View
    承载，点击、截图、OCR 等能力仍由 Runtime 执行。
    """

    view: View | None = None
    score: float = 0.0
    frame: Any = None
    scene_id: int | None = None
    action_shape: Shape | dict[str, Any] | None = None
    meta: dict[str, Any] | None = None

    @property
    def id(self) -> int | None:
        return self.view.id if isinstance(self.view, View) else self.scene_id

    @property
    def title(self) -> str:
        return self.view.title if isinstance(self.view, View) else ""

    @property
    def filename(self) -> str:
        return self.view.filename if isinstance(self.view, View) else ""

    @property
    def raw(self) -> dict[str, Any] | None:
        return self.view.raw if isinstance(self.view, View) else None

    def close(self, runtime: Any) -> Any:
        """按当前识别到的静态帧标注关闭画面。"""

        if not isinstance(self.view, View):
            raise RuntimeError("当前帧缺少对应的静态 View，无法关闭")
        return self.view.close(runtime)


def image_number(image: dict[str, Any] | None) -> int | None:
    """从帧树 image 节点提取编号。"""

    if not isinstance(image, dict):
        return None
    for text in (str(image.get("filename") or ""), str(image.get("title") or ""), str(image.get("id") or "")):
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1))
    return None


def index_images(nodes: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """把帧树索引成 ``{编号: image节点}``。"""

    result: dict[int, dict[str, Any]] = {}

    def visit(items: Iterable[dict[str, Any]]) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                number = image_number(item)
                if number is not None:
                    result[number] = item
            children = item.get("children")
            if isinstance(children, list):
                visit(child for child in children if isinstance(child, dict))

    visit(nodes)
    return result


def flatten_shapes(shapes: Any) -> list[dict[str, Any]]:
    """拉平原始 shape 树，保留原 dict 节点。"""

    result: list[dict[str, Any]] = []
    if not isinstance(shapes, list):
        return result
    for item in shapes:
        if not isinstance(item, dict):
            continue
        result.append(item)
        result.extend(flatten_shapes(item.get("children")))
    return result


def frame_size(image: dict[str, Any] | None) -> tuple[int, int]:
    """返回帧尺寸。"""

    if not isinstance(image, dict):
        return 900, 1600
    return max(1, int(image.get("width") or 900)), max(1, int(image.get("height") or 1600))
