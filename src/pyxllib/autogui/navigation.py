from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .matching import ImagePredicateFunc
from .model import flatten_shapes, image_number


@dataclass(frozen=True)
class SceneNavigator:
    """基于帧树标注规划场景跳转路径。"""

    tree: list[dict[str, Any]]

    def jump_target_text(self, shape: dict[str, Any]) -> str:
        return str(shape.get("sceneJumpTarget") or "").strip()

    def parse_scene_jump_entries(self, value: Any) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for raw_token in str(value or "").split(","):
            token = raw_token.strip()
            if not token:
                continue
            count = 0
            match = re.match(r"^(.*?)\((\d+)\)$", token)
            if match:
                token = match.group(1).strip()
                count = int(match.group(2))
            if token:
                entries.append({"label": token, "count": count})
        return entries

    def serialize_scene_jump_entries(self, entries: list[dict[str, Any]]) -> str:
        normalized: list[dict[str, Any]] = []
        for entry in entries:
            label = str(entry.get("label") or "").strip()
            if not label:
                continue
            count = max(0, int(entry.get("count") or 0))
            normalized.append({"label": label, "count": count})
        normalized.sort(key=lambda item: int(item.get("count") or 0), reverse=True)
        return ",".join(
            f"{item['label']}({item['count']})" if int(item.get("count") or 0) > 0 else item["label"]
            for item in normalized
        )

    def increment_scene_jump_target(self, shape: dict[str, Any], target_scene_id: int) -> bool:
        current_text = self.jump_target_text(shape)
        if current_text in {"-1", "0"}:
            return False
        entries = self.parse_scene_jump_entries(current_text)
        for entry in entries:
            if self.scene_jump_label_number(entry.get("label")) == target_scene_id:
                entry["count"] = int(entry.get("count") or 0) + 1
                shape["sceneJumpTarget"] = self.serialize_scene_jump_entries(entries)
                return True
        return False

    def scene_jump_label_number(self, label: Any) -> int | None:
        text = str(label or "").strip()
        if text.startswith("#"):
            text = text[1:].strip()
        return int(text) if text.isdecimal() else None

    def collect_folder_image_numbers(self, node: dict[str, Any]) -> list[int]:
        result: list[int] = []
        children = node.get("children")
        if not isinstance(children, list):
            return result
        for child in children:
            if not isinstance(child, dict):
                continue
            if child.get("type") == "image":
                number = image_number(child)
                if number is not None:
                    result.append(number)
            result.extend(self.collect_folder_image_numbers(child))
        return result

    def resolve_scene_jump_label(self, label: Any) -> list[int]:
        number = self.scene_jump_label_number(label)
        if number is not None:
            return [number]
        target = str(label or "").strip()
        if not target:
            return []
        found: list[int] = []

        def visit(items: list[dict[str, Any]]) -> None:
            for item in items:
                if not isinstance(item, dict):
                    continue
                if str(item.get("title") or "").strip() == target:
                    if item.get("type") == "image":
                        number = image_number(item)
                        if number is not None:
                            found.append(number)
                    elif item.get("type") == "folder":
                        found.extend(self.collect_folder_image_numbers(item))
                children = item.get("children")
                if isinstance(children, list):
                    visit([child for child in children if isinstance(child, dict)])

        visit(self.tree)
        return found

    def scene_jump_target_ids(self, shape: dict[str, Any]) -> list[int]:
        result: list[int] = []
        for entry in self.parse_scene_jump_entries(self.jump_target_text(shape)):
            for scene_id in self.resolve_scene_jump_label(entry.get("label")):
                if scene_id not in result:
                    result.append(scene_id)
        return result

    def resolve_scene_image_title_ids(self, title: str) -> list[int]:
        target = title.strip()
        if not target:
            return []
        found: list[int] = []

        def visit(items: list[dict[str, Any]]) -> None:
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image" and str(item.get("title") or "").strip() == target:
                    number = image_number(item)
                    if number is not None and number not in found:
                        found.append(number)
                children = item.get("children")
                if isinstance(children, list):
                    visit([child for child in children if isinstance(child, dict)])

        visit(self.tree)
        return found

    def implicit_parent_return_target_ids(
        self,
        shape: dict[str, Any],
        parent_image: dict[str, Any] | None,
        parent_folder_title: str = "",
    ) -> list[int]:
        if self.jump_target_text(shape):
            return []
        title = str(shape.get("title") or "").strip()
        if title not in {"离开", "返回", "关闭", "关闭下方菜单"}:
            return []
        if parent_image:
            parent_id = image_number(parent_image)
            if parent_id is not None:
                return [parent_id]
        return self.resolve_scene_image_title_ids(parent_folder_title)

    def scene_jump_edges(self) -> dict[int, list[dict[str, Any]]]:
        edges: dict[int, list[dict[str, Any]]] = {}

        def visit(
            items: list[dict[str, Any]],
            parent_image: dict[str, Any] | None = None,
            parent_folder_title: str = "",
        ) -> None:
            for item in items:
                if not isinstance(item, dict):
                    continue
                current_parent_image = parent_image
                current_parent_folder_title = parent_folder_title
                if item.get("type") == "folder":
                    current_parent_folder_title = str(item.get("title") or "").strip() or parent_folder_title
                if item.get("type") == "image":
                    source_id = image_number(item)
                    if source_id is not None:
                        for shape in flatten_shapes(item.get("shapes")):
                            if shape.get("kind") == "group":
                                continue
                            target_text = self.jump_target_text(shape)
                            if target_text in {"-1", "0"}:
                                continue
                            target_ids = (
                                self.scene_jump_target_ids(shape)
                                if target_text
                                else self.implicit_parent_return_target_ids(shape, parent_image, parent_folder_title)
                            )
                            if target_ids:
                                edges.setdefault(source_id, []).append({
                                    "source_id": source_id,
                                    "image": item,
                                    "shape": shape,
                                    "target_ids": target_ids,
                                })
                    current_parent_image = item
                children = item.get("children")
                if isinstance(children, list):
                    visit([child for child in children if isinstance(child, dict)], current_parent_image, current_parent_folder_title)

        visit(self.tree)
        return edges

    def find_scene_route(self, start_scene_id: int, target_scene_id: int) -> list[dict[str, Any]] | None:
        if start_scene_id == target_scene_id:
            return []
        edges = self.scene_jump_edges()
        queue: list[tuple[int, list[dict[str, Any]]]] = [(start_scene_id, [])]
        visited = {start_scene_id}
        while queue:
            scene_id, route = queue.pop(0)
            for edge in edges.get(scene_id, []):
                for next_scene_id in edge["target_ids"]:
                    if next_scene_id in visited:
                        continue
                    next_route = [*route, edge]
                    if next_scene_id == target_scene_id:
                        return next_route
                    visited.add(next_scene_id)
                    queue.append((next_scene_id, next_route))
        return None

    def confirmation_scene_ids(self, is_confirmation_image: ImagePredicateFunc) -> list[int]:
        """收集会参与中间确认的场景帧编号。"""

        result: list[int] = []

        def visit(items: list[dict[str, Any]]) -> None:
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image" and is_confirmation_image(item):
                    scene_id = image_number(item)
                    if scene_id is not None and scene_id not in result:
                        result.append(scene_id)
                children = item.get("children")
                if isinstance(children, list):
                    visit([child for child in children if isinstance(child, dict)])

        visit(self.tree)
        return result

    def route_candidate_ids(
        self,
        target_scene_id: int,
        *,
        confirmation_scene_ids: list[int] | None = None,
    ) -> list[int]:
        """生成到达目标场景时优先识别的候选场景编号。"""

        edges = self.scene_jump_edges()
        result: list[int] = [target_scene_id]
        for scene_id in confirmation_scene_ids or []:
            if scene_id not in result:
                result.append(scene_id)
        routed_sources: list[tuple[int, int]] = []
        for source_scene_id in sorted(edges):
            if source_scene_id == target_scene_id:
                continue
            route = self.find_scene_route(source_scene_id, target_scene_id)
            if route is not None:
                routed_sources.append((len(route), source_scene_id))
        for _route_length, source_scene_id in sorted(routed_sources):
            result.append(source_scene_id)
        return result
