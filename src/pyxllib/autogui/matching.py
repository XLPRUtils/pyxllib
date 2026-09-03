from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .model import MatchRole, View, image_number, normalize_match_role


SceneScoreFunc = Callable[[dict[str, Any], dict[str, Any], str], float]
SceneThresholdFunc = Callable[[int], float]
ImageForKeyFunc = Callable[[dict[str, Any], str], dict[str, Any] | None]
KeyThresholdFunc = Callable[[str], float]
ShapeScoreFunc = Callable[[dict[str, Any], dict[str, Any], dict[str, Any], str], float]
ShapeOcrScoreFunc = Callable[[dict[str, Any], dict[str, Any], dict[str, Any], str], float]
DetailLogFunc = Callable[[str], None]
ImagePredicateFunc = Callable[[dict[str, Any]], bool]


def _format_elapsed_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.2f}秒"
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes}分{remaining:05.2f}秒"


@dataclass(frozen=True)
class SceneScorer:
    """根据场景标识 shape 合成单帧场景分数。"""

    shape_score: ShapeScoreFunc
    shape_ocr_score: ShapeOcrScoreFunc
    threshold: float = 80.0
    match_planner: ShapeMatchPlanner | None = None
    log_detail: DetailLogFunc | None = None

    def scene_identity_shape_score(
        self,
        ctx: dict[str, Any],
        image: dict[str, Any],
        shape: dict[str, Any],
        frame_data_url: str,
    ) -> float:
        planner = self.match_planner or ShapeMatchPlanner()
        image_role = planner.image_role(shape)
        ocr_role = planner.ocr_role(shape)
        scores: list[tuple[str, float]] = []
        if image_role != "off":
            scores.append((image_role, float(self.shape_score(ctx, image, shape, frame_data_url) or 0)))
        if ocr_role != "off" and str(shape.get("ocrText") or "").strip():
            try:
                scores.append((ocr_role, float(self.shape_ocr_score(ctx, image, shape, frame_data_url) or 0)))
            except Exception as exc:
                if self.log_detail is not None:
                    self.log_detail(f"OCR匹配失败：{image.get('title')} / {shape.get('title')}：{exc}")
                scores.append((ocr_role, 0.0))
        if not scores:
            return 0.0
        required_scores = [score for role, score in scores if role == "required"]
        if required_scores and any(score < float(self.threshold) for score in required_scores):
            return 0.0
        return max(score for _role, score in scores)

    def scene_score(self, ctx: dict[str, Any], image: dict[str, Any], frame_data_url: str) -> float:
        scores = [
            self.scene_identity_shape_score(ctx, image, shape.raw, frame_data_url)
            for shape in View(image).get_shapes(include_groups=False)
            if shape.is_scene_identity
        ]
        return min(scores) if scores else 0.0


@dataclass(frozen=True)
class SceneRecognizer:
    """根据候选帧分数识别当前场景。"""

    score_image: SceneScoreFunc
    threshold_for_scene_id: SceneThresholdFunc
    image_for_key: ImageForKeyFunc | None = None
    threshold_for_key: KeyThresholdFunc | None = None
    max_parallel_workers: int = 32
    max_candidate_batch_size: int = 16

    def _scene_tree_nodes(self, tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """把资产树投影为 root layer 队列 + frame/subframe structure。

        `layer` 只参与 root frame 的默认识别候选队列；image.children 中的
        image 才构成 frame/subframe 的树形细化关系。
        """

        nodes: list[dict[str, Any]] = []

        def visit(items: list[dict[str, Any]], parent_ids: tuple[int, ...], depth: int) -> None:
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "folder":
                    children = item.get("children")
                    if isinstance(children, list):
                        visit([child for child in children if isinstance(child, dict)], parent_ids, depth)
                    continue
                if item.get("type") == "image":
                    scene_id = image_number(item)
                    current_parent_ids = parent_ids
                    current_depth = depth
                    if scene_id is not None:
                        view = View(item)
                        nodes.append({
                            "scene_id": int(scene_id),
                            "image": item,
                            "parent_ids": parent_ids,
                            "depth": depth,
                            "layer": int(view.layer),
                            "order": len(nodes),
                        })
                        current_parent_ids = (*parent_ids, int(scene_id))
                        current_depth = depth + 1
                    children = item.get("children")
                    if isinstance(children, list):
                        visit([child for child in children if isinstance(child, dict)], current_parent_ids, current_depth)
                    continue

        visit(tree, (), 0)
        return nodes

    def _scene_tree_candidate_ids(
        self,
        ctx: dict[str, Any],
        *,
        preferred_scene_ids: list[int] | None = None,
    ) -> list[int]:
        tree = ctx.get("asset_tree")
        if not isinstance(tree, list):
            return []
        nodes = self._scene_tree_nodes(tree)
        existing = {
            int(scene_id)
            for scene_id, image in (ctx.get("images") or {}).items()
            if isinstance(image, dict)
        }
        by_id = {int(node["scene_id"]): node for node in nodes if int(node["scene_id"]) in existing}
        result: list[int] = []
        if preferred_scene_ids is not None:
            return list(dict.fromkeys(
                int(scene_id)
                for scene_id in preferred_scene_ids
                if int(scene_id) in by_id
            ))
        roots = sorted(
            [node for node in nodes if not node["parent_ids"] and int(node["scene_id"]) in existing],
            key=lambda node: int(node["order"]),
        )
        for layer in (1, 2):
            for root in roots:
                if int(root["layer"]) != layer:
                    continue
                root_id = int(root["scene_id"])
                for node in nodes:
                    scene_id = int(node["scene_id"])
                    if (
                        scene_id not in existing
                        or scene_id in result
                        or int(node.get("layer", 3)) > 2
                    ):
                        continue
                    if scene_id == root_id or root_id in [int(parent_id) for parent_id in node["parent_ids"]]:
                        result.append(scene_id)
        return result

    def scene_matches_id(self, scene_id: int, score: float) -> bool:
        return float(score) >= float(self.threshold_for_scene_id(scene_id))

    def _weak_similarity_scene(self, ctx: dict[str, Any], scene_id: int) -> bool:
        image = (ctx.get("images") or {}).get(int(scene_id))
        if not isinstance(image, dict):
            return False
        if int(View(image).layer) != 3:
            return False
        return not any(shape.is_scene_identity for shape in View(image).get_shapes(include_groups=False))

    def _score_scene_candidates(
        self,
        ctx: dict[str, Any],
        frame_data_url: str,
        scene_ids: list[int],
    ) -> dict[int, float]:
        images = ctx.get("images") or {}
        ids = [int(scene_id) for scene_id in scene_ids if isinstance(images.get(int(scene_id)), dict)]
        if not ids:
            return {}

        def score(scene_id: int) -> tuple[int, float]:
            image = images.get(int(scene_id))
            if not isinstance(image, dict):
                return int(scene_id), 0.0
            return int(scene_id), float(self.score_image(ctx, image, frame_data_url))

        if len(ids) <= 1:
            return dict(score(scene_id) for scene_id in ids)
        workers = max(1, min(len(ids), int(self.max_parallel_workers or 1)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scene-match") as executor:
            return dict(executor.map(score, ids))

    def identify_scene_tree_number(
        self,
        ctx: dict[str, Any],
        frame_data_url: str,
        *,
        preferred_scene_ids: list[int] | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> tuple[int | None, float]:
        if preferred_scene_ids is not None:
            # Layer 0 is the caller's exact dynamic candidate list.  Asset-tree
            # parentage must not add ancestors, descendants, or default scenes.
            return self.identify_scene_number(
                ctx,
                frame_data_url,
                preferred_scene_ids=preferred_scene_ids,
                trace=trace,
            )

        def emit(event: dict[str, Any]) -> None:
            if trace is not None:
                trace.append(event)

        flat_candidate_ids = self._scene_tree_candidate_ids(ctx, preferred_scene_ids=preferred_scene_ids)
        if not flat_candidate_ids:
            emit({
                "event": "fallback_flat",
                "reason": "asset_tree_candidates_empty",
                "preferred_scene_ids": [int(item) for item in preferred_scene_ids or []],
            })
            return self.identify_scene_number(ctx, frame_data_url, preferred_scene_ids=preferred_scene_ids, trace=trace)
        images = ctx.get("images") or {}
        tree = ctx.get("asset_tree") if isinstance(ctx.get("asset_tree"), list) else []
        node_by_id = {
            int(node["scene_id"]): node
            for node in self._scene_tree_nodes(tree)
        }
        children_by_parent: dict[int | None, list[int]] = {}
        for scene_id in flat_candidate_ids:
            node = node_by_id.get(int(scene_id))
            if node is None:
                continue
            parent_ids = [int(parent_id) for parent_id in node["parent_ids"] if int(parent_id) in node_by_id]
            parent_id = parent_ids[-1] if parent_ids else None
            children_by_parent.setdefault(parent_id, []).append(int(scene_id))
        score_by_id: dict[int, float] = {}

        def score_ordered(scene_ids: list[int]) -> list[tuple[int, float]]:
            scores = self._score_scene_candidates(ctx, frame_data_url, scene_ids)
            score_by_id.update(scores)
            return [(int(scene_id), float(scores.get(int(scene_id), 0.0))) for scene_id in scene_ids]

        def describe_candidate(scene_id: int, score: float) -> dict[str, Any]:
            node = node_by_id.get(int(scene_id), {})
            image = images.get(int(scene_id)) if isinstance(images, dict) else None
            return {
                "scene_id": int(scene_id),
                "title": str(image.get("title") or "") if isinstance(image, dict) else "",
                "score": round(float(score), 3),
                "threshold": round(float(self.threshold_for_scene_id(int(scene_id))), 3),
                "matched": self.scene_matches_id(int(scene_id), float(score)),
                "weak": self._weak_similarity_scene(ctx, int(scene_id)),
                "layer": int(node.get("layer", 3) or 3),
                "parent_ids": [int(parent_id) for parent_id in node.get("parent_ids", [])],
            }

        def match_ordered_candidates(
            scene_ids: list[int],
            *,
            stage: str,
            parent_id: int | None = None,
            select_best_score: bool = False,
        ) -> list[tuple[int, float]]:
            """并行评分同一候选组，返回按配置顺序命中的显式场景。

            这里故意不是“最高分获胜”：候选顺序由资产树遍历顺序或调用方
            的候选列表表达。只有整组没有任何显式场景身份候选时，才允许
            layer3 无身份帧用最高全图相似度兜底。父场景下的 children
            属于同一粗场景的细分变体，允许按分数选择更明确的子帧。
            """
            started_at = time.perf_counter()
            explicit_matches: list[tuple[int, float]] = []
            weak_matches: list[tuple[int, float]] = []
            scored: list[tuple[int, float]] = []
            batch_size = max(1, min(len(scene_ids) or 1, int(self.max_candidate_batch_size or len(scene_ids) or 1)))
            has_explicit_candidate = any(not self._weak_similarity_scene(ctx, scene_id) for scene_id in scene_ids)
            stopped_early = False
            for batch_index, offset in enumerate(range(0, len(scene_ids), batch_size), start=1):
                batch_ids = scene_ids[offset : offset + batch_size]
                batch_started_at = time.perf_counter()
                batch_scored = score_ordered(batch_ids)
                batch_elapsed = time.perf_counter() - batch_started_at
                scored.extend(batch_scored)
                batch_explicit_matches: list[tuple[int, float]] = []
                batch_weak_matches: list[tuple[int, float]] = []
                for scene_id, score in batch_scored:
                    if not self.scene_matches_id(scene_id, score):
                        continue
                    if self._weak_similarity_scene(ctx, scene_id):
                        batch_weak_matches.append((scene_id, score))
                    else:
                        batch_explicit_matches.append((scene_id, score))
                explicit_matches.extend(batch_explicit_matches)
                weak_matches.extend(batch_weak_matches)
                emit({
                    "event": "candidate_batch",
                    "stage": stage,
                    "parent_id": parent_id,
                    "batch_index": batch_index,
                    "batch_count": (len(scene_ids) + batch_size - 1) // batch_size,
                    "max_parallel_workers": max(1, int(self.max_parallel_workers or 1)),
                    "candidate_ids": [int(scene_id) for scene_id in batch_ids],
                    "candidate_count": len(batch_ids),
                    "elapsed_seconds": round(float(batch_elapsed), 3),
                    "elapsed_text": _format_elapsed_seconds(batch_elapsed),
                    "matched_ids": [int(scene_id) for scene_id, _score in [*batch_explicit_matches, *batch_weak_matches]],
                    "candidates": [describe_candidate(scene_id, score) for scene_id, score in batch_scored],
                })
                if has_explicit_candidate and batch_explicit_matches and not select_best_score:
                    stopped_early = True
                    break
            selected: list[tuple[int, float]]
            selection_rule: str
            if has_explicit_candidate:
                if select_best_score and explicit_matches:
                    selected = [max(explicit_matches, key=lambda item: item[1])]
                    selection_rule = "best_score_explicit_match"
                else:
                    selected = explicit_matches
                    selection_rule = "ordered_batched_first_explicit_match"
            elif weak_matches:
                selected = [max(weak_matches, key=lambda item: item[1])]
                selection_rule = "batched_weak_fallback_best_score"
            else:
                selected = []
                selection_rule = "no_match"
            elapsed = time.perf_counter() - started_at
            emit({
                "event": "candidate_group",
                "stage": stage,
                "parent_id": parent_id,
                "candidate_ids": [int(scene_id) for scene_id in scene_ids],
                "candidate_count": len(scene_ids),
                "selection_rule": selection_rule,
                "has_explicit_candidate": bool(has_explicit_candidate),
                "batch_size": batch_size,
                "batch_count": (len(scene_ids) + batch_size - 1) // batch_size,
                "max_parallel_workers": max(1, int(self.max_parallel_workers or 1)),
                "processed_count": len(scored),
                "stopped_early": stopped_early,
                "elapsed_seconds": round(float(elapsed), 3),
                "elapsed_text": _format_elapsed_seconds(elapsed),
                "candidates": [describe_candidate(scene_id, score) for scene_id, score in scored],
                "selected_ids": [int(scene_id) for scene_id, _score in selected],
            })
            return selected

        def refine_frame_tree(scene_id: int, score: float, allowed_ids: set[int] | None = None) -> tuple[int | None, float]:
            """父 frame 命中后，只沿 children 继续细化。

            子 frame 的 `layer` 不再触发默认 layer 扫描；如果没有子节点
            命中，就停留在已经命中的 parent frame。
            """
            if not self.scene_matches_id(int(scene_id), float(score)):
                return None, float(score)
            emit({
                "event": "refine_enter",
                "scene_id": int(scene_id),
                "score": round(float(score), 3),
                "allowed_ids": sorted(int(item) for item in allowed_ids) if allowed_ids is not None else None,
            })
            best_id = int(scene_id)
            best_score = float(score)
            children = children_by_parent.get(int(scene_id), [])
            if allowed_ids is not None:
                children = [child_id for child_id in children if int(child_id) in allowed_ids]
            for child_id, child_score in match_ordered_candidates(children, stage="children", parent_id=int(scene_id), select_best_score=True):
                matched_id, matched_score = refine_frame_tree(child_id, child_score, allowed_ids=allowed_ids)
                if matched_id is None:
                    continue
                best_id = matched_id
                best_score = min(float(score), float(matched_score))
                break
            if best_id == int(scene_id):
                emit({
                    "event": "refine_stop_at_parent",
                    "scene_id": int(scene_id),
                    "reason": "no_child_matched",
                })
            else:
                emit({
                    "event": "refine_child_selected",
                    "parent_id": int(scene_id),
                    "selected_scene_id": int(best_id),
                    "score": round(float(best_score), 3),
                })
            return best_id, best_score

        root_ids = children_by_parent.get(None, [])
        root_layer_groups: list[tuple[str, list[int], set[int] | None]] = []
        # 默认只扫描 root frame 的 Layer 1 -> Layer 2 候选队列。
        root_layer_groups.extend(
            [
                (f"layer{layer}", [
                    scene_id
                    for scene_id in root_ids
                    if int(node_by_id.get(int(scene_id), {}).get("layer", 3)) == layer
                ], None)
                for layer in (1, 2)
            ]
        )
        emit({
            "event": "root_layer_queue",
            "preferred_scene_ids": [int(item) for item in preferred_scene_ids or []],
            "flat_candidate_ids": [int(item) for item in flat_candidate_ids],
            "groups": [
                {"stage": stage, "root_ids": [int(item) for item in root_group], "allowed_ids": sorted(int(item) for item in allowed_ids) if allowed_ids is not None else None}
                for stage, root_group, allowed_ids in root_layer_groups
            ],
        })
        for stage, root_group, allowed_ids in root_layer_groups:
            for root_id, root_score in match_ordered_candidates(root_group, stage=stage):
                matched_id, matched_score = refine_frame_tree(root_id, root_score, allowed_ids=allowed_ids)
                if matched_id is not None:
                    emit({
                        "event": "final",
                        "scene_id": int(matched_id),
                        "score": round(float(matched_score), 3),
                        "matched_root_id": int(root_id),
                        "matched_root_stage": stage,
                    })
                    return matched_id, matched_score
        fallback_score = max(score_by_id.values()) if score_by_id else 0.0
        emit({"event": "final", "scene_id": None, "score": round(float(fallback_score), 3)})
        return None, fallback_score

    def identify_scene_number(
        self,
        ctx: dict[str, Any],
        frame_data_url: str,
        *,
        preferred_scene_ids: list[int] | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> tuple[int | None, float]:
        def emit(event: dict[str, Any]) -> None:
            if trace is not None:
                trace.append(event)

        images = ctx.get("images") or {}
        if not isinstance(images, dict):
            emit({"event": "flat_final", "scene_id": None, "score": 0.0, "reason": "images_missing"})
            return None, 0.0
        candidate_ids: list[int] = []
        if preferred_scene_ids is not None:
            for scene_id in preferred_scene_ids:
                image = images.get(int(scene_id))
                if isinstance(image, dict):
                    candidate_ids.append(int(scene_id))
        else:
            for scene_id, image in images.items():
                if isinstance(image, dict) and int(View(image).layer) <= 2:
                    candidate_ids.append(int(scene_id))
        if not candidate_ids:
            emit({"event": "flat_final", "scene_id": None, "score": 0.0, "reason": "candidate_ids_empty"})
            return None, 0.0
        score_by_id = self._score_scene_candidates(ctx, frame_data_url, candidate_ids)
        emit({
            "event": "flat_candidate_group",
            "preferred_scene_ids": [int(item) for item in preferred_scene_ids or []],
            "candidate_ids": [int(item) for item in candidate_ids],
            "selection_rule": "ordered_first_match",
            "candidates": [
                {
                    "scene_id": int(scene_id),
                    "title": str(images.get(int(scene_id), {}).get("title") or "") if isinstance(images.get(int(scene_id)), dict) else "",
                    "score": round(float(score_by_id.get(int(scene_id), 0.0)), 3),
                    "threshold": round(float(self.threshold_for_scene_id(int(scene_id))), 3),
                    "matched": self.scene_matches_id(int(scene_id), float(score_by_id.get(int(scene_id), 0.0))),
                }
                for scene_id in candidate_ids
            ],
        })
        for scene_id in candidate_ids:
            score = float(score_by_id.get(int(scene_id), 0.0))
            if self.scene_matches_id(int(scene_id), score):
                emit({"event": "flat_final", "scene_id": int(scene_id), "score": round(float(score), 3)})
                return int(scene_id), score
        fallback_score = max(score_by_id.values()) if score_by_id else 0.0
        emit({"event": "flat_final", "scene_id": None, "score": round(float(fallback_score), 3)})
        return None, fallback_score

    def scene_matches_key(self, key: str, score: float) -> bool:
        if not key:
            return False
        threshold = self.threshold_for_key(key) if self.threshold_for_key is not None else self.threshold_for_scene_id(0)
        return float(score) >= float(threshold)

    def identify_scene_key(self, ctx: dict[str, Any], frame_data_url: str, *, keys: list[str]) -> tuple[str, float]:
        if self.image_for_key is None:
            raise RuntimeError("SceneRecognizer 缺少 image_for_key，无法按 key 识别场景")
        ordered_keys: list[str] = []
        images_by_key: dict[str, dict[str, Any]] = {}
        for key in keys:
            image = self.image_for_key(ctx, key)
            if image is not None:
                ordered_keys.append(key)
                images_by_key[key] = image
        if not ordered_keys:
            return "", 0.0

        def score(key: str) -> tuple[str, float]:
            return key, float(self.score_image(ctx, images_by_key[key], frame_data_url))

        if len(ordered_keys) <= 1:
            score_by_key = dict(score(key) for key in ordered_keys)
        else:
            workers = max(1, min(len(ordered_keys), int(self.max_parallel_workers or 1)))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scene-key-match") as executor:
                score_by_key = dict(executor.map(score, ordered_keys))
        for key in ordered_keys:
            key_score = float(score_by_key.get(key, 0.0))
            if self.scene_matches_key(key, key_score):
                return key, key_score
        return "", max(score_by_key.values()) if score_by_key else 0.0




@dataclass(frozen=True)
class ShapeMatchPlanner:
    """规划单个 shape 的图像/OCR 匹配策略。"""

    def match_role(self, shape: dict[str, Any], key: str, default: str = "required") -> str:
        role = str(shape.get(key) or default).strip().lower()
        if role == "optional":
            return "optional"
        if role in {"off", "none", "无", "0"}:
            return "off"
        if role in {"required", "must", "必", "1"}:
            return "required"
        if role in {"decisive", "any", "定", "2"}:
            return "decisive"
        if str(default or "").strip().lower() == "optional":
            return "optional"
        normalized = normalize_match_role(default, MatchRole.required)
        if normalized is MatchRole.off:
            return "off"
        if normalized is MatchRole.required:
            return "required"
        if normalized is MatchRole.decisive:
            return "decisive"
        return "required"

    def ocr_role(self, shape: dict[str, Any]) -> str:
        default = "required" if bool(shape.get("ocrEnabled")) and str(shape.get("ocrText") or "").strip() else "off"
        return self.match_role(shape, "ocrMatchRole", default)

    def image_role(self, shape: dict[str, Any]) -> str:
        return self.match_role(shape, "imageMatchRole", "required")

    def ocr_fallback_enabled(self, shape: dict[str, Any]) -> bool:
        if not str(shape.get("ocrText") or "").strip():
            return False
        return self.ocr_role(shape) != "off"

    def shape_match_payload_flags(self, shape: dict[str, Any], *, condition: str = "auto") -> dict[str, Any]:
        ocr_text = str(shape.get("ocrText") or "").strip()
        image_role = self.image_role(shape)
        ocr_role = self.ocr_role(shape)
        force_image = condition == "image"
        force_ocr = condition == "ocr"
        ocr_enabled = bool(not force_image and ocr_role != "off" and ocr_text)
        scan_enabled = bool(shape.get("floating") and not ocr_enabled)
        jitter_enabled = bool(shape.get("jitterEnabled") and not scan_enabled and not ocr_enabled)
        return {
            "image_role": image_role,
            "ocr_role": ocr_role,
            "ocr_enabled": ocr_enabled,
            "scan": scan_enabled,
            "match_strategy": "auto" if (force_ocr or scan_enabled or jitter_enabled) else "anchor_pixel",
        }

    def match_conditions(self, shape: dict[str, Any], *, first: str = "image") -> list[str]:
        """返回 action 匹配的尝试顺序。

        默认先图像，再 OCR。``decisive/定`` 不是强制条件；当图像和 OCR 都是
        ``decisive`` 时，任一条件命中即可。
        """

        conditions: list[str] = []
        image_role = self.image_role(shape)
        ocr_role = self.ocr_role(shape)
        has_ocr = bool(str(shape.get("ocrText") or "").strip() and ocr_role != "off")
        if image_role != "off":
            conditions.append("image")
        if has_ocr:
            conditions.append("ocr")
        if first == "ocr":
            conditions.sort(key=lambda item: 0 if item == "ocr" else 1)
        return conditions
