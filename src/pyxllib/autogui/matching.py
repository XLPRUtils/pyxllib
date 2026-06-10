from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .model import MatchRole, View, normalize_match_role


SceneScoreFunc = Callable[[dict[str, Any], dict[str, Any], str], float]
SceneThresholdFunc = Callable[[int], float]
ImageForKeyFunc = Callable[[dict[str, Any], str], dict[str, Any] | None]
KeyThresholdFunc = Callable[[str], float]
ShapeScoreFunc = Callable[[dict[str, Any], dict[str, Any], dict[str, Any], str], float]
ShapeOcrScoreFunc = Callable[[dict[str, Any], dict[str, Any], dict[str, Any], str], float]
DetailLogFunc = Callable[[str], None]
ImagePredicateFunc = Callable[[dict[str, Any]], bool]


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
        return max(scores) if scores else 0.0


@dataclass(frozen=True)
class SceneRecognizer:
    """根据候选帧分数识别当前场景。"""

    score_image: SceneScoreFunc
    threshold_for_scene_id: SceneThresholdFunc
    image_for_key: ImageForKeyFunc | None = None
    threshold_for_key: KeyThresholdFunc | None = None
    key_priorities: Mapping[str, int] | None = None

    def scene_matches_id(self, scene_id: int, score: float) -> bool:
        return float(score) >= float(self.threshold_for_scene_id(scene_id))

    def identify_scene_number(
        self,
        ctx: dict[str, Any],
        frame_data_url: str,
        *,
        preferred_scene_ids: list[int] | None = None,
    ) -> tuple[int | None, float]:
        images = ctx.get("images") or {}
        if not isinstance(images, dict):
            return None, 0.0
        candidates: list[tuple[int, float, int]] = []
        if preferred_scene_ids:
            for order, scene_id in enumerate(preferred_scene_ids):
                image = images.get(scene_id)
                if isinstance(image, dict):
                    candidates.append((int(scene_id), float(self.score_image(ctx, image, frame_data_url)), int(order)))
        else:
            for scene_id, image in images.items():
                if isinstance(image, dict):
                    candidates.append((int(scene_id), float(self.score_image(ctx, image, frame_data_url)), int(scene_id)))
        candidates.sort(
            key=(
                (lambda item: (item[1], -item[2]))
                if preferred_scene_ids
                else (lambda item: (item[1], -item[0]))
            ),
            reverse=True,
        )
        if not candidates:
            return None, 0.0
        scene_id, score, _order = candidates[0]
        return (scene_id, score) if self.scene_matches_id(scene_id, score) else (None, score)

    def scene_matches_key(self, key: str, score: float) -> bool:
        if not key:
            return False
        threshold = self.threshold_for_key(key) if self.threshold_for_key is not None else self.threshold_for_scene_id(0)
        return float(score) >= float(threshold)

    def identify_scene_key(self, ctx: dict[str, Any], frame_data_url: str, *, keys: list[str]) -> tuple[str, float]:
        if self.image_for_key is None:
            raise RuntimeError("SceneRecognizer 缺少 image_for_key，无法按 key 识别场景")
        priorities = dict(self.key_priorities or {})
        candidates: list[tuple[str, float]] = []
        for key in keys:
            image = self.image_for_key(ctx, key)
            if image is not None:
                candidates.append((key, float(self.score_image(ctx, image, frame_data_url))))
        candidates.sort(key=lambda item: (item[1], priorities.get(item[0], 0)), reverse=True)
        return candidates[0] if candidates else ("", 0.0)




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

    def runtime_match_payload_flags(self, shape: dict[str, Any], *, condition: str = "auto") -> dict[str, Any]:
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
