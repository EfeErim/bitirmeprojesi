"""SAM3 orchestration helpers for the VLM router runtime."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from src.router.analysis_results import (
    build_sam3_analysis_result,
    init_sam3_stage_timings,
    summarize_sam3_stage_timings,
)
from src.router.compatibility_utils import compatible_parts_for_crop
from src.router.confidence_utils import (
    passes_open_set_gate,
    resolve_effective_confidence_threshold,
)
from src.router.pipeline_flow_utils import resolve_effective_max_detections
from src.router.roi_helpers import bbox_area_ratio, extract_roi, sanitize_bbox, suppress_overlapping_detections
from src.router.roi_pipeline import (
    classify_sam3_roi_candidate,
    collect_sam3_roi_candidates,
    run_sam3_roi_classification_stage,
)
from src.router.runtime_settings import build_sam3_runtime_settings, resolve_sam3_stage_order

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sam3RequestContext:
    pil_image: Image.Image
    image_size: Tuple[int, int, int]
    image_width: int
    image_height: int
    effective_threshold: float
    effective_max_detections: Optional[int]
    stage_order: List[str]
    settings: Dict[str, Any]
    sam3_prompt: str
    sam3_threshold: float
    timing_logs_enabled: bool
    preprocess_ms: float


def build_request_context(
    runtime: Any,
    *,
    pil_image: Image.Image,
    image_size: Tuple[int, int, int],
    confidence_threshold: float,
    max_detections: Optional[int],
) -> Sam3RequestContext:
    """Build request-local state once per analyze_image call."""
    started_at = time.perf_counter()
    effective_threshold = resolve_effective_confidence_threshold(confidence_threshold, runtime._policy_value)
    effective_max_detections = resolve_effective_max_detections(max_detections)
    settings = build_sam3_runtime_settings(runtime._policy_value, runtime.vlm_config, effective_threshold)
    stage_order = resolve_sam3_stage_order(runtime._policy_value)
    image_width, image_height = pil_image.size
    preprocess_ms = (time.perf_counter() - started_at) * 1000.0
    return Sam3RequestContext(
        pil_image=pil_image,
        image_size=image_size,
        image_width=image_width,
        image_height=image_height,
        effective_threshold=effective_threshold,
        effective_max_detections=effective_max_detections,
        stage_order=stage_order,
        settings=settings,
        sam3_prompt=str(runtime.vlm_config.get("sam3_text_prompt", "plant")),
        sam3_threshold=settings["sam3_threshold"],
        timing_logs_enabled=bool(runtime.vlm_config.get("timing_logs_enabled", True)),
        preprocess_ms=preprocess_ms,
    )


def postprocess_sam3_detections(
    runtime: Any,
    detections: List[Dict[str, Any]],
    *,
    context: Sam3RequestContext,
) -> List[Dict[str, Any]]:
    """Finalize SAM3 detections with optional dedupe and cap."""
    ordered = sorted(detections, key=lambda d: float(d.get("_quality_score", 0.0)), reverse=True)
    if runtime._policy_enabled("dedupe", True) and "postprocess" in context.stage_order:
        ordered = suppress_overlapping_detections(
            ordered,
            iou_threshold=context.settings["detection_nms_iou_threshold"],
            same_crop_only=context.settings["detection_nms_same_crop_only"],
        )
    if context.effective_max_detections is not None:
        ordered = ordered[: context.effective_max_detections]
    for det in ordered:
        det.pop("_quality_score", None)
    return ordered


def run_sam3_roi_filter_stage(
    runtime: Any,
    boxes: Any,
    scores: Any,
    *,
    context: Sam3RequestContext,
) -> Tuple[List[Dict[str, Any]], int]:
    """Execute ROI candidate filtering stage."""
    apply_roi_filters = runtime._policy_enabled("roi_filter", True) and "roi_filter" in context.stage_order
    return collect_sam3_roi_candidates(
        boxes=boxes,
        scores=scores,
        image_width=context.image_width,
        image_height=context.image_height,
        settings=context.settings,
        apply_roi_filters=apply_roi_filters,
        sanitize_bbox_fn=sanitize_bbox,
        bbox_area_ratio_fn=bbox_area_ratio,
    )


def analyze_sam3_image(runtime: Any, context: Sam3RequestContext) -> Dict[str, Any]:
    """Analyze an image using the SAM3 + BioCLIP runtime."""
    start_time = time.perf_counter()
    stage_timings_ms = init_sam3_stage_timings()
    stage_timings_ms["preprocess"] = context.preprocess_ms

    stage_start = time.perf_counter()
    sam3_results = runtime._run_sam3(
        context.pil_image,
        prompt=context.sam3_prompt,
        threshold=context.sam3_threshold,
    )
    stage_timings_ms["sam3_inference"] = (time.perf_counter() - stage_start) * 1000.0
    masks = sam3_results.get("masks", [])
    boxes = sam3_results.get("boxes", [])
    scores = sam3_results.get("scores", [])

    if torch.is_tensor(masks):
        mask_count = int(masks.shape[0]) if masks.ndim > 0 else int(masks.numel() > 0)
    elif isinstance(masks, (list, tuple)):
        mask_count = len(masks)
    else:
        mask_count = 0

    detections: List[Dict[str, Any]] = []
    roi_seen = 0
    roi_kept = 0
    roi_classification_calls = 0
    if mask_count > 0:
        roi_stage_start = time.perf_counter()
        candidates, roi_seen = run_sam3_roi_filter_stage(runtime, boxes, scores, context=context)

        detections, roi_kept, roi_classification_calls, roi_classification_ms = run_sam3_roi_classification_stage(
            candidates=candidates,
            settings=context.settings,
            stage_order=context.stage_order,
            policy_enabled_fn=runtime._policy_enabled,
            classify_candidate_fn=lambda candidate: classify_sam3_roi_candidate(
                pil_image=context.pil_image,
                candidate=candidate,
                image_width=context.image_width,
                image_height=context.image_height,
                settings=context.settings,
                part_labels=runtime.part_labels,
                crop_labels=runtime.crop_labels,
                policy_enabled_fn=runtime._policy_enabled,
                extract_roi_fn=extract_roi,
                clip_score_labels_ensemble_fn=runtime._clip_score_labels_ensemble,
                compute_leaf_likeness_fn=runtime._compute_leaf_likeness,
                rebalance_part_scores_for_leaf_like_roi_fn=runtime._rebalance_part_scores_for_leaf_like_roi,
                select_best_crop_with_fallback_fn=runtime._select_best_crop_with_fallback,
                compatible_parts_for_crop_fn=lambda crop_label: compatible_parts_for_crop(
                    crop_label,
                    runtime.crop_part_compatibility,
                    runtime.part_labels,
                ),
                score_parts_conditioned_on_crop_fn=runtime._score_parts_conditioned_on_crop,
                apply_generic_part_penalty_fn=runtime._apply_generic_part_penalty,
                select_part_label_with_specificity_fn=runtime._select_part_label_with_specificity,
                apply_leaf_like_override_fn=runtime._apply_leaf_like_override,
            ),
            passes_open_set_gate_fn=passes_open_set_gate,
        )
        stage_timings_ms["roi_classification"] += roi_classification_ms
        stage_timings_ms["roi_total"] = (time.perf_counter() - roi_stage_start) * 1000.0

    stage_start = time.perf_counter()
    detections = postprocess_sam3_detections(runtime, detections, context=context)
    stage_timings_ms["postprocess"] = (time.perf_counter() - stage_start) * 1000.0

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    stage_summary = summarize_sam3_stage_timings(stage_timings_ms, roi_seen, roi_classification_calls)

    if context.timing_logs_enabled:
        logger.info(
            (
                "[TIMING] SAM3 pipeline | total=%.2fms | sam3=%.2fms | roi_total=%.2fms "
                "| roi_class=%.2fms | rois=%d | kept=%d"
            ),
            elapsed_ms,
            stage_timings_ms["sam3_inference"],
            stage_timings_ms["roi_total"],
            stage_timings_ms["roi_classification"],
            roi_seen,
            roi_kept,
        )

    return build_sam3_analysis_result(
        detections=detections,
        image_size=context.image_size,
        elapsed_ms=elapsed_ms,
        stage_summary=stage_summary,
        roi_seen=roi_seen,
        roi_kept=roi_kept,
        roi_classification_calls=roi_classification_calls,
        mask_count=mask_count,
        sam3_threshold=context.sam3_threshold,
    )
