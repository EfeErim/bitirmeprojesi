from PIL import Image

from src.router.roi_helpers import bbox_area_ratio, sanitize_bbox
from src.router.roi_pipeline import (
    collect_sam3_roi_candidates,
    finalize_sam3_roi_candidate,
    run_sam3_roi_classification_stage,
)


def test_collect_sam3_roi_candidates_applies_filters():
    boxes = [[0, 0, 20, 20], [0, 0, 2, 2]]
    scores = [0.9, 0.95]
    settings = {
        "sam3_threshold": 0.85,
        "min_box_side_px": 5,
        "min_box_area_ratio": 0.01,
        "max_rois_for_classification": None,
    }

    candidates, seen = collect_sam3_roi_candidates(
        boxes=boxes,
        scores=scores,
        image_width=100,
        image_height=100,
        settings=settings,
        apply_roi_filters=True,
        sanitize_bbox_fn=sanitize_bbox,
        bbox_area_ratio_fn=bbox_area_ratio,
    )

    assert seen == 2
    assert len(candidates) == 1
    assert candidates[0]["bbox"] == [0.0, 0.0, 20.0, 20.0]


def test_run_sam3_roi_classification_stage_filters_with_open_set_gate():
    candidates = [{"bbox": [0.0, 0.0, 10.0, 10.0], "sam3_score": 0.9}]
    settings = {
        "classification_min_confidence": 0.4,
        "focus_part_mode_enabled": False,
        "focus_fallback_enabled": False,
    }
    stage_order = ["roi_filter", "roi_classification", "open_set_gate", "postprocess"]

    def policy_enabled(_stage, default=True):
        return default

    def classify_candidate(_candidate):
        return (
            {
                "crop": "tomato",
                "part": "leaf",
                "crop_confidence": 0.75,
                "part_confidence": 0.8,
                "bbox": [0.0, 0.0, 10.0, 10.0],
            },
            2,
        )

    def open_set_gate(crop_label, crop_confidence, min_confidence):
        return crop_label == "tomato" and crop_confidence >= min_confidence

    detections, kept, calls, elapsed_ms = run_sam3_roi_classification_stage(
        candidates=candidates,
        settings=settings,
        stage_order=stage_order,
        policy_enabled_fn=policy_enabled,
        classify_candidate_fn=classify_candidate,
        passes_open_set_gate_fn=open_set_gate,
    )

    assert len(detections) == 1
    assert kept == 1
    assert calls == 2
    assert elapsed_ms >= 0.0


def test_run_sam3_roi_classification_stage_returns_early_when_disabled():
    detections, kept, calls, elapsed_ms = run_sam3_roi_classification_stage(
        candidates=[{"bbox": [0.0, 0.0, 1.0, 1.0], "sam3_score": 0.9}],
        settings={"classification_min_confidence": 0.2},
        stage_order=["roi_filter", "postprocess"],
        policy_enabled_fn=lambda *_: True,
        classify_candidate_fn=lambda *_: (None, 0),
        passes_open_set_gate_fn=lambda *_: True,
    )

    assert detections == []
    assert kept == 0
    assert calls == 0
    assert elapsed_ms == 0.0


def test_finalize_sam3_roi_candidate_does_not_force_leaf_from_visual_bias_alone():
    settings = {
        "conditioned_part_weight": 0.45,
        "generic_part_labels": ["whole plant", "whole", "plant", "entire plant"],
        "generic_part_penalty": 0.78,
        "specific_part_override_ratio": 0.45,
        "specific_part_min_confidence": 0.12,
        "preferred_part_labels": [],
        "preferred_part_override_ratio": 0.50,
        "leaf_override_label": "leaf",
        "leaf_override_enabled": False,
        "leaf_visual_override_enabled": True,
        "leaf_visual_likeness_threshold": 0.58,
        "leaf_visual_green_min": 0.18,
        "leaf_visual_min_margin": 0.05,
        "leaf_visual_force_generic": True,
        "leaf_visual_force_without_leaf_score": False,
        "leaf_visual_force_conf_floor": 0.16,
        "leaf_visual_force_part_factor": 0.65,
        "leaf_part_rebalance_enabled": False,
        "weight_crop": 0.65,
        "weight_part": 0.20,
        "weight_sam3": 0.15,
    }

    detection = finalize_sam3_roi_candidate(
        roi_image=Image.new("RGB", (80, 80), color="green"),
        candidate={"bbox": [10.0, 10.0, 90.0, 90.0], "sam3_score": 0.91},
        image_width=100,
        image_height=100,
        settings=settings,
        policy_enabled_fn=lambda stage, default=True: False if stage == "compatibility_fusion" else default,
        part_label="whole plant",
        part_conf=0.42,
        part_scores={"whole plant": 0.42, "fruit": 0.35, "leaf": 0.16},
        crop_label="tomato",
        crop_conf=0.94,
        crop_scores={"tomato": 0.94, "wheat": 0.08},
        compute_leaf_likeness_fn=lambda **_: 0.92,
        rebalance_part_scores_for_leaf_like_roi_fn=lambda **kwargs: kwargs["part_scores"],
        select_best_crop_with_fallback_fn=lambda crop_scores, part_scores, **_: ("tomato", crop_scores["tomato"]),
        compatible_parts_for_crop_fn=lambda _crop: ["leaf", "fruit", "whole plant"],
        score_parts_conditioned_on_crop_fn=lambda *_args, **_kwargs: {},
        apply_generic_part_penalty_fn=lambda scores, *_args, **_kwargs: scores,
        select_part_label_with_specificity_fn=lambda scores, *_args, **_kwargs: max(
            scores.items(), key=lambda item: item[1]
        ),
        apply_leaf_like_override_fn=lambda **kwargs: (kwargs["selected_label"], kwargs["selected_score"]),
        global_crop_scores={"tomato": 0.95},
    )

    assert detection["part"] == "whole plant"
    assert detection["part_confidence"] == 0.42
