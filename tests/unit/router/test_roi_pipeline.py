from src.router.roi_helpers import bbox_area_ratio, sanitize_bbox
from src.router.roi_pipeline import collect_sam3_roi_candidates, run_sam3_roi_classification_stage


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
