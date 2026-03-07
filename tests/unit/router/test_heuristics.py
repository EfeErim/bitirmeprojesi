from PIL import Image

from src.router.heuristics import (
    apply_leaf_like_override,
    compute_leaf_likeness,
    rebalance_part_scores_for_leaf_like_roi,
    select_best_crop_with_fallback,
    select_part_label_with_specificity,
)


def test_compute_leaf_likeness_scores_green_roi_high():
    likeness = compute_leaf_likeness(
        Image.new("RGB", (80, 80), color="green"),
        bbox=[0.0, 0.0, 80.0, 80.0],
        image_width=100,
        image_height=100,
    )

    assert likeness > 0.8


def test_rebalance_part_scores_for_leaf_like_roi_boosts_leaf_and_penalizes_non_foliar():
    adjusted = rebalance_part_scores_for_leaf_like_roi(
        {"leaf": 0.30, "fruit": 0.40, "stem": 0.20},
        leaf_likeness=0.9,
    )

    assert adjusted["leaf"] > 0.30
    assert adjusted["fruit"] < 0.40
    assert adjusted["stem"] == 0.20


def test_apply_leaf_like_override_prefers_leaf_for_generic_selection():
    label, score = apply_leaf_like_override(
        selected_label="whole plant",
        selected_score=0.20,
        part_scores={"whole plant": 0.20, "leaf": 0.15},
        bbox=[10.0, 10.0, 90.0, 60.0],
        image_width=100,
        image_height=100,
    )

    assert label == "leaf"
    assert score == 0.15


def test_select_part_label_with_specificity_prefers_specific_part_when_close():
    label, score = select_part_label_with_specificity(
        {"whole plant": 0.50, "leaf": 0.35},
        generic_part_labels=["whole plant"],
        specific_override_ratio=0.60,
        specific_min_confidence=0.20,
        preferred_part_labels=["leaf"],
        preferred_override_ratio=0.60,
    )

    assert label == "leaf"
    assert score == 0.35


def test_select_best_crop_with_fallback_uses_compatible_part_signal():
    crop, score = select_best_crop_with_fallback(
        {"tomato": 0.49, "potato": 0.50},
        {"leaf": 0.95, "tuber": 0.10},
        {"tomato": ["leaf"], "potato": ["tuber"]},
        ["leaf", "tuber"],
    )

    assert crop == "tomato"
    assert score > 0.50
