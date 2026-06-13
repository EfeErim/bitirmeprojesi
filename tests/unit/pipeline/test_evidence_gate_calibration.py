from src.pipeline.evidence_gate_calibration import (
    calibrate_evidence_gate_report,
    evaluate_policy,
    select_policy,
    split_rows,
)


def _row(image_path, target_id, expected="healthy", diagnosis="healthy", **overrides):
    row = {
        "image_path": image_path,
        "target_id": target_id,
        "expected_label": expected,
        "diagnosis": diagnosis,
        "full_confidence": 0.90,
        "roi_evidence_status": "supports_full",
        "roi_quality_status": "roi_ok",
        "full_ood_is_ood": False,
        "roi_ood_is_ood": False,
    }
    row.update(overrides)
    return row


def test_evaluate_policy_computes_review_metrics():
    policy = {
        "full_confidence_threshold": 0.70,
        "review_on_roi_conflict": True,
        "review_on_roi_quality_bad": False,
        "review_on_full_ood": False,
        "review_on_roi_ood": False,
    }
    rows = [
        _row("wrong.jpg", "tomato__leaf", diagnosis="blight", roi_evidence_status="conflicts_with_full"),
        _row("correct.jpg", "tomato__leaf"),
        _row("reviewed-correct.jpg", "tomato__leaf", roi_evidence_status="conflicts_with_full"),
    ]

    metrics = evaluate_policy(rows, policy)["metrics"]

    assert metrics["sample_count"] == 3
    assert metrics["wrong_count"] == 1
    assert metrics["correct_count"] == 2
    assert metrics["review_count"] == 2
    assert metrics["wrong_capture_rate"] == 1.0
    assert metrics["false_positive_review_rate"] == 0.5
    assert metrics["wrong_missed_count"] == 0


def test_select_policy_chooses_highest_coverage_eligible_policy():
    rows = [
        _row("wrong-low.jpg", "tomato__leaf", diagnosis="blight", full_confidence=0.55),
        _row("wrong-conflict.jpg", "tomato__leaf", diagnosis="blight", roi_evidence_status="conflicts_with_full"),
        _row("correct-high.jpg", "tomato__leaf"),
        _row("correct-low.jpg", "tomato__leaf", full_confidence=0.95),
    ]

    selected = select_policy(rows, min_capture=1.0, max_false_positive_rate=0.0)

    assert selected["status"] == "eligible"
    assert selected["selected"]["policy"]["full_confidence_threshold"] == 0.60
    assert selected["selected"]["policy"]["review_on_roi_conflict"] is True
    assert selected["selected"]["metrics"]["coverage"] == 0.5


def test_select_policy_emits_best_rejected_when_no_policy_is_eligible():
    rows = [
        _row("wrong.jpg", "tomato__leaf", diagnosis="blight", full_confidence=0.95),
        _row("correct.jpg", "tomato__leaf", full_confidence=0.95),
    ]

    selected = select_policy(rows, min_capture=1.0, max_false_positive_rate=0.0)

    assert selected["status"] == "no_eligible_policy"
    assert selected["best_rejected"]["metrics"]["wrong_capture_rate"] == 0.0


def test_calibration_uses_target_policy_when_evidence_is_sufficient():
    rows = []
    for idx in range(1, 31):
        rows.append(
            _row(
                f"target-wrong-{idx}.jpg",
                "tomato__leaf",
                diagnosis="blight",
                full_confidence=0.55,
            )
        )
        rows.append(_row(f"target-correct-{idx}.jpg", "tomato__leaf", full_confidence=0.95))

    payload = calibrate_evidence_gate_report(
        rows,
        min_capture=1.0,
        max_false_positive_rate=0.0,
        min_target_errors=5,
        holdout_ratio=0.30,
        seed=7,
    )

    assert payload["global_policy"]["status"] == "eligible"
    assert payload["target_policies"]["tomato__leaf"]["status"] == "target_specific"


def test_calibration_uses_global_fallback_when_target_errors_are_insufficient():
    rows = []
    for idx in range(1, 31):
        rows.append(_row(f"global-wrong-{idx}.jpg", "tomato__leaf", diagnosis="blight", full_confidence=0.55))
        rows.append(_row(f"global-correct-{idx}.jpg", "tomato__leaf", full_confidence=0.95))
    rows.extend(
        [
            _row("small-wrong.jpg", "strawberry__fruit", diagnosis="ripe", full_confidence=0.55),
            _row("small-correct.jpg", "strawberry__fruit", full_confidence=0.95),
        ]
    )

    payload = calibrate_evidence_gate_report(
        rows,
        min_capture=1.0,
        max_false_positive_rate=0.0,
        min_target_errors=5,
        holdout_ratio=0.30,
        seed=7,
    )

    small_target = payload["target_policies"]["strawberry__fruit"]
    assert small_target["status"] == "global_fallback"
    assert small_target["fallback_reason"].startswith("target_wrong_count_below_min_target_errors")


def test_split_rows_is_stable_across_repeated_runs():
    rows = [_row(f"sample-{idx}.jpg", "tomato__leaf") for idx in range(20)]

    first = split_rows(rows, holdout_ratio=0.30, seed=20260613)
    second = split_rows(rows, holdout_ratio=0.30, seed=20260613)

    assert first == second
