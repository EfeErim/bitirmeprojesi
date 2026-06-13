"""Target-aware calibration helpers for Notebook 16 evidence-gate reports."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "v1_evidence_gate_calibration"
DEFAULT_SOURCE_REPORT = Path("docs/ablation_results/dual_view_inference/multi_target_report.json")
DEFAULT_OUTPUT = Path("docs/ablation_results/dual_view_inference/evidence_gate_calibration.json")
DEFAULT_SEED = 20260613
ROI_QUALITY_BAD = {"roi_too_large", "roi_too_small"}


def load_notebook16_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load rows from a Notebook 16 multi-target report."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Expected a report with a list-valued 'rows' field: {path}")
    return [dict(row) for row in rows]


def calibrate_evidence_gate_report(
    rows: Iterable[dict[str, Any]],
    *,
    source_report: str | Path = DEFAULT_SOURCE_REPORT,
    min_capture: float = 0.70,
    max_false_positive_rate: float = 0.15,
    min_target_errors: int = 20,
    holdout_ratio: float = 0.30,
    seed: int = DEFAULT_SEED,
    include_samples: bool = False,
) -> dict[str, Any]:
    """Build a global plus per-target advisory evidence-gate calibration payload."""
    normalized_rows = [_normalize_row(row) for row in rows]
    calibration_rows, holdout_rows = split_rows(normalized_rows, holdout_ratio=holdout_ratio, seed=seed)
    constraints = {
        "min_capture": float(min_capture),
        "max_false_positive_rate": float(max_false_positive_rate),
        "min_target_errors": int(min_target_errors),
        "holdout_ratio": float(holdout_ratio),
        "seed": int(seed),
    }

    global_selection = select_policy(
        calibration_rows,
        min_capture=min_capture,
        max_false_positive_rate=max_false_positive_rate,
    )
    global_policy = _policy_payload(global_selection, holdout_rows)

    rows_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    calibration_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    holdout_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        rows_by_target[row["target_key"]].append(row)
    for row in calibration_rows:
        calibration_by_target[row["target_key"]].append(row)
    for row in holdout_rows:
        holdout_by_target[row["target_key"]].append(row)

    target_policies = {}
    for target_key, target_rows in sorted(rows_by_target.items()):
        target_calibration_rows = calibration_by_target.get(target_key, [])
        target_holdout_rows = holdout_by_target.get(target_key, [])
        target_policies[target_key] = _target_policy_payload(
            target_rows,
            target_calibration_rows,
            target_holdout_rows,
            global_selection=global_selection,
            min_capture=min_capture,
            max_false_positive_rate=max_false_positive_rate,
            min_target_errors=min_target_errors,
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_report": _as_posix(source_report),
        "constraints": constraints,
        "global_policy": global_policy,
        "target_policies": target_policies,
    }
    if include_samples:
        split_rows_with_labels = [*calibration_rows, *holdout_rows]
        payload["samples"] = [
            {
                "image_path": row["image_path"],
                "target_key": row["target_key"],
                "split": row["split"],
                "expected_label": row["expected_label"],
                "diagnosis": row["diagnosis"],
                "is_correct": row["is_correct"],
            }
            for row in split_rows_with_labels
        ]
    return payload


def write_calibration_report(payload: dict[str, Any], path: str | Path) -> None:
    """Write a calibration report with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def split_rows(
    rows: Sequence[dict[str, Any]],
    *,
    holdout_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows deterministically with a stable image-path plus target-key hash."""
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be between 0 and 1.")
    calibration_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        hash_input = f"{seed}|{item.get('target_key', '')}|{item.get('image_path', '')}"
        value = int(hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
        item["split"] = "holdout" if value < holdout_ratio else "calibration"
        if item["split"] == "holdout":
            holdout_rows.append(item)
        else:
            calibration_rows.append(item)
    return calibration_rows, holdout_rows


def select_policy(
    rows: Sequence[dict[str, Any]],
    *,
    min_capture: float,
    max_false_positive_rate: float,
) -> dict[str, Any]:
    """Select the highest-coverage eligible policy from the deterministic grid."""
    variants = [evaluate_policy(rows, policy) for policy in candidate_policies()]
    eligible = [
        variant
        for variant in variants
        if _rate_at_least(variant["metrics"]["wrong_capture_rate"], min_capture)
        and _rate_at_most(variant["metrics"]["false_positive_review_rate"], max_false_positive_rate)
    ]
    if eligible:
        selected = sorted(eligible, key=_eligible_sort_key)[0]
        return {
            "status": "eligible",
            "selected": selected,
            "eligible_count": len(eligible),
            "evaluated_count": len(variants),
        }

    best_rejected = sorted(variants, key=_rejected_sort_key)[0] if variants else None
    return {
        "status": "no_eligible_policy",
        "best_rejected": best_rejected,
        "eligible_count": 0,
        "evaluated_count": len(variants),
    }


def evaluate_policy(rows: Sequence[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one candidate policy over normalized rows."""
    normalized_rows = [_ensure_normalized_row(row) for row in rows]
    reviewed = 0
    wrong_reviewed = 0
    wrong_count = 0
    correct_reviewed = 0
    correct_count = 0
    comparable_count = 0
    for row in normalized_rows:
        should_review = policy_requires_review(row, policy)
        if should_review:
            reviewed += 1
        if row["is_comparable"]:
            comparable_count += 1
            if row["is_correct"]:
                correct_count += 1
                if should_review:
                    correct_reviewed += 1
            else:
                wrong_count += 1
                if should_review:
                    wrong_reviewed += 1

    sample_count = len(normalized_rows)
    metrics = {
        "sample_count": sample_count,
        "comparable_count": comparable_count,
        "wrong_count": wrong_count,
        "correct_count": correct_count,
        "review_count": reviewed,
        "accepted_count": sample_count - reviewed,
        "review_rate": _safe_rate(reviewed, sample_count),
        "coverage": _safe_rate(sample_count - reviewed, sample_count),
        "wrong_capture_rate": _safe_rate(wrong_reviewed, wrong_count),
        "false_positive_review_rate": _safe_rate(correct_reviewed, correct_count),
        "wrong_missed_count": wrong_count - wrong_reviewed,
        "correct_reviewed_count": correct_reviewed,
    }
    return {"policy": dict(policy), "metrics": metrics}


def policy_requires_review(row: dict[str, Any], policy: dict[str, Any]) -> bool:
    """Return whether a normalized row should be reviewed under a candidate policy."""
    row = _ensure_normalized_row(row)
    if row["full_confidence"] < float(policy["full_confidence_threshold"]):
        return True
    if policy["review_on_roi_conflict"] and row["roi_evidence_status"] == "conflicts_with_full":
        return True
    if policy["review_on_roi_quality_bad"] and row["roi_quality_status"] in ROI_QUALITY_BAD:
        return True
    if policy["review_on_full_ood"] and row["full_ood_is_ood"]:
        return True
    return bool(policy["review_on_roi_ood"] and row["roi_ood_is_ood"])


def candidate_policies() -> list[dict[str, Any]]:
    """Return the small auditable v1 candidate grid."""
    policies: list[dict[str, Any]] = []
    for threshold, roi_conflict, roi_quality_bad, full_ood, roi_ood in itertools.product(
        [0.50, 0.60, 0.70, 0.80, 0.90],
        [False, True],
        [False, True],
        [False, True],
        [False, True],
    ):
        policies.append(
            {
                "full_confidence_threshold": threshold,
                "review_on_roi_conflict": roi_conflict,
                "review_on_roi_quality_bad": roi_quality_bad,
                "review_on_full_ood": full_ood,
                "review_on_roi_ood": roi_ood,
            }
        )
    return policies


def _target_policy_payload(
    target_rows: Sequence[dict[str, Any]],
    calibration_rows: Sequence[dict[str, Any]],
    holdout_rows: Sequence[dict[str, Any]],
    *,
    global_selection: dict[str, Any],
    min_capture: float,
    max_false_positive_rate: float,
    min_target_errors: int,
) -> dict[str, Any]:
    total_wrong = sum(1 for row in target_rows if row["is_comparable"] and not row["is_correct"])
    calibration_wrong = sum(1 for row in calibration_rows if row["is_comparable"] and not row["is_correct"])
    holdout_wrong = sum(1 for row in holdout_rows if row["is_comparable"] and not row["is_correct"])
    if total_wrong < min_target_errors:
        return _global_fallback_payload(
            global_selection,
            holdout_rows,
            fallback_reason=f"target_wrong_count_below_min_target_errors:{total_wrong}<{min_target_errors}",
        )
    if calibration_wrong < 1 or holdout_wrong < 1:
        return _global_fallback_payload(
            global_selection,
            holdout_rows,
            fallback_reason="target_split_missing_wrong_examples",
        )

    target_selection = select_policy(
        calibration_rows,
        min_capture=min_capture,
        max_false_positive_rate=max_false_positive_rate,
    )
    if target_selection["status"] == "eligible":
        selected = target_selection["selected"]
        return {
            "status": "target_specific",
            "policy": selected["policy"],
            "calibration_metrics": selected["metrics"],
            "holdout_metrics": evaluate_policy(holdout_rows, selected["policy"])["metrics"],
            "fallback_reason": "",
        }
    fallback = _global_fallback_payload(
        global_selection,
        holdout_rows,
        fallback_reason="target_selection_no_eligible_policy",
    )
    fallback["target_best_rejected"] = target_selection.get("best_rejected")
    return fallback


def _global_fallback_payload(
    global_selection: dict[str, Any],
    holdout_rows: Sequence[dict[str, Any]],
    *,
    fallback_reason: str,
) -> dict[str, Any]:
    if global_selection["status"] != "eligible":
        return {
            "status": "no_eligible_policy",
            "policy": {},
            "calibration_metrics": {},
            "holdout_metrics": {},
            "fallback_reason": fallback_reason,
            "best_rejected": global_selection.get("best_rejected"),
        }
    selected = global_selection["selected"]
    return {
        "status": "global_fallback",
        "policy": selected["policy"],
        "calibration_metrics": selected["metrics"],
        "holdout_metrics": evaluate_policy(holdout_rows, selected["policy"])["metrics"],
        "fallback_reason": fallback_reason,
    }


def _policy_payload(selection: dict[str, Any], holdout_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if selection["status"] != "eligible":
        return {
            "status": "no_eligible_policy",
            "policy": {},
            "calibration_metrics": {},
            "holdout_metrics": {},
            "best_rejected": selection.get("best_rejected"),
        }
    selected = selection["selected"]
    return {
        "status": "eligible",
        "policy": selected["policy"],
        "calibration_metrics": selected["metrics"],
        "holdout_metrics": evaluate_policy(holdout_rows, selected["policy"])["metrics"],
    }


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    expected = _normalize_label(row.get("expected_label"))
    diagnosis = _normalize_label(row.get("diagnosis") or row.get("full_diagnosis"))
    is_comparable = bool(expected and diagnosis)
    return {
        "image_path": str(row.get("image_path") or ""),
        "target_key": _target_key(row),
        "expected_label": row.get("expected_label"),
        "diagnosis": row.get("diagnosis") or row.get("full_diagnosis"),
        "is_comparable": is_comparable,
        "is_correct": bool(is_comparable and expected == diagnosis),
        "full_confidence": _to_float(row.get("full_confidence", row.get("confidence")), default=0.0),
        "roi_evidence_status": _normalize_label(row.get("roi_evidence_status")),
        "roi_quality_status": _normalize_label(row.get("roi_quality_status")),
        "full_ood_is_ood": _is_true(row.get("full_ood_is_ood") if "full_ood_is_ood" in row else row.get("ood_is_ood")),
        "roi_ood_is_ood": _is_true(row.get("roi_ood_is_ood")),
    }


def _ensure_normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    if "is_comparable" in row and "target_key" in row:
        return dict(row)
    return _normalize_row(row)


def _target_key(row: dict[str, Any]) -> str:
    if row.get("target_id"):
        return str(row["target_id"])
    if row.get("dataset_key"):
        return str(row["dataset_key"])
    crop = str(row.get("crop") or "unknown")
    part = str(row.get("part") or "unknown")
    return f"{crop}__{part}"


def _eligible_sort_key(variant: dict[str, Any]) -> tuple[float, float, float, int, float]:
    metrics = variant["metrics"]
    policy = variant["policy"]
    return (
        -_rate_or_zero(metrics["coverage"]),
        -_rate_or_zero(metrics["wrong_capture_rate"]),
        _rate_or_one(metrics["false_positive_review_rate"]),
        _policy_complexity(policy),
        float(policy["full_confidence_threshold"]),
    )


def _rejected_sort_key(variant: dict[str, Any]) -> tuple[float, float, float, int, float]:
    metrics = variant["metrics"]
    policy = variant["policy"]
    return (
        -_rate_or_zero(metrics["wrong_capture_rate"]),
        _rate_or_one(metrics["false_positive_review_rate"]),
        -_rate_or_zero(metrics["coverage"]),
        _policy_complexity(policy),
        float(policy["full_confidence_threshold"]),
    )


def _policy_complexity(policy: dict[str, Any]) -> int:
    return sum(
        int(bool(policy[key]))
        for key in (
            "review_on_roi_conflict",
            "review_on_roi_quality_bad",
            "review_on_full_ood",
            "review_on_roi_ood",
        )
    )


def _rate_at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _rate_at_most(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def _rate_or_zero(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def _rate_or_one(value: float | None) -> float:
    return 1.0 if value is None else float(value)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return (numerator / denominator) if denominator else None


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


def _to_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _normalize_label(value) in {"1", "true", "yes", "y"}


def _as_posix(path: str | Path) -> str:
    return Path(path).as_posix()
