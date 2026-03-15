"""Deterministic metric helpers for training workflows and notebook artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.shared.json_utils import read_json, write_json

DEFAULT_PLAN_TARGETS = {
    "accuracy": 0.93,
    "ood_auroc": 0.92,
    "ood_false_positive_rate": 0.05,
    "ood_samples": 5.0,
    "in_distribution_samples": 5.0,
    "sure_ds_f1": 0.90,
    "conformal_empirical_coverage": 0.95,
    "conformal_avg_set_size": 2.0,
}

OOD_METRIC_NAMES = (
    "ood_auroc",
    "ood_false_positive_rate",
    "ood_samples",
    "in_distribution_samples",
    "sure_ds_f1",
    "conformal_empirical_coverage",
    "conformal_avg_set_size",
)

_TARGET_FALLBACK_KEYS = {
    "accuracy": ("accuracy", "continual_accuracy"),
    "ood_auroc": ("ood_auroc",),
    "ood_false_positive_rate": ("ood_false_positive_rate",),
    "ood_samples": ("ood_samples", "ood_min_samples", "min_ood_samples"),
    "in_distribution_samples": (
        "in_distribution_samples",
        "in_distribution_min_samples",
        "min_in_distribution_samples",
    ),
    "sure_ds_f1": ("sure_ds_f1",),
    "conformal_empirical_coverage": ("conformal_empirical_coverage",),
    "conformal_avg_set_size": ("conformal_avg_set_size", "conformal_max_avg_set_size"),
}


def load_plan_targets(spec_path: Optional[Path] = None) -> Dict[str, float]:
    if spec_path is None:
        return dict(DEFAULT_PLAN_TARGETS)

    resolved = Path(spec_path)
    if not resolved.exists():
        return dict(DEFAULT_PLAN_TARGETS)

    payload = read_json(resolved, default={}, expect_type=dict)
    targets = payload.get("targets", {}) if isinstance(payload, dict) else {}
    return _resolve_target_values(targets)


def _resolve_target_value(targets: Dict[str, Any], metric_name: str) -> float:
    for key in _TARGET_FALLBACK_KEYS.get(metric_name, (metric_name,)):
        if key in targets:
            return float(targets[key])
    return float(DEFAULT_PLAN_TARGETS[metric_name])


def _resolve_target_values(targets: Optional[Dict[str, Any]]) -> Dict[str, float]:
    raw_targets = dict(targets or {})
    return {
        metric_name: float(_resolve_target_value(raw_targets, metric_name))
        for metric_name in DEFAULT_PLAN_TARGETS
    }


def _build_threshold_checks(
    metrics: Dict[str, Optional[float]],
    target_values: Dict[str, float],
    definitions: Sequence[tuple[str, str]],
) -> Dict[str, Dict[str, Any]]:
    return {
        metric_name: _build_check(
            metrics.get(metric_name),
            float(target_values[metric_name]),
            operator=operator,
        )
        for metric_name, operator in definitions
    }


def _resolve_ood_metric_payload(ood_metrics: Optional[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    resolved_ood_metrics = {name: None for name in OOD_METRIC_NAMES}
    for name, value in dict(ood_metrics or {}).items():
        if name in resolved_ood_metrics:
            resolved_ood_metrics[name] = value
    return resolved_ood_metrics


def _collect_missing_requirements(
    accuracy_check: Dict[str, Any],
    ood_checks: Dict[str, Dict[str, Any]],
    *,
    require_ood: bool,
) -> list[str]:
    missing_requirements: list[str] = []
    if not accuracy_check.get("asserted", False) or not accuracy_check.get("passed", False):
        missing_requirements.append("accuracy")
    for metric_name, detail in ood_checks.items():
        asserted = bool(detail.get("asserted", False))
        passed = bool(detail.get("passed", False))
        if require_ood:
            if not asserted or not passed:
                missing_requirements.append(metric_name)
            continue
        if asserted and not passed:
            missing_requirements.append(metric_name)
    return missing_requirements


def compute_ood_detection_metrics(
    *,
    ood_labels: Optional[Sequence[int]] = None,
    ood_scores: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    ood_auroc: Optional[float] = None
    ood_fpr: Optional[float] = None
    ood_total: Optional[int] = None
    in_dist_total: Optional[int] = None
    if ood_labels is not None and ood_scores is not None:
        if len(ood_labels) != len(ood_scores):
            raise ValueError("ood_labels and ood_scores must have same length")
        if len(ood_labels) > 0:
            labels_t = torch.tensor(list(ood_labels), dtype=torch.long)
            scores_t = torch.tensor(list(ood_scores), dtype=torch.float32)
            ood_total = int((labels_t == 1).sum().item())
            in_dist_total = int((labels_t == 0).sum().item())
            if ood_total > 0 and in_dist_total > 0:
                try:
                    labels_np = labels_t.cpu().numpy()
                    scores_np = scores_t.cpu().numpy()
                    ood_auroc = float(roc_auc_score(labels_np, scores_np))
                    fpr, tpr, _ = roc_curve(labels_np, scores_np, pos_label=1)
                    valid_tpr = tpr >= 0.95
                    if valid_tpr.any():
                        ood_fpr = float(fpr[valid_tpr].min())
                except Exception:
                    ood_auroc = None
                    ood_fpr = None
    return {
        "ood_auroc": ood_auroc,
        "ood_false_positive_rate": ood_fpr,
        "ood_samples": ood_total,
        "in_distribution_samples": in_dist_total,
    }


def compute_plan_metrics(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    ood_labels: Optional[Sequence[int]] = None,
    ood_scores: Optional[Sequence[float]] = None,
    sure_ds_f1: Optional[float] = None,
    conformal_empirical_coverage: Optional[float] = None,
    conformal_avg_set_size: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    if len(y_true) == 0:
        raise ValueError("y_true must not be empty")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    y_true_t = torch.tensor(list(y_true), dtype=torch.long)
    y_pred_t = torch.tensor(list(y_pred), dtype=torch.long)
    accuracy = float((y_true_t == y_pred_t).float().mean().item())

    ood_metrics = compute_ood_detection_metrics(ood_labels=ood_labels, ood_scores=ood_scores)

    return {
        "accuracy": accuracy,
        "ood_auroc": ood_metrics["ood_auroc"],
        "ood_false_positive_rate": ood_metrics["ood_false_positive_rate"],
        "classification_samples": int(y_true_t.numel()),
        "ood_samples": (
            None if ood_metrics["ood_samples"] is None else int(ood_metrics["ood_samples"])
        ),
        "in_distribution_samples": (
            None
            if ood_metrics["in_distribution_samples"] is None
            else int(ood_metrics["in_distribution_samples"])
        ),
        "sure_ds_f1": sure_ds_f1,
        "conformal_empirical_coverage": conformal_empirical_coverage,
        "conformal_avg_set_size": conformal_avg_set_size,
    }


def _build_check(
    value: Optional[float],
    target: float,
    *,
    operator: str,
) -> Dict[str, Any]:
    if operator == ">=":
        passed = bool(value is not None and float(value) >= float(target))
    elif operator == "<=":
        passed = bool(value is not None and float(value) <= float(target))
    else:  # pragma: no cover - internal misuse guard
        raise ValueError(f"Unsupported operator: {operator}")
    return {
        "value": value,
        "target": float(target),
        "operator": operator,
        "asserted": value is not None,
        "passed": passed,
    }


def _finalize_validation(
    checks: Dict[str, Dict[str, Any]],
    *,
    require_metrics: bool,
) -> Dict[str, Any]:
    missing_checks = [name for name, detail in checks.items() if not detail["asserted"]]
    if require_metrics:
        gating_status = "failed" if missing_checks else "ready"
        gating_reason = "missing_required_metrics" if missing_checks else "all_required_metrics_present"
    else:
        gating_status = "soft" if missing_checks else "ready"
        gating_reason = "missing_optional_metrics" if missing_checks else "all_metrics_present"

    all_asserted_passed = all(detail["passed"] for detail in checks.values() if detail["asserted"])
    hard_fail = require_metrics and bool(missing_checks)
    passed = bool(all_asserted_passed and not hard_fail)

    return {
        "passed": passed,
        "checks": checks,
        "gating": {
            "status": gating_status,
            "reason": gating_reason,
            "missing_metrics": missing_checks,
        },
    }


def validate_ood_metrics(
    metrics: Dict[str, Optional[float]],
    targets: Optional[Dict[str, float]] = None,
    *,
    require_ood: bool = False,
) -> Dict[str, Any]:
    target_values = _resolve_target_values(targets)
    checks = _build_threshold_checks(
        metrics,
        target_values,
        (
            ("ood_auroc", ">="),
            ("ood_false_positive_rate", "<="),
            ("ood_samples", ">="),
            ("in_distribution_samples", ">="),
            ("sure_ds_f1", ">="),
            ("conformal_empirical_coverage", ">="),
            ("conformal_avg_set_size", "<="),
        ),
    )
    finalized = _finalize_validation(checks, require_metrics=require_ood)
    return {
        "passed": finalized["passed"],
        "require_ood": bool(require_ood),
        "targets": target_values,
        "checks": checks,
        "gating": finalized["gating"],
    }


def validate_plan_metrics(
    metrics: Dict[str, Optional[float]],
    targets: Optional[Dict[str, float]] = None,
    *,
    require_ood: bool = False,
) -> Dict[str, Any]:
    target_values = _resolve_target_values(targets)
    checks: Dict[str, Dict[str, Any]] = _build_threshold_checks(
        metrics,
        target_values,
        (("accuracy", ">="),),
    )
    ood_validation = validate_ood_metrics(metrics, target_values, require_ood=require_ood)
    checks.update(ood_validation["checks"])
    finalized = _finalize_validation(checks, require_metrics=require_ood)
    return {
        "passed": finalized["passed"],
        "require_ood": bool(require_ood),
        "targets": target_values,
        "checks": checks,
        "gating": finalized["gating"],
    }


def build_production_readiness(
    *,
    classification_metric_gate: Optional[Dict[str, Any]],
    classification_split: str,
    ood_evidence_source: Optional[str],
    ood_metrics: Optional[Dict[str, Optional[float]]],
    targets: Optional[Dict[str, float]] = None,
    context: Optional[Dict[str, Any]] = None,
    require_ood: bool = True,
) -> Dict[str, Any]:
    target_values = _resolve_target_values(targets)
    classification_gate = dict(classification_metric_gate or {})
    classification_metrics = dict(classification_gate.get("metrics", {}))
    classification_eval = dict(classification_gate.get("evaluation", {}))
    classification_checks = dict(classification_eval.get("checks", {}))
    accuracy_check = classification_checks.get(
        "accuracy",
        _build_check(classification_metrics.get("accuracy"), target_values["accuracy"], operator=">="),
    )

    resolved_ood_metrics = _resolve_ood_metric_payload(ood_metrics)
    ood_validation = validate_ood_metrics(resolved_ood_metrics, target_values, require_ood=require_ood)

    missing_requirements = _collect_missing_requirements(
        accuracy_check,
        ood_validation["checks"],
        require_ood=require_ood,
    )
    passed = not missing_requirements
    status = "ready" if passed else "failed"
    return {
        "status": status,
        "passed": bool(passed),
        "ood_evidence_source": str(ood_evidence_source or "unavailable"),
        "classification_evidence": {
            "split_name": str(classification_split),
            "metrics": classification_metrics,
            "evaluation": {
                "checks": {"accuracy": accuracy_check},
            },
        },
        "ood_evidence": {
            "source": str(ood_evidence_source or "unavailable"),
            "metrics": resolved_ood_metrics,
            "evaluation": ood_validation,
        },
        "missing_requirements": missing_requirements,
        "targets": target_values,
        "context": dict(context or {}),
    }


def write_plan_metric_artifact(
    *,
    output_path: Path,
    metrics: Dict[str, Optional[float]],
    targets: Optional[Dict[str, float]] = None,
    require_ood: bool = False,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    artifact = {
        "schema_version": "v6_plan_metric_gate",
        "metrics": metrics,
        "evaluation": validate_plan_metrics(metrics, targets, require_ood=require_ood),
        "context": dict(context or {}),
    }
    write_json(output_path, artifact)
    return artifact
