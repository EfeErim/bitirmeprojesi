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
    "sure_ds_f1": 0.90,
    "conformal_empirical_coverage": 0.95,
}


def load_plan_targets(spec_path: Optional[Path] = None) -> Dict[str, float]:
    if spec_path is None:
        return dict(DEFAULT_PLAN_TARGETS)

    resolved = Path(spec_path)
    if not resolved.exists():
        return dict(DEFAULT_PLAN_TARGETS)

    payload = read_json(resolved, default={}, expect_type=dict)
    targets = payload.get("targets", {}) if isinstance(payload, dict) else {}
    return {
        "accuracy": float(
            targets.get("accuracy", targets.get("continual_accuracy", DEFAULT_PLAN_TARGETS["accuracy"]))
        ),
        "ood_auroc": float(targets.get("ood_auroc", DEFAULT_PLAN_TARGETS["ood_auroc"])),
        "ood_false_positive_rate": float(
            targets.get("ood_false_positive_rate", DEFAULT_PLAN_TARGETS["ood_false_positive_rate"])
        ),
        "sure_ds_f1": float(targets.get("sure_ds_f1", DEFAULT_PLAN_TARGETS["sure_ds_f1"])),
        "conformal_empirical_coverage": float(
            targets.get(
                "conformal_empirical_coverage",
                DEFAULT_PLAN_TARGETS["conformal_empirical_coverage"],
            )
        ),
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

    ood_auroc: Optional[float] = None
    ood_fpr: Optional[float] = None
    ood_total = 0
    in_dist_total = 0
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
        "accuracy": accuracy,
        "ood_auroc": ood_auroc,
        "ood_false_positive_rate": ood_fpr,
        "classification_samples": int(y_true_t.numel()),
        "ood_samples": int(ood_total),
        "in_distribution_samples": int(in_dist_total),
        "sure_ds_f1": sure_ds_f1,
        "conformal_empirical_coverage": conformal_empirical_coverage,
        "conformal_avg_set_size": conformal_avg_set_size,
    }


def validate_plan_metrics(
    metrics: Dict[str, Optional[float]],
    targets: Optional[Dict[str, float]] = None,
    *,
    require_ood: bool = False,
) -> Dict[str, Any]:
    target_values = dict(targets or load_plan_targets())
    checks: Dict[str, Dict[str, Any]] = {}

    acc_value = metrics.get("accuracy")
    checks["accuracy"] = {
        "value": acc_value,
        "target": target_values["accuracy"],
        "operator": ">=",
        "asserted": acc_value is not None,
        "passed": bool(acc_value is not None and float(acc_value) >= float(target_values["accuracy"])),
    }

    auroc_value = metrics.get("ood_auroc")
    checks["ood_auroc"] = {
        "value": auroc_value,
        "target": target_values["ood_auroc"],
        "operator": ">=",
        "asserted": auroc_value is not None,
        "passed": bool(auroc_value is not None and float(auroc_value) >= float(target_values["ood_auroc"])),
    }

    fpr_value = metrics.get("ood_false_positive_rate")
    checks["ood_false_positive_rate"] = {
        "value": fpr_value,
        "target": target_values["ood_false_positive_rate"],
        "operator": "<=",
        "asserted": fpr_value is not None,
        "passed": bool(fpr_value is not None and float(fpr_value) <= float(target_values["ood_false_positive_rate"])),
    }

    # SURE+ DS-F1 (soft gate)
    ds_f1_value = metrics.get("sure_ds_f1")
    ds_f1_target = target_values.get("sure_ds_f1", 0.90)
    checks["sure_ds_f1"] = {
        "value": ds_f1_value,
        "target": ds_f1_target,
        "operator": ">=",
        "asserted": ds_f1_value is not None,
        "passed": bool(ds_f1_value is not None and float(ds_f1_value) >= float(ds_f1_target)),
    }

    # Conformal empirical coverage (soft gate)
    coverage_value = metrics.get("conformal_empirical_coverage")
    coverage_target = target_values.get("conformal_empirical_coverage", 0.95)
    checks["conformal_empirical_coverage"] = {
        "value": coverage_value,
        "target": coverage_target,
        "operator": ">=",
        "asserted": coverage_value is not None,
        "passed": bool(coverage_value is not None and float(coverage_value) >= float(coverage_target)),
    }

    missing_checks = [name for name, detail in checks.items() if not detail["asserted"]]
    if require_ood:
        gating_status = "failed" if missing_checks else "ready"
        gating_reason = "missing_required_metrics" if missing_checks else "all_required_metrics_present"
    else:
        gating_status = "soft" if missing_checks else "ready"
        gating_reason = "missing_optional_metrics" if missing_checks else "all_metrics_present"

    all_asserted_passed = all(detail["passed"] for detail in checks.values() if detail["asserted"])
    hard_fail = require_ood and bool(missing_checks)
    passed = bool(all_asserted_passed and not hard_fail)

    return {
        "passed": passed,
        "require_ood": bool(require_ood),
        "targets": target_values,
        "checks": checks,
        "gating": {
            "status": gating_status,
            "reason": gating_reason,
            "missing_metrics": missing_checks,
        },
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
